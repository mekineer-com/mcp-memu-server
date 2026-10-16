import asyncio
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main
from tests import SavedBatchService
from app.db import sqlite_ensure_conversation_state_schema
from app.services import consolidation, conversation_sources
from app.services.import_routes import ImportPreview, ImportScope, ImportProcess

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "openalma" / "launcher"))
import chat_import


def _endpoint(path):
    return next(route.endpoint for route in main.app.routes if getattr(route, "path", None) == path)


def _guidance_schema(con):
    sqlite_ensure_conversation_state_schema(con)
    con.execute("CREATE TABLE resources (user_id TEXT, soul_id TEXT, conversation_id TEXT, "
                "modality TEXT)")


def test_import_background_readers_keep_historical_rows_out(tmp_path, monkeypatch):
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    monkeypatch.setattr(conversation_sources, "import_source_path", lambda: tmp_path / "imports.db")
    rows, _, _ = chat_import.normalize_messages([
        {"id": str(index), "role": "user", "content": f"Fictional row {index}",
         "timestamp": f"2026-01-01T0{index}:00:00Z"} for index in range(3)])
    chat = chat_import.store_upload(tmp_path / "imports.db", **scope, label="TestApp", messages=rows, history_count=1)
    cid = chat["conversation_id"]
    main._write_conversation_state(cid, **scope, updates={"memorize_chat": False})
    tail = main._load_background_rollup_tail(conversation_id=cid, **scope, rolling_summary_cursor_id=None)
    assert [row["content"] for row in tail] == ["Fictional row 1", "Fictional row 2"]
    with main._sqlite_connect(main._sqlite_current_path(**scope)) as con:
        con.row_factory = sqlite3.Row
        tails = main._cross_history._load_cross_memorize_tails_from_sources(con, **scope, strict=True)
    assert [row["content"] for row in tails[cid]] == [row["content"] for row in tail]
    assert all(row["memorize_chat"] is False for row in tails[cid])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["default", "continuous", "enable", "disable", "failure", "draining", "draining_final"])
async def test_import_task_owns_continuation_and_releases_once(monkeypatch, mode):
    from app.services import import_routes
    scope = {"user_id": "TestOwner", "soul_id": "LoopSoul"}
    cid = "import:dm:loop"
    marker = main._memorize_lock_key(**scope)
    monkeypatch.setattr(main, "_BACKGROUND_TASKS", set())
    monkeypatch.setattr(main, "_FORCED_MEMORIZE_INFLIGHT", {})
    monkeypatch.setattr(main, "_SHUTDOWN_STATE", {"draining": False})
    monkeypatch.setattr(conversation_sources, "import_chat_info", lambda **_: {"conversation_id": cid, "history_end_index": 3})
    main._write_conversation_state(cid, **scope, updates={"import_state": {
        "history_end_index": 3, "memorize_cursor": -1, "pending_segment_ids": [],
        "stage": "memorize", "error": None,
    }})
    if mode == "draining_final":
        main._write_conversation_state(cid, **scope, updates={"import_ordinary_waiting": "chat:waiting"})
        async def no_handoff(*_args, **_kwargs):
            pytest.fail("Shutdown started another paid Memorize")
        monkeypatch.setattr(import_routes, "run_waiting_memorize", no_handoff)
    entered, finish = asyncio.Event(), asyncio.Event()
    calls, releases = [], []
    async def batch(_runtime, **kwargs):
        calls.append(kwargs["retry"])
        assert main._FORCED_MEMORIZE_INFLIGHT[marker] is True
        if len(calls) == 1:
            entered.set()
            await finish.wait()
        failed = mode == "failure"
        main._write_conversation_state(cid, **scope, updates={
            "import_memorize_cursor": 2 if mode == "draining_final" else len(calls) - 1,
            "import_error": "Fictional failure" if failed else None,
        })
        return not failed
    monkeypatch.setattr(import_routes, "run_import_batch", batch)
    release = main._finish_memorize_claim
    async def released(key, success):
        releases.append(success)
        await release(key, success)
    monkeypatch.setattr(main, "_finish_memorize_claim", released)
    request = ImportProcess(**scope, label="TestApp", continuous=mode not in {"default", "enable"})
    await _endpoint("/imports/process")(request)
    task = next(iter(main._BACKGROUND_TASKS))
    await entered.wait()
    assert _endpoint("/imports/status")(**scope, label="TestApp")["running"]
    with pytest.raises(HTTPException, match="still running"):
        await _endpoint("/imports/process")(request)
    if mode in {"enable", "disable"}:
        await _endpoint("/imports/continuation")(request.model_copy(update={"continuous": mode == "enable"}))
    if mode in {"draining", "draining_final"}:
        main._SHUTDOWN_STATE["draining"] = True
    finish.set()
    await task
    assert len(calls) == (3 if mode in {"continuous", "enable"} else 1)
    assert releases == [mode != "failure"]
    assert marker not in main._FORCED_MEMORIZE_INFLIGHT
    if mode == "draining_final":
        assert main._soul_import_state(**scope)[1]["ordinary_waiting"] == "chat:waiting"
    assert not _endpoint("/imports/status")(**scope, label="TestApp")["running"]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "empty", "empty_failure", "failure", "accepted", "unavailable", "not_due",
                                    "no_pending_segments", "in_progress", "memorize_postprocessing_pending",
                                    "failure_requires_retry"])
async def test_import_wait_survives_batches_and_hands_off_to_saved_cross_chat(monkeypatch, outcome):
    scoped = {"user_id": "TestOwner", "soul_id": "ImportSoul"}
    cid, trigger = "import:dm:wait", "chat:trigger"
    marker = main._memorize_lock_key(**scoped)
    monkeypatch.setattr(main, "_FORCED_MEMORIZE_INFLIGHT", {})
    monkeypatch.setattr(main, "_FORCED_MEMORIZE_RECHECK", {})
    monkeypatch.setattr(main, "_CONSOLIDATION_RUNNING", {})
    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_kw: object())
    monkeypatch.setattr(conversation_sources, "import_chat_info", lambda **_: {
        "conversation_id": cid, "history_end_index": 4})
    async def consolidated(**_kwargs):
        if outcome == "empty_failure":
            main._write_conversation_state(cid, **scoped, updates={
                "last_consolidation_error": "Fictional consolidation failure",
                "last_consolidation_error_at": "2026-01-02T00:00:00Z",
            })
            raise ValueError("Fictional consolidation failure")
        return {"status": "skipped", "reason": "no_pending_segments"}
    monkeypatch.setattr(main, "_run_consolidation_pipeline_once", consolidated)
    main._write_conversation_state(cid, **scoped, updates={"import_state": {
        "history_end_index": 4, "memorize_cursor": -1, "pending_segment_ids": [],
        "stage": "memorize", "error": None,
    }})
    assert main._soul_activity_pause(**scoped) is None
    response = await main._memorize_owned({"user": scoped, "conversation_id": trigger,
                                          "conversation": []}, main.BackgroundTasks(), True)
    assert response.status_code == 202 and json.loads(response.body)["status"] == "waiting_for_import"
    scope = main._auto_memorize_scope(trigger, *scoped.values(), {}, [])
    assert main._schedule_auto_memorize({"not": "executed"}, scope) == "coalesced"
    assert main._soul_import_state(**scoped)[1]["ordinary_waiting"] == trigger
    assert main._soul_activity_pause(**scoped) == "Waiting for import before Memorize."
    assert main._soul_activity_pause("TestOwner", "OtherSoul") is None
    main._FORCED_MEMORIZE_INFLIGHT[marker] = True
    await main._finish_memorize_claim(marker, True)
    assert main._soul_activity_pause(**scoped)
    with pytest.raises(HTTPException, match="Finish the import"):
        await main.retry_memorize(**scoped, background_tasks=main.BackgroundTasks())
    main._write_conversation_state(cid, **scoped, updates={"import_memorize_cursor": 3})
    main._FORCED_MEMORIZE_INFLIGHT[marker] = True
    assert main._schedule_auto_memorize({}, scope) == "coalesced"
    assert marker not in main._FORCED_MEMORIZE_RECHECK
    await main._finish_memorize_claim(marker, True)
    def assemble(actual_cid, uid, sid, failure):
        assert (actual_cid, uid, sid) == (trigger, *scoped.values())
        if outcome == "unavailable":
            raise ValueError("Source unavailable")
        if outcome in {"empty", "empty_failure"}:
            return None
        return {"user": scoped, "conversation_id": trigger, "conversation": [
            {"content": "from trigger"}, {"content": "from another chat"}]}
    monkeypatch.setattr(main, "_saved_memorize_payload", assemble)
    async def execute(payload, tasks, force, **kwargs):
        assert kwargs == {"admitted": True, "batch_owned": True, "import_handoff": True, "retry": False}
        assert main._soul_activity_pause(**scoped)
        assert main._soul_activity_pause(**scoped, import_handoff=True) is None
        async def finish():
            if outcome in {"success", "failure"}:
                return {"ok": outcome == "success", "status": "ok" if outcome == "success" else "error"}
            if outcome == "accepted":
                return {"ok": True, "status": "accepted"}
            return {"ok": True, "status": "skipped", "reason": outcome}
        tasks.add_task(finish)
    monkeypatch.setattr(main, "_memorize_owned", execute)
    tasks = main.BackgroundTasks()
    assert (await main.retry_memorize(**scoped, background_tasks=tasks))["status"] == "accepted"
    assert _endpoint("/imports/status")(**scoped, label="TestApp")["followup_error"] is None
    await tasks()
    assert marker not in main._FORCED_MEMORIZE_INFLIGHT
    waiting = outcome not in {"success", "empty", "not_due", "no_pending_segments"}
    assert bool(main._soul_import_state(**scoped)[1]["ordinary_waiting"]) == waiting
    assert bool(main._soul_activity_pause(**scoped)) == waiting
    assert bool(_endpoint("/imports/status")(**scoped, label="TestApp")["followup_error"]) == waiting


@pytest.mark.asyncio
async def test_handoff_controls_follow_import_then_consolidation_before_memorize():
    scoped = {"user_id": "TestOwner", "soul_id": "ImportSoul"}
    cid, trigger = "import:dm:priority", "chat:trigger"
    failure = {"conversation_id": trigger, "paused": True, "error": "Fictional Memorize failure", "targets": {}}
    main._write_conversation_state(cid, **scoped, updates={"import_state": {
        "history_end_index": 1, "memorize_cursor": -1, "pending_segment_ids": [],
        "stage": "memorize", "error": None,
    }, "memorize_failure": failure})
    status = await main.diag_memorize_pending(**scoped)
    assert status["retry_operation"] == "import" and status["pause_reason"] == "Waiting for import before Memorize."
    main._write_conversation_state(cid, **scoped, updates={"import_memorize_cursor": 0,
        "import_ordinary_waiting": trigger, "last_consolidation_error": "Fictional consolidation failure",
        "last_consolidation_error_at": "2026-10-05T00:00:00Z"})
    status = await main.diag_memorize_pending(**scoped)
    assert status["retry_operation"] == "consolidation" and status["pause_reason"] == "Fictional consolidation failure"
    with pytest.raises(HTTPException, match="Retry consolidation"):
        await main.retry_memorize(**scoped, background_tasks=main.BackgroundTasks())
    main._write_conversation_state(cid, **scoped, updates={"last_consolidation_error": None})
    assert (await main.diag_memorize_pending(**scoped))["retry_operation"] == "memorize"


def test_import_interruption_marker_only_pauses_when_not_running(monkeypatch):
    scoped = {"user_id": "TestOwner", "soul_id": "ImportSoul"}
    marker = main._memorize_lock_key(**scoped)
    monkeypatch.setattr(main, "_FORCED_MEMORIZE_INFLIGHT", {marker: True})
    main._write_conversation_state("import:dm:failure", **scoped, updates={"import_state": {
        "history_end_index": 1, "memorize_cursor": -1, "pending_segment_ids": [],
        "stage": "memorize", "error": "Interrupted",
    }})
    assert main._soul_activity_pause(**scoped) is None
    main._FORCED_MEMORIZE_INFLIGHT.clear()
    assert main._soul_activity_pause(**scoped) == "Import failed. Retry in Echo."


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_handoff", [False, True])
async def test_real_import_completion_memorizes_current_messages_across_chats(tmp_path, monkeypatch, fail_handoff):
    source = tmp_path / "imports.db"
    scoped = {"user_id": "TestOwner", "soul_id": "ImportSoul"}
    monkeypatch.setattr(conversation_sources, "import_source_path", lambda: source)
    monkeypatch.setattr(main, "_FORCED_MEMORIZE_INFLIGHT", {})
    monkeypatch.setattr(main, "_FORCED_MEMORIZE_RECHECK", {})
    monkeypatch.setattr(main, "_BACKGROUND_TASKS", set())
    rows, _, _ = chat_import.normalize_messages([
        {"id": "past", "role": "user", "content": "Past story", "timestamp": "2025-01-01"},
        {"id": "current", "role": "user", "content": "Current imported story", "timestamp": "2026-01-01"},
    ])
    chat = chat_import.store_upload(source, **scoped, label="TestApp", messages=rows, history_count=1)
    path = main._sqlite_current_path(**scoped)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        sqlite_ensure_conversation_state_schema(con)
    SavedBatchService.make_engine(scoped).database.close()
    calls = []
    class Service(SavedBatchService):
        memorize_config = SimpleNamespace(category_update_llm_profile="default")
        llm_profiles = SimpleNamespace(profiles={"default": SimpleNamespace(
            context_window_tokens=100000, max_tokens=1000, chat_model="fictional")})
        async def extract(self, **kwargs):
            messages = [row for segment in kwargs["segments"] for row in json.loads(segment["raw_text"])]
            calls.append([row["content"] for row in messages])
            if len(calls) == 2 and fail_handoff:
                raise ValueError("Fictional ordinary extraction failure")
            return [{"memory_item_ids": []} for _ in kwargs["segments"]]
    monkeypatch.setattr(main, "_get_service_from_payload", lambda _, **_kwargs: Service())
    monkeypatch.setattr(consolidation, "_prepare_dossier_consolidation_prompts", lambda *_a, **_kw:
        ([], None, {}, {"dossiers": 0, "anchors": 0, "weekly": 0}))
    async def consolidated(**kwargs):
        if kwargs.get("historical"):
            cid = kwargs["conversation_id"]
            record = main._load_turn_state_and_soul_card(cid, **scoped)[0]["import_state"]
            main._write_conversation_state(cid, **scoped, updates={"import_state": {
                **record, "pending_segment_ids": [], "stage": "complete", "error": None,
            }})
        return {"status": "ok", "result": {}}
    monkeypatch.setattr(main, "_run_consolidation_pipeline_once", consolidated)
    request = ImportProcess(**scoped, label="TestApp")
    _endpoint("/imports/register")(request)
    trigger = "chat:ordinary"
    main._write_conversation_state(trigger, **scoped, updates={})
    conversation_sources.persist_sillytavern_history_snapshot(
        storage_dir=main._get_storage_dir(main._CONFIG), **scoped, conversation_id=trigger,
        history=[{"role": "user", "content": "Current other chat", "received_at": "2026-01-01T12:00:00Z"}])
    assert main._schedule_auto_memorize({}, main._auto_memorize_scope(trigger, *scoped.values(), {}, [])) == "coalesced"
    later_trigger = "chat:later-trigger"
    main._write_conversation_state(later_trigger, **scoped, updates={})
    conversation_sources.persist_sillytavern_history_snapshot(
        storage_dir=main._get_storage_dir(main._CONFIG), **scoped, conversation_id=later_trigger,
        history=[{"role": "user", "content": "Current second chat", "received_at": "2026-01-01T13:00:00Z"}])
    assert main._schedule_auto_memorize({}, main._auto_memorize_scope(later_trigger, *scoped.values(), {}, [])) == "coalesced"
    assert main._soul_import_state(**scoped)[1]["ordinary_waiting"] == later_trigger
    assert (await main.diag_memorize_pending(**scoped))["retry_operation"] == "import"
    assert (await _endpoint("/imports/process")(request)).status_code == 202
    await asyncio.gather(*list(main._BACKGROUND_TASKS))
    assert calls == [["Past story"], ["Current imported story", "Current other chat", "Current second chat"]]
    imported = main._soul_import_state(**scoped)[1]
    assert imported["stage"] == "complete" and bool(imported["ordinary_waiting"]) == fail_handoff
    assert bool(main._soul_activity_pause(**scoped)) == fail_handoff
    assert bool(main._paid_work_state(**scoped)["memorize_failure"]) == fail_handoff
    assert (await main.diag_memorize_pending(**scoped))["retry_operation"] == ("memorize" if fail_handoff else None)
    assert not main._FORCED_MEMORIZE_INFLIGHT
    echo_status = _endpoint("/imports/status")(**scoped, label="TestApp")
    assert bool(echo_status["followup_error"]) == fail_handoff
    assert echo_status["import_state"]["stage"] == "complete" and echo_status["import_state"]["error"] is None
    if fail_handoff:
        tasks = main.BackgroundTasks()
        await main.retry_memorize(**scoped, background_tasks=tasks)
        await tasks()
        assert len(calls) == 3 and calls[-1] == calls[1]
        assert main._soul_import_state(**scoped)[1]["ordinary_waiting"] is None
        assert main._soul_activity_pause(**scoped) is None
        assert _endpoint("/imports/status")(**scoped, label="TestApp")["followup_error"] is None
    else:
        # The cursors committed, but the process stopped before clearing the wait.
        main._write_conversation_state(chat["conversation_id"], **scoped,
                                       updates={"import_ordinary_waiting": trigger})
        assert _endpoint("/imports/status")(**scoped, label="TestApp")["followup_error"] == (
            "Waiting for Memorize after import.")
        tasks = main.BackgroundTasks()
        await main.retry_memorize(**scoped, background_tasks=tasks)
        await tasks()
        assert len(calls) == 2 and main._soul_activity_pause(**scoped) is None

        competing = "atomic:competing"
        main._write_conversation_state(competing, **scoped, updates={})
        storage = main._get_storage_dir(main._CONFIG)
        history = [{"role": "user", "content": "Pending competing chat", "received_at": "2026-01-02T12:00:00Z"}]
        conversation_sources.persist_atomic_history_snapshot(storage_dir=storage, **scoped,
            conversation_id=competing, history=history)
        snapshot = conversation_sources._chat_snapshot_path(storage_dir=storage, **scoped,
            conversation_id=competing, source_label="atomic")
        snapshot.unlink()
        main._write_conversation_state(chat["conversation_id"], **scoped, updates={"import_ordinary_waiting": trigger})
        tasks = main.BackgroundTasks()
        await main.retry_memorize(**scoped, background_tasks=tasks)
        await tasks()
        assert len(calls) == 2 and main._soul_import_state(**scoped)[1]["ordinary_waiting"] == trigger
        conversation_sources.persist_atomic_history_snapshot(storage_dir=storage, **scoped,
            conversation_id=competing, history=history)
        tasks = main.BackgroundTasks()
        await main.retry_memorize(**scoped, background_tasks=tasks)
        await tasks()
        assert calls[-1] == ["Pending competing chat"] and main._soul_activity_pause(**scoped) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["remaining_source", "missing_source", "no_pending_segments", "failure"])
async def test_consolidation_retry_reconciles_completed_import_wait(tmp_path, monkeypatch, outcome):
    scoped = {"user_id": "TestOwner", "soul_id": "RetrySoul"}
    source = tmp_path / "imports.db"
    monkeypatch.setattr(conversation_sources, "import_source_path", lambda: source)
    monkeypatch.setattr(main, "_FORCED_MEMORIZE_INFLIGHT", {})
    monkeypatch.setattr(main, "_FORCED_MEMORIZE_RECHECK", {})
    monkeypatch.setattr(main, "_CONSOLIDATION_RUNNING", {})
    monkeypatch.setattr(main, "_BACKGROUND_TASKS", set())
    chat = chat_import.store_upload(source, **scoped, label="TestApp", messages=[], history_count=0)
    _endpoint("/imports/register")(ImportScope(**scoped, label="TestApp"))
    import_cid, trigger = chat["conversation_id"], "chat:retry-trigger"
    clock = "2026-01-01T00:00:00Z"
    main._write_conversation_state(trigger, **scoped, updates={
        "last_consolidation_at": clock, "last_consolidation_error": "Fictional post-commit failure",
        "last_consolidation_error_at": "2026-01-02T00:00:00Z",
    })
    main._write_conversation_state(import_cid, **scoped, updates={"import_ordinary_waiting": trigger})
    conversation_sources.persist_sillytavern_history_snapshot(
        storage_dir=main._get_storage_dir(main._CONFIG), **scoped, conversation_id=trigger,
        history=[] if outcome == "no_pending_segments" else [
            {"role": "user", "content": "Remaining fictional story", "received_at": "2026-01-03T12:00:00Z"}],
    )
    if outcome == "missing_source":
        competing = "atomic:missing"
        main._write_conversation_state(competing, **scoped, updates={})
        storage = main._get_storage_dir(main._CONFIG)
        conversation_sources.persist_atomic_history_snapshot(storage_dir=storage, **scoped,
            conversation_id=competing, history=[{"role": "user", "content": "Missing fictional story",
                                                "received_at": "2026-01-03T13:00:00Z"}])
        snapshot = conversation_sources._chat_snapshot_path(storage_dir=storage, **scoped,
            conversation_id=competing, source_label="atomic")
    extracts, releases, runs = [], [], []
    class Service(SavedBatchService):
        async def extract(self, **kwargs):
            extracts.extend(json.loads(segment["raw_text"]) for segment in kwargs["segments"])
            return [{"memory_item_ids": []} for _ in kwargs["segments"]]
    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_kw: Service())
    pipeline = main._run_consolidation_pipeline_once
    async def consolidated(**kwargs):
        runs.append(kwargs["force"])
        if outcome == "no_pending_segments":
            return await pipeline(**kwargs)
        if not kwargs["force"]:
            return {"status": "skipped", "reason": "not_due"}
        if outcome == "failure":
            raise ValueError("Fictional repeated consolidation failure")
        main._write_conversation_state(trigger, **scoped, updates={
            "last_consolidation_error": None, "last_consolidation_error_at": None,
        })
        if outcome == "missing_source":
            snapshot.unlink()
        return {"status": "ok", "result": {}}
    monkeypatch.setattr(main, "_run_consolidation_pipeline_once", consolidated)
    finish = main._finish_memorize_claim
    async def finished(marker, success):
        releases.append(success)
        await finish(marker, success)
    monkeypatch.setattr(main, "_finish_memorize_claim", finished)
    assert _endpoint("/imports/status")(**scoped, label="TestApp")["followup_error"]
    assert (await main.retry_consolidation(trigger, {"user": scoped})).status_code == 202
    marker = main._memorize_lock_key(**scoped)
    assert main._FORCED_MEMORIZE_INFLIGHT[marker] is False
    await asyncio.gather(*list(main._BACKGROUND_TASKS))
    state = main._paid_work_state(**scoped)
    failed = outcome in {"failure", "missing_source"}
    assert releases == [not failed] and marker not in main._FORCED_MEMORIZE_INFLIGHT
    assert bool(main._soul_import_state(**scoped)[1]["ordinary_waiting"]) == failed
    assert bool(main._soul_activity_pause(**scoped)) == failed
    assert bool(_endpoint("/imports/status")(**scoped, label="TestApp")["followup_error"]) == failed
    assert state["last_consolidation_at"] == clock
    if outcome == "missing_source":
        assert state["memorize_failure"]["paused"]
        assert competing in state["memorize_failure"]["error"]
    else:
        assert state["memorize_failure"] is None
    assert bool(state["last_consolidation_error"]) == (outcome == "failure")
    assert bool(state["last_consolidation_error_at"]) == (outcome == "failure")
    assert runs == ([True] if failed else [True, False])
    assert [[row["content"] for row in batch] for batch in extracts] == (
        [["Remaining fictional story"]] if outcome == "remaining_source" else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("force,release", [(False, True), (True, False)])
async def test_no_pending_consolidation_does_not_acknowledge_error_outside_retry(monkeypatch, force, release):
    scoped = {"user_id": "TestOwner", "soul_id": "AckSoul"}
    cid = "chat:no-pending"
    error = "Fictional post-commit failure"
    error_at = "2026-01-02T00:00:00Z"
    monkeypatch.setattr(main, "_CONSOLIDATION_RUNNING", {})
    marker = main._memorize_lock_key(**scoped)
    monkeypatch.setattr(main, "_FORCED_MEMORIZE_INFLIGHT", {marker: False})
    monkeypatch.setattr(main, "_FORCED_MEMORIZE_RECHECK", {})
    main._write_conversation_state(cid, **scoped, updates={
        "last_consolidation_error": error, "last_consolidation_error_at": error_at,
    })
    result = await main._run_consolidation_task(
        object(), conversation_id=cid, uid=scoped["user_id"], soul_id=scoped["soul_id"],
        force=force, release_memorize_claim=release,
    )
    assert result == {"ok": True, "status": "skipped", "reason": "no_pending_segments"}
    state = main._paid_work_state(**scoped)
    assert state["last_consolidation_error"] == error and state["last_consolidation_error_at"] == error_at
    assert state["last_consolidation_at"] is None


def test_new_soul_first_preview_initializes_schema_without_model_calls(tmp_path, monkeypatch):
    from app.services import owner

    owner.create_owner(main._CONFIG, "TestOwner")
    source = tmp_path / "imports.db"
    monkeypatch.setattr(conversation_sources, "import_source_path", lambda: source)
    client = TestClient(main.app, client=("127.0.0.1", 50000))
    assert client.post("/souls", json={"soul_id": "FreshImportSoul", "use_existing": False,
                                       "user_name": "TestOwner"}).status_code == 200
    path = main._sqlite_current_path("TestOwner", "FreshImportSoul")
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall() == [("soul_state",)]
        assert con.execute("SELECT user_name FROM soul_state WHERE id = 1").fetchone() == ("TestOwner",)
    monkeypatch.setattr(main, "_build_turn_prompt", lambda **_: pytest.fail("All-history preview built a turn"))
    response = client.post("/imports/validate", json={
        "user_id": "TestOwner", "soul_id": "FreshImportSoul", "label": "Replika",
        "conversation_id": "import:dm:fresh", "current_messages": [],
    })
    assert response.status_code == 200
    assert response.json() == {"ok": True, "estimated_tokens": 0, "input_budget": None,
        "pending_start_day": None, "processed_start_day": None, "processed_end_day": None, "deferred_history": False}
    svc = main._get_service_from_payload({"user": {"user_id": "TestOwner", "soul_id": "FreshImportSoul"}})
    assert not svc._llm_clients and svc._claude_cli_client is None
    svc.database.close()
    assert not source.exists()
    rows, _, _ = chat_import.normalize_messages([{"id": "first", "role": "user", "content": "fictional",
                                                 "timestamp": "2025-01-01"}])
    chat_import.store_upload(source, user_id="TestOwner", soul_id="FreshImportSoul", label="Replika",
                             messages=rows, history_count=1)
    registered = client.post("/imports/register", json={"user_id": "TestOwner", "soul_id": "FreshImportSoul", "label": "REPLIKA"})
    assert registered.status_code == 200 and registered.json()["import_state"]["history_end_index"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["due", "small_broken", "small_summed", "no_gap", "turned_off", "turned_off_summed"])
async def test_auto_memorize_during_import_preserves_eligibility_and_retry_admission(monkeypatch, mode):
    from app.services import import_routes
    scoped = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    cid, import_cid = "chat:trigger", "import:dm:history"
    main._write_conversation_state(import_cid, **scoped, updates={"import_state": {
        "history_end_index": 1, "memorize_cursor": -1, "pending_segment_ids": [],
        "stage": "memorize", "error": None}})
    main._write_conversation_state(cid, **scoped, updates={"memorize_chat": True})
    monkeypatch.setattr(main, "_MIN_CHUNK_TOKENS", 100)
    history = [{"role": "user", "content": "fictional " * (100 if mode == "due" or mode.startswith("turned_off") else 10),
                "ts_ms": 1_735_689_600_000 + i * (1000 if mode == "no_gap" else 86_400_000)} for i in range(2)]
    calls = []
    def build(*_args, **_kwargs):
        calls.append(True)
        if mode.startswith("turned_off"):
            rows = [{**row, "memorize_chat": False} for row in history]
            return {"conversation": rows + ([{"content": "fictional " * 100}]
                if mode == "turned_off_summed" else [])}
        if mode != "small_summed":
            raise OSError("Fictional source unreadable")
        return {"conversation": history + [{"content": "fictional " * 100}]}
    monkeypatch.setattr(main, "_build_cross_conversation_payload", build)
    state = main._load_turn_state_and_soul_card(cid, **scoped)[0]
    safe = {"memorize_chat": False} if mode.startswith("turned_off") else {}
    _tokens, payload = main._prepare_auto_memorize(cid, *scoped.values(), safe, state, history, dry_run=False)
    if mode == "turned_off":
        assert _tokens == 0
    if payload:
        assert main._schedule_auto_memorize(payload, main._auto_memorize_scope(
            cid, *scoped.values(), {}, history)) == "coalesced"
    waiting = main._soul_import_state(**scoped)[1].get("ordinary_waiting")
    assert bool(waiting) == (mode in {"due", "small_summed", "turned_off_summed"})
    assert len(calls) == (0 if mode in {"due", "no_gap"} else 1)
    assert not main._paid_work_state(**scoped).get("memorize_failure")
    assert main._soul_activity_pause("TestOwner", "OtherSoul") is None
    monkeypatch.setattr(conversation_sources, "import_chat_info", lambda **_kw: {
        "conversation_id": import_cid, "label": "TestApp", "title": None})
    async def finish(runtime, **_kwargs):
        await runtime._finish_memorize_claim(runtime._memorize_lock_key(**scoped), True)
    monkeypatch.setattr(import_routes, "run_import", finish)
    response = await _endpoint("/imports/process")(ImportProcess(**scoped, label="TestApp"))
    assert response.status_code == 202
    await asyncio.gather(*tuple(main._BACKGROUND_TASKS))


@pytest.mark.parametrize("prior_segment", [False, True])
@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.asyncio
async def test_registration_uses_segments_not_incidental_memories(tmp_path, monkeypatch, prior_segment, failed):
    source, db = tmp_path / "imports.db", tmp_path / "TestSoul.db"
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul", "label": "Replika"}
    monkeypatch.setattr(conversation_sources, "import_source_path", lambda: source)
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_kw: db)
    monkeypatch.setattr(main, "_get_service_from_payload", lambda _, **_kwargs: None)
    with sqlite3.connect(db) as con:
        _guidance_schema(con)
        con.execute("CREATE TABLE memory_items (user_id TEXT, soul_id TEXT, memory_type TEXT)")
        con.execute("INSERT INTO memory_items VALUES ('TestOwner', 'TestSoul', 'subconscious')")
        if prior_segment:
            con.execute("INSERT INTO resources VALUES ('TestOwner', 'TestSoul', 'chat:earlier', 'conversation')")
    rows, _, _ = chat_import.normalize_messages([{"id": "one", "role": "user", "content": "fictional",
                                                 "timestamp": "2025-01-01"}])
    chat_import.store_upload(source, **scope, messages=rows, history_count=1)
    later, _, _ = chat_import.normalize_messages([{"id": "later", "role": "user", "content": "later file",
                                                 "timestamp": "2025-02-01"}])
    chat_import.store_upload(source, **scope, messages=later, history_count=1)
    scoped = {key: scope[key] for key in ("user_id", "soul_id")}
    failure = {"conversation_id": "chat:failed", "paused": True, "error": "Memorize failed", "targets": {}}
    if failed:
        main._write_conversation_state("chat:failed", **scoped, updates={"memorize_failure": failure})
        with pytest.raises(HTTPException, match="Retry Memorize"):
            _endpoint("/imports/register")(ImportScope(**scope))
        main._write_conversation_state("chat:failed", **scoped, updates={"memorize_failure": None})
    result = _endpoint("/imports/register")(ImportScope(**scope))["import_state"]
    assert result["history_end_index"] == (0 if prior_segment else 1)
    if prior_segment:
        assert result["stage"] == "complete"
        with pytest.raises(HTTPException, match="No eligible history"):
            await _endpoint("/imports/process")(ImportProcess(**scope))
    elif failed:
        main._write_conversation_state("chat:failed", **scoped, updates={"memorize_failure": failure})
        with pytest.raises(HTTPException, match="Retry Memorize"):
            await _endpoint("/imports/process")(ImportProcess(**scope))


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["dedupe", "review"])
@pytest.mark.parametrize("fail,error", [(False, ""), (True, "Review still failed"), (True, "")])
@pytest.mark.parametrize("stage", ["memorize", "consolidation"])
async def test_import_retry_recovers_zero_item_segment_before_pending_shortcut(monkeypatch, phase, fail, error, stage):
    scoped = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    cid = "import:dm:zero"
    main._write_conversation_state(cid, **scoped, updates={"import_state": {
        "history_end_index": 1, "memorize_cursor": 0, "pending_segment_ids": ["saved"],
        "stage": stage, "error": "Review failed"}})
    main._write_conversation_state(cid, **scoped, updates={"import_segment_work": {"saved": phase}})
    engine = SavedBatchService.make_engine(scoped)
    engine.database.resource_repo.create_resource(url="saved", modality="conversation",
        local_path="saved", caption=None, embedding=None, user_data=scoped,
        segment_id="saved", conversation_id=cid)
    for profile in engine.llm_profiles.profiles.values():
        profile.context_window_tokens = 1_000_000
    entered, release = asyncio.Event(), asyncio.Event()
    async def review(state, context, *, enforce_input_budget):
        assert enforce_input_budget and not state["items"]
        assert main._soul_activity_pause(**scoped) == "Import failed. Retry in Echo."
        entered.set()
        await release.wait()
        if fail:
            raise RuntimeError(error)
        return state
    monkeypatch.setattr(engine, "_memorize_persist_and_index", review)
    monkeypatch.setattr(main, "_get_service_from_payload", lambda _payload, **_kwargs: engine)
    monkeypatch.setattr(conversation_sources, "import_chat_info", lambda **_kw: {
        "conversation_id": cid, "label": "TestApp", "title": None})
    def tail(**_kwargs):
        assert stage == "memorize"
        assert not main._soul_import_state(**scoped)[1].get("segment_work")
        return []
    monkeypatch.setattr(conversation_sources, "load_import_tail", tail)
    calls = []
    async def consolidate(**kwargs):
        record = main._soul_import_state(**scoped)[1]
        assert not record.get("segment_work")
        assert record["error"] == ("Review failed" if stage == "consolidation" else None)
        calls.append(kwargs)
        main._write_conversation_state(cid, **scoped, updates={"import_state": {
            **record, "pending_segment_ids": [], "error": None, "stage": "complete"}})
        return {"status": "ok"}
    monkeypatch.setattr(main, "_run_consolidation_pipeline_once", consolidate)
    tasks = main._BACKGROUND_TASKS
    try:
        response = await _endpoint("/imports/retry")(ImportProcess(**scoped, label="TestApp"))
        assert response.status_code == 202
        task = next(task for task in tasks if task.get_name() == f"import:{main._memorize_lock_key(**scoped)}")
        await asyncio.wait_for(entered.wait(), 2)
        assert main._soul_activity_pause("TestOwner", "OtherSoul") is None
        release.set()
        await task
        record = main._soul_import_state(**scoped)[1]
        assert bool(record.get("segment_work")) == fail
        assert bool(calls) != fail
        assert bool(record["error"]) == fail
        if fail:
            assert record["error"] == (error or "RuntimeError")
            assert main._soul_activity_pause(**scoped) == "Import failed. Retry in Echo."
        assert main._memorize_lock_key(**scoped) not in main._FORCED_MEMORIZE_INFLIGHT
    finally:
        release.set()
        await asyncio.gather(*tuple(tasks), return_exceptions=True)
        engine.database.close()


@pytest.mark.parametrize("running", ["memorize", "consolidation"])
def test_first_registration_waits_for_memory_work(tmp_path, monkeypatch, running):
    source, db = tmp_path / "imports.db", tmp_path / "TestSoul.db"
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul", "label": "Replika"}
    monkeypatch.setattr(conversation_sources, "import_source_path", lambda: source)
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_kw: db)
    monkeypatch.setattr(main, "_get_service_from_payload", lambda _, **_kwargs: None)
    with sqlite3.connect(db) as con:
        _guidance_schema(con)
    rows, _, _ = chat_import.normalize_messages([{"id": "one", "role": "user", "content": "fictional",
                                                 "timestamp": "2025-01-01"}])
    chat = chat_import.store_upload(source, **scope, messages=rows, history_count=1)
    marker = main._memorize_lock_key("TestOwner", "TestSoul")
    claims = {marker: False} if running == "memorize" else {("TestOwner", "TestSoul"): False}
    monkeypatch.setattr(main, "_FORCED_MEMORIZE_INFLIGHT" if running == "memorize" else "_CONSOLIDATION_RUNNING", claims)
    register = _endpoint("/imports/register")
    with pytest.raises(HTTPException) as refused:
        register(ImportScope(**scope))
    assert refused.value.status_code == 409
    state, _, _ = main._load_turn_state_and_soul_card(chat["conversation_id"], user_id="TestOwner", soul_id="TestSoul")
    assert state["import_state"] is None
    claims.clear()
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO resources VALUES ('TestOwner', 'TestSoul', 'chat:earlier', 'conversation')")
    assert register(ImportScope(**scope))["import_state"]["history_end_index"] == 0
    claims[marker if running == "memorize" else ("TestOwner", "TestSoul")] = False
    assert register(ImportScope(**scope))["import_state"]["history_end_index"] == 0


def test_registration_keeps_first_file_bound_without_resetting_progress(tmp_path, monkeypatch):
    source, db = tmp_path / "imports.db", tmp_path / "TestSoul.db"
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul", "label": "Replika"}
    monkeypatch.setattr(conversation_sources, "import_source_path", lambda: source)
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_kw: db)
    with sqlite3.connect(db) as con:
        _guidance_schema(con)
    rows, _, _ = chat_import.normalize_messages([
        {"id": str(i), "role": "user", "content": f"message {i}", "timestamp": "2025-01-01"} for i in range(5)])
    upload = chat_import.store_upload(source, **scope, messages=rows, history_count=2)
    register = _endpoint("/imports/register")
    request = ImportScope(**scope)
    result = register(request)
    assert result["import_state"] == {"history_end_index": 2, "memorize_cursor": -1,
        "pending_segment_ids": [], "stage": "memorize", "error": None}
    monkeypatch.setattr(main, "_resolve_cross_source_paths", lambda: (tmp_path, None, None, None))
    preview = _endpoint("/imports/validate")(ImportPreview(**scope, conversation_id=upload["conversation_id"],
        current_messages=[], history_end_index=6))
    assert preview["deferred_history"] is True  # Prospective later history, before any segment exists.
    saved = {**result["import_state"], "memorize_cursor": 1, "pending_segment_ids": ["test-segment"],
             "stage": "consolidation", "error": "Historical failure"}
    main._write_conversation_state(upload["conversation_id"], user_id="TestOwner", soul_id="TestSoul",
                                  updates={"import_state": saved, "digest_cursor": 3,
                                           "last_memorize_at": "2025-01-01T12:00:00Z"})
    chat_import.store_upload(source, **scope, messages=rows, history_count=5)
    assert register(request)["import_state"] == saved
    older, _, _ = chat_import.normalize_messages([
        {"id": "older", "role": "assistant", "content": "older", "timestamp": "2024-01-01"}])
    chat_import.store_upload(source, **scope, messages=older, history_count=1)
    assert register(request)["import_state"] == saved
    # Extraction patches cannot extend eligibility to the later file.
    main._write_conversation_state(upload["conversation_id"], user_id="TestOwner", soul_id="TestSoul",
        updates={"import_error": "Interrupted"})
    main._write_conversation_state(upload["conversation_id"], user_id="TestOwner", soul_id="TestSoul",
        updates={"import_memorize_cursor": 1, "append_import_pending_segment_ids": ["later-segment"], "import_error": None})
    patched = register(request)["import_state"]
    assert patched["history_end_index"] == 2 and patched["memorize_cursor"] == 1
    assert patched["pending_segment_ids"] == ["test-segment", "later-segment"] and patched["error"] is None
    state, _, _ = main._load_turn_state_and_soul_card(upload["conversation_id"], user_id="TestOwner", soul_id="TestSoul")
    assert state["digest_cursor"] == 3 and state["last_memorize_at"] == "2025-01-01T12:00:00Z"
    assert register(request)["import_state"] == patched
    main._write_conversation_state(upload["conversation_id"], user_id="TestOwner", soul_id="TestSoul",
        updates={"import_state": {**patched, "pending_segment_ids": [], "stage": "memorize"}})
    advanced, _ = main._write_conversation_state(upload["conversation_id"], user_id="TestOwner", soul_id="TestSoul",
        updates={"import_memorize_cursor": 1, "import_error": None})
    assert advanced["import_state"]["stage"] == "complete"
    late, _ = main._write_conversation_state(upload["conversation_id"], user_id="TestOwner", soul_id="TestSoul",
        updates={"import_memorize_cursor": 0, "import_error": "Late failure"})
    assert late["import_state"]["memorize_cursor"] == 1 and late["import_state"]["history_end_index"] == 2
    with pytest.raises(HTTPException) as refused:
        register(ImportScope(**{**scope, "soul_id": "OtherSoul"}))
    assert refused.value.status_code == 404


def test_preview_counts_stored_display_and_cross_context_without_calls_or_inserts(tmp_path, monkeypatch):
    workspace = str(tmp_path / "fictional-workspace")
    monkeypatch.setitem(main._CONFIG, "claude_code_workspace", workspace)
    systems = []
    make_system = main._make_turn_system_prompt
    def capture_system(*args, **kwargs):
        text = make_system(*args, **kwargs)
        systems.append(text)
        return text
    monkeypatch.setattr(main, "_make_turn_system_prompt", capture_system)
    source, db = tmp_path / "imports.db", tmp_path / "TestSoul.db"
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul", "label": "Nomi"}
    monkeypatch.setattr(conversation_sources, "import_source_path", lambda: source)
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_kw: db)
    monkeypatch.setattr(main, "_resolve_cross_source_paths", lambda: (tmp_path, None, None, None))
    main._souls.publish_soul_db(db, "TestHuman")
    with sqlite3.connect(db) as con:
        _guidance_schema(con)
    rows, _, _ = chat_import.normalize_messages([
        {"id": str(i), "role": "user", "name": "TestSpeaker", "content": f"stored {i}", "timestamp": "2025-01-01"}
        for i in range(3)])
    upload = chat_import.store_upload(source, **scope, messages=rows, history_count=0)
    cid = upload["conversation_id"]
    _endpoint("/imports/register")(ImportScope(**scope))
    main._write_conversation_state(cid, user_id="TestOwner", soul_id="TestSoul",
        updates={"digest_cursor": 2, "last_memorize_at": "2025-01-01T12:00:00Z",
                 "last_display_segment_start_index": 0, "last_display_segment_end_index": 2,
                 "last_display_segment_at": "2025-01-01T12:00:00Z", "prior_context": [{"id": "pending-memory"}]})
    other = "chat:other-fictional"
    main._write_conversation_state(other, user_id="TestOwner", soul_id="TestSoul", updates={})
    conversation_sources.persist_sillytavern_history_snapshot(
        storage_dir=tmp_path, user_id="TestOwner", soul_id="TestSoul", conversation_id=other,
        history=[{"role": "user", "name": "OtherSpeaker", "content": "another chat " * 200}])
    monkeypatch.setattr(main, "_load_activity_tail_for_ai", lambda *_a, **_kw: [{
        "conversation_id": "activity:dm:TestSoul", "role": "assistant", "content": "sitting recap " * 200}])
    profile = SimpleNamespace(context_window_tokens=100000, max_tokens=1000, chat_model="fictional-model")
    anchors = [SimpleNamespace(anchor_role="soul", summary="I paint coastal sketches."),
               SimpleNamespace(anchor_role="user", summary="My human prefers early walks.")]
    svc = SimpleNamespace(user_config=SimpleNamespace(user_name="TestHuman"),
                          _claude_code=False, llm_profiles=SimpleNamespace(profiles={"default": profile}),
                          list_active_dossiers=lambda _: anchors, build_dossier_index=lambda _, **_kw: "Known dossier index",
                          _get_database=lambda: SimpleNamespace(memory_item_repo=SimpleNamespace(
                              list_items_by_ids=lambda ids, identity: {"pending-memory": {"id": "pending-memory",
                                  "memory_type": "knowledge", "summary": "Current pending memory wording"}})),
                          _model_dump_without_embeddings=dict)
    monkeypatch.setattr(main, "_get_service_from_payload", lambda _, **_kwargs: svc)
    prompts = []
    build = main._build_turn_prompt
    def capture(**kwargs):
        text = build(**kwargs)
        prompts.append(text)
        return text
    monkeypatch.setattr(main, "_build_turn_prompt", capture)
    validate = _endpoint("/imports/validate")
    message = {"role": "user", "content": "proposed", "timestamp": "2025-01-02T12:00:00Z",
               "source_day": "2025-01-02", "position": 3}
    preview = ImportPreview(**scope, conversation_id=cid, current_messages=[message])
    first = validate(preview)
    assert f"- attachment: absolute path inside {workspace}/" in systems[-1]
    assert first["processed_start_day"] == first["processed_end_day"] == "2025-01-01"
    assert first["ok"] and prompts[-1].count("[TestSpeaker] stored 2") == 1
    assert "another chat" in prompts[-1] and "sitting recap" in prompts[-1]
    assert "My Nomi Conversations:" in prompts[-1] and "Known dossier index" in prompts[-1]
    assert "Prior Context:" in prompts[-1] and "Current pending memory wording" in prompts[-1]
    client = TestClient(main.app)
    response = client.post("/imports/validate", json=preview.model_dump(mode="json"))
    assert response.status_code == 200 and response.json()["ok"]
    anchors[0].summary += " sensory details" * 4000
    assert validate(preview)["estimated_tokens"] > first["estimated_tokens"]
    invalid = {**preview.model_dump(mode="json"), "current_messages": [{**message, "role": "system"}]}
    assert client.post("/imports/validate", json=invalid).status_code == 422
    profile.context_window_tokens = 1000 + (first["estimated_tokens"] + 10) * 5 // 4
    with pytest.raises(HTTPException, match="context"):
        validate(preview)
    anchors[0].summary = "I paint coastal sketches."
    larger = ImportPreview(**scope, conversation_id=cid,
                          current_messages=[{**message, "content": "additional words " * 100}])
    with pytest.raises(HTTPException, match="context"):
        validate(larger)
    with sqlite3.connect(source) as con:
        assert con.execute("SELECT COUNT(*) FROM imported_messages").fetchone()[0] == 3


def test_import_guidance_uses_soul_period_but_selected_chat_processed_dates(tmp_path, monkeypatch):
    source, db = tmp_path / "imports.db", tmp_path / "TestSoul.db"
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul", "label": "Replika"}
    monkeypatch.setattr(conversation_sources, "import_source_path", lambda: source)
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_kw: db)
    monkeypatch.setattr(main, "_resolve_cross_source_paths", lambda: (tmp_path, None, None, None))
    rows, _, _ = chat_import.normalize_messages([
        {"id": str(i), "role": "user", "content": "fictional", "timestamp": day}
        for i, day in enumerate(("2020-01-01", "2025-01-01", "2025-03-01", "2025-02-01"))])
    upload = chat_import.store_upload(source, **scope, messages=rows, history_count=1)
    cid = upload["conversation_id"]
    with sqlite3.connect(db) as con:
        _guidance_schema(con)
        # An existing conversation segment makes the upload deferred history.
        con.execute("INSERT INTO resources VALUES (?, ?, ?, ?)",
                    ("TestOwner", "TestSoul", cid, "conversation"))
    _endpoint("/imports/register")(ImportScope(**scope))
    main._write_conversation_state(cid, user_id="TestOwner", soul_id="TestSoul",
        updates={"digest_cursor": 1, "last_memorize_at": "2025-01-02T12:00:00Z"})
    # All-history guidance must not build a turn or call a model.
    monkeypatch.setattr(main, "_get_service_from_payload", lambda _, **_kwargs: None)
    monkeypatch.setattr(main, "_build_turn_prompt", lambda **_: pytest.fail("All-history guidance built a turn"))
    validate = _endpoint("/imports/validate")
    result = validate(ImportPreview(**scope, conversation_id=cid, current_messages=[]))
    assert result == {"ok": True, "estimated_tokens": 0, "input_budget": None,
        "pending_start_day": "2025-02-01", "processed_start_day": "2025-01-01", "processed_end_day": "2025-01-01",
        "deferred_history": True}
    main._write_conversation_state(cid, user_id="TestOwner", soul_id="TestSoul", updates={"digest_cursor": 3})
    assert validate(ImportPreview(**scope, conversation_id=cid, current_messages=[]))["pending_start_day"] is None
    other = validate(ImportPreview(**{**scope, "label": "Nomi"}, conversation_id="import:dm:new", current_messages=[]))
    assert other["pending_start_day"] is other["processed_start_day"] is other["processed_end_day"] is None
    other_cid = "chat:other"
    main._write_conversation_state(other_cid, user_id="TestOwner", soul_id="TestSoul", updates={})
    conversation_sources.persist_sillytavern_history_snapshot(
        storage_dir=tmp_path, user_id="TestOwner", soul_id="TestSoul", conversation_id=other_cid,
        history=[{"role": "user", "content": "fictional other chat", "ts_ms": 1_704_067_200_000}])
    result = validate(ImportPreview(**scope, conversation_id=cid, current_messages=[]))
    assert result["pending_start_day"] == datetime.fromtimestamp(1_704_067_200).date().isoformat()
    assert result["processed_start_day"] == "2025-01-01" and result["processed_end_day"] == "2025-03-01"
    snapshot = conversation_sources._chat_snapshot_path(
        storage_dir=tmp_path, user_id="TestOwner", soul_id="TestSoul", conversation_id=other_cid,
        source_label="sillytavern")
    for memorize_chat in (True, False):
        main._write_conversation_state(other_cid, user_id="TestOwner", soul_id="TestSoul",
                                       updates={"memorize_chat": memorize_chat})
        for raw in ("{", "[]", '{"history":{}}', None):
            if raw is None:
                snapshot.unlink()
            else:
                snapshot.write_text(raw, encoding="utf-8")
            with pytest.raises(HTTPException) as refused:
                validate(ImportPreview(**scope, conversation_id=cid, current_messages=[]))
            assert refused.value.status_code == 409 and other_cid in refused.value.detail and "unavailable" in refused.value.detail
            with sqlite3.connect(db) as con:
                con.row_factory = sqlite3.Row
                assert main._load_cross_memorize_tails_from_sources(con, user_id="TestOwner", soul_id="TestSoul") == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["calendar", "calendar_retry", "capacity", "continuous", "cancel_extraction", "failure", "error_write", "consolidation_failure", "missing_capacity"])
async def test_import_batch_reuses_extraction_and_one_consolidation(tmp_path, monkeypatch, stop):
    import asyncio
    from app.services import memorize_endpoint
    source, db = tmp_path / "imports.db", tmp_path / "TestSoul.db"
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul", "label": "Replika"}
    scoped = {key: scope[key] for key in ("user_id", "soul_id")}
    monkeypatch.setattr(conversation_sources, "import_source_path", lambda: source)
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_kw: db)
    monkeypatch.setattr(memorize_endpoint, "_FORCE_MEMORIZE_MAX_CHUNK_TOKENS", 80)
    with sqlite3.connect(db) as con:
        sqlite_ensure_conversation_state_schema(con)
    SavedBatchService.make_engine(scoped).database.close()
    main._write_conversation_state("chat:import-background", **scoped, updates={
        "memorize_chat": False, "rolling_summary": "Import supporting summary"})
    rows, _, _ = chat_import.normalize_messages([
        {"id": str(i), "role": "user", "content": "fictional story " * 20,
         "timestamp": "2025-01-09" if stop in {"calendar", "calendar_retry", "no_memories_calendar"} and i else "2025-01-01"}
        for i in range(3)])
    stored = chat_import.store_upload(source, **scope, messages=rows, history_count=3)
    request = ImportProcess(**scope, continuous=stop in {"continuous", "cancel_extraction"})
    _endpoint("/imports/register")(request)
    cid = stored["conversation_id"]
    marker = main._memorize_lock_key(**scoped)
    extracts, consolidations, releases = [], [], []
    fail = stop in {"failure", "calendar_retry", "error_write", "consolidation_failure"}
    if stop == "error_write":
        write = main._write_conversation_state
        def failing_write(*args, **kwargs):
            if str(kwargs.get("updates", {}).get("import_error", "")).startswith("Fictional"):
                raise sqlite3.OperationalError("Fictional error-write lock")
            return write(*args, **kwargs)
        monkeypatch.setattr(main, "_write_conversation_state", failing_write)
    profile = SimpleNamespace(context_window_tokens=100, max_tokens=0, chat_model="fictional")
    profiles = {"default": profile}
    if stop == "missing_capacity":
        profiles["consolidation"] = SimpleNamespace(context_window_tokens=None, max_tokens=0, chat_model="fictional")
    class Service(SavedBatchService):
        memorize_config = SimpleNamespace(category_update_llm_profile="default")
        llm_profiles = SimpleNamespace(profiles=profiles)
        async def extract(self, **kwargs):
            assert kwargs["enforce_input_budget"] is True
            assert main._FORCED_MEMORIZE_INFLIGHT[marker] is True
            extracts.append([row["segment"]["segment_id"] for row in kwargs["segments"]])
            if stop == "cancel_extraction" and len(extracts) == 2:
                assert (await main.memorize_cancel(scoped))["status"] == "cancel_requested"
            with pytest.raises(HTTPException) as busy:
                await _endpoint("/imports/process")(request)
            assert busy.value.status_code == 409
            if stop in {"failure", "calendar_retry", "error_write"} and fail and len(extracts) == 2:
                raise ValueError("Fictional extraction failure")
            return [{"pending_segment_ids": [row["segment"]["segment_id"]]} for row in kwargs["segments"]]
    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_, **_kwargs: Service())
    def gather(_deps, **kwargs):
        pairs = kwargs["selected_segments"]
        assert all(owner == cid for owner, _ in pairs)
        return {"segment_inputs": [{} for _ in pairs], "current_chat_messages": [
            {"source_day": rows[int(segment_id.rsplit(":", 1)[1].split("-")[0])]["source_day"]}
            for _owner, segment_id in pairs]}
    monkeypatch.setattr(consolidation, "gather_consolidation_inputs", gather)
    def prepare(*_a, **kw):
        assert "Import supporting summary" in kw["inputs"]["all_chat_history"]
        return ([], "", "", {"dossiers": 60 if stop in {"capacity", "continuous", "cancel_extraction", "consolidation_failure"}
                and len(kw["inputs"]["segment_inputs"]) >= 2 else 10})
    monkeypatch.setattr(consolidation, "_prepare_dossier_consolidation_prompts", prepare)
    def record():
        return main._load_turn_state_and_soul_card(cid, **scoped)[0]["import_state"]
    entered, finish = asyncio.Event(), asyncio.Event()
    async def pipeline(**kwargs):
        assert kwargs["historical"] and main._FORCED_MEMORIZE_INFLIGHT[marker] is True
        consolidations.append(kwargs["selected_segments"])
        assert kwargs["selected_segments"] == {(cid, segment_id) for segment_id in record()["pending_segment_ids"]}
        if stop not in {"continuous", "cancel_extraction"}:
            assert (await main.memorize_cancel(scoped))["status"] == "cancel_requested"
        entered.set()
        await finish.wait()
        if stop == "consolidation_failure" and fail:
            raise ValueError("Fictional consolidation failure")
        current = record()
        # Simulate the existing fitting-prefix selector leaving one pending span.
        remaining = current["pending_segment_ids"][1:]
        main._write_conversation_state(cid, **scoped, updates={"import_state": {
            **current, "pending_segment_ids": remaining, "error": None,
            "stage": "consolidation" if remaining else "memorize",
        }})
        return {"status": "ok", "result": {"revised": 1}}
    monkeypatch.setattr(main, "_run_consolidation_pipeline_once", pipeline)
    release = main._finish_memorize_claim
    async def finished(key, success):
        assert key not in main._MEMORIZE_CANCEL
        releases.append((key, success))
        await release(key, success)
    monkeypatch.setattr(main, "_finish_memorize_claim", finished)
    response = await _endpoint("/imports/process")(request)
    assert response.status_code == 202
    tasks = list(main._BACKGROUND_TASKS)
    if stop == "missing_capacity":
        recoveries = []
        async def resume(**kwargs):
            recoveries.append(kwargs)
        monkeypatch.setattr(Service, "resume_memorize_segment", staticmethod(resume))
        await asyncio.gather(*tasks)
        assert not recoveries and not extracts and not consolidations
        assert record()["error"] == "context_window_tokens is required for model fictional"
        assert record()["stage"] == "memorize"
        assert record()["memorize_cursor"] == -1 and releases == [(marker, False)]
        assert marker not in main._FORCED_MEMORIZE_INFLIGHT
        for stage in ("memorize", "consolidation"):
            main._write_conversation_state(cid, **scoped, updates={
                "import_state": {**record(), "stage": stage, "pending_segment_ids": ["saved"],
                                 "segment_work": {"saved": "review"}},
            })
            response = await _endpoint("/imports/retry")(request)
            assert response.status_code == 202
            await asyncio.gather(*list(main._BACKGROUND_TASKS))
            assert not recoveries and not extracts and not consolidations
            assert record()["stage"] == stage
            assert record()["error"] == "context_window_tokens is required for model fictional"
            assert record()["segment_work"] == {"saved": "review"}
            assert record()["pending_segment_ids"] == ["saved"] and record()["memorize_cursor"] == -1
            assert releases[-1] == (marker, False) and marker not in main._FORCED_MEMORIZE_INFLIGHT
        return
    if stop not in {"failure", "calendar_retry", "error_write"}:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(HTTPException) as busy:
            await _endpoint("/imports/process")(request)
        assert busy.value.status_code == 409
        finish.set()
    await asyncio.gather(*tasks)
    assert len(extracts) == (3 if stop == "continuous" else 2) and marker not in main._FORCED_MEMORIZE_INFLIGHT
    assert releases == [(marker, stop not in {"failure", "calendar_retry", "error_write", "consolidation_failure"})]
    if stop in {"failure", "calendar_retry", "error_write"}:
        assert record()["stage"] == "memorize" and record()["error"]
        assert record()["memorize_cursor"] == 0 and not consolidations
        with pytest.raises(HTTPException, match="Retry"):
            await _endpoint("/imports/process")(request)
        fail = False
        finish.set()
        await _endpoint("/imports/retry")(request)
        await asyncio.gather(*list(main._BACKGROUND_TASKS))
        assert len(extracts) == (3 if stop == "calendar_retry" else 4) and len(consolidations) == 1
    elif stop == "consolidation_failure":
        assert record()["stage"] == "consolidation" and record()["error"]
        assert len(consolidations) == 1
        with pytest.raises(HTTPException, match="Retry"):
            await _endpoint("/imports/process")(request)
        fail = False
        await _endpoint("/imports/retry")(request)
        await asyncio.gather(*list(main._BACKGROUND_TASKS))
        assert len(extracts) == 2 and len(consolidations) == 2
    elif stop == "cancel_extraction":
        assert len(consolidations) == 1 and record()["pending_segment_ids"]
    elif stop == "continuous":
        assert record()["stage"] == "complete" and len(consolidations) == 3
    else:
        assert len(consolidations) == 1 and record()["pending_segment_ids"]
        await _endpoint("/imports/process")(request)
        await asyncio.gather(*list(main._BACKGROUND_TASKS))
        assert len(extracts) == 2 and len(consolidations) == 2
    assert marker not in main._FORCED_MEMORIZE_INFLIGHT
    status = _endpoint("/imports/status")(**scope)
    assert not status["running"] and status["progress"]["active"] is False
