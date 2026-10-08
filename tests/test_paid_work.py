import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks, HTTPException

from app import main
from tests import SavedBatchService
from app.services import soul_state
from app.services import consolidation, service_factory


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", [False, True])
async def test_registered_import_handoff_keeps_history_out_of_normal_memorize(monkeypatch, tail):
    uid, sid, cid = "TestOwner", "TestSoul", "replika:test-import"
    main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={"import_state": {
        "history_end_index": 3, "memorize_cursor": -1, "pending_segment_ids": [],
        "stage": "memorize", "error": None,
    }})
    calls = []
    class Service(SavedBatchService):
        async def extract(self, **kwargs):
            calls.append(kwargs)
            return [{"pending_segment_ids": [row["segment"]["segment_id"]]} for row in kwargs["segments"]]
    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_: Service())
    storage = main._get_storage_dir(main._CONFIG)
    source = dict(storage_dir=storage, user_id=uid, soul_id=sid, conversation_id=cid, source_label="replika")
    history = [{"role": "user", "content": text} for text in ("Old one", "", "Old two", "Current tail")]
    main._conversation_sources.persist_chat_history_snapshot(**source, history=history)
    def read(record, *, historical, cursor):
        return main._conversation_sources.load_chat_snapshot_tail(
            **source, since_cursor=cursor, recent_fallback_messages=0,
            import_state=record, historical=historical,
        )
    state, _, _ = main._load_turn_state_and_soul_card(cid, user_id=uid, soul_id=sid)
    prefix = read(state["import_state"], historical=True, cursor=-1)
    assert [row["source_conversation_index"] for row in prefix] == [0, 2]
    payload = {"user": {"user_id": uid, "soul_id": sid}, "conversation_id": cid, "conversation": prefix}
    tasks = BackgroundTasks()
    response = await main._memorize_owned(payload, tasks, True, historical=True, tail=tail)
    assert response.status_code == 202
    await tasks()
    state, _, _ = main._load_turn_state_and_soul_card(cid, user_id=uid, soul_id=sid)
    import_state = state["import_state"]
    assert import_state["memorize_cursor"] == 2 and len(import_state["pending_segment_ids"]) == 1
    assert state["pending_segment_ids"] == [] and state["last_memorize_at"] is None
    assert read(import_state, historical=True, cursor=2) == []
    tasks = BackgroundTasks()
    stale_tail = read(import_state, historical=True, cursor=0)
    response = await main._memorize_owned({**payload, "conversation": stale_tail}, tasks, True, historical=True, tail=tail)
    assert response.status_code == 200
    await tasks()
    tasks = BackgroundTasks()
    current = read(import_state, historical=False, cursor=-1)
    assert [row["source_conversation_index"] for row in current] == [3]
    await main.memorize({**payload, "conversation": current}, tasks, True, tail=tail)
    await tasks()
    assert [[row["content"] for segment in call["segments"] for row in json.loads(segment["raw_text"])]
            for call in calls] == [["Old one", "Old two"], ["Current tail"]]
    state, _, _ = main._load_turn_state_and_soul_card(cid, user_id=uid, soul_id=sid)
    assert state["import_state"] == import_state
    assert state["digest_cursor"] == 3 and len(state["pending_segment_ids"]) == 1


def test_pause_record_survives_restart_and_retry_without_pausing_other_soul():
    state = soul_state.defaults()
    state["memorize_failure"] = {
        "conversation_id": "chat:saved-chat", "error": "Memorize failed",
        "paused": False, "targets": {"chat:saved-chat": {"cursor": 2}},
    }
    assert soul_state.activity_pause(state, memorize_running=True, consolidation_running=False) is None
    assert soul_state.activity_pause(state, memorize_running=False, consolidation_running=False)
    state["memorize_failure"]["paused"] = True
    assert soul_state.activity_pause(state, memorize_running=True, consolidation_running=False)
    other = soul_state.defaults()
    assert soul_state.activity_pause(other, memorize_running=False, consolidation_running=False) is None
    state.update(last_consolidation_error="Reflection failed", last_consolidation_error_at=datetime.now(UTC).isoformat())
    state["memorize_failure"] = None
    assert soul_state.activity_pause(state, memorize_running=False, consolidation_running=True) == "Reflection failed"


def test_atomic_profile_gates_only_paused_soul_before_returning_credentials(monkeypatch):
    from fastapi.testclient import TestClient

    main._write_conversation_state("chat:saved-chat", user_id="TestOwner", soul_id="TestSoul", updates={
        "memorize_failure": {"conversation_id": "chat:saved-chat", "error": "Failed", "paused": True, "targets": {}},
    })
    calls = []
    monkeypatch.setattr(main, "_atomic_chat_settings_from_config", lambda _config: calls.append(True) or {"api_key": "fake"})
    client = TestClient(main.app)
    response = client.get("/integration/atomic/chat_profile", params={"user_id": "TestOwner", "soul_id": "TestSoul"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "soul_paused"
    assert not calls
    response = client.get("/integration/atomic/chat_profile", params={"user_id": "TestOwner", "soul_id": "OtherSoul"})
    assert response.status_code == 200 and calls == [True]
    assert client.get("/integration/atomic/chat_profile").status_code == 422


@pytest.mark.asyncio
async def test_invalid_memorize_source_is_rejected_before_admission(monkeypatch):
    monkeypatch.setattr(main, "_safe_payload", lambda payload: payload)
    with pytest.raises(HTTPException) as error:
        await main.memorize({"user": {"user_id": "TestOwner", "soul_id": "TestSoul"}, "conversation": []}, BackgroundTasks(), True)
    assert error.value.status_code == 400
    assert error.value.detail == "conversation_id is required"
    with pytest.raises(HTTPException, match="supported history reader"):
        await main.memorize({"user": {"user_id": "TestOwner", "soul_id": "TestSoul"}, "conversation_id": "hermes:unsupported"}, BackgroundTasks(), True)
    marker = main._memorize_lock_key("TestOwner", "TestSoul")
    main._FORCED_MEMORIZE_INFLIGHT[marker] = False
    with pytest.raises(HTTPException, match="supported history reader"):
        await main._memorize_admitted({"user": {"user_id": "TestOwner", "soul_id": "TestSoul"}, "conversation_id": "hermes:unsupported"}, BackgroundTasks(), True)
    assert marker not in main._FORCED_MEMORIZE_INFLIGHT


@pytest.mark.asyncio
async def test_admitted_stale_turn_cannot_hide_interrupted_record(monkeypatch):
    uid, sid, cid = "TestOwner", "TestSoul", "chat:stale"
    failure = {"conversation_id": cid, "error": "Interrupted", "paused": False, "targets": {cid: {"cursor": 1}}}
    main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={"memorize_failure": failure})
    marker = main._memorize_lock_key(uid, sid)
    main._FORCED_MEMORIZE_INFLIGHT[marker] = False
    monkeypatch.setattr(main._memorize_endpoint, "memorize_endpoint", lambda *_args, **_kwargs: pytest.fail("Must not schedule paid work"))
    with pytest.raises(HTTPException) as blocked:
        await main._memorize_admitted({"user": {"user_id": uid, "soul_id": sid}, "conversation_id": cid}, BackgroundTasks(), True)
    assert blocked.value.detail["code"] == "soul_paused"
    assert marker not in main._FORCED_MEMORIZE_INFLIGHT


@pytest.mark.asyncio
async def test_direct_and_tool_retrieve_refuse_paused_scope(monkeypatch):
    uid, sid, cid = "TestOwner", "TestSoul", "chat:paused"
    main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={"memorize_failure": {
        "conversation_id": cid, "error": "Failed", "paused": True, "targets": {cid: {"cursor": 1}},
    }})
    monkeypatch.setattr(main._retrieve_orchestration, "_run_retrieve", lambda *_args, **_kwargs: pytest.fail("Must not retrieve"))
    with pytest.raises(HTTPException) as blocked:
        await main.retrieve({"where": {"user_id": uid, "soul_id": sid}, "queries": ["Test"]})
    assert blocked.value.detail["code"] == "soul_paused"
    with pytest.raises(HTTPException) as tool_blocked:
        await main.mcp_memu_retrieve(main._mcp_tools.MemuRetrieveRequest(user_id=uid, soul_id=sid, query="Test"))
    assert tool_blocked.value.detail["code"] == "soul_paused"


@pytest.mark.asyncio
async def test_memorize_uses_one_scope_for_admission_runner_and_release(monkeypatch):
    uid, sid = "TestOwner", "TestSoul"
    async def endpoint(payload, *_args, **_kwargs):
        assert payload["user"]["soul_id"] == sid
        assert main._memorize_lock_key(uid, sid) in main._FORCED_MEMORIZE_INFLIGHT
        return {"ok": True}
    monkeypatch.setattr(main._memorize_endpoint, "memorize_endpoint", endpoint)
    await main.memorize({"user_id": uid, "soul_id": sid,
                        "user": {"user_id": uid, "soul_id": "OtherSoul"}, "conversation_id": "chat:scope"}, BackgroundTasks(), True)
    assert main._memorize_lock_key(uid, sid) not in main._FORCED_MEMORIZE_INFLIGHT


@pytest.mark.asyncio
async def test_waiting_consolidation_retry_never_lifts_pause(monkeypatch):
    uid, sid, cid = "TestOwner", "TestSoul", "chat:interrupted"
    main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={
        "last_consolidation_error": soul_state.CONSOLIDATION_UNFINISHED,
        "last_consolidation_error_at": datetime.now(UTC).isoformat(),
    })
    lock = main._get_memorize_lock(main._memorize_lock_key(uid, sid))
    await lock.acquire()
    retry = asyncio.create_task(main._run_consolidation_pipeline_once(
        svc=object(), deps=main._make_consolidation_deps(), state_lock=lock,
        running=main._CONSOLIDATION_RUNNING, load_cross_tail_for_ai=lambda **_kwargs: [],
        format_all_chat_history_for_ai=lambda **_kwargs: '',
        conversation_id=cid, soul_id=sid, user_id=uid, force=True,
    ))
    await asyncio.sleep(0)
    try:
        assert (uid, sid) in main._CONSOLIDATION_RUNNING
        assert main._soul_activity_pause(uid, sid)
        with pytest.raises(HTTPException):
            main._require_soul_active(uid, sid)
    finally:
        retry.cancel()
        with pytest.raises(asyncio.CancelledError):
            await retry
        lock.release()


@pytest.mark.asyncio
async def test_cross_checkpoint_failure_retains_published_history_and_retry_uses_remaining_source(monkeypatch):
    uid, sid, first, second = "TestOwner", "TestSoul", "chat:audit-a", "chat:audit-b"
    history = [{"role": "user", "content": "Test", "ts_ms": 1}, {"role": "assistant", "content": "Reply", "ts_ms": 2}]
    for cid in (first, second):
        main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={"memorize_chat": True})
        main._conversation_sources.persist_sillytavern_history_snapshot(
            storage_dir=main._get_storage_dir(main._CONFIG), user_id=uid, soul_id=sid,
            conversation_id=cid, history=history,
        )
    payload = main._build_cross_conversation_payload(first, uid, sid, {}, history, -1)
    real_write = main._write_conversation_state
    calls = []
    class Service(SavedBatchService):
        async def extract(self, **kwargs):
            calls.append(kwargs)
            return [{"pending_segment_ids": [segment["segment"]["segment_id"]]} for segment in kwargs["segments"]]
    def write(cid, **kwargs):
        if cid == second and "digest_cursor" in kwargs["updates"]:
            raise RuntimeError("Checkpoint failed")
        return real_write(cid, **kwargs)
    monkeypatch.setattr(main, "_write_conversation_state", write)
    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_args: Service())
    monkeypatch.setattr(main, "_run_consolidation_task", AsyncMock(return_value={"status": "skipped"}))
    tasks = BackgroundTasks()
    await main.memorize(payload, tasks, True)
    with pytest.raises(RuntimeError, match="Checkpoint failed"):
        await tasks()
    state, _, _ = main._load_turn_state_and_soul_card(first, user_id=uid, soul_id=sid)
    assert not state["pending_segment_ids"]
    assert not any(Path(segment["local_path"]).is_file() for segment in calls[0]["segments"])
    assert main._paid_work_state(uid, sid)["memorize_failure"]
    monkeypatch.setattr(main, "_write_conversation_state", real_write)
    reader = main._cross_history._load_tail_for_source_conversation
    def unavailable_second(**kwargs):
        if kwargs["conversation_id"] == second:
            raise FileNotFoundError("Fictional source unavailable")
        return reader(**kwargs)
    monkeypatch.setattr(main._cross_history, "_load_tail_for_source_conversation", unavailable_second)
    with pytest.raises(HTTPException, match="unavailable"):
        await main.retry_memorize(uid, sid, BackgroundTasks())
    assert len(calls) == 1 and second in main._paid_work_state(uid, sid)["memorize_failure"]["targets"]
    monkeypatch.setattr(main._cross_history, "_load_tail_for_source_conversation", reader)
    main._conversation_sources.persist_sillytavern_history_snapshot(
        storage_dir=main._get_storage_dir(main._CONFIG), user_id=uid, soul_id=sid,
        conversation_id=second, history=history[:1],
    )
    tasks = BackgroundTasks()
    await main.retry_memorize(uid, sid, tasks)
    with pytest.raises(RuntimeError, match="unfinished source checkpoints"):
        await tasks()
    assert main._paid_work_state(uid, sid)["memorize_failure"]["targets"][second]["cursor"] == 1
    main._conversation_sources.persist_sillytavern_history_snapshot(
        storage_dir=main._get_storage_dir(main._CONFIG), user_id=uid, soul_id=sid,
        conversation_id=second, history=history,
    )
    third = "chat:audit-new"
    main._write_conversation_state(third, user_id=uid, soul_id=sid, updates={"memorize_chat": True})
    main._conversation_sources.persist_sillytavern_history_snapshot(
        storage_dir=main._get_storage_dir(main._CONFIG), user_id=uid, soul_id=sid,
        conversation_id=third, history=history,
    )
    def fail_later_checkpoint(cid, **kwargs):
        if cid == third and "digest_cursor" in kwargs["updates"]:
            raise RuntimeError("Later checkpoint failed")
        return real_write(cid, **kwargs)
    monkeypatch.setattr(main, "_write_conversation_state", fail_later_checkpoint)
    tasks = BackgroundTasks()
    await main.retry_memorize(uid, sid, tasks)
    await tasks()
    assert "Later checkpoint failed" in main._paid_work_state(uid, sid)["memorize_failure"]["error"]
    state, _, _ = main._load_turn_state_and_soul_card(first, user_id=uid, soul_id=sid)
    assert calls[-1]["segments"][0]["segment"]["segment_id"] not in state["pending_segment_ids"]
    assert not any(Path(segment["local_path"]).is_file() for segment in calls[-1]["segments"])
    monkeypatch.setattr(main, "_write_conversation_state", real_write)
    tasks = BackgroundTasks()
    await main.retry_memorize(uid, sid, tasks)
    await tasks()
    assert len(calls) == 4 and calls[-1]["conversation_id"] == first
    assert main._paid_work_state(uid, sid)["memorize_failure"] is None
    assert main._memorize_targets_complete(uid, sid, payload["_final_cursors"])
    state, _, _ = main._load_turn_state_and_soul_card(first, user_id=uid, soul_id=sid)
    assert calls[-1]["segments"][0]["segment"]["segment_id"] in state["pending_segment_ids"]


@pytest.mark.asyncio
async def test_retry_assembles_remaining_activity_under_original_owner(monkeypatch):
    uid, sid, cid = "TestOwner", "TestSoul", "chat:activity-owner"
    main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={"digest_cursor": 0, "last_memorize_at": datetime.now(UTC).isoformat()})
    main._conversation_sources.persist_sillytavern_history_snapshot(
        storage_dir=main._get_storage_dir(main._CONFIG), user_id=uid, soul_id=sid, conversation_id=cid,
        history=[{"role": "user", "content": "Already consumed"}],
    )
    main._record_activity_message(user_id=uid, soul_id=sid, recap="Completed a fictional task")
    activity = main._activity_messages.activity_conversation_id(sid)
    targets = {activity: {"cursor": 1, "memory_producing": True}}
    main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={"memorize_failure": {
        "conversation_id": cid, "error": "Failed", "paused": True, "targets": targets,
    }})
    class Service(SavedBatchService):
        async def extract(self, **kwargs):
            assert kwargs["conversation_id"] == cid
            assert "Completed a fictional task" in kwargs["segments"][0]["raw_text"]
            return [{"pending_segment_ids": [segment["segment"]["segment_id"]]} for segment in kwargs["segments"]]
    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_args: Service())
    monkeypatch.setattr(main, "_run_consolidation_task", AsyncMock(return_value={"status": "skipped"}))
    tasks = BackgroundTasks()
    await main.retry_memorize(uid, sid, tasks)
    await tasks()
    assert main._memorize_targets_complete(uid, sid, targets)
    assert main._paid_work_state(uid, sid)["memorize_failure"] is None
    state, _, _ = main._load_turn_state_and_soul_card(cid, user_id=uid, soul_id=sid)
    assert state["pending_segment_ids"] and all(item.startswith(cid + ':') for item in state["pending_segment_ids"])


@pytest.mark.asyncio
@pytest.mark.parametrize("recovering", [False, True])
async def test_context_only_remainder_recovers_existing_failure_without_creating_new_one(monkeypatch, recovering):
    uid, sid, cid, background = "TestOwner", "TestSoul", "chat:context-owner", "chat:background"
    main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={"digest_cursor": 0, "last_memorize_at": datetime.now(UTC).isoformat()})
    main._conversation_sources.persist_sillytavern_history_snapshot(
        storage_dir=main._get_storage_dir(main._CONFIG), user_id=uid, soul_id=sid, conversation_id=cid,
        history=[{"role": "user", "content": "Already consumed"}],
    )
    main._write_conversation_state(background, user_id=uid, soul_id=sid, updates={"memorize_chat": False})
    main._conversation_sources.persist_sillytavern_history_snapshot(
        storage_dir=main._get_storage_dir(main._CONFIG), user_id=uid, soul_id=sid, conversation_id=background,
        history=[{"role": "user", "content": "Earlier context"}, {"role": "assistant", "content": "Remaining context"}],
    )
    targets = {background: {"cursor": 1, "memory_producing": False}}
    if recovering:
        main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={
            "memorize_failure": {"conversation_id": cid, "error": "Failed", "paused": True, "targets": targets},
            "last_consolidation_error": "Reflection failed", "last_consolidation_error_at": datetime.now(UTC).isoformat(),
        })
    class Service(SavedBatchService):
        async def extract(self, **kwargs):
            failure = main._paid_work_state(uid, sid)["memorize_failure"]
            assert (failure is not None and failure["paused"]) if recovering else failure is None
            assert all(segment["segment"]["context_only"] for segment in kwargs["segments"])
            return [{"context_only": True} for _segment in kwargs["segments"]]
    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_args: Service())
    monkeypatch.setattr(main, "_run_consolidation_task", AsyncMock(return_value={"status": "skipped"}))
    tasks = BackgroundTasks()
    if recovering:
        await main.retry_memorize(uid, sid, tasks)
    else:
        payload = main._build_cross_conversation_payload(cid, uid, sid, {}, [], 0)
        await main.memorize(payload, tasks, True)
    await tasks()
    assert main._memorize_targets_complete(uid, sid, targets)
    assert main._paid_work_state(uid, sid)["memorize_failure"] is None
    if recovering:
        assert main._soul_activity_pause(uid, sid) == "Reflection failed"


@pytest.mark.asyncio
async def test_direct_checkpoint_cannot_commit_without_pending_bookkeeping(monkeypatch):
    uid, sid, cid = "TestOwner", "TestSoul", "chat:bookkeeping"
    main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={})
    con = main._sqlite_connect(main._sqlite_current_path(uid, sid))
    con.execute("CREATE TRIGGER reject_pending BEFORE UPDATE OF pending_segment_ids ON conversations "
                "WHEN NEW.pending_segment_ids IS NOT OLD.pending_segment_ids "
                "BEGIN SELECT RAISE(ABORT, 'Bookkeeping failed'); END")
    con.commit()
    con.close()
    class Service(SavedBatchService):
        async def extract(self, **kwargs):
            return [{"pending_segment_ids": [segment["segment"]["segment_id"]]} for segment in kwargs["segments"]]
    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_args: Service())
    tasks = BackgroundTasks()
    await main.memorize({"user": {"user_id": uid, "soul_id": sid}, "conversation_id": cid,
                        "conversation": [{"role": "user", "content": "Test"}]}, tasks, True)
    with pytest.raises(sqlite3.IntegrityError, match="Bookkeeping failed"):
        await tasks()
    state, _, _ = main._load_turn_state_and_soul_card(cid, user_id=uid, soul_id=sid)
    assert main._effective_digest_cursor_from_row(state) == -1
    assert state["memorize_failure"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_automatic_admission_reaches_real_endpoint_once(monkeypatch, cancel):
    uid, sid, cid = "TestOwner", "TestSoul", "chat:saved-chat"
    main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={})
    monkeypatch.setattr(main, "_safe_payload", lambda payload: payload)
    marker = main._memorize_lock_key(uid, sid)
    calls = []

    class Service(SavedBatchService):
        async def extract(self, **kwargs):
            calls.append(kwargs)
            assert marker in main._FORCED_MEMORIZE_INFLIGHT
            assert main._paid_work_state(uid, sid)["memorize_failure"]
            assert main._soul_activity_pause(uid, sid) is None
            return [{"memory_item_ids": []} for _ in kwargs["segments"]]

    monkeypatch.setattr(main, "_get_service_from_payload", lambda _payload: Service())
    payload = {
        "user": {"user_id": uid, "soul_id": sid, "conversation_id": cid},
        "conversation_id": cid,
        "conversation": [{"role": "user", "content": "First"}, {"role": "assistant", "content": "Second"}],
    }
    main._FORCED_MEMORIZE_INFLIGHT[marker] = False
    if cancel:
        main._MEMORIZE_CANCEL.add(marker)
    try:
        await main._run_forced_memorize_from_turn(payload)
        assert len(calls) == int(not cancel)
        assert marker not in main._FORCED_MEMORIZE_INFLIGHT
        assert main._paid_work_state(uid, sid)["memorize_failure"] is None
    finally:
        main._FORCED_MEMORIZE_INFLIGHT.pop(marker, None)
        main._MEMORIZE_CANCEL.discard(marker)


@pytest.mark.asyncio
@pytest.mark.parametrize("import_pending", [False, True])
async def test_automatic_admission_failure_before_jobs_pauses_and_exposes_retry(monkeypatch, import_pending):
    uid, sid, cid = "TestOwner", "TestSoul", "chat:failed-admission"
    unreadable = "chat:unreadable-source"
    storage = main._get_storage_dir(main._CONFIG)
    monkeypatch.setattr(main, "_resolve_cross_source_paths", lambda: (storage, None, None, None))
    history = [{"role": "user", "content": "Fictional source", "ts_ms": 1_577_836_800_000}]
    for source_cid in (cid, unreadable):
        main._write_conversation_state(source_cid, user_id=uid, soul_id=sid, updates={"memorize_chat": True})
        main._conversation_sources.persist_sillytavern_history_snapshot(
            storage_dir=storage, user_id=uid, soul_id=sid, conversation_id=source_cid, history=history,
        )
    source = dict(storage_dir=storage, user_id=uid, soul_id=sid,
                  conversation_id=unreadable, source_label="sillytavern")
    main._conversation_sources._chat_snapshot_path(**source).write_text("{unreadable")
    with pytest.raises(json.JSONDecodeError):
        main._conversation_sources.load_chat_snapshot_tail(**source, since_cursor=-1, recent_fallback_messages=0)
    other_before, _, _ = main._load_turn_state_and_soul_card(cid, user_id=uid, soul_id="OtherSoul")
    before = {source_cid: main._load_turn_state_and_soul_card(source_cid, user_id=uid, soul_id=sid)[0]
              for source_cid in (cid, unreadable)}
    extract = AsyncMock(return_value=[{}])
    service = SavedBatchService()
    monkeypatch.setattr(service, "extract", extract, raising=False)
    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_args: service)
    monkeypatch.setattr(main, "_run_consolidation_task", AsyncMock(return_value={"status": "skipped"}))
    monkeypatch.setattr(main, "_MIN_CHUNK_TOKENS", 1)
    monkeypatch.setattr(main, "_unmemorized_sleep_gap_detected", lambda *_args, **_kwargs: True)
    if import_pending:
        main._write_conversation_state("import:dm:history", user_id=uid, soul_id=sid, updates={"import_state": {
            "history_end_index": 1, "memorize_cursor": -1, "pending_segment_ids": [],
            "stage": "memorize", "error": None}})
    _, payload = main._prepare_auto_memorize(cid, uid, sid, {}, before[cid], history, dry_run=False)
    if import_pending:
        assert payload is None
        assert not main._paid_work_state(uid, sid).get("memorize_failure")
        assert main._soul_import_state(uid, sid)[1]["ordinary_waiting"] == cid
        main._write_conversation_state("import:dm:history", user_id=uid, soul_id=sid, updates={
            "import_memorize_cursor": 0, "import_stage": "complete",
        })
        marker = main._memorize_lock_key(uid, sid)
        main._FORCED_MEMORIZE_INFLIGHT[marker] = False
        await main._import_routes.run_waiting_memorize(main, scoped={"user_id": uid, "soul_id": sid},
                                                    import_cid="import:dm:history")
        assert marker not in main._FORCED_MEMORIZE_INFLIGHT
    elif payload is not None:
        await main._run_forced_memorize_from_turn(payload)
    other_after, _, _ = main._load_turn_state_and_soul_card(cid, user_id=uid, soul_id="OtherSoul")
    assert other_after == other_before
    assert main._soul_activity_pause(uid, "OtherSoul") is None
    extract.assert_not_awaited()
    assert payload is None
    for source_cid in (cid, unreadable):
        after, _, _ = main._load_turn_state_and_soul_card(source_cid, user_id=uid, soul_id=sid)
        for field in ("digest_cursor", "rolling_summary_cursor_id", "last_memorize_at", "pending_segment_ids"):
            assert after[field] == before[source_cid][field]
    failure = main._paid_work_state(uid, sid)["memorize_failure"]
    assert failure["conversation_id"] == cid and failure["paused"]
    assert unreadable in failure["error"]
    assert main._soul_activity_pause(uid, sid)
    with pytest.raises(HTTPException) as blocked:
        main._require_soul_active(uid, sid)
    assert blocked.value.detail["code"] == "soul_paused"
    tasks = BackgroundTasks()
    if import_pending:
        assert (await main.retry_memorize(uid, sid, tasks))["status"] == "accepted"
        await tasks()
    else:
        with pytest.raises(HTTPException, match="unavailable"):
            await main.retry_memorize(uid, sid, tasks)
    extract.assert_not_awaited()
    assert main._paid_work_state(uid, sid)["memorize_failure"]["paused"]


@pytest.mark.asyncio
async def test_retry_of_completed_targets_clears_only_memorize(monkeypatch):
    uid, sid, cid = "TestOwner", "TestSoul", "chat:saved-chat"
    failure = {"conversation_id": cid, "error": "Failed", "paused": True, "targets": {cid: {"cursor": 0}}}
    main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={"memorize_failure": failure})
    assert not main._memorize_targets_complete(uid, sid, failure["targets"])
    main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={
        "digest_cursor": 0, "last_memorize_at": datetime.now(UTC).isoformat(),
        "last_consolidation_error": "Reflection failed", "last_consolidation_error_at": datetime.now(UTC).isoformat(),
    })
    monkeypatch.setattr(main, "_safe_payload", lambda payload: payload)
    schedule = AsyncMock(return_value={"status": "skipped"})
    monkeypatch.setattr(main, "_run_consolidation_task", schedule)
    tasks = BackgroundTasks()
    assert (await main.retry_memorize(uid, sid, tasks))["status"] == "already_memorized"
    await tasks()
    schedule.assert_awaited_once()
    state = main._paid_work_state(uid, sid)
    assert state["memorize_failure"] is None
    assert main._soul_activity_pause(uid, sid) == "Reflection failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("phase", ["dedupe", "review"])
async def test_retry_recovers_before_completed_shortcut_and_consolidation(monkeypatch, waiting, fail, phase):
    uid, sid, cid = "TestOwner", "TestSoul", "chat:review-retry"
    scoped = {"user_id": uid, "soul_id": sid}
    failure = {"conversation_id": cid, "paused": False, "error": "Review failed",
               "targets": {cid: {"cursor": 0}}}
    main._write_conversation_state(cid, **scoped, updates={"digest_cursor": 0, "memorize_failure": failure,
        "last_memorize_at": datetime.now(UTC).isoformat(),
        "last_consolidation_error": "Reflection failed",
        "last_consolidation_error_at": datetime.now(UTC).isoformat()})
    main._write_conversation_state(cid, **scoped, updates={"memorize_segment_work": {"saved-segment": phase}})
    if waiting:
        main._write_conversation_state("import:dm:history", **scoped, updates={"import_state": {
            "history_end_index": 1, "memorize_cursor": 0, "pending_segment_ids": [],
            "stage": "complete", "error": None, "ordinary_waiting": cid}})
    engine = SavedBatchService.make_engine(scoped)
    entered, release = asyncio.Event(), asyncio.Event()
    async def review(state, context, **kwargs):
        assert main._paid_work_state(uid, sid)["memorize_failure"]["paused"]
        assert main._soul_activity_pause(uid, sid)
        entered.set()
        await release.wait()
        if fail:
            raise RuntimeError("Review still failed")
        return state
    monkeypatch.setattr(engine, "_memorize_persist_and_index", review)
    monkeypatch.setattr(main, "_get_service_from_payload", lambda _payload: engine)
    monkeypatch.setattr(main, "_saved_memorize_payload", lambda *_args: pytest.fail("Completed checkpoint must not extract"))
    consolidation = AsyncMock(return_value={"status": "skipped"})
    monkeypatch.setattr(main, "_run_consolidation_pipeline_once", consolidation)
    if waiting:
        assert (await main.diag_memorize_pending(**scoped))["retry_operation"] == "memorize"
    tasks = BackgroundTasks()
    assert (await main.retry_memorize(uid, sid, tasks))["status"] == "accepted"
    assert main._paid_work_state(uid, sid)["memorize_failure"]["paused"]
    runner = asyncio.create_task(tasks())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        with pytest.raises(HTTPException):
            await main.retry_memorize(uid, sid, BackgroundTasks())
        release.set()
        await runner
        failure = main._paid_work_state(uid, sid)["memorize_failure"]
        assert bool(failure) == (fail or waiting)
        assert bool((failure or {}).get("segment_work")) == fail
        if fail:
            assert "Review still failed" in failure["error"]
            assert main._MEMORIZE_PROGRESS[main._memorize_lock_key(uid, sid)]["last_result"] == "failure"
        elif waiting:
            assert main._soul_import_state(uid, sid)[1]["ordinary_waiting"] == cid
            assert (await main.diag_memorize_pending(**scoped))["retry_operation"] == "consolidation"
        assert consolidation.await_count == (0 if fail or waiting else 1)
        if not fail and not waiting:
            progress = main._MEMORIZE_PROGRESS[main._memorize_lock_key(uid, sid)]
            assert not progress["active"] and progress["last_result"] == "skipped"
        assert main._memorize_lock_key(uid, sid) not in main._FORCED_MEMORIZE_INFLIGHT
    finally:
        release.set()
        await runner
        engine.database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "skipped", "failure", "cancel"])
async def test_recovery_cancel_flag_and_consolidation_interruption_preserve_targets(monkeypatch, outcome):
    scoped = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    cid = "chat:completed-review"
    targets = {cid: {"cursor": 0}}
    main._write_conversation_state(cid, **scoped, updates={"digest_cursor": 0,
        "last_memorize_at": datetime.now(UTC).isoformat(), "memorize_failure": {
        "conversation_id": cid, "paused": False, "error": "Review failed", "targets": targets}})
    main._write_conversation_state(cid, **scoped, updates={"memorize_segment_work": {"saved": "review"}})
    engine = SavedBatchService.make_engine(scoped)
    async def review(state, context, **_kwargs):
        assert (await main.memorize_cancel(scoped))["status"] == "cancel_requested"
        return state
    async def consolidate(**_kwargs):
        if outcome == "cancel":
            raise asyncio.CancelledError()
        if outcome == "failure":
            raise RuntimeError("Reflection failed")
        return {"status": "ok", "result": {}} if outcome == "success" else {"status": "skipped"}
    monkeypatch.setattr(engine, "_memorize_persist_and_index", review)
    monkeypatch.setattr(main, "_get_service_from_payload", lambda _payload: engine)
    monkeypatch.setattr(main, "_run_consolidation_pipeline_once", consolidate)
    tasks = BackgroundTasks()
    await main.retry_memorize(**scoped, background_tasks=tasks)
    assert main._paid_work_state(**scoped)["memorize_failure"]["paused"]
    try:
        if outcome == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await tasks()
            assert main._paid_work_state(**scoped)["memorize_failure"]["targets"] == targets
            assert main._memorize_targets_complete(*scoped.values(), targets)
            assert (await main.retry_memorize(**scoped, background_tasks=BackgroundTasks()))["status"] == "already_memorized"
        else:
            await tasks()
            assert main._paid_work_state(**scoped)["memorize_failure"] is None
        marker = main._memorize_lock_key(**scoped)
        assert marker not in main._MEMORIZE_CANCEL and marker not in main._FORCED_MEMORIZE_INFLIGHT
        assert not main._MEMORIZE_PROGRESS[marker]["active"]
        assert main._MEMORIZE_PROGRESS[marker]["last_result"] == (
            "failure" if outcome in {"failure", "cancel"} else outcome
        )
        if outcome in {"failure", "cancel"}:
            assert main._MEMORIZE_PROGRESS[marker]["error"].startswith(
                "RuntimeError: Reflection failed" if outcome == "failure" else "CancelledError:"
            )
    finally:
        engine.database.close()


@pytest.mark.asyncio
async def test_recovery_dedupes_trailing_segment_before_soul_wide_review(monkeypatch):
    from app.services import import_routes
    scoped = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    cid = "chat:mixed-phases"
    main._write_conversation_state(cid, **scoped, updates={"memorize_failure": {
        "conversation_id": cid, "paused": True, "error": "Dedupe failed", "targets": {}}})
    main._write_conversation_state(cid, **scoped, updates={"memorize_segment_work": {
        "earlier": "review", "later": "dedupe"}})
    engine = SavedBatchService.make_engine(scoped)
    for _ in range(10):
        item = engine.database.memory_item_repo.create_item(
            memory_type="knowledge", summary="A fictional garden", embedding=[1.0, 0.0],
            user_data=scoped, segment_id="later")
        engine.database.dossier_candidate_repo.add_candidate(proposed_name="Garden", item_id=item.id, where=scoped)
    monkeypatch.setattr(engine, "generate_dynamic_category_review", AsyncMock(
        side_effect=AssertionError("Duplicate evidence must collapse before review")))
    monkeypatch.setattr(main, "_get_service_from_payload", lambda _payload: engine)
    try:
        await import_routes.recover_memorize(main, scoped=scoped, cid=cid)
        assert len(engine.database.memory_item_repo.list_items(scoped)) == 1
        assert not main._paid_work_state(**scoped)["memorize_failure"].get("segment_work")
        engine.generate_dynamic_category_review.assert_not_awaited()
    finally:
        engine.database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_detached_consolidation_owns_terminal_progress_and_cancel_cleanup(monkeypatch, cancel):
    scoped = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    marker = main._memorize_lock_key(**scoped)
    entered, release = asyncio.Event(), asyncio.Event()
    async def pipeline(**kwargs):
        key = (kwargs["user_id"], kwargs["soul_id"])
        main._CONSOLIDATION_RUNNING[key] = False
        try:
            entered.set()
            await release.wait()
            return {"status": "ok", "result": {}}
        finally:
            main._CONSOLIDATION_RUNNING.pop(key)
    monkeypatch.setattr(main, "_run_consolidation_pipeline_once", pipeline)
    task = asyncio.create_task(main._run_consolidation_task(
        object(), conversation_id="chat:detached", uid=scoped["user_id"], soul_id=scoped["soul_id"],
        progress_key=marker, memorize_progress=main._MEMORIZE_PROGRESS))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert (await main.memorize_cancel(scoped))["status"] == "cancel_requested"
        assert marker in main._MEMORIZE_CANCEL
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            await task
        assert marker not in main._MEMORIZE_CANCEL
        assert not main._MEMORIZE_PROGRESS[marker]["active"]
        assert main._MEMORIZE_PROGRESS[marker]["last_result"] == ("failure" if cancel else "success")
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_retry_stays_paused_and_duplicate_is_refused(monkeypatch):
    uid, sid, cid = "TestOwner", "TestSoul", "chat:saved-chat"
    failure = {"conversation_id": cid, "error": "Failed", "paused": False, "targets": {cid: {"cursor": 1}}}
    main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={"memorize_failure": failure})
    monkeypatch.setattr(main, "_safe_payload", lambda payload: payload)
    payload = {"user": {"user_id": uid, "soul_id": sid, "conversation_id": cid}, "conversation_id": cid,
               "conversation": [{"role": "user", "content": "First"}, {"role": "assistant", "content": "Second"}]}
    monkeypatch.setattr(main._cross_history, "_load_tail_for_source_conversation", lambda **_kwargs: payload["conversation"])
    monkeypatch.setattr(main, "_build_cross_conversation_payload", lambda *_args, **_kwargs: payload)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    class Service(SavedBatchService):
        async def extract(self, **kwargs):
            calls.append(kwargs)
            entered.set()
            await release.wait()
            return [{"memory_item_ids": []} for _ in kwargs["segments"]]
    monkeypatch.setattr(main, "_get_service_from_payload", lambda _payload: Service())
    tasks = BackgroundTasks()
    await main.retry_memorize(uid, sid, tasks)
    marker = main._memorize_lock_key(uid, sid)
    runner = asyncio.create_task(tasks())
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        with pytest.raises(HTTPException) as paused:
            main._require_soul_active(uid, sid)
        assert paused.value.detail["code"] == "soul_paused"
        with pytest.raises(HTTPException) as duplicate:
            await main.retry_memorize(uid, sid, BackgroundTasks())
        assert duplicate.value.status_code == 409
        release.set()
        await runner
        assert len(calls) == 1
        assert main._paid_work_state(uid, sid)["memorize_failure"] is None
    finally:
        release.set()
        await runner
        main._FORCED_MEMORIZE_INFLIGHT.pop(marker, None)


@pytest.mark.asyncio
async def test_interrupted_consolidation_is_durable_and_retry_remains_paused(monkeypatch):
    uid, sid, cid = "TestOwner", "TestSoul", "chat:saved-chat"
    main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={})
    path = main._sqlite_current_path(uid, sid)
    calls = []
    def gather(*_args, force=False, **_kwargs):
        state = main._paid_work_state(uid, sid)
        if soul_state.consolidation_failure(state) and not force:
            return {"status": "skip", "reason": "failure_requires_retry"}
        return {"status": "ready", "db_path": path, "state": state, "current_chat_messages": []}
    async def prepare(*_args, **_kwargs):
        calls.append("paid")
        assert main._paid_work_state(uid, sid)["last_consolidation_error"]
        if len(calls) == 1:
            assert main._soul_activity_pause(uid, sid) is None
            raise asyncio.CancelledError()
        assert main._soul_activity_pause(uid, sid)
    async def llm(*_args, **_kwargs):
        return {}
    def finish(*_args, **_kwargs):
        main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={
            "last_consolidation_error": None, "last_consolidation_error_at": None,
        })
        return {}
    monkeypatch.setattr(consolidation, "gather_consolidation_inputs", gather)
    monkeypatch.setattr(consolidation, "prepare_dossier_consolidation_context", prepare)
    monkeypatch.setattr(consolidation, "preflight_consolidation_profiles", lambda *_args: None)
    monkeypatch.setattr(service_factory, "_resolve_profile_if_configured", lambda *_args: "test")
    monkeypatch.setattr(consolidation, "run_consolidation_llm", llm)
    monkeypatch.setattr(consolidation, "write_consolidation_outputs", finish)
    kwargs = dict(
        svc=SimpleNamespace(database=SimpleNamespace(dsn=f"sqlite:///{path}")), deps=main._make_consolidation_deps(),
        state_lock=asyncio.Lock(), running=main._CONSOLIDATION_RUNNING,
        load_cross_tail_for_ai=lambda **_kwargs: [], format_all_chat_history_for_ai=lambda **_kwargs: "",
        conversation_id=cid, soul_id=sid, user_id=uid,
    )
    with pytest.raises(asyncio.CancelledError):
        await main._run_consolidation_pipeline_once(**kwargs)
    assert main._soul_activity_pause(uid, sid)
    assert (await main._run_consolidation_pipeline_once(**kwargs))["reason"] == "failure_requires_retry"
    assert len(calls) == 1
    assert (await main._run_consolidation_pipeline_once(**kwargs, force=True))["status"] == "ok"
    assert main._soul_activity_pause(uid, sid) is None


@pytest.mark.asyncio
async def test_paused_atomic_session_creates_no_state_or_snapshot(monkeypatch):
    from fastapi.testclient import TestClient
    uid, sid = "TestOwner", "TestSoul"
    main._write_conversation_state("chat:failure", user_id=uid, soul_id=sid, updates={
        "memorize_failure": {"conversation_id": "chat:failure", "error": "Failed", "paused": True, "targets": {}},
    })
    monkeypatch.setattr(main, "_write_conversation_state", lambda *_args, **_kwargs: pytest.fail("must not create a session"))
    monkeypatch.setattr(main._conversation_sources, "persist_atomic_history_snapshot", lambda **_kwargs: pytest.fail("must not write history"))
    response = TestClient(main.app).post("/integration/atomic/session_start", json={
        "user_id": uid, "soul_id": sid, "conversation_id": "chat:atomic-test",
    })
    assert response.status_code == 409 and response.json()["detail"]["code"] == "soul_paused"


@pytest.mark.asyncio
async def test_retry_names_unreadable_source_without_clearing_failure(monkeypatch):
    uid, sid, cid = "TestOwner", "TestSoul", "chat:unreadable"
    main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={
        "memorize_failure": {"conversation_id": cid, "error": "Failed", "paused": True,
                             "targets": {cid: {"cursor": 1, "memory_producing": True}}},
    })
    def broken(**_kwargs):
        raise OSError("Unreadable snapshot")
    monkeypatch.setattr(main._cross_history, "_load_tail_for_source_conversation", broken)
    with pytest.raises(HTTPException) as refused:
        await main.retry_memorize(uid, sid, BackgroundTasks())
    assert refused.value.status_code == 409
    assert cid in refused.value.detail and "restore" in refused.value.detail
    assert main._paid_work_state(uid, sid)["memorize_failure"]["paused"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["save", "review", "review_error_write"])
async def test_partial_publication_retains_only_published_files_and_retry_reuses_resource(monkeypatch, tmp_path, request, failure_stage):
    from pydantic import BaseModel
    from memu.app.service import MemoryService
    class Scope(BaseModel):
        user_id: str | None = None
        soul_id: str | None = None
    uid, sid, cid = "TestOwner", "TestSoul", "chat:partial-publish"
    path = main._sqlite_current_path(uid, sid)
    svc = MemoryService(database_config={"metadata_store": {"provider": "sqlite", "dsn": f"sqlite:///{path}"}}, user_config={"model": Scope})
    request.addfinalizer(svc.database.close)
    scope = {"user_id": uid, "soul_id": sid}
    main._write_conversation_state(cid, **scope, updates={})
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"segments": [{"start": i, "end": i} for i in range(2)]}))
    routes = []
    class Router:
        chat_model = "fictional-router"
        async def chat(self, prompt):
            routes.append(prompt)
            return json.dumps({"episodes": [{"title": "Fictional garden", "episode_summary": "A garden.",
                "episode_item": "A fictional garden project.", "categories": ["Garden"], "day": self.day}],
                "excluded_types": []})
    route = svc._route_segment
    async def routed(text, types, **kwargs):
        kwargs["llm_client"].day = kwargs["source_days"][0]
        return await route(text, types, **kwargs)
    monkeypatch.setattr(svc, "_route_segment", routed)
    monkeypatch.setattr(svc, "_generate_entries_from_text", AsyncMock(return_value=[]))
    monkeypatch.setattr(svc, "_select_chat_client", lambda *_a, **_kw: Router())
    monkeypatch.setattr(svc, "_select_embedding_client", lambda *_a, **_kw: SimpleNamespace(
        embed=AsyncMock(side_effect=lambda texts: [[1.0, 0.0] for _ in texts])))
    writer = main._write_conversation_state
    def fail_second(*args, **kwargs):
        if failure_stage == "save" and kwargs["updates"].get("digest_cursor") == 2:
            raise RuntimeError("Second publication failed")
        if failure_stage == "review_error_write" and (kwargs["updates"].get("memorize_failure") or {}).get("paused"):
            raise RuntimeError("Failure status write failed")
        return writer(*args, **kwargs)
    review = svc._memorize_persist_and_index
    review_failed = False
    async def fail_review(state, context, **kwargs):
        nonlocal review_failed
        if failure_stage != "save" and state.get("resources") and not review_failed:
            review_failed = True
            raise RuntimeError("Saved segment review failed")
        return await review(state, context, **kwargs)
    monkeypatch.setattr(svc, "_memorize_persist_and_index", fail_review)
    monkeypatch.setattr(main, "_write_conversation_state", fail_second)
    monkeypatch.setattr(main, "_run_consolidation_task", AsyncMock(return_value={"status": "skipped"}))
    messages = [[{"role": "user", "content": word, "ts_ms": 1_577_836_800_000}] for word in ("First", "Second")]
    async def run(indices):
        return await main._run_memorize_segments(
            memorize_segments=[("unused", [{**messages[0][0], "memorize_chat": False}], 0, 0, None)]
                + [("unused", messages[i], i + 1, i + 1, (i, i)) for i in indices],
            svc=svc, scope=scope, conversation_id=cid, soul_id=sid, uid=uid,
            processed_cursor=-1, safe={}, resource_url="unused", chat_key=None, merged_len=2,
            force=True, sleep_stats=None, segments_dir=segments_dir,
        )
    with pytest.raises(RuntimeError, match="Second publication|Saved segment review|Failure status write"):
        await run([0, 1])
    assert [file.name for file in segments_dir.iterdir()] == ["2020-01-01.json"]
    assert json.loads(manifest.read_text())["segments"] == [{"start": 0, "end": 0}]
    assert len(svc.database.resource_repo.list_resources(scope)) == 1
    state, _, _ = main._load_turn_state_and_soul_card(cid, **scope)
    assert state["digest_cursor"] == 1 and state["pending_segment_ids"] == [f"{cid}:0-0"]
    assert state["memorize_failure"]["segment_work"] == {f"{cid}:0-0": "review"}
    monkeypatch.setattr(main, "_write_conversation_state", writer)
    svc.database.close()
    monkeypatch.setattr(main, "_get_service_from_payload", lambda _payload: svc)
    monkeypatch.setattr(main, "_saved_memorize_payload", lambda *_args: {"user": scope, "conversation_id": cid})
    async def remainder(payload, tasks, force, **kwargs):
        assert kwargs["admitted"] and kwargs["batch_owned"] and kwargs["retry"]
        assert main._paid_work_state(uid, sid)["memorize_failure"]["paused"]
        manifest.write_text(json.dumps({"segments": [{"start": i, "end": i} for i in range(2)]}))
        tasks.add_task(run, [1])
    monkeypatch.setattr(main, "_memorize_owned", remainder)
    entered, release = asyncio.Event(), asyncio.Event()
    async def recovery_review(state, context, **kwargs):
        assert len(routes) == 2
        assert main._paid_work_state(uid, sid)["memorize_failure"]["paused"]
        entered.set()
        await release.wait()
        return await review(state, context, **kwargs)
    monkeypatch.setattr(svc, "_memorize_persist_and_index", recovery_review)
    tasks = BackgroundTasks()
    assert (await main.retry_memorize(uid, sid, tasks))["status"] == "accepted"
    runner = asyncio.create_task(tasks())
    await asyncio.wait_for(entered.wait(), timeout=2)
    assert main._MEMORIZE_PROGRESS[main._memorize_lock_key(uid, sid)]["active"]
    with pytest.raises(HTTPException):
        await main.retry_memorize(uid, sid, BackgroundTasks())
    release.set()
    monkeypatch.setattr(svc, "_memorize_persist_and_index", review)
    await runner
    assert len(routes) == 3 and len(svc.database.resource_repo.list_resources(scope)) == 2
    assert len(list(segments_dir.iterdir())) == 2
    state, _, _ = main._load_turn_state_and_soul_card(cid, **scope)
    assert state["rolling_summary_cursor_id"] == 0 and state["digest_cursor"] == 2
    assert state["memorize_failure"] is None


@pytest.mark.asyncio
async def test_no_new_tail_still_checks_pending_consolidation_with_a_recent_clock(monkeypatch):
    uid, sid, cid = "TestOwner", "TestSoul", "chat:pending-cadence"
    main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={
        "pending_segment_ids": [f"{cid}:0-0"], "last_consolidation_at": datetime.now(UTC).isoformat(),
        "digest_cursor": 0, "last_memorize_at": datetime.now(UTC).isoformat(),
    })
    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_args: object())
    schedule = AsyncMock(return_value={"status": "skipped"})
    monkeypatch.setattr(main, "_run_consolidation_task", schedule)
    tasks = BackgroundTasks()
    response = await main.memorize({"user": {"user_id": uid, "soul_id": sid},
        "conversation_id": cid, "conversation": [{"role": "user", "content": "Already memorized"}]}, tasks, True, tail=True)
    assert response.status_code == 202
    await tasks()
    schedule.assert_awaited_once()
    assert schedule.await_args.kwargs["conversation_id"] == cid
    assert not schedule.await_args.kwargs.get("force", False)


@pytest.mark.parametrize("historical", [False, True])
def test_consolidation_defers_scoped_unfinished_memorize(historical):
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    cid = "chat:unfinished"
    updates = {"pending_segment_ids": [f"{cid}:older"]}
    if historical:
        updates["import_state"] = {"history_end_index": 2, "memorize_cursor": 0,
            "stage": "consolidation", "error": "Failed", "pending_segment_ids": [f"{cid}:0-0"],
            "segment_work": {f"{cid}:0-0": "review"}}
    else:
        updates["memorize_failure"] = {"conversation_id": cid, "error": "Failed", "paused": True,
            "targets": {cid: {"cursor": 0}}, "segment_work": {f"{cid}:0-0": "dedupe"}}
    if not historical:
        work = updates["memorize_failure"].pop("segment_work")
    main._write_conversation_state(cid, **scope, updates=updates)
    if not historical:
        main._write_conversation_state(cid, **scope, updates={"memorize_segment_work": work})
    result = consolidation.gather_consolidation_inputs(
        main._make_consolidation_deps(), conversation_id=cid, **scope, force=True, historical=historical,
    )
    assert result == {"status": "skip", "reason": "memorize_postprocessing_pending"}
    other = {**scope, "soul_id": "OtherSoul"}
    main._write_conversation_state(cid, **other, updates={})
    assert consolidation.gather_consolidation_inputs(
        main._make_consolidation_deps(), conversation_id=cid, **other, force=True,
    )["reason"] == "no_pending_segments"
    if not historical:
        main._write_conversation_state(cid, **scope, updates={"memorize_failure": {
            "conversation_id": "chat:later-source", "error": "Later source failure", "paused": True, "targets": {},
        }})
        failure = main._paid_work_state(**scope)["memorize_failure"]
        assert failure["segment_work"] == work
        assert failure["targets"] == {cid: {"cursor": 0}}
        assert failure["conversation_id"] == cid
        with pytest.raises(RuntimeError, match="postprocessing is still unfinished"):
            main._write_conversation_state(cid, **scope, updates={"memorize_failure": None})


@pytest.mark.asyncio
async def test_cancel_before_segment_file_releases_manifest_reservation(tmp_path):
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    cid = "chat:early-cancel"
    main._write_conversation_state(cid, **scope, updates={})
    segments = tmp_path / "segments"
    segments.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"segments": [{"start": 0, "end": 0}]}))
    key = main._memorize_lock_key(**scope)
    main._MEMORIZE_CANCEL.add(key)
    assert not await main._run_memorize_segments(
        memorize_segments=[("unused", [{"role": "user", "content": "Fictional story"}], 0, 0, (0, 0))],
        svc=object(), scope=scope, conversation_id=cid, soul_id=scope["soul_id"], uid=scope["user_id"],
        processed_cursor=-1, safe={}, resource_url="unused", chat_key=None, merged_len=1,
        force=True, sleep_stats=None, segments_dir=segments,
    )
    assert json.loads(manifest.read_text())["segments"] == []
    assert not list(segments.iterdir())
