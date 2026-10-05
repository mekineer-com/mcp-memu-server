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
    class Service:
        async def memorize_segments_batch(self, **kwargs):
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
    class Service:
        async def memorize_segments_batch(self, **kwargs):
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
    assert state["pending_segment_ids"]
    assert all(Path(segment["local_path"]).is_file() for segment in calls[0]["segments"])
    assert main._paid_work_state(uid, sid)["memorize_failure"]
    monkeypatch.setattr(main, "_write_conversation_state", real_write)
    reader = main._cross_history._load_tail_for_source_conversation
    monkeypatch.setattr(main._cross_history, "_load_tail_for_source_conversation", lambda **kwargs: [] if kwargs["conversation_id"] == second else reader(**kwargs))
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
    with pytest.raises(RuntimeError, match="Later checkpoint failed"):
        await tasks()
    state, _, _ = main._load_turn_state_and_soul_card(first, user_id=uid, soul_id=sid)
    assert calls[-1]["segments"][0]["segment"]["segment_id"] in state["pending_segment_ids"]
    assert all(Path(segment["local_path"]).is_file() for segment in calls[-1]["segments"])
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
    class Service:
        async def memorize_segments_batch(self, **kwargs):
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
    class Service:
        async def memorize_segments_batch(self, **kwargs):
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
    class Service:
        async def memorize_segments_batch(self, **kwargs):
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
async def test_automatic_admission_reaches_real_endpoint_once(monkeypatch):
    uid, sid, cid = "TestOwner", "TestSoul", "chat:saved-chat"
    main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={})
    monkeypatch.setattr(main, "_safe_payload", lambda payload: payload)
    marker = main._memorize_lock_key(uid, sid)
    calls = []

    class Service:
        async def memorize_segments_batch(self, **kwargs):
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
    try:
        await main._run_forced_memorize_from_turn(payload)
        assert len(calls) == 1
        assert marker not in main._FORCED_MEMORIZE_INFLIGHT
        assert main._paid_work_state(uid, sid)["memorize_failure"] is None
    finally:
        main._FORCED_MEMORIZE_INFLIGHT.pop(marker, None)


@pytest.mark.asyncio
async def test_automatic_admission_failure_before_jobs_pauses_and_exposes_retry(monkeypatch):
    uid, sid, cid = "TestOwner", "TestSoul", "chat:failed-admission"
    async def fail(*_args):
        raise ValueError("Fictional source error")
    monkeypatch.setattr(main, "_memorize_admitted", fail)
    assert await main._run_forced_memorize_from_turn({
        "user": {"user_id": uid, "soul_id": sid}, "conversation_id": cid,
    }) is False
    assert main._paid_work_state(uid, sid)["memorize_failure"]["conversation_id"] == cid
    assert main._soul_activity_pause(uid, sid)


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
    class Service:
        async def memorize_segments_batch(self, **kwargs):
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
async def test_partial_publication_retains_only_published_files_and_retry_reuses_resource(monkeypatch, tmp_path, request):
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
    resources = []
    async def batch(**kwargs):
        for segment in kwargs["segments"]:
            resource = svc.database.resource_repo.create_resource(
                url=segment["resource_url"], local_path=segment["local_path"], modality="conversation",
                caption=None, embedding=None, user_data=scope, conversation_id=cid,
                segment_id=segment["segment"]["segment_id"],
            )
            resources.append(resource.id)
        return [{"pending_segment_ids": [segment["segment"]["segment_id"]]} for segment in kwargs["segments"]]
    svc.memorize_segments_batch = batch
    writer = main._write_conversation_state
    def fail_second(*args, **kwargs):
        if kwargs["updates"].get("digest_cursor") == 1:
            raise RuntimeError("Second publication failed")
        return writer(*args, **kwargs)
    monkeypatch.setattr(main, "_write_conversation_state", fail_second)
    monkeypatch.setattr(main, "_run_consolidation_task", AsyncMock(return_value={"status": "skipped"}))
    messages = [[{"role": "user", "content": word, "ts_ms": 1_577_836_800_000}] for word in ("First", "Second")]
    async def run(indices):
        await main._run_memorize_segments(
            memorize_segments=[("unused", messages[i], i, i, (i, i)) for i in indices],
            svc=svc, scope=scope, conversation_id=cid, soul_id=sid, uid=uid,
            processed_cursor=-1, safe={}, resource_url="unused", chat_key=None, merged_len=2,
            force=True, sleep_stats=None, segments_dir=segments_dir,
        )
    with pytest.raises(RuntimeError, match="Second publication"):
        await run([0, 1])
    assert [file.name for file in segments_dir.iterdir()] == ["2020-01-01.json"]
    monkeypatch.setattr(main, "_write_conversation_state", writer)
    await run([1])
    assert resources[1] == resources[2]
    assert len(list(segments_dir.iterdir())) == 2


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
