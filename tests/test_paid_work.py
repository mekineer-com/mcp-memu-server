import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException

from app import main
from app.services import soul_state
from app.services import consolidation, service_factory


def test_pause_record_survives_restart_and_retry_without_pausing_other_soul():
    state = soul_state.defaults()
    state["memorize_failure"] = {
        "conversation_id": "saved-chat", "error": "Memorize failed",
        "paused": False, "targets": {"saved-chat": {"cursor": 2}},
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


@pytest.mark.asyncio
async def test_invalid_memorize_source_is_rejected_before_admission(monkeypatch):
    monkeypatch.setattr(main, "_safe_payload", lambda payload: payload)
    with pytest.raises(HTTPException) as error:
        await main.memorize({"user": {"user_id": "TestOwner", "soul_id": "TestSoul"}, "conversation": []}, BackgroundTasks(), True)
    assert error.value.status_code == 400
    assert error.value.detail == "conversation_id is required"


@pytest.mark.asyncio
async def test_automatic_admission_reaches_real_endpoint_once(monkeypatch):
    uid, sid, cid = "TestOwner", "TestSoul", "saved-chat"
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
    main._FORCED_MEMORIZE_INFLIGHT.add(marker)
    try:
        await main._run_forced_memorize_from_turn(payload)
        assert len(calls) == 1
        assert marker not in main._FORCED_MEMORIZE_INFLIGHT
        assert main._paid_work_state(uid, sid)["memorize_failure"] is None
    finally:
        main._FORCED_MEMORIZE_INFLIGHT.discard(marker)


@pytest.mark.asyncio
async def test_retry_of_completed_targets_clears_only_memorize(monkeypatch):
    uid, sid, cid = "TestOwner", "TestSoul", "saved-chat"
    failure = {"conversation_id": cid, "error": "Failed", "paused": True, "targets": {cid: {"cursor": 0}}}
    main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={"memorize_failure": failure})
    assert not main._memorize_targets_complete(uid, sid, failure["targets"])
    main._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={
        "digest_cursor": 0, "last_memorize_at": datetime.now(UTC).isoformat(),
        "last_consolidation_error": "Reflection failed", "last_consolidation_error_at": datetime.now(UTC).isoformat(),
    })
    monkeypatch.setattr(main, "_safe_payload", lambda payload: payload)
    monkeypatch.setattr(main, "_consolidation_due", lambda _state: False)
    assert (await main.retry_memorize(uid, sid, BackgroundTasks()))["status"] == "already_memorized"
    state = main._paid_work_state(uid, sid)
    assert state["memorize_failure"] is None
    assert main._soul_activity_pause(uid, sid) == "Reflection failed"


@pytest.mark.asyncio
async def test_retry_stays_paused_and_duplicate_is_refused(monkeypatch):
    uid, sid, cid = "TestOwner", "TestSoul", "saved-chat"
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
        main._FORCED_MEMORIZE_INFLIGHT.discard(marker)


@pytest.mark.asyncio
async def test_interrupted_consolidation_is_durable_and_retry_remains_paused(monkeypatch):
    uid, sid, cid = "TestOwner", "TestSoul", "saved-chat"
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
