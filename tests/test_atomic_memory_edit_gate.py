import asyncio
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest
from fastapi import BackgroundTasks, FastAPI, HTTPException

from app import main


SCOPE = {"user_id": "TestOwner", "soul_id": "TestSoul"}


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.setattr(main, "_FORCED_MEMORIZE_INFLIGHT", {})
    monkeypatch.setattr(main, "_FORCED_MEMORIZE_RECHECK", {})
    monkeypatch.setattr(main, "_CONSOLIDATION_RUNNING", {})
    monkeypatch.setattr(main, "_MEMORIZE_PROGRESS", {})
    monkeypatch.setattr(main, "_MEMORIZE_CANCEL", set())
    monkeypatch.setattr(main, "_soul_import_state", lambda *_args: (None, {}))
    monkeypatch.setattr(main, "_paid_work_state", lambda *_args: {})
    monkeypatch.setattr(main, "_defer_memorize_for_import", lambda *_args: False)
    monkeypatch.setattr(main, "_register_memorize_bounds", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(main._SHUTDOWN_STATE, "draining", False)
    return main._memorize_lock_key(**SCOPE)


@pytest.mark.asyncio
@pytest.mark.parametrize("job", ["memorize", "import", "consolidation"])
async def test_running_memory_work_refuses_every_atomic_mutation(gate, monkeypatch, job):
    if job == "consolidation":
        main._CONSOLIDATION_RUNNING[tuple(SCOPE.values())] = False
    else:
        main._FORCED_MEMORIZE_INFLIGHT[gate] = job == "import"
    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_kw: pytest.fail("mutation reached service"))
    cases = [
        ("PATCH", "/memory/m1"), ("POST", "/memory/m1/approve"), ("DELETE", "/memory/m1"),
        ("PATCH", "/category/c1"), ("POST", "/category/c1/approve"),
        ("PUT", "/category/c1/memory/m1"), ("DELETE", "/category/c1/memory/m1"),
        ("PATCH", "/soul-summary/narrative_self"), ("POST", "/soul-summary/narrative_self/approve"),
        ("POST", "/integration/atomic/memories"), ("POST", "/integration/atomic/entities"),
        ("PATCH", "/integration/atomic/entities/e1"), ("DELETE", "/integration/atomic/entities/e1"),
        ("POST", "/integration/atomic/entities/e1/merge"), ("POST", "/integration/atomic/entities/e1/ignore"),
        ("POST", "/integration/atomic/entities/e1/restore"),
        ("PUT", "/integration/atomic/memories/m1/entities/e1"),
        ("DELETE", "/integration/atomic/memories/m1/entities/e1"),
        ("POST", "/souls/TestSoul/relationships"), ("PATCH", "/souls/TestSoul/relationships/entity:e1"),
        ("DELETE", "/souls/TestSoul/relationships/entity:e1"),
        ("POST", "/souls/TestSoul/narrative_suggestion"),
    ]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        for method, path in cases:
            params = {**SCOPE, "displayed_summary": "Original"}
            if path == "/integration/atomic/memories" or (path.startswith("/souls/") and method != "DELETE"):
                params = {"user_id": "OtherOwner", "soul_id": "OtherSoul"}
            response = await client.request(method, path, params=params, json={**SCOPE, "text": "A fictional memory"})
            assert response.status_code == 409, (method, path, response.text)
            assert "try again" in response.json()["detail"]
    assert (gate in main._FORCED_MEMORIZE_INFLIGHT) == (job != "consolidation")


@pytest.mark.asyncio
async def test_reads_workspace_and_other_soul_stay_available(gate, monkeypatch):
    main._FORCED_MEMORIZE_INFLIGHT[gate] = False
    calls = []
    svc = SimpleNamespace(
        graph_atomic_entities=lambda **_kw: {"entities": []},
        graph_create_entity=lambda *_a, **kw: calls.append(kw["where"]) or {"id": "e1"},
    )
    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_kw: svc)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        assert (await client.get("/integration/atomic/entities", params=SCOPE)).status_code == 200
        other = {**SCOPE, "soul_id": "OtherSoul"}
        assert (await client.post("/integration/atomic/entities", params=other,
                                  json={"name": "Fictional Garden", "entity_type": "place"})).status_code == 200
    assert calls == [other]
    assert main._FORCED_MEMORIZE_INFLIGHT == {gate: False}
    app = FastAPI()
    app.middleware("http")(main._trace_requests)
    app.add_api_route("/workspace/notes", lambda: {"saved": True}, methods=["POST"])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.post("/workspace/notes", params=SCOPE)).json() == {"saved": True}


@pytest.mark.asyncio
async def test_edit_claim_preserves_interrupted_memorize_retry_status(gate, monkeypatch):
    main._write_conversation_state("atomic:dm:TestSoul", **SCOPE, updates={"memorize_failure": {
        "conversation_id": "atomic:dm:TestSoul", "error": "Interrupted Memorize", "paused": False,
        "targets": {},
    }})
    main._FORCED_MEMORIZE_INFLIGHT[gate] = None
    monkeypatch.setattr(main, "_load_cross_memorize_tails_from_sources", lambda *_a, **_kw: {})
    status = await main.diag_memorize_pending(**SCOPE)
    assert status["paused"] and status["retry_operation"] == "memorize"


@pytest.mark.asyncio
async def test_malformed_manual_memory_body_keeps_validation(gate, monkeypatch):
    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_kw: pytest.fail("invalid body reached service"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        response = await client.post("/integration/atomic/memories", content="{", headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert main._FORCED_MEMORIZE_INFLIGHT == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("error, status", [
    (HTTPException(status_code=422, detail="Invalid Soul name"), 422),
    (main._owner.OwnerIdentityError("Owner mismatch"), 409),
    (main._owner.OwnerStorageError("Owner unavailable"), 503),
])
async def test_gate_scope_errors_keep_original_response(gate, monkeypatch, error, status):
    def unavailable(*_args, **_kwargs):
        raise error
    monkeypatch.setattr(main, "_sqlite_current_path", unavailable)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        response = await client.post("/integration/atomic/memories", json={**SCOPE, "text": "Fictional"})
    assert response.status_code == status
    assert main._FORCED_MEMORIZE_INFLIGHT == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_edit_claim_blocks_jobs_and_releases_queued_work(gate, monkeypatch, outcome):
    entered, release = asyncio.Event(), asyncio.Event()
    queued = []
    schedule = main._schedule_auto_memorize
    monkeypatch.setattr(main, "_schedule_auto_memorize", lambda payload, scope: queued.append((payload, scope)) or "launched")
    monkeypatch.setattr(main, "_safe_payload", deepcopy)

    async def create(text, *, where):
        assert where == SCOPE and text == "A fictional memory"
        entered.set()
        await release.wait()
        if outcome == "error":
            raise ValueError("Fictional embedding failed")
        return {"id": "m1"}

    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_kw: SimpleNamespace(graph_create_manual_memory=create))
    payload = {"conversation_id": "atomic:dm:TestSoul", "user": SCOPE, "conversation": []}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        task = asyncio.create_task(client.post("/integration/atomic/memories", json={**SCOPE, "text": "A fictional memory"}))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert gate in main._FORCED_MEMORIZE_INFLIGHT
            assert main._FORCED_MEMORIZE_INFLIGHT[gate] is None
            failure = {"memorize_failure": {"paused": False, "error": "Interrupted Memorize"}}
            assert main._soul_activity_pause(**SCOPE, state=failure) == "Interrupted Memorize"
            with pytest.raises(HTTPException) as refused:
                await main._memorize_owned(payload, BackgroundTasks(), True)
            assert refused.value.status_code == 409
            with pytest.raises(HTTPException) as refused:
                main._consolidation_request_context(payload["conversation_id"], payload)
            assert refused.value.status_code == 409
            assert schedule(payload, {**SCOPE, "conversation_id": payload["conversation_id"]}) == "coalesced"
            assert not queued
            if outcome == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                release.set()
                assert (await task).status_code == (200 if outcome == "success" else 400)
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    assert gate not in main._FORCED_MEMORIZE_INFLIGHT
    assert main._FORCED_MEMORIZE_RECHECK == {}
    assert len(queued) == 1
