"""Contract test for /memorize: the endpoint's BackgroundTasks must actually run.

History: the endpoint adds `_run_memorize_segments` via `background_tasks.add_task(...)`
but returns a `JSONResponse(status_code=202, ...)` built inline. In FastAPI,
when an endpoint returns a Response object directly, tasks on the injected
`BackgroundTasks` parameter are NOT auto-attached — you have to pass
`background=background_tasks` to the Response. Without it, add_task is a
silent no-op and the memorize batches never run.

Turn-triggered memorize goes through `_run_forced_memorize_from_turn` which
creates its own BackgroundTasks and awaits it manually, so it's unaffected.
Only direct `POST /memorize` calls (Re-memorize chat button, st-api.sh, and
every agent smoke-test driver) hit this hole.

This test is the regression guard: it stubs `_run_memorize_segments` with a
recorder, posts to /memorize, and asserts the task actually ran.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_memorize_background_task_runs(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[dict] = []

    async def fake_run(**kwargs) -> None:
        recorded.append({"ran": True, "segment_count": len(kwargs.get("memorize_segments") or [])})

    # Stub the heavy work so we can verify the task fires without LLM calls.
    monkeypatch.setattr(main_module, "_run_memorize_segments", fake_run)

    # Stub _get_service_from_payload to avoid real service construction.
    class _FakeSvc:
        pass

    monkeypatch.setattr(main_module, "_get_service_from_payload", lambda *a, **k: _FakeSvc())

    payload = {
        "user": {"user_id": "test_user", "soul_id": "test_soul", "conversation_id": "cid-1"},
        "conversation_id": "cid-1",
        "conversation": [
            {"role": "user", "name": "test_user", "content": "hello"},
            {"role": "assistant", "name": "test_soul", "content": "hi"},
        ],
        "llm_profiles": {"default": {"provider": "openai", "api_key": "x", "base_url": "x", "chat_model": "x", "embed_model": "x"}},
    }
    resp = client.post("/memorize?rebuild=true", json=payload)
    assert resp.status_code == 202, f"unexpected status: {resp.status_code} body={resp.text[:300]}"
    body = resp.json()
    assert body.get("status") == "accepted"
    assert isinstance(body.get("segment_count"), int)
    assert body.get("segment_count", 0) >= 1

    # TestClient blocks until BackgroundTasks complete, so by the time .post()
    # returns, our fake_run should have been invoked.
    assert recorded, (
        "background task did NOT run — JSONResponse is missing background=background_tasks; "
        "check app/main.py memorize endpoint return"
    )
    assert recorded[0]["ran"] is True
    assert recorded[0]["segment_count"] == body.get("segment_count")


def test_memorize_tail_retries_consolidation_when_pending_segments_exist(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    recorded: list[dict] = []

    async def fake_consolidation_task(
        _svc,
        *,
        conversation_id: str,
        soul_id: str,
        uid: str,
        progress_key: str | None = None,
        memorize_progress: dict | None = None,
    ) -> None:
        recorded.append({"conversation_id": conversation_id, "soul_id": soul_id, "uid": uid})

    async def fake_run_memorize_segments(**_kwargs) -> None:  # pragma: no cover - should not run in this branch
        raise AssertionError("run_memorize_segments should not run when no tail segments exist")

    class _FakeSvc:
        pass

    def fake_write_conversation_state(*_args, **_kwargs):
        return (
            {
                "last_memorize_at": "2026-05-09T00:00:00+00:00",
                "digest_cursor": 1,  # with 2-message input below, tail is empty
                "pending_segment_ids": ["cid-2:0-1"],
            },
            tmp_path / "Echo.db",
        )

    monkeypatch.setattr(main_module, "_get_service_from_payload", lambda *a, **k: _FakeSvc())
    monkeypatch.setattr(main_module, "_run_memorize_segments", fake_run_memorize_segments)
    monkeypatch.setattr(main_module, "_run_consolidation_task", fake_consolidation_task)
    monkeypatch.setattr(main_module, "_write_conversation_state", fake_write_conversation_state)
    monkeypatch.setattr(main_module, "_get_storage_dir", lambda _cfg: tmp_path)

    payload = {
        "user": {"user_id": "test_user", "soul_id": "test_soul", "conversation_id": "cid-2"},
        "conversation_id": "cid-2",
        "conversation": [
            {"role": "user", "name": "test_user", "content": "hello"},
            {"role": "assistant", "name": "test_soul", "content": "hi"},
        ],
        "llm_profiles": {"default": {"provider": "openai", "api_key": "x", "base_url": "x", "chat_model": "x", "embed_model": "x"}},
    }

    resp = client.post("/memorize?tail=true", json=payload)
    assert resp.status_code == 202, f"unexpected status: {resp.status_code} body={resp.text[:300]}"
    body = resp.json()
    assert body.get("pending_segment_retry") is True
    assert recorded == [{"conversation_id": "cid-2", "soul_id": "test_soul", "uid": "test_user"}]


def test_memorize_tail_nothing_to_memorize_sets_terminal_progress(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _FakeSvc:
        pass

    def fake_write_conversation_state(*_args, **_kwargs):
        return (
            {
                "last_memorize_at": "2026-05-09T00:00:00+00:00",
                "digest_cursor": 1,  # with 2-message input below, tail is empty
                "pending_segment_ids": [],
            },
            tmp_path / "Echo.db",
        )

    monkeypatch.setattr(main_module, "_get_service_from_payload", lambda *a, **k: _FakeSvc())
    monkeypatch.setattr(main_module, "_write_conversation_state", fake_write_conversation_state)
    monkeypatch.setattr(main_module, "_get_storage_dir", lambda _cfg: tmp_path)

    payload = {
        "user": {"user_id": "test_user", "soul_id": "test_soul", "conversation_id": "cid-3"},
        "conversation_id": "cid-3",
        "conversation": [
            {"role": "user", "name": "test_user", "content": "hello"},
            {"role": "assistant", "name": "test_soul", "content": "hi"},
        ],
        "llm_profiles": {"default": {"provider": "openai", "api_key": "x", "base_url": "x", "chat_model": "x", "embed_model": "x"}},
    }

    resp = client.post("/memorize?tail=true", json=payload)
    assert resp.status_code == 200, f"unexpected status: {resp.status_code} body={resp.text[:300]}"
    body = resp.json()
    assert body.get("status") == "nothing_to_memorize"

    progress = client.get("/memorize/progress?user_id=test_user&soul_id=test_soul")
    assert progress.status_code == 200
    out = progress.json()
    assert out.get("active") is False
    assert out.get("last_result") == "nothing_to_memorize"


# ---- force vs rebuild split regression tests ----

def _make_stub_payload(uid: str, soul_id: str, cid: str) -> dict:
    return {
        "user": {"user_id": uid, "soul_id": soul_id, "conversation_id": cid},
        "conversation_id": cid,
        "conversation": [
            {"role": "user", "name": uid, "content": "hello"},
            {"role": "assistant", "name": soul_id, "content": "hi"},
        ],
        "llm_profiles": {"default": {"provider": "openai", "api_key": "x", "base_url": "x", "chat_model": "x", "embed_model": "x"}},
    }


async def _noop_run(**kwargs) -> None:
    pass


def test_force_without_rebuild_does_not_archive_db(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """force=True (turn-valve path) must NOT rename / archive the DB file."""
    db_file = tmp_path / "TestSoul.db"
    db_file.write_text("fake-db")

    def fake_sqlite_current_path(uid: str, soul_id: str):
        return db_file

    def fake_write_conversation_state(*_a, **_k):
        return ({}, tmp_path / "fake_state.db")

    monkeypatch.setattr(main_module, "_get_service_from_payload", lambda *a, **k: object())
    monkeypatch.setattr(main_module, "_run_memorize_segments", _noop_run)
    monkeypatch.setattr(main_module, "_sqlite_current_path", fake_sqlite_current_path)
    monkeypatch.setattr(main_module, "_write_conversation_state", fake_write_conversation_state)
    monkeypatch.setattr(main_module, "_get_storage_dir", lambda _cfg: tmp_path)

    resp = client.post("/memorize?force=true", json=_make_stub_payload("u1", "TestSoul", "cid-f"))
    assert resp.status_code in (200, 202), f"status={resp.status_code} body={resp.text[:300]}"
    # DB file must still exist — force-only must not rename it.
    assert db_file.exists(), "force=True without rebuild must NOT archive the DB file"
    bak_files = list(tmp_path.glob("*.bak-*"))
    assert not bak_files, f"force=True without rebuild must not create .bak files, got: {bak_files}"


def test_rebuild_archives_db_and_service_reacquired_after_archive(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """rebuild=True must archive the DB and re-acquire the service AFTER the archive."""
    db_file = tmp_path / "TestSoul.db"
    db_file.write_text("fake-db")

    call_order: list[str] = []

    def fake_sqlite_current_path(uid: str, soul_id: str):
        return db_file

    def fake_clear_cached_services() -> None:
        call_order.append("cleared")

    def fake_get_service(payload):
        call_order.append("get_service")
        return object()

    def fake_write_conversation_state(*_a, **_k):
        return ({}, tmp_path / "fake_state.db")

    monkeypatch.setattr(main_module, "_get_service_from_payload", fake_get_service)
    monkeypatch.setattr(main_module, "_run_memorize_segments", _noop_run)
    monkeypatch.setattr(main_module, "_sqlite_current_path", fake_sqlite_current_path)
    monkeypatch.setattr(main_module, "_clear_cached_services", fake_clear_cached_services)
    monkeypatch.setattr(main_module, "_write_conversation_state", fake_write_conversation_state)
    monkeypatch.setattr(main_module, "_get_storage_dir", lambda _cfg: tmp_path)

    resp = client.post("/memorize?rebuild=true", json=_make_stub_payload("u1", "TestSoul", "cid-r"))
    assert resp.status_code in (200, 202), f"status={resp.status_code} body={resp.text[:300]}"

    # DB must have been renamed — .bak file should exist, original should not.
    bak_files = list(tmp_path.glob("*.bak-*"))
    assert bak_files, "rebuild=True must create a .bak archive file"
    assert not db_file.exists(), (
        f"rebuild=True must rename the original DB; call_order={call_order}; "
        f"tmp contents={list(tmp_path.iterdir())}"
    )

    # Service must be re-acquired AFTER the cache was cleared.
    assert "cleared" in call_order, f"clear_cached_services not called; order={call_order}"
    assert "get_service" in call_order, f"get_service_from_payload not called; order={call_order}"
    cleared_idx = call_order.index("cleared")
    get_service_idx = call_order.index("get_service")
    assert get_service_idx > cleared_idx, (
        "get_service_from_payload must be called AFTER clear_cached_services; "
        f"order was {call_order}"
    )
