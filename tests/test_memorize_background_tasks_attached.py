"""Contract test for /memorize: the endpoint's BackgroundTasks must actually run.

History: the endpoint adds `_run_memorize_episodes` via `background_tasks.add_task(...)`
but returns a `JSONResponse(status_code=202, ...)` built inline. In FastAPI,
when an endpoint returns a Response object directly, tasks on the injected
`BackgroundTasks` parameter are NOT auto-attached — you have to pass
`background=background_tasks` to the Response. Without it, add_task is a
silent no-op and the memorize batches never run.

Turn-triggered memorize goes through `_run_forced_memorize_from_turn` which
creates its own BackgroundTasks and awaits it manually, so it's unaffected.
Only direct `POST /memorize` calls (Re-memorize chat button, st-api.sh, and
every agent smoke-test driver) hit this hole.

This test is the regression guard: it stubs `_run_memorize_episodes` with a
recorder, posts to /memorize, and asserts the task actually ran.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

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
    monkeypatch.setattr(main_module, "_run_memorize_episodes", fake_run)

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
    resp = client.post("/memorize?force=true", json=payload)
    assert resp.status_code == 202, f"unexpected status: {resp.status_code} body={resp.text[:300]}"
    body = resp.json()
    assert body.get("status") == "accepted"
    assert body.get("segment_count") == 1

    # TestClient blocks until BackgroundTasks complete, so by the time .post()
    # returns, our fake_run should have been invoked.
    assert recorded, (
        "background task did NOT run — JSONResponse is missing background=background_tasks; "
        "check app/main.py memorize endpoint return"
    )
    assert recorded[0]["ran"] is True
    assert recorded[0]["segment_count"] >= 1
