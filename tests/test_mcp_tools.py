from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app import main
from app.services import mcp_tools


def test_mcp_tool_registration_exposes_expected_operations() -> None:
    assert main._has_mcp is True
    operation_ids = set(getattr(main, "mcp").operation_map.keys())
    assert operation_ids == {
        "memu_turn",
        "memu_retrieve",
        "memu_memorize",
        "memu_consolidate",
        "memu_intentions",
    }


@pytest.mark.asyncio
async def test_memu_turn_orchestrates_retrieve_then_turn() -> None:
    captured: list[tuple[str, str, dict[str, Any]]] = []

    async def fake_retrieve(conversation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured.append(("retrieve", conversation_id, payload))
        return {
            "turn_system_prompt": "system",
            "turn_user_prompt": "user",
            "memory_cache": ["a"],
            "intentions_active": {"items": []},
            "result": {"categories": [], "items": [], "resources": []},
            "retrieve_ms": 42,
        }

    async def fake_turn(conversation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured.append(("turn", conversation_id, payload))
        return {
            "ok": True,
            "conversation_id": conversation_id,
            "response": "done",
            "response_target": "private",
            "response_peer": "",
            "apimw": "not_started",
            "retrieve_ms": 42,
            "turn_ms": 9,
        }

    out = await mcp_tools.memu_turn_endpoint(
        mcp_tools.MemuTurnRequest(
            conversation_id="c1",
            user_id="u1",
            soul_id="s1",
            message="hello",
            history=[{"role": "user", "content": "hello"}],
            user_name="Alice",
            chat_name="Alice",
            chat_type="dm",
            memorize_chat=False,
        ),
        conversation_retrieve=fake_retrieve,
        conversation_turn=fake_turn,
    )

    assert out["ok"] is True
    assert out["response"] == "done"
    assert out["response_target"] == "private"
    assert out["response_peer"] == ""
    assert [row[0] for row in captured] == ["retrieve", "turn"]
    retrieve_payload = captured[0][2]
    turn_payload = captured[1][2]
    assert retrieve_payload.get("user_name") == "Alice"
    assert retrieve_payload.get("chat_name") == "Alice"
    assert retrieve_payload.get("chat_type") == "dm"
    assert retrieve_payload.get("memorize_chat") is False
    assert turn_payload.get("user_name") == "Alice"
    assert turn_payload.get("chat_name") == "Alice"
    assert turn_payload.get("chat_type") == "dm"
    assert turn_payload.get("memorize_chat") is False
    override_payload = turn_payload.get("prompt_override_payload")
    assert isinstance(override_payload, dict)
    assert override_payload.get("retrieve_ms") == 42
    assert override_payload.get("user_prompt") == "user"


@pytest.mark.asyncio
async def test_memu_turn_raises_when_retrieve_returns_empty_prompt() -> None:
    async def fake_retrieve(_conversation_id: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {"turn_user_prompt": ""}

    async def fake_turn(_conversation_id: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {}

    with pytest.raises(HTTPException) as exc:
        await mcp_tools.memu_turn_endpoint(
            mcp_tools.MemuTurnRequest(
                conversation_id="c1",
                user_id="u1",
                soul_id="s1",
                message="hello",
            ),
            conversation_retrieve=fake_retrieve,
            conversation_turn=fake_turn,
        )

    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_memu_turn_omits_blank_chat_fields() -> None:
    captured: list[tuple[str, str, dict[str, Any]]] = []

    async def fake_retrieve(conversation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured.append(("retrieve", conversation_id, payload))
        return {
            "turn_system_prompt": "system",
            "turn_user_prompt": "user",
            "memory_cache": [],
            "intentions_active": {"items": []},
            "result": {"categories": [], "items": [], "resources": []},
        }

    async def fake_turn(conversation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured.append(("turn", conversation_id, payload))
        return {
            "ok": True,
            "conversation_id": conversation_id,
            "response": "done",
            "apimw": "not_started",
            "retrieve_ms": 1,
            "turn_ms": 1,
        }

    await mcp_tools.memu_turn_endpoint(
        mcp_tools.MemuTurnRequest(
            conversation_id="c1",
            user_id="u1",
            soul_id="s1",
            message="hello",
            chat_name="   ",
            chat_type="",
        ),
        conversation_retrieve=fake_retrieve,
        conversation_turn=fake_turn,
    )

    retrieve_payload = captured[0][2]
    turn_payload = captured[1][2]
    assert "chat_name" not in retrieve_payload
    assert "chat_type" not in retrieve_payload
    assert "memorize_chat" not in retrieve_payload
    assert "chat_name" not in turn_payload
    assert "chat_type" not in turn_payload
    assert "memorize_chat" not in turn_payload


def test_memu_retrieve_request_requires_query_or_queries() -> None:
    with pytest.raises(ValidationError):
        mcp_tools.MemuRetrieveRequest(
            user_id="u1",
            soul_id="s1",
            query="   ",
        )

    with pytest.raises(ValidationError):
        mcp_tools.MemuRetrieveRequest(
            user_id="u1",
            soul_id="s1",
            queries=[],
        )


@pytest.mark.asyncio
async def test_mcp_memu_memorize_returns_before_background_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    finished = asyncio.Event()

    async def slow_background_work() -> None:
        await asyncio.sleep(0.2)
        finished.set()

    async def fake_memorize(
        _payload: dict[str, Any],
        background_tasks: Any,
        _force: bool,
    ) -> JSONResponse:
        background_tasks.add_task(slow_background_work)
        return JSONResponse(status_code=202, content={"ok": True, "status": "accepted"})

    monkeypatch.setattr(main, "memorize", fake_memorize)

    req = mcp_tools.MemuMemorizeRequest(
        user_id="u1",
        soul_id="s1",
        conversation=[{"role": "user", "content": "hello"}],
    )
    out = await main.mcp_memu_memorize(req)

    assert out["ok"] is True
    assert out["status"] == "accepted"
    assert not finished.is_set()

    await asyncio.wait_for(finished.wait(), timeout=1.0)


@pytest.mark.asyncio
async def test_memu_state_get_and_patch_routes() -> None:
    calls: list[tuple[str, str, Any, Any]] = []

    async def fake_get(conversation_id: str, soul_id: str | None, user_id: str | None) -> dict[str, Any]:
        calls.append(("get", conversation_id, soul_id, user_id))
        return {"ok": True, "state": {"digest_cursor": 1}}

    async def fake_patch(
        conversation_id: str,
        payload: dict[str, Any] | None,
        soul_id: str | None,
        user_id: str | None,
    ) -> dict[str, Any]:
        calls.append(("patch", conversation_id, soul_id, user_id))
        return {"ok": True, "state": payload or {}}

    get_out = await mcp_tools.memu_state_endpoint(
        mcp_tools.MemuStateRequest(
            action="get",
            conversation_id="c1",
            user_id="u1",
            soul_id="s1",
        ),
        get_state=fake_get,
        patch_state=fake_patch,
    )
    patch_out = await mcp_tools.memu_state_endpoint(
        mcp_tools.MemuStateRequest(
            action="patch",
            conversation_id="c1",
            user_id="u1",
            soul_id="s1",
            updates={"digest_cursor": 9},
        ),
        get_state=fake_get,
        patch_state=fake_patch,
    )

    assert get_out["ok"] is True
    assert patch_out["ok"] is True
    assert calls[0] == ("get", "c1", "s1", "u1")
    assert calls[1] == ("patch", "c1", "s1", "u1")
