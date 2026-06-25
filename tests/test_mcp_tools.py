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
            "turn_prompt_active_since": 100.0,
        }

    async def fake_turn(conversation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured.append(("turn", conversation_id, payload))
        return {
            "ok": True,
            "conversation_id": conversation_id,
            "response": "done",
            "response_target": "private",
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
            allow_public_response=False,
        ),
        conversation_retrieve=fake_retrieve,
        conversation_turn=fake_turn,
    )

    assert out["ok"] is True
    assert out["response"] == "done"
    assert out["response_target"] == "private"
    assert isinstance(out.get("trace_id"), str)
    assert len(str(out.get("trace_id") or "")) == 32
    assert [row[0] for row in captured] == ["retrieve", "turn"]
    retrieve_payload = captured[0][2]
    turn_payload = captured[1][2]
    assert retrieve_payload.get("user_name") == "Alice"
    assert retrieve_payload.get("chat_name") == "Alice"
    assert retrieve_payload.get("chat_type") == "dm"
    assert retrieve_payload.get("memorize_chat") is False
    assert retrieve_payload.get("allow_public_response") is False
    assert "mental_health_addon" not in retrieve_payload
    assert "time_zone" not in retrieve_payload
    assert "time_zone_offset_min" not in retrieve_payload
    assert "load_source_history" not in retrieve_payload
    assert turn_payload.get("user_name") == "Alice"
    assert turn_payload.get("chat_name") == "Alice"
    assert turn_payload.get("chat_type") == "dm"
    assert turn_payload.get("memorize_chat") is False
    assert turn_payload.get("allow_public_response") is False
    assert "time_zone" not in turn_payload
    assert "time_zone_offset_min" not in turn_payload
    assert "load_source_history" not in turn_payload
    assert retrieve_payload.get("trace_id") == out.get("trace_id")
    assert turn_payload.get("trace_id") == out.get("trace_id")
    override_payload = turn_payload.get("prompt_override_payload")
    assert isinstance(override_payload, dict)
    assert override_payload.get("retrieve_ms") == 42
    assert override_payload.get("user_prompt") == "user"
    assert override_payload.get("generated_by") == "conversation_retrieve"
    assert override_payload.get("active_since") == 100.0


@pytest.mark.asyncio
async def test_memu_turn_requests_source_history_for_whatsapp() -> None:
    captured: list[tuple[str, str, dict[str, Any]]] = []

    async def fake_retrieve(conversation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured.append(("retrieve", conversation_id, payload))
        return {
            "turn_system_prompt": "system",
            "turn_user_prompt": "user",
            "turn_history": [{"role": "user", "content": "from retrieve history"}],
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
            "response_target": "respond",
            "apimw": "not_started",
            "retrieve_ms": 1,
            "turn_ms": 1,
        }

    await mcp_tools.memu_turn_endpoint(
        mcp_tools.MemuTurnRequest(
            conversation_id="whatsapp:dm:15133278228@s.whatsapp.net",
            user_id="u1",
            soul_id="Siri",
            message="hello",
            external_message_id="CURRENT",
        ),
        conversation_retrieve=fake_retrieve,
        conversation_turn=fake_turn,
    )

    retrieve_payload = captured[0][2]
    turn_payload = captured[1][2]
    assert captured[0][1] == "whatsapp:dm:15133278228@s.whatsapp.net"
    assert captured[1][1] == "whatsapp:dm:15133278228@s.whatsapp.net"
    assert retrieve_payload["user"]["conversation_id"] == "whatsapp:dm:15133278228@s.whatsapp.net"
    assert turn_payload["user"]["conversation_id"] == "whatsapp:dm:15133278228@s.whatsapp.net"
    assert retrieve_payload["load_source_history"] is True
    assert retrieve_payload["is_live_turn"] is True
    assert retrieve_payload["external_message_id"] == "CURRENT"
    assert turn_payload["load_source_history"] is True
    assert turn_payload["is_live_turn"] is True
    assert turn_payload["history"] == []
    assert turn_payload["external_message_id"] == "CURRENT"


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
            "response_target": "listen",
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
    assert "mental_health_addon" not in retrieve_payload
    assert "chat_name" not in turn_payload
    assert "chat_type" not in turn_payload
    assert "memorize_chat" not in turn_payload


@pytest.mark.asyncio
async def test_memu_turn_forwards_mental_health_addon_override() -> None:
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
            "response_target": "listen",
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
            mental_health_addon=False,
        ),
        conversation_retrieve=fake_retrieve,
        conversation_turn=fake_turn,
    )

    assert captured[0][2].get("mental_health_addon") is False


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
