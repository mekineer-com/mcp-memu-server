from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field, model_validator


class MemuTurnRequest(BaseModel):
    conversation_id: str
    user_id: str
    soul_id: str
    message: str
    history: list[dict[str, Any]] | None = None
    soul_card: str | None = None
    run_apimw: bool = True
    apply_turn_maintenance: bool = True
    debug: bool = False
    temperature: float | None = None
    channel_mode: str | None = None
    sender_name: str | None = None
    chat_name: str | None = None


class MemuRetrieveRequest(BaseModel):
    user_id: str
    soul_id: str
    query: str | None = None
    queries: list[dict[str, Any]] | None = None
    conversation_id: str | None = None
    method: str | None = None
    top_k: int | None = None
    as_of: str | None = None
    debug: bool = False
    history: list[dict[str, Any]] | None = None
    soul_card: str | None = None

    @model_validator(mode="after")
    def _require_query_or_queries(self) -> "MemuRetrieveRequest":
        query_text = str(self.query or "").strip()
        has_queries = isinstance(self.queries, list) and len(self.queries) > 0
        if not query_text and not has_queries:
            raise ValueError("Either query or queries is required")
        return self


class MemuMemorizeRequest(BaseModel):
    user_id: str
    soul_id: str
    conversation: list[dict[str, Any]]
    conversation_id: str | None = None
    force: bool = False
    time_zone: str | None = None
    time_zone_offset_min: int | None = None


class MemuConsolidateRequest(BaseModel):
    conversation_id: str
    user_id: str
    soul_id: str


class MemuIntentionsRequest(BaseModel):
    user_id: str
    soul_id: str
    status: str = "active"


class MemuStateRequest(BaseModel):
    action: str = Field(default="get", pattern="^(get|patch)$")
    conversation_id: str
    user_id: str
    soul_id: str
    updates: dict[str, Any] | None = None


def _scope(*, user_id: str, soul_id: str, conversation_id: str | None = None) -> dict[str, str]:
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    cid = str(conversation_id or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="user_id is required")
    if not sid:
        raise HTTPException(status_code=400, detail="soul_id is required")
    scope: dict[str, str] = {"user_id": uid, "soul_id": sid}
    if cid:
        scope["conversation_id"] = cid
    return scope


async def memu_turn_endpoint(
    req: MemuTurnRequest,
    *,
    conversation_retrieve: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
    conversation_turn: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    conversation_id = str(req.conversation_id or "").strip()
    if not conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id is required")
    message = str(req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")
    scope = _scope(user_id=req.user_id, soul_id=req.soul_id, conversation_id=conversation_id)

    retrieve_payload: dict[str, Any] = {
        "user": scope,
        "message": message,
        "query": message,
        "history": list(req.history or []),
        "build_turn_prompt": True,
        "debug": bool(req.debug),
    }
    if req.soul_card:
        retrieve_payload["soul_card"] = str(req.soul_card)
    if req.channel_mode:
        retrieve_payload["channel_mode"] = str(req.channel_mode)
    if req.sender_name:
        retrieve_payload["sender_name"] = str(req.sender_name)
    if req.chat_name:
        retrieve_payload["chat_name"] = str(req.chat_name)

    retrieve_out = await conversation_retrieve(conversation_id, retrieve_payload)
    should_respond = retrieve_out.get("should_respond", True)

    turn_user_prompt = str(retrieve_out.get("turn_user_prompt") or "").strip()
    turn_system_prompt = str(retrieve_out.get("turn_system_prompt") or "").strip()
    if not turn_user_prompt:
        raise HTTPException(status_code=502, detail="conversation_retrieve returned empty turn_user_prompt")

    prompt_override_payload = {
        "system_prompt": turn_system_prompt,
        "user_prompt": turn_user_prompt,
        "memory_cache": retrieve_out.get("memory_cache") or [],
        "intentions_active": retrieve_out.get("intentions_active") or {"items": []},
        "retrieve_rag": retrieve_out.get("result") or {"categories": [], "items": [], "resources": []},
    }
    retrieve_ms = retrieve_out.get("retrieve_ms")
    if isinstance(retrieve_ms, (int, float)):
        prompt_override_payload["retrieve_ms"] = int(retrieve_ms)

    turn_payload: dict[str, Any] = {
        "user": scope,
        "message": message,
        "history": list(req.history or []),
        "prompt_override_payload": prompt_override_payload,
        "run_apimw": bool(req.run_apimw),
        "apply_turn_maintenance": bool(req.apply_turn_maintenance),
        "should_respond": bool(should_respond),
        "debug": bool(req.debug),
    }
    if req.soul_card:
        turn_payload["soul_card"] = str(req.soul_card)
    if req.temperature is not None:
        turn_payload["temperature"] = float(req.temperature)

    turn_out = await conversation_turn(conversation_id, turn_payload)
    compact: dict[str, Any] = {
        "ok": bool(turn_out.get("ok", False)),
        "should_respond": bool(should_respond),
        "conversation_id": str(turn_out.get("conversation_id") or conversation_id),
        "response": str(turn_out.get("response") or ""),
        "apimw": turn_out.get("apimw"),
        "retrieve_ms": turn_out.get("retrieve_ms"),
        "turn_ms": turn_out.get("turn_ms"),
    }
    if req.debug:
        compact["debug_payload"] = turn_out
    return compact


async def memu_retrieve_endpoint(
    req: MemuRetrieveRequest,
    *,
    retrieve: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    query_text = str(req.query or "").strip()
    has_queries = isinstance(req.queries, list) and len(req.queries) > 0
    if not query_text and not has_queries:
        raise HTTPException(status_code=400, detail="Either query or queries is required")

    scope = _scope(user_id=req.user_id, soul_id=req.soul_id, conversation_id=req.conversation_id)
    payload: dict[str, Any] = {
        "user": scope,
        "debug": bool(req.debug),
    }
    if query_text:
        payload["query"] = query_text
    if has_queries:
        payload["queries"] = req.queries
    if req.method is not None:
        payload["method"] = str(req.method)
    if req.top_k is not None:
        payload["top_k"] = int(req.top_k)
    if req.as_of is not None:
        payload["as_of"] = str(req.as_of)
    if req.history is not None:
        payload["history"] = list(req.history)
    if req.soul_card:
        payload["soul_card"] = str(req.soul_card)
    return await retrieve(payload)


async def memu_memorize_endpoint(
    req: MemuMemorizeRequest,
    *,
    memorize: Callable[[dict[str, Any], bool], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    conversation_id = str(req.conversation_id or "").strip()
    scope = _scope(user_id=req.user_id, soul_id=req.soul_id, conversation_id=conversation_id)
    payload: dict[str, Any] = {
        "user": scope,
        "conversation": list(req.conversation or []),
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    if req.time_zone:
        payload["time_zone"] = str(req.time_zone)
    if req.time_zone_offset_min is not None:
        payload["time_zone_offset_min"] = int(req.time_zone_offset_min)
    return await memorize(payload, bool(req.force))


async def memu_consolidate_endpoint(
    req: MemuConsolidateRequest,
    *,
    force_consolidation: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    conversation_id = str(req.conversation_id or "").strip()
    if not conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id is required")
    scope = _scope(user_id=req.user_id, soul_id=req.soul_id, conversation_id=conversation_id)
    return await force_consolidation(conversation_id, {"user": scope, "conversation_id": conversation_id})


async def memu_intentions_endpoint(
    req: MemuIntentionsRequest,
    *,
    list_intentions: Callable[[str, str, str], Awaitable[Any]],
) -> dict[str, Any]:
    soul_id = _scope(user_id=req.user_id, soul_id=req.soul_id)["soul_id"]
    user_id = str(req.user_id).strip()
    items = await list_intentions(soul_id=soul_id, user_id=user_id, status=str(req.status or "active"))
    return {"intentions": items}


async def memu_state_endpoint(
    req: MemuStateRequest,
    *,
    get_state: Callable[[str, str | None, str | None], Awaitable[dict[str, Any]]],
    patch_state: Callable[[str, dict[str, Any] | None, str | None, str | None], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    conversation_id = str(req.conversation_id or "").strip()
    if not conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id is required")
    scope = _scope(user_id=req.user_id, soul_id=req.soul_id, conversation_id=conversation_id)
    if req.action == "get":
        return await get_state(conversation_id=conversation_id, soul_id=scope["soul_id"], user_id=scope["user_id"])
    return await patch_state(
        conversation_id=conversation_id,
        payload=dict(req.updates or {}),
        soul_id=scope["soul_id"],
        user_id=scope["user_id"],
    )
