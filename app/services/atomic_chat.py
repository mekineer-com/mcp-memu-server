"""Scoped Claude transport for Atomic; Atomic retains transcript/tool ownership."""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, ConfigDict, Field

from app.services import owner, service_factory, state as conversation_state


class AtomicChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    soul_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    operation: Literal["new", "continue", "retry"]
    purpose: Literal["reply", "recap"] = "reply"
    prompt: str = Field(min_length=1)
    system_prompt: str | None = None
    bridge_url: str | None = None
    profile_fingerprint: str


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def claude_chat_profile(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("claude_code") is not True:
        raise HTTPException(status_code=409, detail="Atomic Claude transport requires claude_code=true")
    capacity = config.get("claude_code_context_window_tokens")
    if type(capacity) is not int or capacity <= 20000:
        raise HTTPException(status_code=409, detail="Set claude_code_context_window_tokens above 20000 to the usable allowance")
    root = Path(config.get("claude_code_workspace") or Path.home() / ".cache" / "memu-claude-workspace").expanduser().resolve()
    profile = {
        "backend": "claude_code", "model": str(config.get("claude_code_model", "claude-opus-4-7")).strip() or "claude-opus-4-7",
        "context_window_tokens": capacity,
        "timeout_seconds": service_factory._cfg_int(config, "claude_code_timeout_seconds", 3600, minimum=1),
        "effort": str(config.get("claude_code_effort", "medium")).strip(),
    }
    return {**profile, "fingerprint": _hash([profile, str(root)]), "workspace_root": root}


async def _bridge_tools(url: str, authorization: str) -> list[str]:
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
            or parsed.path != "/openalma/chat-tools/mcp" or parsed.query or parsed.fragment or parsed.username is not None):
        raise HTTPException(status_code=422, detail="bridge_url must be the local Atomic scoped-tool endpoint")
    async with asyncio.timeout(10), httpx.AsyncClient(headers={"Authorization": authorization}, timeout=10, trust_env=False) as client:
        async with streamable_http_client(url, http_client=client) as (read, write, _session):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                if tools.nextCursor:
                    raise HTTPException(status_code=502, detail="Atomic chat tools must fit one tools/list response")
                return [f"mcp__atomic__{tool.name}" for tool in tools.tools]


def register_atomic_chat_routes(app: FastAPI, *, runtime: Any) -> None:
    @app.post("/integration/atomic/chat", tags=["integration"])
    async def atomic_chat(body: AtomicChatRequest, authorization: str | None = Header(default=None)):
        config = deepcopy(runtime._CONFIG)
        scope = runtime._extract_scope({"user_id": body.user_id, "soul_id": body.soul_id})
        if not scope.get("user_id") or not scope.get("soul_id"):
            raise HTTPException(status_code=422, detail="Owner and Soul are required")
        try:
            scope["user_id"] = owner.require_owner(config, scope.get("user_id"))
        except owner.OwnerIdentityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        runtime._require_soul_active(**scope)
        profile = claude_chat_profile(config)
        if body.profile_fingerprint != profile["fingerprint"]:
            raise HTTPException(status_code=409, detail="Atomic chat profile changed; reload it before sending")
        if body.operation == "new" and not body.system_prompt:
            raise HTTPException(status_code=422, detail="A new sitting requires its Soul Presence")
        if body.operation != "new" and body.system_prompt is not None:
            raise HTTPException(status_code=422, detail="Resume uses the original Soul Presence")
        cid = body.conversation_id.strip()
        if not cid:
            raise HTTPException(status_code=422, detail="conversation_id is required")
        if not cid.startswith("chat:atomic-"):
            if cid.startswith("chat:"):
                raise HTTPException(status_code=422, detail="An Atomic conversation is required")
            cid = f"chat:atomic-{cid}"
        service = runtime._get_service_from_payload({"user": scope}, operation="other_reply")
        db_path = runtime._sqlite_current_path(**scope)
        if db_path is None:
            raise HTTPException(status_code=409, detail="Atomic chat requires a scoped Soul database")
        # The cache may outlive config reload; never launch a different captured profile.
        if (not service._claude_code or service._claude_code_model != profile["model"]
                or service._claude_code_context_window_tokens != profile["context_window_tokens"]
                or (service._claude_code_effort or "") != profile["effort"]
                or service._claude_code_timeout_seconds != profile["timeout_seconds"]):
            raise HTTPException(status_code=409, detail="Claude service profile changed; reload before sending")
        token = (authorization or "").removeprefix("Bearer ")
        tools = {"mcp_servers": {}, "allowed_tools": []}
        if body.purpose == "reply":
            if not authorization or not authorization.startswith("Bearer ") or not token or not body.bridge_url:
                raise HTTPException(status_code=422, detail="Reply requires a scoped bridge URL and Bearer credential")
            try:
                names = await _bridge_tools(body.bridge_url, authorization)
            except HTTPException:
                raise
            except Exception as exc:
                runtime.logger.warning("Atomic tool discovery failed: %s", str(exc).replace(token, "[redacted]"))
                raise HTTPException(status_code=502, detail="Atomic scoped tool discovery failed") from None
            tools = {"mcp_servers": {"atomic": {"type": "http", "url": body.bridge_url,
                                               "headers": {"Authorization": authorization}}}, "allowed_tools": names}
        workspace = profile["workspace_root"] / "atomic" / _hash([scope["user_id"], scope["soul_id"]])
        marker = "atomic-chat:" + _hash([scope, cid])
        if any(not task.done() and task.get_name() == marker for task in runtime._BACKGROUND_TASKS):
            raise HTTPException(status_code=409, detail="This Atomic sitting is still running")
        con = runtime._sqlite_connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            runtime._sqlite_ensure_conversation_state_schema(con)
            con.execute("BEGIN IMMEDIATE")
            state = conversation_state.conversation_state_from_row(
                conversation_state.conversation_state_row(con, cid, **scope)) or {}
            if state.get("atomic_session_ended_at"):
                raise HTTPException(status_code=409, detail="This Atomic sitting has ended")
            native_id = state.get("atomic_native_session_id")
            if body.operation == "new":
                if native_id:
                    raise HTTPException(status_code=409, detail="This sitting already has a native session; do not replay its first turn")
                native_id = str(uuid.uuid4())
                conversation_state.write_conversation_state(cid, **scope, connection=con,
                    sqlite_current_path=lambda *_args: db_path, updates={
                        "atomic_native_session_id": native_id, "atomic_native_workspace": str(workspace),
                    })
                con.commit()
            elif not native_id:
                raise HTTPException(status_code=409, detail="This sitting has no native session to resume")
            elif state.get("atomic_native_workspace") != str(workspace):
                raise HTTPException(status_code=409, detail="The Claude workspace changed; restore it before resuming")
        finally:
            con.close()

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        handed_off = asyncio.Event()
        attached = True

        def detach() -> None:
            nonlocal attached
            attached = False
            while not queue.empty():
                queue.get_nowait()
            handed_off.set()

        def publish(event: dict[str, Any], *, terminal: bool = False) -> None:
            if attached:
                queue.put_nowait({**event, "terminal": terminal})

        async def produce() -> None:
            candidate = None
            async def on_event(event):
                nonlocal candidate
                if event.get("type") == "result":
                    candidate = event
                else:
                    publish(event)
            public_profile = {key: value for key, value in profile.items() if key != "workspace_root"}
            publish({"type": "session", "session_id": native_id, "turn_id": body.turn_id, "profile": public_profile})
            try:
                await service.chat(body.prompt, system_prompt=body.system_prompt, op="atomic_chat", trace_id=body.turn_id,
                    session_id=native_id if body.operation == "new" else None,
                    resume_session_id=native_id if body.operation != "new" else None,
                    workspace=workspace, cli_tools=tools, on_event=on_event)
                if candidate is None:
                    raise RuntimeError("Claude completed without its terminal result")
                runtime._record_call("atomic_chat", {"user": scope}, ok=True, info={"conversation_id": cid, "turn_id": body.turn_id})
                publish(candidate, terminal=True)
            except asyncio.CancelledError:
                publish({"type": "error", "error": "Atomic chat producer stopped"}, terminal=True)
                raise
            except Exception as exc:
                message = str(exc).replace(token, "[redacted]") if token else str(exc)
                runtime._record_call("atomic_chat", {"user": scope}, ok=False, error=message)
                publish({"type": "error", "error": message}, terminal=True)
            await handed_off.wait()

        task = asyncio.create_task(produce(), name=marker)
        runtime._BACKGROUND_TASKS.add(task)
        def finished(done):
            runtime._BACKGROUND_TASKS.discard(done)
            if not done.cancelled() and (error := done.exception()) is not None:
                runtime.logger.error("Atomic chat producer failed: %s", str(error).replace(token, "[redacted]") if token else str(error))
        task.add_done_callback(finished)

        async def stream():
            try:
                while True:
                    event = await queue.get()
                    yield json.dumps(event, ensure_ascii=False) + "\n"
                    if event["terminal"]:
                        handed_off.set()
                        return
            finally:
                detach()

        class ChatStream(StreamingResponse):
            async def __call__(self, scope, receive, send):
                try:
                    await super().__call__(scope, receive, send)
                finally:
                    detach()

        return ChatStream(stream(), media_type="application/x-ndjson")
