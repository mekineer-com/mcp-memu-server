"""Scoped Claude transport for Atomic; Atomic retains transcript/tool ownership."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
import psutil
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, ConfigDict, Field
from memu.llm.claude_cli import claude_session_path

from app.services import owner, service_factory, state as conversation_state

_DELIVERY_BUFFER_BYTES = 16 * 1024 * 1024


def _task_name(conversation_id: str, user_id: str, soul_id: str) -> str:
    return "atomic-chat:" + _hash([{"user_id": user_id, "soul_id": soul_id}, conversation_id])


def require_sitting_idle(runtime: Any, conversation_id: str, *, user_id: str, soul_id: str) -> None:
    marker = _task_name(conversation_id, user_id, soul_id)
    if any(not task.done() and task.get_name() == marker for task in runtime._BACKGROUND_TASKS):
        raise HTTPException(status_code=409, detail="This Atomic sitting is still running")
    status = _native_status(runtime, conversation_id, {"user_id": user_id, "soul_id": soul_id})
    if status["writer_idle"] is not True:
        raise HTTPException(status_code=409, detail="Native writer exit is not established. Keep this history; start a new sitting if necessary.")


def _native_state(runtime: Any, cid: str, scope: dict[str, str]) -> dict[str, Any]:
    path = runtime._sqlite_current_path(**scope)
    if path is None:
        raise HTTPException(status_code=409, detail="Atomic chat requires a scoped Soul database")
    con = runtime._sqlite_connect(path)
    try:
        con.row_factory = sqlite3.Row
        runtime._sqlite_ensure_conversation_state_schema(con)
        return conversation_state.conversation_state_from_row(
            conversation_state.conversation_state_row(con, cid, **scope)) or {}
    finally:
        con.close()


def _writer_idle(receipt: dict[str, Any]) -> bool | None:
    if receipt.get("writer_exited"):
        return True
    process = receipt.get("process")
    if not process:
        return None
    if not receipt.get("parent_reaped"):
        try:
            if psutil.Process(process["pid"]).create_time() == process["created_at"]:
                return False
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            return None
    # A dead parent alone is not evidence that its process group stopped writing.
    if os.name == "posix":
        try:
            os.killpg(process["pid"], 0)
        except ProcessLookupError:
            return True
        except OSError:
            return None
    return None


def _reported_failure(receipt: dict[str, Any]) -> bool:
    result = receipt.get("native_result") or {}
    return (result.get("is_error") is True and isinstance(result.get("subtype"), str)
            and (result["subtype"] == "success" or result["subtype"].startswith("error_")))


def _native_status(runtime: Any, cid: str, scope: dict[str, str]) -> dict[str, Any]:
    saved = _native_state(runtime, cid, scope)
    receipt = saved.get("atomic_native_turn")
    if receipt is None:
        bound = saved.get("atomic_native_session_id")
        return {"session_id": bound, "turn": None, "writer_idle": None if bound else True}
    live = any(not task.done() and task.get_name() == _task_name(cid, **scope)
               for task in runtime._BACKGROUND_TASKS)
    idle = False if live else _writer_idle(receipt)
    unsubmitted = idle is True and receipt.get("submission") == "not_submitted"
    known_error = idle is True and _reported_failure(receipt)
    if not live and (receipt["status"] in {"prepared", "running"}
                     or (receipt["status"] == "uncertain" and (unsubmitted or known_error))):
        receipt = {**receipt, "status": "failed" if unsubmitted or known_error else "uncertain",
                   "error": ("This attempt was not submitted. " + str(receipt.get("error", ""))) if unsubmitted
                   else receipt.get("error") or "Native completion was not recorded; do not replay this turn."}
        if known_error:
            receipt["submission"] = "recorded"
            receipt["input_recorded"] = True
    if idle is True and not receipt.get("writer_exited"):
        receipt = {**receipt, "writer_exited": True}
    if receipt != saved["atomic_native_turn"]:
        _save_receipt(runtime, cid, scope, receipt, runtime._sqlite_current_path(**scope))
    return {"session_id": saved.get("atomic_native_session_id"), "turn": receipt, "writer_idle": idle}


def _save_receipt(runtime: Any, cid: str, scope: dict[str, str], receipt: dict[str, Any], path: Path) -> None:
    con = runtime._sqlite_connect(path)
    try:
        con.row_factory = sqlite3.Row
        runtime._sqlite_ensure_conversation_state_schema(con)
        con.execute("BEGIN IMMEDIATE")
        saved = conversation_state.conversation_state_from_row(
            conversation_state.conversation_state_row(con, cid, **scope)) or {}
        previous = saved.get("atomic_native_turn") or {}
        if (saved.get("atomic_native_session_id") != receipt["session_id"]
                or previous.get("turn_id") != receipt["turn_id"]
                or previous.get("attempt_id") != receipt.get("attempt_id")):
            raise RuntimeError("Native turn binding changed before checkpoint")
        conversation_state.write_conversation_state(cid, **scope, connection=con,
            sqlite_current_path=lambda *_args: path, updates={"atomic_native_turn": receipt})
        con.commit()
    finally:
        con.close()


def _scope_and_id(runtime: Any, user_id: str, soul_id: str, cid: str, *, admit: bool = True) -> tuple[dict[str, str], str]:
    scope = runtime._extract_scope({"user_id": user_id, "soul_id": soul_id})
    if not scope.get("user_id") or not scope.get("soul_id"):
        raise HTTPException(status_code=422, detail="Owner and Soul are required")
    try:
        scope["user_id"] = owner.require_owner(runtime._CONFIG, scope.get("user_id"))
    except owner.OwnerIdentityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if admit:
        runtime._require_soul_active(**scope)
    cid = cid.strip()
    if not cid or (cid.startswith("chat:") and not cid.startswith("chat:atomic-")):
        raise HTTPException(status_code=422, detail="An Atomic conversation is required")
    return scope, cid if cid.startswith("chat:atomic-") else f"chat:atomic-{cid}"


class AtomicChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    soul_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    operation: Literal["new", "continue", "retry"]
    purpose: Literal["reply", "recap"] = "reply"
    prompt: str = Field(min_length=1)
    system_prompt: str | None = None
    bridge_url: str | None = None
    profile_fingerprint: str


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _wire_events(event: dict[str, Any]) -> list[dict[str, Any]]:
    if event["type"] == "context_usage":
        return [{"type": "usage", "message_id": event["message_id"], "usage": event["usage"]}]
    if event["type"] not in {"assistant", "user"}:
        return []
    message = event["message"]
    identity = {"record_id": event["uuid"], "message_id": message.get("id"),
                "parent_call_id": event.get("parent_tool_use_id")}
    events = []
    content = message["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    for index, block in enumerate(content):
        base = {**identity, "block_index": index}
        if event["type"] == "assistant" and block["type"] == "text":
            events.append({"type": "text", **base, "mode": "replace", "text": block["text"]})
        elif event["type"] == "assistant" and block["type"] == "tool_use":
            events.append({"type": "tool_call", **base, "call_id": block["id"],
                           "name": block["name"].removeprefix("mcp__atomic__"), "input": block["input"]})
        elif event["type"] == "user" and block["type"] == "tool_result":
            events.append({"type": "tool_result", **base, "call_id": block["tool_use_id"],
                           "content": block.get("content"), "is_error": block.get("is_error", False)})
    return events


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
    names = [tool.name for tool in tools.tools]
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
        raise HTTPException(status_code=502, detail="Atomic bridge returned an invalid tool name")
    return [f"mcp__atomic__{name}" for name in names]


def register_atomic_chat_routes(app: FastAPI, *, runtime: Any) -> None:
    @app.get("/integration/atomic/chat/status", tags=["integration"])
    async def atomic_chat_status(user_id: str, soul_id: str, conversation_id: str, turn_id: str):
        scope, cid = _scope_and_id(runtime, user_id, soul_id, conversation_id, admit=False)
        status = _native_status(runtime, cid, scope)
        if status["turn"] is not None and status["turn"]["turn_id"] != turn_id:
            earlier = status["turn"].get("uncertain_turn")
            if earlier is None or earlier["turn_id"] != turn_id:
                raise HTTPException(status_code=409, detail={
                    "code": "atomic_native_turn_missing",
                    "error": "Latest native receipt belongs to a different turn; this request's submission cannot be inferred",
                    "session_id": status["session_id"], "turn": None,
                    "writer_idle": status["writer_idle"],
                })
            status = {**status, "turn": earlier}
        return status

    @app.post("/integration/atomic/chat", tags=["integration"])
    async def atomic_chat(body: AtomicChatRequest, authorization: str | None = Header(default=None)):
        config = deepcopy(runtime._CONFIG)
        scope, cid = _scope_and_id(runtime, body.user_id, body.soul_id, body.conversation_id)
        profile = claude_chat_profile(config)
        if body.profile_fingerprint != profile["fingerprint"]:
            raise HTTPException(status_code=409, detail="Atomic chat profile changed; reload it before sending")
        if body.operation == "new" and not body.system_prompt:
            raise HTTPException(status_code=422, detail="A new sitting requires its Soul Presence")
        if body.operation == "new" and body.purpose == "recap":
            raise HTTPException(status_code=422, detail="A recap must resume the sitting; one-shot close-out is a separate operation")
        if body.operation != "new" and body.system_prompt is not None:
            raise HTTPException(status_code=422, detail="Resume uses the original Soul Presence")
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
        marker = _task_name(cid, **scope)
        require_sitting_idle(runtime, cid, **scope)
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
            if native_id and state.get("atomic_native_workspace") != str(workspace):
                raise HTTPException(status_code=409, detail=f"The Claude workspace changed from {state.get('atomic_native_workspace')} to {workspace}; restore it before resuming")
            if not native_id and body.operation == "new":
                native_id = str(uuid.uuid4())
            elif not native_id:
                raise HTTPException(status_code=409, detail="This sitting has no native session to resume")
            previous = state.get("atomic_native_turn")
            if previous:
                if previous.get("attempt_id") == body.attempt_id:
                    raise HTTPException(status_code=409, detail="This attempt already exists; recover it through chat/status without another call")
                if body.purpose == "reply" and previous.get("uncertain_turn"):
                    raise HTTPException(status_code=409, detail="The ordinary outcome remains unknown; only close-out may continue")
                if previous["turn_id"] == body.turn_id and previous["purpose"] != body.purpose:
                    raise HTTPException(status_code=409, detail="Saved turn purpose does not match")
                if previous["turn_id"] == body.turn_id and previous["status"] == "complete":
                    raise HTTPException(status_code=409, detail="This turn is complete; recover it through chat/status without another call")
                if previous["status"] in {"prepared", "running", "uncertain"} and body.purpose != "recap":
                    raise HTTPException(status_code=409, detail="The previous outcome is unknown. Do not replay it; close out after writer exit.")
            exists = claude_session_path(workspace, native_id).exists()
            if body.operation == "new" and (exists or (previous and previous.get("initialized"))):
                raise HTTPException(status_code=409, detail="This native session is already initialized; reconcile and resume it, do not repeat initialization")
            if body.operation != "new" and not exists:
                raise HTTPException(status_code=409, detail="The native session file is missing; do not initialize a replacement")
            receipt = {"session_id": native_id, "turn_id": body.turn_id, "attempt_id": body.attempt_id, "purpose": body.purpose,
                       "status": "prepared", "writer_exited": False, "initialized": exists,
                       "submission": "not_submitted",
                       # Retry replaces the attempt, not the input's recorded history.
                       "input_recorded": bool(previous and previous["turn_id"] == body.turn_id
                           and (previous.get("input_recorded") or previous.get("submission") == "recorded"))}
            if previous:
                uncertain = (previous if previous["purpose"] == "reply" and previous["status"] == "uncertain"
                             else previous.get("uncertain_turn"))
                if uncertain:
                    receipt["uncertain_turn"] = uncertain
            conversation_state.write_conversation_state(cid, **scope, connection=con,
                sqlite_current_path=lambda *_args: db_path, updates={
                    "atomic_native_session_id": native_id, "atomic_native_workspace": str(workspace),
                    "atomic_native_turn": receipt,
                })
            con.commit()
        finally:
            con.close()

        queue: asyncio.Queue[tuple[bytes, bool]] = asyncio.Queue()
        queued_bytes = 0
        handed_off = asyncio.Event()
        attached = True

        def detach() -> None:
            nonlocal attached, queued_bytes
            attached = False
            while not queue.empty():
                queue.get_nowait()
            queued_bytes = 0
            handed_off.set()

        def publish(event: dict[str, Any], *, terminal: bool = False) -> None:
            nonlocal queued_bytes
            if attached:
                frame = (json.dumps({**event, "terminal": terminal}, ensure_ascii=False) + "\n").encode("utf-8")
                if queued_bytes + len(frame) > _DELIVERY_BUFFER_BYTES:
                    detach()
                    frame = (json.dumps({"type": "error", "code": "delivery_overflow", "terminal": True,
                        "error": "Atomic event delivery overflowed; native work may still be running. Do not replay this turn."}) + "\n").encode()
                    queued_bytes = len(frame)
                    queue.put_nowait((frame, True))
                    return
                queued_bytes += len(frame)
                queue.put_nowait((frame, terminal))

        async def produce() -> None:
            public_profile = {key: value for key, value in profile.items() if key != "workspace_root"}
            session = {"type": "session", "session_id": native_id, "turn_id": body.turn_id,
                       "attempt_id": body.attempt_id, "profile": public_profile}
            async def on_event(event):
                if event.get("type") == "process":
                    if event["state"] == "started":
                        receipt["status"] = "running"
                        receipt["process"] = {"pid": event["pid"], "created_at": None}
                        try:
                            child = psutil.Process(event["pid"])
                            receipt["process"] = {"pid": child.pid, "created_at": child.create_time()}
                        except psutil.NoSuchProcess:
                            # Early launch rejection may exit before identity lookup.
                            pass
                    elif event["state"] == "input_started":
                        _save_receipt(runtime, cid, scope, {**receipt, "submission": "unknown"}, db_path)
                        receipt["submission"] = "unknown"
                        return
                    elif event["state"] == "reaped":
                        receipt["parent_reaped"] = True
                        if _writer_idle(receipt) is True:
                            receipt["writer_exited"] = True
                    _save_receipt(runtime, cid, scope, receipt, db_path)
                elif event.get("type") == "result":
                    receipt["native_result"] = event
                    _save_receipt(runtime, cid, scope, receipt, db_path)
                elif event.get("type") == "system" and event.get("subtype") == "init":
                    if event["session_id"] != native_id:
                        raise RuntimeError("Claude returned a different native session")
                    receipt["initialized"] = True
                    _save_receipt(runtime, cid, scope, receipt, db_path)
                    publish({**session, "actual_model": event.get("model")})
                else:
                    if event.get("type") == "context_usage":
                        receipt["usage"] = {"message_id": event["message_id"], "usage": event["usage"]}
                    for frame in _wire_events(event):
                        publish(frame)
                    # A buffered native burst must let the ready HTTP reader run.
                    await asyncio.sleep(0)
            publish(session)
            try:
                await service.chat(body.prompt, system_prompt=body.system_prompt, op="atomic_chat", trace_id=body.turn_id,
                    session_id=native_id if body.operation == "new" else None,
                    resume_session_id=native_id if body.operation != "new" else None,
                    workspace=workspace, cli_tools=tools, on_event=on_event)
                candidate = receipt.get("native_result")
                if candidate is None:
                    raise RuntimeError("Claude completed without its terminal result")
                empty_recap = body.purpose == "recap" and not candidate["result"].strip()
                receipt.update(status="failed" if empty_recap else "complete", text=candidate["result"])
                receipt["submission"] = "recorded"
                receipt["input_recorded"] = True
                receipt["initialized"] = True
                if empty_recap:
                    receipt["error"] = "The recap was empty. Retry close-out."
                _save_receipt(runtime, cid, scope, receipt, db_path)
                runtime._record_call("atomic_chat", {"user": scope}, ok=not empty_recap,
                    error=receipt.get("error"), info={"conversation_id": cid, "turn_id": body.turn_id})
                publish({"type": "error", "error": receipt["error"]} if empty_recap else
                        {"type": "complete", "session_id": native_id, "text": candidate["result"]}, terminal=True)
            except asyncio.CancelledError:
                reason = ("Planned shutdown" if runtime._SHUTDOWN_STATE.get("draining")
                          else "Atomic chat producer cancelled")
                receipt.update(status="uncertain", error=f"{reason}; native outcome unknown")
                try:
                    _save_receipt(runtime, cid, scope, receipt, db_path)
                finally:
                    publish({"type": "error", "error": receipt["error"]}, terminal=True)
                raise
            except Exception as exc:
                message = str(exc).replace(token, "[redacted]") if token else str(exc)
                if receipt["status"] == "prepared":
                    receipt["writer_exited"] = True
                # Delivery/checkpoint errors must not erase a validated answer.
                if receipt["status"] != "complete":
                    known_error = _reported_failure(receipt) and _writer_idle(receipt) is True
                    receipt["status"] = "failed" if receipt["submission"] == "not_submitted" or known_error else "uncertain"
                    if known_error:
                        receipt["submission"] = "recorded"
                        receipt["input_recorded"] = True
                        receipt["writer_exited"] = True
                receipt["error"] = message
                receipt["initialized"] = receipt["initialized"] or claude_session_path(workspace, native_id).exists()
                try:
                    _save_receipt(runtime, cid, scope, receipt, db_path)
                except Exception:
                    runtime.logger.exception("Could not checkpoint failed Atomic native turn")
                runtime._record_call("atomic_chat", {"user": scope}, ok=False, error=message)
                publish({"type": "error", "error": message}, terminal=True)
            # Native work has settled; HTTP handoff still participates in drain,
            # but must not keep the conversation's writer marked busy.
            asyncio.current_task().set_name(marker + ":delivery")
            await handed_off.wait()

        task = asyncio.create_task(produce(), name=marker)
        runtime._BACKGROUND_TASKS.add(task)
        def finished(done):
            runtime._BACKGROUND_TASKS.discard(done)
            if not done.cancelled() and (error := done.exception()) is not None:
                runtime.logger.error("Atomic chat producer failed: %s", str(error).replace(token, "[redacted]") if token else str(error))
        task.add_done_callback(finished)

        async def stream():
            nonlocal queued_bytes
            try:
                while True:
                    frame, terminal = await queue.get()
                    queued_bytes -= len(frame)
                    yield frame
                    if terminal:
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
