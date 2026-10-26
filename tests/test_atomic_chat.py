import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from app import db, main
from app.services import atomic_chat, state


@pytest.fixture
def chat_runtime(monkeypatch, tmp_path):
    monkeypatch.setitem(main._CONFIG, "claude_code", True)
    monkeypatch.setitem(main._CONFIG, "claude_code_model", "fictional-model")
    monkeypatch.setitem(main._CONFIG, "claude_code_effort", "medium")
    monkeypatch.setitem(main._CONFIG, "claude_code_context_window_tokens", 200000)
    monkeypatch.setitem(main._CONFIG, "claude_code_timeout_seconds", 90)
    monkeypatch.setitem(main._CONFIG, "claude_code_workspace", str(tmp_path / "claude"))
    monkeypatch.setattr(main, "_BACKGROUND_TASKS", set())
    services, records, discoveries = [], [], []
    native_ids = set()

    class Service:
        def __init__(self, scope):
            self._claude_code = True
            self._claude_code_model = main._CONFIG["claude_code_model"]
            self._claude_code_context_window_tokens = main._CONFIG["claude_code_context_window_tokens"]
            self._claude_code_effort = main._CONFIG["claude_code_effort"]
            self._claude_code_timeout_seconds = main._CONFIG["claude_code_timeout_seconds"]
            self.scope = scope
            self.calls = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.release.set()
            self.failure = None
            self.failure_before_create = None
        async def chat(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            # The authoritative binding exists before native work can start.
            native = main._load_turn_state_and_soul_card("chat:atomic-sitting", **self.scope)[0]
            assert native["atomic_native_session_id"] == (kwargs["session_id"] or kwargs["resume_session_id"])
            if self.failure_before_create:
                raise self.failure_before_create
            if kwargs["session_id"]:
                if kwargs["session_id"] in native_ids:
                    raise RuntimeError("Session ID is already in use")
                native_ids.add(kwargs["session_id"])
            self.started.set()
            await kwargs["on_event"]({"type": "assistant", "uuid": "record-reply", "message": {
                "id": "model-message", "content": [{"type": "text", "text": "reply"}]}})
            await self.release.wait()
            await kwargs["on_event"]({"type": "result", "subtype": "success", "is_error": False, "result": "reply"})
            if self.failure:
                raise self.failure
            return "reply"

    def factory(payload, *, operation):
        assert operation == "other_reply" and set(payload) == {"user"}
        service = Service(payload["user"])
        services.append(service)
        return service
    async def discover(url, authorization):
        discoveries.append((url, authorization))
        return ["mcp__atomic__search_atoms"]

    monkeypatch.setattr(main, "_get_service_from_payload", factory)
    monkeypatch.setattr(main, "_record_call", lambda *args, **kwargs: records.append((args, kwargs)))
    monkeypatch.setattr(atomic_chat, "_bridge_tools", discover)
    app = FastAPI()
    atomic_chat.register_atomic_chat_routes(app, runtime=main)
    endpoint = next(route.endpoint for route in app.routes if route.path == "/integration/atomic/chat")
    return SimpleNamespace(app=app, endpoint=endpoint, services=services, records=records,
                           discoveries=discoveries, factory=factory, native_ids=native_ids)


def request(**changes):
    return atomic_chat.AtomicChatRequest(**{
        "user_id": "TestOwner", "soul_id": "Birch", "conversation_id": "sitting", "turn_id": "turn-1",
        "operation": "new", "purpose": "reply", "prompt": "fictional question", "system_prompt": "fictional Presence",
        "bridge_url": "http://127.0.0.1:9999/openalma/chat-tools/mcp",
        "profile_fingerprint": atomic_chat.claude_chat_profile(main._CONFIG)["fingerprint"], **changes,
    })


async def read(response):
    events = [json.loads(line) async for line in response.body_iterator]
    await asyncio.gather(*main._BACKGROUND_TASKS)
    return events


@pytest.mark.asyncio
async def test_new_then_resume_uses_server_binding_and_tool_free_recap(chat_runtime):
    first = await read(await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret"))
    await asyncio.sleep(0)
    native_id = first[0]["session_id"]
    assert first[-1]["terminal"] is True and first[-1]["type"] == "complete"
    assert first[0]["profile"]["model"] == "fictional-model"
    assert "workspace_root" not in first[0]["profile"]
    second = await read(await chat_runtime.endpoint(request(operation="continue", system_prompt=None,
        purpose="recap", bridge_url=None, turn_id="turn-2"), authorization=None))
    await asyncio.sleep(0)
    assert second[0]["session_id"] == native_id
    kwargs = chat_runtime.services[-1].calls[0][1]
    assert kwargs["resume_session_id"] == native_id and kwargs["session_id"] is None
    assert kwargs["system_prompt"] is None
    assert kwargs["cli_tools"] == {"mcp_servers": {}, "allowed_tools": []}
    assert len(chat_runtime.discoveries) == 1
    assert not main._BACKGROUND_TASKS
    assert "fictional-secret" not in repr(chat_runtime.records)


@pytest.mark.asyncio
async def test_same_soul_serialized_other_soul_independent_and_disconnect_does_not_stop_work(chat_runtime, monkeypatch):
    def held_factory(payload, *, operation):
        service = chat_runtime.factory(payload, operation=operation)
        service.release.clear()
        return service
    monkeypatch.setattr(main, "_get_service_from_payload", held_factory)
    response = await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret")
    first = json.loads(await anext(response.body_iterator))
    service = chat_runtime.services[0]
    await service.started.wait()
    with pytest.raises(HTTPException, match="still running"):
        await chat_runtime.endpoint(request(operation="continue", system_prompt=None), authorization="Bearer fictional-secret")
    other = await chat_runtime.endpoint(request(soul_id="Birch-v2"), authorization="Bearer fictional-secret")
    other_first = json.loads(await anext(other.body_iterator))
    assert first["session_id"] != other_first["session_id"]
    await chat_runtime.services[-1].started.wait()
    assert chat_runtime.services[0].calls[0][1]["workspace"] != chat_runtime.services[-1].calls[0][1]["workspace"]
    await response.body_iterator.aclose()
    assert len(main._BACKGROUND_TASKS) == 2 and not service.release.is_set()
    tasks = list(main._BACKGROUND_TASKS)
    for item in chat_runtime.services:
        item.release.set()
    await read(other)
    await asyncio.gather(*tasks)
    await asyncio.sleep(0)
    assert not main._BACKGROUND_TASKS


@pytest.mark.asyncio
async def test_success_terminal_waits_for_wrapper_and_retains_binding_after_failure(chat_runtime, monkeypatch):
    def failed_factory(payload, *, operation):
        service = chat_runtime.factory(payload, operation=operation)
        service.failure = RuntimeError("fictional-secret wrapper failure")
        return service
    monkeypatch.setattr(main, "_get_service_from_payload", failed_factory)
    events = await read(await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret"))
    await asyncio.sleep(0)
    assert [event["type"] for event in events] == ["session", "text", "error"]
    assert events[-1]["error"] == "[redacted] wrapper failure"
    assert not main._BACKGROUND_TASKS
    again = await read(await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret"))
    assert again[0]["session_id"] == events[0]["session_id"]
    assert again[-1]["type"] == "error" and "already in use" in again[-1]["error"]
    assert len(chat_runtime.native_ids) == 1
    assert "fictional-secret" not in repr(chat_runtime.records)


@pytest.mark.asyncio
async def test_failed_initialization_retries_same_id_with_presence(chat_runtime, monkeypatch):
    def failed_once(payload, *, operation):
        service = chat_runtime.factory(payload, operation=operation)
        if len(chat_runtime.services) == 1:
            service.failure_before_create = RuntimeError("fictional login unavailable")
        return service
    monkeypatch.setattr(main, "_get_service_from_payload", failed_once)
    first = await read(await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret"))
    assert first[-1]["type"] == "error" and not chat_runtime.native_ids
    second = await read(await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret"))
    assert second[-1]["type"] == "complete"
    assert first[0]["session_id"] == second[0]["session_id"]
    kwargs = chat_runtime.services[-1].calls[0][1]
    assert kwargs["session_id"] == first[0]["session_id"] and kwargs["system_prompt"] == "fictional Presence"


@pytest.mark.asyncio
async def test_explicit_retry_resumes_without_duplicating_initial_input(chat_runtime):
    await read(await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret"))
    control = "Fictional continuation of a known failed response."
    await read(await chat_runtime.endpoint(request(operation="retry", system_prompt=None, prompt=control),
                                         authorization="Bearer fictional-secret"))
    prompt, kwargs = chat_runtime.services[-1].calls[0]
    assert prompt == control and kwargs["session_id"] is None and kwargs["resume_session_id"] in chat_runtime.native_ids


@pytest.mark.asyncio
async def test_new_recap_does_not_consume_primary_session_slot(chat_runtime):
    with pytest.raises(HTTPException, match="A recap must resume"):
        await chat_runtime.endpoint(request(purpose="recap", bridge_url=None), authorization=None)
    assert chat_runtime.services == []


@pytest.mark.asyncio
async def test_start_and_end_refuse_active_native_work_then_ended_session_stays_closed(chat_runtime, monkeypatch):
    def held_factory(payload, *, operation):
        service = chat_runtime.factory(payload, operation=operation)
        service.release.clear()
        return service
    monkeypatch.setattr(main, "_get_service_from_payload", held_factory)
    response = await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret")
    await anext(response.body_iterator)
    service = chat_runtime.services[0]
    await service.started.wait()
    scope = {"user_id": "TestOwner", "soul_id": "Birch", "conversation_id": "chat:atomic-sitting"}
    start = main.AtomicSessionStartRequest(**scope)
    end = main.AtomicSessionEndRequest(**scope)
    for endpoint, body in ((main.atomic_session_start, start), (main.atomic_session_end, end)):
        with pytest.raises(HTTPException, match="still running"):
            await endpoint(body)
    service.release.set()
    # Leave terminal bytes unread: native completion is not writer-busy status.
    async with asyncio.timeout(2):
        while not chat_runtime.records:
            await asyncio.sleep(0)
    atomic_chat.require_sitting_idle(main, scope["conversation_id"], user_id="TestOwner", soul_id="Birch")
    assert main._BACKGROUND_TASKS
    assert (await main.atomic_session_end(end))["status"] == "ended"
    with pytest.raises(HTTPException, match="has ended"):
        await main.atomic_session_start(start)
    with pytest.raises(HTTPException, match="has ended"):
        await chat_runtime.endpoint(request(operation="continue", system_prompt=None), authorization="Bearer fictional-secret")
    await read(response)


def test_memorize_retry_upgrades_native_columns_before_state_read(chat_runtime, monkeypatch):
    scope = {"user_id": "TestOwner", "soul_id": "Birch"}
    cid = "chat:atomic-old"
    main._write_conversation_state(cid, **scope, updates={"digest_cursor": 8, "last_memorize_at": "2026-10-01T00:00:00Z"})
    path = main._sqlite_current_path(**scope)
    with sqlite3.connect(path) as con:
        con.execute("ALTER TABLE conversations DROP COLUMN atomic_native_session_id")
        con.execute("ALTER TABLE conversations DROP COLUMN atomic_native_workspace")
    monkeypatch.setattr(main, "_sillytavern_source_deleted", lambda *_args: False)
    assert main._memorize_targets_complete("TestOwner", "Birch", {cid: {"cursor": 7}})
    with sqlite3.connect(path) as con:
        assert "atomic_native_session_id" in db.sqlite_table_columns(con, "conversations")


@pytest.mark.asyncio
async def test_captured_service_profile_mismatch_refuses_before_native_work(chat_runtime, monkeypatch):
    def stale(payload, *, operation):
        service = chat_runtime.factory(payload, operation=operation)
        service._claude_code_model = "stale-model"
        return service
    monkeypatch.setattr(main, "_get_service_from_payload", stale)
    with pytest.raises(HTTPException, match="service profile changed"):
        await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret")
    assert not main._BACKGROUND_TASKS and not chat_runtime.services[0].calls


@pytest.mark.asyncio
async def test_preflight_refuses_paused_or_changed_profile_before_work(chat_runtime, monkeypatch):
    changed = request(profile_fingerprint="old")
    with pytest.raises(HTTPException, match="profile changed"):
        await chat_runtime.endpoint(changed, authorization="Bearer fictional-secret")
    def paused(**_scope):
        raise HTTPException(status_code=409, detail={"code": "soul_paused"})
    monkeypatch.setattr(main, "_require_soul_active", paused)
    with pytest.raises(HTTPException) as error:
        await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret")
    assert error.value.detail == {"code": "soul_paused"}
    assert chat_runtime.services == [] and not main._BACKGROUND_TASKS


@pytest.mark.asyncio
async def test_http_validation_does_not_echo_authorization_and_rejects_overrides(chat_runtime):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=chat_runtime.app), base_url="http://test") as client:
        response = await client.post("/integration/atomic/chat", json={**request().model_dump(), "session_id": "foreign"},
                                     headers={"Authorization": "Bearer fictional-secret"})
    assert response.status_code == 422 and "fictional-secret" not in response.text
    assert not main._BACKGROUND_TASKS


@pytest.mark.asyncio
async def test_http_stream_framing_and_registration(chat_runtime):
    assert sum(route.path == "/integration/atomic/chat" for route in main.app.routes) == 1
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=chat_runtime.app), base_url="http://test") as client:
        response = await client.post("/integration/atomic/chat", json=request().model_dump(),
                                     headers={"Authorization": "Bearer fictional-secret"})
    assert response.status_code == 200 and response.headers["content-type"] == "application/x-ndjson"
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events] == ["session", "text", "complete"]
    assert [event["terminal"] for event in events] == [False, False, True]
    await asyncio.gather(*main._BACKGROUND_TASKS)


@pytest.mark.asyncio
async def test_disconnect_before_first_body_does_not_orphan_producer(chat_runtime, monkeypatch):
    def held_factory(payload, *, operation):
        service = chat_runtime.factory(payload, operation=operation)
        service.release.clear()
        return service
    monkeypatch.setattr(main, "_get_service_from_payload", held_factory)
    response = await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret")
    async def send(_message):
        raise OSError("fictional connection closed")
    async def receive():
        return {"type": "http.disconnect"}
    from starlette.requests import ClientDisconnect
    with pytest.raises(ClientDisconnect):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    service = chat_runtime.services[0]
    await service.started.wait()
    assert main._BACKGROUND_TASKS
    service.release.set()
    await asyncio.wait_for(asyncio.gather(*main._BACKGROUND_TASKS), 2)
    await asyncio.sleep(0)
    assert not main._BACKGROUND_TASKS


@pytest.mark.asyncio
async def test_delivery_overflow_is_bounded_without_cancelling_native_work(chat_runtime, monkeypatch):
    monkeypatch.setattr(atomic_chat, "_DELIVERY_BUFFER_BYTES", 1024)
    def held_factory(payload, *, operation):
        service = chat_runtime.factory(payload, operation=operation)
        service.release.clear()
        return service
    monkeypatch.setattr(main, "_get_service_from_payload", held_factory)
    response = await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret")
    await anext(response.body_iterator)
    service = chat_runtime.services[0]
    await service.started.wait()
    await service.calls[0][1]["on_event"]({"type": "assistant", "uuid": "record-overflow", "message": {
        "id": "model-message", "content": [{"type": "text", "text": "x" * 2048}]}})
    event = json.loads(await anext(response.body_iterator))
    assert event["terminal"] and event["code"] == "delivery_overflow"
    assert main._BACKGROUND_TASKS and not service.release.is_set()
    service.release.set()
    await response.body_iterator.aclose()
    await asyncio.wait_for(asyncio.gather(*main._BACKGROUND_TASKS), 2)
    await asyncio.sleep(0)
    assert not main._BACKGROUND_TASKS


@pytest.mark.asyncio
@pytest.mark.parametrize("lag", [False, True])
async def test_event_burst_survives_fast_or_lagging_consumer(chat_runtime, monkeypatch, lag):
    def burst_factory(payload, *, operation):
        service = chat_runtime.factory(payload, operation=operation)
        original = service.chat
        async def burst(prompt, **kwargs):
            for number in range(300):
                await kwargs["on_event"]({"type": "assistant", "uuid": f"record-{number}", "message": {
                    "id": "model-message", "content": [{"type": "text", "text": str(number)}]}})
            return await original(prompt, **kwargs)
        service.chat = burst
        if lag:
            service.release.clear()
        return service
    monkeypatch.setattr(main, "_get_service_from_payload", burst_factory)
    response = await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret")
    first = []
    if lag:
        first.append(json.loads(await anext(response.body_iterator)))
        await chat_runtime.services[0].started.wait()
        chat_runtime.services[0].release.set()
    events = first + await read(response)
    assert len(events) == 303 and events[-1]["type"] == "complete"


def test_normalized_events_preserve_records_calls_results_and_final_usage():
    def message(kind, record, content, parent=None):
        return {"type": kind, "uuid": record, "parent_tool_use_id": parent,
                "message": {"id": "same-model-message", "content": content}}
    raw = [
        message("assistant", "text-1", [{"type": "text", "text": "first"}]),
        message("assistant", "text-2", [{"type": "text", "text": "second"}]),
        message("assistant", "call-record", [{"type": "tool_use", "id": "call-1",
                "name": "mcp__atomic__search_atoms", "input": {"query": "fictional"}}], parent="outer-call"),
        message("user", "result-record", [{"type": "tool_result", "tool_use_id": "call-1",
                "content": [{"type": "text", "text": "full tool output"}], "is_error": True}], parent="outer-call"),
        {"type": "context_usage", "message_id": "same-model-message",
         "usage": {"input_tokens": 2, "cache_read_input_tokens": 30, "output_tokens": 16}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"text": "not duplicated"}}},
        message("assistant", "thought", [{"type": "thinking", "thinking": "private"}]),
    ]
    events = [frame for event in raw for frame in atomic_chat._wire_events(event)]
    assert [frame["type"] for frame in events] == ["text", "text", "tool_call", "tool_result", "usage"]
    assert [frame["record_id"] for frame in events[:2]] == ["text-1", "text-2"]
    assert all(frame["mode"] == "replace" for frame in events[:2])
    assert events[2]["call_id"] == events[3]["call_id"] == "call-1"
    assert events[2]["name"] == "search_atoms" and events[2]["input"] == {"query": "fictional"}
    assert events[3]["content"] == [{"type": "text", "text": "full tool output"}]
    assert events[3]["is_error"] and events[3]["parent_call_id"] == "outer-call"
    assert events[-1]["usage"] == raw[4]["usage"]


@pytest.mark.asyncio
async def test_active_turn_keeps_captured_service_during_reload(chat_runtime, monkeypatch):
    def held_factory(payload, *, operation):
        service = chat_runtime.factory(payload, operation=operation)
        service.release.clear()
        return service
    monkeypatch.setattr(main, "_get_service_from_payload", held_factory)
    response = await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret")
    session = json.loads(await anext(response.body_iterator))
    service = chat_runtime.services[0]
    await service.started.wait()
    monkeypatch.setitem(main._CONFIG, "claude_code_model", "new-fictional-model")
    assert session["profile"]["model"] == service._claude_code_model == "fictional-model"
    service.release.set()
    assert (await read(response))[-1]["type"] == "complete"


@pytest.mark.asyncio
async def test_shutdown_waits_for_native_work_and_terminal_handoff(chat_runtime, monkeypatch):
    def held_factory(payload, *, operation):
        service = chat_runtime.factory(payload, operation=operation)
        service.release.clear()
        return service
    monkeypatch.setattr(main, "_get_service_from_payload", held_factory)
    signals = []
    monkeypatch.setattr(main.signal, "raise_signal", signals.append)
    monkeypatch.setattr(main, "_ACTIVE_WORK_REQUESTS", 0)
    monkeypatch.setattr(main, "_SHUTDOWN_STATE", dict(main._SHUTDOWN_STATE))
    response = await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret")
    await anext(response.body_iterator)
    service = chat_runtime.services[0]
    await service.started.wait()
    shutdown = asyncio.create_task(main._shutdown_when_idle(max_wait_sec=0))
    await asyncio.sleep(0.02)
    assert signals == []
    service.release.set()
    await asyncio.sleep(0.02)
    assert signals == [] and main._BACKGROUND_TASKS
    await read(response)
    await asyncio.wait_for(shutdown, 2)
    assert signals == [main.signal.SIGINT]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["search_atoms", "invalid*"])
async def test_installed_mcp_client_discovers_bridge_tools_without_network(monkeypatch, tool_name):
    requests = []
    def handler(req):
        assert req.headers["authorization"] == "Bearer fictional-secret"
        message = json.loads(req.content)
        requests.append(message["method"])
        if message["method"] == "notifications/initialized":
            return httpx.Response(202)
        if message["method"] == "initialize":
            result = {"protocolVersion": message["params"]["protocolVersion"],
                      "capabilities": {"tools": {}}, "serverInfo": {"name": "fictional", "version": "1"}}
        else:
            assert message["method"] == "tools/list"
            result = {"tools": [{"name": tool_name, "inputSchema": {"type": "object"}}]}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})
    real_client = httpx.AsyncClient
    monkeypatch.setattr(atomic_chat.httpx, "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    url = "http://127.0.0.1:9999/openalma/chat-tools/mcp"
    if tool_name == "invalid*":
        with pytest.raises(HTTPException, match="invalid tool name"):
            await atomic_chat._bridge_tools(url, "Bearer fictional-secret")
    else:
        assert await atomic_chat._bridge_tools(url, "Bearer fictional-secret") == ["mcp__atomic__search_atoms"]
    assert requests == ["initialize", "notifications/initialized", "tools/list"]


@pytest.mark.parametrize("capacity", [None, 20000, True])
def test_profile_requires_explicit_usable_capacity(chat_runtime, monkeypatch, capacity):
    monkeypatch.setitem(main._CONFIG, "claude_code_context_window_tokens", capacity)
    with pytest.raises(HTTPException, match="claude_code_context_window_tokens"):
        atomic_chat.claude_chat_profile(main._CONFIG)


@pytest.mark.asyncio
async def test_binding_scope_and_workspace_changes_refused(chat_runtime, monkeypatch, tmp_path):
    await read(await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret"))
    await asyncio.sleep(0)
    with pytest.raises(HTTPException, match="no native session"):
        await chat_runtime.endpoint(request(soul_id="Cedar", operation="continue", system_prompt=None),
                                    authorization="Bearer fictional-secret")
    with pytest.raises(HTTPException, match="no native session"):
        await chat_runtime.endpoint(request(conversation_id="different", operation="continue", system_prompt=None),
                                    authorization="Bearer fictional-secret")
    monkeypatch.setitem(main._CONFIG, "claude_code_workspace", str(tmp_path / "different-root"))
    with pytest.raises(HTTPException, match="workspace changed"):
        await chat_runtime.endpoint(request(operation="continue", system_prompt=None), authorization="Bearer fictional-secret")


def test_schema_upgrades_existing_conversation_without_replacing_it():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE conversations (conversation_id TEXT PRIMARY KEY, user_id TEXT, soul_id TEXT)")
    con.execute("INSERT INTO conversations VALUES ('kept', 'TestOwner', 'Birch')")
    db.sqlite_ensure_conversation_state_schema(con)
    assert con.execute("SELECT conversation_id FROM conversations").fetchall() == [("kept",)]
    assert {"atomic_native_session_id", "atomic_native_workspace"} <= set(db.sqlite_table_columns(con, "conversations"))
    con.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["https://example.test/openalma/chat-tools/mcp", "http://localhost/api/search",
                                  "http://token@localhost/openalma/chat-tools/mcp",
                                  "http://:fictional-secret@localhost/openalma/chat-tools/mcp"])
async def test_tool_discovery_refuses_non_bridge_urls(url):
    with pytest.raises(HTTPException) as error:
        await atomic_chat._bridge_tools(url, "Bearer fictional-secret")
    assert error.value.status_code == 422
