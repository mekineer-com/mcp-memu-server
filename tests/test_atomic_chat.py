import asyncio
import json
import os
import sqlite3
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from app import db, main
from app.services import atomic_chat, state


@pytest.fixture
def chat_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "native-home"))
    monkeypatch.setitem(main._CONFIG, "claude_code", True)
    monkeypatch.setitem(main._CONFIG, "claude_code_model", "fictional-model")
    monkeypatch.setitem(main._CONFIG, "claude_code_effort", "medium")
    monkeypatch.setitem(main._CONFIG, "claude_code_context_window_tokens", 200000)
    monkeypatch.setitem(main._CONFIG, "claude_code_timeout_seconds", 90)
    monkeypatch.setitem(main._CONFIG, "claude_code_workspace", str(tmp_path / "claude"))
    monkeypatch.setattr(main, "_BACKGROUND_TASKS", set())
    services, records, discoveries = [], [], []
    native_ids = set()
    original_idle = atomic_chat._writer_idle
    def fake_idle(receipt):
        if receipt.get("process", {}).get("pid") == os.getpid():
            return bool(receipt.get("parent_reaped"))
        return original_idle(receipt)
    monkeypatch.setattr(atomic_chat, "_writer_idle", fake_idle)

    class Service:
        def __init__(self, scope):
            self._claude_code = True
            self._claude_code_model = main._CONFIG["claude_code_model"]
            self._claude_code_context_window_tokens = main._CONFIG["claude_code_context_window_tokens"]
            self._claude_code_effort = main._CONFIG["claude_code_effort"]
            self._claude_code_timeout_seconds = main._CONFIG["claude_code_timeout_seconds"]
            self.scope = scope
            self.calls = []
            self.inputs = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.release.set()
            self.failure = None
            self.error_result = None
            self.failure_before_create = None
            self.usage = None
            self.result = "reply"
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
                path = atomic_chat.claude_session_path(kwargs["workspace"], kwargs["session_id"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fictional native transcript\n")
            await kwargs["on_event"]({"type":"process", "state":"started", "pid":os.getpid()})
            try:
                await kwargs["on_event"]({"type":"process", "state":"input_started", "pid":os.getpid()})
            except BaseException:
                await kwargs["on_event"]({"type":"process", "state":"reaped", "pid":os.getpid()})
                raise
            self.inputs.append(prompt)
            self.started.set()
            message_id = self.usage["message_id"] if self.usage else "model-message"
            await kwargs["on_event"]({"type": "context_start", "message_id": message_id})
            await kwargs["on_event"]({"type": "assistant", "uuid": "record-reply", "message": {
                "id": message_id, "content": [{"type": "text", "text": "reply"}]}})
            await self.release.wait()
            if self.usage:
                await kwargs["on_event"]({"type":"context_usage", **self.usage})
            await kwargs["on_event"]({"type":"process", "state":"reaped", "pid":os.getpid()})
            await kwargs["on_event"](self.error_result or {"type": "result", "subtype": "success", "is_error": False, "result": self.result})
            if self.error_result:
                raise RuntimeError("fictional recorded Claude failure")
            if self.failure:
                raise self.failure
            return self.result

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
        "attempt_id": str(uuid.uuid4()),
        "operation": "new", "purpose": "reply", "prompt": "fictional question", "system_prompt": "fictional Presence",
        "bridge_url": "http://127.0.0.1:9999/openalma/chat-tools/mcp",
        "profile_fingerprint": atomic_chat.claude_chat_profile(main._CONFIG)["fingerprint"],
        "prompt_tokens": 100, "overhead_tokens": 50,
        "system_tokens": 0 if changes.get("system_prompt", "presence") is None else 25, **changes,
    })


async def read(response):
    events = [json.loads(line) async for line in response.body_iterator]
    await asyncio.gather(*main._BACKGROUND_TASKS)
    return events


def test_context_coverage_rebases_at_input_boundary_not_usage_arrival():
    receipt = {"context": {"estimated_tokens": 200, "overhead_tokens": 50, "system_tokens": 25}}
    context = atomic_chat._Context(receipt, 60)
    context.observe({"type":"context_start", "message_id":"m1"})
    result = {"type":"tool_result", "tool_use_id":"call-1", "content":"native tool output"}
    user = {"type":"user", "uuid":"result", "message":{"content":[result]}}
    assistant = {"type":"assistant", "uuid":"assistant", "message":{
        "id":"m1", "content":[{"type":"text", "text":"provisional answer"}]}}
    context.observe(user)
    context.observe(assistant)
    first_output = receipt["context"]["estimated_tokens"]
    context.observe({**assistant, "uuid":"second-record"})
    assert receipt["context"]["estimated_tokens"] == first_output + atomic_chat._block_tokens(assistant["message"]["content"][0])
    context.observe({**assistant, "uuid":"second-record"})
    assert receipt["context"]["estimated_tokens"] == first_output + atomic_chat._block_tokens(assistant["message"]["content"][0])
    usage = {"input_tokens":10, "cache_creation_input_tokens":20, "cache_read_input_tokens":30, "output_tokens":5}
    context.observe({"type":"context_usage", "message_id":"m1", "usage":usage})
    expected = 65 + atomic_chat._block_tokens(result)
    assert receipt["context"] == {"estimated_tokens":expected, "overhead_tokens":60, "system_tokens":25}
    context.observe(assistant)  # A late full snapshot is already covered by output_tokens.
    context.observe(user)  # Replacement/duplicate result, not another model-facing message.
    context.observe({**user, "parent_tool_use_id":"nested"})
    assert receipt["context"]["estimated_tokens"] == expected
    context.observe({"type":"context_start", "message_id":"m2"})
    context.observe({**assistant, "message":{**assistant["message"], "id":"m2"}})
    growing = receipt["context"]["estimated_tokens"]
    assert growing > expected
    context.observe({"type":"context_start", "message_id":"m3"})
    assert receipt["context"]["estimated_tokens"] == growing  # m2 never had valid usage.
    context.observe({"type":"context_usage", "message_id":"m3", "usage":usage})
    assert receipt["context"]["estimated_tokens"] == 65


def test_context_unknown_coverage_is_not_zero_and_requires_new_measurement():
    receipt = {"context": None}
    assert atomic_chat._input_context(None, 100, 50) is None
    context = atomic_chat._Context(receipt, 50)
    context.observe({"type":"context_start", "message_id":"m1"})
    context.observe({"type":"user", "uuid":"opaque", "message":{"content":[{
        "type":"tool_result", "tool_use_id":"call", "content":[{"type":"image", "source":{}}]}]}})
    usage = {"input_tokens":10, "cache_creation_input_tokens":0, "cache_read_input_tokens":0, "output_tokens":5}
    context.observe({"type":"context_usage", "message_id":"m1", "usage":usage})
    assert receipt["context"] is None  # These post-input results are not covered yet.
    context.observe({"type":"context_start", "message_id":"m2"})
    context.observe({"type":"context_usage", "message_id":"m2", "usage":usage})
    assert receipt["context"]["estimated_tokens"] == 15
    with pytest.raises(RuntimeError, match="coverage boundary"):
        context.observe({"type":"context_usage", "message_id":"wrong", "usage":usage})


def test_context_overhead_growth_is_counted_once_without_reinjecting_presence():
    original = {"estimated_tokens":1000, "overhead_tokens":100, "system_tokens":25}
    same = atomic_chat._input_context(original, 10, 100)
    assert original["estimated_tokens"] == 1000  # Input checkpoint prepares a copy.
    assert same["estimated_tokens"] == 1010
    larger = atomic_chat._input_context(same, 20, 150)
    assert larger["estimated_tokens"] == 1080
    smaller = atomic_chat._input_context(larger, 10, 50)
    assert atomic_chat._input_context(smaller, 10, 150)["estimated_tokens"] == 1100


@pytest.mark.asyncio
async def test_new_then_resume_uses_server_binding_and_tool_free_recap(chat_runtime):
    first = await read(await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret"))
    await asyncio.sleep(0)
    native_id = first[0]["session_id"]
    assert first[-1]["terminal"] is True and first[-1]["type"] == "complete"
    assert first[0]["profile"]["model"] == "fictional-model"
    assert "workspace_root" not in first[0]["profile"]
    scope = {"user_id":"TestOwner", "soul_id":"Birch"}
    first_receipt = atomic_chat._native_status(main, "chat:atomic-sitting", scope)["turn"]
    assert "usage" not in first_receipt  # An estimate is not fabricated provider usage.
    first_context = first_receipt["context"]["estimated_tokens"]
    assert first_context > 150
    second = await read(await chat_runtime.endpoint(request(operation="continue", system_prompt=None,
        purpose="recap", bridge_url=None, turn_id="turn-2"), authorization=None))
    await asyncio.sleep(0)
    assert second[0]["session_id"] == native_id
    second_receipt = atomic_chat._native_status(main, "chat:atomic-sitting", scope)["turn"]
    assert second_receipt["context"]["estimated_tokens"] == first_context + 100 + atomic_chat._block_tokens({"type":"text", "text":"reply"})
    assert second_receipt["context"]["overhead_tokens"] == 50
    kwargs = chat_runtime.services[-1].calls[0][1]
    assert kwargs["resume_session_id"] == native_id and kwargs["session_id"] is None
    assert kwargs["system_prompt"] is None
    assert kwargs["cli_tools"] == {"mcp_servers": {}, "allowed_tools": []}
    assert len(chat_runtime.discoveries) == 1
    assert not main._BACKGROUND_TASKS
    assert "fictional-secret" not in repr(chat_runtime.records)


@pytest.mark.asyncio
async def test_empty_recap_is_failed_and_retries_same_native_session(chat_runtime, monkeypatch):
    await read(await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret"))
    def blank_once(payload, *, operation):
        service = chat_runtime.factory(payload, operation=operation)
        if len(chat_runtime.services) == 2:
            service.result = " \n\t"
        return service
    monkeypatch.setattr(main, "_get_service_from_payload", blank_once)
    recap = request(operation="continue", system_prompt=None, bridge_url=None,
                    purpose="recap", turn_id="recap-1")
    events = await read(await chat_runtime.endpoint(recap, authorization=None))
    scope = {"user_id":"TestOwner", "soul_id":"Birch"}
    saved = atomic_chat._native_status(main, "chat:atomic-sitting", scope)
    assert events[-1]["type"] == "error" and "empty" in events[-1]["error"]
    assert saved["turn"]["status"] == "failed" and saved["writer_idle"] is True
    assert saved["turn"]["submission"] == "recorded"
    retried = await read(await chat_runtime.endpoint(request(operation="retry", system_prompt=None,
        bridge_url=None, purpose="recap", turn_id="recap-1"), authorization=None))
    assert retried[-1]["type"] == "complete"
    assert retried[0]["session_id"] == saved["session_id"]
    assert retried[0]["attempt_id"] != recap.attempt_id


@pytest.mark.asyncio
async def test_retry_discovery_exposes_old_attempt_but_cannot_checkpoint_over_new_one(chat_runtime, monkeypatch):
    def error_once(payload, *, operation):
        service = chat_runtime.factory(payload, operation=operation)
        if len(chat_runtime.services) == 1:
            service.error_result = {"type":"result", "subtype":"error_max_turns", "is_error":True}
        return service
    monkeypatch.setattr(main, "_get_service_from_payload", error_once)
    await read(await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret"))
    scope = {"user_id":"TestOwner", "soul_id":"Birch"}
    cid = "chat:atomic-sitting"
    old = atomic_chat._native_status(main, cid, scope)["turn"]
    entered, release = asyncio.Event(), asyncio.Event()
    async def delayed_discovery(*_args):
        entered.set()
        await release.wait()
        return ["mcp__atomic__search_atoms"]
    monkeypatch.setattr(atomic_chat, "_bridge_tools", delayed_discovery)
    retry = request(operation="retry", system_prompt=None)
    pending = asyncio.create_task(chat_runtime.endpoint(retry, authorization="Bearer fictional-secret"))
    await entered.wait()
    status = atomic_chat._native_status(main, cid, scope)
    assert status["writer_idle"] is True and status["turn"]["status"] == "failed"
    assert status["turn"]["turn_id"] == retry.turn_id
    assert status["turn"]["attempt_id"] != retry.attempt_id
    release.set()
    events = await read(await pending)
    assert events[0]["attempt_id"] == retry.attempt_id
    assert events[-1]["type"] == "complete"
    with pytest.raises(RuntimeError, match="binding changed"):
        atomic_chat._save_receipt(main, cid, scope, old, main._sqlite_current_path(**scope))
    assert atomic_chat._native_status(main, cid, scope)["turn"]["attempt_id"] == retry.attempt_id
    calls = sum(len(service.calls) for service in chat_runtime.services)
    with pytest.raises(HTTPException, match="attempt already exists"):
        await chat_runtime.endpoint(retry, authorization="Bearer fictional-secret")
    assert sum(len(service.calls) for service in chat_runtime.services) == calls


@pytest.mark.asyncio
async def test_status_recovers_exact_completed_turn_without_another_call(chat_runtime, monkeypatch):
    def with_usage(payload, *, operation):
        service = chat_runtime.factory(payload, operation=operation)
        service.usage = {"message_id":"primary", "usage":{"input_tokens":10, "output_tokens":5,
            "cache_creation_input_tokens":0, "cache_read_input_tokens":0}}
        return service
    monkeypatch.setattr(main, "_get_service_from_payload", with_usage)
    await read(await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret"))
    count = sum(len(service.calls) for service in chat_runtime.services)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=chat_runtime.app), base_url="http://test") as client:
        params = {"user_id":"TestOwner", "soul_id":"Birch", "conversation_id":"sitting", "turn_id":"turn-1"}
        response = await client.get("/integration/atomic/chat/status", params=params)
        assert response.status_code == 200
        saved = response.json()
        assert saved["turn"]["text"] == "reply" and saved["turn"]["status"] == "complete"
        assert saved["writer_idle"] is True
        assert saved["turn"]["writer_exited"] is True
        assert saved["turn"]["usage"] == chat_runtime.services[0].usage
        def paused(**_scope):
            raise HTTPException(status_code=409, detail={"code":"soul_paused"})
        with monkeypatch.context() as pause:
            pause.setattr(main, "_require_soul_active", paused)
            assert (await client.get("/integration/atomic/chat/status", params=params)).status_code == 200
        assert (await client.get("/integration/atomic/chat/status", params={**params, "turn_id":"wrong"})).status_code == 409
        foreign = (await client.get("/integration/atomic/chat/status", params={**params, "soul_id":"Cedar"})).json()
        assert foreign["turn"] is None and foreign["session_id"] is None
    with pytest.raises(HTTPException, match="complete"):
        await chat_runtime.endpoint(request(operation="retry", system_prompt=None), authorization="Bearer fictional-secret")
    binding = atomic_chat._native_state(main, "chat:atomic-sitting", {"user_id":"TestOwner", "soul_id":"Birch"})
    atomic_chat.claude_session_path(binding["atomic_native_workspace"], binding["atomic_native_session_id"]).unlink()
    with pytest.raises(HTTPException, match="already initialized"):
        await chat_runtime.endpoint(request(turn_id="turn-2"), authorization="Bearer fictional-secret")
    assert sum(len(service.calls) for service in chat_runtime.services) == count


@pytest.mark.asyncio
@pytest.mark.parametrize("writer_idle", [True, False, None])
async def test_status_missing_turn_preserves_writer_evidence(chat_runtime, monkeypatch, writer_idle):
    await read(await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret"))
    original = atomic_chat._native_status
    def status_with_exit_evidence(*args):
        return {**original(*args), "writer_idle": writer_idle}
    monkeypatch.setattr(atomic_chat, "_native_status", status_with_exit_evidence)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=chat_runtime.app), base_url="http://test") as client:
        response = await client.get("/integration/atomic/chat/status", params={
            "user_id":"TestOwner", "soul_id":"Birch", "conversation_id":"sitting", "turn_id":"refused-next-turn",
        })
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "atomic_native_turn_missing"
    assert detail["turn"] is None and detail["writer_idle"] is writer_idle
    assert detail["session_id"] == chat_runtime.services[0].calls[0][1]["session_id"]
    assert detail["context"]["estimated_tokens"] > 0


@pytest.mark.asyncio
async def test_writer_exit_is_latched_before_pid_can_be_reused(chat_runtime, monkeypatch):
    await read(await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret"))
    scope = {"user_id":"TestOwner", "soul_id":"Birch"}
    cid = "chat:atomic-sitting"
    receipt = atomic_chat._native_state(main, cid, scope)["atomic_native_turn"]
    receipt["writer_exited"] = False
    atomic_chat._save_receipt(main, cid, scope, receipt, main._sqlite_current_path(**scope))
    reused = False
    def proof(saved):
        return True if saved.get("writer_exited") or not reused else None
    monkeypatch.setattr(atomic_chat, "_writer_idle", proof)
    assert atomic_chat._native_status(main, cid, scope)["writer_idle"] is True
    reused = True
    atomic_chat.require_sitting_idle(main, cid, **scope)
    assert atomic_chat._native_state(main, cid, scope)["atomic_native_turn"]["writer_exited"] is True


@pytest.mark.asyncio
async def test_failed_input_checkpoint_keeps_turn_unsubmitted(chat_runtime, monkeypatch):
    save = atomic_chat._save_receipt
    failed = False
    def checkpoint(runtime, cid, scope, receipt, path):
        nonlocal failed
        if receipt["submission"] == "unknown" and not failed:
            failed = True
            raise RuntimeError("fictional checkpoint failure")
        return save(runtime, cid, scope, receipt, path)
    monkeypatch.setattr(atomic_chat, "_save_receipt", checkpoint)
    events = await read(await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret"))
    assert events[-1]["type"] == "error" and failed
    assert chat_runtime.services[0].inputs == []
    saved = atomic_chat._native_state(main, "chat:atomic-sitting", {"user_id":"TestOwner", "soul_id":"Birch"})["atomic_native_turn"]
    assert saved["status"] == "failed" and saved["submission"] == "not_submitted"
    assert saved["initialized"] is True
    assert saved["context"]["estimated_tokens"] == 25  # Presence may be installed before stdin.
    await read(await chat_runtime.endpoint(request(operation="retry", system_prompt=None), authorization="Bearer fictional-secret"))
    resumed = atomic_chat._native_status(main, "chat:atomic-sitting", {"user_id":"TestOwner", "soul_id":"Birch"})
    assert resumed["context"]["estimated_tokens"] == 175 + atomic_chat._block_tokens({"type":"text", "text":"reply"})


@pytest.mark.asyncio
async def test_restart_evidence_is_not_completion_or_permission_to_replay(chat_runtime, monkeypatch):
    await read(await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret"))
    scope = {"user_id":"TestOwner", "soul_id":"Birch"}
    cid = "chat:atomic-sitting"
    saved = atomic_chat._native_state(main, cid, scope)["atomic_native_turn"]
    saved.update(status="running", writer_exited=False)
    saved.pop("text")
    saved.pop("native_result")
    atomic_chat._save_receipt(main, cid, scope, saved, main._sqlite_current_path(**scope))
    assert not main._BACKGROUND_TASKS
    monkeypatch.setattr(atomic_chat, "_writer_idle", lambda _receipt: None)
    assert atomic_chat._native_status(main, cid, scope)["turn"]["status"] == "uncertain"
    assert atomic_chat._native_status(main, cid, scope)["context"] is None
    with pytest.raises(HTTPException, match="exit is not established"):
        atomic_chat.require_sitting_idle(main, cid, **scope)
    monkeypatch.setattr(atomic_chat, "_writer_idle", lambda _receipt: True)
    atomic_chat.require_sitting_idle(main, cid, **scope)
    with pytest.raises(HTTPException, match="outcome is unknown"):
        await chat_runtime.endpoint(request(operation="retry", system_prompt=None), authorization="Bearer fictional-secret")
    assert sum(len(service.calls) for service in chat_runtime.services) == 1


def test_writer_exit_checks_identity_and_process_group(monkeypatch):
    receipt = {"process":{"pid":12345, "created_at":10}}
    monkeypatch.setattr(atomic_chat.psutil, "Process", lambda _pid: SimpleNamespace(create_time=lambda:10))
    assert atomic_chat._writer_idle(receipt) is False
    monkeypatch.setattr(atomic_chat.psutil, "Process", lambda _pid: SimpleNamespace(create_time=lambda:11))
    if os.name == "posix":
        def absent(_pid, signal):
            assert signal == 0
            raise ProcessLookupError
        monkeypatch.setattr(atomic_chat.os, "killpg", absent)
        assert atomic_chat._writer_idle(receipt) is True
        monkeypatch.setattr(atomic_chat.os, "killpg", lambda *_args:None)
        assert atomic_chat._writer_idle(receipt) is None
        assert atomic_chat._writer_idle({**receipt, "parent_reaped":True}) is None
    assert atomic_chat._writer_idle({}) is None
    assert atomic_chat._writer_idle({**receipt, "writer_exited":True}) is True


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
    receipt = atomic_chat._native_state(main, "chat:atomic-sitting", {"user_id":"TestOwner", "soul_id":"Birch"})["atomic_native_turn"]
    assert receipt["native_result"]["result"] == "reply"
    assert receipt["status"] == "uncertain", "native text is evidence, not proof of wrapper success"
    with pytest.raises(HTTPException, match="outcome is unknown"):
        await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret")
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
    before_retry = atomic_chat._native_status(main, "chat:atomic-sitting", {"user_id":"TestOwner", "soul_id":"Birch"})
    assert before_retry["context"]["estimated_tokens"] == 25
    second = await read(await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret"))
    assert second[-1]["type"] == "complete"
    assert first[0]["session_id"] == second[0]["session_id"]
    kwargs = chat_runtime.services[-1].calls[0][1]
    assert kwargs["session_id"] == first[0]["session_id"] and kwargs["system_prompt"] == "fictional Presence"
    after_retry = atomic_chat._native_status(main, "chat:atomic-sitting", {"user_id":"TestOwner", "soul_id":"Birch"})
    assert after_retry["context"]["estimated_tokens"] == 175 + atomic_chat._block_tokens({"type":"text", "text":"reply"})


@pytest.mark.asyncio
async def test_uncertain_submission_only_admits_same_session_recap(chat_runtime, monkeypatch):
    def failed_once(payload, *, operation):
        service = chat_runtime.factory(payload, operation=operation)
        if len(chat_runtime.services) == 1:
            service.failure = RuntimeError("fictional failed turn")
        return service
    monkeypatch.setattr(main, "_get_service_from_payload", failed_once)
    await read(await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret"))
    control = "Fictional continuation of a known failed response."
    with pytest.raises(HTTPException, match="outcome is unknown"):
        await chat_runtime.endpoint(request(operation="retry", system_prompt=None, prompt=control),
                                    authorization="Bearer fictional-secret")
    await read(await chat_runtime.endpoint(request(operation="continue", system_prompt=None,
        purpose="recap", turn_id="recap-1"), authorization=None))
    saved = atomic_chat._native_state(main, "chat:atomic-sitting", {"user_id":"TestOwner", "soul_id":"Birch"})["atomic_native_turn"]
    assert saved["uncertain_turn"]["turn_id"] == "turn-1"
    assert saved["uncertain_turn"]["status"] == "uncertain"
    resumed = chat_runtime.services[-1].calls[0][1]
    assert resumed["resume_session_id"] == saved["session_id"] and resumed["session_id"] is None
    assert resumed["cli_tools"] == {"mcp_servers":{}, "allowed_tools":[]}
    assert sum(len(service.calls) for service in chat_runtime.services) == 2
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=chat_runtime.app), base_url="http://test") as client:
        earlier = await client.get("/integration/atomic/chat/status", params={"user_id":"TestOwner", "soul_id":"Birch", "conversation_id":"sitting", "turn_id":"turn-1"})
        assert earlier.status_code == 200 and earlier.json()["turn"]["status"] == "uncertain"
    with pytest.raises(HTTPException, match="ordinary outcome remains unknown"):
        await chat_runtime.endpoint(request(operation="retry", system_prompt=None), authorization="Bearer fictional-secret")
    assert sum(len(service.calls) for service in chat_runtime.services) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("delayed_exit", [False, True])
@pytest.mark.parametrize("api_error", [False, True])
async def test_recorded_claude_error_is_retryable_without_reinitialization(chat_runtime, monkeypatch, delayed_exit, api_error):
    idle = atomic_chat._writer_idle
    exit_known = not delayed_exit
    monkeypatch.setattr(atomic_chat, "_writer_idle", lambda receipt: idle(receipt) if exit_known else None)
    def failed_once(payload, *, operation):
        service = chat_runtime.factory(payload, operation=operation)
        if len(chat_runtime.services) == 1:
            service.error_result = ({"type":"result", "subtype":"success", "is_error":True,
                                     "api_error_status":429, "result":"fictional rate limit"} if api_error else
                                    {"type":"result", "subtype":"error_max_turns", "is_error":True, "errors":["fictional round limit"]})
        elif len(chat_runtime.services) in {2, 4}:
            service.failure_before_create = RuntimeError("fictional launch refusal")
        return service
    monkeypatch.setattr(main, "_get_service_from_payload", failed_once)
    await read(await chat_runtime.endpoint(request(), authorization="Bearer fictional-secret"))
    if delayed_exit:
        saved = atomic_chat._native_state(main, "chat:atomic-sitting", {"user_id":"TestOwner", "soul_id":"Birch"})["atomic_native_turn"]
        assert saved["status"] == "uncertain"
        exit_known = True
        assert atomic_chat._native_status(main, "chat:atomic-sitting", {"user_id":"TestOwner", "soul_id":"Birch"})["turn"]["status"] == "failed"
    saved = atomic_chat._native_state(main, "chat:atomic-sitting", {"user_id":"TestOwner", "soul_id":"Birch"})["atomic_native_turn"]
    assert saved["status"] == "failed" and saved["submission"] == "recorded"
    assert saved["input_recorded"] is True
    original_context = saved["context"]["estimated_tokens"]
    await read(await chat_runtime.endpoint(request(operation="retry", system_prompt=None, prompt="Continue the recorded failed turn"), authorization="Bearer fictional-secret"))
    scope = {"user_id":"TestOwner", "soul_id":"Birch"}
    interrupted = atomic_chat._native_state(main, "chat:atomic-sitting", scope)["atomic_native_turn"]
    atomic_chat._save_receipt(main, "chat:atomic-sitting", scope,
        {**interrupted, "status":"prepared"}, main._sqlite_current_path(**scope))
    refused = atomic_chat._native_status(main, "chat:atomic-sitting", {"user_id":"TestOwner", "soul_id":"Birch"})["turn"]
    assert refused["submission"] == "not_submitted" and refused["input_recorded"] is True
    assert refused["error"].startswith("This attempt was not submitted.")
    assert refused["context"]["estimated_tokens"] == original_context
    await read(await chat_runtime.endpoint(request(operation="retry", system_prompt=None, prompt="Continue the recorded failed turn"), authorization="Bearer fictional-secret"))
    assert chat_runtime.services[-1].calls[0][1]["resume_session_id"] == saved["session_id"]
    after_retry = atomic_chat._native_status(main, "chat:atomic-sitting", scope)["context"]["estimated_tokens"]
    assert after_retry == original_context + 100 + atomic_chat._block_tokens({"type":"text", "text":"reply"})
    await read(await chat_runtime.endpoint(request(operation="continue", system_prompt=None,
        turn_id="turn-2", prompt="A different logical turn"), authorization="Bearer fictional-secret"))
    other = atomic_chat._native_status(main, "chat:atomic-sitting", {"user_id":"TestOwner", "soul_id":"Birch"})["turn"]
    assert other["submission"] == "not_submitted" and other["input_recorded"] is False


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


@pytest.mark.parametrize("kind", ["assistant", "user"])
def test_normalized_plain_string_content(kind):
    event = {"type": kind, "uuid": "plain-record", "message": {"id": "model-message", "content": "plain text"}}
    frames = atomic_chat._wire_events(event)
    if kind == "assistant":
        assert frames == [{"type": "text", "record_id": "plain-record", "message_id": "model-message",
                           "parent_call_id": None, "block_index": 0, "mode": "replace", "text": "plain text"}]
    else:
        assert frames == []  # Atomic already owns the user turn.


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
