from __future__ import annotations

import asyncio
import copy
import inspect
import io
import json
from concurrent.futures import CancelledError as FutureCancelledError, ThreadPoolExecutor
from datetime import UTC, datetime
from types import SimpleNamespace
from threading import Event
from typing import Any, Callable

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.services import mentra_routes
from app.services.mentra_routes import register_mentra_routes


AUTH = {"Authorization": "Bearer test-secret"}
START = {
    "user_id": "Fictional User",
    "soul_id": "Codexia",
    "device_session_id": "phone-1",
    "mode": "continuous",
}


@pytest.fixture(autouse=True)
def clear_leases() -> None:
    mentra_routes._leases.clear()
    mentra_routes._start_claims.clear()
    yield
    mentra_routes._leases.clear()
    mentra_routes._start_claims.clear()


def _configured() -> dict[str, Any]:
    return {
        "mentra": {
            "enabled": True,
            "integration_bearer_token": "test-secret",
            "gemini_api_key": "permanent-secret",
            "model": "gemini-2.5-flash-native-audio-preview-12-2025",
            "voice": "Kore",
        }
    }


def _session_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    token_results: list[str | BaseException] | None = None,
    state_results: list[Exception | None] | None = None,
    raise_server_exceptions: bool = True,
    prepare_auto_memorize: Callable[..., tuple[int, dict[str, Any] | None]] | None = None,
    schedule_auto_memorize: Callable[..., str] | None = None,
    retrieve_result: dict[str, Any] | BaseException | None = None,
    retrieve_hook: Callable[[str, dict[str, Any]], Any] | None = None,
) -> tuple[TestClient, dict[str, Any], dict[str, Any]]:
    config = _configured()
    calls: dict[str, Any] = {
        "service": 0,
        "token": [],
        "state": [],
        "cross": [],
        "state_writes": [],
        "memorize_checks": [],
        "recalls": [],
        "route_telemetry": [],
    }
    results = iter(token_results or ["ephemeral-1", "ephemeral-2", "ephemeral-3"])
    writes = iter(state_results or [])
    sitting_ids = iter(("sitting-1", "sitting-2", "sitting-3", "sitting-4"))
    soul_lock = asyncio.Lock()

    class Service:
        async def ensure_dossier_anchors(self, scope: dict[str, str]) -> dict[str, Any]:
            calls["anchors"] = dict(scope)
            return {
                "soul": SimpleNamespace(summary="I am Codexia.", description="soul"),
                "user": SimpleNamespace(summary="The user likes careful work.", description="user"),
            }

    def get_service(scope: dict[str, str]) -> Service:
        calls["service"] += 1
        calls["service_scope"] = dict(scope)
        return Service()

    def load_state(conversation_id: str, **scope: str) -> tuple[dict[str, Any], str, None]:
        calls["state"].append((conversation_id, scope))
        return {}, "I value continuity.", None

    def load_cross(**kwargs: str) -> str:
        calls["cross"].append(kwargs)
        return "[dm][Fictional Friend]\nFictional Friend: hello"

    def load_current(**kwargs: str) -> list[dict[str, Any]]:
        calls["current"] = kwargs
        return [
            {
                "role": "user",
                "content": "A fictional current-chat line.",
                "source_conversation_index": 7,
            }
        ]

    async def retrieve(conversation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls["recalls"].append((conversation_id, copy.deepcopy(payload)))
        calls["lease_locked_during_retrieve"] = mentra_routes._lease_lock.locked()
        calls["soul_locked_during_retrieve"] = soul_lock.locked()
        if retrieve_hook is not None:
            return await retrieve_hook(conversation_id, payload)
        if isinstance(retrieve_result, BaseException):
            raise retrieve_result
        return retrieve_result or {
            "ok": True,
            "retrieve_ms": 123,
            "result": {
                "categories": [{"name": "Fictional Places", "summary": "A remembered coast."}],
                "items": [
                    {
                        "id": "memory-secret-id",
                        "memory_type": "profile",
                        "summary": "The lighthouse keeper preferred cedar tea.",
                    }
                ],
                "resources": [],
            },
        }

    async def mint(**kwargs: str) -> str:
        calls["token"].append(kwargs)
        result = next(results)
        if isinstance(result, BaseException):
            raise result
        return result

    def write_state(*args: Any, **kwargs: Any) -> None:
        calls["state_writes"].append((args, kwargs))
        result = next(writes, None)
        if isinstance(result, Exception):
            raise result

    def prepare(*args: Any, **kwargs: Any) -> tuple[int, dict[str, Any] | None]:
        calls["memorize_checks"].append((args, kwargs))
        return (prepare_auto_memorize or (lambda *_a, **_k: (0, None)))(*args, **kwargs)

    def schedule(*args: Any, **kwargs: Any) -> str:
        return (schedule_auto_memorize or (lambda *_a, **_k: "launched"))(*args, **kwargs)

    def record_call(*args: Any, **kwargs: Any) -> None:
        calls["route_telemetry"].append((args, copy.deepcopy(kwargs)))

    monkeypatch.setattr(mentra_routes, "_mint_gemini_token", mint)
    monkeypatch.setattr(mentra_routes.secrets, "token_urlsafe", lambda _length: next(sitting_ids))
    app = FastAPI()
    register_mentra_routes(
        app,
        get_config=lambda: config,
        get_service_from_scope=get_service,
        load_turn_state_and_soul_card=load_state,
        build_identity_context=lambda soul_id: f"Today is server time.\nYou are {soul_id}.",
        load_cross_chat_context=load_cross,
        load_current_history=load_current,
        conversation_retrieve=retrieve,
        record_call=record_call,
        get_storage_dir=lambda: tmp_path,
        get_soul_lock=lambda _user_id, _soul_id: soul_lock,
        write_conversation_state=write_state,
        prepare_auto_memorize=prepare,
        schedule_auto_memorize=schedule,
    )
    return TestClient(app, raise_server_exceptions=raise_server_exceptions), calls, config


def test_mentra_health_requires_enabled_configured_bearer() -> None:
    config = {"mentra": {"enabled": False, "integration_bearer_token": ""}}
    app = FastAPI()
    register_mentra_routes(app, get_config=lambda: config)
    client = TestClient(app)

    assert client.get("/integration/mentra/health").status_code == 404

    config["mentra"]["enabled"] = True
    assert client.get("/integration/mentra/health").status_code == 503

    config["mentra"]["integration_bearer_token"] = "test-secret"
    missing = client.get("/integration/mentra/health")
    wrong = client.get("/integration/mentra/health", headers={"Authorization": "Bearer wrong"})
    assert missing.status_code == wrong.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"

    accepted = client.get("/integration/mentra/health", headers=AUTH)
    assert accepted.status_code == 200
    assert accepted.json() == {"ok": True}


def test_start_auth_and_validation_precede_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(monkeypatch, tmp_path)

    assert client.post("/integration/mentra/session/start", json={}, headers=AUTH).status_code == 422
    assert client.post("/integration/mentra/session/start", json={}).status_code == 401
    same_identity = {**START, "user_id": " CODEXIA "}
    assert client.post(
        "/integration/mentra/session/start", json=same_identity, headers=AUTH
    ).status_code == 409
    rejected_call = calls["route_telemetry"][-1]
    assert rejected_call[1]["ok"] is False
    assert set(rejected_call[1]["info"]) == {"totalMs"}
    invalid_device = {**START, "device_session_id": "bad/id"}
    assert client.post(
        "/integration/mentra/session/start", json=invalid_device, headers=AUTH
    ).status_code == 422
    assert calls["service"] == 0
    assert calls["token"] == []


def test_start_requires_model_and_voice_before_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, config = _session_app(monkeypatch, tmp_path)

    config["mentra"]["model"] = ""
    assert client.post("/integration/mentra/session/start", json=START, headers=AUTH).status_code == 503
    config["mentra"]["model"] = "model"
    config["mentra"]["voice"] = ""
    assert client.post("/integration/mentra/session/start", json=START, headers=AUTH).status_code == 503
    assert calls["service"] == 0


def test_start_builds_bounded_instruction_and_returns_only_client_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(monkeypatch, tmp_path)

    response = client.post("/integration/mentra/session/start", json=START, headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "ok": True,
        "session_id": "sitting-1",
        "conversation_id": "mentra:phone-1",
        "next_transcript_sequence": 1,
        "model": "gemini-2.5-flash-native-audio-preview-12-2025",
        "voice": "Kore",
        "ephemeral_token": "ephemeral-1",
        "websocket": {
            "api_version": "v1alpha",
            "method": "BidiGenerateContentConstrained",
            "input_audio_rate_hz": 16000,
            "output_audio_rate_hz": 24000,
        },
        "lease_seconds": 90,
    }
    assert calls["anchors"] == {"user_id": "Fictional User", "soul_id": "Codexia"}
    assert calls["state"][0][0] == "mentra:phone-1"
    assert calls["cross"][0]["conversation_id"] == "mentra:phone-1"
    prompt = calls["token"][0]["system_instruction"]
    headings = [
        "Today is server time.",
        "My narrative self:",
        "My identity and lived experience:",
        "The user's identity and lived experience:",
        "Recent conversations and activities:",
        "Use recall_memory when relevant context is missing.",
        "Speak naturally and concisely",
    ]
    assert [prompt.index(text) for text in headings] == sorted(prompt.index(text) for text in headings)
    serialized = json.dumps(body)
    assert "permanent-secret" not in serialized
    assert "I am Codexia" not in serialized
    assert "Fictional Friend" not in serialized
    start_call = calls["route_telemetry"][-1]
    assert start_call[0] == (
        "mentra.start",
        {"user": {"user_id": "Fictional User", "soul_id": "Codexia"}},
    )
    assert set(start_call[1]["info"]) == {
        "setupMs",
        "anchorsMs",
        "stateMs",
        "contextMs",
        "tokenMs",
        "totalMs",
    }


def test_lease_resume_heartbeat_and_end_are_scoped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(monkeypatch, tmp_path)

    first = client.post("/integration/mentra/session/start", json=START, headers=AUTH)
    resumed = client.post("/integration/mentra/session/start", json=START, headers=AUTH)
    assert first.status_code == resumed.status_code == 200
    assert first.json()["session_id"] == "sitting-1"
    assert resumed.json()["session_id"] == "sitting-2"
    assert first.json()["ephemeral_token"] != resumed.json()["ephemeral_token"]

    competing = {**START, "device_session_id": "phone-2"}
    assert client.post(
        "/integration/mentra/session/start", json=competing, headers=AUTH
    ).status_code == 409
    other_user = {**START, "user_id": "Another Fictional User"}
    assert client.post(
        "/integration/mentra/session/start", json=other_user, headers=AUTH
    ).status_code == 409
    assert calls["service"] == 2

    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    assert client.post(
        "/integration/mentra/session/wrong/heartbeat", json=scope, headers=AUTH
    ).status_code == 404
    wrong_user_scope = {**scope, "user_id": "Another Fictional User"}
    assert client.post(
        "/integration/mentra/session/sitting-2/heartbeat", json=wrong_user_scope, headers=AUTH
    ).status_code == 404
    assert client.post(
        "/integration/mentra/session/sitting-2/heartbeat", json=scope, headers=AUTH
    ).status_code == 200
    stale_append = client.post(
        "/integration/mentra/session/sitting-1/transcripts/append",
        json={
            **scope,
            "events": [{
                "event_id": "sitting-1:1",
                "sequence": 1,
                "event_kind": "transcript",
                "role": "user",
                "content": "A stale fictional line.",
                "status": "complete",
            }],
        },
        headers=AUTH,
    )
    assert stale_append.status_code == 404
    assert client.post(
        "/integration/mentra/session/wrong/end", json=scope, headers=AUTH
    ).status_code == 409
    assert client.post(
        "/integration/mentra/session/sitting-1/end", json=scope, headers=AUTH
    ).status_code == 409
    assert client.post(
        "/integration/mentra/session/sitting-2/end", json=scope, headers=AUTH
    ).status_code == 200
    assert client.post(
        "/integration/mentra/session/sitting-2/end", json=scope, headers=AUTH
    ).status_code == 200


def test_failed_replacement_start_preserves_healthy_sitting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(
        monkeypatch,
        tmp_path,
        token_results=[RuntimeError("upstream"), "token", RuntimeError("upstream")],
    )

    failed = client.post("/integration/mentra/session/start", json=START, headers=AUTH)
    assert failed.status_code == 502
    failed_call = calls["route_telemetry"][-1]
    assert failed_call[1]["ok"] is False
    assert set(failed_call[1]["info"]) == {
        "setupMs",
        "anchorsMs",
        "stateMs",
        "contextMs",
        "tokenMs",
        "totalMs",
    }
    assert failed_call[1]["error"].startswith("HTTPException:")
    competing = {**START, "device_session_id": "phone-2"}
    healthy = client.post(
        "/integration/mentra/session/start", json=competing, headers=AUTH
    )
    assert healthy.status_code == 200
    healthy_sitting = healthy.json()["session_id"]
    lease_before = mentra_routes._leases[START["soul_id"]]
    failed_renewal = client.post(
        "/integration/mentra/session/start", json=competing, headers=AUTH
    )
    assert failed_renewal.status_code == 502
    assert mentra_routes._leases[START["soul_id"]] == lease_before
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    assert client.post(
        f"/integration/mentra/session/{healthy_sitting}/heartbeat",
        json=scope,
        headers=AUTH,
    ).status_code == 200


def test_replacement_start_rejects_context_that_changed_during_mint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, _, _ = _session_app(monkeypatch, tmp_path)
    mint_count = 0

    async def mint(**_kwargs: str) -> str:
        nonlocal mint_count
        mint_count += 1
        if mint_count == 2:
            mentra_routes.conversation_sources.persist_mentra_history_snapshot(
                storage_dir=tmp_path,
                user_id=START["user_id"],
                soul_id=START["soul_id"],
                conversation_id="mentra:phone-1",
                history=[
                    {
                        "event_id": "sitting-1:1",
                        "sequence": 1,
                        "event_kind": "transcript",
                        "role": "user",
                        "content": "A fictional late line.",
                        "transcript_status": "complete",
                        "received_at": "2026-08-23T20:00:00.000Z",
                    }
                ],
            )
        return f"token-{mint_count}"

    monkeypatch.setattr(mentra_routes, "_mint_gemini_token", mint)
    healthy = client.post("/integration/mentra/session/start", json=START, headers=AUTH)
    assert healthy.status_code == 200
    healthy_sitting = healthy.json()["session_id"]

    stale = client.post("/integration/mentra/session/start", json=START, headers=AUTH)
    assert stale.status_code == 409
    assert stale.json()["detail"] == {"code": "mentra_history_changed"}
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    assert client.post(
        f"/integration/mentra/session/{healthy_sitting}/heartbeat",
        json=scope,
        headers=AUTH,
    ).status_code == 200

    refreshed = client.post("/integration/mentra/session/start", json=START, headers=AUTH)
    assert refreshed.status_code == 200
    assert refreshed.json()["next_transcript_sequence"] == 2


def test_failed_start_claim_cleanup_cannot_remove_newer_claim() -> None:
    mentra_routes._start_claims["Codexia"] = "newer-sitting"

    asyncio.run(mentra_routes._release_start_claim_if_owned("Codexia", "older-sitting"))

    assert mentra_routes._start_claims["Codexia"] == "newer-sitting"


def test_second_start_is_rejected_while_same_soul_start_is_in_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    mentra_routes._start_claims[START["soul_id"]] = "in-progress"

    response = client.post("/integration/mentra/session/start", json=START, headers=AUTH)

    assert response.status_code == 409
    assert calls["service"] == 0


def test_cancelled_start_always_releases_in_progress_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(
        monkeypatch,
        tmp_path,
        token_results=[asyncio.CancelledError()],
    )

    with pytest.raises((asyncio.CancelledError, FutureCancelledError)):
        client.post("/integration/mentra/session/start", json=START, headers=AUTH)

    assert START["soul_id"] not in mentra_routes._start_claims


def test_route_registration_has_explicit_transcript_state_seams() -> None:
    parameters = inspect.signature(register_mentra_routes).parameters
    assert {
        "get_storage_dir",
        "get_soul_lock",
        "write_conversation_state",
        "load_current_history",
        "conversation_retrieve",
        "record_call",
    } <= set(parameters)


def test_recall_is_sitting_scoped_read_only_compact_and_unlocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START, headers=AUTH
    ).json()["session_id"]
    endpoint = f"/integration/mentra/session/{sitting_id}/recall"
    request = {
        "user_id": START["user_id"],
        "soul_id": START["soul_id"],
        "query": "remember the fictional lighthouse",
    }

    invalid = client.post(
        endpoint,
        json={**request, "query": "rejected text", "extra": True},
        headers=AUTH,
    )
    assert invalid.status_code == 422
    assert invalid.json()["detail"][0]["type"] == "extra_forbidden"
    assert client.post(endpoint.replace(sitting_id, "wrong"), json=request, headers=AUTH).status_code == 404
    response = client.post(endpoint, json=request, headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "context": (
            "Dossiers:\n[Fictional Places] A remembered coast.\n\n"
            "Memories:\nKey: [profile] what's said or declared\n"
            "- [profile] The lighthouse keeper preferred cedar tea."
        ),
        "retrieve_ms": 123,
    }
    conversation_id, payload = calls["recalls"][0]
    assert conversation_id == "mentra:phone-1"
    assert payload["history"][0]["content"] == "A fictional current-chat line."
    assert payload["force_retrieve"] is True
    assert payload["_read_only_retrieve"] is True
    assert payload["mental_health_addon"] is False
    assert "memory-secret-id" not in response.text
    assert calls["lease_locked_during_retrieve"] is False
    assert calls["soul_locked_during_retrieve"] is False
    telemetry_args, telemetry_kwargs = calls["route_telemetry"][-1]
    assert telemetry_args[0] == "mentra.recall"
    assert telemetry_kwargs == {
        "ok": True,
        "info": {
            "wallMs": pytest.approx(0, abs=1000),
            "retrieveMs": 123,
            "resultCounts": {"categories": 1, "items": 1},
        },
    }
    assert "remember the fictional lighthouse" not in json.dumps(calls["route_telemetry"])


def test_recall_failure_is_generic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(
        monkeypatch,
        tmp_path,
        retrieve_result=RuntimeError("private fictional query leaked"),
    )
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START, headers=AUTH
    ).json()["session_id"]
    response = client.post(
        f"/integration/mentra/session/{sitting_id}/recall",
        json={
            "user_id": START["user_id"],
            "soul_id": START["soul_id"],
            "query": "private fictional query",
        },
        headers=AUTH,
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Mentra memory recall failed"}
    assert "private fictional" not in response.text
    assert calls["route_telemetry"][-1][1] == {
        "ok": False,
        "error": "RuntimeError: private fictional query leaked",
    }


def test_recall_timeout_is_temporary_and_config_failure_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    timeout_client, _, _ = _session_app(
        monkeypatch,
        tmp_path,
        retrieve_result=HTTPException(status_code=504, detail="private timeout"),
    )
    timeout_sitting = timeout_client.post(
        "/integration/mentra/session/start", json=START, headers=AUTH
    ).json()["session_id"]
    request = {"user_id": START["user_id"], "soul_id": START["soul_id"], "query": "private"}
    timeout = timeout_client.post(
        f"/integration/mentra/session/{timeout_sitting}/recall",
        json=request,
        headers=AUTH,
    )
    assert timeout.status_code == 502
    assert timeout.json() == {"detail": "Mentra memory recall failed"}

    mentra_routes._leases.clear()
    config_client, _, _ = _session_app(
        monkeypatch,
        tmp_path,
        retrieve_result=HTTPException(status_code=503, detail="private config"),
    )
    config_sitting = config_client.post(
        "/integration/mentra/session/start", json=START, headers=AUTH
    ).json()["session_id"]
    config = config_client.post(
        f"/integration/mentra/session/{config_sitting}/recall",
        json=request,
        headers=AUTH,
    )
    assert config.status_code == 503
    assert config.json() == {"detail": "Mentra memory recall failed"}


def test_recall_enforces_server_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def stalled_retrieve(
        _conversation_id: str, _payload: dict[str, Any]
    ) -> dict[str, Any]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(mentra_routes, "_RECALL_TIMEOUT_SECONDS", 0.01)
    client, calls, _ = _session_app(
        monkeypatch,
        tmp_path,
        retrieve_hook=stalled_retrieve,
    )
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START, headers=AUTH
    ).json()["session_id"]
    response = client.post(
        f"/integration/mentra/session/{sitting_id}/recall",
        json={
            "user_id": START["user_id"],
            "soul_id": START["soul_id"],
            "query": "private fictional query",
        },
        headers=AUTH,
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Mentra memory recall failed"}
    assert calls["route_telemetry"][-1][1] == {
        "ok": False,
        "error": "TimeoutError",
    }


def test_slow_recall_does_not_hold_lease_or_soul_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    started = Event()
    release = Event()

    async def slow_retrieve(_conversation_id: str, _payload: dict[str, Any]) -> dict[str, Any]:
        started.set()
        await asyncio.to_thread(release.wait, 5)
        return {"ok": True, "retrieve_ms": 1, "result": {}}

    client, _, _ = _session_app(monkeypatch, tmp_path, retrieve_hook=slow_retrieve)
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START, headers=AUTH
    ).json()["session_id"]
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    recall_path = f"/integration/mentra/session/{sitting_id}/recall"

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            client.post,
            recall_path,
            json={**scope, "query": "slow fictional recall"},
            headers=AUTH,
        )
        assert started.wait(2)
        assert client.post(
            f"/integration/mentra/session/{sitting_id}/heartbeat",
            json=scope,
            headers=AUTH,
        ).status_code == 200
        assert client.post(
            f"/integration/mentra/session/{sitting_id}/transcripts/append",
            json={
                **scope,
                "events": [
                    {
                        "event_id": f"{sitting_id}:1",
                        "sequence": 1,
                        "event_kind": "transcript",
                        "role": "user",
                        "content": "A fictional line while recall runs.",
                        "status": "complete",
                    }
                ],
            },
            headers=AUTH,
        ).status_code == 200
        release.set()
        assert future.result(timeout=2).status_code == 200


def test_transcript_append_is_contiguous_idempotent_and_initializes_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START, headers=AUTH
    ).json()["session_id"]
    endpoint = f"/integration/mentra/session/{sitting_id}/transcripts/append"
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    events = [
        {
            "event_id": f"{sitting_id}:1",
            "sequence": 1,
            "event_kind": "transcript",
            "role": "user",
            "content": "A fictional question.",
            "status": "complete",
        },
        {
            "event_id": f"{sitting_id}:2",
            "sequence": 2,
            "event_kind": "transcript",
            "role": "assistant",
            "content": "A fictional answer.",
            "status": "complete",
        },
    ]

    first = client.post(endpoint, json={**scope, "events": events}, headers=AUTH)
    replay = client.post(endpoint, json={**scope, "events": events}, headers=AUTH)
    suffix = client.post(
        endpoint,
        json={
            **scope,
            "events": [
                events[1],
                {
                    "event_id": f"{sitting_id}:3",
                    "sequence": 3,
                    "event_kind": "sitting_summary",
                    "role": "assistant",
                    "content": "I noticed a fictional shift.",
                },
            ],
        },
        headers=AUTH,
    )
    restarted = client.post("/integration/mentra/session/start", json=START, headers=AUTH)

    assert first.json() == {
        "ok": True,
        "conversation_id": "mentra:phone-1",
        "ack_sequence": 2,
        "accepted": 2,
        "duplicates": 0,
        "memorize_check_queued": False,
    }
    assert replay.json()["accepted"] == 0
    assert replay.json()["duplicates"] == 2
    assert suffix.json()["accepted"] == 1
    assert suffix.json()["duplicates"] == 1
    assert restarted.json()["next_transcript_sequence"] == 4
    history = mentra_routes.conversation_sources.load_mentra_history_snapshot(
        storage_dir=tmp_path,
        user_id=START["user_id"],
        soul_id=START["soul_id"],
        conversation_id="mentra:phone-1",
    )
    assert [row["sequence"] for row in history] == [1, 2, 3]
    assert all(row["received_at"].endswith("Z") for row in history)
    assert len(calls["state_writes"]) == 3
    assert calls["state_writes"][0][1]["updates"] == {"memorize_chat": True}


def test_transcript_append_queues_shared_auto_memorize_only_for_new_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scheduled: list[tuple[Any, ...]] = []
    payload = {"conversation": [{"content": "fictional", "memorize_chat": True}]}
    client, calls, _ = _session_app(
        monkeypatch,
        tmp_path,
        prepare_auto_memorize=lambda *_a, **_k: (9000, payload),
        schedule_auto_memorize=lambda *args: scheduled.append(args) or "launched",
    )
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START, headers=AUTH
    ).json()["session_id"]
    endpoint = f"/integration/mentra/session/{sitting_id}/transcripts/append"
    body = {
        "user_id": START["user_id"],
        "soul_id": START["soul_id"],
        "events": [
            {
                "event_id": f"{sitting_id}:1",
                "sequence": 1,
                "event_kind": "transcript",
                "role": "user",
                "content": "A fictional partial line.",
                "status": "interrupted",
            },
            {
                "event_id": f"{sitting_id}:2",
                "sequence": 2,
                "event_kind": "sitting_summary",
                "role": "assistant",
                "content": "I noticed a fictional shift.",
            },
        ],
    }

    first = client.post(endpoint, json=body, headers=AUTH)
    replay = client.post(endpoint, json=body, headers=AUTH)

    assert first.json()["memorize_check_queued"] is True
    assert replay.json()["memorize_check_queued"] is False
    assert len(calls["memorize_checks"]) == 1
    assert len(scheduled) == 1
    assert scheduled[0][0] is payload
    assert scheduled[0][1:4] == ("mentra:phone-1", START["user_id"], START["soul_id"])
    projected = calls["memorize_checks"][0][0][5]
    assert [row["source_conversation_index"] for row in projected] == [1, 2]
    assert all(isinstance(row["ts_ms"], int) for row in projected)
    assert [row["speaker"] for row in projected] == [START["user_id"], START["soul_id"]]
    assert projected[0]["content"].endswith("[interrupted]")
    assert projected[1]["content"].startswith("[End-of-sitting reflection]")


def test_sequence_and_duplicate_lookup_tolerate_compacted_prefix() -> None:
    history = [
        {
            "event_id": "fictional-sitting:40",
            "sequence": 40,
            "event_kind": "transcript",
            "role": "user",
            "content": "Retained fictional line.",
            "transcript_status": "complete",
            "received_at": "2026-08-23T10:00:00.000Z",
        },
        {
            "event_id": "fictional-sitting:42",
            "sequence": 42,
            "event_kind": "transcript",
            "role": "assistant",
            "content": "Retained fictional reply.",
            "transcript_status": "complete",
            "received_at": "2026-08-23T10:00:01.000Z",
        },
    ]
    events = mentra_routes.MentraTranscriptBatch.model_validate(
        {
            "user_id": "Fictional User",
            "soul_id": "Codexia",
            "events": [
                {
                    "event_id": "fictional-sitting:42",
                    "sequence": 42,
                    "event_kind": "transcript",
                    "role": "assistant",
                    "content": "Retained fictional reply.",
                    "status": "complete",
                },
                {
                    "event_id": "fictional-sitting:43",
                    "sequence": 43,
                    "event_kind": "transcript",
                    "role": "user",
                    "content": "New fictional line.",
                    "status": "complete",
                },
            ],
        }
    ).events

    merged, accepted, duplicates = mentra_routes._merge_transcript_events(
        history,
        events,
        sitting_id="fictional-sitting",
    )

    assert mentra_routes._next_transcript_sequence(history) == 43
    assert [row["sequence"] for row in merged] == [40, 42, 43]
    assert (accepted, duplicates) == (1, 1)
    assert merged[-1]["received_at"].endswith("Z")


def test_transcript_append_rejects_holes_conflicts_and_redacts_validation_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, _, _ = _session_app(monkeypatch, tmp_path)
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START, headers=AUTH
    ).json()["session_id"]
    endpoint = f"/integration/mentra/session/{sitting_id}/transcripts/append"
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}

    hole = client.post(
        endpoint,
        json={
            **scope,
            "events": [{
                "event_id": f"{sitting_id}:2",
                "sequence": 2,
                "event_kind": "transcript",
                "role": "user",
                "content": "Fictional hole.",
                "status": "complete",
            }],
        },
        headers=AUTH,
    )
    private_text = "PRIVATE-FICTIONAL-TRANSCRIPT"
    invalid = client.post(
        endpoint,
        json={
            **scope,
            "events": [{
                "event_id": f"{sitting_id}:1",
                "sequence": 1,
                "event_kind": "transcript",
                "role": "user",
                "content": private_text,
                "status": "complete",
                "timestamp": "client-time-is-forbidden",
            }],
        },
        headers=AUTH,
    )

    assert hole.status_code == 409
    assert hole.json()["detail"] == {"expected_sequence": 1}
    assert invalid.status_code == 422
    assert invalid.json() == {"detail": "Invalid transcript batch"}
    assert private_text not in invalid.text


def test_transcript_retry_repairs_state_after_snapshot_was_written(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(
        monkeypatch,
        tmp_path,
        state_results=[RuntimeError("fictional state failure"), None],
        raise_server_exceptions=False,
    )
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START, headers=AUTH
    ).json()["session_id"]
    endpoint = f"/integration/mentra/session/{sitting_id}/transcripts/append"
    payload = {
        "user_id": START["user_id"],
        "soul_id": START["soul_id"],
        "events": [{
            "event_id": f"{sitting_id}:1",
            "sequence": 1,
            "event_kind": "transcript",
            "role": "user",
            "content": "A recoverable fictional line.",
            "status": "interrupted",
        }],
    }

    failed = client.post(endpoint, json=payload, headers=AUTH)
    repaired = client.post(endpoint, json=payload, headers=AUTH)

    assert failed.status_code == 500
    assert repaired.status_code == 200
    assert repaired.json()["accepted"] == 0
    assert repaired.json()["duplicates"] == 1
    assert len(calls["state_writes"]) == 2


def test_token_mint_uses_measured_constrained_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: Any) -> None:
            self.close()

    def urlopen(request: Any, timeout: int) -> Response:
        captured["request"] = request
        captured["timeout"] = timeout
        return Response(b'{"name":"ephemeral"}')

    monkeypatch.setattr(mentra_routes.urllib.request, "urlopen", urlopen)
    token = asyncio.run(
        mentra_routes._mint_gemini_token(
            api_key="secret",
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            voice="Kore",
            system_instruction="fictional context",
        )
    )

    assert token == "ephemeral"
    request = captured["request"]
    payload = json.loads(request.data)
    setup = payload["bidiGenerateContentSetup"]
    assert captured["timeout"] == 10
    assert request.get_header("X-goog-api-key") == "secret"
    assert payload["uses"] == 1
    assert payload["fieldMask"] == (
        "model,generationConfig,systemInstruction,tools,realtimeInputConfig,"
        "inputAudioTranscription,outputAudioTranscription,contextWindowCompression,"
        "proactivity,historyConfig"
    )
    assert setup["model"] == "models/gemini-2.5-flash-native-audio-preview-12-2025"
    assert setup["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "recall_memory",
                    "description": mentra_routes._RECALL_TOOL_DESCRIPTION,
                    "behavior": "NON_BLOCKING",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {"query": {"type": "STRING"}},
                        "required": ["query"],
                    },
                }
            ]
        }
    ]
    assert setup["realtimeInputConfig"] == {
        "automaticActivityDetection": {"silenceDurationMs": 1_500},
    }
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert setup["inputAudioTranscription"] == {}
    assert setup["outputAudioTranscription"] == {}
    assert "sessionResumption" not in setup
    assert setup["contextWindowCompression"] == {"slidingWindow": {}}
    assert setup["proactivity"] == {"proactiveAudio": False}
    assert setup["historyConfig"] == {}
    assert set(payload["fieldMask"].split(",")) == set(setup)
    expires = datetime.fromisoformat(payload["expireTime"].replace("Z", "+00:00"))
    new_session = datetime.fromisoformat(
        payload["newSessionExpireTime"].replace("Z", "+00:00")
    )
    now = datetime.now(UTC)
    assert 29 * 60 < (expires - now).total_seconds() <= 30 * 60
    assert 50 < (new_session - now).total_seconds() <= 60


def test_partial_mentra_config_update_preserves_omitted_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main

    config = copy.deepcopy(main._CONFIG)
    config["mentra"] = _configured()["mentra"]
    saved: dict[str, Any] = {}
    monkeypatch.setattr(main, "_CONFIG", config)
    monkeypatch.setattr(main, "_save_config", lambda value: saved.update(value))
    monkeypatch.setattr(main, "_clear_cached_services", lambda: None)

    response = TestClient(main.app).post("/config", json={"mentra": {"enabled": False}})

    assert response.status_code == 200
    assert saved["mentra"] == {**_configured()["mentra"], "enabled": False}
    assert response.json()["config"]["mentra"]["gemini_api_key"] == "***"
    assert response.json()["config"]["mentra"]["integration_bearer_token"] == "***"
