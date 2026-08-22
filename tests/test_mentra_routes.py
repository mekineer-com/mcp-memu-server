from __future__ import annotations

import asyncio
import copy
import io
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
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
    yield
    mentra_routes._leases.clear()


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
    *,
    token_results: list[str | Exception] | None = None,
) -> tuple[TestClient, dict[str, Any], dict[str, Any]]:
    config = _configured()
    calls: dict[str, Any] = {"service": 0, "token": [], "state": [], "cross": []}
    results = iter(token_results or ["ephemeral-1", "ephemeral-2", "ephemeral-3"])

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

    async def mint(**kwargs: str) -> str:
        calls["token"].append(kwargs)
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(mentra_routes, "_mint_gemini_token", mint)
    app = FastAPI()
    register_mentra_routes(
        app,
        get_config=lambda: config,
        get_service_from_scope=get_service,
        load_turn_state_and_soul_card=load_state,
        build_identity_context=lambda soul_id: f"Today is server time.\nYou are {soul_id}.",
        load_cross_chat_context=load_cross,
    )
    return TestClient(app), calls, config


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
) -> None:
    client, calls, _ = _session_app(monkeypatch)

    assert client.post("/integration/mentra/session/start", json={}, headers=AUTH).status_code == 422
    assert client.post("/integration/mentra/session/start", json={}).status_code == 401
    same_identity = {**START, "user_id": " CODEXIA "}
    assert client.post(
        "/integration/mentra/session/start", json=same_identity, headers=AUTH
    ).status_code == 409
    invalid_device = {**START, "device_session_id": "bad/id"}
    assert client.post(
        "/integration/mentra/session/start", json=invalid_device, headers=AUTH
    ).status_code == 422
    assert calls["service"] == 0
    assert calls["token"] == []


def test_start_builds_bounded_instruction_and_returns_only_client_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, calls, _ = _session_app(monkeypatch)

    response = client.post("/integration/mentra/session/start", json=START, headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "ok": True,
        "session_id": "phone-1",
        "conversation_id": "mentra:phone-1",
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
        "Speak naturally and concisely",
    ]
    assert [prompt.index(text) for text in headings] == sorted(prompt.index(text) for text in headings)
    serialized = json.dumps(body)
    assert "permanent-secret" not in serialized
    assert "I am Codexia" not in serialized
    assert "Fictional Friend" not in serialized


def test_lease_resume_heartbeat_and_end_are_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, calls, _ = _session_app(monkeypatch)

    first = client.post("/integration/mentra/session/start", json=START, headers=AUTH)
    resumed = client.post("/integration/mentra/session/start", json=START, headers=AUTH)
    assert first.status_code == resumed.status_code == 200
    assert first.json()["ephemeral_token"] != resumed.json()["ephemeral_token"]

    competing = {**START, "device_session_id": "phone-2"}
    assert client.post(
        "/integration/mentra/session/start", json=competing, headers=AUTH
    ).status_code == 409
    assert calls["service"] == 2

    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    assert client.post(
        "/integration/mentra/session/wrong/heartbeat", json=scope, headers=AUTH
    ).status_code == 404
    assert client.post(
        "/integration/mentra/session/phone-1/heartbeat", json=scope, headers=AUTH
    ).status_code == 200
    assert client.post(
        "/integration/mentra/session/wrong/end", json=scope, headers=AUTH
    ).status_code == 409
    assert client.post(
        "/integration/mentra/session/phone-1/end", json=scope, headers=AUTH
    ).status_code == 200
    assert client.post(
        "/integration/mentra/session/phone-1/end", json=scope, headers=AUTH
    ).status_code == 200


def test_failed_token_mint_rolls_back_new_and_renewed_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _ = _session_app(
        monkeypatch,
        token_results=[RuntimeError("upstream"), "token", RuntimeError("upstream")],
    )

    failed = client.post("/integration/mentra/session/start", json=START, headers=AUTH)
    assert failed.status_code == 502
    competing = {**START, "device_session_id": "phone-2"}
    assert client.post(
        "/integration/mentra/session/start", json=competing, headers=AUTH
    ).status_code == 200
    lease_before = mentra_routes._leases[(START["user_id"], START["soul_id"])]
    failed_renewal = client.post(
        "/integration/mentra/session/start", json=competing, headers=AUTH
    )
    assert failed_renewal.status_code == 502
    assert mentra_routes._leases[(START["user_id"], START["soul_id"])] == lease_before


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
    assert setup["model"] == "models/gemini-2.5-flash-native-audio-preview-12-2025"
    assert setup["tools"] == []
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert setup["inputAudioTranscription"] == {}
    assert setup["outputAudioTranscription"] == {}
    assert setup["sessionResumption"] == {}
    assert setup["contextWindowCompression"] == {"slidingWindow": {}}
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
