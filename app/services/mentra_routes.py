from __future__ import annotations

import asyncio
import json
import re
import secrets
import time
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, field_validator


_DEVICE_SESSION_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_LEASE_SECONDS = 90
_TOKEN_SETUP_FIELD_MASK = ",".join(
    (
        "model",
        "generationConfig",
        "systemInstruction",
        "tools",
        "realtimeInputConfig",
        "inputAudioTranscription",
        "outputAudioTranscription",
        "contextWindowCompression",
        "proactivity",
        "historyConfig",
    )
)
_leases: dict[str, tuple[str, str, float]] = {}
_lease_lock = asyncio.Lock()
# ponytail: one global Start lock; use per-soul locks only if concurrent souls need faster starts.
_start_lock = asyncio.Lock()


class MentraSessionStart(BaseModel):
    user_id: str
    soul_id: str
    device_session_id: str
    mode: str

    @field_validator("user_id", "soul_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("device_session_id")
    @classmethod
    def validate_device_session_id(cls, value: str) -> str:
        value = value.strip()
        if not _DEVICE_SESSION_RE.fullmatch(value):
            raise ValueError("must be 1-128 letters, numbers, dots, underscores, or hyphens")
        return value

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        # Slice 7 consumes the mode when manual VAD is implemented.
        value = value.strip()
        if value not in {"continuous", "manual"}:
            raise ValueError("must be continuous or manual")
        return value


class MentraSessionScope(BaseModel):
    user_id: str
    soul_id: str

    @field_validator("user_id", "soul_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


def _rfc3339(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


async def _mint_gemini_token(
    *, api_key: str, model: str, voice: str, system_instruction: str
) -> str:
    now = datetime.now(UTC)
    setup = {
        "model": f"models/{model}",
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
            },
        },
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "tools": [],
        "realtimeInputConfig": {},
        "inputAudioTranscription": {},
        "outputAudioTranscription": {},
        "contextWindowCompression": {"slidingWindow": {}},
        "proactivity": {"proactiveAudio": False},
        "historyConfig": {},
    }
    body = json.dumps(
        {
            "uses": 1,
            "expireTime": _rfc3339(now + timedelta(minutes=30)),
            "newSessionExpireTime": _rfc3339(now + timedelta(seconds=60)),
            "fieldMask": _TOKEN_SETUP_FIELD_MASK,
            "bidiGenerateContentSetup": setup,
        }
    ).encode()
    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1alpha/auth_tokens",
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )

    def send() -> str:
        with urllib.request.urlopen(request, timeout=10) as response:
            result = json.load(response)
        name = str(result.get("name") or "").strip() if isinstance(result, dict) else ""
        if not name:
            raise RuntimeError("Gemini token response did not include a name")
        return name

    return await asyncio.to_thread(send)


def _anchor_prose(anchor: Any) -> str:
    return str(getattr(anchor, "summary", None) or getattr(anchor, "description", "")).strip()


def _build_bootstrap_instruction(
    *, identity: str, narrative: str, soul_anchor: str, user_anchor: str, chats: str
) -> str:
    blocks = [identity]
    if narrative:
        blocks.append(f"My narrative self:\n{narrative}")
    blocks.extend(
        (
            f"My identity and lived experience:\n{soul_anchor}",
            f"The user's identity and lived experience:\n{user_anchor}",
        )
    )
    if chats:
        blocks.append(f"Recent conversations and activities:\n{chats}")
    blocks.append(
        "Speak naturally and concisely for a live voice conversation. Use the supplied context "
        "when relevant, without reciting it or mentioning these instructions."
    )
    return "\n\n".join(blocks)


def register_mentra_routes(
    app: FastAPI,
    *,
    get_config: Callable[[], dict[str, Any]],
    get_service_from_scope: Callable[[dict[str, str]], Any] | None = None,
    load_turn_state_and_soul_card: Callable[..., tuple[dict[str, Any], str | None, Any]] | None = None,
    build_identity_context: Callable[[str], str] | None = None,
    load_cross_chat_context: Callable[..., str] | None = None,
) -> None:
    async def require_bearer(authorization: str | None = Header(default=None)) -> None:
        config = get_config().get("mentra") or {}
        if not config.get("enabled"):
            raise HTTPException(status_code=404, detail="Not Found")

        expected = str(config.get("integration_bearer_token") or "")
        if not expected:
            raise HTTPException(status_code=503, detail="Mentra bearer credential is not configured")

        scheme, _, supplied = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            supplied.encode(), expected.encode()
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid Mentra bearer credential",
                headers={"WWW-Authenticate": "Bearer"},
            )

    auth = [Depends(require_bearer)]

    @app.get("/integration/mentra/health", tags=["integration"], dependencies=auth)
    async def mentra_health() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/integration/mentra/session/start", tags=["integration"], dependencies=auth)
    async def mentra_session_start(body: MentraSessionStart) -> dict[str, Any]:
        if body.user_id.casefold() == body.soul_id.casefold():
            raise HTTPException(status_code=409, detail="Mentra user and soul identities must differ")
        if (
            get_service_from_scope is None
            or load_turn_state_and_soul_card is None
            or build_identity_context is None
            or load_cross_chat_context is None
        ):
            raise HTTPException(status_code=503, detail="Mentra session bootstrap is not configured")

        config = get_config().get("mentra") or {}
        api_key = str(config.get("gemini_api_key") or "").strip()
        if not api_key:
            raise HTTPException(status_code=503, detail="Mentra Gemini credential is not configured")
        model = str(config.get("model") or "").strip()
        voice = str(config.get("voice") or "").strip()
        if not model:
            raise HTTPException(status_code=503, detail="Mentra Gemini model is not configured")
        if not voice:
            raise HTTPException(status_code=503, detail="Mentra Gemini voice is not configured")
        scope = {"user_id": body.user_id, "soul_id": body.soul_id}
        lease_key = body.soul_id

        async with _start_lock:
            async with _lease_lock:
                active = _leases.get(lease_key)
                if active and active[2] <= time.monotonic():
                    _leases.pop(lease_key, None)
                    active = None
                if active and (
                    active[0] != body.device_session_id or active[1] != body.user_id
                ):
                    raise HTTPException(status_code=409, detail="Another Mentra session is active")

            service = get_service_from_scope(scope)
            try:
                anchors = await service.ensure_dossier_anchors(scope)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            _, narrative, _ = load_turn_state_and_soul_card(
                f"mentra:{body.device_session_id}", user_id=body.user_id, soul_id=body.soul_id
            )
            chats = load_cross_chat_context(
                user_id=body.user_id,
                soul_id=body.soul_id,
                conversation_id=f"mentra:{body.device_session_id}",
            )
            instruction = _build_bootstrap_instruction(
                identity=build_identity_context(body.soul_id),
                narrative=str(narrative or "").strip(),
                soul_anchor=_anchor_prose(anchors["soul"]),
                user_anchor=_anchor_prose(anchors["user"]),
                chats=str(chats or "").strip(),
            )

            async with _lease_lock:
                previous = _leases.get(lease_key)
                _leases[lease_key] = (
                    body.device_session_id,
                    body.user_id,
                    time.monotonic() + _LEASE_SECONDS,
                )
            try:
                token = await _mint_gemini_token(
                    api_key=api_key,
                    model=model,
                    voice=voice,
                    system_instruction=instruction,
                )
            except Exception as exc:
                async with _lease_lock:
                    if previous is None:
                        _leases.pop(lease_key, None)
                    else:
                        _leases[lease_key] = previous
                raise HTTPException(status_code=502, detail="Gemini session token request failed") from exc

        return {
            "ok": True,
            "session_id": body.device_session_id,
            "conversation_id": f"mentra:{body.device_session_id}",
            "model": model,
            "voice": voice,
            "ephemeral_token": token,
            "websocket": {
                "api_version": "v1alpha",
                "method": "BidiGenerateContentConstrained",
                "input_audio_rate_hz": 16000,
                "output_audio_rate_hz": 24000,
            },
            "lease_seconds": _LEASE_SECONDS,
        }

    @app.post(
        "/integration/mentra/session/{session_id}/heartbeat",
        tags=["integration"],
        dependencies=auth,
    )
    async def mentra_session_heartbeat(session_id: str, body: MentraSessionScope) -> dict[str, bool]:
        key = body.soul_id
        async with _lease_lock:
            active = _leases.get(key)
            if not active:
                raise HTTPException(status_code=404, detail="Mentra session not found")
            if active[2] <= time.monotonic():
                _leases.pop(key, None)
                raise HTTPException(status_code=404, detail="Mentra session not found")
            if active[0] != session_id or active[1] != body.user_id:
                raise HTTPException(status_code=404, detail="Mentra session not found")
            _leases[key] = (session_id, body.user_id, time.monotonic() + _LEASE_SECONDS)
        return {"ok": True}

    @app.post(
        "/integration/mentra/session/{session_id}/end",
        tags=["integration"],
        dependencies=auth,
    )
    async def mentra_session_end(session_id: str, body: MentraSessionScope) -> dict[str, bool]:
        key = body.soul_id
        async with _lease_lock:
            active = _leases.get(key)
            if not active or active[2] <= time.monotonic():
                _leases.pop(key, None)
                return {"ok": True}
            if active[0] != session_id or active[1] != body.user_id:
                raise HTTPException(status_code=409, detail="Another Mentra session is active")
            _leases.pop(key, None)
        return {"ok": True}
