from __future__ import annotations

import asyncio
import json
import re
import secrets
import time
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, NamedTuple

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from app.services import conversation_sources


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

class _Lease(NamedTuple):
    sitting_id: str
    user_id: str
    device_session_id: str
    expires_at: float


_leases: dict[str, _Lease] = {}
_start_claims: dict[str, str] = {}
_lease_lock = asyncio.Lock()


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


class MentraTranscriptEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    sequence: StrictInt
    event_kind: Literal["transcript", "sitting_summary"]
    role: Literal["user", "assistant"]
    content: str
    status: Literal["complete", "interrupted"] | None = None

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value) > 256:
            raise ValueError("must be 1-256 characters")
        return value

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        if len(value) > 16_000:
            raise ValueError("is too long")
        return value

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be positive")
        return value

    @model_validator(mode="after")
    def validate_kind(self) -> MentraTranscriptEvent:
        if self.event_kind == "transcript" and self.status is None:
            raise ValueError("transcript status is required")
        if self.event_kind == "sitting_summary" and (
            self.role != "assistant" or self.status is not None
        ):
            raise ValueError("sitting_summary must be an assistant event without status")
        return self


class MentraTranscriptBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    soul_id: str
    events: list[MentraTranscriptEvent]

    @field_validator("user_id", "soul_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("events")
    @classmethod
    def validate_events(cls, value: list[MentraTranscriptEvent]) -> list[MentraTranscriptEvent]:
        if not 1 <= len(value) <= 16:
            raise ValueError("must contain 1-16 events")
        sequences = [event.sequence for event in value]
        if any(right != left + 1 for left, right in zip(sequences, sequences[1:], strict=False)):
            raise ValueError("event sequences must be contiguous and increasing")
        return value


class _SequenceConflict(Exception):
    def __init__(self, expected_sequence: int):
        self.expected_sequence = expected_sequence


def _transcript_rows_by_sequence(history: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    previous = 0
    for row in history:
        sequence = row.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= previous:
            raise RuntimeError("mentra snapshot sequence is not strictly increasing")
        rows[sequence] = row
        previous = sequence
    return rows


def _next_transcript_sequence(history: list[dict[str, Any]]) -> int:
    rows = _transcript_rows_by_sequence(history)
    return max(rows, default=0) + 1


def _stored_event(event: MentraTranscriptEvent, *, received_at: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "event_id": event.event_id,
        "sequence": event.sequence,
        "event_kind": event.event_kind,
        "role": event.role,
        "content": event.content,
        "received_at": received_at,
    }
    if event.status is not None:
        row["transcript_status"] = event.status
    return row


def _same_event(row: dict[str, Any], event: MentraTranscriptEvent) -> bool:
    return (
        row.get("event_id") == event.event_id
        and row.get("sequence") == event.sequence
        and row.get("event_kind") == event.event_kind
        and row.get("role") == event.role
        and row.get("content") == event.content
        and row.get("transcript_status") == event.status
    )


def _merge_transcript_events(
    history: list[dict[str, Any]],
    events: list[MentraTranscriptEvent],
    *,
    sitting_id: str,
) -> tuple[list[dict[str, Any]], int, int]:
    rows_by_sequence = _transcript_rows_by_sequence(history)
    expected = max(rows_by_sequence, default=0) + 1
    accepted = 0
    duplicates = 0
    merged = list(history)
    for event in events:
        if event.event_id != f"{sitting_id}:{event.sequence}":
            raise _SequenceConflict(expected)
        if event.sequence < expected:
            existing = rows_by_sequence.get(event.sequence)
            if existing is None or not _same_event(existing, event):
                raise _SequenceConflict(expected)
            duplicates += 1
            continue
        if event.sequence != expected:
            raise _SequenceConflict(expected)
        merged.append(_stored_event(event, received_at=_rfc3339(datetime.now(UTC))))
        expected += 1
        accepted += 1
    return merged, accepted, duplicates


async def _release_start_claim_if_owned(lease_key: str, sitting_id: str) -> None:
    async with _lease_lock:
        if _start_claims.get(lease_key) == sitting_id:
            _start_claims.pop(lease_key, None)


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
    get_storage_dir: Callable[[], Path] | None = None,
    get_soul_lock: Callable[[str, str], asyncio.Lock] | None = None,
    write_conversation_state: Callable[..., Any] | None = None,
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
            or get_storage_dir is None
            or get_soul_lock is None
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
        conversation_id = f"mentra:{body.device_session_id}"
        sitting_id = secrets.token_urlsafe(18)

        async with _lease_lock:
            active = _leases.get(lease_key)
            if active and active.expires_at <= time.monotonic():
                _leases.pop(lease_key, None)
                active = None
            if active and (
                active.device_session_id != body.device_session_id
                or active.user_id != body.user_id
            ):
                raise HTTPException(status_code=409, detail="Another Mentra session is active")
            if lease_key in _start_claims:
                raise HTTPException(status_code=409, detail="Mentra session start is already in progress")
            _start_claims[lease_key] = sitting_id

        try:
            service = get_service_from_scope(scope)
            anchors = await service.ensure_dossier_anchors(scope)
        except ValueError as exc:
            await _release_start_claim_if_owned(lease_key, sitting_id)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception:
            await _release_start_claim_if_owned(lease_key, sitting_id)
            raise
        try:
            _, narrative, _ = load_turn_state_and_soul_card(
                conversation_id, user_id=body.user_id, soul_id=body.soul_id
            )
            chats = load_cross_chat_context(
                user_id=body.user_id,
                soul_id=body.soul_id,
                conversation_id=conversation_id,
            )
            instruction = _build_bootstrap_instruction(
                identity=build_identity_context(body.soul_id),
                narrative=str(narrative or "").strip(),
                soul_anchor=_anchor_prose(anchors["soul"]),
                user_anchor=_anchor_prose(anchors["user"]),
                chats=str(chats or "").strip(),
            )
        except Exception:
            await _release_start_claim_if_owned(lease_key, sitting_id)
            raise
        try:
            token = await _mint_gemini_token(
                api_key=api_key,
                model=model,
                voice=voice,
                system_instruction=instruction,
            )
        except Exception as exc:
            await _release_start_claim_if_owned(lease_key, sitting_id)
            raise HTTPException(status_code=502, detail="Gemini session token request failed") from exc

        try:
            async with get_soul_lock(body.user_id, body.soul_id):
                history = conversation_sources.load_mentra_history_snapshot(
                    storage_dir=get_storage_dir(),
                    user_id=body.user_id,
                    soul_id=body.soul_id,
                    conversation_id=conversation_id,
                )
                next_sequence = _next_transcript_sequence(history)
                async with _lease_lock:
                    if _start_claims.get(lease_key) != sitting_id:
                        raise HTTPException(status_code=409, detail="Mentra session was superseded")
                    active = _leases.get(lease_key)
                    if active and active.expires_at <= time.monotonic():
                        _leases.pop(lease_key, None)
                        active = None
                    if active and (
                        active.device_session_id != body.device_session_id
                        or active.user_id != body.user_id
                    ):
                        raise HTTPException(
                            status_code=409, detail="Another Mentra session is active"
                        )
                    _leases[lease_key] = _Lease(
                        sitting_id,
                        body.user_id,
                        body.device_session_id,
                        time.monotonic() + _LEASE_SECONDS,
                    )
                    _start_claims.pop(lease_key, None)
        except Exception:
            await _release_start_claim_if_owned(lease_key, sitting_id)
            raise

        return {
            "ok": True,
            "session_id": sitting_id,
            "conversation_id": conversation_id,
            "next_transcript_sequence": next_sequence,
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
            if active.expires_at <= time.monotonic():
                _leases.pop(key, None)
                raise HTTPException(status_code=404, detail="Mentra session not found")
            if active.sitting_id != session_id or active.user_id != body.user_id:
                raise HTTPException(status_code=404, detail="Mentra session not found")
            _leases[key] = active._replace(expires_at=time.monotonic() + _LEASE_SECONDS)
        return {"ok": True}

    @app.post(
        "/integration/mentra/session/{session_id}/end",
        tags=["integration"],
        dependencies=auth,
    )
    async def mentra_session_end(session_id: str, body: MentraSessionScope) -> dict[str, bool]:
        if get_soul_lock is None:
            raise HTTPException(status_code=503, detail="Mentra session teardown is not configured")
        key = body.soul_id
        async with get_soul_lock(body.user_id, body.soul_id):
            async with _lease_lock:
                active = _leases.get(key)
                if not active or active.expires_at <= time.monotonic():
                    _leases.pop(key, None)
                    return {"ok": True}
                if active.sitting_id != session_id or active.user_id != body.user_id:
                    raise HTTPException(status_code=409, detail="Another Mentra session is active")
                _leases.pop(key, None)
        return {"ok": True}

    @app.post(
        "/integration/mentra/session/{sitting_id}/transcripts/append",
        tags=["integration"],
        dependencies=auth,
    )
    async def mentra_transcripts_append(sitting_id: str, request: Request) -> dict[str, Any]:
        if get_storage_dir is None or get_soul_lock is None or write_conversation_state is None:
            raise HTTPException(status_code=503, detail="Mentra transcript persistence is not configured")
        try:
            raw = await request.json()
            body = MentraTranscriptBatch.model_validate(raw)
        except (ValueError, ValidationError):
            raise HTTPException(status_code=422, detail="Invalid transcript batch") from None

        conversation_id: str
        async with get_soul_lock(body.user_id, body.soul_id):
            storage_dir = get_storage_dir()
            async with _lease_lock:
                active = _leases.get(body.soul_id)
                if not active or active.expires_at <= time.monotonic():
                    _leases.pop(body.soul_id, None)
                    raise HTTPException(status_code=404, detail="Mentra session not found")
                if active.sitting_id != sitting_id or active.user_id != body.user_id:
                    raise HTTPException(status_code=404, detail="Mentra session not found")
                conversation_id = f"mentra:{active.device_session_id}"

            history = conversation_sources.load_mentra_history_snapshot(
                storage_dir=storage_dir,
                user_id=body.user_id,
                soul_id=body.soul_id,
                conversation_id=conversation_id,
            )
            try:
                merged, accepted, duplicates = _merge_transcript_events(
                    history,
                    body.events,
                    sitting_id=sitting_id,
                )
            except _SequenceConflict as exc:
                raise HTTPException(
                    status_code=409,
                    detail={"expected_sequence": exc.expected_sequence},
                ) from None
            conversation_sources.persist_mentra_history_snapshot(
                storage_dir=storage_dir,
                user_id=body.user_id,
                soul_id=body.soul_id,
                conversation_id=conversation_id,
                history=merged,
            )
            write_conversation_state(
                conversation_id,
                soul_id=body.soul_id,
                user_id=body.user_id,
                updates={"memorize_chat": True},
            )

        return {
            "ok": True,
            "conversation_id": conversation_id,
            "ack_sequence": _next_transcript_sequence(merged) - 1,
            "accepted": accepted,
            "duplicates": duplicates,
            "memorize_check_queued": False,
        }
