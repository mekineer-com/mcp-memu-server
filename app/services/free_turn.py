"""Free-turn continuation and follow-up helpers."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any


def _build_free_turn_prompt(
    *,
    reason: str,
    continuation_index: int,
    origin_conversation_id: str,
    previous_contract: dict[str, Any],
    allow_public_response: bool,
) -> str:
    response_target = str(previous_contract.get("response_target") or "").strip().lower()
    response = str(previous_contract.get("response") or "").strip()
    rehearsal = str(previous_contract.get("rehearsal") or "").strip()
    target_instruction = (
        "If you choose response_target respond/private, it can be sent through WhatsApp."
        if allow_public_response
        else "Valid response_target values here are observe/private. Do not use listen/respond."
    )
    return "\n".join(
        [
            f"You chose continue_reason={reason!r} after the live turn from {origin_conversation_id}.",
            f"This is continuation turn {continuation_index} of 3.",
            "Continue only the specific task/research/diary purpose you chose.",
            "Return the same strict turn-contract JSON. Do not invent a new user message.",
            target_instruction,
            "",
            "Previous turn outcome:",
            f"- response_target: {response_target or 'unknown'}",
            f"- response: {response or '(empty)'}",
            f"- rehearsal: {rehearsal or '(empty)'}",
        ]
    )


def _attachment_workspace(config: Mapping[str, Any]) -> str | None:
    workspace = str(config.get("claude_code_workspace") or "").strip()
    return workspace or None


def _parse_free_turn_contract(
    raw: Any,
    *,
    allow_public_response: bool,
    config: Mapping[str, Any],
    parse_turn_contract: Callable[..., dict[str, Any]],
    logger: Any,
) -> dict[str, Any]:
    try:
        return parse_turn_contract(
            raw,
            allow_public_response=allow_public_response,
            attachment_workspace=_attachment_workspace(config),
        )
    except (ValueError, json.JSONDecodeError):
        text = str(raw or "").strip()
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("No complete JSON object found")
            parsed = json.loads(text[start : end + 1])
        except (ValueError, json.JSONDecodeError):
            raise
        logger.warning("free_turn: Claude Code returned prose around JSON; extracted turn contract")
        return parse_turn_contract(
            json.dumps(parsed),
            allow_public_response=allow_public_response,
            attachment_workspace=_attachment_workspace(config),
        )


def _turn_generation_metadata(payload: dict[str, Any], *, config: Mapping[str, Any]) -> dict[str, str]:
    if bool(config.get("claude_code", False)):
        model = str(config.get("claude_code_model") or "").strip()
        return {"api": "claude_code", "model": model} if model else {"api": "claude_code"}

    profiles = payload.get("llm_profiles") if isinstance(payload.get("llm_profiles"), dict) else {}
    default_profile = profiles.get("default") if isinstance(profiles.get("default"), dict) else {}
    api = str(default_profile.get("provider") or "").strip()
    model = str(default_profile.get("chat_model") or "").strip()
    out: dict[str, str] = {}
    if api:
        out["api"] = api
    if model:
        out["model"] = model
    return out


async def _run_free_turn_chain(
    *,
    marker: str,
    service: Any,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    session_id: str,
    initial_reason: str,
    initial_contract: dict[str, Any],
    system_prompt: str,
    allow_public_response: bool,
    safe_payload: dict[str, Any],
    soul_card: str | None,
    system_prompt_has_activity_recap: bool = False,
    config: Mapping[str, Any],
    make_turn_system_prompt: Callable[..., str],
    parse_free_turn_contract: Callable[..., dict[str, Any]],
    record_activity_message: Callable[..., bool],
    activity_recap_from_contract: Callable[[dict[str, Any]], str],
    insert_whatsapp_outbound: Callable[..., str],
    schedule_free_turn_continuation: Callable[..., str | None],
    clear_inflight: Callable[[set[str], str], None],
    free_turn_inflight: set[str],
    logger: Any,
) -> None:
    reason = initial_reason
    previous_contract = initial_contract
    free_turn_system_prompt = system_prompt
    if not system_prompt_has_activity_recap:
        free_turn_system_prompt = make_turn_system_prompt(
            soul_id,
            soul_card=soul_card,
            response_sentences=int(config.get("turn_response_sentences", 3)),
            allow_public_response=allow_public_response,
            include_activity_recap=True,
        )
    try:
        for continuation_index in range(1, 4):
            prompt = _build_free_turn_prompt(
                reason=reason,
                continuation_index=continuation_index,
                origin_conversation_id=conversation_id,
                previous_contract=previous_contract,
                allow_public_response=allow_public_response,
            )
            raw = await service.chat(
                prompt,
                system_prompt=free_turn_system_prompt,
                response_format={"type": "json_object"},
                op="free_turn",
                step=f"continue_{continuation_index}",
                resume_session_id=session_id,
            )
            contract = parse_free_turn_contract(raw, allow_public_response=allow_public_response)
            record_activity_message(
                user_id=user_id,
                soul_id=soul_id,
                recap=activity_recap_from_contract(contract),
            )
            response_target = str(contract.get("response_target") or "").strip().lower()
            response = str(contract.get("response") or "").strip()
            media_path = str(contract.get("attachment") or "").strip() or None
            if response_target in {"respond", "private"} and (response or media_path):
                if conversation_id.startswith("whatsapp:"):
                    out_id = insert_whatsapp_outbound(
                        user_id=user_id,
                        soul_id=soul_id,
                        origin_conversation_id=conversation_id,
                        target=response_target,
                        response_text=response,
                        media_path=media_path,
                        metadata={
                            "reason": reason,
                            "continuation_index": continuation_index,
                            "source": "free_turn",
                        },
                    )
                    logger.info("free_turn: queued WhatsApp outbound %s target=%s", out_id, response_target)
                else:
                    logger.info("free_turn: response ignored for non-WhatsApp conversation")
            cache_entry = str(contract.get("cache_entry") or "").strip()
            annulments = contract.get("annulments") if isinstance(contract.get("annulments"), list) else []
            if cache_entry or annulments:
                logger.info("free_turn: cache_entry/annulments intentionally ignored for continuation state")
            next_reason = str(contract.get("continue_reason") or "").strip().lower()
            next_continue_at = str(contract.get("continue_at") or "").strip()
            if not next_reason:
                return
            if next_continue_at:
                schedule_free_turn_continuation(
                    user_id=user_id,
                    soul_id=soul_id,
                    conversation_id=conversation_id,
                    continue_at=next_continue_at,
                    continue_reason=next_reason,
                    safe_payload=safe_payload,
                )
                return
            reason = next_reason
            previous_contract = contract
    except Exception:
        logger.exception("free_turn: continuation chain failed for %s", marker)
    finally:
        clear_inflight(free_turn_inflight, marker)


def _queue_free_turn_chain(
    *,
    service: Any,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    session_id: str,
    initial_reason: str,
    initial_contract: dict[str, Any],
    system_prompt: str,
    allow_public_response: bool,
    safe_payload: dict[str, Any],
    soul_card: str | None,
    system_prompt_has_activity_recap: bool = False,
    mark_inflight: Callable[[set[str], str], bool],
    free_turn_inflight: set[str],
    background_tasks: set[asyncio.Task],
    run_free_turn_chain: Callable[..., Any],
    logger: Any,
) -> bool:
    marker = f"{user_id}::{soul_id}"
    if not mark_inflight(free_turn_inflight, marker):
        logger.info("free_turn: continuation skipped because one is already running for %s", marker)
        return False
    task = asyncio.create_task(
        run_free_turn_chain(
            marker=marker,
            service=service,
            user_id=user_id,
            soul_id=soul_id,
            conversation_id=conversation_id,
            session_id=session_id,
            initial_reason=initial_reason,
            initial_contract=initial_contract,
            system_prompt=system_prompt,
            allow_public_response=allow_public_response,
            safe_payload=safe_payload,
            soul_card=soul_card,
            system_prompt_has_activity_recap=system_prompt_has_activity_recap,
        )
    )
    background_tasks.add(task)

    def _on_done(done_task: asyncio.Task) -> None:
        background_tasks.discard(done_task)

    task.add_done_callback(_on_done)
    return True


def _ensure_free_turn_continuations_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
CREATE TABLE IF NOT EXISTS free_turn_continuations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    soul_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    continue_at TEXT NOT NULL,
    due_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    claimed_at TEXT,
    completed_at TEXT,
    failed_at TEXT,
    last_error TEXT,
    payload_json TEXT NOT NULL
)
"""
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_free_turn_continuations_due "
        "ON free_turn_continuations(status, due_at)"
    )
    con.commit()


def _free_turn_continuation_row(row: sqlite3.Row, *, json_from_db: Callable[[Any], Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "soul_id": row["soul_id"],
        "conversation_id": row["conversation_id"],
        "continue_at": row["continue_at"],
        "due_at": row["due_at"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "claimed_at": row["claimed_at"],
        "completed_at": row["completed_at"],
        "failed_at": row["failed_at"],
        "last_error": row["last_error"],
        "payload": json_from_db(row["payload_json"]) or {},
    }


def _parse_free_turn_continue_at(raw: str, *, server_timezone: Callable[[], tzinfo]) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=server_timezone())
        return dt.astimezone(UTC)
    except ValueError:
        pass

    zone = server_timezone()
    candidates = [text, text.rsplit(" ", 1)[0]]
    for candidate in candidates:
        for fmt in ("%A, %B %d, %Y %H:%M", "%B %d, %Y %H:%M"):
            try:
                dt = datetime.strptime(candidate, fmt)
            except ValueError:
                continue
            return dt.replace(tzinfo=zone).astimezone(UTC)
    return None


def _free_turn_continuation_payload(safe: dict[str, Any]) -> dict[str, Any]:
    kept: dict[str, Any] = {}
    for key in (
        "user",
        "user_name",
        "chat_name",
        "chat_type",
        "channel_mode",
        "soul_card",
        "memorize_chat",
        "allow_public_response",
    ):
        if key in safe:
            kept[key] = safe[key]
    return kept


def _schedule_free_turn_continuation(
    *,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    continue_at: str,
    continue_reason: str,
    safe_payload: dict[str, Any],
    parse_free_turn_continue_at: Callable[[str], datetime | None],
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    sqlite_ensure_nonempty: Callable[[Path], None],
    sqlite_connect: Callable[[Path], sqlite3.Connection],
    json_to_db: Callable[[Any], str],
    touch_poll_marker: Callable[[Path, str, str], None],
    logger: Any,
) -> str | None:
    reason = str(continue_reason or "").strip()
    if not reason:
        logger.warning("free_turn: scheduled continuation ignored because continue_reason is missing")
        return None
    due_at = parse_free_turn_continue_at(continue_at)
    if due_at is None:
        logger.warning("free_turn: invalid continue_at ignored: %r", continue_at)
        return None
    db_path = sqlite_current_path(user_id, soul_id)
    if db_path is None:
        logger.warning("free_turn: scheduled continuation ignored because sqlite path is unavailable")
        return None
    sqlite_ensure_nonempty(db_path)
    now_iso = datetime.now(UTC).isoformat()
    continuation_id = f"wafup_{uuid.uuid4().hex}"
    payload = _free_turn_continuation_payload(safe_payload)
    payload["continue_reason"] = reason
    con = sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        _ensure_free_turn_continuations_schema(con)
        con.execute(
            """
INSERT INTO free_turn_continuations (
    id, user_id, soul_id, conversation_id, continue_at, due_at,
    status, created_at, updated_at, payload_json
) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
""",
            (
                continuation_id,
                user_id,
                soul_id,
                conversation_id,
                continue_at,
                due_at.isoformat(),
                now_iso,
                now_iso,
                json_to_db(payload),
            ),
        )
        con.commit()
    finally:
        con.close()
    touch_poll_marker(db_path, "free-turn-continuations", str(due_at.timestamp()))
    logger.info("free_turn: scheduled continuation %s due_at=%s", continuation_id, due_at.isoformat())
    return continuation_id


def _free_turn_continuation_db_paths(
    *,
    storage_status: Mapping[str, Any],
    config: Mapping[str, Any],
    sqlite_dir_from_cfg: Callable[..., Path],
    logger: Any,
) -> list[Path]:
    base_dsn = str(storage_status.get("dsn") or "")
    sqlite_dir = sqlite_dir_from_cfg(config, fallback_dsn=base_dsn)
    try:
        return sorted(path for path in sqlite_dir.glob("*.db") if path.is_file())
    except OSError:
        logger.exception("free_turn: failed to scan continuation sqlite dir %s", sqlite_dir)
        return []


def _claim_due_free_turn_continuations(
    db_path: Path,
    *,
    now: datetime,
    json_from_db: Callable[[Any], Any],
    sqlite_connect: Callable[[Path], sqlite3.Connection],
    poll_marker_due: Callable[..., bool],
    sqlite_has_rows_quietly: Callable[..., bool],
    remove_poll_marker: Callable[[Path, str], None],
    limit: int = 5,
    claim_timeout_seconds: int = 7200,
) -> list[dict[str, Any]]:
    claimed: list[dict[str, Any]] = []
    now_iso = now.astimezone(UTC).isoformat()
    stale_before = (now.astimezone(UTC) - timedelta(seconds=max(1, int(claim_timeout_seconds)))).isoformat()
    if not poll_marker_due(db_path, "free-turn-continuations", now=now):
        return []
    if not sqlite_has_rows_quietly(
        db_path,
        table="free_turn_continuations",
        where_sql="(status = 'pending' AND due_at <= ?) OR (status = 'running' AND claimed_at < ?)",
        params=(now_iso, stale_before),
    ):
        remove_poll_marker(db_path, "free-turn-continuations")
        return []
    con = sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        table = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'free_turn_continuations'"
        ).fetchone()
        if table is None:
            return []
        rows = con.execute(
            """
SELECT id
FROM free_turn_continuations
WHERE (status = 'pending' AND due_at <= ?)
   OR (status = 'running' AND claimed_at < ?)
ORDER BY due_at ASC
LIMIT ?
""",
            (now_iso, stale_before, max(1, min(20, int(limit)))),
        ).fetchall()
        for row in rows:
            continuation_id = str(row["id"])
            cur = con.execute(
                """
UPDATE free_turn_continuations
SET status = 'running', claimed_at = ?, updated_at = ?
WHERE id = ?
  AND ((status = 'pending' AND due_at <= ?) OR (status = 'running' AND claimed_at < ?))
""",
                (now_iso, now_iso, continuation_id, now_iso, stale_before),
            )
            if cur.rowcount != 1:
                continue
            claimed_row = con.execute(
                "SELECT * FROM free_turn_continuations WHERE id = ? LIMIT 1",
                (continuation_id,),
            ).fetchone()
            if claimed_row is not None:
                claimed.append(_free_turn_continuation_row(claimed_row, json_from_db=json_from_db))
        con.commit()
    finally:
        con.close()
    if not claimed:
        remove_poll_marker(db_path, "free-turn-continuations")
    return claimed


def _mark_free_turn_continuation(
    db_path: Path,
    continuation_id: str,
    *,
    status: str,
    sqlite_connect: Callable[[Path], sqlite3.Connection],
    error: str | None = None,
) -> None:
    final_status = str(status or "").strip().lower()
    if final_status not in {"completed", "failed"}:
        raise ValueError("continuation status must be completed|failed")
    now_iso = datetime.now(UTC).isoformat()
    completed_at = now_iso if final_status == "completed" else None
    failed_at = now_iso if final_status == "failed" else None
    con = sqlite_connect(db_path)
    try:
        con.execute(
            """
UPDATE free_turn_continuations
SET status = ?, updated_at = ?, completed_at = ?, failed_at = ?, last_error = ?
WHERE id = ? AND status = 'running'
""",
            (
                final_status,
                now_iso,
                completed_at,
                failed_at,
                str(error or "").strip() or None,
                continuation_id,
            ),
        )
        con.commit()
    finally:
        con.close()


async def _run_free_turn_continuation(
    row: dict[str, Any],
    db_path: Path,
    *,
    mark_inflight: Callable[[set[str], str], bool],
    free_turn_scheduled_inflight: set[str],
    conversation_retrieve: Callable[..., Any],
    conversation_turn: Callable[..., Any],
    build_prompt_override_payload: Callable[[dict[str, Any]], dict[str, Any]],
    insert_whatsapp_outbound: Callable[..., str],
    mark_free_turn_continuation: Callable[..., None],
    clear_inflight: Callable[[set[str], str], None],
    logger: Any,
) -> None:
    continuation_id = str(row.get("id") or "").strip()
    marker = f"continuation::{continuation_id}"
    if not mark_inflight(free_turn_scheduled_inflight, marker):
        return
    try:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        user_id = str(row.get("user_id") or "").strip()
        soul_id = str(row.get("soul_id") or "").strip()
        conversation_id = str(row.get("conversation_id") or "").strip()
        user_scope = {"user_id": user_id, "soul_id": soul_id, "conversation_id": conversation_id}
        continue_reason = str(payload.get("continue_reason") or "").strip()
        trace_id = uuid.uuid4().hex
        message = (
            f"Scheduled continuation due now. You asked to wake at {row.get('continue_at')}. "
            f"Reason you gave: {continue_reason}. "
            "Use fresh memory and current chat history, then decide whether to send a WhatsApp message."
        )
        retrieve_payload = {
            **payload,
            "user": user_scope,
            "self_turn_directive": message,
            "self_turn_label": "Scheduled wake",
            "history": [],
            "build_turn_prompt": True,
            "load_source_history": conversation_id.startswith("whatsapp:"),
            "is_live_turn": False,
            "trace_id": trace_id,
        }
        retrieve_out = await conversation_retrieve(conversation_id, retrieve_payload)
        prompt_override_payload = build_prompt_override_payload(retrieve_out)
        if not str(prompt_override_payload.get("user_prompt") or "").strip():
            raise RuntimeError("conversation_retrieve returned empty turn_user_prompt")
        load_source_history = conversation_id.startswith("whatsapp:")
        turn_history = [] if load_source_history else (
            retrieve_out.get("turn_history") if isinstance(retrieve_out.get("turn_history"), list) else []
        )
        turn_payload = {
            **payload,
            "user": user_scope,
            "self_turn_directive": message,
            "self_turn_label": "Scheduled wake",
            "history": turn_history,
            "prompt_override_payload": prompt_override_payload,
            "load_source_history": load_source_history,
            "is_live_turn": False,
            "trace_id": trace_id,
        }
        result = await conversation_turn(conversation_id, turn_payload)
        response_target = str(result.get("response_target") or "").strip().lower()
        response = str(result.get("response") or "").strip()
        media_path = str(result.get("attachment") or "").strip() or None
        if response_target in {"respond", "private"} and (response or media_path):
            outbound_target = response_target if conversation_id.startswith("whatsapp:") else "private"
            insert_whatsapp_outbound(
                user_id=user_id,
                soul_id=soul_id,
                origin_conversation_id=conversation_id,
                target=outbound_target,
                response_text=response,
                media_path=media_path,
                metadata={
                    "source": "free_turn_scheduled",
                    "continuation_id": continuation_id,
                    "requested_target": response_target,
                },
            )
        mark_free_turn_continuation(db_path, continuation_id, status="completed")
    except Exception as exc:
        logger.exception("free_turn: scheduled continuation failed id=%s", continuation_id)
        mark_free_turn_continuation(db_path, continuation_id, status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        clear_inflight(free_turn_scheduled_inflight, marker)


async def _run_due_free_turn_continuations_once(
    *,
    free_turn_continuation_db_paths: Callable[[], list[Path]],
    claim_due_free_turn_continuations: Callable[..., list[dict[str, Any]]],
    run_free_turn_continuation: Callable[[dict[str, Any], Path], Any],
) -> int:
    count = 0
    for db_path in free_turn_continuation_db_paths():
        for row in claim_due_free_turn_continuations(db_path, now=datetime.now(UTC)):
            count += 1
            await run_free_turn_continuation(row, db_path)
    return count


async def _free_turn_continuation_scheduler(
    *,
    run_due_free_turn_continuations_once: Callable[[], Any],
    logger: Any,
) -> None:
    while True:
        try:
            await run_due_free_turn_continuations_once()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("free_turn: continuation scheduler pass failed")
        await asyncio.sleep(30)
