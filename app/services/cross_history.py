"""Cross-conversation history composition helpers."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.services import conversation_sources as _conversation_sources
from app.services import memorize_endpoint as _memorize_endpoint
from app.services import message_log as _message_log
from app.services.conversation_id import canonical_conversation_id
from app.services.payload import (
    _normalize_conversation,
    _normalize_turn_history,
    _parse_turn_ts_ms,
    _pick_str,
)
from app.services.state import (
    effective_digest_cursor_from_row as _effective_digest_cursor_from_row,
)
from app.services.state import (
    memorize_chat_from_row as _memorize_chat_from_row,
)

_main: Any = None


def _m() -> Any:
    if _main is None:
        raise RuntimeError("cross_history is not bound to app.main")
    return _main


def message_sort_key(msg: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(msg.get("received_at") or ""),
        str(msg.get("source_conversation_id") or msg.get("conversation_id") or ""),
        int(msg.get("source_conversation_index") or 0),
    )


def _resolve_cross_source_paths() -> tuple[Path, Path | None, Path | None, Path | None]:
    hermes_cfg = _m()._CONFIG.get("hermes") if isinstance(_m()._CONFIG.get("hermes"), dict) else {}
    hermes_home_raw = str(hermes_cfg.get("home") or "").strip()
    sessions_index_raw = str(hermes_cfg.get("sessions_index_path") or "").strip()
    state_db_raw = str(hermes_cfg.get("state_db_path") or "").strip()
    hermes_home_path = Path(hermes_home_raw).expanduser().resolve() if hermes_home_raw else None
    sessions_index_path = Path(sessions_index_raw).expanduser().resolve() if sessions_index_raw else None
    state_db_path = Path(state_db_raw).expanduser().resolve() if state_db_raw else None
    return _m()._get_storage_dir(_m()._CONFIG), hermes_home_path, sessions_index_path, state_db_path


def _resolve_whatsapp_web_source_config() -> tuple[Path | None, str]:
    hermes_cfg = _m()._CONFIG.get("hermes") if isinstance(_m()._CONFIG.get("hermes"), dict) else {}
    db_raw = str(hermes_cfg.get("whatsapp_web_source_db") or "").strip()
    db_path = Path(db_raw).expanduser().resolve() if db_raw else None
    reply_prefix = str(hermes_cfg.get("whatsapp_reply_prefix") or "")
    return db_path, reply_prefix


def _resolve_source_cursor(
    conversation_id: str,
    cursor: int,
    source_message_id: str | None,
    source_ts: int | None,
    *,
    rolling: bool,
    hermes_home_path: Path | None,
) -> tuple[int, int | None, bool]:
    if not _message_log.derive_source_label(conversation_id).startswith("whatsapp:"):
        return cursor, None, False
    db_path, _ = _m()._resolve_whatsapp_web_source_config()
    resolved, min_timestamp = _conversation_sources.resolve_whatsapp_web_source_cursor(
        conversation_id,
        cursor,
        source_message_id,
        source_ts,
        db_path,
        rolling=rolling,
        hermes_home=hermes_home_path,
    )
    return resolved, min_timestamp, True


def _load_soul_active_since(
    soul_id: str,
    *,
    hermes_home_path: Path | None,
    state_db_path: Path | None,
) -> float | None:
    return _conversation_sources.load_soul_active_since(
        soul_id=soul_id,
        hermes_home=hermes_home_path,
        state_db_path=state_db_path,
    )


_ACTIVE_SINCE_UNSET = object()


def _current_whatsapp_active_since_for_soul(
    conversation_id: str,
    soul_id: str,
) -> float | None:
    if not _message_log.derive_source_label(conversation_id).startswith("whatsapp:"):
        return None
    _storage_dir, hermes_home_path, _sessions_index_path, state_db_path = _m()._resolve_cross_source_paths()
    return _m()._load_soul_active_since(
        soul_id,
        hermes_home_path=hermes_home_path,
        state_db_path=state_db_path,
    )


def _filter_current_whatsapp_history_for_soul(
    conversation_id: str,
    soul_id: str,
    history: list[dict[str, Any]],
    *,
    active_since: Any = _ACTIVE_SINCE_UNSET,
) -> list[dict[str, Any]]:
    if not history or not _message_log.derive_source_label(conversation_id).startswith("whatsapp:"):
        return history
    if active_since is _ACTIVE_SINCE_UNSET:
        active_since = _m()._current_whatsapp_active_since_for_soul(conversation_id, soul_id)
    if active_since is None:
        return history

    threshold_ms = active_since * 1000.0
    out: list[dict[str, Any]] = []
    for i, msg in enumerate(history):
        ts_ms = _parse_turn_ts_ms(msg.get("ts_ms"))
        if ts_ms is None:
            ts_ms = _parse_turn_ts_ms(msg.get("received_at"))
        if ts_ms is None:
            raise HTTPException(
                status_code=400,
                detail=f"WhatsApp history row {i} is missing ts_ms for soul active_since cutoff",
            )
        if ts_ms >= threshold_ms:
            out.append(msg)
    return out


def _degrade_live_whatsapp_history_after_filter_error(
    *,
    conversation_id: str,
    soul_id: str,
    history: list[dict[str, Any]],
    active_since: Any,
    exc: HTTPException,
    op: str,
) -> list[dict[str, Any]]:
    _m().logger.warning(
        "%s: live WhatsApp history cutoff failed; continuing with valid in-scope rows only "
        "conversation_id=%s soul_id=%s detail=%s",
        op,
        conversation_id,
        soul_id,
        exc.detail,
    )
    if active_since is None:
        return history
    threshold_ms = active_since * 1000.0
    out: list[dict[str, Any]] = []
    dropped = 0
    for msg in history:
        ts_ms = _parse_turn_ts_ms(msg.get("ts_ms"))
        if ts_ms is None:
            ts_ms = _parse_turn_ts_ms(msg.get("received_at"))
        if ts_ms is None:
            dropped += 1
            continue
        if ts_ms >= threshold_ms:
            out.append(msg)
    if dropped:
        _m().logger.warning(
            "%s: dropped %d live WhatsApp history row(s) without ts_ms conversation_id=%s soul_id=%s",
            op,
            dropped,
            conversation_id,
            soul_id,
        )
    return out


def _source_id_matches_external(source_id: Any, external_message_id: Any) -> bool:
    source = str(source_id or "").strip()
    external = str(external_message_id or "").strip()
    return bool(source and external and (source == external or external in source))


def _filter_external_message_from_history(
    history: list[dict[str, Any]],
    external_message_id: Any,
) -> list[dict[str, Any]]:
    if not external_message_id:
        return history
    return [
        row for row in history
        if not _m()._source_id_matches_external(row.get("source_message_id"), external_message_id)
    ]


def _load_current_whatsapp_history_from_source(
    conversation_id: str,
    soul_id: str,
    *,
    active_since: float | None,
    external_message_id: Any = None,
    exclude_external_message: bool = True,
) -> list[dict[str, Any]] | None:
    if not _message_log.derive_source_label(conversation_id).startswith("whatsapp:"):
        return None
    _storage_dir, hermes_home_path, sessions_index_path, state_db_path = _m()._resolve_cross_source_paths()
    web_source_db_path, reply_prefix = _m()._resolve_whatsapp_web_source_config()
    assistant_ids = _conversation_sources.load_whatsapp_assistant_source_message_ids(
        conversation_id=conversation_id,
        hermes_home=hermes_home_path,
        sessions_index_path=sessions_index_path,
        state_db_path=state_db_path,
    )
    rows = _conversation_sources.load_whatsapp_web_source_tail(
        conversation_id=conversation_id,
        since_cursor=-1,
        recent_fallback_messages=0,
        soul_id=soul_id,
        reply_prefix=reply_prefix,
        hermes_home=hermes_home_path,
        web_source_db_path=web_source_db_path,
        min_timestamp=active_since,
        assistant_source_message_ids=assistant_ids,
    )
    if not external_message_id or not exclude_external_message:
        return rows
    return _m()._filter_external_message_from_history(rows, external_message_id)


def _prepare_current_whatsapp_history(
    *,
    conversation_id: str,
    soul_id: str,
    raw_history: Any,
    active_since: float | None,
    load_source_history: bool,
    external_message_id: Any,
    is_live_turn: bool,
    op: str,
    exclude_external_message: bool = True,
) -> list[dict[str, Any]]:
    history = _normalize_turn_history(raw_history)
    if load_source_history:
        try:
            source_history = _m()._load_current_whatsapp_history_from_source(
                conversation_id,
                soul_id,
                active_since=active_since,
                external_message_id=external_message_id,
                exclude_external_message=exclude_external_message,
            )
        except Exception as exc:
            if not is_live_turn:
                raise
            _m().logger.warning(
                "%s: live WhatsApp source history load failed; continuing with payload history "
                "conversation_id=%s soul_id=%s: %s",
                op,
                conversation_id,
                soul_id,
                exc,
            )
            source_history = None
        if source_history is not None:
            history = _normalize_turn_history(source_history)
    try:
        return _m()._filter_current_whatsapp_history_for_soul(
            conversation_id,
            soul_id,
            history,
            active_since=active_since,
        )
    except HTTPException as exc:
        if not is_live_turn:
            raise
        return _m()._degrade_live_whatsapp_history_after_filter_error(
            conversation_id=conversation_id,
            soul_id=soul_id,
            history=history,
            active_since=active_since,
            exc=exc,
            op=op,
        )


def _stamp_assistant_display_name(messages: list[dict[str, Any]], soul_name: str) -> None:
    display = str(soul_name or "").strip()
    if not display:
        return
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip().lower()
        if role != "assistant":
            continue
        existing = str(msg.get("name") or msg.get("speaker") or "").strip()
        if existing:
            continue
        msg["name"] = display
        msg["speaker"] = display


def _persist_completed_sillytavern_turn_snapshot(
    *,
    safe: dict[str, Any],
    history: list[dict[str, Any]],
    conversation_id: str,
    user_id: str,
    soul_id: str,
    message: str,
    response_text: str,
    response_target: str,
) -> None:
    if _message_log.derive_source_label(conversation_id) != "sillytavern":
        return
    completed = _normalize_conversation(history)
    user_text = str(message or "").strip()
    user_ts_ms = _parse_turn_ts_ms(safe.get("message_ts_ms"))
    if user_ts_ms is None:
        user_ts_ms = _parse_turn_ts_ms(safe.get("messageTsMs"))
    if user_text:
        user_row: dict[str, Any] = {
            "role": "user",
            "content": user_text,
            "name": _pick_str(safe, "user_name") or "",
            "source_message_id": _pick_str(safe, "message_source_id", "messageSourceId") or str(len(completed)),
        }
        if user_ts_ms is not None:
            user_row["ts_ms"] = user_ts_ms
        completed.append(user_row)

    soul_text = str(response_text or "").strip()
    if response_target == "respond" and soul_text:
        soul_ts_ms = int(time.time() * 1000)
        if user_ts_ms is not None and soul_ts_ms <= user_ts_ms:
            soul_ts_ms = user_ts_ms + 1
        completed.append(
            {
                "role": "soul",
                "content": soul_text,
                "name": _pick_str(safe, "chat_name") or soul_id,
                "source_message_id": str(len(completed)),
                "ts_ms": soul_ts_ms,
            }
        )

    if not completed:
        return
    _conversation_sources.persist_sillytavern_history_snapshot(
        storage_dir=_m()._get_storage_dir(_m()._CONFIG),
        user_id=user_id,
        soul_id=soul_id,
        conversation_id=conversation_id,
        history=completed,
        chat_name=_pick_str(safe, "chat_name") or None,
    )


def _load_tail_for_source_conversation(
    *,
    conversation_id: str,
    user_id: str,
    soul_id: str,
    since_cursor: int,
    recent_fallback_messages: int,
    storage_dir: Path,
    hermes_home_path: Path | None,
    sessions_index_path: Path | None,
    state_db_path: Path | None,
    min_timestamp: int | None = None,
    include_floor_without_new: bool = False,
) -> list[dict[str, Any]]:
    source_label = _message_log.derive_source_label(conversation_id)
    if source_label.startswith("whatsapp:"):
        active_since = _m()._load_soul_active_since(
            soul_id,
            hermes_home_path=hermes_home_path,
            state_db_path=state_db_path,
        )
        timestamp_floor = max(
            (value for value in (active_since, min_timestamp) if value is not None),
            default=None,
        )
        web_source_db_path, reply_prefix = _m()._resolve_whatsapp_web_source_config()
        assistant_ids = _conversation_sources.load_whatsapp_assistant_source_message_ids(
            conversation_id=conversation_id,
            hermes_home=hermes_home_path,
            sessions_index_path=sessions_index_path,
            state_db_path=state_db_path,
        )
        return _conversation_sources.load_whatsapp_web_source_tail(
            conversation_id=conversation_id,
            since_cursor=since_cursor,
            recent_fallback_messages=recent_fallback_messages,
            soul_id=soul_id,
            reply_prefix=reply_prefix,
            hermes_home=hermes_home_path,
            web_source_db_path=web_source_db_path,
            min_timestamp=timestamp_floor,
            assistant_source_message_ids=assistant_ids,
            include_floor_without_new=include_floor_without_new,
        )
    if source_label == "sillytavern":
        return _conversation_sources.load_sillytavern_tail(
            storage_dir=storage_dir,
            user_id=user_id,
            soul_id=soul_id,
            conversation_id=conversation_id,
            since_cursor=since_cursor,
            recent_fallback_messages=recent_fallback_messages,
        )
    if source_label == "atomic":
        return _conversation_sources.load_atomic_tail(
            storage_dir=storage_dir,
            user_id=user_id,
            soul_id=soul_id,
            conversation_id=conversation_id,
            since_cursor=since_cursor,
            recent_fallback_messages=recent_fallback_messages,
        )
    return []


def _latest_saved_segment_display_ranges(
    *,
    soul_id: str,
) -> dict[str, tuple[int, int]]:
    chats_dir = (_m()._get_storage_dir(_m()._CONFIG) / "st_chats").resolve()
    agent_slug = _m()._sanitize_db_filename(soul_id)
    if not chats_dir.exists() or not agent_slug:
        return {}
    segment_paths = sorted(
        chats_dir.glob(f"{agent_slug}_*/segments/*.json"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    for segment_path in segment_paths:
        try:
            raw = json.loads(segment_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, list):
            continue
        ranges = _memorize_endpoint._segment_display_ranges(raw)
        if ranges:
            return ranges
    return {}


def _load_cross_tail_from_sources(
    con: sqlite3.Connection,
    *,
    user_id: str,
    soul_id: str,
    exclude_conversation_id: str | None = None,
) -> list[dict[str, Any]]:
    storage_dir, hermes_home_path, sessions_index_path, state_db_path = _m()._resolve_cross_source_paths()
    excluded_id = str(exclude_conversation_id or "").strip()
    excluded_key = _source_conversation_key(excluded_id)
    cursor_rows = con.execute(
        "SELECT conversation_id, memorize_chat, digest_cursor, last_memorize_at, "
        "digest_cursor_source_message_id, digest_cursor_ts, "
        "last_display_segment_start_index, last_display_segment_end_index, last_display_segment_at "
        "FROM conversations"
    ).fetchall()
    newest_display_at = max(
        (
            str(row["last_display_segment_at"] or "").strip()
            for row in cursor_rows
            if str(row["last_display_segment_at"] or "").strip()
        ),
        default="",
    )
    all_messages: list[dict[str, Any]] = []
    resource_display_ranges: dict[str, tuple[int, int]] | None = None
    for row in cursor_rows:
        cid = str(row["conversation_id"] or "").strip()
        if not cid or cid == excluded_id or (excluded_key and _source_conversation_key(cid) == excluded_key):
            continue
        if cid.startswith("activity:"):
            continue
        source_label = _message_log.derive_source_label(cid)
        cursor = _effective_digest_cursor_from_row(row)
        try:
            cursor, min_timestamp, web_source = _resolve_source_cursor(
                cid,
                cursor,
                row["digest_cursor_source_message_id"],
                row["digest_cursor_ts"],
                rolling=False,
                hermes_home_path=hermes_home_path,
            )
            display_start = row["last_display_segment_start_index"]
            display_end = row["last_display_segment_end_index"]
            display_at = str(row["last_display_segment_at"] or "").strip()
            latest_display_participant = bool(
                web_source
                and row["last_memorize_at"]
                and display_start is not None
                and display_end is not None
                and newest_display_at
                and display_at == newest_display_at
            )
            if (
                not newest_display_at
                and not web_source
                and row["last_memorize_at"]
                and (display_start is None or display_end is None)
            ):
                if resource_display_ranges is None:
                    resource_display_ranges = _m()._latest_saved_segment_display_ranges(
                        soul_id=soul_id,
                    )
                fallback_range = resource_display_ranges.get(cid)
                if fallback_range is not None:
                    display_start, display_end = fallback_range
            if (
                not web_source
                and row["last_memorize_at"]
                and display_start is not None
                and display_end is not None
                and (not newest_display_at or display_at == newest_display_at)
            ):
                try:
                    segment_start = max(0, int(display_start))
                    segment_end = max(segment_start, int(display_end))
                except (TypeError, ValueError, OverflowError):
                    segment_start = segment_end = -1
                if segment_end >= 0:
                    cursor = max(
                        -1,
                        max(segment_start, segment_end - (_message_log.DEFAULT_CROSS_RECENT_FALLBACK_MESSAGES - 1)) - 1,
                    )
            tail = _m()._load_tail_for_source_conversation(
                conversation_id=cid,
                user_id=user_id,
                soul_id=soul_id,
                since_cursor=cursor,
                recent_fallback_messages=_message_log.DEFAULT_CROSS_RECENT_FALLBACK_MESSAGES,
                storage_dir=storage_dir,
                hermes_home_path=hermes_home_path,
                sessions_index_path=sessions_index_path,
                state_db_path=state_db_path,
                min_timestamp=min_timestamp,
                include_floor_without_new=latest_display_participant,
            )
        except Exception as exc:
            _m().logger.error("cross-context source read failed for conversation_id=%s: %s", cid, exc)
            is_lid_gap = "LID↔phone mapping gap" in str(exc)
            if source_label.startswith("whatsapp:") and not is_lid_gap:
                raise RuntimeError(f"WhatsApp web_source read failed for {cid}: {exc}") from exc
            continue
        _m()._stamp_assistant_display_name(tail, soul_id)
        all_messages.extend(tail)
    all_messages.sort(key=message_sort_key)
    return all_messages


def _source_conversation_key(conversation_id: str) -> str:
    return canonical_conversation_id(conversation_id)


def _clear_last_display_segments_for_nonparticipants(
    *,
    user_id: str,
    soul_id: str,
    participant_conversation_ids: set[str],
    run_started_at: str,
) -> None:
    db_path = _m()._sqlite_current_path(user_id or None, soul_id or None)
    if db_path is None or not db_path.exists():
        return
    con = _m()._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        _m()._sqlite_ensure_conversation_state_schema(con)
        participants = {str(cid or "").strip() for cid in participant_conversation_ids if str(cid or "").strip()}
        where = ["memorize_chat = 1"]
        where.append("(last_display_segment_at IS NULL OR last_display_segment_at <= ?)")
        params: list[Any] = [run_started_at]
        if participants:
            placeholders = ",".join("?" for _ in participants)
            where.append(f"conversation_id NOT IN ({placeholders})")
            params.extend(sorted(participants))
        con.execute(
            "UPDATE conversations "
            "SET last_display_segment_start_index = NULL, "
            "last_display_segment_end_index = NULL, "
            "last_display_segment_at = NULL "
            f"WHERE {' AND '.join(where)}",
            tuple(params),
        )
        con.commit()
    finally:
        con.close()


def _chat_label_for_prompt(safe: Mapping[str, Any] | None) -> str | None:
    payload = safe or {}
    chat_name = str(payload.get("chat_name") or "").strip()
    chat_type = str(payload.get("chat_type") or "").strip()
    if chat_name and chat_type:
        return f"[{chat_type}][{chat_name}]"
    if chat_name:
        return f"[{chat_name}]"
    return None


def _chat_label_from_history(
    history_rows: list[dict[str, Any]],
    conversation_id: str,
) -> str | None:
    for msg in reversed(history_rows):
        if not isinstance(msg, dict):
            continue
        chat_name = str(msg.get("chat_name") or "").strip()
        if not chat_name:
            continue
        chat_type = str(msg.get("chat_type") or "").strip()
        if not chat_type:
            chat_type = "group" if str(conversation_id or "").startswith("whatsapp:group:") else "dm"
        return f"[{chat_type}][{chat_name}]"
    return None


def _format_cross_tail_for_ai(cross_tail: list[dict[str, Any]], *, soul_id: str) -> str:
    activity_tail = [
        msg for msg in cross_tail
        if str(msg.get("conversation_id") or "").startswith("activity:")
    ]
    chat_tail = [
        msg for msg in cross_tail
        if not str(msg.get("conversation_id") or "").startswith("activity:")
    ]
    activity_text = _message_log.format_merged_history(activity_tail, soul_name=soul_id) if activity_tail else ""
    chat_text = _message_log.format_merged_history(chat_tail, soul_name=soul_id) if chat_tail else ""
    return "\n\n".join(part for part in (activity_text, chat_text) if part).strip()


def _format_all_chat_history_for_ai(
    *,
    current_history: list[dict[str, Any]] | None,
    cross_tail: list[dict[str, Any]] | None,
    conversation_id: str,
    soul_id: str,
    chat_label: str | None = None,
    current_user_text: str | None = None,
    current_user_name: str | None = None,
    self_turn_directive: str | None = None,
    mark_current_chat: bool = True,
) -> str:
    cross_text = _m()._format_cross_tail_for_ai(cross_tail or [], soul_id=soul_id) if cross_tail else ""
    history_rows = current_history or []
    if not history_rows:
        if current_user_text or self_turn_directive:
            return _m()._build_conversations_block(
                history=[],
                cross_conversation_history=cross_text,
                conversation_id=conversation_id,
                chat_label=chat_label,
                soul_name=soul_id,
                current_user_text=current_user_text,
                current_user_name=current_user_name,
                self_turn_directive=self_turn_directive,
                mark_current_chat=mark_current_chat,
            )
        return cross_text
    return _m()._build_conversations_block(
        history=history_rows,
        cross_conversation_history=cross_text,
        conversation_id=conversation_id,
        chat_label=chat_label or _m()._chat_label_from_history(history_rows, conversation_id),
        soul_name=soul_id,
        current_user_text=current_user_text,
        current_user_name=current_user_name,
        self_turn_directive=self_turn_directive,
        mark_current_chat=mark_current_chat,
    )


def _load_cross_tail_for_ai(
    *,
    user_id: str,
    soul_id: str,
    conversation_id: str,
) -> list[dict[str, Any]]:
    db_path = _m()._sqlite_current_path(user_id, soul_id)
    if db_path is None or not db_path.exists():
        return []
    con = _m()._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        _m()._sqlite_ensure_conversation_state_schema(con)
        return [
            *_m()._load_activity_tail_for_ai(con, user_id=user_id, soul_id=soul_id),
            *_m()._load_cross_tail_from_sources(
                con,
                user_id=user_id,
                soul_id=soul_id,
                exclude_conversation_id=conversation_id,
            ),
        ]
    finally:
        con.close()


def _load_cross_tail_with_activities_from_sources(
    con: sqlite3.Connection,
    *,
    user_id: str,
    soul_id: str,
    exclude_conversation_id: str | None = None,
) -> list[dict[str, Any]]:
    return [
        *_m()._load_activity_tail_for_ai(con, user_id=user_id, soul_id=soul_id),
        *_m()._load_cross_tail_from_sources(
            con,
            user_id=user_id,
            soul_id=soul_id,
            exclude_conversation_id=exclude_conversation_id,
        ),
    ]


def _load_cross_memorize_tails_from_sources(
    con: sqlite3.Connection,
    *,
    user_id: str,
    soul_id: str,
    exclude_conversation_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    storage_dir, hermes_home_path, sessions_index_path, state_db_path = _m()._resolve_cross_source_paths()
    excluded_id = str(exclude_conversation_id or "").strip()
    rows = con.execute(
        "SELECT conversation_id, digest_cursor, digest_cursor_source_message_id, digest_cursor_ts, "
        "last_memorize_at, memorize_chat, rolling_summary_cursor_id, "
        "rolling_summary_cursor_source_message_id, rolling_summary_cursor_ts "
        "FROM conversations"
    ).fetchall()
    tails: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cid = str(row["conversation_id"] or "").strip()
        if not cid or cid == excluded_id:
            continue
        try:
            if cid.startswith("activity:"):
                cursor = _effective_digest_cursor_from_row(row)
                tail = _m()._activity_message_rows(
                    con,
                    user_id=user_id,
                    soul_id=soul_id,
                    since_cursor=cursor,
                    recent_fallback_messages=0,
                )
                if tail:
                    tails[cid] = tail
                continue
            memorize_chat = _memorize_chat_from_row(row)
            if memorize_chat:
                cursor = _effective_digest_cursor_from_row(row)
                cursor, min_timestamp, _ = _resolve_source_cursor(
                    cid,
                    cursor,
                    row["digest_cursor_source_message_id"],
                    row["digest_cursor_ts"],
                    rolling=False,
                    hermes_home_path=hermes_home_path,
                )
                tail = _m()._load_tail_for_source_conversation(
                    conversation_id=cid,
                    user_id=user_id,
                    soul_id=soul_id,
                    since_cursor=cursor,
                    recent_fallback_messages=0,
                    storage_dir=storage_dir,
                    hermes_home_path=hermes_home_path,
                    sessions_index_path=sessions_index_path,
                    state_db_path=state_db_path,
                    min_timestamp=min_timestamp,
                )
                _m()._stamp_assistant_display_name(tail, soul_id)
                if not tail:
                    continue
                for i, msg in enumerate(tail):
                    msg["source_conversation_id"] = cid
                    if msg.get("source_conversation_index") is None:
                        msg["source_conversation_index"] = cursor + 1 + i
                    msg["memorize_chat"] = True
                tails[cid] = tail
                continue

            rolling_cursor_id = row["rolling_summary_cursor_id"]
            source_label = _message_log.derive_source_label(cid)
            if source_label.startswith("whatsapp:"):
                active_since = _m()._load_soul_active_since(
                    soul_id,
                    hermes_home_path=hermes_home_path,
                    state_db_path=state_db_path,
                )
                web_source_db_path, reply_prefix = _m()._resolve_whatsapp_web_source_config()
                rolling_cursor_id, checkpoint_floor, _ = _resolve_source_cursor(
                    cid,
                    int(rolling_cursor_id or 0),
                    row["rolling_summary_cursor_source_message_id"],
                    row["rolling_summary_cursor_ts"],
                    rolling=True,
                    hermes_home_path=hermes_home_path,
                )
                assistant_ids = _conversation_sources.load_whatsapp_assistant_source_message_ids(
                    conversation_id=cid,
                    hermes_home=hermes_home_path,
                    sessions_index_path=sessions_index_path,
                    state_db_path=state_db_path,
                )
                tail = _conversation_sources.load_whatsapp_web_source_tail_after_rowid(
                    conversation_id=cid,
                    after_rowid=int(rolling_cursor_id) if rolling_cursor_id is not None else None,
                    soul_id=soul_id,
                    reply_prefix=reply_prefix,
                    hermes_home=hermes_home_path,
                    web_source_db_path=web_source_db_path,
                    min_timestamp=max(
                        (value for value in (active_since, checkpoint_floor) if value is not None),
                        default=None,
                    ),
                    assistant_source_message_ids=assistant_ids,
                )
            elif source_label == "sillytavern":
                tail = _conversation_sources.load_sillytavern_tail(
                    storage_dir=storage_dir,
                    user_id=user_id,
                    soul_id=soul_id,
                    conversation_id=cid,
                    since_cursor=int(rolling_cursor_id) if rolling_cursor_id is not None else -1,
                    recent_fallback_messages=0,
                )
            else:
                continue
            _m()._stamp_assistant_display_name(tail, soul_id)
            if not tail:
                continue
            for msg in tail:
                msg["source_conversation_id"] = cid
                if msg.get("source_conversation_index") is None:
                    raise RuntimeError(
                        f"cross-memorize background/non-memorize tail missing source_conversation_index for {cid}"
                    )
                msg["source_conversation_index"] = int(msg["source_conversation_index"])
                msg["memorize_chat"] = False
            tails[cid] = tail
        except Exception as exc:
            _m().logger.error("cross-memorize source read failed for conversation_id=%s: %s", cid, exc)
            is_lid_gap = "LID↔phone mapping gap" in str(exc)
            if _message_log.derive_source_label(cid).startswith("whatsapp:") and not is_lid_gap:
                raise RuntimeError(f"WhatsApp web_source read failed for {cid}: {exc}") from exc
            continue
    return tails


def _read_background_rolling_summaries_from_conversations(
    con: sqlite3.Connection,
    *,
    exclude_conversation_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    rows = con.execute(
        "SELECT conversation_id, memorize_chat, rolling_summary, rolling_summary_updated_at "
        "FROM conversations"
    ).fetchall()
    excluded_id = str(exclude_conversation_id or "").strip()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = str(row["conversation_id"] or "").strip()
        if not cid or cid == excluded_id:
            continue
        memorize_chat = _memorize_chat_from_row(row)
        if memorize_chat:
            continue
        summary = str(row["rolling_summary"] or "").strip()
        if not summary:
            continue
        out[cid] = {
            "source_conversation_id": cid,
            "source_label": _message_log.derive_source_label(cid),
            "summary": summary,
            "updated_at": row["rolling_summary_updated_at"],
        }
    return out


TURN_HISTORY_WINDOW_MESSAGES = 8


def _turn_history_with_floor(
    history: list[dict[str, Any]],
    resolved_cursor: int,
    min_timestamp: int | None = None,
) -> list[dict[str, Any]]:
    if not history:
        return []
    if min_timestamp is not None:
        history = [
            row for row in history
            if int(row.get("ts_ms") or 0) >= min_timestamp * 1000
        ]
    if resolved_cursor < 0:
        return list(history[-TURN_HISTORY_WINDOW_MESSAGES:])
    return _conversation_sources.slice_tail_with_floor(
        history,
        since_cursor=resolved_cursor,
        recent_fallback_messages=TURN_HISTORY_WINDOW_MESSAGES,
    )


def _source_cursor_checkpoint(
    messages: list[dict[str, Any]],
    *,
    web_source: bool,
) -> dict[str, Any] | None:
    final: dict[str, Any] | None = None
    for msg in messages:
        try:
            idx = int(msg.get("source_conversation_index"))
        except (TypeError, ValueError, OverflowError):
            continue
        if idx < 0:
            continue
        if final is None or idx > final["cursor"]:
            final = {"cursor": idx}
            if web_source:
                source_id = str(msg.get("source_message_id") or "").strip()
                ts_ms = msg.get("ts_ms")
                if not source_id or ts_ms is None:
                    raise RuntimeError(
                        "WhatsApp web_source tail is missing checkpoint metadata; "
                        "repair the conversation cursor with patch_conversation_state_endpoint"
                    )
                final.update(source_message_id=source_id, ts=int(ts_ms) // 1000)
    return final
