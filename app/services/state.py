from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.db import (
    json_from_db,
    json_to_db,
    merge_unique_text_lists,
    normalize_text_list,
    sqlite_connect,
    sqlite_ensure_conversation_state_schema,
    sqlite_ensure_nonempty,
)
from app.services import soul_state as _soul_state


_MISSING = object()


def _row_value(row: Any, key: str) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return _MISSING


def effective_digest_cursor_from_row(row: Any) -> int:
    if row is None:
        return -1
    last_memorize_at = _row_value(row, "last_memorize_at")
    if last_memorize_at is _MISSING:
        raise KeyError("last_memorize_at")
    if not last_memorize_at:
        return -1
    digest_cursor = _row_value(row, "digest_cursor")
    if digest_cursor is _MISSING:
        raise KeyError("digest_cursor")
    return int(digest_cursor or 0)


def memorize_chat_from_row(row: Any) -> bool:
    if row is None:
        return True
    raw = _row_value(row, "memorize_chat")
    if raw is _MISSING:
        raise KeyError("memorize_chat")
    return True if raw is None else bool(int(raw))


def conversation_state_from_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    digest_cursor = int(row["digest_cursor"] or 0)
    prior_context = row["prior_context"]
    return {
        "conversation_id": row["conversation_id"],
        "soul_id": row["soul_id"],
        "user_id": row["user_id"],
        "memorize_chat": memorize_chat_from_row(row),
        "digest_cursor": max(0, digest_cursor),
        "digest_cursor_source_message_id": (
            str(row["digest_cursor_source_message_id"] or "").strip() or None
        ),
        "digest_cursor_ts": (
            int(row["digest_cursor_ts"]) if row["digest_cursor_ts"] is not None else None
        ),
        "rolling_summary": (
            (str(row["rolling_summary"] or "").strip() or None)
            if "rolling_summary" in row.keys()
            else None
        ),
        "rolling_summary_cursor_id": (
            int(row["rolling_summary_cursor_id"])
            if "rolling_summary_cursor_id" in row.keys() and row["rolling_summary_cursor_id"] is not None
            else None
        ),
        "rolling_summary_cursor_source_message_id": (
            str(row["rolling_summary_cursor_source_message_id"] or "").strip() or None
        ),
        "rolling_summary_cursor_ts": (
            int(row["rolling_summary_cursor_ts"])
            if row["rolling_summary_cursor_ts"] is not None
            else None
        ),
        "rolling_summary_updated_at": row["rolling_summary_updated_at"] if "rolling_summary_updated_at" in row.keys() else None,
        "prior_context": None if prior_context is None else str(prior_context),
        "pending_segment_ids": normalize_text_list(row["pending_segment_ids"]),
        "last_memorize_at": row["last_memorize_at"],
        "last_display_segment_start_index": (
            int(row["last_display_segment_start_index"])
            if "last_display_segment_start_index" in row.keys() and row["last_display_segment_start_index"] is not None
            else None
        ),
        "last_display_segment_end_index": (
            int(row["last_display_segment_end_index"])
            if "last_display_segment_end_index" in row.keys() and row["last_display_segment_end_index"] is not None
            else None
        ),
        "last_display_segment_at": row["last_display_segment_at"] if "last_display_segment_at" in row.keys() else None,
        "updated_at": row["updated_at"],
        "undo_snapshot": json_from_db(row["undo_snapshot"]),
        "last_background_error": row["last_background_error"] if "last_background_error" in row.keys() else None,
        "last_background_error_at": row["last_background_error_at"] if "last_background_error_at" in row.keys() else None,
        "last_consolidation_error": row["last_consolidation_error"] if "last_consolidation_error" in row.keys() else None,
        "last_consolidation_error_at": row["last_consolidation_error_at"] if "last_consolidation_error_at" in row.keys() else None,
        "atomic_session_started_at": row["atomic_session_started_at"] if "atomic_session_started_at" in row.keys() else None,
        "atomic_session_ended_at": row["atomic_session_ended_at"] if "atomic_session_ended_at" in row.keys() else None,
    }


def conversation_state_row(con: sqlite3.Connection, conversation_id: str) -> sqlite3.Row | None:
    return con.execute(
        "SELECT conversation_id, soul_id, user_id, memorize_chat, digest_cursor, "
        "digest_cursor_source_message_id, digest_cursor_ts, "
        "rolling_summary, rolling_summary_cursor_id, "
        "rolling_summary_cursor_source_message_id, rolling_summary_cursor_ts, "
        "rolling_summary_updated_at, prior_context, "
        "pending_segment_ids, last_memorize_at, "
        "last_display_segment_start_index, last_display_segment_end_index, last_display_segment_at, "
        "updated_at, undo_snapshot, "
        "last_background_error, last_background_error_at, "
        "last_consolidation_error, last_consolidation_error_at, "
        "atomic_session_started_at, atomic_session_ended_at "
        "FROM conversations WHERE conversation_id = ? LIMIT 1",
        (conversation_id,),
    ).fetchone()


def conversation_state_empty(
    conversation_id: str,
    soul_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "soul_id": soul_id,
        "user_id": user_id,
        "memorize_chat": True,
        "digest_cursor": 0,
        "digest_cursor_source_message_id": None,
        "digest_cursor_ts": None,
        "rolling_summary": None,
        "rolling_summary_cursor_id": None,
        "rolling_summary_cursor_source_message_id": None,
        "rolling_summary_cursor_ts": None,
        "rolling_summary_updated_at": None,
        "prior_context": None,
        "pending_segment_ids": [],
        "last_memorize_at": None,
        "last_display_segment_start_index": None,
        "last_display_segment_end_index": None,
        "last_display_segment_at": None,
        "updated_at": None,
        "undo_snapshot": None,
        "last_background_error": None,
        "last_background_error_at": None,
        "last_consolidation_error": None,
        "last_consolidation_error_at": None,
        "atomic_session_started_at": None,
        "atomic_session_ended_at": None,
    }


def write_conversation_state(
    conversation_id: str,
    *,
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    soul_id: str | None = None,
    user_id: str | None = None,
    updates: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    cid = str(conversation_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="conversation_id is required")

    scoped_soul = str(soul_id or "").strip() or None
    scoped_user = str(user_id or "").strip() or None

    db_path: Path | None = sqlite_current_path(scoped_user, scoped_soul) if scoped_soul else None
    existing_state: dict[str, Any] | None = None
    if db_path is None:
        raise HTTPException(status_code=400, detail="soul_id is required")

    sqlite_ensure_nonempty(db_path)
    con = sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        sqlite_ensure_conversation_state_schema(con)

        if existing_state is None:
            existing_state = conversation_state_from_row(conversation_state_row(con, cid))

        if existing_state is None:
            if scoped_soul is None:
                raise HTTPException(status_code=400, detail="soul_id is required")
            seed = conversation_state_empty(cid, scoped_soul, scoped_user)
            seed["updated_at"] = datetime.now(UTC).isoformat()
            con.execute(
                """
INSERT OR IGNORE INTO conversations (
    conversation_id, soul_id, user_id, memorize_chat,
    digest_cursor, rolling_summary, rolling_summary_cursor_id, rolling_summary_updated_at,
    prior_context, pending_segment_ids,
    last_memorize_at,
    updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
                (
                    seed["conversation_id"],
                    seed.get("soul_id"),
                    seed.get("user_id"),
                    1 if seed.get("memorize_chat", True) else 0,
                    int(seed.get("digest_cursor") or 0),
                    seed.get("rolling_summary"),
                    seed.get("rolling_summary_cursor_id"),
                    seed.get("rolling_summary_updated_at"),
                    seed.get("prior_context"),
                    json_to_db(seed.get("pending_segment_ids") or []),
                    seed.get("last_memorize_at"),
                    seed.get("updated_at"),
                ),
            )
            con.commit()
            existing_state = conversation_state_from_row(conversation_state_row(con, cid)) or seed

        raw_updates = dict(updates) if updates else {}
        soul_updates = {k: raw_updates.pop(k) for k in list(raw_updates) if k in _soul_state._VALID_FIELDS}
        for append_key, field in (
            ("append_retrieval_ids_since_consolidation", "retrieval_ids_since_consolidation"),
            ("append_prior_context_ids_since_consolidation", "prior_context_ids_since_consolidation"),
        ):
            appended = raw_updates.pop(append_key, None)
            if appended is not None:
                current = _soul_state.read(con)
                soul_updates[field] = merge_unique_text_lists(current.get(field), appended)
        if soul_updates:
            _soul_state.write(con, soul_updates)
        append_pending_segment_ids = raw_updates.pop("append_pending_segment_ids", None)
        field_updates: dict[str, Any] = {}

        for cursor, source_id, source_ts in (
            ("digest_cursor", "digest_cursor_source_message_id", "digest_cursor_ts"),
            (
                "rolling_summary_cursor_id",
                "rolling_summary_cursor_source_message_id",
                "rolling_summary_cursor_ts",
            ),
        ):
            has_cursor = cursor in raw_updates
            has_source_id = source_id in raw_updates
            has_source_ts = source_ts in raw_updates
            if (has_source_id or has_source_ts) and not has_cursor:
                raise HTTPException(status_code=400, detail=f"{cursor} is required with its checkpoint")
            if not has_cursor:
                continue
            if has_source_id != has_source_ts:
                raise HTTPException(status_code=400, detail=f"{cursor} checkpoint must be complete")
            if has_source_id and (
                (raw_updates[source_id] is None) != (raw_updates[source_ts] is None)
            ):
                raise HTTPException(status_code=400, detail=f"{cursor} checkpoint must be complete")
            if not has_source_id:
                raw_updates[source_id] = None
                raw_updates[source_ts] = None

        for key, value in raw_updates.items():
            if key in {
                "memorize_chat",
                "digest_cursor",
                "digest_cursor_source_message_id",
                "digest_cursor_ts",
                "rolling_summary",
                "rolling_summary_cursor_id",
                "rolling_summary_cursor_source_message_id",
                "rolling_summary_cursor_ts",
                "rolling_summary_updated_at",
                "prior_context",
                "pending_segment_ids",
                "last_memorize_at",
                "last_display_segment_start_index",
                "last_display_segment_end_index",
                "last_display_segment_at",
                "undo_snapshot",
                "last_background_error",
                "last_background_error_at",
                "last_consolidation_error",
                "last_consolidation_error_at",
                "atomic_session_started_at",
                "atomic_session_ended_at",
            }:
                field_updates[key] = value

        for cursor, source_id, source_ts in (
            ("digest_cursor", "digest_cursor_source_message_id", "digest_cursor_ts"),
            (
                "rolling_summary_cursor_id",
                "rolling_summary_cursor_source_message_id",
                "rolling_summary_cursor_ts",
            ),
        ):
            if cursor not in field_updates or field_updates[source_id] is None:
                continue
            source_message_id = str(field_updates[source_id] or "").strip()
            if not source_message_id or field_updates[source_ts] is None:
                raise HTTPException(status_code=400, detail=f"{cursor} checkpoint must be complete")
            try:
                timestamp = int(field_updates[source_ts])
            except (TypeError, ValueError, OverflowError) as exc:
                raise HTTPException(status_code=400, detail=f"{source_ts} must be an integer") from exc
            field_updates[source_id] = source_message_id
            field_updates[source_ts] = timestamp

        if append_pending_segment_ids is not None:
            base_pending = field_updates.get(
                "pending_segment_ids", existing_state.get("pending_segment_ids")
            )
            field_updates["pending_segment_ids"] = merge_unique_text_lists(
                base_pending,
                append_pending_segment_ids,
            )
        remove_pending_segment_ids = raw_updates.pop("remove_pending_segment_ids", None)
        if remove_pending_segment_ids is not None:
            base_pending = field_updates.get(
                "pending_segment_ids", existing_state.get("pending_segment_ids")
            )
            removed = set(normalize_text_list(remove_pending_segment_ids))
            field_updates["pending_segment_ids"] = [
                segment_id
                for segment_id in normalize_text_list(base_pending)
                if segment_id not in removed
            ]

        if "digest_cursor" in field_updates:
            try:
                field_updates["digest_cursor"] = max(0, int(field_updates.get("digest_cursor") or 0))
            except (TypeError, ValueError, OverflowError) as exc:
                raise HTTPException(status_code=400, detail="digest_cursor must be an integer") from exc
        if "rolling_summary_cursor_id" in field_updates:
            raw_cursor_id = field_updates.get("rolling_summary_cursor_id")
            if raw_cursor_id is None:
                field_updates["rolling_summary_cursor_id"] = None
            else:
                try:
                    field_updates["rolling_summary_cursor_id"] = max(0, int(raw_cursor_id))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise HTTPException(status_code=400, detail="rolling_summary_cursor_id must be an integer") from exc
        if "rolling_summary" in field_updates:
            raw_summary = field_updates.get("rolling_summary")
            field_updates["rolling_summary"] = None if raw_summary is None else (str(raw_summary).strip() or None)
        if "rolling_summary_updated_at" in field_updates:
            raw_rs_at = field_updates.get("rolling_summary_updated_at")
            field_updates["rolling_summary_updated_at"] = None if raw_rs_at is None else (str(raw_rs_at).strip() or None)
        if "memorize_chat" in field_updates:
            field_updates["memorize_chat"] = 1 if bool(field_updates.get("memorize_chat")) else 0
        if "last_memorize_at" in field_updates:
            raw_last = field_updates.get("last_memorize_at")
            field_updates["last_memorize_at"] = None if raw_last is None else (str(raw_last).strip() or None)
        for key in ("last_display_segment_start_index", "last_display_segment_end_index"):
            if key not in field_updates:
                continue
            raw_index = field_updates.get(key)
            if raw_index is None:
                field_updates[key] = None
                continue
            try:
                field_updates[key] = max(0, int(raw_index))
            except (TypeError, ValueError, OverflowError) as exc:
                raise HTTPException(status_code=400, detail=f"{key} must be an integer") from exc
        if "last_display_segment_at" in field_updates:
            raw_display_at = field_updates.get("last_display_segment_at")
            field_updates["last_display_segment_at"] = None if raw_display_at is None else (str(raw_display_at).strip() or None)
        if "prior_context" in field_updates:
            raw_prior_context = field_updates.get("prior_context")
            field_updates["prior_context"] = None if raw_prior_context is None else str(raw_prior_context)
        if "pending_segment_ids" in field_updates:
            field_updates["pending_segment_ids"] = normalize_text_list(field_updates["pending_segment_ids"])

        if scoped_soul is not None:
            field_updates["soul_id"] = scoped_soul
        if scoped_user is not None:
            field_updates["user_id"] = scoped_user

        if field_updates:
            field_updates["updated_at"] = datetime.now(UTC).isoformat()
            assignments: list[str] = []
            params: list[Any] = []
            for key, value in field_updates.items():
                assignments.append(f"{key} = ?")
                if key in {"pending_segment_ids", "undo_snapshot"}:
                    params.append(json_to_db(value))
                elif key == "memorize_chat":
                    params.append(1 if bool(value) else 0)
                elif key == "digest_cursor":
                    params.append(int(value or 0))
                elif key in {"last_display_segment_start_index", "last_display_segment_end_index"}:
                    params.append(None if value is None else int(value or 0))
                else:
                    params.append(value)
            params.append(cid)
            con.execute(
                f"UPDATE conversations SET {', '.join(assignments)} WHERE conversation_id = ?",
                tuple(params),
            )
        if soul_updates or field_updates:
            con.commit()

        state_out = conversation_state_from_row(conversation_state_row(con, cid))
        if state_out is None:
            state_out = conversation_state_empty(cid, scoped_soul, scoped_user)
        state_out.update(_soul_state.read(con))
        return state_out, db_path
    finally:
        con.close()
