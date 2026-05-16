from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


def conversation_state_from_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    digest_cursor = int(row["digest_cursor"] or 0)
    prior_context = row["prior_context"]
    return {
        "conversation_id": row["conversation_id"],
        "soul_id": row["soul_id"],
        "user_id": row["user_id"],
        "memorize_chat": bool(int(row["memorize_chat"])) if "memorize_chat" in row.keys() and row["memorize_chat"] is not None else True,
        "digest_cursor": max(0, digest_cursor),
        "prior_context": None if prior_context is None else str(prior_context),
        "pending_episode_ids": normalize_text_list(row["pending_episode_ids"]),
        "last_retrieval_ids": json_from_db(row["last_retrieval_ids"]),
        "last_memorize_at": row["last_memorize_at"],
        "updated_at": row["updated_at"],
        "undo_snapshot": json_from_db(row["undo_snapshot"]),
        "last_background_error": row["last_background_error"] if "last_background_error" in row.keys() else None,
        "last_background_error_at": row["last_background_error_at"] if "last_background_error_at" in row.keys() else None,
    }


def conversation_state_row(con: sqlite3.Connection, conversation_id: str) -> sqlite3.Row | None:
    return con.execute(
        "SELECT conversation_id, soul_id, user_id, memorize_chat, digest_cursor, prior_context, "
        "pending_episode_ids, last_retrieval_ids, last_memorize_at, "
        "updated_at, undo_snapshot, "
        "last_background_error, last_background_error_at "
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
        "prior_context": None,
        "pending_episode_ids": [],
        "last_retrieval_ids": None,
        "last_memorize_at": None,
        "updated_at": None,
        "undo_snapshot": None,
        "last_background_error": None,
        "last_background_error_at": None,
    }


def sqlite_agent_db_paths(sqlite_dir: Path) -> list[Path]:
    if not sqlite_dir.exists():
        return []
    return sorted([p.resolve() for p in sqlite_dir.glob("*.db") if p.is_file()])


def find_conversation_state_across_dbs(
    conversation_id: str,
    sqlite_dir: Path,
) -> tuple[Path | None, dict[str, Any] | None]:
    for db_path in sqlite_agent_db_paths(sqlite_dir):
        con = sqlite_connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            sqlite_ensure_conversation_state_schema(con)
            row = conversation_state_row(con, conversation_id)
            if row is not None:
                return db_path, conversation_state_from_row(row)
        except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
            logger.warning("Skipping unreadable sqlite db during state scan: %s (%s)", db_path, exc)
        finally:
            con.close()
    return None, None


def write_conversation_state(
    conversation_id: str,
    *,
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    sqlite_dir: Path,
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
        db_path, existing_state = find_conversation_state_across_dbs(cid, sqlite_dir)
        if db_path is None:
            raise HTTPException(status_code=400, detail="soul_id is required when creating new conversation state")

    sqlite_ensure_nonempty(db_path)
    con = sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        sqlite_ensure_conversation_state_schema(con)

        if existing_state is None:
            existing_state = conversation_state_from_row(conversation_state_row(con, cid))

        if existing_state is None:
            if scoped_soul is None:
                raise HTTPException(status_code=400, detail="soul_id is required when creating new conversation state")
            seed = conversation_state_empty(cid, scoped_soul, scoped_user)
            seed["updated_at"] = datetime.now(UTC).isoformat()
            con.execute(
                """
INSERT OR IGNORE INTO conversations (
    conversation_id, soul_id, user_id, memorize_chat,
    digest_cursor, prior_context, pending_episode_ids,
    last_retrieval_ids, last_memorize_at,
    updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
                (
                    seed["conversation_id"],
                    seed.get("soul_id"),
                    seed.get("user_id"),
                    1 if seed.get("memorize_chat", True) else 0,
                    int(seed.get("digest_cursor") or 0),
                    seed.get("prior_context"),
                    json_to_db(seed.get("pending_episode_ids") or []),
                    json_to_db(seed.get("last_retrieval_ids")),
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
        append_pending_episode_ids = raw_updates.pop("append_pending_episode_ids", None)
        append_retrieval_ids = None
        append_prior_context_ids = None
        field_updates: dict[str, Any] = {}

        for key, value in raw_updates.items():
            if key in {
                "memorize_chat",
                "digest_cursor",
                "prior_context",
                "pending_episode_ids",
                "last_retrieval_ids",
                "last_memorize_at",
                "undo_snapshot",
                "last_background_error",
                "last_background_error_at",
            }:
                field_updates[key] = value

        if append_prior_context_ids is not None:
            base_pc = field_updates.get(
                "prior_context_ids_since_consolidation", existing_state.get("prior_context_ids_since_consolidation")
            )
            field_updates["prior_context_ids_since_consolidation"] = merge_unique_text_lists(
                base_pc,
                append_prior_context_ids,
            )

        if append_retrieval_ids is not None:
            base_ids = field_updates.get(
                "retrieval_ids_since_consolidation", existing_state.get("retrieval_ids_since_consolidation")
            )
            field_updates["retrieval_ids_since_consolidation"] = merge_unique_text_lists(
                base_ids,
                append_retrieval_ids,
            )

        if append_pending_episode_ids is not None:
            base_pending = field_updates.get(
                "pending_episode_ids", existing_state.get("pending_episode_ids")
            )
            field_updates["pending_episode_ids"] = merge_unique_text_lists(
                base_pending,
                append_pending_episode_ids,
            )

        if "digest_cursor" in field_updates:
            try:
                field_updates["digest_cursor"] = max(0, int(field_updates.get("digest_cursor") or 0))
            except (TypeError, ValueError, OverflowError) as exc:
                raise HTTPException(status_code=400, detail="digest_cursor must be an integer") from exc
        if "memorize_chat" in field_updates:
            field_updates["memorize_chat"] = 1 if bool(field_updates.get("memorize_chat")) else 0
        if "last_memorize_at" in field_updates:
            raw_last = field_updates.get("last_memorize_at")
            field_updates["last_memorize_at"] = None if raw_last is None else (str(raw_last).strip() or None)
        if "last_consolidation_at" in field_updates:
            raw_last_consolidation = field_updates.get("last_consolidation_at")
            field_updates["last_consolidation_at"] = (
                None if raw_last_consolidation is None else (str(raw_last_consolidation).strip() or None)
            )
        if "prior_context" in field_updates:
            raw_prior_context = field_updates.get("prior_context")
            field_updates["prior_context"] = None if raw_prior_context is None else str(raw_prior_context)
        if "all_categories_summary" in field_updates:
            raw_acs = field_updates.get("all_categories_summary")
            field_updates["all_categories_summary"] = None if raw_acs is None else (str(raw_acs) or None)
        if "intentions_active" in field_updates:
            field_updates["intentions_active"] = normalize_intentions_stack(field_updates.get("intentions_active"))
        if "memory_cache" in field_updates:
            field_updates["memory_cache"] = normalize_memory_cache(field_updates.get("memory_cache"))
        if "pending_episode_ids" in field_updates:
            field_updates["pending_episode_ids"] = normalize_text_list(field_updates["pending_episode_ids"])

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
                if key in {"pending_episode_ids", "last_retrieval_ids", "undo_snapshot"}:
                    params.append(json_to_db(value))
                elif key == "memorize_chat":
                    params.append(1 if bool(value) else 0)
                elif key == "digest_cursor":
                    params.append(int(value or 0))
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
