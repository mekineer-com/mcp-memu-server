"""Activity-message helpers for self-turn/free-turn recaps."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.db import (
    sqlite_connect,
    sqlite_ensure_conversation_state_schema,
    sqlite_ensure_nonempty,
)
from app.services import conversation_sources
from app.services.state import effective_digest_cursor_from_row


def activity_conversation_id(soul_id: str) -> str:
    return f"activity:dm:{str(soul_id or '').strip() or 'soul'}"


def ensure_activity_messages_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
CREATE TABLE IF NOT EXISTS activity_messages (
    source_conversation_index INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    soul_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    speaker TEXT NOT NULL,
    content TEXT NOT NULL,
    received_at TEXT NOT NULL
)
"""
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_activity_messages_scope "
        "ON activity_messages(user_id, soul_id, source_conversation_index)"
    )
    con.commit()


def activity_message_rows(
    con: sqlite3.Connection,
    *,
    user_id: str,
    soul_id: str,
    since_cursor: int,
    recent_fallback_messages: int,
) -> list[dict[str, Any]]:
    ensure_activity_messages_schema(con)
    activity_cid = activity_conversation_id(soul_id)
    rows = con.execute(
        """
SELECT source_conversation_index, conversation_id, speaker, content, received_at
FROM activity_messages
WHERE user_id = ? AND soul_id = ?
ORDER BY source_conversation_index ASC
""",
        (user_id, soul_id),
    ).fetchall()
    messages = [
        {
            "conversation_id": row["conversation_id"],
            "source_conversation_id": activity_cid,
            "source_conversation_index": int(row["source_conversation_index"]),
            "source_label": "activity",
            "role": "assistant",
            "speaker": row["speaker"],
            "name": row["speaker"],
            "chat_name": row["speaker"],
            "content": row["content"],
            "received_at": row["received_at"],
            "memorize_chat": True,
        }
        for row in rows
    ]
    return conversation_sources.slice_tail_with_floor(
        messages,
        since_cursor=since_cursor,
        recent_fallback_messages=recent_fallback_messages,
    )


def load_activity_tail_for_ai(
    con: sqlite3.Connection,
    *,
    user_id: str,
    soul_id: str,
    recent_fallback_messages: int,
) -> list[dict[str, Any]]:
    sqlite_ensure_conversation_state_schema(con)
    activity_cid = activity_conversation_id(soul_id)
    row = con.execute(
        "SELECT digest_cursor, last_memorize_at FROM conversations WHERE conversation_id = ?",
        (activity_cid,),
    ).fetchone()
    cursor = effective_digest_cursor_from_row(row)
    return activity_message_rows(
        con,
        user_id=user_id,
        soul_id=soul_id,
        since_cursor=cursor,
        recent_fallback_messages=recent_fallback_messages,
    )


def record_activity_message(
    *,
    user_id: str,
    soul_id: str,
    recap: str,
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    logger: logging.Logger,
    platform_name: str = "Claude Code",
    happened_at: datetime | None = None,
) -> bool:
    text = str(recap or "").strip()
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    if not uid or not sid or not text:
        return False
    db_path = sqlite_current_path(uid, sid)
    if db_path is None:
        logger.warning("activity recap skipped: sqlite path unavailable")
        return False
    sqlite_ensure_nonempty(db_path)
    activity_cid = activity_conversation_id(sid)
    message_cid = f"activity:dm:{str(platform_name or '').strip() or 'Claude Code'}"
    now_iso = (happened_at or datetime.now(UTC)).astimezone(UTC).isoformat()
    con = sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        sqlite_ensure_conversation_state_schema(con)
        ensure_activity_messages_schema(con)
        con.execute(
            """
INSERT OR IGNORE INTO conversations (
    conversation_id, soul_id, user_id, memorize_chat, digest_cursor, updated_at
) VALUES (?, ?, ?, 1, 0, ?)
""",
            (activity_cid, sid, uid, now_iso),
        )
        con.execute(
            """
INSERT INTO activity_messages (
    user_id, soul_id, conversation_id, speaker, content, received_at
) VALUES (?, ?, ?, ?, ?, ?)
""",
            (uid, sid, message_cid, sid, text, now_iso),
        )
        con.commit()
        return True
    finally:
        con.close()


def activity_recap_from_contract(contract: dict[str, Any]) -> str:
    recap = str(contract.get("activity_recap") or "").strip()
    if recap:
        return recap
    cache_entry = str(contract.get("cache_entry") or "").strip()
    if cache_entry:
        return cache_entry
    return str(contract.get("rehearsal") or "").strip()
