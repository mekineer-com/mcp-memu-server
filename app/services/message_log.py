"""Message log — append-only per-conversation message store for cross-conversational context."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any


def derive_source_label(conversation_id: str) -> str:
    cid = str(conversation_id or "").strip()
    if cid.startswith("whatsapp:"):
        suffix = cid.split(":", 1)[1] if ":" in cid else ""
        if "@g.us" in suffix:
            return "whatsapp:group"
        return "whatsapp:dm"
    if cid.startswith("sillytavern"):
        return "sillytavern"
    if cid.startswith("cron:"):
        return "cron"
    return cid.split(":")[0] if ":" in cid else "unknown"


def append_messages(
    con: sqlite3.Connection,
    conversation_id: str,
    messages: list[dict[str, Any]],
    source_label: str | None = None,
) -> int:
    """Append messages not already stored. Returns count of new messages appended."""
    if not messages:
        return 0

    label = source_label or derive_source_label(conversation_id)
    existing_count = con.execute(
        "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()[0]

    new_messages = messages[existing_count:]
    if not new_messages:
        return 0

    now_iso = datetime.now(UTC).isoformat()
    rows = []
    for msg in new_messages:
        role = str(msg.get("role") or "user").strip()
        content = str(msg.get("content") or msg.get("text") or "").strip()
        if isinstance(msg.get("content"), dict):
            content = str(msg["content"].get("text") or "").strip()
        speaker = str(msg.get("name") or msg.get("speaker") or "").strip() or None
        if not content:
            continue
        rows.append((conversation_id, role, speaker, content, label, now_iso))

    if not rows:
        return 0

    con.executemany(
        "INSERT INTO messages (conversation_id, role, speaker, content, source_label, received_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def read_tail(
    con: sqlite3.Connection,
    conversation_id: str,
    after_cursor: int = 0,
) -> list[dict[str, Any]]:
    """Read messages for a conversation after the given cursor (message count offset)."""
    rows = con.execute(
        "SELECT role, speaker, content, source_label, received_at FROM messages "
        "WHERE conversation_id = ? ORDER BY id ASC LIMIT -1 OFFSET ?",
        (conversation_id, after_cursor),
    ).fetchall()
    return [
        {
            "role": row["role"],
            "speaker": row["speaker"],
            "content": row["content"],
            "source_label": row["source_label"],
            "received_at": row["received_at"],
        }
        for row in rows
    ]


MAX_CROSS_TAIL_MESSAGES = 50


def read_all_tails(
    con: sqlite3.Connection,
    exclude_conversation_id: str | None = None,
    max_messages: int = MAX_CROSS_TAIL_MESSAGES,
) -> list[dict[str, Any]]:
    """Read unmemorized tails from all conversations, merged chronologically.

    Uses each conversation's digest_cursor from the conversations table as the boundary.
    Excludes the current conversation (its history comes fresh from the payload).
    Capped at max_messages most recent to bound prompt size.
    """
    cursor_rows = con.execute(
        "SELECT conversation_id, digest_cursor FROM conversations"
    ).fetchall()

    all_messages: list[dict[str, Any]] = []
    for row in cursor_rows:
        cid = str(row["conversation_id"])
        if cid == exclude_conversation_id:
            continue
        cursor = int(row["digest_cursor"] or 0)
        tail = read_tail(con, cid, after_cursor=cursor + 1)
        all_messages.extend(tail)

    all_messages.sort(key=lambda m: m.get("received_at") or "")
    return all_messages[-max_messages:]


def read_all_tails_for_memorize(
    con: sqlite3.Connection,
    exclude_conversation_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Read unmemorized tails from all conversations, keyed by conversation_id.

    Each message carries source_label and its position within its conversation.
    No cap — memorize needs all unmemorized messages.
    """
    cursor_rows = con.execute(
        "SELECT conversation_id, digest_cursor, last_memorize_at FROM conversations"
    ).fetchall()

    result: dict[str, list[dict[str, Any]]] = {}
    for row in cursor_rows:
        cid = str(row["conversation_id"])
        if cid == exclude_conversation_id:
            continue
        cursor = int(row["digest_cursor"] or 0) if row["last_memorize_at"] else -1
        tail = read_tail(con, cid, after_cursor=cursor + 1)
        if tail:
            for i, msg in enumerate(tail):
                msg["source_conversation_id"] = cid
                msg["source_conversation_index"] = cursor + 1 + i
            result[cid] = tail
    return result


def format_merged_history(
    messages: list[dict[str, Any]],
    current_source: str | None = None,
) -> str:
    """Format merged messages with source labels for the soul's context."""
    lines: list[str] = []
    for msg in messages:
        source = msg.get("source_label") or "unknown"
        speaker = msg.get("speaker") or msg.get("role") or "unknown"
        content = msg.get("content") or ""
        lines.append(f"[{source}] [{speaker}]: {content}")
    return "\n".join(lines)
