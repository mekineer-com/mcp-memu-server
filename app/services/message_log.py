"""Message log — append-only per-conversation message store for cross-conversational context."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

from app.services.turn_contract import format_relative_time_label


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
    now_iso = datetime.now(UTC).isoformat()
    incoming_rows: list[tuple[str, str | None, str]] = []
    for msg in messages:
        role = str(msg.get("role") or "user").strip()
        content = str(msg.get("content") or msg.get("text") or "").strip()
        if isinstance(msg.get("content"), dict):
            content = str(msg["content"].get("text") or "").strip()
        speaker = str(msg.get("name") or msg.get("speaker") or "").strip() or None
        if not content:
            continue
        incoming_rows.append((role, speaker, content))

    if not incoming_rows:
        return 0

    existing_count = con.execute(
        "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()[0]

    # Fast path: cumulative full-history payloads (legacy behavior).
    new_rows_data: list[tuple[str, str | None, str]]
    if len(incoming_rows) > existing_count:
        new_rows_data = incoming_rows[existing_count:]
    else:
        new_rows_data = []

    # Fallback: incremental payloads (latest message(s) only).
    if not new_rows_data:
        recent_rows = con.execute(
            "SELECT role, speaker, content FROM messages "
            "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (conversation_id, len(incoming_rows)),
        ).fetchall()
        existing_tail = list(reversed([
            (str(row["role"] or "").strip(), str(row["speaker"] or "").strip() or None, str(row["content"] or "").strip())
            for row in recent_rows
        ]))
        max_overlap = min(len(existing_tail), len(incoming_rows))
        overlap = 0
        for k in range(max_overlap, 0, -1):
            if existing_tail[-k:] == incoming_rows[:k]:
                overlap = k
                break
        new_rows_data = incoming_rows[overlap:]

    if not new_rows_data:
        return 0

    rows = [
        (conversation_id, role, speaker, content, label, now_iso)
        for role, speaker, content in new_rows_data
    ]

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
DEFAULT_CROSS_RECENT_FALLBACK_MESSAGES = 8


def read_all_tails(
    con: sqlite3.Connection,
    exclude_conversation_id: str | None = None,
    max_messages: int = MAX_CROSS_TAIL_MESSAGES,
    recent_fallback_per_conversation: int = DEFAULT_CROSS_RECENT_FALLBACK_MESSAGES,
) -> list[dict[str, Any]]:
    """Read unmemorized tails from all conversations, merged chronologically.

    Uses each conversation's digest_cursor from the conversations table as the boundary.
    Excludes the current conversation (its history comes fresh from the payload).
    If a conversation has no unmemorized tail, falls back to recent messages so
    cross-conversation context does not disappear solely due to cursor drift or
    full memorization.
    Capped at max_messages most recent to bound prompt size.
    """
    cursor_rows = con.execute(
        "SELECT conversation_id, digest_cursor, last_memorize_at FROM conversations"
    ).fetchall()

    all_messages: list[dict[str, Any]] = []
    for row in cursor_rows:
        cid = str(row["conversation_id"])
        if cid == exclude_conversation_id:
            continue
        cursor = int(row["digest_cursor"] or 0) if row["last_memorize_at"] else -1
        tail = read_tail(con, cid, after_cursor=cursor + 1)
        if tail:
            for msg in tail:
                msg["conversation_id"] = cid
        if not tail and recent_fallback_per_conversation > 0:
            recent = read_recent(con, cid, limit=recent_fallback_per_conversation)
            tail = [
                {
                    "role": msg.get("role"),
                    "speaker": msg.get("name"),
                    "content": msg.get("content"),
                    "source_label": msg.get("source_label"),
                    "received_at": msg.get("received_at"),
                    "conversation_id": cid,
                }
                for msg in recent
            ]
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


def read_recent(
    con: sqlite3.Connection,
    conversation_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Read the most recent `limit` messages for a conversation (memorized + tail)."""
    rows = con.execute(
        "SELECT role, speaker, content, source_label, received_at FROM messages "
        "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
        (conversation_id, limit),
    ).fetchall()
    return list(reversed([
        {
            "role": row["role"],
            "name": row["speaker"],
            "content": row["content"],
            "source_label": row["source_label"],
            "received_at": row["received_at"],
        }
        for row in rows
    ]))


def format_merged_history(
    messages: list[dict[str, Any]],
    current_source: str | None = None,
) -> str:
    """Format merged messages with source labels and date separators for the soul's context."""
    # Build canonical speaker labels from explicit names already present.
    # This keeps labels stable when some payload rows omit name/speaker.
    canonical_speaker: dict[tuple[str, str], str] = {}
    for msg in messages:
        cid = str(msg.get("conversation_id") or "").strip()
        role = str(msg.get("role") or "").strip()
        speaker = str(msg.get("speaker") or "").strip()
        if not cid or not role or not speaker:
            continue
        if speaker.lower() == role.lower():
            continue
        key = (cid, role)
        if key not in canonical_speaker:
            canonical_speaker[key] = speaker

    lines: list[str] = []
    last_time_label: str | None = None
    for msg in messages:
        time_label = format_relative_time_label(msg.get("received_at"))
        if time_label and time_label != last_time_label:
            if lines:
                lines.append("")
            lines.append(f"--- {time_label} ---")
            last_time_label = time_label
        source = msg.get("source_label") or "unknown"
        role = str(msg.get("role") or "").strip()
        cid = str(msg.get("conversation_id") or "").strip()
        speaker = str(msg.get("speaker") or "").strip()
        if not speaker and cid and role:
            speaker = canonical_speaker.get((cid, role), "")
        if not speaker:
            speaker = role or "unknown"
        content = msg.get("content") or ""
        lines.append(f"[{source}] [{speaker}]: {content}")
    return "\n".join(lines)
