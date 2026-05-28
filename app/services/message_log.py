"""Message log — append-only per-conversation message store for cross-conversational context."""

from __future__ import annotations

import json
import os
import re
import sqlite3

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.services.turn_contract import format_relative_time_label

_SHARED_GROUP_PREFIX_RE = re.compile(r"^\[([^\]]+)\]\s+(.+)$")


def _parse_shared_group_sender_prefix(content: str) -> tuple[str, str] | None:
    match = _SHARED_GROUP_PREFIX_RE.match(content)
    if not match:
        return None
    sender = str(match.group(1) or "").strip()
    message = str(match.group(2) or "").strip()
    if not sender or not message:
        return None
    return sender, message


def _looks_like_shared_group_conversation(conversation_id: str) -> bool:
    cid = str(conversation_id or "").strip().lower()
    return ":group:" in cid or cid.endswith(":group")


def _canonical_conversation_id(conversation_id: str) -> str:
    cid = str(conversation_id or "").strip()
    prefix = "whatsapp:dm:"
    if not cid.startswith(prefix):
        return cid
    identifier = cid[len(prefix):]
    aliases = _expand_whatsapp_alias_identifiers(identifier)
    if not aliases:
        normalized = _normalize_whatsapp_identifier(identifier)
        return f"{prefix}{normalized}" if normalized else cid
    canonical = sorted(aliases, key=lambda item: (len(item), item))[0]
    return f"{prefix}{canonical}"


def _normalize_row_for_overlap(
    conversation_id: str,
    role: str,
    speaker: str | None,
    content: str,
) -> tuple[str, str | None, str]:
    norm_role = str(role or "").strip()
    norm_speaker = str(speaker or "").strip() or None
    norm_content = str(content or "").strip()
    if norm_role == "user" and _looks_like_shared_group_conversation(conversation_id):
        parsed = _parse_shared_group_sender_prefix(norm_content)
        if parsed is not None:
            parsed_speaker, parsed_content = parsed
            norm_content = parsed_content
            norm_speaker = parsed_speaker
    return norm_role, norm_speaker, norm_content


def derive_source_label(conversation_id: str) -> str:
    cid = str(conversation_id or "").strip()
    if cid.startswith("whatsapp:"):
        suffix = cid.split(":", 1)[1] if ":" in cid else ""
        if "@g.us" in suffix:
            return "whatsapp:group"
        return "whatsapp:dm"
    if cid.startswith(("sillytavern", "integrity:", "chat:")):
        return "sillytavern"
    if cid.startswith("cron:"):
        return "cron"
    return cid.split(":")[0] if ":" in cid else "unknown"


def append_messages(
    con: sqlite3.Connection,
    conversation_id: str,
    messages: list[dict[str, Any]],
    source_label: str | None = None,
    chat_name: str | None = None,
) -> int:
    """Append messages not already stored. Returns count of new messages appended."""
    if not messages:
        return 0

    label = source_label or derive_source_label(conversation_id)
    chat_name_value = str(chat_name or "").strip() or None
    now_iso = datetime.now(UTC).isoformat()
    incoming_rows: list[tuple[str, str | None, str]] = []
    for msg in messages:
        role = str(msg.get("role") or "user").strip()
        content = str(msg.get("content") or msg.get("text") or "").strip()
        if isinstance(msg.get("content"), dict):
            content = str(msg["content"].get("text") or "").strip()
        speaker = str(msg.get("name") or msg.get("speaker") or "").strip() or None
        role, speaker, content = _normalize_row_for_overlap(
            conversation_id, role, speaker, content
        )
        if not content:
            continue
        incoming_rows.append((role, speaker, content))

    if not incoming_rows:
        return 0

    new_rows_data: list[tuple[str, str | None, str]] = []

    # Policy: exact tail overlap is treated as duplicate noise and suppressed.
    # This is intentional for retry/replay-prone paths (including listen_only
    # message ingestion) where duplicate rows hurt context quality.
    recent_rows = con.execute(
        "SELECT role, speaker, content FROM messages "
        "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
        (conversation_id, len(incoming_rows)),
    ).fetchall()
    existing_tail = list(reversed([
        _normalize_row_for_overlap(
            conversation_id,
            str(row["role"] or "").strip(),
            str(row["speaker"] or "").strip() or None,
            str(row["content"] or "").strip(),
        )
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
        (conversation_id, role, speaker, chat_name_value, content, label, now_iso)
        for role, speaker, content in new_rows_data
    ]

    con.executemany(
        "INSERT INTO messages (conversation_id, role, speaker, chat_name, content, source_label, received_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
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
        "SELECT role, speaker, chat_name, content, source_label, received_at FROM messages "
        "WHERE conversation_id = ? ORDER BY id ASC LIMIT -1 OFFSET ?",
        (conversation_id, after_cursor),
    ).fetchall()
    return [
        {
            "role": row["role"],
            "speaker": row["speaker"],
            "chat_name": row["chat_name"],
            "content": row["content"],
            "source_label": row["source_label"],
            "received_at": row["received_at"],
        }
        for row in rows
    ]


def read_tail_after_message_id(
    con: sqlite3.Connection,
    conversation_id: str,
    after_message_id: int | None = None,
) -> list[dict[str, Any]]:
    """Read messages for a conversation after a message row-id boundary."""
    cursor_id = int(after_message_id or 0)
    rows = con.execute(
        "SELECT id, role, speaker, chat_name, content, source_label, received_at FROM messages "
        "WHERE conversation_id = ? AND id > ? ORDER BY id ASC",
        (conversation_id, cursor_id),
    ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "role": row["role"],
            "speaker": row["speaker"],
            "chat_name": row["chat_name"],
            "content": row["content"],
            "source_label": row["source_label"],
            "received_at": row["received_at"],
        }
        for row in rows
    ]


def last_message_row_id(con: sqlite3.Connection, conversation_id: str) -> int | None:
    row = con.execute(
        "SELECT id FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT 1",
        (conversation_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        return int(row["id"])
    except (TypeError, ValueError):
        return None


def delete_messages_through_id(
    con: sqlite3.Connection,
    conversation_id: str,
    max_message_id: int | None,
) -> int:
    if max_message_id is None:
        return 0
    cursor_id = int(max_message_id)
    cur = con.execute(
        "DELETE FROM messages WHERE conversation_id = ? AND id <= ?",
        (conversation_id, cursor_id),
    )
    return int(cur.rowcount or 0)


MAX_CROSS_TAIL_MESSAGES = 0
DEFAULT_CROSS_RECENT_FALLBACK_MESSAGES = 8


def _normalize_whatsapp_identifier(value: str) -> str:
    normalized = (
        str(value or "")
        .strip()
        .replace("+", "", 1)
        .split(":", 1)[0]
        .split("@", 1)[0]
    )
    # Conversation IDs flow into lid-mapping file lookups. Reject path-like
    # values so alias expansion cannot traverse outside the session dir.
    if not normalized or "/" in normalized or "\\" in normalized:
        return ""
    if normalized in {".", ".."}:
        return ""
    return normalized


def _read_lid_mapping_value(path: Path) -> str:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(raw, str):
        return ""
    return _normalize_whatsapp_identifier(raw)


def _load_whatsapp_creds_aliases(session_dir: Path) -> dict[str, str]:
    creds_path = session_dir / "creds.json"
    if not creds_path.exists():
        return {}
    try:
        parsed = json.loads(creds_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    me = parsed.get("me") if isinstance(parsed, dict) else None
    if not isinstance(me, dict):
        return {}
    phone_id = _normalize_whatsapp_identifier(me.get("id"))
    lid_id = _normalize_whatsapp_identifier(me.get("lid"))
    if not phone_id or not lid_id or phone_id == lid_id:
        return {}
    return {
        phone_id: lid_id,
        lid_id: phone_id,
    }


def _expand_whatsapp_alias_identifiers(identifier: str) -> set[str]:
    normalized = _normalize_whatsapp_identifier(identifier)
    if not normalized:
        return set()

    hermes_home = Path(os.getenv("HERMES_HOME") or "~/.hermes").expanduser().resolve()
    session_dir = hermes_home / "whatsapp" / "session"
    creds_aliases = _load_whatsapp_creds_aliases(session_dir)
    resolved: set[str] = set()
    queue = [normalized]
    while queue:
        current = queue.pop(0)
        if not current or current in resolved:
            continue
        resolved.add(current)

        for suffix in ("", "_reverse"):
            mapping_path = session_dir / f"lid-mapping-{current}{suffix}.json"
            if not mapping_path.exists():
                continue
            mapped = _read_lid_mapping_value(mapping_path)
            if mapped and mapped not in resolved:
                queue.append(mapped)

        mapped_from_creds = creds_aliases.get(current, "")
        if mapped_from_creds and mapped_from_creds not in resolved:
            queue.append(mapped_from_creds)

    return resolved


def conversation_aliases(conversation_id: str) -> list[str]:
    cid = str(conversation_id or "").strip()
    if not cid:
        return []
    prefix = "whatsapp:dm:"
    if not cid.startswith(prefix):
        return [cid]
    identifier = cid[len(prefix):]
    aliases = _expand_whatsapp_alias_identifiers(identifier) or {_normalize_whatsapp_identifier(identifier)}
    out = {cid}
    for alias in aliases:
        if alias:
            out.add(f"{prefix}{alias}")
    return sorted(out, key=lambda item: (len(item), item))


def read_all_tails(
    con: sqlite3.Connection,
    exclude_conversation_id: str | None = None,
    max_messages: int = MAX_CROSS_TAIL_MESSAGES,
    recent_fallback_per_conversation: int = DEFAULT_CROSS_RECENT_FALLBACK_MESSAGES,
) -> list[dict[str, Any]]:
    """Read unmemorized tails from all conversations, merged chronologically.

    Uses each conversation's digest_cursor from the conversations table as the boundary.
    Excludes the current conversation (its history comes fresh from the payload).
    If a conversation has fewer than recent_fallback_per_conversation unmemorized
    messages, backfills from recent messages to that floor so context continuity
    is preserved after memorize runs.
    If max_messages > 0, returns only the most recent max_messages entries.
    Default behavior is uncapped so full unmemorized tails are preserved.
    """
    excluded_ids = set(conversation_aliases(exclude_conversation_id or ""))
    cursor_rows = con.execute(
        "SELECT conversation_id, digest_cursor, last_memorize_at FROM conversations"
    ).fetchall()

    all_messages: list[dict[str, Any]] = []
    for row in cursor_rows:
        cid = str(row["conversation_id"])
        if cid in excluded_ids:
            continue
        canonical_cid = _canonical_conversation_id(cid)
        cursor = int(row["digest_cursor"] or 0) if row["last_memorize_at"] else -1
        tail = read_tail(con, cid, after_cursor=cursor + 1)
        if tail:
            for msg in tail:
                msg["conversation_id"] = canonical_cid
        if len(tail) < recent_fallback_per_conversation and recent_fallback_per_conversation > 0:
            recent = read_recent(con, cid, limit=recent_fallback_per_conversation)
            if len(recent) > len(tail):
                tail = [
                    {
                        "role": msg.get("role"),
                        "speaker": msg.get("name"),
                        "chat_name": msg.get("chat_name"),
                        "content": msg.get("content"),
                        "source_label": msg.get("source_label"),
                        "received_at": msg.get("received_at"),
                        "conversation_id": canonical_cid,
                    }
                    for msg in recent
                ]
        all_messages.extend(tail)

    all_messages.sort(key=lambda m: m.get("received_at") or "")
    if max_messages > 0:
        return all_messages[-max_messages:]
    return all_messages


def read_all_tails_for_memorize(
    con: sqlite3.Connection,
    exclude_conversation_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Read unmemorized tails from all conversations, keyed by conversation_id.

    Each message carries source_label and its position within its conversation.
    No cap — memorize needs all unmemorized messages.
    """
    cursor_rows = con.execute(
        "SELECT conversation_id, digest_cursor, last_memorize_at, memorize_chat, rolling_summary_cursor_id "
        "FROM conversations"
    ).fetchall()

    result: dict[str, list[dict[str, Any]]] = {}
    for row in cursor_rows:
        cid = str(row["conversation_id"])
        if cid == exclude_conversation_id:
            continue
        cursor = int(row["digest_cursor"] or 0) if row["last_memorize_at"] else -1
        memorize_chat = True if row["memorize_chat"] is None else bool(int(row["memorize_chat"]))
        if memorize_chat:
            tail = read_tail(con, cid, after_cursor=cursor + 1)
        else:
            rolling_cursor_id = row["rolling_summary_cursor_id"]
            tail = read_tail_after_message_id(con, cid, rolling_cursor_id)
        if tail:
            for i, msg in enumerate(tail):
                msg["source_conversation_id"] = cid
                msg["source_conversation_index"] = (
                    int(msg["id"]) if (not memorize_chat and msg.get("id") is not None) else (cursor + 1 + i)
                )
                msg["memorize_chat"] = memorize_chat
            result[cid] = tail
    return result


def read_background_rolling_summaries(
    con: sqlite3.Connection,
    *,
    exclude_conversation_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    rows = con.execute(
        "SELECT conversation_id, memorize_chat, rolling_summary, rolling_summary_updated_at "
        "FROM conversations"
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = str(row["conversation_id"] or "").strip()
        if not cid or cid == exclude_conversation_id:
            continue
        memorize_chat = True if row["memorize_chat"] is None else bool(int(row["memorize_chat"]))
        if memorize_chat:
            continue
        summary = str(row["rolling_summary"] or "").strip()
        if not summary:
            continue
        out[cid] = {
            "source_conversation_id": cid,
            "source_label": derive_source_label(cid),
            "summary": summary,
            "updated_at": row["rolling_summary_updated_at"],
        }
    return out


def read_recent(
    con: sqlite3.Connection,
    conversation_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Read the most recent `limit` messages for a conversation (memorized + tail)."""
    rows = con.execute(
        "SELECT role, speaker, chat_name, content, source_label, received_at FROM messages "
        "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
        (conversation_id, limit),
    ).fetchall()
    return list(reversed([
        {
            "role": row["role"],
            "name": row["speaker"],
            "chat_name": row["chat_name"],
            "content": row["content"],
            "source_label": row["source_label"],
            "received_at": row["received_at"],
        }
        for row in rows
    ]))


def read_recent_for_conversation_ids(
    con: sqlite3.Connection,
    conversation_ids: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    ids = [str(cid or "").strip() for cid in conversation_ids if str(cid or "").strip()]
    if not ids:
        return []
    if len(ids) == 1:
        return read_recent(con, ids[0], limit)
    placeholders = ",".join("?" for _ in ids)
    rows = con.execute(
        "SELECT role, speaker, chat_name, content, source_label, received_at FROM messages "
        f"WHERE conversation_id IN ({placeholders}) ORDER BY id DESC LIMIT ?",
        [*ids, limit],
    ).fetchall()
    return list(reversed([
        {
            "role": row["role"],
            "name": row["speaker"],
            "chat_name": row["chat_name"],
            "content": row["content"],
            "source_label": row["source_label"],
            "received_at": row["received_at"],
        }
        for row in rows
    ]))


def format_merged_history(messages: list[dict[str, Any]]) -> str:
    """Format merged messages as grouped markdown for the soul's cross-chat context."""
    numeric_like_re = re.compile(r"^[0-9+\-() .]+$")

    def _conversation_kind_and_key(conversation_id: str) -> tuple[str, str]:
        cid = str(conversation_id or "").strip()
        if cid.startswith("whatsapp:group:"):
            return ("whatsapp_group", cid[len("whatsapp:group:"):].strip())
        if cid.startswith("whatsapp:dm:"):
            return ("whatsapp_dm", cid[len("whatsapp:dm:"):].strip())
        if cid.startswith("sillytavern:"):
            return ("sillytavern_dm", cid[len("sillytavern:"):].strip() or "sillytavern")
        if cid.startswith("integrity:"):
            return ("sillytavern_dm", cid)
        if cid.startswith("chat:"):
            return ("sillytavern_dm", cid)
        if cid == "sillytavern":
            return ("sillytavern_dm", "sillytavern")
        return ("sillytavern_dm", cid or "sillytavern")

    def _load_whatsapp_directory_names() -> dict[str, str]:
        hermes_home = Path(os.getenv("HERMES_HOME") or "~/.hermes").expanduser().resolve()
        directory_path = hermes_home / "channel_directory.json"
        out: dict[str, str] = {}
        if not directory_path.exists():
            return out
        try:
            raw = json.loads(directory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return out
        platforms = raw.get("platforms") if isinstance(raw, dict) else None
        rows = platforms.get("whatsapp") if isinstance(platforms, dict) else None
        if not isinstance(rows, list):
            rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("id") or "").strip()
            rname = str(row.get("name") or "").strip()
            if not rid or not rname:
                continue
            out[rid] = rname
            normalized = _normalize_whatsapp_identifier(rid)
            if normalized and normalized not in out:
                out[normalized] = rname

        # memU Stack can enrich raw group ids with bridge-resolved names and
        # persist them here. Use that cache so payload headings match launcher UI.
        cache_path = hermes_home / "whatsapp_group_names.json"
        if cache_path.exists():
            try:
                cache_raw = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cache_raw = {}
            if isinstance(cache_raw, dict):
                for key, value in cache_raw.items():
                    chat_id = str(key or "").strip()
                    group_name = str(value or "").strip()
                    if not chat_id or not group_name:
                        continue
                    out[chat_id] = group_name
                    normalized = _normalize_whatsapp_identifier(chat_id)
                    if normalized:
                        out[normalized] = group_name
        return out

    def _lookup_whatsapp_name(key: str, names: dict[str, str]) -> str:
        key_norm = _normalize_whatsapp_identifier(key)
        candidates: list[str] = []
        seen: set[str] = set()

        def _push(candidate_key: str) -> None:
            value = str(names.get(candidate_key) or "").strip()
            if value and value not in seen:
                seen.add(value)
                candidates.append(value)

        if key:
            _push(key)
        if key_norm:
            _push(key_norm)

        for alias in sorted(_expand_whatsapp_alias_identifiers(key_norm), key=lambda item: (len(item), item)):
            _push(alias)
            _push(f"{alias}@lid")
            _push(f"{alias}@s.whatsapp.net")

        if not candidates:
            return ""

        def _score(name: str) -> tuple[int, int, int]:
            normalized_name = _normalize_whatsapp_identifier(name)
            same_as_key = int(bool(key_norm) and normalized_name == key_norm)
            numeric_like = int(bool(numeric_like_re.fullmatch(name)))
            return (same_as_key, numeric_like, len(name))

        return min(candidates, key=_score)

    def _conversation_heading(
        kind: str,
        key: str,
        names: dict[str, str],
        chat_name: str | None,
    ) -> str:
        if kind == "whatsapp_group":
            pretty = _lookup_whatsapp_name(key, names) or key or "group"
            return f"[group][{pretty}]"
        if kind == "whatsapp_dm":
            pretty = _lookup_whatsapp_name(key, names) or key or "contact"
            return f"[dm][{pretty}]"
        if kind == "sillytavern_dm":
            pretty = (chat_name or "").strip() or key or "sillytavern"
            return f"[dm][{pretty}]"
        return f"[dm][{key or 'sillytavern'}]"

    def _section_title(kind: str) -> str:
        if kind.startswith("sillytavern_"):
            return "## My SillyTavern Conversations:"
        if kind.startswith("whatsapp_"):
            return "## My WhatsApp Conversations:"
        return "## My SillyTavern Conversations:"

    by_conversation: dict[str, list[dict[str, Any]]] = {}
    for msg in messages:
        cid = str(msg.get("conversation_id") or "").strip() or "unknown"
        by_conversation.setdefault(cid, []).append(msg)

    dir_names = _load_whatsapp_directory_names()

    sections: dict[str, list[str]] = {}
    for cid, rows in by_conversation.items():
        kind, key = _conversation_kind_and_key(cid)
        section_key = _section_title(kind)
        blocks = sections.setdefault(section_key, [])
        chat_name = ""
        for msg in reversed(rows):
            candidate = str(msg.get("chat_name") or "").strip()
            if candidate:
                chat_name = candidate
                break
        conv_lines: list[str] = [
            _conversation_heading(kind, key, dir_names, chat_name or None)
        ]
        last_time_label: str | None = None
        for msg in rows:
            time_label = format_relative_time_label(msg.get("received_at"))
            if time_label and time_label != last_time_label:
                conv_lines.append(f"--- {time_label} ---")
                last_time_label = time_label
            role = str(msg.get("role") or "").strip()
            speaker = str(msg.get("speaker") or "").strip()
            content = str(msg.get("content") or "")
            # Legacy fallback: pre-eb51570 group rows stored the raw "[Sender] body"
            # text. The content prefix is the authoritative sender for groups.
            if role == "user" and kind == "whatsapp_group":
                parsed = _parse_shared_group_sender_prefix(content)
                if parsed is not None:
                    speaker, content = parsed
            if not speaker:
                speaker = role or "unknown"
            conv_lines.append(f"[{speaker}]: {content}")
        blocks.append("\n".join(conv_lines))

    lines: list[str] = []
    for section_title, blocks in sections.items():
        if not blocks:
            continue
        if lines:
            lines.append("")
        lines.append(section_title)
        lines.append("")
        lines.append("\n\n".join(blocks))
    return "\n".join(lines).strip()
