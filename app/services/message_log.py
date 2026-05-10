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
        role, speaker, content = _normalize_row_for_overlap(
            conversation_id, role, speaker, content
        )
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
    If a conversation has no unmemorized tail, falls back to recent messages so
    cross-conversation context does not disappear solely due to cursor drift or
    full memorization.
    Capped at max_messages most recent to bound prompt size.
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
        if not tail and recent_fallback_per_conversation > 0:
            recent = read_recent(con, cid, limit=recent_fallback_per_conversation)
            tail = [
                {
                    "role": msg.get("role"),
                    "speaker": msg.get("name"),
                    "content": msg.get("content"),
                    "source_label": msg.get("source_label"),
                    "received_at": msg.get("received_at"),
                    "conversation_id": canonical_cid,
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
        "SELECT role, speaker, content, source_label, received_at FROM messages "
        f"WHERE conversation_id IN ({placeholders}) ORDER BY id DESC LIMIT ?",
        [*ids, limit],
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


def format_merged_history(messages: list[dict[str, Any]]) -> str:
    """Format merged messages as grouped markdown for the soul's cross-chat context."""

    def _conversation_kind_and_key(conversation_id: str) -> tuple[str, str]:
        cid = str(conversation_id or "").strip()
        if cid.startswith("whatsapp:group:"):
            return ("whatsapp_group", cid[len("whatsapp:group:"):].strip())
        if cid.startswith("whatsapp:dm:"):
            return ("whatsapp_dm", cid[len("whatsapp:dm:"):].strip())
        if cid.startswith("sillytavern:"):
            return ("sillytavern_dm", cid[len("sillytavern:"):].strip() or "sillytavern")
        if cid == "sillytavern":
            return ("sillytavern_dm", "sillytavern")
        return ("other", cid or "unknown")

    def _load_whatsapp_directory_names() -> dict[str, str]:
        hermes_home = Path(os.getenv("HERMES_HOME") or "~/.hermes").expanduser().resolve()
        directory_path = hermes_home / "channel_directory.json"
        if not directory_path.exists():
            return {}
        try:
            raw = json.loads(directory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        platforms = raw.get("platforms") if isinstance(raw, dict) else None
        rows = platforms.get("whatsapp") if isinstance(platforms, dict) else None
        if not isinstance(rows, list):
            return {}
        out: dict[str, str] = {}
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
        return out

    def _conversation_heading(kind: str, key: str, names: dict[str, str]) -> str:
        if kind == "whatsapp_group":
            pretty = names.get(key) or names.get(_normalize_whatsapp_identifier(key)) or key or "group"
            return f"[group][{pretty}]"
        if kind == "whatsapp_dm":
            pretty = names.get(key) or names.get(_normalize_whatsapp_identifier(key)) or key or "contact"
            return f"[dm][{pretty}]"
        if kind == "sillytavern_dm":
            pretty = key or "sillytavern"
            return f"[dm][{pretty}]"
        return f"[other][{key or 'unknown'}]"

    def _section_title(kind: str) -> str:
        if kind.startswith("sillytavern_"):
            return "## My SillyTavern Conversations:"
        if kind.startswith("whatsapp_"):
            return "## My WhatsApp Conversations:"
        return "## Other Conversations:"

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

        conv_lines: list[str] = [_conversation_heading(kind, key, dir_names)]
        last_time_label: str | None = None
        # Legacy alias splits + overlap drift can produce repeated rows for the
        # same sender/text within one day bucket. Suppress duplicates in prompt
        # rendering so cross-chat context stays semantically dense.
        seen_day_lines: set[tuple[str, str, str]] = set()
        for msg in rows:
            time_label = format_relative_time_label(msg.get("received_at"))
            if time_label and time_label != last_time_label:
                conv_lines.append(f"--- {time_label} ---")
                last_time_label = time_label
            role = str(msg.get("role") or "").strip()
            speaker = str(msg.get("speaker") or "").strip()
            if not speaker and cid and role:
                speaker = canonical_speaker.get((cid, role), "")
            if not speaker:
                speaker = role or "unknown"
            content = str(msg.get("content") or "")
            if role == "user" and kind == "whatsapp_group":
                parsed = _parse_shared_group_sender_prefix(content)
                if parsed is not None:
                    parsed_speaker, parsed_content = parsed
                    speaker = parsed_speaker
                    content = parsed_content
            day_bucket = time_label or "unknown-day"
            dedupe_key = (day_bucket, speaker.strip().lower(), content.strip())
            if dedupe_key in seen_day_lines:
                continue
            seen_day_lines.add(dedupe_key)
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
