from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import sanitize_db_filename
from app.services import memorize_endpoint

_NUMERIC_LIKE_RE = re.compile(r"^[0-9+\-() .]+$")
_ST_SNAPSHOT_FILE = "latest_history.json"
_CHAT_SNAPSHOT_DIRS = {
    "sillytavern": "st_chats",
    "atomic": "atomic_chats",
    "mentra": "transcripts",
}
_GATEWAY_NOTICE_PREFIXES = (
    "⚠️ Gateway shutting down — ",
    "⚠️ Gateway restarting — ",
    "memU turn failed",
)


def mentra_storage_dir() -> Path:
    data_home = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return (data_home / "openalma" / "mentra").expanduser().resolve()


def _normalize_whatsapp_identifier(value: str) -> str:
    return (
        str(value or "")
        .strip()
        .replace("+", "", 1)
        .split(":", 1)[0]
        .split("@", 1)[0]
    )


def _normalize_whatsapp_match_token(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "@" not in raw:
        return raw.replace("+", "", 1).split(":", 1)[0]
    local, _, domain = raw.partition("@")
    local = local.replace("+", "", 1).split(":", 1)[0].strip()
    domain = domain.strip()
    return f"{local}@{domain}" if local and domain else raw


def _hermes_base(hermes_home: Path | None = None) -> Path:
    fallback = Path(__file__).resolve().parents[3] / "hermes-channels" / "data"
    raw = hermes_home or Path(os.getenv("HERMES_HOME") or os.getenv("CHANNELS_HOME") or fallback)
    return raw.expanduser().resolve()


def _resolve_hermes_paths(
    *,
    hermes_home: Path | None = None,
    sessions_index_path: Path | None = None,
    state_db_path: Path | None = None,
) -> tuple[Path, Path]:
    base = _hermes_base(hermes_home)
    sessions_path = sessions_index_path or (base / "sessions" / "sessions.json")
    db_path = state_db_path or (base / "state.db")
    return sessions_path, db_path


def _load_sessions_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"sessions index missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"sessions index must be an object: {path}")
    return raw


def load_soul_active_since(
    *,
    soul_id: str,
    hermes_home: Path | None = None,
    state_db_path: Path | None = None,
) -> float | None:
    selected = str(soul_id or "").strip()
    if not selected:
        return None
    _sessions_path, db_path = _resolve_hermes_paths(
        hermes_home=hermes_home,
        state_db_path=state_db_path,
    )
    if not db_path.exists():
        return None
    con = sqlite3.connect(db_path)
    try:
        row = con.execute(
            "SELECT active_since FROM souls WHERE soul_id = ?",
            (selected,),
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return None
        raise
    finally:
        con.close()
    if row is None:
        return None
    try:
        return float(row[0])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"invalid active_since for soul {selected!r} in {db_path}: {row[0]!r}"
        ) from exc


def _parse_session_key_chat_token(session_key: str, *, chat_type: str) -> str:
    marker = f":whatsapp:{chat_type}:"
    idx = str(session_key).find(marker)
    if idx < 0:
        return ""
    tail = session_key[idx + len(marker) :]
    if not tail:
        return ""
    return tail.split(":", 1)[0]


def _split_whatsapp_conversation_id(conversation_id: str) -> tuple[str, str]:
    cid = str(conversation_id or "").strip()
    if cid.startswith("whatsapp:group:"):
        return "group", cid[len("whatsapp:group:") :]
    if cid.startswith("whatsapp:dm:"):
        return "dm", cid[len("whatsapp:dm:") :]
    return "", ""


def _collect_whatsapp_session_entries(
    sessions_index: dict[str, Any],
    *,
    conversation_id: str,
) -> tuple[list[dict[str, Any]], str]:
    chat_type, target = _split_whatsapp_conversation_id(conversation_id)
    if not chat_type:
        return [], ""

    target_norm = _normalize_whatsapp_match_token(target)
    out: list[dict[str, Any]] = []
    for session_key, entry in sessions_index.items():
        if not isinstance(entry, dict):
            continue
        origin = entry.get("origin") if isinstance(entry.get("origin"), dict) else {}
        platform = str(entry.get("platform") or origin.get("platform") or "").strip().lower()
        if platform != "whatsapp":
            continue
        entry_chat_type = str(entry.get("chat_type") or origin.get("chat_type") or "").strip().lower()
        if entry_chat_type != chat_type:
            continue
        session_id = str(entry.get("session_id") or "").strip()
        if not session_id:
            continue

        token_candidates = [
            _normalize_whatsapp_match_token(str(origin.get("chat_id") or "")),
            _normalize_whatsapp_match_token(_parse_session_key_chat_token(str(session_key), chat_type=chat_type)),
        ]
        if target_norm:
            if target_norm not in token_candidates:
                continue

        out.append(
            {
                "session_id": session_id,
                "chat_name": str(origin.get("chat_name") or entry.get("display_name") or "").strip(),
                "user_name": str(origin.get("user_name") or "").strip(),
            }
        )
    return out, chat_type


def _expand_session_ids_with_lineage(db_path: Path, session_ids: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for sid in session_ids:
        value = str(sid or "").strip()
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    if not ordered:
        return ordered

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        queue = list(ordered)
        while queue:
            current = queue.pop(0)
            try:
                row = con.execute(
                    "SELECT parent_session_id FROM sessions WHERE id = ? LIMIT 1",
                    (current,),
                ).fetchone()
            except sqlite3.OperationalError as exc:
                if "no such table" in str(exc).lower():
                    break
                raise
            parent = str((row["parent_session_id"] if row else "") or "").strip()
            if parent and parent not in seen:
                seen.add(parent)
                ordered.append(parent)
                queue.append(parent)
        return ordered
    finally:
        con.close()


def _source_id_matches_any(source_message_id: str, expected_ids: set[str]) -> bool:
    source = str(source_message_id or "").strip()
    if not source:
        return False
    for raw in expected_ids:
        value = str(raw or "").strip()
        if not value:
            continue
        if value == source:
            return True
        if source.endswith(value):
            prefix = source[: -len(value)]
            if prefix and prefix[-1] in "_-:":
                return True
    return False


def _is_gateway_notice(content: str) -> bool:
    text = str(content or "").strip()
    text = re.sub(r"^\s*(?:✦\s*)?\*{0,2}[^:\[\]\n]{1,80}\*{0,2}\s*:\s*", "", text)
    text = re.sub(r"^\s*\[[^\]\n]{1,80}\]\s*", "", text)
    return any(text.startswith(prefix) for prefix in _GATEWAY_NOTICE_PREFIXES)


def load_whatsapp_assistant_source_message_ids(
    *,
    conversation_id: str,
    hermes_home: Path | None = None,
    sessions_index_path: Path | None = None,
    state_db_path: Path | None = None,
) -> set[str]:
    sessions_path, db_path = _resolve_hermes_paths(
        hermes_home=hermes_home,
        sessions_index_path=sessions_index_path,
        state_db_path=state_db_path,
    )
    if not sessions_path.exists() or not db_path.exists():
        return set()
    entries, _chat_type = _collect_whatsapp_session_entries(
        _load_sessions_index(sessions_path),
        conversation_id=conversation_id,
    )
    session_ids = [
        sid
        for entry in entries
        if (sid := str(entry.get("session_id") or "").strip())
    ]
    session_ids = _expand_session_ids_with_lineage(db_path, session_ids)
    if not session_ids:
        return set()

    placeholders = ",".join("?" for _ in session_ids)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            f"SELECT source_message_id FROM messages "
            f"WHERE session_id IN ({placeholders}) AND role = 'assistant' "
            "AND source_message_id IS NOT NULL AND TRIM(source_message_id) != ''",
            session_ids,
        ).fetchall()
    finally:
        con.close()
    return {str(row["source_message_id"] or "").strip() for row in rows if str(row["source_message_id"] or "").strip()}


def _to_iso_utc(value: Any) -> str:
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return ""
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def _to_epoch_ms(value: Any) -> int | None:
    try:
        return int(float(value) * 1000)
    except (TypeError, ValueError, OverflowError):
        return None


def _resolve_whatsapp_row_speaker(
    *,
    sender_name: Any,
    sender_id: Any,
    session_user_name: str,
) -> str:
    name = str(sender_name or "").strip()
    if name:
        return name
    normalized_sender = _normalize_whatsapp_identifier(str(sender_id or ""))
    if normalized_sender:
        return normalized_sender
    return str(session_user_name or "").strip()


def _strip_soul_prefix(body: str, soul_id: str, reply_prefix: str = "") -> tuple[bool, str]:
    text = str(body or "")
    soul = str(soul_id or "").strip()
    prefix = str(reply_prefix or "").replace("\\n", "\n")
    if prefix.strip() and text.startswith(prefix):
        return True, text[len(prefix):].strip()
    if not soul:
        return False, text
    escaped = re.escape(soul)
    patterns = [
        rf"^\s*(?:✦\s*)?\*{{0,2}}{escaped}\*{{0,2}}\s*:\s*",
        rf"^\s*\[{escaped}\]\s*",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return True, text[match.end():].strip()
    return False, text


def _contact_name(row: sqlite3.Row, prefix: str) -> str:
    values: list[str] = []
    for suffix in ("short_name", "name", "push_name", "verified_name"):
        key = f"{prefix}_{suffix}" if prefix else suffix
        value = row[key] if key in row.keys() else None
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return next(
        (value for value in values if not _NUMERIC_LIKE_RE.fullmatch(value)),
        values[0] if values else "",
    )


def _web_source_db_path(*, hermes_home: Path | None, web_source_db_path: Path | None) -> Path:
    if web_source_db_path is not None:
        return web_source_db_path.expanduser().resolve()
    return _hermes_base(hermes_home) / "whatsapp" / "web_source.db"


def _load_whatsapp_source_ref_map(web_source_db_path: Path) -> dict[str, str]:
    path = web_source_db_path.with_name("contact_store.json")
    if not path.exists():
        return {}
    contacts = json.loads(path.read_text(encoding="utf-8")).get("contacts")
    if not isinstance(contacts, dict):
        raise RuntimeError(f"WhatsApp contact store must contain a contacts object: {path}")
    source_refs: dict[str, str] = {}
    for record in contacts.values():
        if not isinstance(record, dict):
            raise RuntimeError(f"WhatsApp contact store contains an invalid contact: {path}")
        preferred = _normalize_whatsapp_match_token(str(record.get("preferred_jid") or ""))
        if not preferred:
            continue
        aliases = record.get("aliases") or []
        if not isinstance(aliases, list):
            raise RuntimeError(f"WhatsApp contact aliases must be a list: {path}")
        for value in [preferred, *aliases]:
            alias = _normalize_whatsapp_match_token(str(value or ""))
            owner = source_refs.get(alias)
            if owner and owner != preferred:
                raise RuntimeError(f"WhatsApp alias {alias!r} has multiple preferred JIDs in {path}")
            if alias:
                source_refs[alias] = preferred
    return source_refs


def _web_source_chat_match(
    conversation_id: str,
) -> tuple[str, set[str], str]:
    chat_type, target = _split_whatsapp_conversation_id(conversation_id)
    if not chat_type:
        return "", set(), ""
    target_local = _normalize_whatsapp_identifier(target)
    return target, {target_local} if target_local else set(), chat_type


def whatsapp_web_source_message_rowid(
    conversation_id: str,
    source_message_id: str,
    *,
    hermes_home: Path | None = None,
    web_source_db_path: Path | None = None,
) -> int | None:
    db_path = _web_source_db_path(hermes_home=hermes_home, web_source_db_path=web_source_db_path)
    target, local_ids, chat_type = _web_source_chat_match(conversation_id)
    if not chat_type or not local_ids:
        return None
    if not db_path.exists():
        raise FileNotFoundError(f"WhatsApp web_source db missing: {db_path}")
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute(
            f"SELECT rowid FROM whatsapp_messages WHERE msg_key = ? "
            f"AND (chat_id = ? OR chat_local_id IN ({','.join('?' for _ in local_ids)})) "
            "LIMIT 1",
            [str(source_message_id), target, *sorted(local_ids)],
        ).fetchone()
        return int(row[0]) if row is not None else None
    finally:
        con.close()


def resolve_whatsapp_web_source_cursor(
    conversation_id: str,
    cursor: int,
    source_message_id: str | None,
    source_ts: int | None,
    web_source_db_path: Path | None,
    *,
    rolling: bool,
    hermes_home: Path | None = None,
) -> tuple[int, int | None]:
    source_id = str(source_message_id or "").strip()
    if not source_id:
        return int(cursor), None
    rowid = whatsapp_web_source_message_rowid(
        conversation_id,
        source_id,
        hermes_home=hermes_home,
        web_source_db_path=web_source_db_path,
    )
    if rowid is not None:
        return rowid, None
    return (0 if rolling else -1), int(source_ts) if source_ts is not None else None


def _render_reactions(reactions_json: str | None, contact_map: dict[str, str]) -> str:
    if not reactions_json:
        return ""
    reactions: dict[str, str] = json.loads(reactions_json)
    if not isinstance(reactions, dict) or not reactions:
        return ""
    parts = []
    for sender_local_id, emoji in reactions.items():
        if not emoji:
            continue
        name = contact_map.get(sender_local_id) or sender_local_id
        parts.append(f"{emoji} — {name}")
    if not parts:
        return ""
    return f" [reacted {', '.join(parts)}]"


def _whatsapp_role_field(role: str, *, owner_human: bool = False) -> dict[str, str]:
    if role == "assistant" or (owner_human and role == "user"):
        return {"role": role}
    return {}


def _web_source_row_to_tail(
    row: sqlite3.Row,
    *,
    conversation_id: str,
    chat_type: str,
    chat_name: str,
    soul_id: str,
    reply_prefix: str,
    assistant_source_message_ids: set[str],
    contact_map: dict[str, str],
    source_ref_map: dict[str, str],
) -> dict[str, Any] | None:
    body = str(row["body"] or "").strip()
    if not body or _is_gateway_notice(body):
        return None
    resolved_chat_name = _contact_name(row, "chat") or chat_name
    from_me = bool(row["from_me"])
    source_message_id = str(row["msg_key"] or "")
    is_soul = _source_id_matches_any(source_message_id, assistant_source_message_ids)
    prefix_is_soul, stripped = _strip_soul_prefix(body, soul_id, reply_prefix)
    is_soul = is_soul or prefix_is_soul
    if from_me and is_soul:
        role = "assistant"
        content = stripped or body
        speaker = soul_id
    else:
        role = "user"
        content = body
        speaker = ""
        if row["author_id"]:
            speaker = _contact_name(row, "author")
        if not speaker and row["from_id"]:
            speaker = _contact_name(row, "from")
        if not speaker:
            sender = _normalize_whatsapp_identifier(str(row["author_id"] or row["from_id"] or ""))
            speaker = contact_map.get(sender) or sender
    reaction_tag = _render_reactions(row["reactions"], contact_map)
    if reaction_tag:
        content = content + reaction_tag
    source_ref: dict[str, str] = {}
    if not from_me:
        participant_id = _normalize_whatsapp_match_token(str(row["author_id"] or row["from_id"] or ""))
        if participant_id:
            source_ref["source_ref"] = f"whatsapp:{source_ref_map.get(participant_id, participant_id)}"
    return {
        "id": int(row["rowid"]),
        "source_message_id": source_message_id,
        **_whatsapp_role_field(role, owner_human=from_me),
        **source_ref,
        "speaker": speaker,
        "chat_name": resolved_chat_name,
        "content": content,
        "source_label": f"whatsapp:{chat_type}",
        "ts_ms": _to_epoch_ms(row["timestamp"]),
        "received_at": _to_iso_utc(row["timestamp"]),
        "conversation_id": conversation_id,
        "source_conversation_id": conversation_id,
        "source_conversation_index": int(row["rowid"]),
    }


def load_whatsapp_web_source_tail(
    *,
    conversation_id: str,
    since_cursor: int,
    recent_fallback_messages: int,
    soul_id: str,
    reply_prefix: str,
    hermes_home: Path | None = None,
    web_source_db_path: Path | None = None,
    min_timestamp: float | None = None,
    assistant_source_message_ids: set[str] | None = None,
    include_floor_without_new: bool = False,
) -> list[dict[str, Any]]:
    return _load_whatsapp_web_source_tail(
        conversation_id=conversation_id,
        cursor=int(since_cursor),
        cursor_is_rowid=False,
        recent_fallback_messages=recent_fallback_messages,
        soul_id=soul_id,
        reply_prefix=reply_prefix,
        hermes_home=hermes_home,
        web_source_db_path=web_source_db_path,
        min_timestamp=min_timestamp,
        assistant_source_message_ids=assistant_source_message_ids,
        include_floor_without_new=include_floor_without_new,
    )


def load_whatsapp_web_source_tail_after_rowid(
    *,
    conversation_id: str,
    after_rowid: int | None,
    soul_id: str,
    reply_prefix: str,
    hermes_home: Path | None = None,
    web_source_db_path: Path | None = None,
    min_timestamp: float | None = None,
    assistant_source_message_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    return _load_whatsapp_web_source_tail(
        conversation_id=conversation_id,
        cursor=int(after_rowid or 0),
        cursor_is_rowid=True,
        recent_fallback_messages=0,
        soul_id=soul_id,
        reply_prefix=reply_prefix,
        hermes_home=hermes_home,
        web_source_db_path=web_source_db_path,
        min_timestamp=min_timestamp,
        assistant_source_message_ids=assistant_source_message_ids,
    )


def _load_whatsapp_web_source_tail(
    *,
    conversation_id: str,
    cursor: int,
    cursor_is_rowid: bool,
    recent_fallback_messages: int,
    soul_id: str,
    reply_prefix: str,
    hermes_home: Path | None,
    web_source_db_path: Path | None,
    min_timestamp: float | None,
    assistant_source_message_ids: set[str] | None,
    include_floor_without_new: bool = False,
) -> list[dict[str, Any]]:
    db_path = _web_source_db_path(hermes_home=hermes_home, web_source_db_path=web_source_db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"WhatsApp web_source db missing: {db_path}")
    target, local_ids, chat_type = _web_source_chat_match(conversation_id)
    if not chat_type or not local_ids:
        return []

    where = [
        "m.revoked = 0",
        "m.body IS NOT NULL",
        "trim(m.body) != ''",
        f"(m.chat_id = ? OR m.chat_local_id IN ({','.join('?' for _ in local_ids)}))",
    ]
    params: list[Any] = [target, *sorted(local_ids)]
    if cursor_is_rowid:
        where.append("m.rowid > ?")
        params.append(int(cursor))
    if min_timestamp is not None:
        where.append("m.timestamp >= ?")
        params.append(float(min_timestamp))

    select_sql = f"""
        SELECT
          m.rowid, m.msg_key, m.chat_id, m.from_me, m.timestamp, m.body,
          m.author_id, m.from_id, m.reactions,
          cc.name AS chat_name, cc.short_name AS chat_short_name,
          cc.push_name AS chat_push_name, cc.verified_name AS chat_verified_name,
          ca.name AS author_name, ca.short_name AS author_short_name,
          ca.push_name AS author_push_name, ca.verified_name AS author_verified_name,
          cf.name AS from_name, cf.short_name AS from_short_name,
          cf.push_name AS from_push_name, cf.verified_name AS from_verified_name
        FROM whatsapp_messages m
        LEFT JOIN whatsapp_contacts cc ON cc.contact_id = m.chat_id
        LEFT JOIN whatsapp_contacts ca ON ca.contact_id = m.author_id
        LEFT JOIN whatsapp_contacts cf ON cf.contact_id = m.from_id
        WHERE {" AND ".join(where)}
    """
    sql = f"{select_sql} ORDER BY m.timestamp ASC, m.rowid ASC"
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        msg_cols = {
            str(r[1])
            for r in con.execute("PRAGMA table_info(whatsapp_messages)").fetchall()
        }
        if "reactions" not in msg_cols:
            raise RuntimeError(
                f"web_source.db schema is outdated — reactions column missing: {db_path}. "
                "Restart the web-source daemon to apply the migration."
            )
        rows = con.execute(sql, params).fetchall()
        if not rows and not cursor_is_rowid and cursor < 0:
            any_rows = con.execute(
                f"SELECT 1 FROM whatsapp_messages WHERE "
                f"(chat_id = ? OR chat_local_id IN ({','.join('?' for _ in local_ids)})) LIMIT 1",
                [target, *sorted(local_ids)],
            ).fetchone()
            if not any_rows:
                raise RuntimeError(
                    f"web_source returned no rows for conversation_id={conversation_id!r} "
                    f"— likely canonical WhatsApp ID mismatch or web_source clone not synced"
                )
        contact_rows = con.execute(
            "SELECT contact_local_id, name, short_name, push_name, verified_name "
            "FROM whatsapp_contacts"
        ).fetchall()
    finally:
        con.close()

    contact_map: dict[str, str] = {}
    for cr in contact_rows:
        local_id = str(cr["contact_local_id"] or "").strip()
        if not local_id:
            continue
        if name := _contact_name(cr, ""):
            contact_map[local_id] = name
    source_ref_map = _load_whatsapp_source_ref_map(db_path)

    chat_name = str(conversation_id).split(":", 2)[-1].strip() or "contact"
    assistant_ids = {
        str(value or "").strip()
        for value in (assistant_source_message_ids or set())
        if str(value or "").strip()
    }
    all_rows = [
        item
        for row in rows
        if (item := _web_source_row_to_tail(
            row,
            conversation_id=conversation_id,
            chat_type=chat_type,
            chat_name=chat_name,
            soul_id=soul_id,
            reply_prefix=reply_prefix,
            assistant_source_message_ids=assistant_ids,
            contact_map=contact_map,
            source_ref_map=source_ref_map,
        ))
    ]
    if cursor_is_rowid:
        return all_rows
    return slice_tail_with_floor(
        all_rows,
        since_cursor=cursor,
        recent_fallback_messages=recent_fallback_messages,
        include_floor_without_new=include_floor_without_new,
    )


def slice_tail_with_floor(
    all_rows: Sequence[dict[str, Any]],
    *,
    since_cursor: int,
    recent_fallback_messages: int,
    include_floor_without_new: bool = False,
) -> list[dict[str, Any]]:
    def _row_cursor_value(row: dict[str, Any], fallback_idx: int) -> int:
        raw = row.get("source_conversation_index")
        if isinstance(raw, bool):
            return fallback_idx
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            return int(raw)
        try:
            return int(str(raw).strip())
        except (TypeError, ValueError):
            return fallback_idx

    if since_cursor < 0:
        tail = list(all_rows)
    else:
        tail = [
            row
            for idx, row in enumerate(all_rows)
            if _row_cursor_value(row, idx) > since_cursor
        ]
    if (
        since_cursor >= 0
        and recent_fallback_messages > 0
        and len(tail) < recent_fallback_messages
        and len(all_rows) > len(tail)
        and (tail or include_floor_without_new)
    ):
        tail = list(all_rows[-recent_fallback_messages:])
    return tail


def _chat_snapshot_path(
    *,
    storage_dir: Path,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    source_label: str,
) -> Path:
    directory = _CHAT_SNAPSHOT_DIRS.get(source_label)
    if directory is None:
        raise ValueError(f"unsupported chat snapshot source: {source_label}")
    chats_dir = (storage_dir / directory).resolve()
    chat_dir, _chat_key, _source = memorize_endpoint.resolve_chat_storage_dir(
        chats_dir,
        user_id,
        soul_id,
        conversation_id,
        sanitize_db_filename,
    )
    chat_dir.mkdir(parents=True, exist_ok=True)
    return (chat_dir / _ST_SNAPSHOT_FILE).resolve()


def persist_chat_history_snapshot(
    *,
    storage_dir: Path,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    history: list[dict[str, Any]],
    chat_name: str | None = None,
    source_label: str,
) -> None:
    payload = {
        "conversation_id": str(conversation_id or "").strip(),
        "chat_name": str(chat_name or "").strip(),
        "source_label": source_label,
        "updated_at": datetime.now(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "history": list(history or []),
    }
    path = _chat_snapshot_path(
        storage_dir=storage_dir,
        user_id=user_id,
        soul_id=soul_id,
        conversation_id=conversation_id,
        source_label=source_label,
    )
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_name = tmp.name
            tmp.write(json.dumps(payload, ensure_ascii=False))
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass


def persist_sillytavern_history_snapshot(
    *,
    storage_dir: Path,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    history: list[dict[str, Any]],
    chat_name: str | None = None,
) -> None:
    persist_chat_history_snapshot(
        storage_dir=storage_dir,
        user_id=user_id,
        soul_id=soul_id,
        conversation_id=conversation_id,
        history=history,
        chat_name=chat_name,
        source_label="sillytavern",
    )


def persist_atomic_history_snapshot(
    *,
    storage_dir: Path,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    history: list[dict[str, Any]],
    chat_name: str | None = None,
) -> None:
    persist_chat_history_snapshot(
        storage_dir=storage_dir,
        user_id=user_id,
        soul_id=soul_id,
        conversation_id=conversation_id,
        history=history,
        chat_name=chat_name,
        source_label="atomic",
    )


def persist_mentra_history_snapshot(
    *,
    storage_dir: Path,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    history: list[dict[str, Any]],
) -> None:
    persist_chat_history_snapshot(
        storage_dir=storage_dir,
        user_id=user_id,
        soul_id=soul_id,
        conversation_id=conversation_id,
        history=history,
        chat_name="Smartglasses",
        source_label="mentra",
    )


def load_mentra_history_snapshot(
    *,
    storage_dir: Path,
    user_id: str,
    soul_id: str,
    conversation_id: str,
) -> list[dict[str, Any]]:
    path = _chat_snapshot_path(
        storage_dir=storage_dir,
        user_id=user_id,
        soul_id=soul_id,
        conversation_id=conversation_id,
        source_label="mentra",
    )
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("conversation_id") != conversation_id:
        raise RuntimeError(f"mentra snapshot identity mismatch: {path}")
    if raw.get("source_label") != "mentra" or not isinstance(raw.get("history"), list):
        raise RuntimeError(f"mentra snapshot is invalid: {path}")
    if not all(isinstance(row, dict) for row in raw["history"]):
        raise RuntimeError(f"mentra snapshot history contains an invalid row: {path}")
    return list(raw["history"])


def load_chat_snapshot_tail(
    *,
    storage_dir: Path,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    since_cursor: int,
    recent_fallback_messages: int,
    source_label: str,
    include_floor_without_new: bool = False,
) -> list[dict[str, Any]]:
    path = _chat_snapshot_path(
        storage_dir=storage_dir,
        user_id=user_id,
        soul_id=soul_id,
        conversation_id=conversation_id,
        source_label=source_label,
    )
    if not path.exists():
        raise FileNotFoundError(f"{source_label} snapshot missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"{source_label} snapshot must be an object: {path}")
    history = raw.get("history")
    if not isinstance(history, list):
        raise RuntimeError(f"{source_label} snapshot history must be a list: {path}")
    chat_name = str(raw.get("chat_name") or "").strip()
    all_rows: list[dict[str, Any]] = []
    for idx, item in enumerate(history):
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        role = str(item.get("role") or "").strip() or "unknown"
        speaker = str(item.get("name") or "").strip()
        if source_label in {"atomic", "mentra"} and not speaker:
            if role == "user":
                speaker = user_id
            elif role in {"assistant", "soul"}:
                speaker = soul_id
        ts_ms = item.get("ts_ms")
        received_at = _to_iso_utc((float(ts_ms) / 1000.0) if isinstance(ts_ms, (int, float)) else "")
        if not received_at:
            received_at = str(item.get("received_at") or item.get("created_at") or "").strip()
        source_index = item.get("sequence") if source_label == "mentra" else idx
        row = {
            "role": role,
            "speaker": speaker,
            "chat_name": chat_name,
            "content": (
                f"[End-of-sitting reflection] {content}"
                if item.get("event_kind") == "sitting_summary"
                else (
                    f"{content} [interrupted]"
                    if item.get("transcript_status") == "interrupted"
                    else content
                )
            ),
            "source_label": source_label,
            "received_at": received_at,
            "conversation_id": conversation_id,
            "source_conversation_id": conversation_id,
            "source_conversation_index": source_index,
        }
        if source_label == "mentra":
            try:
                received = datetime.fromisoformat(received_at.replace("Z", "+00:00"))
                if received.tzinfo is None:
                    raise ValueError
                row["ts_ms"] = int(received.timestamp() * 1000)
            except (ValueError, OverflowError):
                raise RuntimeError(f"mentra snapshot row has invalid received_at: {path}") from None
            for key in ("event_id", "sequence", "event_kind", "transcript_status"):
                if item.get(key) is not None:
                    row[key] = item[key]
        all_rows.append(row)
    return slice_tail_with_floor(
        all_rows,
        since_cursor=since_cursor,
        recent_fallback_messages=recent_fallback_messages,
        include_floor_without_new=include_floor_without_new,
    )


def load_sillytavern_tail(
    *,
    storage_dir: Path,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    since_cursor: int,
    recent_fallback_messages: int,
) -> list[dict[str, Any]]:
    return load_chat_snapshot_tail(
        storage_dir=storage_dir,
        user_id=user_id,
        soul_id=soul_id,
        conversation_id=conversation_id,
        since_cursor=since_cursor,
        recent_fallback_messages=recent_fallback_messages,
        source_label="sillytavern",
    )


def load_atomic_tail(
    *,
    storage_dir: Path,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    since_cursor: int,
    recent_fallback_messages: int,
) -> list[dict[str, Any]]:
    return load_chat_snapshot_tail(
        storage_dir=storage_dir,
        user_id=user_id,
        soul_id=soul_id,
        conversation_id=conversation_id,
        since_cursor=since_cursor,
        recent_fallback_messages=recent_fallback_messages,
        source_label="atomic",
    )


def load_mentra_tail(
    *,
    storage_dir: Path,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    since_cursor: int,
    recent_fallback_messages: int,
    include_floor_without_new: bool = False,
) -> list[dict[str, Any]]:
    try:
        return load_chat_snapshot_tail(
            storage_dir=storage_dir,
            user_id=user_id,
            soul_id=soul_id,
            conversation_id=conversation_id,
            since_cursor=since_cursor,
            recent_fallback_messages=recent_fallback_messages,
            source_label="mentra",
            include_floor_without_new=include_floor_without_new,
        )
    except FileNotFoundError:
        return []
