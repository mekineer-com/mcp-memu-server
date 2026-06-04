from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import deque
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import sanitize_db_filename
from app.services import memorize_endpoint

_NUMERIC_LIKE_RE = re.compile(r"^[0-9+\-() .]+$")
_ST_SNAPSHOT_FILE = "latest_history.json"
_LID_MAPPING_FILE_RE = re.compile(r"^lid-mapping-(.+?)(?:_reverse)?\.json$")
_GATEWAY_NOTICE_PREFIXES = (
    "⚠️ Gateway shutting down — ",
    "⚠️ Gateway restarting — ",
)


def _normalize_whatsapp_identifier(value: str) -> str:
    return (
        str(value or "")
        .strip()
        .replace("+", "", 1)
        .split(":", 1)[0]
        .split("@", 1)[0]
    )


def _resolve_hermes_paths(
    *,
    hermes_home: Path | None = None,
    sessions_index_path: Path | None = None,
    state_db_path: Path | None = None,
) -> tuple[Path, Path]:
    base = (hermes_home or Path(os.getenv("HERMES_HOME") or "~/.hermes")).expanduser().resolve()
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


def _messages_source_message_id_select(con: sqlite3.Connection) -> str:
    columns = {
        str(row[1])
        for row in con.execute("PRAGMA table_info(messages)").fetchall()
        if len(row) > 1
    }
    if "source_message_id" in columns:
        return "source_message_id"
    return "NULL AS source_message_id"


def _parse_session_key_chat_token(session_key: str, *, chat_type: str) -> str:
    marker = f":whatsapp:{chat_type}:"
    idx = str(session_key).find(marker)
    if idx < 0:
        return ""
    tail = session_key[idx + len(marker) :]
    if not tail:
        return ""
    return tail.split(":", 1)[0]


def _load_whatsapp_alias_graph(*, session_dir: Path) -> dict[str, set[str]]:
    # Transitional Plan E read-path safety net.
    # Remove in Phase 5 once source-side canonicalization is fully proven.
    graph: dict[str, set[str]] = {}

    def _link(a: str, b: str) -> None:
        if not a or not b or a == b:
            return
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)

    if session_dir.exists():
        for path in session_dir.iterdir():
            if not path.is_file():
                continue
            match = _LID_MAPPING_FILE_RE.match(path.name)
            if not match:
                continue
            key_local = _normalize_whatsapp_identifier(match.group(1))
            if not key_local:
                continue
            try:
                mapped_local = _normalize_whatsapp_identifier(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
            _link(key_local, mapped_local)

        creds_path = session_dir / "creds.json"
        if creds_path.exists():
            try:
                parsed = json.loads(creds_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                parsed = None
            me = parsed.get("me") if isinstance(parsed, dict) else None
            if isinstance(me, dict):
                phone_local = _normalize_whatsapp_identifier(me.get("id"))
                lid_local = _normalize_whatsapp_identifier(me.get("lid"))
                _link(phone_local, lid_local)

    return graph


def _expand_whatsapp_aliases(identifier: str, *, alias_graph: dict[str, set[str]]) -> set[str]:
    start = _normalize_whatsapp_identifier(identifier)
    if not start:
        return set()
    out: set[str] = set()
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current in out:
            continue
        out.add(current)
        for nxt in alias_graph.get(current, set()):
            if nxt and nxt not in out:
                queue.append(nxt)
    return out


def _collect_whatsapp_session_entries(
    sessions_index: dict[str, Any],
    *,
    conversation_id: str,
    alias_graph: dict[str, set[str]] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    cid = str(conversation_id or "").strip()
    if cid.startswith("whatsapp:group:"):
        chat_type = "group"
        target = cid[len("whatsapp:group:") :]
    elif cid.startswith("whatsapp:dm:"):
        chat_type = "dm"
        target = cid[len("whatsapp:dm:") :]
    else:
        return [], ""

    target_norm = _normalize_whatsapp_identifier(target)
    target_aliases = (
        _expand_whatsapp_aliases(target_norm, alias_graph=alias_graph or {})
        if chat_type == "dm"
        else {target_norm}
    )
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
            _normalize_whatsapp_identifier(str(origin.get("chat_id") or "")),
            _normalize_whatsapp_identifier(_parse_session_key_chat_token(str(session_key), chat_type=chat_type)),
        ]
        if target_norm:
            if chat_type == "dm":
                if not any(token and token in target_aliases for token in token_candidates):
                    continue
            elif target_norm not in token_candidates:
                continue

        out.append(
            {
                "session_id": session_id,
                "chat_name": str(origin.get("chat_name") or entry.get("display_name") or "").strip(),
                "user_name": str(origin.get("user_name") or "").strip(),
            }
        )
    return out, chat_type


def _resolve_hermes_base(
    *,
    hermes_home: Path | None,
    sessions_path: Path,
) -> Path:
    if isinstance(hermes_home, Path):
        return hermes_home.expanduser().resolve()
    if sessions_path.name == "sessions.json" and sessions_path.parent.name == "sessions":
        return sessions_path.parent.parent
    return sessions_path.parent


def _pick_chat_name(entries: list[dict[str, Any]], fallback: str, *, chat_type: str) -> str:
    def collect_names(keys: tuple[str, ...]) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            for key in keys:
                value = str(entry.get(key) or "").strip()
                if value and value not in seen:
                    seen.add(value)
                    names.append(value)
        return names

    if chat_type != "group":
        chat_names = collect_names(("chat_name",))
        non_numeric_chat_names = [name for name in chat_names if not _NUMERIC_LIKE_RE.fullmatch(name)]
        if non_numeric_chat_names:
            return min(non_numeric_chat_names, key=lambda name: (len(name), name))

        user_names = collect_names(("user_name",))
        non_numeric_user_names = [name for name in user_names if not _NUMERIC_LIKE_RE.fullmatch(name)]
        if non_numeric_user_names:
            return min(non_numeric_user_names, key=lambda name: (len(name), name))
        if chat_names:
            return min(chat_names, key=lambda name: (len(name), name))
        if user_names:
            return min(user_names, key=lambda name: (len(name), name))
        return fallback

    names: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        value = str(entry.get("chat_name") or "").strip()
        if value and value not in seen:
            seen.add(value)
            names.append(value)
    if not names:
        for entry in entries:
            value = str(entry.get("user_name") or "").strip()
            if value and value not in seen:
                seen.add(value)
                names.append(value)
    if not names:
        return fallback
    return min(names, key=lambda name: (int(bool(_NUMERIC_LIKE_RE.fullmatch(name))), len(name), name))


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
            row = con.execute(
                "SELECT parent_session_id FROM sessions WHERE id = ? LIMIT 1",
                (current,),
            ).fetchone()
            parent = str((row["parent_session_id"] if row else "") or "").strip()
            if parent and parent not in seen:
                seen.add(parent)
                ordered.append(parent)
                queue.append(parent)
        return ordered
    except sqlite3.OperationalError:
        return ordered
    finally:
        con.close()


def _source_id_matches_any(source_message_id: str, expected_ids: set[str]) -> bool:
    source = str(source_message_id or "").strip()
    return bool(source and any(value == source or value in source for value in expected_ids))


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
    base_home = _resolve_hermes_base(hermes_home=hermes_home, sessions_path=sessions_path)
    alias_graph = _load_whatsapp_alias_graph(session_dir=(base_home / "whatsapp" / "session"))
    entries, _chat_type = _collect_whatsapp_session_entries(
        _load_sessions_index(sessions_path),
        conversation_id=conversation_id,
        alias_graph=alias_graph,
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
        columns = {
            str(row[1])
            for row in con.execute("PRAGMA table_info(messages)").fetchall()
            if len(row) > 1
        }
        if "source_message_id" not in columns:
            return set()
        rows = con.execute(
            f"SELECT source_message_id FROM messages "
            f"WHERE session_id IN ({placeholders}) AND role = 'assistant' "
            "AND source_message_id IS NOT NULL AND TRIM(source_message_id) != ''",
            session_ids,
        ).fetchall()
    except sqlite3.OperationalError:
        return set()
    finally:
        con.close()
    return {str(row["source_message_id"] or "").strip() for row in rows if str(row["source_message_id"] or "").strip()}


def _to_iso_utc(value: Any) -> str:
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return ""
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


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
    for suffix in ("short_name", "name", "push_name", "verified_name"):
        key = f"{prefix}_{suffix}"
        value = row[key] if key in row.keys() else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _web_source_db_path(*, hermes_home: Path | None, web_source_db_path: Path | None) -> Path:
    if web_source_db_path is not None:
        return web_source_db_path.expanduser().resolve()
    base = (hermes_home or Path(os.getenv("HERMES_HOME") or "~/.hermes")).expanduser().resolve()
    return base / "whatsapp" / "web_source.db"


def _web_source_chat_match(
    conversation_id: str,
    *,
    hermes_home: Path | None,
) -> tuple[str, set[str], str]:
    cid = str(conversation_id or "").strip()
    if cid.startswith("whatsapp:group:"):
        chat_type = "group"
        target = cid[len("whatsapp:group:") :]
    elif cid.startswith("whatsapp:dm:"):
        chat_type = "dm"
        target = cid[len("whatsapp:dm:") :]
    else:
        return "", set(), ""
    target_local = _normalize_whatsapp_identifier(target)
    if chat_type == "dm":
        base = (hermes_home or Path(os.getenv("HERMES_HOME") or "~/.hermes")).expanduser().resolve()
        alias_graph = _load_whatsapp_alias_graph(session_dir=(base / "whatsapp" / "session"))
        locals_ = _expand_whatsapp_aliases(target_local, alias_graph=alias_graph) or {target_local}
    else:
        locals_ = {target_local}
    return target, {value for value in locals_ if value}, chat_type


def _web_source_row_to_tail(
    row: sqlite3.Row,
    *,
    conversation_id: str,
    chat_type: str,
    chat_name: str,
    soul_id: str,
    reply_prefix: str,
    assistant_source_message_ids: set[str],
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
            speaker = _normalize_whatsapp_identifier(str(row["author_id"] or row["from_id"] or ""))
    return {
        "id": int(row["rowid"]),
        "source_message_id": source_message_id,
        "role": role,
        "speaker": speaker,
        "chat_name": resolved_chat_name,
        "content": content,
        "source_label": f"whatsapp:{chat_type}",
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
    max_messages: int | None = None,
    assistant_source_message_ids: set[str] | None = None,
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
        max_messages=max_messages,
        assistant_source_message_ids=assistant_source_message_ids,
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
        max_messages=None,
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
    max_messages: int | None,
    assistant_source_message_ids: set[str] | None,
) -> list[dict[str, Any]]:
    db_path = _web_source_db_path(hermes_home=hermes_home, web_source_db_path=web_source_db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"WhatsApp web_source db missing: {db_path}")
    target, local_ids, chat_type = _web_source_chat_match(conversation_id, hermes_home=hermes_home)
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

    limit_tail = (
        max(1, int(max_messages))
        if max_messages is not None and not cursor_is_rowid and cursor < 0 and recent_fallback_messages <= 0
        else None
    )
    select_sql = f"""
        SELECT
          m.rowid, m.msg_key, m.chat_id, m.from_me, m.timestamp, m.body,
          m.author_id, m.from_id,
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
    if limit_tail is not None:
        sql = f"""
            SELECT *
            FROM (
              {select_sql}
              ORDER BY m.timestamp DESC, m.rowid DESC
              LIMIT ?
            )
            ORDER BY timestamp ASC, rowid ASC
        """
        params.append(limit_tail)
    else:
        sql = f"{select_sql} ORDER BY m.timestamp ASC, m.rowid ASC"
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(sql, params).fetchall()
    finally:
        con.close()

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
        ))
    ]
    if cursor_is_rowid:
        return all_rows
    return _slice_tail_with_floor(
        all_rows,
        since_cursor=cursor,
        recent_fallback_messages=recent_fallback_messages,
    )


def _slice_tail_with_floor(
    all_rows: Sequence[dict[str, Any]],
    *,
    since_cursor: int,
    recent_fallback_messages: int,
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
    if recent_fallback_messages > 0 and len(tail) < recent_fallback_messages and len(all_rows) > len(tail):
        tail = list(all_rows[-recent_fallback_messages:])
    return tail


def _sillytavern_snapshot_path(
    *,
    storage_dir: Path,
    user_id: str,
    soul_id: str,
    conversation_id: str,
) -> Path:
    chats_dir = (storage_dir / "st_chats").resolve()
    chat_dir, _chat_key, _source = memorize_endpoint.resolve_chat_storage_dir(
        chats_dir,
        user_id,
        soul_id,
        conversation_id,
        sanitize_db_filename,
    )
    chat_dir.mkdir(parents=True, exist_ok=True)
    return (chat_dir / _ST_SNAPSHOT_FILE).resolve()


def persist_sillytavern_history_snapshot(
    *,
    storage_dir: Path,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    history: list[dict[str, Any]],
    chat_name: str | None = None,
) -> None:
    payload = {
        "conversation_id": str(conversation_id or "").strip(),
        "chat_name": str(chat_name or "").strip(),
        "updated_at": datetime.now(UTC).isoformat(),
        "history": list(history or []),
    }
    path = _sillytavern_snapshot_path(
        storage_dir=storage_dir,
        user_id=user_id,
        soul_id=soul_id,
        conversation_id=conversation_id,
    )
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_sillytavern_tail(
    *,
    storage_dir: Path,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    since_cursor: int,
    recent_fallback_messages: int,
) -> list[dict[str, Any]]:
    path = _sillytavern_snapshot_path(
        storage_dir=storage_dir,
        user_id=user_id,
        soul_id=soul_id,
        conversation_id=conversation_id,
    )
    if not path.exists():
        raise FileNotFoundError(f"sillytavern snapshot missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"sillytavern snapshot must be an object: {path}")
    history = raw.get("history")
    if not isinstance(history, list):
        raise RuntimeError(f"sillytavern snapshot history must be a list: {path}")
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
        ts_ms = item.get("ts_ms")
        received_at = _to_iso_utc((float(ts_ms) / 1000.0) if isinstance(ts_ms, (int, float)) else "")
        all_rows.append(
            {
                "role": role,
                "speaker": speaker,
                "chat_name": chat_name,
                "content": content,
                "source_label": "sillytavern",
                "received_at": received_at,
                "conversation_id": conversation_id,
                "source_conversation_id": conversation_id,
                "source_conversation_index": idx,
            }
        )
    return _slice_tail_with_floor(
        all_rows,
        since_cursor=since_cursor,
        recent_fallback_messages=recent_fallback_messages,
    )


def load_whatsapp_tail(
    *,
    conversation_id: str,
    since_cursor: int,
    recent_fallback_messages: int,
    hermes_home: Path | None = None,
    sessions_index_path: Path | None = None,
    state_db_path: Path | None = None,
    min_timestamp: float | None = None,
    max_messages: int | None = None,
) -> list[dict[str, Any]]:
    sessions_path, db_path = _resolve_hermes_paths(
        hermes_home=hermes_home,
        sessions_index_path=sessions_index_path,
        state_db_path=state_db_path,
    )
    base_home = _resolve_hermes_base(hermes_home=hermes_home, sessions_path=sessions_path)
    alias_graph = _load_whatsapp_alias_graph(session_dir=(base_home / "whatsapp" / "session"))
    entries, chat_type = _collect_whatsapp_session_entries(
        _load_sessions_index(sessions_path),
        conversation_id=conversation_id,
        alias_graph=alias_graph,
    )
    if not entries:
        raise RuntimeError(f"no WhatsApp session mapping for conversation_id={conversation_id!r}")
    if not db_path.exists():
        raise FileNotFoundError(f"Hermes state db missing: {db_path}")

    fallback_name = str(conversation_id).split(":", 2)[-1].strip() or "contact"
    chat_name = _pick_chat_name(entries, fallback_name, chat_type=chat_type)
    session_ids: list[str] = []
    session_user_name: dict[str, str] = {}
    for entry in entries:
        sid = str(entry.get("session_id") or "").strip()
        if sid and sid not in session_user_name:
            session_ids.append(sid)
            session_user_name[sid] = str(entry.get("user_name") or "").strip()
    primary_session_ids = set(session_ids)
    session_ids = _expand_session_ids_with_lineage(db_path, session_ids)

    placeholders = ",".join("?" for _ in session_ids)
    where = (
        f"session_id IN ({placeholders}) AND role IN ('user', 'assistant') "
        "AND content IS NOT NULL AND TRIM(content) != ''"
    )
    params: list[Any] = list(session_ids)
    if min_timestamp is not None:
        where += " AND timestamp >= ?"
        params.append(float(min_timestamp))
    limit_tail = (
        max(1, int(max_messages))
        if max_messages is not None and since_cursor < 0 and recent_fallback_messages <= 0
        else None
    )
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        source_message_id_select = _messages_source_message_id_select(con)
        select_sql = (
            "SELECT id, session_id, role, content, timestamp, sender_id, sender_name, "
            f"{source_message_id_select} FROM messages "
            f"WHERE {where}"
        )
        if limit_tail is not None:
            rows = con.execute(
                f"SELECT * FROM ({select_sql} ORDER BY timestamp DESC, id DESC LIMIT ?) "
                "ORDER BY timestamp ASC, id ASC",
                [*params, limit_tail],
            ).fetchall()
        else:
            rows = con.execute(
                f"{select_sql} ORDER BY timestamp ASC, id ASC",
                params,
            ).fetchall()
    finally:
        con.close()

    source_label = f"whatsapp:{chat_type}"
    all_rows: list[dict[str, Any]] = []
    primary_index = 0
    lineage_index = -1
    for row in rows:
        sid = str(row["session_id"] or "").strip()
        if sid in primary_session_ids:
            source_index = primary_index
            primary_index += 1
        else:
            # Parent-session rows are historical context. Keep active-session
            # cursors non-negative and stable across lineage expansion.
            source_index = lineage_index
            lineage_index -= 1
        content = str(row["content"] or "").strip()
        if not content or _is_gateway_notice(content):
            continue
        role = str(row["role"] or "").strip().lower()
        speaker = ""
        if role in {"user", "assistant"}:
            fallback_name = session_user_name.get(sid, "") if role == "user" else ""
            speaker = _resolve_whatsapp_row_speaker(
                sender_name=row["sender_name"],
                sender_id=row["sender_id"],
                session_user_name=fallback_name,
            )
        all_rows.append(
            {
                "role": role,
                "speaker": speaker,
                "chat_name": chat_name,
                "content": content,
                "source_message_id": str(row["source_message_id"] or ""),
                "source_label": source_label,
                "received_at": _to_iso_utc(row["timestamp"]),
                "conversation_id": conversation_id,
                "source_conversation_id": conversation_id,
                "source_conversation_index": source_index,
            }
        )

    return _slice_tail_with_floor(
        all_rows,
        since_cursor=since_cursor,
        recent_fallback_messages=recent_fallback_messages,
    )


def load_whatsapp_tail_after_message_id(
    *,
    conversation_id: str,
    after_message_id: int | None,
    hermes_home: Path | None = None,
    sessions_index_path: Path | None = None,
    state_db_path: Path | None = None,
    min_timestamp: float | None = None,
) -> list[dict[str, Any]]:
    sessions_path, db_path = _resolve_hermes_paths(
        hermes_home=hermes_home,
        sessions_index_path=sessions_index_path,
        state_db_path=state_db_path,
    )
    entries, chat_type = _collect_whatsapp_session_entries(
        _load_sessions_index(sessions_path),
        conversation_id=conversation_id,
        alias_graph=_load_whatsapp_alias_graph(
            session_dir=(_resolve_hermes_base(hermes_home=hermes_home, sessions_path=sessions_path) / "whatsapp" / "session")
        ),
    )
    if not entries:
        raise RuntimeError(f"no WhatsApp session mapping for conversation_id={conversation_id!r}")
    if not db_path.exists():
        raise FileNotFoundError(f"Hermes state db missing: {db_path}")

    fallback_name = str(conversation_id).split(":", 2)[-1].strip() or "contact"
    chat_name = _pick_chat_name(entries, fallback_name, chat_type=chat_type)
    session_ids: list[str] = []
    session_user_name: dict[str, str] = {}
    for entry in entries:
        sid = str(entry.get("session_id") or "").strip()
        if sid and sid not in session_user_name:
            session_ids.append(sid)
            session_user_name[sid] = str(entry.get("user_name") or "").strip()
    session_ids = _expand_session_ids_with_lineage(db_path, session_ids)

    placeholders = ",".join("?" for _ in session_ids)
    cursor_id = int(after_message_id or 0)
    where = (
        f"session_id IN ({placeholders}) AND role IN ('user', 'assistant') AND id > ? "
        "AND content IS NOT NULL AND TRIM(content) != ''"
    )
    params: list[Any] = [*session_ids, cursor_id]
    if min_timestamp is not None:
        where += " AND timestamp >= ?"
        params.append(float(min_timestamp))
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        source_message_id_select = _messages_source_message_id_select(con)
        rows = con.execute(
            "SELECT id, session_id, role, content, timestamp, sender_id, sender_name, "
            f"{source_message_id_select} FROM messages "
            f"WHERE {where} "
            "ORDER BY timestamp ASC, id ASC",
            params,
        ).fetchall()
    finally:
        con.close()

    source_label = f"whatsapp:{chat_type}"
    out: list[dict[str, Any]] = []
    for row in rows:
        content = str(row["content"] or "").strip()
        if not content or _is_gateway_notice(content):
            continue
        role = str(row["role"] or "").strip().lower()
        sid = str(row["session_id"] or "").strip()
        speaker = ""
        if role in {"user", "assistant"}:
            fallback_name = session_user_name.get(sid, "") if role == "user" else ""
            speaker = _resolve_whatsapp_row_speaker(
                sender_name=row["sender_name"],
                sender_id=row["sender_id"],
                session_user_name=fallback_name,
            )
        out.append(
            {
                "id": int(row["id"]),
                "role": role,
                "speaker": speaker,
                "chat_name": chat_name,
                "content": content,
                "source_message_id": str(row["source_message_id"] or ""),
                "source_label": source_label,
                "received_at": _to_iso_utc(row["timestamp"]),
                "conversation_id": conversation_id,
                "source_conversation_id": conversation_id,
                "source_conversation_index": int(row["id"]),
            }
        )
    return out
