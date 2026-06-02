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
    names: list[str] = []
    seen: set[str] = set()
    candidate_keys = ("chat_name",) if chat_type == "group" else ("chat_name", "user_name")
    for entry in entries:
        for key in candidate_keys:
            value = str(entry.get(key) or "").strip()
            if value and value not in seen:
                seen.add(value)
                names.append(value)
    if not names and chat_type == "group":
        for entry in entries:
            value = str(entry.get("user_name") or "").strip()
            if value and value not in seen:
                seen.add(value)
                names.append(value)
    if not names:
        return fallback
    return min(names, key=lambda name: (int(bool(_NUMERIC_LIKE_RE.fullmatch(name))), len(name), name))


def _to_iso_utc(value: Any) -> str:
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return ""
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def _slice_tail_with_floor(
    all_rows: Sequence[dict[str, Any]],
    *,
    since_cursor: int,
    recent_fallback_messages: int,
) -> list[dict[str, Any]]:
    if since_cursor < 0:
        tail = list(all_rows)
    else:
        tail = list(all_rows[since_cursor + 1 :])
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

    placeholders = ",".join("?" for _ in session_ids)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT id, session_id, role, content, timestamp FROM messages "
            f"WHERE session_id IN ({placeholders}) AND role IN ('user', 'assistant') "
            "ORDER BY timestamp ASC, id ASC",
            session_ids,
        ).fetchall()
    finally:
        con.close()

    source_label = f"whatsapp:{chat_type}"
    all_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        content = str(row["content"] or "").strip()
        if not content:
            continue
        role = str(row["role"] or "").strip().lower()
        sid = str(row["session_id"] or "").strip()
        all_rows.append(
            {
                "role": role,
                "speaker": session_user_name.get(sid) if role == "user" else "",
                "chat_name": chat_name,
                "content": content,
                "source_label": source_label,
                "received_at": _to_iso_utc(row["timestamp"]),
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


def load_whatsapp_tail_after_message_id(
    *,
    conversation_id: str,
    after_message_id: int | None,
    hermes_home: Path | None = None,
    sessions_index_path: Path | None = None,
    state_db_path: Path | None = None,
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

    placeholders = ",".join("?" for _ in session_ids)
    cursor_id = int(after_message_id or 0)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT id, session_id, role, content, timestamp FROM messages "
            f"WHERE session_id IN ({placeholders}) AND role IN ('user', 'assistant') AND id > ? "
            "ORDER BY timestamp ASC, id ASC",
            [*session_ids, cursor_id],
        ).fetchall()
    finally:
        con.close()

    source_label = f"whatsapp:{chat_type}"
    out: list[dict[str, Any]] = []
    for row in rows:
        content = str(row["content"] or "").strip()
        if not content:
            continue
        role = str(row["role"] or "").strip().lower()
        sid = str(row["session_id"] or "").strip()
        out.append(
            {
                "id": int(row["id"]),
                "role": role,
                "speaker": session_user_name.get(sid) if role == "user" else "",
                "chat_name": chat_name,
                "content": content,
                "source_label": source_label,
                "received_at": _to_iso_utc(row["timestamp"]),
                "conversation_id": conversation_id,
                "source_conversation_id": conversation_id,
                "source_conversation_index": int(row["id"]),
            }
        )
    return out
