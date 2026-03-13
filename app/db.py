from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def sqlite_ensure_nonempty(path: Path) -> None:
    """Ensure the sqlite file is not a confusing 0-byte placeholder."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                if path.stat().st_size > 0:
                    return
            except Exception:
                pass
        con = sqlite3.connect(str(path), timeout=5.0)
        con.execute("PRAGMA user_version=1")
        con.commit()
        con.close()
    except Exception:
        pass


def sqlite_connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path), timeout=5.0)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=3000")
    return con


def sqlite_table_columns(con: sqlite3.Connection, table: str) -> list[str]:
    cur = con.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall() if len(r) > 1]
    return [c for c in cols if isinstance(c, str)]


def sqlite_pragmas(con: sqlite3.Connection) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ("journal_mode", "synchronous", "busy_timeout", "foreign_keys", "cache_size", "temp_store"):
        try:
            r = con.execute(f"PRAGMA {k}").fetchone()
            out[k] = r[0] if r else None
        except Exception:
            out[k] = None
    return out


def sqlite_ensure_diary_tables(con: sqlite3.Connection) -> None:
    con.execute(
        """
CREATE TABLE IF NOT EXISTS memu_self_model (
    id TEXT PRIMARY KEY,
    soul_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    trait_invariants TEXT,
    narrative_self TEXT,
    contextual_state TEXT,
    related_memory_ids TEXT,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""
    )
    cols = set(sqlite_table_columns(con, "memu_self_model"))
    if "related_memory_ids" not in cols:
        con.execute("ALTER TABLE memu_self_model ADD COLUMN related_memory_ids TEXT")
    con.execute(
        """
CREATE TABLE IF NOT EXISTS memu_intentions (
    id TEXT PRIMARY KEY,
    soul_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    source TEXT,
    confidence REAL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    target_date TEXT,
    related_memory_ids TEXT,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_self_model_soul_user "
        "ON memu_self_model(soul_id, user_id, updated_at DESC)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_intentions_soul_user "
        "ON memu_intentions(soul_id, user_id, status)"
    )


def sqlite_ensure_conversation_state_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
CREATE TABLE IF NOT EXISTS memu_conversation_state (
    conversation_id VARCHAR PRIMARY KEY,
    soul_id VARCHAR,
    user_id VARCHAR,
    digest_cursor INTEGER DEFAULT 0,
    working_note TEXT,
    active_intentions JSON,
    pending_diary_memory_ids JSON DEFAULT '[]',
    self_model_id VARCHAR,
    last_retrieval_ids JSON,
    last_memorize_at DATETIME,
    updated_at DATETIME
)
"""
    )
    cols = set(sqlite_table_columns(con, "memu_conversation_state"))
    alters: list[str] = []
    if "soul_id" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN soul_id VARCHAR")
    if "user_id" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN user_id VARCHAR")
    if "digest_cursor" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN digest_cursor INTEGER DEFAULT 0")
    if "working_note" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN working_note TEXT")
    if "active_intentions" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN active_intentions JSON")
    if "pending_diary_memory_ids" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN pending_diary_memory_ids JSON DEFAULT '[]'")
    if "self_model_id" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN self_model_id VARCHAR")
    if "last_retrieval_ids" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN last_retrieval_ids JSON")
    if "last_memorize_at" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN last_memorize_at DATETIME")
    if "updated_at" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN updated_at DATETIME")
    for stmt in alters:
        con.execute(stmt)
    sqlite_ensure_diary_tables(con)
    con.commit()


def json_to_db(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def json_from_db(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return value
    return value
