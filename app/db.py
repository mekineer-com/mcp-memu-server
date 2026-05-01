from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def sqlite_ensure_nonempty(path: Path) -> None:
    """Ensure the sqlite file is not a confusing 0-byte placeholder."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                if path.stat().st_size > 0:
                    return
            except OSError:
                pass
        con = sqlite3.connect(str(path), timeout=5.0)
        con.execute("PRAGMA user_version=1")
        con.commit()
        con.close()
    except (OSError, sqlite3.Error):
        logger.warning("sqlite_ensure_nonempty failed for %s", path, exc_info=True)


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
        except sqlite3.Error:
            out[k] = None
    return out


def sqlite_ensure_self_model_tables(con: sqlite3.Connection) -> None:
    con.execute(
        """
CREATE TABLE IF NOT EXISTS memu_self_model (
    id TEXT PRIMARY KEY,
    soul_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    narrative_self TEXT,
    related_memory_ids TEXT,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""
    )
    cols = set(sqlite_table_columns(con, "memu_self_model"))
    if "related_memory_ids" not in cols:
        con.execute("ALTER TABLE memu_self_model ADD COLUMN related_memory_ids TEXT")
        cols = set(sqlite_table_columns(con, "memu_self_model"))
    if "trait_invariants" in cols or "contextual_state" in cols:
        con.execute(
            """
CREATE TABLE memu_self_model__new (
    id TEXT PRIMARY KEY,
    soul_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    narrative_self TEXT,
    related_memory_ids TEXT,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""
        )
        con.execute(
            """
INSERT INTO memu_self_model__new (
    id, soul_id, user_id, narrative_self, related_memory_ids, updated_at
)
SELECT id, soul_id, user_id, narrative_self, related_memory_ids, updated_at
FROM memu_self_model
"""
        )
        con.execute("DROP TABLE memu_self_model")
        con.execute("ALTER TABLE memu_self_model__new RENAME TO memu_self_model")
    tables = {
        str(row[0])
        for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if row and row[0]
    }
    if "memu_intentions" in tables and "intentions_life_goals" not in tables:
        con.execute("ALTER TABLE memu_intentions RENAME TO intentions_life_goals")

    con.execute(
        """
CREATE TABLE IF NOT EXISTS intentions_life_goals (
    id TEXT PRIMARY KEY,
    soul_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    resolution_note TEXT,
    source TEXT,
    confidence REAL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    target_date TEXT,
    related_memory_ids TEXT,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""
    )
    intention_cols = set(sqlite_table_columns(con, "intentions_life_goals"))
    if "resolution_note" not in intention_cols:
        con.execute("ALTER TABLE intentions_life_goals ADD COLUMN resolution_note TEXT")
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_self_model_soul_user ON memu_self_model(soul_id, user_id, updated_at DESC)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_intentions_soul_user ON intentions_life_goals(soul_id, user_id, status)"
    )


def sqlite_ensure_conversation_state_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
CREATE TABLE IF NOT EXISTS memu_conversation_state (
    conversation_id VARCHAR PRIMARY KEY,
    soul_id VARCHAR,
    user_id VARCHAR,
    digest_cursor INTEGER DEFAULT 0,
    prior_context TEXT,
    intentions_active JSON,
    memory_cache JSON DEFAULT '[]',
    pending_episode_ids JSON DEFAULT '[]',
    self_model_id VARCHAR,
    last_retrieval_ids JSON,
    last_memorize_at DATETIME,
    last_consolidation_at DATETIME,
    consolidation_in_progress BOOLEAN DEFAULT 0,
    consolidation_started_at DATETIME,
    subconscious_message TEXT,
    updated_at DATETIME,
    undo_snapshot JSON
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
    if "prior_context" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN prior_context TEXT")
    if "intentions_active" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN intentions_active JSON")
    if "active_intentions" in cols and "intentions_active" not in cols:
        alters.append("ALTER TABLE memu_conversation_state RENAME COLUMN active_intentions TO intentions_active")
    if "memory_cache" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN memory_cache JSON DEFAULT '[]'")
    if "pending_episode_ids" not in cols:
        if "pending_diary_episode_ids" in cols:
            alters.append("ALTER TABLE memu_conversation_state RENAME COLUMN pending_diary_episode_ids TO pending_episode_ids")
        else:
            alters.append("ALTER TABLE memu_conversation_state ADD COLUMN pending_episode_ids JSON DEFAULT '[]'")
    if "self_model_id" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN self_model_id VARCHAR")
    if "last_retrieval_ids" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN last_retrieval_ids JSON")
    if "last_memorize_at" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN last_memorize_at DATETIME")
    if "last_consolidation_at" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN last_consolidation_at DATETIME")
    if "consolidation_in_progress" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN consolidation_in_progress BOOLEAN DEFAULT 0")
    if "consolidation_started_at" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN consolidation_started_at DATETIME")
    if "updated_at" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN updated_at DATETIME")
    if "undo_snapshot" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN undo_snapshot JSON")
    if "all_categories_summary" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN all_categories_summary TEXT")
    if "last_chat_x" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN last_chat_x TEXT")
    if "last_chat_x_prev" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN last_chat_x_prev TEXT")
    if "retrieve_rewrite_angle" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN retrieve_rewrite_angle INTEGER DEFAULT 0")
    if "retrieval_ids_since_consolidation" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN retrieval_ids_since_consolidation JSON")
    if "prior_context_ids_since_consolidation" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN prior_context_ids_since_consolidation JSON")
    if "subconscious_message" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN subconscious_message TEXT")
    if "last_background_error" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN last_background_error TEXT")
    if "last_background_error_at" not in cols:
        alters.append("ALTER TABLE memu_conversation_state ADD COLUMN last_background_error_at DATETIME")
    for stmt in alters:
        try:
            con.execute(stmt)
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise
    sqlite_ensure_self_model_tables(con)
    con.commit()


def normalize_text_list(value: Any) -> list[str]:
    parsed = json_from_db(value)
    if not isinstance(parsed, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def merge_unique_text_lists(left: Any, right: Any) -> list[str]:
    return normalize_text_list([*normalize_text_list(left), *normalize_text_list(right)])


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
        except (json.JSONDecodeError, ValueError):
            return None
    return value
