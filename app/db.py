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


def sqlite_ensure_soul_tables(con: sqlite3.Connection) -> None:
    con.execute(
        """
CREATE TABLE IF NOT EXISTS narrative_history (
    id TEXT PRIMARY KEY,
    narrative_self TEXT NOT NULL,
    related_memory_ids JSON DEFAULT '[]',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""
    )
    con.execute(
        """
CREATE TABLE IF NOT EXISTS intentions (
    id TEXT PRIMARY KEY,
    soul_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    resolution_note TEXT,
    source TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""
    )
    con.execute(
        """
CREATE TABLE IF NOT EXISTS life_goals (
    id TEXT PRIMARY KEY,
    soul_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""
    )
    intention_cols = set(sqlite_table_columns(con, "intentions"))
    if "resolution_note" not in intention_cols:
        con.execute("ALTER TABLE intentions ADD COLUMN resolution_note TEXT")
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_intentions_soul_user ON intentions(soul_id, user_id, status)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_life_goals_soul_user ON life_goals(soul_id, user_id, status)"
    )


def sqlite_ensure_conversation_state_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    soul_id TEXT,
    user_id TEXT,
    memorize_chat INTEGER DEFAULT 1,
    digest_cursor INTEGER DEFAULT 0,
    rolling_summary TEXT,
    rolling_summary_cursor_id INTEGER,
    rolling_summary_updated_at DATETIME,
    prior_context TEXT,
    apimw_message_to_self TEXT,
    pending_segment_ids JSON DEFAULT '[]',
    last_memorize_at DATETIME,
    last_display_segment_start_index INTEGER,
    last_display_segment_end_index INTEGER,
    last_display_segment_at DATETIME,
    updated_at DATETIME,
    undo_snapshot JSON,
    last_background_error TEXT,
    last_background_error_at DATETIME,
    last_consolidation_error TEXT,
    last_consolidation_error_at DATETIME
)
"""
    )
    conversation_cols = set(sqlite_table_columns(con, "conversations"))
    if "memorize_chat" not in conversation_cols:
        con.execute("ALTER TABLE conversations ADD COLUMN memorize_chat INTEGER DEFAULT 1")
    if "pending_segment_ids" not in conversation_cols:
        con.execute("ALTER TABLE conversations ADD COLUMN pending_segment_ids JSON DEFAULT '[]'")
        if "pending_episode_ids" in conversation_cols:
            con.execute(
                """
                UPDATE conversations
                SET pending_segment_ids = pending_episode_ids
                WHERE pending_episode_ids IS NOT NULL
                  AND trim(CAST(pending_episode_ids AS TEXT)) != ''
                  AND (pending_segment_ids IS NULL OR pending_segment_ids = '[]')
                """
            )
    if "rolling_summary" not in conversation_cols:
        con.execute("ALTER TABLE conversations ADD COLUMN rolling_summary TEXT")
    if "rolling_summary_cursor_id" not in conversation_cols:
        con.execute("ALTER TABLE conversations ADD COLUMN rolling_summary_cursor_id INTEGER")
    if "rolling_summary_updated_at" not in conversation_cols:
        con.execute("ALTER TABLE conversations ADD COLUMN rolling_summary_updated_at DATETIME")
    if "apimw_message_to_self" not in conversation_cols:
        con.execute("ALTER TABLE conversations ADD COLUMN apimw_message_to_self TEXT")
    if "last_consolidation_error" not in conversation_cols:
        con.execute("ALTER TABLE conversations ADD COLUMN last_consolidation_error TEXT")
    if "last_consolidation_error_at" not in conversation_cols:
        con.execute("ALTER TABLE conversations ADD COLUMN last_consolidation_error_at DATETIME")
    if "last_display_segment_start_index" not in conversation_cols:
        con.execute("ALTER TABLE conversations ADD COLUMN last_display_segment_start_index INTEGER")
    if "last_display_segment_end_index" not in conversation_cols:
        con.execute("ALTER TABLE conversations ADD COLUMN last_display_segment_end_index INTEGER")
    if "last_display_segment_at" not in conversation_cols:
        con.execute("ALTER TABLE conversations ADD COLUMN last_display_segment_at DATETIME")
    sqlite_ensure_soul_tables(con)
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
