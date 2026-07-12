"""Soul-level state — single row, shared across all conversations."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

from app.db import json_from_db, json_to_db, normalize_text_list
from app.services.intention_state import normalize_intentions_stack, normalize_memory_cache


def ensure_schema(con: sqlite3.Connection) -> None:
    table_exists = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'soul_state'"
    ).fetchone() is not None
    existing_cols = (
        {row[1] for row in con.execute("PRAGMA table_info(soul_state)").fetchall()}
        if table_exists
        else set()
    )
    added_columns = {
        "apimw_message_to_self": "TEXT",
        "narrative_self_previous": "TEXT",
        "narrative_self_approved": "TEXT",
        "all_categories_summary_previous": "TEXT",
        "all_categories_summary_approved": "TEXT",
        "summaries_revision": "INTEGER NOT NULL DEFAULT 0",
    }
    missing_row = table_exists and con.execute("SELECT COUNT(*) FROM soul_state").fetchone()[0] == 0
    needs_migration = (
        not table_exists
        or missing_row
        or any(name not in existing_cols for name in added_columns)
    )
    owns_migration = needs_migration and not con.in_transaction
    if owns_migration:
        con.execute("BEGIN")

    con.execute("""
CREATE TABLE IF NOT EXISTS soul_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    narrative_self TEXT,
    narrative_self_previous TEXT,
    narrative_self_approved TEXT,
    all_categories_summary TEXT,
    all_categories_summary_previous TEXT,
    all_categories_summary_approved TEXT,
    summaries_revision INTEGER NOT NULL DEFAULT 0,
    memory_cache JSON DEFAULT '[]',
    intentions_active JSON,
    retrieve_rewrite_angle INTEGER DEFAULT 0,
    retrieval_ids_since_consolidation JSON DEFAULT '[]',
    prior_context_ids_since_consolidation JSON DEFAULT '[]',
    last_consolidation_at DATETIME,
    consolidation_in_progress BOOLEAN DEFAULT 0,
    consolidation_started_at DATETIME,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
)""")
    cols = {row[1] for row in con.execute("PRAGMA table_info(soul_state)").fetchall()}
    for name, definition in added_columns.items():
        if name in cols:
            continue
        con.execute(f"ALTER TABLE soul_state ADD COLUMN {name} {definition}")
        if name == "narrative_self_approved":
            con.execute("UPDATE soul_state SET narrative_self_approved = narrative_self")
        elif name == "all_categories_summary_approved":
            con.execute("UPDATE soul_state SET all_categories_summary_approved = all_categories_summary")
    if con.execute("SELECT COUNT(*) FROM soul_state").fetchone()[0] == 0:
        con.execute("INSERT INTO soul_state (id, updated_at) VALUES (1, ?)", (datetime.now(UTC).isoformat(),))
    if owns_migration:
        con.commit()


def read(con: sqlite3.Connection) -> dict[str, Any]:
    ensure_schema(con)
    row = con.execute("SELECT * FROM soul_state WHERE id = 1").fetchone()
    if row is None:
        return defaults()
    return {
        "narrative_self": row["narrative_self"],
        "narrative_self_previous": row["narrative_self_previous"],
        "narrative_self_approved": row["narrative_self_approved"],
        "all_categories_summary": row["all_categories_summary"],
        "all_categories_summary_previous": row["all_categories_summary_previous"],
        "all_categories_summary_approved": row["all_categories_summary_approved"],
        "summaries_revision": int(row["summaries_revision"] or 0),
        "memory_cache": normalize_memory_cache(json_from_db(row["memory_cache"])),
        "intentions_active": normalize_intentions_stack(json_from_db(row["intentions_active"])),
        "retrieve_rewrite_angle": int(row["retrieve_rewrite_angle"] or 0),
        "retrieval_ids_since_consolidation": normalize_text_list(row["retrieval_ids_since_consolidation"]),
        "prior_context_ids_since_consolidation": normalize_text_list(row["prior_context_ids_since_consolidation"]),
        "apimw_message_to_self": (str(row["apimw_message_to_self"] or "").strip() or None),
        "last_consolidation_at": row["last_consolidation_at"],
        "consolidation_in_progress": bool(row["consolidation_in_progress"]),
        "consolidation_started_at": row["consolidation_started_at"],
        "updated_at": row["updated_at"],
    }


def defaults() -> dict[str, Any]:
    return {
        "narrative_self": None,
        "narrative_self_previous": None,
        "narrative_self_approved": None,
        "all_categories_summary": None,
        "all_categories_summary_previous": None,
        "all_categories_summary_approved": None,
        "summaries_revision": 0,
        "memory_cache": [],
        "intentions_active": normalize_intentions_stack(None),
        "retrieve_rewrite_angle": 0,
        "retrieval_ids_since_consolidation": [],
        "prior_context_ids_since_consolidation": [],
        "apimw_message_to_self": None,
        "last_consolidation_at": None,
        "consolidation_in_progress": False,
        "consolidation_started_at": None,
        "updated_at": None,
    }


_JSON_FIELDS = {
    "intentions_active", "memory_cache",
    "retrieval_ids_since_consolidation", "prior_context_ids_since_consolidation",
}

_VALID_FIELDS = {
    "memory_cache", "intentions_active",
    "retrieve_rewrite_angle", "retrieval_ids_since_consolidation",
    "prior_context_ids_since_consolidation", "apimw_message_to_self",
    "last_consolidation_at", "consolidation_in_progress", "consolidation_started_at",
}


def write(con: sqlite3.Connection, updates: dict[str, Any]) -> None:
    """Update soul_state fields. Does NOT commit — caller owns the transaction."""
    ensure_schema(con)
    fields = {k: v for k, v in updates.items() if k in _VALID_FIELDS}
    if not fields:
        return
    if "intentions_active" in fields:
        fields["intentions_active"] = normalize_intentions_stack(fields["intentions_active"])
    if "memory_cache" in fields:
        fields["memory_cache"] = normalize_memory_cache(fields["memory_cache"])
    if "retrieval_ids_since_consolidation" in fields:
        fields["retrieval_ids_since_consolidation"] = normalize_text_list(fields["retrieval_ids_since_consolidation"])
    if "prior_context_ids_since_consolidation" in fields:
        fields["prior_context_ids_since_consolidation"] = normalize_text_list(fields["prior_context_ids_since_consolidation"])
    if "consolidation_in_progress" in fields:
        fields["consolidation_in_progress"] = bool(fields["consolidation_in_progress"])
    if "apimw_message_to_self" in fields:
        raw_message = fields["apimw_message_to_self"]
        fields["apimw_message_to_self"] = None if raw_message is None else (str(raw_message).strip() or None)

    fields["updated_at"] = datetime.now(UTC).isoformat()
    assignments = []
    params = []
    for key, value in fields.items():
        assignments.append(f"{key} = ?")
        if key in _JSON_FIELDS:
            params.append(json_to_db(value))
        elif key == "retrieve_rewrite_angle":
            params.append(int(value or 0))
        elif key == "consolidation_in_progress":
            params.append(1 if value else 0)
        else:
            params.append(value)
    con.execute(f"UPDATE soul_state SET {', '.join(assignments)} WHERE id = 1", tuple(params))
