"""Soul-level state — single row, shared across all conversations."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

from app.db import json_from_db, json_to_db, normalize_text_list
from app.services.intention_state import normalize_intentions_stack, normalize_memory_cache


def ensure_schema(con: sqlite3.Connection) -> None:
    con.execute("""
CREATE TABLE IF NOT EXISTS soul_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    narrative_self TEXT,
    all_categories_summary TEXT,
    memory_cache JSON DEFAULT '[]',
    intentions_active JSON,
    retrieve_rewrite_angle INTEGER DEFAULT 0,
    retrieval_ids_since_consolidation JSON DEFAULT '[]',
    prior_context_ids_since_consolidation JSON DEFAULT '[]',
    subconscious_message TEXT,
    last_consolidation_at DATETIME,
    consolidation_in_progress BOOLEAN DEFAULT 0,
    consolidation_started_at DATETIME,
    self_model_id TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
)""")
    if con.execute("SELECT COUNT(*) FROM soul_state").fetchone()[0] == 0:
        con.execute("INSERT INTO soul_state (id, updated_at) VALUES (1, ?)", (datetime.now(UTC).isoformat(),))


def read(con: sqlite3.Connection) -> dict[str, Any]:
    ensure_schema(con)
    row = con.execute("SELECT * FROM soul_state WHERE id = 1").fetchone()
    if row is None:
        return defaults()
    return {
        "narrative_self": row["narrative_self"],
        "all_categories_summary": row["all_categories_summary"],
        "memory_cache": normalize_memory_cache(json_from_db(row["memory_cache"])),
        "intentions_active": normalize_intentions_stack(json_from_db(row["intentions_active"])),
        "retrieve_rewrite_angle": int(row["retrieve_rewrite_angle"] or 0),
        "retrieval_ids_since_consolidation": normalize_text_list(row["retrieval_ids_since_consolidation"]),
        "prior_context_ids_since_consolidation": normalize_text_list(row["prior_context_ids_since_consolidation"]),
        "subconscious_message": row["subconscious_message"],
        "last_consolidation_at": row["last_consolidation_at"],
        "consolidation_in_progress": bool(row["consolidation_in_progress"]),
        "consolidation_started_at": row["consolidation_started_at"],
        "self_model_id": row["self_model_id"],
        "updated_at": row["updated_at"],
    }


def defaults() -> dict[str, Any]:
    return {
        "narrative_self": None,
        "all_categories_summary": None,
        "memory_cache": [],
        "intentions_active": normalize_intentions_stack(None),
        "retrieve_rewrite_angle": 0,
        "retrieval_ids_since_consolidation": [],
        "prior_context_ids_since_consolidation": [],
        "subconscious_message": None,
        "last_consolidation_at": None,
        "consolidation_in_progress": False,
        "consolidation_started_at": None,
        "self_model_id": None,
        "updated_at": None,
    }


_JSON_FIELDS = {
    "intentions_active", "memory_cache",
    "retrieval_ids_since_consolidation", "prior_context_ids_since_consolidation",
}

_VALID_FIELDS = {
    "narrative_self", "all_categories_summary", "memory_cache", "intentions_active",
    "retrieve_rewrite_angle", "retrieval_ids_since_consolidation",
    "prior_context_ids_since_consolidation", "subconscious_message",
    "last_consolidation_at", "consolidation_in_progress", "consolidation_started_at",
    "self_model_id",
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


def seed_from_legacy(con: sqlite3.Connection) -> None:
    """One-time: populate soul_state from old tables if empty."""
    ensure_schema(con)
    existing = con.execute("SELECT narrative_self, all_categories_summary FROM soul_state WHERE id = 1").fetchone()
    if existing and (existing["narrative_self"] or existing["all_categories_summary"]):
        return

    tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    updates: dict[str, Any] = {}

    if "memu_self_model" in tables:
        sm = con.execute("SELECT narrative_self FROM memu_self_model ORDER BY updated_at DESC LIMIT 1").fetchone()
        if sm and sm["narrative_self"]:
            updates["narrative_self"] = str(sm["narrative_self"]).strip()

    if "memu_conversation_state" in tables:
        row = con.execute(
            "SELECT all_categories_summary, memory_cache, intentions_active, retrieve_rewrite_angle, "
            "retrieval_ids_since_consolidation, prior_context_ids_since_consolidation, subconscious_message, "
            "last_consolidation_at, consolidation_in_progress, consolidation_started_at, self_model_id "
            "FROM memu_conversation_state ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        if row:
            for f in ("all_categories_summary", "memory_cache", "intentions_active", "retrieve_rewrite_angle",
                      "retrieval_ids_since_consolidation", "prior_context_ids_since_consolidation",
                      "subconscious_message", "last_consolidation_at", "consolidation_in_progress",
                      "consolidation_started_at", "self_model_id"):
                if row[f] is not None:
                    updates[f] = row[f]

    if "memu_memory_categories" in tables and not updates.get("all_categories_summary"):
        cats = con.execute(
            "SELECT name, summary FROM memu_memory_categories WHERE summary IS NOT NULL AND summary != '' ORDER BY name"
        ).fetchall()
        if cats:
            parts = [f"## {c['name']}\n{c['summary']}" for c in cats if c["name"] and c["summary"]]
            if parts:
                updates["all_categories_summary"] = "\n\n".join(parts)

    if updates:
        write(con, updates)
