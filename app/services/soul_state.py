"""Soul-level state — single row, shared across all conversations."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

from app.db import json_from_db, json_to_db, normalize_text_list
from app.services.intention_state import validate_intentions, normalize_memory_cache
from app.services.payload import parse_iso_datetime

CONSOLIDATION_UNFINISHED = "Consolidation interrupted before completion. Retry in launcher."


def consolidation_failure(state: dict[str, Any]) -> str | None:
    error = state.get("last_consolidation_error")
    failed_at = parse_iso_datetime(state.get("last_consolidation_error_at"))
    completed_at = parse_iso_datetime(state.get("last_consolidation_at"))
    if error and failed_at and (completed_at is None or failed_at > completed_at):
        return str(error)
    return None


def activity_pause(state: dict[str, Any], *, memorize_running: bool, consolidation_running: bool) -> str | None:
    failure = state.get("memorize_failure")
    if failure and (failure.get("paused") or not memorize_running):
        return str(failure["error"])
    error = consolidation_failure(state)
    if error and (error != CONSOLIDATION_UNFINISHED or not consolidation_running):
        return error
    return None


def ensure_schema(con: sqlite3.Connection) -> None:
    table_exists = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'soul_state'"
    ).fetchone() is not None
    missing_row = table_exists and con.execute("SELECT COUNT(*) FROM soul_state").fetchone()[0] == 0
    owns_initialization = (not table_exists or missing_row) and not con.in_transaction
    if owns_initialization:
        con.execute("BEGIN")

    con.execute("""
CREATE TABLE IF NOT EXISTS soul_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    narrative_self TEXT,
    narrative_self_previous TEXT,
    narrative_self_approved TEXT,
    summaries_revision INTEGER NOT NULL DEFAULT 0,
    memory_cache JSON DEFAULT '[]',
    intentions_active JSON NOT NULL DEFAULT '[]',
    retrieve_rewrite_angle INTEGER DEFAULT 0,
    retrieval_ids_since_consolidation JSON DEFAULT '[]',
    prior_context_ids_since_consolidation JSON DEFAULT '[]',
    apimw_message_to_self TEXT,
    last_consolidation_at DATETIME,
    last_consolidation_error TEXT,
    last_consolidation_error_at DATETIME,
    memorize_failure TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
)""")
    if con.execute("SELECT COUNT(*) FROM soul_state").fetchone()[0] == 0:
        con.execute(
            "INSERT INTO soul_state (id, intentions_active, updated_at) VALUES (1, '[]', ?)",
            (datetime.now(UTC).isoformat(),),
        )
    if owns_initialization:
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
        "summaries_revision": int(row["summaries_revision"] or 0),
        "memory_cache": normalize_memory_cache(json_from_db(row["memory_cache"])),
        "intentions_active": validate_intentions(json_from_db(row["intentions_active"])),
        "retrieve_rewrite_angle": int(row["retrieve_rewrite_angle"] or 0),
        "retrieval_ids_since_consolidation": normalize_text_list(row["retrieval_ids_since_consolidation"]),
        "prior_context_ids_since_consolidation": normalize_text_list(row["prior_context_ids_since_consolidation"]),
        "apimw_message_to_self": (str(row["apimw_message_to_self"] or "").strip() or None),
        "last_consolidation_at": row["last_consolidation_at"],
        "last_consolidation_error": row["last_consolidation_error"],
        "last_consolidation_error_at": row["last_consolidation_error_at"],
        "memorize_failure": json_from_db(row["memorize_failure"]),
        "updated_at": row["updated_at"],
    }


def defaults() -> dict[str, Any]:
    return {
        "narrative_self": None,
        "narrative_self_previous": None,
        "narrative_self_approved": None,
        "summaries_revision": 0,
        "memory_cache": [],
        "intentions_active": [],
        "retrieve_rewrite_angle": 0,
        "retrieval_ids_since_consolidation": [],
        "prior_context_ids_since_consolidation": [],
        "apimw_message_to_self": None,
        "last_consolidation_at": None,
        "last_consolidation_error": None,
        "last_consolidation_error_at": None,
        "memorize_failure": None,
        "updated_at": None,
    }


_JSON_FIELDS = {
    "memorize_failure",
    "intentions_active", "memory_cache",
    "retrieval_ids_since_consolidation", "prior_context_ids_since_consolidation",
}

_VALID_FIELDS = {
    "memorize_failure",
    "memory_cache", "intentions_active",
    "retrieve_rewrite_angle", "retrieval_ids_since_consolidation",
    "prior_context_ids_since_consolidation", "apimw_message_to_self",
    "last_consolidation_at", "last_consolidation_error",
    "last_consolidation_error_at",
}


def write(con: sqlite3.Connection, updates: dict[str, Any]) -> None:
    """Update soul_state fields. Does NOT commit — caller owns the transaction."""
    ensure_schema(con)
    fields = {k: v for k, v in updates.items() if k in _VALID_FIELDS}
    if not fields:
        return
    if "intentions_active" in fields:
        fields["intentions_active"] = validate_intentions(fields["intentions_active"])
    if "memory_cache" in fields:
        fields["memory_cache"] = normalize_memory_cache(fields["memory_cache"])
    if fields.get("memorize_failure") is not None:
        failure = fields["memorize_failure"]
        if not isinstance(failure, dict) or not isinstance(failure.get("conversation_id"), str) or not isinstance(failure.get("error"), str) or not isinstance(failure.get("paused"), bool) or not isinstance(failure.get("targets"), dict):
            raise ValueError("invalid memorize failure record")
    if "retrieval_ids_since_consolidation" in fields:
        fields["retrieval_ids_since_consolidation"] = normalize_text_list(fields["retrieval_ids_since_consolidation"])
    if "prior_context_ids_since_consolidation" in fields:
        fields["prior_context_ids_since_consolidation"] = normalize_text_list(fields["prior_context_ids_since_consolidation"])
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
        else:
            params.append(value)
    con.execute(f"UPDATE soul_state SET {', '.join(assignments)} WHERE id = 1", tuple(params))
