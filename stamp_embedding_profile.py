#!/usr/bin/env python3
"""Validate and explicitly stamp a stopped legacy OpenAI memU database."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

PROFILE = "text-embedding-3-large:3072"
VECTOR_BYTES = 3072 * 4
TABLES = ("memory_items", "categories", "resources")


def stamp(database: Path, *, apply: bool) -> dict[str, object]:
    database = database.expanduser().resolve()
    if not database.is_file():
        raise ValueError(f"database does not exist: {database}")

    backup: Path | None = None
    with sqlite3.connect(database) as conn:
        invalid = {
            table: conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE embedding IS NOT NULL "
                "AND (typeof(embedding) != 'blob' OR length(embedding) != ?)",
                (VECTOR_BYTES,),
            ).fetchone()[0]
            for table in TABLES
        }
        if any(invalid.values()):
            raise ValueError(f"non-canonical embeddings: {invalid}")

        has_profile_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'embedding_profile'"
        ).fetchone()
        existing = (
            conn.execute("SELECT profile FROM embedding_profile WHERE id = 1").fetchone()
            if has_profile_table
            else None
        )
        if existing and existing[0] != PROFILE:
            raise ValueError(f"database already stamped as {existing[0]}")

        if apply and not existing:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            backup = database.with_name(f"{database.stem}-pre-profile-{stamp}{database.suffix}")
            with sqlite3.connect(backup) as destination:
                conn.backup(destination)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embedding_profile "
                "(id INTEGER PRIMARY KEY CHECK (id = 1), profile TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO embedding_profile (id, profile) VALUES (1, ?)",
                (PROFILE,),
            )
            conn.commit()
    return {
        "database": str(database),
        "profile": existing[0] if existing else PROFILE,
        "status": "already_stamped" if existing else "stamped" if apply else "ready",
        "backup": str(backup) if backup else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        result = stamp(args.database, apply=args.apply)
    except (ValueError, sqlite3.Error, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
