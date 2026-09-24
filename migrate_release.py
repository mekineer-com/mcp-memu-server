#!/usr/bin/env python3
"""Apply one release's rerunnable schema and data migrations to every soul DB."""
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from app.config import load_config, sqlite_dir_from_cfg, sqlite_dsn_from_path
from memu.database.sqlite.sqlite import SQLiteStore


class ReleaseScope(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


Migration = tuple[str, Callable[[sqlite3.Connection], None]]
MIGRATIONS: tuple[Migration, ...] = ()


def _initialize_schema(database: Path, embedding_profile: str) -> None:
    store = SQLiteStore(
        dsn=sqlite_dsn_from_path(database),
        scope_model=ReleaseScope,
        embedding_profile=embedding_profile,
    )
    store.close()


def migrate_database(
    database: Path,
    embedding_profile: str,
    migrations: tuple[Migration, ...] = MIGRATIONS,
) -> None:
    _initialize_schema(database, embedding_profile)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS openalma_release_migrations ("
            "migration_id TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        applied = {
            row[0] for row in connection.execute(
                "SELECT migration_id FROM openalma_release_migrations"
            )
        }
        for migration_id, migration in migrations:
            if migration_id in applied:
                continue
            migration(connection)
            connection.execute(
                "INSERT INTO openalma_release_migrations (migration_id) VALUES (?)",
                (migration_id,),
            )


def main() -> None:
    config = load_config()
    metadata = (config.get("storage") or {}).get("metadata_store") or {}
    embedding_profile = str(metadata.get("embedding_profile") or "").strip()
    if not embedding_profile:
        raise RuntimeError("storage.metadata_store.embedding_profile is required for release migration")
    sqlite_dir = sqlite_dir_from_cfg(config, str(metadata.get("dsn") or ""))
    databases = sorted(sqlite_dir.glob("*.db"))
    if not databases:
        raise RuntimeError(f"No soul databases found in {sqlite_dir}")
    for database in databases:
        migrate_database(database, embedding_profile)
        print(f"Migrated {database.name}")


if __name__ == "__main__":
    main()
