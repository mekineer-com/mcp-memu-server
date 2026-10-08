#!/usr/bin/env python3
"""Apply one release's rerunnable schema and data migrations to every soul DB."""
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from app.config import (
    database_config_from_cfg,
    load_config,
    procedural_db_path,
    sqlite_dir_from_cfg,
    sqlite_dsn_from_path,
    sqlite_file_from_dsn,
)
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
        connection.commit()
        for migration_id, migration in migrations:
            if migration_id in applied:
                continue
            try:
                connection.execute("BEGIN")
                migration(connection)
                connection.execute(
                    "INSERT INTO openalma_release_migrations (migration_id) VALUES (?)",
                    (migration_id,),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise


def soul_databases(config: dict) -> list[Path]:
    metadata = database_config_from_cfg(config)["metadata_store"]
    sqlite_dir = sqlite_dir_from_cfg(config, metadata["dsn"])
    base = sqlite_file_from_dsn(metadata["dsn"])
    procedural = procedural_db_path(config)
    return sorted(
        path for path in sqlite_dir.glob("*.db")
        if not path.name.startswith(".")
        and not path.is_symlink()
        and path.is_file()
        and (base is None or path.resolve() != base.resolve())
        and path.resolve() != procedural.resolve()
    )


def main() -> None:
    config = load_config()
    metadata = database_config_from_cfg(config)["metadata_store"]
    embedding_profile = metadata["embedding_profile"]
    databases = soul_databases(config)
    for database in databases:
        try:
            migrate_database(database, embedding_profile)
        except Exception:
            raise RuntimeError("Soul database migration failed") from None
    print(f"Migrated {len(databases)} soul database(s)")


if __name__ == "__main__":
    main()
