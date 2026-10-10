from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StrictBool

from app.config import (
    SoulIdError,
    SoulNameConflictError,
    normalize_sqlite_dsn,
    procedural_db_path,
    sqlite_dir_from_cfg,
    sqlite_file_from_dsn,
    sqlite_path_for_scope,
    validate_soul_id,
)
from app.services.owner import read_owner, validate_user_id
from app.services import soul_state


_CREATE_LOCK = threading.Lock()


class SoulCreate(BaseModel):
    soul_id: str
    use_existing: StrictBool
    user_name: str | None = None


def _base_dsn(config: dict[str, Any]) -> str:
    return normalize_sqlite_dsn(str((config.get("storage", {}).get("metadata_store") or {}).get("dsn") or ""))


def _soul_id(value: Any) -> str:
    try:
        return validate_soul_id(value)
    except SoulIdError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _path(config: dict[str, Any], soul_id: str) -> Path:
    try:
        path = sqlite_path_for_scope(config, _base_dsn(config), {"soul_id": soul_id})
    except SoulNameConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if path is None:
        raise RuntimeError("soul_id is required")
    base_path = sqlite_file_from_dsn(_base_dsn(config))
    if base_path is not None and path.resolve() == base_path.resolve():
        raise HTTPException(status_code=409, detail="Soul name is reserved by the base database")
    if path.resolve() == procedural_db_path(config).resolve():
        raise HTTPException(status_code=409, detail="Soul name is reserved by the procedural database")
    if path.is_symlink():
        raise HTTPException(status_code=409, detail="A symlink occupies this soul name")
    if path.exists() and not path.is_file():
        raise HTTPException(status_code=409, detail="A non-file occupies this soul name")
    return path


def publish_soul_db(path: Path, user_name: str) -> bool:
    """Publish the Soul and its immutable human name together."""
    user_name = validate_user_id(user_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        suffix=".tmp",
        delete_on_close=False,
    ) as staged:
        with closing(sqlite3.connect(staged.name)) as con:
            con.execute("PRAGMA user_version=1")
            soul_state.ensure_schema(con)
            con.execute("UPDATE soul_state SET user_name = ? WHERE id = 1", (user_name,))
            con.commit()
        try:
            os.link(staged.name, path)
        except FileExistsError:
            return False
    return True


def read_soul_name(path: Path | None) -> str:
    """Read existing identity without creating files, schema, state or WAL."""
    missing = HTTPException(status_code=409, detail="Soul user name is missing; this database has no persona binding")
    if path is None or not path.is_file() or path.is_symlink():
        raise missing
    try:
        # The write-once name is committed in the staged main DB, never in WAL.
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)) as con:
            row = con.execute("SELECT user_name FROM soul_state WHERE id = 1").fetchone()
        if row is None or not isinstance(row[0], str):
            raise missing
        return validate_user_id(row[0])
    except (sqlite3.Error, ValueError) as exc:
        raise missing from exc


def list_souls(config: dict[str, Any]) -> list[str]:
    directory = sqlite_dir_from_cfg(config, _base_dsn(config))
    base_path = sqlite_file_from_dsn(_base_dsn(config))
    procedural_path = procedural_db_path(config).resolve()
    try:
        return sorted(
            path.stem
            for path in directory.iterdir()
            if path.suffix == ".db"
            and not path.is_symlink()
            and path.is_file()
            and (base_path is None or path.resolve() != base_path.resolve())
            and path.resolve() != procedural_path
        )
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise HTTPException(status_code=503, detail="Soul database directory is unavailable") from exc


def create_soul(config: dict[str, Any], body: SoulCreate) -> dict[str, Any]:
    soul_id = _soul_id(body.soul_id)
    owner_id = read_owner(config)
    if owner_id is None:
        raise HTTPException(status_code=409, detail="OpenAlma owner has not been created")
    if owner_id.casefold() == soul_id.casefold():
        raise HTTPException(status_code=422, detail="Soul name must differ from the owner name")
    with _CREATE_LOCK:
        path = _path(config, soul_id)
        created = False
        if not path.exists():
            try:
                user_name = validate_user_id(body.user_name if body.user_name is not None else owner_id)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if user_name.casefold() == soul_id.casefold():
                raise HTTPException(status_code=422, detail="Soul name must differ from the user's name")
            created = publish_soul_db(path, user_name)
        if not created and not body.use_existing:
            raise HTTPException(
                status_code=409,
                detail={"reason": "existing_exact", "message": "Soul already exists. Use its existing database?"},
            )
        if not created:
            read_soul_name(path)
        return {"soul_id": soul_id, "created": created}


def register_soul_routes(
    app: FastAPI,
    *,
    get_config: Callable[[], dict[str, Any]],
    prefix: str = "",
    dependencies: list[Any] | None = None,
) -> None:
    @app.get(f"{prefix}/souls", dependencies=dependencies)
    def souls() -> dict[str, list[str]]:
        return {"souls": list_souls(get_config())}

    @app.post(f"{prefix}/souls", dependencies=dependencies)
    def souls_create(body: SoulCreate) -> dict[str, Any]:
        return create_soul(get_config(), body)

    @app.get(f"{prefix}/souls/{{soul_id}}", dependencies=dependencies)
    def soul_metadata(soul_id: str) -> dict[str, str]:
        soul_id = _soul_id(soul_id)
        return {"soul_id": soul_id, "user_name": read_soul_name(_path(get_config(), soul_id))}
