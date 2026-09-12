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
    sqlite_dir_from_cfg,
    sqlite_file_from_dsn,
    sqlite_path_for_scope,
    validate_soul_id,
)
from app.services.owner import read_owner


_CREATE_LOCK = threading.Lock()


class SoulCreate(BaseModel):
    soul_id: str
    use_existing: StrictBool


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
    if path.is_symlink():
        raise HTTPException(status_code=409, detail="A symlink occupies this soul name")
    if path.exists() and not path.is_file():
        raise HTTPException(status_code=409, detail="A non-file occupies this soul name")
    return path


def publish_soul_db(path: Path) -> bool:
    """Atomically publish a minimal SQLite file; return whether this call won."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        suffix=".tmp",
        delete_on_close=False,
    ) as staged:
        with closing(sqlite3.connect(staged.name)) as con:
            con.execute("PRAGMA user_version=1")
            con.commit()
        try:
            os.link(staged.name, path)
        except FileExistsError:
            return False
    return True


def list_souls(config: dict[str, Any]) -> list[str]:
    directory = sqlite_dir_from_cfg(config, _base_dsn(config))
    base_path = sqlite_file_from_dsn(_base_dsn(config))
    try:
        return sorted(
            path.stem
            for path in directory.iterdir()
            if path.suffix == ".db"
            and not path.is_symlink()
            and path.is_file()
            and (base_path is None or path.resolve() != base_path.resolve())
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
        created = publish_soul_db(path) if not path.exists() else False
        if not created and not body.use_existing:
            raise HTTPException(
                status_code=409,
                detail={"reason": "existing_exact", "message": "Soul already exists. Use its existing database?"},
            )
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
