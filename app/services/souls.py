from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StrictBool

from app.config import normalize_sqlite_dsn, sanitize_db_filename, sqlite_dir_from_cfg, sqlite_path_for_scope


_IDENTITY_TABLE = "soul_identity"
# ponytail: serialize short setup writes; per-target locks if setup throughput matters.
_CREATE_LOCK = threading.Lock()


class SoulCreate(BaseModel):
    user_id: str
    soul_id: str
    use_existing: StrictBool


def _scope(body: SoulCreate) -> tuple[str, str]:
    user_id, soul_id = body.user_id.strip(), body.soul_id.strip()
    if not user_id or not soul_id:
        raise HTTPException(status_code=422, detail="user_id and soul_id are required")
    if user_id.casefold() == soul_id.casefold():
        raise HTTPException(status_code=422, detail="user_id and soul_id must differ")
    if sanitize_db_filename(soul_id) == "unknown" and soul_id.casefold() != "unknown":
        raise HTTPException(status_code=422, detail="Invalid soul_id")
    return user_id, soul_id


def _canonical_path(config: dict[str, Any], soul_id: str) -> Path:
    path = sqlite_path_for_scope(config, _base_dsn(config), {"soul_id": soul_id})
    if path is None:
        raise RuntimeError("soul_id is required")
    return path


def _base_dsn(config: dict[str, Any]) -> str:
    return normalize_sqlite_dsn(str((config.get("storage", {}).get("metadata_store") or {}).get("dsn") or ""))


def _readonly_connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)


def _identities(path: Path) -> set[tuple[str, str]]:
    try:
        con = _readonly_connect(path)
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail="Soul database discovery is unavailable") from exc
    try:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        identities: set[tuple[str, str]] = set()
        if _IDENTITY_TABLE in tables:
            row = con.execute(
                "SELECT user_id, soul_id FROM soul_identity WHERE id = 1"
            ).fetchone()
            if row:
                identities.add(row)
        for table, query in (
            ("categories", "SELECT DISTINCT user_id, soul_id FROM categories WHERE user_id != '' AND soul_id != ''"),
            ("conversations", "SELECT DISTINCT user_id, soul_id FROM conversations WHERE user_id != '' AND soul_id != ''"),
        ):
            if table not in tables:
                continue
            columns = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
            if {"user_id", "soul_id"} <= columns:
                identities.update(con.execute(query))
        return identities
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail="Soul database discovery is unavailable") from exc
    finally:
        con.close()


def _known_at_path(config: dict[str, Any], path: Path, user_id: str, soul_id: str) -> bool:
    try:
        identities = _identities(path)
        return (user_id, soul_id) in identities and {sid for _, sid in identities} == {soul_id}
    except HTTPException:
        _collision()


def list_souls(config: dict[str, Any], user_id: str) -> list[str]:
    user_id = user_id.strip()
    if not user_id:
        raise HTTPException(status_code=422, detail="user_id is required")
    sqlite_dir = sqlite_dir_from_cfg(config, _base_dsn(config))
    souls: set[str] = set()
    try:
        candidates = list(sqlite_dir.iterdir())
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise HTTPException(status_code=503, detail="Soul database directory is unavailable") from exc
    for path in candidates:
        if path.suffix != ".db":
            continue
        if path.is_symlink() or not path.is_file():
            continue
        identities = _identities(path)
        for stored_user, soul_id in identities:
            if stored_user == user_id and path == _canonical_path(config, soul_id):
                souls.add(soul_id)
    return sorted(souls)


def _collision() -> None:
    raise HTTPException(
        status_code=409,
        detail={
            "reason": "sanitized_collision",
            "message": "A database already occupies this sanitized soul name.",
        },
    )


def create_soul(config: dict[str, Any], body: SoulCreate) -> dict[str, Any]:
    user_id, soul_id = _scope(body)
    path = _canonical_path(config, soul_id)
    directory = sqlite_dir_from_cfg(config, _base_dsn(config))
    target = directory / f"{sanitize_db_filename(soul_id)}.db"
    if target.is_symlink() or path.parent != directory:
        _collision()
    with _CREATE_LOCK:
        if path.exists():
            if _known_at_path(config, path, user_id, soul_id):
                if body.use_existing:
                    return {"soul_id": soul_id, "created": False}
                raise HTTPException(
                    status_code=409,
                    detail={
                        "reason": "existing_exact",
                        "message": "Soul already exists; set use_existing to true to reuse it.",
                    },
                )
            _collision()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError:
            _collision()
        os.close(fd)
        try:
            con = sqlite3.connect(path)
            try:
                con.execute(
                    "CREATE TABLE soul_identity (id INTEGER PRIMARY KEY CHECK (id = 1), user_id TEXT NOT NULL, soul_id TEXT NOT NULL)"
                )
                con.execute(
                    "INSERT INTO soul_identity (id, user_id, soul_id) VALUES (1, ?, ?)",
                    (user_id, soul_id),
                )
                con.commit()
            finally:
                con.close()
        except sqlite3.Error as exc:
            raise HTTPException(status_code=500, detail="Could not initialize soul identity") from exc
    return {"soul_id": soul_id, "created": True}


def register_soul_routes(
    app: FastAPI,
    *,
    get_config: Callable[[], dict[str, Any]],
    prefix: str = "",
    dependencies: list[Any] | None = None,
) -> None:
    route_dependencies = dependencies or []

    @app.get(f"{prefix}/souls", dependencies=route_dependencies)
    def souls(user_id: str) -> dict[str, list[str]]:
        return {"souls": list_souls(get_config(), user_id)}

    @app.post(f"{prefix}/souls", dependencies=route_dependencies)
    def souls_create(body: SoulCreate) -> dict[str, Any]:
        return create_soul(get_config(), body)
