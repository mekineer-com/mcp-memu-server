from __future__ import annotations

import os
import logging
import sqlite3
import tempfile
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StrictBool

from app.config import normalize_sqlite_dsn, sanitize_db_filename, sqlite_dir_from_cfg, sqlite_path_for_scope


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
        if "soul_identity" in tables:
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


def _write_identity(con: sqlite3.Connection, user_id: str, soul_id: str) -> None:
    con.execute("CREATE TABLE IF NOT EXISTS soul_identity (id INTEGER PRIMARY KEY CHECK (id = 1), user_id TEXT NOT NULL, soul_id TEXT NOT NULL)")
    con.execute("INSERT INTO soul_identity VALUES (1, ?, ?)", (user_id, soul_id))


def _reuse(path: Path, user_id: str, soul_id: str, consent: bool) -> dict[str, Any]:
    try:
        identities = _identities(path)
    except HTTPException:
        _collision()
    if not identities:
        with closing(sqlite3.connect(path)) as con:
            con.execute("BEGIN IMMEDIATE")
            identities = _identities(path)
            if not identities:
                tables = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
                if any(con.execute('SELECT 1 FROM "' + name.replace('"', '""') + '" LIMIT 1').fetchone() for (name,) in tables):
                    _collision()
                if not consent:
                    raise HTTPException(status_code=409, detail={
                        "reason": "existing_exact", "message": "An empty database exists. Assign it to this soul?",
                    })
                _write_identity(con, user_id, soul_id)
                con.commit()
                identities = {(user_id, soul_id)}
    if (user_id, soul_id) not in identities or {sid for _, sid in identities} != {soul_id}:
        _collision()
    if not consent:
        raise HTTPException(status_code=409, detail={
            "reason": "existing_exact", "message": "Soul already exists. Use its existing database?",
        })
    return {"soul_id": soul_id, "created": False}


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
        if path.suffix != ".db" or path.is_symlink() or not path.is_file():
            continue
        try:
            identities = _identities(path)
        except HTTPException:
            logging.getLogger(__name__).warning("Skipping unreadable soul database %s", path)
            continue
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
    if path.exists():
        return _reuse(path, user_id, soul_id, body.use_existing)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp") as staged:
        with closing(sqlite3.connect(staged.name)) as con:
            _write_identity(con, user_id, soul_id)
            con.commit()
        try:
            os.link(staged.name, path)  # Publish only a complete DB; never replace an occupied name.
        except FileExistsError:
            return _reuse(path, user_id, soul_id, body.use_existing)
    return {"soul_id": soul_id, "created": True}


def register_soul_routes(
    app: FastAPI,
    *,
    get_config: Callable[[], dict[str, Any]],
    prefix: str = "",
    dependencies: list[Any] | None = None,
) -> None:
    @app.get(f"{prefix}/souls", dependencies=dependencies)
    def souls(user_id: str) -> dict[str, list[str]]:
        return {"souls": list_souls(get_config(), user_id)}

    @app.post(f"{prefix}/souls", dependencies=dependencies)
    def souls_create(body: SoulCreate) -> dict[str, Any]:
        return create_soul(get_config(), body)
