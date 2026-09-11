from __future__ import annotations

import os
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import sqlite_dir_from_cfg


class OwnerIdentityError(RuntimeError):
    pass


class OwnerMissingError(OwnerIdentityError):
    pass


class OwnerMismatchError(OwnerIdentityError):
    pass


class OwnerStorageError(OwnerIdentityError):
    pass


class OwnerCreate(BaseModel):
    user_id: str


_CREATE_LOCK = threading.Lock()


def validate_user_id(value: Any) -> str:
    user_id = str(value or "").strip()
    if not user_id or any(ord(char) < 32 or ord(char) == 127 for char in user_id):
        raise ValueError("Invalid user name")
    return user_id


def owner_path(config: dict[str, Any]) -> Path:
    metadata = (config.get("storage") or {}).get("metadata_store") or {}
    return sqlite_dir_from_cfg(config, str(metadata.get("dsn") or "")) / "owner.txt"


def read_owner(config: dict[str, Any]) -> str | None:
    path = owner_path(config)
    try:
        return validate_user_id(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except ValueError as exc:
        raise OwnerStorageError(f"Invalid OpenAlma owner file: {path}") from exc
    except OSError as exc:
        raise OwnerStorageError(f"Cannot read OpenAlma owner file: {path}") from exc


def create_owner(config: dict[str, Any], value: Any) -> tuple[str, bool]:
    user_id = validate_user_id(value)
    path = owner_path(config)
    with _CREATE_LOCK:
        existing = read_owner(config)
        if existing is not None:
            if existing != user_id:
                raise OwnerMismatchError(f"OpenAlma owner is {existing!r}, not {user_id!r}")
            return existing, False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent) as staged:
                staged.write(f"{user_id}\n")
                staged.flush()
                os.fsync(staged.fileno())
                try:
                    os.link(staged.name, path)
                except FileExistsError:
                    existing = read_owner(config)
                    if existing != user_id:
                        raise OwnerMismatchError(
                            f"OpenAlma owner is {existing!r}, not {user_id!r}"
                        )
                    return user_id, False
        except OSError as exc:
            raise OwnerStorageError(f"Cannot create OpenAlma owner file: {path}") from exc
    return user_id, True


def require_owner(config: dict[str, Any], value: Any) -> str:
    try:
        user_id = validate_user_id(value)
    except ValueError as exc:
        raise OwnerMismatchError("A valid OpenAlma owner is required") from exc
    owner = read_owner(config)
    if owner is None:
        raise OwnerMissingError("OpenAlma owner has not been created")
    if owner != user_id:
        raise OwnerMismatchError(f"OpenAlma owner is {owner!r}, not {user_id!r}")
    return owner


def register_owner_routes(
    app: FastAPI,
    *,
    get_config: Callable[[], dict[str, Any]],
    prefix: str = "",
    dependencies: list[Any] | None = None,
) -> None:
    @app.get(f"{prefix}/owner", dependencies=dependencies)
    def owner_get() -> dict[str, str | None]:
        return {"user_id": read_owner(get_config())}

    @app.post(f"{prefix}/owner", dependencies=dependencies)
    def owner_create(body: OwnerCreate) -> dict[str, str | bool]:
        try:
            user_id, created = create_owner(get_config(), body.user_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OwnerMismatchError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"user_id": user_id, "created": created}
