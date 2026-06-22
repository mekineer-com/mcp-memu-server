"""Pytest bootstrap: add the memu engine to sys.path so `from memu.app import ...` works.

The server normally loads memu via run.py's sys.path injection (driven by config.json).
Tests don't go through run.py, so without this bootstrap `from app import main` fails
on `from memu.app import MemoryService` and any test that imports main silently skips.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent


def _add_memu_to_syspath() -> None:
    cfg_path = _ROOT / "config.json"
    if not cfg_path.exists():
        return
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return
    memu_obj = cfg.get("memu") if isinstance(cfg.get("memu"), dict) else {}
    memu_path = memu_obj.get("path") if isinstance(memu_obj.get("path"), str) else ""
    if not memu_path.strip():
        return
    resolved = (_ROOT / memu_path).resolve() if not Path(memu_path).is_absolute() else Path(memu_path)
    if resolved.exists() and str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))


_add_memu_to_syspath()


@pytest.fixture(autouse=True)
def _isolate_memu_sqlite_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep tests from ever resolving the operator's live sqlite directory.

    Several server helpers resolve the active DB through app.main._CONFIG and
    app.main._STORAGE_STATUS at runtime. If an individual test forgets to patch
    one background path, it must still land in this test-local directory, never
    in memu/sqlite/Siri.db.
    """
    from app import main

    sqlite_dir = tmp_path / "sqlite"
    resources_dir = tmp_path / "resources"
    sqlite_dir.mkdir()
    resources_dir.mkdir()
    dsn = f"sqlite:///{(sqlite_dir / 'memu.db').as_posix()}"

    cfg = deepcopy(main._CONFIG)
    storage = cfg.setdefault("storage", {})
    storage["sqlite_dir"] = str(sqlite_dir)
    storage["resources_dir"] = str(resources_dir)
    metadata_store = storage.setdefault("metadata_store", {})
    metadata_store["provider"] = "sqlite"
    metadata_store["dsn"] = dsn

    status = deepcopy(main._STORAGE_STATUS)
    status["dsn"] = dsn
    status["sqlite_dir"] = str(sqlite_dir)
    status["sqlite_path"] = str(sqlite_dir / "memu.db")
    status["sqlite_parent"] = str(sqlite_dir)

    live_sqlite_dir = (_ROOT.parent / "memu" / "sqlite").resolve()
    original_connect = main._sqlite_connect
    original_ensure_nonempty = main._sqlite_ensure_nonempty

    def _reject_live_path(path: object) -> Path:
        resolved = Path(path).expanduser().resolve()
        if resolved == live_sqlite_dir or live_sqlite_dir in resolved.parents:
            raise AssertionError(f"test attempted to touch live sqlite path: {resolved}")
        return resolved

    def _guarded_connect(path: object, *args, **kwargs):
        return original_connect(_reject_live_path(path), *args, **kwargs)

    def _guarded_ensure_nonempty(path: object, *args, **kwargs):
        return original_ensure_nonempty(_reject_live_path(path), *args, **kwargs)

    monkeypatch.setattr(main, "_CONFIG", cfg)
    monkeypatch.setattr(main, "_STORAGE_STATUS", status)
    monkeypatch.setattr(main, "_sqlite_connect", _guarded_connect)
    monkeypatch.setattr(main, "_sqlite_ensure_nonempty", _guarded_ensure_nonempty)
