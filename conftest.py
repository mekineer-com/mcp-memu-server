"""Pytest bootstrap: add the memu engine to sys.path so `from memu.app import ...` works.

The server normally loads memu via run.py's sys.path injection (driven by config.json).
Tests don't go through run.py, so without this bootstrap `from app import main` fails
on `from memu.app import MemoryService` and any test that imports main silently skips.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
