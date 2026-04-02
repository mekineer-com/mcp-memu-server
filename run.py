#!/usr/bin/env python3
"""Start the mcp-memu-server without relying on a shell.

Design goals (for your setup):
- The plugin spawns Python directly (no /bin/sh, no run.sh).
- Server reads *all* paths (memu path, sqlite/postgres, resources, host/port) from config.json.
- Optional: run the server using a different Python (e.g. memu's own venv). We therefore do NOT
  auto-reexec into .venv unless explicitly requested.
- Write a pidfile so the plugin can stop/restart the server reliably.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _config_path() -> Path:
    return (ROOT / "config.json").resolve()


def _config_dir() -> Path:
    return ROOT


def _resolve_cfg_path(raw: str) -> Path:
    p = Path(str(raw or "")).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (_config_dir() / p).resolve()


def _read_json(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _pidfile_path(cfg: dict) -> Path:
    pf = cfg.get("pid_file") if isinstance(cfg.get("pid_file"), str) else None
    if pf and pf.strip():
        return _resolve_cfg_path(pf)
    return (ROOT / ".memu-server.pid").resolve()


def _write_pidfile(cfg: dict) -> None:
    pf = _pidfile_path(cfg)
    _write_text(pf, str(os.getpid()) + "\n")

    def _cleanup() -> None:
        try:
            if pf.exists():
                pf.unlink()
        except Exception:
            pass

    atexit.register(_cleanup)


def _parse_pid(text: str) -> int | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        pid = int(raw, 10)
    except Exception:
        return None
    if pid <= 0:
        return None
    return pid


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def _enforce_single_instance(cfg: dict) -> None:
    pf = _pidfile_path(cfg)
    if not pf.exists():
        return
    existing = _parse_pid(pf.read_text(encoding="utf-8"))
    if existing is None:
        return
    if existing == os.getpid():
        return
    if _is_pid_alive(existing):
        print(f"mcp-memu-server already running (pid={existing}, pid_file={pf})", file=sys.stderr)
        raise SystemExit(1)
    try:
        pf.unlink()
    except Exception:
        pass


def _maybe_add_memu_to_syspath(cfg: dict) -> None:
    memu_obj = cfg.get("memu") if isinstance(cfg.get("memu"), dict) else {}
    memu_path = memu_obj.get("path") if isinstance(memu_obj.get("path"), str) else ""
    if not memu_path.strip():
        return
    p = _resolve_cfg_path(memu_path)
    # Accept either a repo root (contains pyproject.toml) or the package dir.
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _should_force_venv(cfg: dict) -> bool:
    py = cfg.get("python") if isinstance(cfg.get("python"), dict) else {}
    return bool(py.get("force_venv"))


def _venv_python(root: Path) -> Path:
    # Linux/macOS
    for p in (root / ".venv" / "bin" / "python", root / ".venv" / "bin" / "python3"):
        if p.exists():
            return p
    # Windows
    return root / ".venv" / "Scripts" / "python.exe"


def _reexec_into_venv_if_requested(cfg: dict) -> None:
    if not _should_force_venv(cfg):
        return
    vpy = _venv_python(ROOT)
    if not vpy.exists():
        return
    try:
        if Path(sys.executable).resolve() == vpy.resolve():
            return
    except Exception:
        pass
    os.execv(str(vpy), [str(vpy), str(__file__), *sys.argv[1:]])


def _listen(cfg: dict) -> tuple[str, int]:
    listen = cfg.get("listen") if isinstance(cfg.get("listen"), dict) else {}
    cfg_host = listen.get("host") if isinstance(listen.get("host"), str) else None
    cfg_port = listen.get("port")
    try:
        cfg_port_i = int(cfg_port) if cfg_port is not None else None
    except Exception:
        cfg_port_i = None

    host = cfg_host or "127.0.0.1"
    port = cfg_port_i or 8099

    return host, port


def main() -> None:
    os.chdir(str(ROOT))

    cfg_path = _config_path()
    cfg = _read_json(cfg_path)

    # Optional: force server-local venv.
    _reexec_into_venv_if_requested(cfg)

    # Ensure memu import works even if memu is not installed into this venv.
    _maybe_add_memu_to_syspath(cfg)

    # Single-instance guard: refuse startup if another live server owns pid_file.
    _enforce_single_instance(cfg)

    # pidfile for plugin stop/restart.
    _write_pidfile(cfg)

    host, port = _listen(cfg)

    try:
        import uvicorn  # type: ignore
    except Exception:
        print("Missing uvicorn (or venv not set up).", file=sys.stderr)
        print("If you want a dedicated venv here:", file=sys.stderr)
        print("  python3 -m venv .venv", file=sys.stderr)
        print("  .venv/bin/pip install -e .", file=sys.stderr)
        raise

    uvicorn.run("app.main:app", host=host, port=port)


if __name__ == "__main__":
    main()
