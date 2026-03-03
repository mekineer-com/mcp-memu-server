import hashlib
import json
import os
import traceback
import time
import uuid
import sqlite3
import re
import math
import sys
from datetime import datetime, timezone, timedelta, time as dtime
from pathlib import Path
from typing import Any, Optional

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


try:
    import pwd
except Exception:  # pragma: no cover
    pwd = None  # type: ignore

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from memu.app import MemoryService

app = FastAPI(title="mcp-memu-server", version="0.4.0")

# Build marker (helps verify you restarted into the expected code)
_BUILD_ID: str = "fix48.debloat.bloatRemoval.concepts"

# Sleep-based daily split guardrails
_SLEEP_SPLIT_MIN_LULL_SECONDS: int = 3 * 60 * 60  # 3 hours
_SLEEP_SPLIT_MAX_SPAN_DAYS: int = 2  # allow at most 2 days when no qualifying lull is found


def _has_category_content(c: dict[str, Any]) -> bool:
    """Return True if this category has any user-visible content.

    We intentionally do NOT fabricate content. We only consider fields that
    already exist on the category object.
    """
    try:
        summary = str(c.get("summary") or "").strip()
        desc = str(c.get("description") or "").strip()
        return bool(summary or desc)
    except Exception:
        return False


# -------------------------
# Process identity (for restart detection)
# -------------------------

_SERVER_INSTANCE_ID: str = os.getenv("MEMU_SERVER_INSTANCE_ID") or str(uuid.uuid4())
_SERVER_STARTED_AT_UNIX: float = time.time()

# -------------------------
# Recent request trace (debug)
# -------------------------
_LAST_CALLS: list[dict[str, Any]] = []

# Full HTTP trace (method/path/status/elapsed). This answers:
# "Is anything reaching the server from the plugin?"
_LAST_HTTP: list[dict[str, Any]] = []


@app.middleware("http")
async def _trace_requests(request: Request, call_next):
    t0 = time.time()
    status = 500
    try:
        resp = await call_next(request)
        status = getattr(resp, "status_code", 200)
        return resp
    finally:
        try:
            dt_ms = int((time.time() - t0) * 1000)
            _LAST_HTTP.append({
                "t": time.time(),
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "ms": dt_ms,
            })
            if len(_LAST_HTTP) > 200:
                del _LAST_HTTP[0: len(_LAST_HTTP) - 200]
        except Exception:
            pass



def _is_ephemeral_db(cfg: dict[str, Any]) -> bool:
    """Return True when the metadata store is expected to be wiped on server restart."""
    try:
        storage = cfg.get("storage") if isinstance(cfg.get("storage"), dict) else {}
        ms = storage.get("metadata_store") if isinstance(storage.get("metadata_store"), dict) else {}
        provider = str(ms.get("provider") or "").lower()
        dsn = str(ms.get("dsn") or "")

        if provider in ("memory", "inmemory", "in-memory"):
            return True
        if provider == "sqlite" and ":memory:" in dsn:
            return True
    except Exception:
        pass
    return False

# -------------------------
# User scoping (SillyTavern)
# -------------------------
# memU 1.4's DefaultUserModel includes user_id only. For SillyTavern we want per-user + per-character
# (+ optional session) isolation without patching memU itself.
class STUserModel(BaseModel):
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None


# -------------------------
# Config file (single source of truth for "default" service)
# -------------------------

def _home_dir() -> Path:
    h = os.getenv("HOME") or os.getenv("USERPROFILE") or "."
    return Path(h).expanduser().resolve()


def _default_config() -> dict[str, Any]:
    home = _home_dir()

    # If you keep memu source in a versioned folder (e.g. ~/apps/memu-1.4.0), default to that if present.
    memu_guess = None
    for cand in (home / "apps" / "memu-1.4.0", home / "apps" / "memu"):
        if cand.exists():
            memu_guess = cand
            break

    sqlite_path = Path(":memory:")  # placeholder; per-agent dbs resolved per-request
    resources_dir = home / "apps" / "memu" / "resources"

    return {
        # Optional: where memu source lives. run.py can add this to sys.path.
        "memu": {
            "path": str(memu_guess) if memu_guess else "",
        },
        # Optional: tell the plugin which python to spawn (e.g. memu's own venv).
        "python": {
            "executable": "",
            "force_venv": False,
        },
        "pid_file": str((home / "apps" / "mcp-memu-server" / ".memu-server.pid")),
        "listen": {"host": "127.0.0.1", "port": 8099},
        "llm": {
            "provider": "openai",
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "chat_model": "gpt-4o-mini",
            "embed_model": "text-embedding-3-small",
            "client_backend": "httpx",
            "endpoint_overrides": {},
        },
        "storage": {
            "resources_dir": str(resources_dir),
            "metadata_store": {
                "provider": "sqlite",
                # absolute path DSN form (4 slashes after sqlite:)
                "dsn": "sqlite:///:memory:" ,
                "ddl_mode": "create",
            },
            "sqlite_dir": str(sqlite_path.parent),
            "shared_sqlite_path": str(sqlite_path.parent / "shared.db"),
        },
        "categories": {
            "defaults": ["personal_info", "preferences", "relationships", "goals"],
            "max_total": 12,
            "allow_dynamic": True,
        },
        "mcp": {"http_path": "/mcp", "sse_path": "/sse"},
    }


def _config_path() -> Path:
    p = os.getenv("MEMU_SERVER_CONFIG") or os.getenv("MCP_MEMU_CONFIG") or "./config.json"
    return Path(p).expanduser().resolve()


def _config_dir() -> Path:
    try:
        return _config_path().parent
    except Exception:
        return Path.cwd()


def _load_config() -> dict[str, Any]:
    p = _config_path()
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        cfg = _default_config()
        p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        _ensure_storage_paths(cfg)
        return cfg
    try:
        raw = p.read_text(encoding="utf-8")
        cfg = json.loads(raw) if raw.strip() else _default_config()
        if not isinstance(cfg, dict):
            cfg = _default_config()
        _ensure_storage_paths(cfg)
        return cfg
    except Exception:
        traceback.print_exc()
        cfg = _default_config()
        _ensure_storage_paths(cfg)
        return cfg


# -------------------------
# Storage path helpers
# -------------------------

_STARTUP_WARNINGS: list[str] = []

# Last known storage diagnostics (used by /health).
_STORAGE_STATUS: dict[str, Any] = {
    "ok": None,
    "provider": None,
    "dsn": None,
    "sqlite_path": None,
    "sqlite_parent": None,
    "sqlite_exists": None,
    "sqlite_open_ok": None,
    "sqlite_mode": "per_agent_only",
    "sqlite_dir": None,
    "shared_db_path": None,
    "error": None,
}


def _sqlite_file_from_dsn(dsn: str) -> Path | None:
    """Parse a sqlite DSN into a filesystem path (minimal).

    Supports:
      sqlite:////abs/path.db
      sqlite:////abs/path.db?query
      sqlite:///abs-or-relative.db

    Returns None for :memory: or unknown formats.
    """
    if not isinstance(dsn, str):
        return None
    dsn = dsn.strip()
    if not dsn or ':memory:' in dsn:
        return None

    base = dsn.split('?', 1)[0]
    if base.startswith('sqlite:////'):
        return Path('/' + base[len('sqlite:////'):])
    if base.startswith('sqlite:///'):
        return Path(base[len('sqlite:///'):])
    return None


def _normalize_sqlite_dsn(dsn_or_path: str) -> str:
    """Accept either a sqlite DSN or a plain filesystem path.

    Users frequently paste a path like "/home/.../memu.db" instead of a DSN.
    This converts it into a correct absolute sqlite DSN.
    """
    raw = str(dsn_or_path or "").strip()
    if not raw:
        return raw
    if raw.startswith("sqlite:"):
        return raw
    if raw == ":memory:" or raw.lower() == "memory":
        return "sqlite:///:memory:"

    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (_config_dir() / p).resolve()
    # sqlite absolute path DSN needs 4 slashes after the scheme.
    return f"sqlite:////{p.as_posix().lstrip('/')}"


# -------------------------
# Per-agent SQLite isolation (always enabled)
# -------------------------

def _sqlite_dir_from_cfg(cfg: dict[str, Any], fallback_dsn: str | None = None) -> Path:
    storage = cfg.get('storage') if isinstance(cfg.get('storage'), dict) else {}
    d = storage.get('sqlite_dir')
    if isinstance(d, str) and d.strip():
        return Path(d).expanduser().resolve()

    if fallback_dsn:
        f = _sqlite_file_from_dsn(str(fallback_dsn))
        if f is not None:
            return f.expanduser().resolve().parent

    return (_config_dir() / 'sqlite').resolve()


def _ensure_storage_paths(cfg: dict[str, Any]) -> None:
    """Create directories needed for resources + sqlite on-disk storage.

    This keeps setup smooth (no manual mkdir) and converts a confusing
    sqlite "unable to open database file" into a clear startup warning.
    """
    global _STORAGE_STATUS
    try:
        storage = cfg.get('storage') if isinstance(cfg.get('storage'), dict) else {}

        # resources_dir
        resources_dir = storage.get('resources_dir')
        if isinstance(resources_dir, str) and resources_dir.strip():
            Path(resources_dir).expanduser().mkdir(parents=True, exist_ok=True)

        # sqlite db file parent + file
        ms = storage.get('metadata_store') if isinstance(storage.get('metadata_store'), dict) else {}
        provider = str(ms.get('provider') or '').lower()
        dsn = str(ms.get('dsn') or '')

        _STORAGE_STATUS = {
            "ok": True,
            "provider": provider or None,
            "dsn": dsn or None,
            "sqlite_path": None,
            "sqlite_parent": None,
            "sqlite_exists": None,
            "sqlite_open_ok": None,
            "sqlite_mode": "per_agent_only",
            "sqlite_dir": str(_sqlite_dir_from_cfg(cfg, dsn)),
            "error": None,
        }

        if provider == 'sqlite':
            dsn = _normalize_sqlite_dsn(dsn)
            # Keep normalized DSN visible in /health.
            _STORAGE_STATUS["dsn"] = dsn or None
            # Also persist normalization back into config in-memory (no disk write).
            try:
                ms["dsn"] = dsn
            except Exception:
                pass

            sqlite_dir = _sqlite_dir_from_cfg(cfg, dsn)
            if sqlite_dir is not None:
                try:
                    sqlite_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
            # KISS: do not create/open any DB at startup.
            # Just ensure the directory exists and is writable.
            try:
                test_path = (sqlite_dir / '.write_test') if sqlite_dir is not None else None
                if test_path is not None:
                    test_path.write_text('ok', encoding='utf-8')
                    test_path.unlink(missing_ok=True)
            except Exception as e:
                _STORAGE_STATUS['sqlite_open_ok'] = False
                _STORAGE_STATUS['ok'] = False
                _STORAGE_STATUS['error'] = f"sqlite_dir_write: {type(e).__name__}: {e}"
                _STARTUP_WARNINGS.append(_STORAGE_STATUS['error'])
    except Exception as e:
        _STORAGE_STATUS["ok"] = False
        _STORAGE_STATUS["error"] = f"storage_paths: {type(e).__name__}: {e}"
        _STARTUP_WARNINGS.append(_STORAGE_STATUS["error"])
def _save_config(cfg: dict[str, Any]) -> None:
    p = _config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _mask_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(cfg))
    try:
        key = out.get("llm", {}).get("api_key", "")
        if isinstance(key, str) and key:
            out["llm"]["api_key"] = key[:4] + "…" + key[-4:]
    except Exception:
        pass
    return out


_CONFIG: dict[str, Any] = _load_config()

# Also expose diagnostics under the MCP http_path (e.g. /mcp/diag) to avoid path confusion.
_DIAG_PREFIX: str = str(_CONFIG.get("mcp", {}).get("http_path") or os.getenv("MCP_HTTP_PATH") or "/mcp").rstrip("/")
if _DIAG_PREFIX == "":
    _DIAG_PREFIX = "/mcp"


# -------------------------
# Category policy (server config.json)
# -------------------------

_DEFAULT_CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "personal_info": "Personal information about the user",
    "preferences": "User likes/dislikes and stable choices",
    "relationships": "Relationships with others (including the assistant)",
    "goals": "Goals, plans, and objectives",
}


def _categories_from_cfg(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    cats_cfg = cfg.get("categories") if isinstance(cfg.get("categories"), dict) else {}
    raw = cats_cfg.get("defaults") if isinstance(cats_cfg.get("defaults"), list) else []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for c in raw:
        name = None
        desc = ""
        if isinstance(c, str):
            name = c.strip()
            desc = _DEFAULT_CATEGORY_DESCRIPTIONS.get(name, "")
        elif isinstance(c, dict):
            name = str(c.get("name") or "").strip()
            desc = str(c.get("description") or "").strip()
            if not desc:
                desc = _DEFAULT_CATEGORY_DESCRIPTIONS.get(name, "")
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "description": desc})

    # If user didn't define anything, fall back to a small sane default.
    if not out:
        out = [
            {"name": "personal_info", "description": _DEFAULT_CATEGORY_DESCRIPTIONS["personal_info"]},
            {"name": "preferences", "description": _DEFAULT_CATEGORY_DESCRIPTIONS["preferences"]},
            {"name": "relationships", "description": _DEFAULT_CATEGORY_DESCRIPTIONS["relationships"]},
            {"name": "goals", "description": _DEFAULT_CATEGORY_DESCRIPTIONS["goals"]},
        ]
    return out


def _category_policy_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    cats_cfg = cfg.get("categories") if isinstance(cfg.get("categories"), dict) else {}
    allow_dynamic = bool(cats_cfg.get("allow_dynamic", True))
    max_total = cats_cfg.get("max_total", 12)
    try:
        max_total = int(max_total) if max_total is not None else 0
    except Exception:
        max_total = 12
    return {
        "memory_categories": _categories_from_cfg(cfg),
        "allow_dynamic_categories": allow_dynamic,
        "max_categories_total": max_total,
    }


def _merge_memory_categories(fixed: list[dict[str, Any]], user_list: Any) -> list[dict[str, Any]]:
    # Keep fixed categories first, then append user categories by name.
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(name: str, desc: str = ""):
        key = (name or "").strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        out.append({"name": name.strip(), "description": (desc or "").strip()})

    for c in fixed:
        if isinstance(c, dict):
            add(str(c.get("name") or ""), str(c.get("description") or ""))

    if isinstance(user_list, list):
        for c in user_list:
            if isinstance(c, str):
                add(c, _DEFAULT_CATEGORY_DESCRIPTIONS.get(c.strip(), ""))
            elif isinstance(c, dict):
                add(str(c.get("name") or ""), str(c.get("description") or ""))

    return out


def _get_storage_dir(cfg: dict[str, Any]) -> Path:
    env = os.getenv("STORAGE_PATH") or os.getenv("MEMU_STORAGE_DIR") or os.getenv("MEMU_STORAGE_PATH")
    if env:
        d = Path(env).expanduser().resolve()
    else:
        d = Path(cfg.get("storage", {}).get("resources_dir") or "./storage").expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d



def _sanitize_db_filename(name: str) -> str:
    s = str(name or '').strip()
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', s)
    s = s.strip('._-')
    if not s:
        s = 'unknown'
    return s[:80]




def _sqlite_dsn_for_scope(cfg: dict[str, Any], base_dsn: str, scope: dict[str, Any] | None) -> str:
    """Resolve the sqlite DSN for this request.

    Policy (minimal):
      - Per-character DBs for SillyTavern traffic (agent_id == character name).
    """
    if not isinstance(scope, dict):
        scope = {}

    agent_id = str(scope.get('agent_id') or '').strip()

    sqlite_dir = _sqlite_dir_from_cfg(cfg, fallback_dsn=base_dsn)
    sqlite_dir.mkdir(parents=True, exist_ok=True)

    # No scope provided: keep the base DSN (typically :memory:).
    if not agent_id:
        return base_dsn

    # KISS: agent_id *is* the character name (provided by the extension/plugin).
    basename = _sanitize_db_filename(agent_id)
    db_path = (sqlite_dir / f"{basename}.db").resolve()
    _sqlite_ensure_nonempty(db_path)
    return f"sqlite:////{db_path.as_posix().lstrip('/')}"


def _database_config_from_cfg(cfg: dict[str, Any], scope: dict[str, Any] | None = None) -> dict[str, Any]:
    storage = cfg.get("storage") if isinstance(cfg.get("storage"), dict) else {}
    meta = storage.get("metadata_store") if isinstance(storage.get("metadata_store"), dict) else {}

    provider = os.getenv("METADATA_STORE_PROVIDER") or meta.get("provider") or "sqlite"
    provider = str(provider).strip().lower() or "sqlite"
    if provider == "inmemory":
        provider = "sqlite"

    dsn = os.getenv("DATABASE_URL") or os.getenv("MEMU_DB_DSN") or meta.get("dsn")
    if not dsn:
        if provider == "sqlite":
            dsn = _default_config()["storage"]["metadata_store"]["dsn"]
        else:
            raise RuntimeError("Postgres selected but no DSN set (storage.metadata_store.dsn or DATABASE_URL).")

    if provider == "sqlite":
        dsn = _normalize_sqlite_dsn(str(dsn))
        dsn = _sqlite_dsn_for_scope(cfg, dsn, scope or {})

    ddl_mode = os.getenv("DDL_MODE") or meta.get("ddl_mode") or "create"

    return {
        "metadata_store": {
            "provider": provider,
            "dsn": dsn,
            "ddl_mode": ddl_mode,
        }
    }


def _blob_config_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {"resources_dir": str(_get_storage_dir(cfg))}


# -------------------------
# Payload-based services (SillyTavern plugin)
# -------------------------

_SERVICES: dict[str, MemoryService] = {}
_SERVICE_STORAGE_FP: dict[str, dict[str, Any]] = {}


def _close_service_quiet(svc: MemoryService | None) -> None:
    """Best-effort close of a cached MemoryService database handle."""
    if svc is None:
        return
    try:
        db = getattr(svc, "database", None)
        close_fn = getattr(db, "close", None)
        if callable(close_fn):
            close_fn()
    except Exception:
        pass


def _clear_cached_services() -> None:
    """Drop all cached services and release underlying DB handles."""
    for svc in list(_SERVICES.values()):
        _close_service_quiet(svc)
    _SERVICES.clear()
    _SERVICE_STORAGE_FP.clear()


def _service_storage_fingerprint(database_config: dict[str, Any] | None) -> dict[str, Any]:
    """Small storage fingerprint used to invalidate stale cached services.

    For sqlite we track file path + inode identity so deleting/recreating the
    file forces a new service (otherwise SQLAlchemy may keep writing to an
    unlinked inode through a pooled connection).
    """
    if not isinstance(database_config, dict):
        return {"provider": None}

    ms = database_config.get("metadata_store")
    if not isinstance(ms, dict):
        return {"provider": None}

    provider = str(ms.get("provider") or "").strip().lower() or None
    if provider != "sqlite":
        return {"provider": provider}

    dsn = str(ms.get("dsn") or "")
    f = _sqlite_file_from_dsn(dsn)
    if f is None:
        return {"provider": "sqlite", "dsn": dsn, "path": None, "dev": None, "ino": None}

    p = f.expanduser().resolve()
    dev = None
    ino = None
    try:
        st = p.stat()
        dev = int(st.st_dev)
        ino = int(st.st_ino)
    except Exception:
        pass

    return {"provider": "sqlite", "dsn": dsn, "path": str(p), "dev": dev, "ino": ino}


def _safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out.pop("api_key", None)
    out.pop("OPENAI_API_KEY", None)
    out.pop("NANOGPT_API_KEY", None)
    return out


def _payload_signature(payload: dict[str, Any]) -> str:
    # Hash only config-y fields so changing per-step overrides forces a new service.
    keys = ["llm_profiles", "database_config", "blob_config", "memorize_config", "retrieve_config", "user_config"]
    snap = {k: payload.get(k) for k in keys if k in payload}
    raw = json.dumps(snap, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _get_service_from_payload(payload: dict[str, Any]) -> MemoryService:
    service_key_raw = str(payload.get("service_key") or "default")

    llm_profiles = payload.get("llm_profiles")
    database_config = payload.get("database_config")
    blob_config = payload.get("blob_config")

    # Local-first UX: plugin sends llm_profiles + step routing, while storage paths live in server config.json.
    if not isinstance(llm_profiles, dict):
        raise HTTPException(status_code=400, detail='llm_profiles required')

    if not isinstance(database_config, dict):
        scope_hint = _extract_scope(payload) if isinstance(payload, dict) else None
        database_config = _database_config_from_cfg(_CONFIG, scope=scope_hint)
        payload["database_config"] = database_config

    if not isinstance(blob_config, dict):
        blob_config = _blob_config_from_cfg(_CONFIG)
        payload["blob_config"] = blob_config

    # Enforce per-agent sqlite isolation even if the payload provided a database_config.
    # This prevents cross-character memory mixing at the storage boundary.
    try:
        if isinstance(database_config, dict):
            ms = database_config.get('metadata_store')
            if isinstance(ms, dict) and str(ms.get('provider') or '').lower() == 'sqlite':
                scope_hint2 = _extract_scope(payload)
                agent_id2 = str((scope_hint2 or {}).get('agent_id') or '').strip()
                if not agent_id2:
                    raise HTTPException(status_code=400, detail='agent_id required for sqlite (use character name)')
                base = _normalize_sqlite_dsn(str(ms.get('dsn') or ''))
                ms['dsn'] = _sqlite_dsn_for_scope(_CONFIG, base, scope_hint2)
    except HTTPException:
        raise
    except Exception:
        pass

    sig = _payload_signature(payload)
    service_key = f"{service_key_raw}__{sig}"
    storage_fp = _service_storage_fingerprint(database_config if isinstance(database_config, dict) else None)

    svc = _SERVICES.get(service_key)
    if svc is not None:
        prev_fp = _SERVICE_STORAGE_FP.get(service_key)
        if prev_fp == storage_fp:
            return svc

        # Backing storage changed (e.g. sqlite file deleted+recreated): recycle
        # this service so writes don't continue to an unlinked inode.
        _close_service_quiet(svc)
        _SERVICES.pop(service_key, None)
        _SERVICE_STORAGE_FP.pop(service_key, None)

    blob_config = payload.get("blob_config") or {}
    memorize_config = payload.get("memorize_config") or {}

    # Enforce fixed categories + dynamic policy from server config.json unless the payload overrides.
    try:
        fixed_cats = _categories_from_cfg(_CONFIG)
        if isinstance(memorize_config, dict):
            user_cats = memorize_config.get("memory_categories")
            memorize_config["memory_categories"] = _merge_memory_categories(fixed_cats, user_cats)
            if "allow_dynamic_categories" not in memorize_config:
                memorize_config["allow_dynamic_categories"] = bool((_CONFIG.get("categories") or {}).get("allow_dynamic", True))
            if "max_categories_total" not in memorize_config:
                memorize_config["max_categories_total"] = int(((_CONFIG.get("categories") or {}).get("max_total", 12)) or 0)
    except Exception:
        pass
    retrieve_config = payload.get("retrieve_config") or {}
    user_config = payload.get("user_config") or {}

    # Force STUserModel so agent_id/session_id filters are accepted.
    user_config = {**(user_config if isinstance(user_config, dict) else {}), "model": STUserModel}

    # Small UX: disable conversation preprocess prompt unless explicitly set.
    try:
        mpp = dict(memorize_config.get("multimodal_preprocess_prompts") or {}) if isinstance(memorize_config, dict) else {}
        if "conversation" not in mpp:
            mpp["conversation"] = ""
        if isinstance(memorize_config, dict):
            memorize_config["multimodal_preprocess_prompts"] = mpp
    except Exception:
        pass

    svc = MemoryService(
        llm_profiles=llm_profiles,
        blob_config=blob_config,
        database_config=database_config,
        memorize_config=memorize_config,
        retrieve_config=retrieve_config,
        user_config=user_config,
    )

    # Cap cached payload-services without a thundering-herd full wipe.
    if len(_SERVICES) >= 50:
        # Dict preserves insertion order in Python 3.7+; drop the oldest.
        try:
            oldest_key = next(iter(_SERVICES))
            _close_service_quiet(_SERVICES.get(oldest_key))
            _SERVICES.pop(oldest_key, None)
            _SERVICE_STORAGE_FP.pop(oldest_key, None)
        except Exception:
            _clear_cached_services()

    _SERVICES[service_key] = svc
    _SERVICE_STORAGE_FP[service_key] = storage_fp
    return svc


def _pick_str(payload: dict[str, Any], *keys: str) -> str | None:
    for k in keys:
        v = payload.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def _extract_scope(payload: dict[str, Any]) -> dict[str, Any]:
    user_id = _pick_str(payload, "user_id", "userId", "userID", "userid")
    agent_id = _pick_str(payload, "agent_id", "agentId", "agentID", "agentid")
    agent_name = _pick_str(payload, "agent_name", "agentName", "character_name", "characterName", "character")
    session_id = _pick_str(payload, "session_id", "sessionId", "sessionID", "sessionid", "session_date", "sessionDate", "sessiondate")

    # SillyTavern local plugin sends scope primarily under payload.user.
    user_obj = payload.get("user")
    if isinstance(user_obj, dict):
        if not user_id:
            user_id = _pick_str(user_obj, "user_id", "userId", "userID", "userid")
        if not agent_id:
            agent_id = _pick_str(user_obj, "agent_id", "agentId", "agentID", "agentid")
        if not agent_name:
            agent_name = _pick_str(user_obj, "agent_name", "agentName", "character_name", "characterName", "character")
        if not session_id:
            session_id = _pick_str(
                user_obj,
                "session_id",
                "sessionId",
                "sessionID",
                "sessionid",
                "session_date",
                "sessionDate",
                "sessiondate",
            )

    # Fallback: if an explicit agent_id was not provided, use agent_name for scoping.
    # This keeps per-agent SQLite practical for SillyTavern where the stable ID may be absent.
    if not agent_id and agent_name:
        agent_id = agent_name

    scope: dict[str, Any] = {}
    if user_id:
        scope["user_id"] = user_id
    if agent_id:
        scope["agent_id"] = agent_id
    if session_id:
        scope["session_id"] = session_id
    return scope


def _normalize_conversation(conv: Any) -> Any:
    if not isinstance(conv, list):
        return conv
    out = []
    for m in conv:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "participant":
            role = "user"

        # Preserve timestamps when available (epoch ms preferred).
        ts_ms: int | None = None
        try:
            raw_ts = m.get('ts_ms') if isinstance(m, dict) else None
            if raw_ts is None:
                raw_ts = m.get('timestamp') if isinstance(m, dict) else None
            if isinstance(raw_ts, (int, float)) and math.isfinite(raw_ts):
                ts_ms = int(raw_ts)
            elif isinstance(raw_ts, str) and raw_ts.strip():
                # ISO send_date
                try:
                    dt = datetime.fromisoformat(raw_ts.replace('Z', '+00:00'))
                    ts_ms = int(dt.timestamp() * 1000)
                except Exception:
                    ts_ms = None
        except Exception:
            ts_ms = None

        out.append({
            "role": role or "unknown",
            "name": m.get("name"),
            "content": m.get("content") or "",
            **({"ts_ms": ts_ms} if ts_ms is not None else {}),
        })
    return out


def _record_call(op: str, payload: dict[str, Any] | None, *, ok: bool, info: Any = None, error: str | None = None) -> None:
    try:
        scope = None
        if isinstance(payload, dict):
            u = payload.get('user')
            if isinstance(u, dict):
                scope = u
            else:
                scope = _extract_scope(payload) or None
        item = {
            't': time.time(),
            'op': op,
            'ok': ok,
            'scope': scope,
            'info': info,
            'error': error,
        }
        _LAST_CALLS.append(item)
        # keep small
        if len(_LAST_CALLS) > 50:
            del _LAST_CALLS[0: len(_LAST_CALLS) - 50]
    except Exception:
        pass


def _sqlite_current_path(user_id: str | None = None, agent_id: str | None = None) -> Path | None:
    try:
        base_dsn = str(_STORAGE_STATUS.get('dsn') or '')
        if not agent_id:
            return None
        scope = {'agent_id': agent_id}
        dsn = _sqlite_dsn_for_scope(_CONFIG, base_dsn, scope)
        f = _sqlite_file_from_dsn(dsn)
        return f.expanduser().resolve() if f is not None else None
    except Exception:
        return None


def _sqlite_ensure_nonempty(path: Path) -> None:
    """Ensure the sqlite file is not a confusing 0-byte placeholder.

    sqlite will create a 0-byte file on connect; that looks like "nothing happened".
    We write a harmless header change (user_version) so the file becomes non-empty.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                if path.stat().st_size > 0:
                    return
            except Exception:
                pass
        con = sqlite3.connect(str(path), timeout=5.0)
        con.execute('PRAGMA user_version=1')
        con.commit()
        con.close()
    except Exception:
        # Best-effort only; a later real DB write will make it non-empty.
        pass


def _sqlite_connect(path: Path) -> sqlite3.Connection:
    # timeout helps with transient lock contention
    return sqlite3.connect(str(path), timeout=5.0)


def _sqlite_table_columns(con: sqlite3.Connection, table: str) -> list[str]:
    cur = con.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall() if len(r) > 1]
    return [c for c in cols if isinstance(c, str)]


def _sqlite_build_scope_where(cols: list[str], user_id: str | None, agent_id: str | None, session_id: str | None) -> tuple[str, list[Any]]:
    where = []
    params: list[Any] = []
    if user_id and 'user_id' in cols:
        where.append('user_id = ?')
        params.append(user_id)
    if agent_id and 'agent_id' in cols:
        where.append('agent_id = ?')
        params.append(agent_id)
    if session_id and 'session_id' in cols:
        where.append('session_id = ?')
        params.append(session_id)
    if not where:
        return '', params
    return ' WHERE ' + ' AND '.join(where), params


def _sqlite_file_info(p: Path) -> dict[str, Any]:
    try:
        st = p.stat()
        return {
            'exists': p.exists(),
            'path': str(p),
            'size': int(st.st_size),
            'mtime': float(st.st_mtime),
        }
    except Exception as e:
        return {'exists': p.exists(), 'path': str(p), 'error': f"{type(e).__name__}: {e}"}


def _sqlite_pragmas(con: sqlite3.Connection) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ('journal_mode','synchronous','busy_timeout','foreign_keys','cache_size','temp_store'): 
        try:
            r = con.execute(f"PRAGMA {k}").fetchone()
            out[k] = r[0] if r else None
        except Exception:
            out[k] = None
    return out



# -------------------------
# Meta
# -------------------------

@app.get("/health", operation_id="health")
async def health():
    storage_dir = _get_storage_dir(_CONFIG)
    uid = getattr(os, "geteuid", lambda: None)()
    gid = getattr(os, "getegid", lambda: None)()
    user = None
    try:
        if uid is not None and pwd is not None:
            user = pwd.getpwuid(uid).pw_name
    except Exception:
        pass
    return {
        # "ok" reflects whether storage paths look usable.
        "ok": (_STORAGE_STATUS.get("ok") is not False),
        "serverInstanceId": _SERVER_INSTANCE_ID,
        "buildId": _BUILD_ID,
        "startedAtUnix": _SERVER_STARTED_AT_UNIX,
        "process": {"uid": uid, "gid": gid, "user": user},
        "ephemeralDb": _is_ephemeral_db(_CONFIG),
        "config_path": str(_config_path()),
        "storage_dir": str(storage_dir),
        "storage": _STORAGE_STATUS,
        "services_cached": len(_SERVICES),
        "startup_warnings": _STARTUP_WARNINGS,
        "mcp": {
            "enabled": _has_mcp,
            "http_path": str(_CONFIG.get("mcp", {}).get("http_path") or os.getenv("MCP_HTTP_PATH") or "/mcp"),
            "sse_path": str(_CONFIG.get("mcp", {}).get("sse_path") or os.getenv("MCP_SSE_PATH") or "/sse"),
        },
    }


@app.get("/version", operation_id="version")
async def version():
    return {"ok": True, "buildId": _BUILD_ID, "serverInstanceId": _SERVER_INSTANCE_ID, "startedAtUnix": _SERVER_STARTED_AT_UNIX}


@app.get(f"{_DIAG_PREFIX}/diag")
@app.get("/diag", operation_id="diag_page")
async def diag_page():
    return HTMLResponse(content="""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>mcp-memu-server diagnostics</title>
<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:20px;line-height:1.4;max-width:980px}code,pre{background:#f2f2f2;border-radius:10px}code{padding:2px 6px}pre{padding:12px;overflow:auto}</style>
</head><body>
<h2>mcp-memu-server diagnostics</h2>
<ul>
<li><a href='/health'>/health</a></li>
<li><a href='/diag/http'>/diag/http</a> <small>(last 200 HTTP requests seen by this server)</small></li>
<li><a href='/diag/calls'>/diag/calls</a> <small>(last 50 memorize/retrieve calls)</small></li>
<li><a href='/diag/sqlite'>/diag/sqlite</a></li>
<li><a href='/diag/sqlite/counts'>/diag/sqlite/counts</a> <small>(add ?user_id=...&agent_id=...)</small></li>
<li><a href='/diag/sqlite/recent?table=memu_memory_items&limit=10'>/diag/sqlite/recent</a> <small>(add scope params)</small></li>
</ul>
<p><b>Scope tip:</b> if your ST extension uses <code>user_id</code> + <code>agent_id</code>, but your tests omit one, retrieval can look empty. Use the same scope in <code>/diag/sqlite/*</code>.</p>
</body></html>""")


@app.get(f"{_DIAG_PREFIX}/diag/calls")
@app.get("/diag/calls", operation_id="diag_calls")
async def diag_calls():
    return {"ok": True, "calls": _LAST_CALLS}


@app.get(f"{_DIAG_PREFIX}/diag/http")
@app.get("/diag/http", operation_id="diag_http")
async def diag_http():
    return {"ok": True, "http": _LAST_HTTP}


@app.get(f"{_DIAG_PREFIX}/diag/sqlite")
@app.get("/diag/sqlite", operation_id="diag_sqlite")
async def diag_sqlite(user_id: str = "", agent_id: str = ""):
    try:
        storage = _CONFIG.get('storage') if isinstance(_CONFIG.get('storage'), dict) else {}
        meta = storage.get('metadata_store') if isinstance(storage.get('metadata_store'), dict) else {}
        provider = str(meta.get('provider') or '').lower()
        if provider not in ('sqlite', 'sqlite3'):
            return {"ok": False, "reason": "provider_not_sqlite", "provider": provider, "storage": _STORAGE_STATUS}

        p = _sqlite_current_path(user_id or None, agent_id or None)
        if p is None:
            return {"ok": False, "reason": "no_sqlite_path", "storage": _STORAGE_STATUS}

        info = _sqlite_file_info(p)
        if not p.exists():
            return {"ok": False, "reason": "sqlite_file_missing", **info, "storage": _STORAGE_STATUS}

        con = _sqlite_connect(p)
        try:
            tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
            return {
                "ok": True,
                "storage": _STORAGE_STATUS,
                "file": info,
                "tables": tables,
                "pragmas": _sqlite_pragmas(con),
            }
        finally:
            con.close()
    except Exception as e:
        return {"ok": False, "reason": "exception", "error": f"{type(e).__name__}: {e}", "storage": _STORAGE_STATUS}


@app.get(f"{_DIAG_PREFIX}/diag/sqlite/counts")
@app.get("/diag/sqlite/counts", operation_id="diag_sqlite_counts")
async def diag_sqlite_counts(user_id: str | None = None, agent_id: str | None = None, session_id: str | None = None):
    p = _sqlite_current_path(user_id or None, agent_id or None)
    if p is None or not p.exists():
        return {"ok": False, "reason": "sqlite_file_missing", "path": str(p) if p else None, "storage": _STORAGE_STATUS}

    allowed = ["memu_resources", "memu_memory_categories", "memu_memory_items", "memu_category_items"]
    con = _sqlite_connect(p)
    try:
        out: dict[str, Any] = {"ok": True, "path": str(p), "scope": {"user_id": user_id, "agent_id": agent_id, "session_id": session_id}, "tables": {}}
        for t in allowed:
            cols = _sqlite_table_columns(con, t)
            where, params = _sqlite_build_scope_where(cols, user_id, agent_id, session_id)
            total = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            scoped = con.execute(f"SELECT COUNT(*) FROM {t}{where}", params).fetchone()[0] if where else total
            out["tables"][t] = {"total": int(total), "scoped": int(scoped), "scope_cols": [c for c in ('user_id','agent_id','session_id') if c in cols]}
        return out
    finally:
        con.close()


@app.get(f"{_DIAG_PREFIX}/diag/sqlite/recent")
@app.get("/diag/sqlite/recent", operation_id="diag_sqlite_recent")
async def diag_sqlite_recent(table: str = 'memu_memory_items', limit: int = 20, user_id: str | None = None, agent_id: str | None = None, session_id: str | None = None):
    allowed = {"memu_resources", "memu_memory_categories", "memu_memory_items", "memu_category_items"}
    if table not in allowed:
        raise HTTPException(status_code=400, detail=f"table must be one of: {sorted(allowed)}")
    limit = max(1, min(int(limit or 20), 200))

    p = _sqlite_current_path(user_id or None, agent_id or None)
    if p is None or not p.exists():
        return {"ok": False, "reason": "sqlite_file_missing", "path": str(p) if p else None, "storage": _STORAGE_STATUS}

    con = _sqlite_connect(p)
    try:
        cols = _sqlite_table_columns(con, table)
        scope_where, params = _sqlite_build_scope_where(cols, user_id, agent_id, session_id)

        # Avoid dumping big JSON embeddings/extras by default.
        prefer = ["id","created_at","updated_at","user_id","agent_id","session_id","name","description","summary","memory_type","happened_at","resource_id","url","modality","local_path","caption","item_id","category_id"]
        ban = {"embedding", "extra"}
        sel = [c for c in prefer if c in cols and c not in ban]
        if not sel:
            sel = [c for c in cols if c not in ban][:12]
        order_col = 'created_at' if 'created_at' in cols else ('updated_at' if 'updated_at' in cols else 'id')

        sql = f"SELECT {', '.join(sel)} FROM {table}{scope_where} ORDER BY {order_col} DESC LIMIT ?"
        rows = con.execute(sql, [*params, limit]).fetchall()
        out_rows = []
        for r in rows:
            d = {sel[i]: r[i] for i in range(len(sel))}
            # light truncation for readability
            for k,v in list(d.items()):
                if isinstance(v, str) and len(v) > 400:
                    d[k] = v[:400] + '…'
            out_rows.append(d)
        return {"ok": True, "path": str(p), "table": table, "columns": sel, "rows": out_rows}
    finally:
        con.close()



@app.get("/config", operation_id="get_config")
async def get_config(include_secrets: bool = False):
    return JSONResponse(content={"ok": True, "config": _CONFIG if include_secrets else _mask_config(_CONFIG)})


@app.post("/config", operation_id="set_config")
async def set_config(req: Request):
    global _CONFIG
    try:
        body = await req.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Config must be a JSON object")
        merged = {**_CONFIG, **body}
        _save_config(merged)
        _CONFIG = merged
        _ensure_storage_paths(_CONFIG)
        _clear_cached_services()
        return JSONResponse(content={"ok": True, "config": _mask_config(_CONFIG)})
    except HTTPException as he:
        try:
            _record_call('config.set', body if isinstance(body, dict) else None, ok=False, error=str(getattr(he, 'detail', he)))
        except Exception:
            pass
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reload", operation_id="reload")
async def reload_config():
    global _CONFIG
    _CONFIG = _load_config()
    _clear_cached_services()
    return JSONResponse(content={"ok": True})


@app.get("/", operation_id="root")
async def root():
    """Serve memU-ui if bundled next to the server, otherwise show a JSON health stub."""
    try:
        bundle_root = Path(__file__).resolve().parents[2]
        ui_index = bundle_root / 'memu-ui' / 'dist' / 'index.html'
        if ui_index.exists():
            return FileResponse(str(ui_index))
    except Exception:
        pass
    return {"message": "mcp-memu-server", "mcp": "enabled" if _has_mcp else "disabled"}


# -------------------------
# SillyTavern plugin endpoints (payload-driven)
# -------------------------



# -----------------------------
# ST conversation resource helpers
# -----------------------------

def _msg_key(m: dict[str, Any]) -> tuple[str, str, str, str]:
    # Include timestamp when present so identical text repeated at different times isn't dropped.
    return (
        str(m.get('role') or ''),
        str(m.get('name') or ''),
        str(m.get('content') or ''),
        str(m.get('ts_ms') or ''),
    )


def _read_list(p: Path) -> list[dict[str, Any]]:
    try:
        if not p.exists():
            return []
        raw = p.read_text(encoding='utf-8')
        obj = json.loads(raw) if raw.strip() else []
        return [m for m in obj if isinstance(m, dict)] if isinstance(obj, list) else []
    except Exception:
        return []


def _write_list_if_changed(p: Path, old: list[dict[str, Any]], new: list[dict[str, Any]]) -> None:
    if new == old:
        return
    p.write_text(json.dumps(new, ensure_ascii=False), encoding='utf-8')


def _merge_conv(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not old:
        return new
    if not new:
        return old
    # common prefix
    if len(new) >= len(old) and old == new[: len(old)]:
        return new
    if len(old) >= len(new) and new == old[: len(new)]:
        return old
    # overlap on the end (cap to keep it cheap)
    max_k = min(len(old), len(new), 80)
    for k in range(max_k, 0, -1):
        if old[-k:] == new[:k]:
            return old + new[k:]
    # fallback: append only unseen messages
    seen = {_msg_key(m) for m in old}
    out = list(old)
    for m in new:
        k = _msg_key(m)
        if k in seen:
            continue
        seen.add(k)
        out.append(m)
    return out


def _local_dt(ts_ms: int, zi: Any | None) -> datetime:
    # ts_ms is UTC epoch ms
    dt_utc = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    return dt_utc.astimezone(zi) if zi is not None else dt_utc


def _date_label(ts_ms: int | None, zi: Any | None) -> str:
    if ts_ms is None:
        return 'undated'
    try:
        return _local_dt(ts_ms, zi).date().isoformat()
    except Exception:
        return 'undated'


def _split_indices_by_sleep(msgs: list[dict[str, Any]], zi: Any | None, tz_ok: bool, min_lull_seconds: int) -> tuple[list[int], dict[str, Any]]:
    # Return split indices (start of a new day) within msgs.
    # Choose the largest no-message gap overlapping the local night window (22:00 → 08:00),
    # accepting only when overlap >= min_lull_seconds.
    if not tz_ok:
        return ([], {"tz_ok": False})

    ts: list[int | None] = []
    for m in msgs:
        v = m.get('ts_ms')
        ts.append(int(v) if isinstance(v, int) else None)
    if sum(1 for x in ts if x is not None) < 2:
        return ([], {"tz_ok": True, "timestamps_ok": False})

    best: dict[Any, tuple[float, int]] = {}
    for i in range(len(ts) - 1):
        a = ts[i]
        b = ts[i + 1]
        if a is None or b is None:
            continue
        if b <= a:
            continue

        t0 = _local_dt(a, zi)
        t1 = _local_dt(b, zi)
        if t1 <= t0:
            continue

        d0 = (t0.date() - timedelta(days=1))
        d1 = t1.date()
        # Cap the day-iteration for extremely long gaps.
        max_days = min((d1 - d0).days, 14)
        for k in range(max_days + 1):
            d = d0 + timedelta(days=k)
            win_start = datetime.combine(d, dtime(22, 0), tzinfo=zi)
            win_end = datetime.combine(d + timedelta(days=1), dtime(8, 0), tzinfo=zi)
            overlap = (min(t1, win_end) - max(t0, win_start)).total_seconds()
            if overlap <= 0:
                continue
            prev = best.get(d)
            if prev is None or overlap > prev[0]:
                best[d] = (overlap, i + 1)

    min_lull = float(min_lull_seconds)
    nights_total = len(best)
    nights_qual = sum(1 for (score, _idx) in best.values() if isinstance(score, (int, float)) and score >= min_lull)

    splits = sorted({
        idx for (score, idx) in best.values()
        if isinstance(idx, int) and 0 < idx < len(msgs) and isinstance(score, (int, float)) and score >= min_lull
    })
    return (splits, {"tz_ok": True, "timestamps_ok": True, "nights_total": nights_total, "nights_qual": nights_qual, "min_lull_seconds": min_lull_seconds})


def _first_ts_in_range(ts_list: list[int | None], a_i: int, b_i: int) -> int | None:
    for j in range(max(0, a_i), min(len(ts_list), b_i)):
        v = ts_list[j]
        if isinstance(v, int):
            return v
    return None


def _enforce_max_span(boundaries_in: list[int], ts_list: list[int | None], zi: Any | None, tz_ok: bool, max_span_days: int, min_lull_seconds: int) -> tuple[list[int], int]:
    # Insert forced boundaries so a segment never spans more than max_span_days.
    if not tz_ok:
        return (boundaries_in, 0)

    out: list[int] = []
    forced = 0
    for nxt in boundaries_in:
        if not out:
            out.append(nxt)
            continue

        while True:
            cur = out[-1]
            if nxt <= cur + 1:
                break

            start_ts = _first_ts_in_range(ts_list, cur, nxt)
            if start_ts is None:
                break

            try:
                cap_local = _local_dt(start_ts, zi) + timedelta(days=max_span_days)
            except Exception:
                break

            idx_force: int | None = None
            for j in range(cur + 1, nxt):
                v = ts_list[j]
                if not isinstance(v, int):
                    continue
                try:
                    if _local_dt(v, zi) >= cap_local:
                        idx_force = j
                        break
                except Exception:
                    continue

            if idx_force is None or idx_force <= cur or idx_force >= nxt:
                break

            out.append(idx_force)
            forced += 1
            try:
                sys.stderr.write(
                    f"memu-server: sleep-split: no lull >= {min_lull_seconds//3600}h found; forcing split at {_local_dt(ts_list[idx_force], zi).isoformat()} (max {max_span_days}d)\n"
                )
                sys.stderr.flush()
            except Exception:
                pass

        out.append(nxt)

    out2: list[int] = []
    for x in out:
        if not out2 or x != out2[-1]:
            out2.append(x)
    return (out2, forced)

@app.post("/memorize", operation_id="memorize")
async def memorize(payload: dict[str, Any]):
    """Memorize a SillyTavern conversation.

    Preferred: send the full memU payload (llm_profiles/database_config/etc) so per-step routing works.
    """
    try:
        safe = _safe_payload(payload)
        svc = _get_service_from_payload(safe)

        user_scope = safe.get("user")
        if not isinstance(user_scope, dict):
            user_scope = _extract_scope(safe) or None


        # Per-agent-only: SillyTavern traffic must include user.agent_id (or agent_name).
        if not isinstance(user_scope, dict) or not str(user_scope.get('agent_id') or '').strip():
            raise HTTPException(status_code=400, detail='Missing user.agent_id (per-agent DBs required)')

        conversation = safe.get("conversation")
        if conversation is None:
            conversation = safe.get("content")
        if conversation is None and isinstance(payload, list):
            conversation = payload
        if not isinstance(conversation, list) or not conversation:
            raise HTTPException(status_code=400, detail="Missing or empty 'conversation' list")

        conv_norm = _normalize_conversation(conversation)

        # Keep resources inside memU for traceability:
        # - One canonical full log per chat (deduped)
        # - Daily resources split by sleep gap (22:00–08:00 local)
        #   using the largest no-chat gap intersecting that window.


        uid = str((user_scope or {}).get('user_id') or 'user') if isinstance(user_scope, dict) else 'user'
        aid = str((user_scope or {}).get('agent_id') or 'agent') if isinstance(user_scope, dict) else 'agent'

        chat_file = _pick_str(safe, 'chatFileName', 'chat_file_name', 'chat_filename', 'chatFile')
        hint = chat_file or _pick_str(safe, 'resource_url') or ''
        chat_key_src = f"{uid}|{aid}|{hint}".encode('utf-8', 'ignore')
        chat_key = hashlib.sha1(chat_key_src).hexdigest()[:16]
        agent_slug = _sanitize_db_filename(aid)

        storage_dir = _get_storage_dir(_CONFIG)
        chats_dir = (storage_dir / 'st_chats').resolve()
        chat_dir = (chats_dir / f"{agent_slug}_{chat_key}").resolve()
        days_dir = (chat_dir / 'days').resolve()
        chat_dir.mkdir(parents=True, exist_ok=True)
        days_dir.mkdir(parents=True, exist_ok=True)

        full_path = (chat_dir / 'full.json').resolve()
        manifest_path = (chat_dir / 'manifest.json').resolve()

        prev_full = _read_list(full_path)
        prev_len = len(prev_full)
        merged_len = len(conv_norm) if isinstance(conv_norm, list) else 0
        merged = prev_full
        if isinstance(conv_norm, list):
            merged = _merge_conv(prev_full, conv_norm)
            merged_len = len(merged)
            _write_list_if_changed(full_path, prev_full, merged)

        # Timezone hint (IANA) from client. Offset is only a fallback for logging.
        tz_name = _pick_str(safe, 'timeZone', 'timezone', 'time_zone')
        tz_off_raw = safe.get('timeZoneOffsetMin')
        tz_off_min = int(tz_off_raw) if isinstance(tz_off_raw, (int, float)) and math.isfinite(tz_off_raw) else None

        tz_ok = False
        zi = None
        if tz_name and ZoneInfo is not None:
            try:
                zi = ZoneInfo(str(tz_name))
                tz_ok = True
            except Exception:
                zi = None
                tz_ok = False
        # Fallback: fixed offset (no DST) when tzdata is missing or tz is invalid.
        if not tz_ok and tz_off_min is not None:
            try:
                zi = timezone(timedelta(minutes=-tz_off_min))
                tz_ok = True
                if not tz_name:
                    tz_name = f"offset({tz_off_min})"
            except Exception:
                zi = None
                tz_ok = False

        # Load manifest (segments for older days) so we only rebuild the tail.
        manifest: dict[str, Any] = {}
        try:
            rawm = manifest_path.read_text(encoding='utf-8') if manifest_path.exists() else ''
            manifest = json.loads(rawm) if rawm.strip() else {}
        except Exception:
            manifest = {}
        segments: list[dict[str, Any]] = manifest.get('segments') if isinstance(manifest.get('segments'), list) else []


        # Build/update daily files only when we have tz + timestamps.
        resource_url = str(full_path)
        days_written = 0
        forced_splits = 0
        sleep_stats: Any | None = None
        if tz_ok and isinstance(merged, list) and any(isinstance(m.get('ts_ms'), int) for m in merged):
            tail_n = 2500
            if not segments:
                rebuild_from = 0
                keep_segments: list[dict[str, Any]] = []
            else:
                tail_start = max(0, len(merged) - tail_n)
                rebuild_from = tail_start
                keep_segments = []
                for s in segments:
                    try:
                        st_i = int(s.get('start'))
                        en_i = int(s.get('end'))
                        if st_i <= tail_start <= en_i:
                            rebuild_from = st_i
                        if st_i < rebuild_from:
                            keep_segments.append(s)
                    except Exception:
                        continue

            ctx_start = max(0, rebuild_from - 1)
            # Split indices are only accepted when the best night-overlap gap is >= min lull.
            splits_rel, sleep_stats = _split_indices_by_sleep(merged[ctx_start:], zi, tz_ok, _SLEEP_SPLIT_MIN_LULL_SECONDS)
            splits = [ctx_start + i for i in splits_rel if (ctx_start + i) > rebuild_from]

            # If we found no qualifying lull at all, emit a one-line notice.
            try:
                if isinstance(sleep_stats, dict) and sleep_stats.get('nights_total') and not sleep_stats.get('nights_qual'):
                    sys.stderr.write(
                        f"memu-server: sleep-split: no qualifying lull >= {_SLEEP_SPLIT_MIN_LULL_SECONDS//3600}h found in night windows; allowing up to {_SLEEP_SPLIT_MAX_SPAN_DAYS} days per file\n"
                    )
                    sys.stderr.flush()
            except Exception:
                pass

            boundaries_raw = [rebuild_from] + splits + [len(merged)]

            # Enforce max span so we don't get unbounded multi-day files.
            # Build ts_list for the full merged list once.
            ts_list: list[int | None] = []
            for m in merged:
                v = m.get('ts_ms') if isinstance(m, dict) else None
                ts_list.append(int(v) if isinstance(v, int) else None)
            boundaries, forced_splits = _enforce_max_span(boundaries_raw, ts_list, zi, tz_ok, _SLEEP_SPLIT_MAX_SPAN_DAYS, _SLEEP_SPLIT_MIN_LULL_SECONDS)
            new_segments: list[dict[str, Any]] = []
            for a_i, b_i in zip(boundaries, boundaries[1:]):
                if b_i <= a_i:
                    continue
                seg = merged[a_i:b_i]
                if not seg:
                    continue
                first_ts = seg[0].get('ts_ms') if isinstance(seg[0], dict) else None
                last_ts = seg[-1].get('ts_ms') if isinstance(seg[-1], dict) else None
                date = _date_label(first_ts if isinstance(first_ts, int) else None, zi)
                end_date = _date_label(last_ts if isinstance(last_ts, int) else None, zi)
                fn = f"{date}.json" if end_date == date else f"{date}__{end_date}.json"
                fp = (days_dir / fn).resolve()
                old_seg = _read_list(fp)
                _write_list_if_changed(fp, old_seg, seg)
                days_written += 1 if seg != old_seg else 0
                new_segments.append({"date": date, "end_date": end_date, "start": a_i, "end": b_i - 1, "file": fn})

            segments = keep_segments + new_segments
            try:
                manifest_out = {"v": 1, "tz": str(tz_name or ''), "segments": segments, "split": {"min_lull_seconds": _SLEEP_SPLIT_MIN_LULL_SECONDS, "max_span_days": _SLEEP_SPLIT_MAX_SPAN_DAYS}, "source": {"chatFileName": chat_file or '', "resource_url_in": _pick_str(safe, 'resource_url') or '', "timeZoneOffsetMin": tz_off_min if tz_off_min is not None else None}}
                manifest_path.write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception:
                pass

            if segments:
                last_file = segments[-1].get('file')
                if isinstance(last_file, str) and last_file:
                    resource_url = str((days_dir / last_file).resolve())

        result = await svc.memorize(resource_url=resource_url, modality="conversation", user=user_scope)

        _record_call('memorize', safe, ok=True, info={
            'resource_url': resource_url,
            'chatFileName': chat_file,
            'timeZone': tz_name,
            'messages_prev': prev_len,
            'messages_in': len(conv_norm) if isinstance(conv_norm, list) else None,
            'messages_merged': len(merged) if isinstance(merged, list) else None,
            'days_written': days_written,
            'sleepSplitMinLullSeconds': _SLEEP_SPLIT_MIN_LULL_SECONDS,
            'sleepSplitMaxSpanDays': _SLEEP_SPLIT_MAX_SPAN_DAYS,
            'sleepSplitForcedSplits': forced_splits if 'forced_splits' in locals() else 0,
            'sleepSplitStats': sleep_stats if 'sleep_stats' in locals() else None,
        })
        return {"ok": True, "result": result, "resource_url": resource_url}
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        try:
            _record_call('memorize', payload if isinstance(payload, dict) else None, ok=False, error=f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(exc)) from exc




@app.get("/default-categories", operation_id="default_categories")
async def default_categories():
    """Return default category policy from server config.json.

    This endpoint is intentionally scope-free (no user_id/agent_id required).
    """
    cats_cfg = _CONFIG.get('categories') if isinstance(_CONFIG, dict) else None
    cats_cfg = cats_cfg if isinstance(cats_cfg, dict) else {}

    defaults_in = cats_cfg.get('defaults')
    defaults_in = defaults_in if isinstance(defaults_in, list) else []

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for c in defaults_in:
        name = ''
        desc = ''
        if isinstance(c, str):
            name = c.strip()
        elif isinstance(c, dict):
            name = str(c.get('name') or '').strip()
            desc = str(c.get('description') or '').strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({'name': name, 'description': desc})

    allow_dynamic = bool(cats_cfg.get('allow_dynamic') if 'allow_dynamic' in cats_cfg else True)
    max_total_raw = cats_cfg.get('max_total', 12)
    try:
        max_total = int(max_total_raw)
    except Exception:
        max_total = 12

    return {"ok": True, "categories": out, "allow_dynamic": allow_dynamic, "max_total": max_total}

@app.get("/categories", operation_id="list_memory_categories")
async def list_memory_categories(user_id: str = "", agent_id: str = "", include_empty: bool = False):
    # Scope is required: this server runs per-agent SQLite databases (no shared DB by default).
    if not agent_id.strip():
        raise HTTPException(status_code=400, detail="agent_id required")
    where: dict[str, Any] = {"agent_id": agent_id.strip()}
    if user_id.strip():
        where["user_id"] = user_id.strip()

    # Build a minimal payload using server config so this GET endpoint can list categories
    # without requiring the caller to send llm_profiles.
    llm = _CONFIG.get("llm", {}) if isinstance(_CONFIG.get("llm"), dict) else {}
    api_key = str(os.getenv("OPENAI_API_KEY") or os.getenv("NANOGPT_API_KEY") or llm.get("api_key") or "")
    base_url = str(os.getenv("OPENAI_BASE_URL") or llm.get("base_url") or "https://api.openai.com/v1")
    chat_model = str(os.getenv("DEFAULT_CHAT_MODEL") or llm.get("chat_model") or "gpt-4o-mini")
    embed_model = str(os.getenv("DEFAULT_EMBED_MODEL") or llm.get("embed_model") or "text-embedding-3-small")

    payload = {
        "llm_profiles": {
            "default": {
                "provider": str(os.getenv("LLM_PROVIDER") or llm.get("provider") or "openai"),
                "api_key": api_key,
                "base_url": base_url,
                "chat_model": chat_model,
                "embed_model": embed_model,
                "client_backend": str(os.getenv("LLM_CLIENT_BACKEND") or llm.get("client_backend") or "httpx"),
                "endpoint_overrides": llm.get("endpoint_overrides") or {},
            }
        },
        "user": where,
    }
    svc = _get_service_from_payload(payload)

    try:
        cats = await svc.list_memory_categories(where=where)
        if isinstance(cats, dict):
            cats = cats.get("categories")
        out = []
        for c in cats or []:
            if not isinstance(c, dict):
                continue
            nm = c.get("name")
            if not isinstance(nm, str) or not nm.strip():
                continue
            cc = {**c, "name": nm, "summary": str(c.get("summary") or "")}
            if include_empty or _has_category_content(cc):
                out.append(cc)
        return {"categories": out}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
@app.post("/categories/search", operation_id="search_memory_categories")
async def search_memory_categories(payload: dict[str, Any]):
    """Payload-driven category listing (matches SillyTavern plugin's local mode)."""
    try:
        safe = _safe_payload(payload)
        svc = _get_service_from_payload(safe)

        where = safe.get("where")
        if where is not None and not isinstance(where, dict):
            raise HTTPException(status_code=400, detail="'where' must be an object")
        if where is None:
            where = safe.get("user") if isinstance(safe.get("user"), dict) else (_extract_scope(safe) or None)

        include_empty = bool(safe.get("include_empty"))

        cats = await svc.list_memory_categories(where=where)
        if isinstance(cats, dict):
            cats = cats.get("categories")
        out = []
        for c in cats or []:
            if not isinstance(c, dict):
                continue
            nm = c.get("name")
            if not isinstance(nm, str) or not nm.strip():
                continue
            cc = {**c, "name": nm, "summary": str(c.get("summary") or "")}
            if include_empty or _has_category_content(cc):
                out.append(cc)
        _record_call('categories.search', safe, ok=True, info={'returned': len(out)})
        return {"categories": out}
    except HTTPException:
        _record_call('categories.search', payload if isinstance(payload, dict) else None, ok=False, error='HTTPException')
        raise
    except Exception as exc:
        _record_call('categories.search', payload if isinstance(payload, dict) else None, ok=False, error=f"{type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/retrieve", operation_id="retrieve")
async def retrieve(payload: dict[str, Any]):
    if "query" not in payload and "queries" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'query' or 'queries' in request body")
    try:
        safe = _safe_payload(payload)
        svc = _get_service_from_payload(safe)

        where = safe.get("where")
        if where is not None and not isinstance(where, dict):
            raise HTTPException(status_code=400, detail="'where' must be an object")
        if where is None:
            where = safe.get("user") if isinstance(safe.get("user"), dict) else (_extract_scope(safe) or None)

        queries = safe.get("queries")
        if queries is not None:
            if not isinstance(queries, list) or not queries:
                raise HTTPException(status_code=400, detail="'queries' must be a non-empty list")
            memu_queries: list[dict[str, Any]] = []
            for q in queries:
                if isinstance(q, str):
                    memu_queries.append({"role": "user", "content": {"text": q}})
                elif isinstance(q, dict):
                    if "content" in q:
                        memu_queries.append(q)
                    elif "query" in q:
                        memu_queries.append({"role": q.get("role", "user"), "content": {"text": str(q.get("query"))}})
                    else:
                        raise HTTPException(status_code=400, detail="Each query object must have 'content' or 'query'")
                else:
                    raise HTTPException(status_code=400, detail="Each query must be a string or object")
        else:
            memu_queries = [{"role": "user", "content": {"text": str(safe.get("query", ""))}}]

        result = await svc.retrieve(memu_queries, where=where)
        _record_call('retrieve', safe, ok=True, info={'queries': len(memu_queries) if isinstance(memu_queries, list) else None, 'where': where})
        # Return plain dict so FastAPI can encode datetime/UUID safely.
        return {"ok": True, "result": result}
    except HTTPException as he:
        try:
            _record_call('retrieve', payload if isinstance(payload, dict) else None, ok=False, error=str(getattr(he, 'detail', he)))
        except Exception:
            pass
        raise
    except Exception as exc:
        try:
            _record_call('retrieve', payload if isinstance(payload, dict) else None, ok=False, error=f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# memU-ui compatibility: it calls /api/*
@app.post("/api/memorize", operation_id="api_memorize")
async def api_memorize(payload: dict[str, Any] = Body(...)):
    return await memorize(payload)


@app.post("/api/retrieve", operation_id="api_retrieve")
async def api_retrieve(payload: dict[str, Any] = Body(...)):
    return await retrieve(payload)


# -------------------------
# MCP mounting (optional)
# -------------------------

_has_mcp = False
try:
    from fastapi_mcp import FastApiMCP

    mcp = FastApiMCP(app)
    http_path = str(_CONFIG.get("mcp", {}).get("http_path") or os.getenv("MCP_HTTP_PATH") or "/mcp")
    sse_path = str(_CONFIG.get("mcp", {}).get("sse_path") or os.getenv("MCP_SSE_PATH") or "/sse")
    mcp.mount(http_path=http_path, sse_path=sse_path)
    _has_mcp = True
except Exception:
    _has_mcp = False


# -------------------------
# Bundle UI mounting (optional)
# -------------------------
try:
    _BUNDLE_ROOT = Path(__file__).resolve().parents[2]
    _UI_DIST = _BUNDLE_ROOT / 'memu-ui' / 'dist'
    if _UI_DIST.exists():
        # Serve SPA assets (e.g. /assets/*). API routes defined above still win.
        app.mount('/', StaticFiles(directory=str(_UI_DIST), html=True), name='ui')
except Exception:
    pass
