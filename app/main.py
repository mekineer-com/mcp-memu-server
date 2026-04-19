import asyncio
import hashlib
import json
import logging
import math
import os
import random
import re
import signal
import sqlite3
import threading
import time
import traceback
import uuid
from collections.abc import Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path
from typing import Any

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


try:
    import pwd
except Exception:  # pragma: no cover
    pwd = None  # type: ignore

from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from memu.app import MemoryService
from pydantic import BaseModel

from app.db import (
    json_to_db as _json_to_db,
)
from app.db import (
    normalize_text_list as _normalize_text_list,
)
from app.db import (
    sqlite_connect as _sqlite_connect,
)
from app.db import (
    sqlite_ensure_conversation_state_schema as _sqlite_ensure_conversation_state_schema,
)
from app.db import (
    sqlite_ensure_nonempty as _sqlite_ensure_nonempty,
)
from app.db import (
    sqlite_pragmas as _sqlite_pragmas,
)
from app.db import (
    sqlite_table_columns as _sqlite_table_columns,
)
from app.services.consolidation import (
    ConsolidationDeps,
)
from app.services.consolidation import (
    gather_consolidation_inputs as _gather_consolidation_inputs,
)
from app.services.consolidation import (
    run_consolidation_llm as _run_consolidation_llm,
)
from app.services.consolidation import (
    write_consolidation_outputs as _write_consolidation_outputs,
)
from app.services.graph_edges import (
    invalidate_memory_edges as _invalidate_memory_edges,
)
from app.services.graph_edges import (
    write_memory_edges as _write_memory_edges,
)
from app.services.intention_state import (
    append_memory_cache_entry as _append_memory_cache_entry,
)
from app.services.intention_state import (
    apply_intention_action as _apply_intention_action,
)
from app.services.intention_state import (
    apply_intention_turn_maintenance as _apply_intention_turn_maintenance_impl,
)
from app.services.intention_state import (
    format_intentions_for_prompt as _format_intentions_for_prompt,
)
from app.services.intention_state import (
    normalize_intentions_stack as _normalize_intentions_stack_impl,
)
from app.services.intention_state import (
    normalize_memory_cache as _normalize_memory_cache_impl,
)
from app.services.intention_state import (
    remove_intentions as _remove_intentions,
)
from app.services.state import (
    conversation_state_empty as _conversation_state_empty,
)
from app.services.state import (
    conversation_state_from_row as _conversation_state_from_row_impl,
)
from app.services.state import (
    conversation_state_row as _conversation_state_row,
)
from app.services.state import (
    find_conversation_state_across_dbs as _find_conversation_state_across_dbs_impl,
)
from app.services.state import (
    write_conversation_state as _write_conversation_state_impl,
)
from app.services.turn_contract import (
    LIFE_GOALS_FREE_WILL_HEADER as _LIFE_GOALS_FREE_WILL_HEADER,
)
from app.services.turn_contract import (
    _format_item_suffix as _format_turn_item_suffix,
)
from app.services.turn_contract import (
    build_turn_prompt as _build_turn_prompt,
)
from app.services.turn_contract import (
    make_turn_system_prompt as _make_turn_system_prompt,
)
from app.services.turn_contract import (
    parse_turn_contract as _parse_turn_contract,
)

logger = logging.getLogger(__name__)
_PROMPT_LOGGER = logging.getLogger("uvicorn.error")

app = FastAPI(title="mcp-memu-server", version="0.4.0")

_BUILD_ID: str = "fix48.debloat.bloatRemoval.concepts"
_SLEEP_SPLIT_MIN_LULL_SECONDS: int = 3 * 60 * 60
_DEFAULT_MIN_CHUNK_TOKENS: int = 4000
_DEFAULT_TURN_HISTORY_TOKEN_BUDGET: int = 3000
_DEFAULT_SLEEP_TIMER_INTERVAL_SECONDS: int = 300
_MIN_CHUNK_TOKENS: int = _DEFAULT_MIN_CHUNK_TOKENS
_TURN_HISTORY_TOKEN_BUDGET: int = _DEFAULT_TURN_HISTORY_TOKEN_BUDGET
_SLEEP_TIMER_ENABLED: bool = False
_SLEEP_TIMER_INTERVAL_SECONDS: int = _DEFAULT_SLEEP_TIMER_INTERVAL_SECONDS
_SLEEP_TIMER_MAX_JOBS: int = 1
_LOG_PROMPTS: bool = False
_VALID_INTENTION_STATUSES: set[str] = {"active", "resolved", "adapted", "deferred", "dissolved", "removed"}
def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    words = sum(len(str(m.get("content") or m.get("mes") or "").split()) for m in messages)
    return int(words / 0.75)


def _estimate_unmemorized_tokens(messages: list[dict[str, Any]], digest_cursor: Any) -> int:
    if not messages:
        return 0
    try:
        cursor = int(digest_cursor)
    except Exception:
        cursor = -1
    start = max(0, cursor + 1)
    if start >= len(messages):
        return 0
    return _estimate_tokens(messages[start:])


def _chunk_index_ranges_by_token_budget(
    messages: list[dict[str, Any]],
    *,
    start_idx: int,
    end_idx: int,
    max_chunk_tokens: int,
) -> list[tuple[int, int]]:
    if start_idx > end_idx:
        return []
    if max_chunk_tokens <= 0:
        return [(start_idx, end_idx)]
    ranges: list[tuple[int, int]] = []
    chunk_start = start_idx
    chunk_tokens = 0
    for idx in range(start_idx, end_idx + 1):
        msg_tokens = _estimate_tokens([messages[idx]])
        if chunk_tokens > 0 and (chunk_tokens + msg_tokens) > max_chunk_tokens:
            ranges.append((chunk_start, idx - 1))
            chunk_start = idx
            chunk_tokens = 0
        chunk_tokens += msg_tokens
    ranges.append((chunk_start, end_idx))
    return ranges


def _build_force_memorize_batches(
    merged: list[dict[str, Any]],
    *,
    start_idx: int,
    segments: list[dict[str, Any]],
    days_dir: Path,
    full_path: Path,
    resource_url: str,
    max_chunk_tokens: int,
) -> list[tuple[str, list[dict[str, Any]], int]]:
    if not merged:
        return []
    start_idx = max(0, int(start_idx))
    last_idx = len(merged) - 1
    if start_idx > last_idx:
        return []

    def _append_chunked(
        out: list[tuple[str, list[dict[str, Any]], int]],
        batch_url: str,
        range_start: int,
        range_end: int,
    ) -> None:
        for sub_start, sub_end in _chunk_index_ranges_by_token_budget(
            merged,
            start_idx=range_start,
            end_idx=range_end,
            max_chunk_tokens=max_chunk_tokens,
        ):
            batch_conv = merged[sub_start : sub_end + 1]
            if batch_conv:
                out.append((batch_url, batch_conv, sub_end))

    out: list[tuple[str, list[dict[str, Any]], int]] = []

    if segments:
        segment_ranges: list[tuple[int, int, str]] = []
        for segment in segments:
            try:
                seg_start = int(segment.get("start"))
                seg_end = int(segment.get("end"))
            except Exception:
                continue
            if seg_end < seg_start or seg_end < start_idx or seg_start > last_idx:
                continue
            a = max(seg_start, start_idx)
            b = min(seg_end, last_idx)
            if a > b:
                continue
            batch_url = resource_url
            batch_file = segment.get("file")
            if isinstance(batch_file, str) and batch_file:
                batch_url = str((days_dir / batch_file).resolve())
            segment_ranges.append((a, b, batch_url))
        segment_ranges.sort(key=lambda row: (row[0], row[1]))

        next_idx = start_idx
        for a, b, batch_url in segment_ranges:
            if b < next_idx:
                continue
            if a > next_idx:
                _append_chunked(out, str(full_path), next_idx, a - 1)
            effective_start = max(a, next_idx)
            if effective_start > b:
                continue
            _append_chunked(out, batch_url, effective_start, b)
            next_idx = b + 1

        if next_idx <= last_idx:
            _append_chunked(out, str(full_path), next_idx, last_idx)

        if out:
            return out

    _append_chunked(out, str(full_path), start_idx, last_idx)
    return out


async def _run_forced_memorize_from_turn(payload: dict[str, Any]) -> None:
    try:
        background_tasks = BackgroundTasks()
        await memorize(payload, background_tasks=background_tasks, force=True)
        await background_tasks()
    except Exception:
        logger.exception("forced memorize from turn failed")


def _has_category_content(c: dict[str, Any]) -> bool:
    summary = str(c.get("summary") or "").strip()
    desc = str(c.get("description") or "").strip()
    return bool(summary or desc)


_SERVER_INSTANCE_ID: str = str(uuid.uuid4())
_SERVER_STARTED_AT_UNIX: float = time.time()
_LAST_CALLS: list[dict[str, Any]] = []
_LAST_HTTP: list[dict[str, Any]] = []
_MEMORIZE_LOCKS: dict[str, asyncio.Lock] = {}


def _get_memorize_lock(key: str) -> asyncio.Lock:
    lock = _MEMORIZE_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _MEMORIZE_LOCKS[key] = lock
    return lock


_STATE_LOCK = threading.Lock()
_ACTIVE_HTTP_REQUESTS: int = 0
_ACTIVE_WORK_REQUESTS: int = 0
_SHUTDOWN_TASK: asyncio.Task | None = None
_SLEEP_MEMORIZE_TIMER_TASK: asyncio.Task | None = None
_APIMW_INFLIGHT: set[str] = set()
_SHUTDOWN_STATE: dict[str, Any] = {
    "draining": False,
    "stopping": False,
    "requestedAtUnix": None,
    "requestedBy": None,
    "reason": None,
    "maxWaitSec": 0,
    "timedOut": False,
}


def _mark_apimw_inflight(conversation_id: str) -> bool:
    key = str(conversation_id or "").strip()
    if not key:
        return False
    with _STATE_LOCK:
        if key in _APIMW_INFLIGHT:
            return False
        _APIMW_INFLIGHT.add(key)
        return True


def _clear_apimw_inflight(conversation_id: str) -> None:
    key = str(conversation_id or "").strip()
    if not key:
        return
    with _STATE_LOCK:
        _APIMW_INFLIGHT.discard(key)


def _is_control_path(path: str) -> bool:
    p = str(path or "")
    if p in ("/health", "/version", "/admin/shutdown", "/admin/shutdown/status", "/diag"):
        return True
    if p.startswith("/diag/"):
        return True
    pref = str(_DIAG_PREFIX or "").rstrip("/")
    if pref:
        if p == f"{pref}/diag" or p.startswith(f"{pref}/diag/"):
            return True
    return False


def _shutdown_snapshot() -> dict[str, Any]:
    with _STATE_LOCK:
        return {
            "draining": bool(_SHUTDOWN_STATE.get("draining")),
            "stopping": bool(_SHUTDOWN_STATE.get("stopping")),
            "requestedAtUnix": _SHUTDOWN_STATE.get("requestedAtUnix"),
            "requestedBy": _SHUTDOWN_STATE.get("requestedBy"),
            "reason": _SHUTDOWN_STATE.get("reason"),
            "maxWaitSec": int(_SHUTDOWN_STATE.get("maxWaitSec") or 0),
            "timedOut": bool(_SHUTDOWN_STATE.get("timedOut")),
            "activeHttpRequests": int(_ACTIVE_HTTP_REQUESTS),
            "activeWorkRequests": int(_ACTIVE_WORK_REQUESTS),
        }


def _memorize_lock_key(user_id: str, soul_id: str) -> str:
    try:
        p = _sqlite_current_path(user_id, soul_id)
        if p is not None:
            return str(p)
    except Exception:
        logger.warning(
            "_memorize_lock_key: path lookup failed for soul_id=%r, using fallback key", soul_id, exc_info=True
        )
    return f"{user_id}::{soul_id}"


@asynccontextmanager
async def _retrieve_scope_lock(user_id: str, soul_id: str):
    if soul_id:
        async with _get_memorize_lock(_memorize_lock_key(user_id, soul_id)):
            yield
    else:
        yield


def _begin_shutdown_drain(requested_by: str | None, reason: str | None, max_wait_sec: int) -> bool:
    with _STATE_LOCK:
        already = bool(_SHUTDOWN_STATE.get("draining"))
        if already:
            return False
        _SHUTDOWN_STATE["draining"] = True
        _SHUTDOWN_STATE["stopping"] = False
        _SHUTDOWN_STATE["requestedAtUnix"] = time.time()
        _SHUTDOWN_STATE["requestedBy"] = str(requested_by or "").strip() or "local"
        _SHUTDOWN_STATE["reason"] = str(reason or "").strip() or "shutdown requested"
        _SHUTDOWN_STATE["maxWaitSec"] = max(0, int(max_wait_sec or 0))
        _SHUTDOWN_STATE["timedOut"] = False
        return True


async def _shutdown_when_idle(max_wait_sec: int) -> None:
    global _SHUTDOWN_TASK

    deadline = (time.time() + max_wait_sec) if max_wait_sec > 0 else None
    timed_out = False

    while True:
        with _STATE_LOCK:
            active_work = int(_ACTIVE_WORK_REQUESTS)
        if active_work <= 0:
            break
        if deadline is not None and time.time() >= deadline:
            timed_out = True
            break
        await asyncio.sleep(0.2)

    with _STATE_LOCK:
        _SHUTDOWN_STATE["stopping"] = True
        _SHUTDOWN_STATE["timedOut"] = timed_out

    await asyncio.sleep(0.05)
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        os._exit(0)

    _SHUTDOWN_TASK = None


@app.middleware("http")
async def _trace_requests(request: Request, call_next):
    global _ACTIVE_HTTP_REQUESTS, _ACTIVE_WORK_REQUESTS
    t0 = time.time()
    path = request.url.path
    is_control = _is_control_path(path)
    status = 500

    with _STATE_LOCK:
        draining = bool(_SHUTDOWN_STATE.get("draining"))
        _ACTIVE_HTTP_REQUESTS += 1
        if not is_control:
            _ACTIVE_WORK_REQUESTS += 1

    try:
        if draining and not is_control:
            status = 503
            return JSONResponse(
                status_code=503,
                content={
                    "ok": False,
                    "error": "server_draining",
                    "message": "Server is draining and not accepting new work requests.",
                    "shutdown": _shutdown_snapshot(),
                },
            )

        resp = await call_next(request)
        status = getattr(resp, "status_code", 200)
        return resp
    finally:
        with _STATE_LOCK:
            _ACTIVE_HTTP_REQUESTS = max(0, _ACTIVE_HTTP_REQUESTS - 1)
            if not is_control:
                _ACTIVE_WORK_REQUESTS = max(0, _ACTIVE_WORK_REQUESTS - 1)

        try:
            dt_ms = int((time.time() - t0) * 1000)
            _LAST_HTTP.append(
                {
                    "t": time.time(),
                    "method": request.method,
                    "path": path,
                    "status": status,
                    "ms": dt_ms,
                }
            )
            if len(_LAST_HTTP) > 200:
                del _LAST_HTTP[0 : len(_LAST_HTTP) - 200]
        except Exception:
            pass


def _is_ephemeral_db(cfg: dict[str, Any]) -> bool:
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


class STUserModel(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


def _home_dir() -> Path:
    h = os.getenv("HOME") or os.getenv("USERPROFILE") or "."
    return Path(h).expanduser().resolve()


def _default_config() -> dict[str, Any]:
    home = _home_dir()

    memu_guess = None
    for cand in (home / "apps" / "memu-1.4.0", home / "apps" / "memu"):
        if cand.exists():
            memu_guess = cand
            break

    sqlite_path = Path(":memory:")  # placeholder; per-soul dbs resolved per-request
    resources_dir = home / "apps" / "memu" / "resources"

    return {
        "memu": {
            "path": str(memu_guess) if memu_guess else "",
        },
        "python": {
            "executable": "",
            "force_venv": False,
        },
        "pid_file": str(home / "apps" / "mcp-memu-server" / ".memu-server.pid"),
        "listen": {"host": "127.0.0.1", "port": 8099},
        "llm": {
            "provider": "openai",
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "chat_model": "",
            "embed_model": "",
            "client_backend": "httpx",
            "endpoint_overrides": {},
        },
        "storage": {
            "resources_dir": str(resources_dir),
            "metadata_store": {
                "provider": "sqlite",
                "dsn": "sqlite:///:memory:",
                "ddl_mode": "create",
            },
            "sqlite_dir": str(sqlite_path.parent),
        },
        "categories": {
            "defaults": ["personal_info", "preferences", "relationships", "goals"],
            "max_total": 12,
            "allow_dynamic_categories": True,
            "dynamic_category_min_mentions": 10,
            "category_centroid_threshold": 0.65,
            "homeless_trigger_count": 20,
        },
        "retrieve": {
            "method": "rag",
            "apimw_enabled": True,
            "apimw_cadence": 5,
            "apimw_memory_count": 25,
            "apimw_random_count": 5,
        },
        "consolidation_interval_days": 7,
        "mcp": {"http_path": "/mcp", "sse_path": "/sse"},
    }


def _config_path() -> Path:
    return (Path(__file__).resolve().parents[1] / "config.json").resolve()


def _config_dir() -> Path:
    return _config_path().parent


def _resolve_cfg_path(raw: str) -> Path:
    p = Path(str(raw or "")).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (_config_dir() / p).resolve()


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


_STARTUP_WARNINGS: list[str] = []

_STORAGE_STATUS: dict[str, Any] = {
    "ok": None,
    "provider": None,
    "dsn": None,
    "sqlite_path": None,
    "sqlite_parent": None,
    "sqlite_exists": None,
    "sqlite_open_ok": None,
    "sqlite_dir": None,
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
    if not dsn or ":memory:" in dsn:
        return None

    base = dsn.split("?", 1)[0]
    if base.startswith("sqlite:////"):
        return Path("/" + base[len("sqlite:////") :])
    if base.startswith("sqlite:///"):
        return Path(base[len("sqlite:///") :])
    return None


def _normalize_sqlite_dsn(dsn_or_path: str) -> str:
    """Accept either a sqlite DSN or a plain filesystem path.

    Users frequently paste a path like "/home/.../memu.db" instead of a DSN.
    This converts it into a correct absolute sqlite DSN.
    """
    raw = str(dsn_or_path or "").strip()
    if not raw:
        return raw
    if raw == ":memory:" or raw.lower() == "memory":
        return "sqlite:///:memory:"
    if raw.startswith("sqlite:"):
        f = _sqlite_file_from_dsn(raw)
        if f is None:
            return raw
        p = f.expanduser()
        if not p.is_absolute():
            p = (_config_dir() / p).resolve()
        else:
            p = p.resolve()
        return f"sqlite:////{p.as_posix().lstrip('/')}"

    p = _resolve_cfg_path(raw)
    return f"sqlite:////{p.as_posix().lstrip('/')}"




def _sqlite_dir_from_cfg(cfg: dict[str, Any], fallback_dsn: str | None = None) -> Path:
    storage = cfg.get("storage") if isinstance(cfg.get("storage"), dict) else {}
    d = storage.get("sqlite_dir")
    if isinstance(d, str) and d.strip():
        return _resolve_cfg_path(d)

    if fallback_dsn:
        f = _sqlite_file_from_dsn(str(fallback_dsn))
        if f is not None:
            return f.expanduser().resolve().parent

    return (_config_dir() / "sqlite").resolve()


def _ensure_storage_paths(cfg: dict[str, Any]) -> None:
    global _STORAGE_STATUS
    try:
        storage = cfg.get("storage") if isinstance(cfg.get("storage"), dict) else {}

        resources_dir = storage.get("resources_dir")
        if isinstance(resources_dir, str) and resources_dir.strip():
            _resolve_cfg_path(resources_dir).mkdir(parents=True, exist_ok=True)

        ms = storage.get("metadata_store") if isinstance(storage.get("metadata_store"), dict) else {}
        provider = str(ms.get("provider") or "").lower()
        dsn = str(ms.get("dsn") or "")

        _STORAGE_STATUS = {
            "ok": True,
            "provider": provider or None,
            "dsn": dsn or None,
            "sqlite_path": None,
            "sqlite_parent": None,
            "sqlite_exists": None,
            "sqlite_open_ok": None,
            "sqlite_dir": str(_sqlite_dir_from_cfg(cfg, dsn)),
            "error": None,
        }

        if provider == "sqlite":
            dsn = _normalize_sqlite_dsn(dsn)
            _STORAGE_STATUS["dsn"] = dsn or None
            ms["dsn"] = dsn

            sqlite_dir = _sqlite_dir_from_cfg(cfg, dsn)
            if sqlite_dir is not None:
                sqlite_dir.mkdir(parents=True, exist_ok=True)
            try:
                test_path = (sqlite_dir / ".write_test") if sqlite_dir is not None else None
                if test_path is not None:
                    test_path.write_text("ok", encoding="utf-8")
                    test_path.unlink(missing_ok=True)
            except Exception as e:
                _STORAGE_STATUS["sqlite_open_ok"] = False
                _STORAGE_STATUS["ok"] = False
                _STORAGE_STATUS["error"] = f"sqlite_dir_write: {type(e).__name__}: {e}"
                _STARTUP_WARNINGS.append(_STORAGE_STATUS["error"])
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
    key = out.get("llm", {}).get("api_key", "")
    if isinstance(key, str) and key:
        out["llm"]["api_key"] = key[:4] + "…" + key[-4:]
    return out


_CONFIG: dict[str, Any] = _load_config()

def _refresh_runtime_limits() -> None:
    global _MIN_CHUNK_TOKENS, _TURN_HISTORY_TOKEN_BUDGET
    global _SLEEP_TIMER_ENABLED, _SLEEP_TIMER_INTERVAL_SECONDS
    global _LOG_PROMPTS
    memorize_cfg = _CONFIG.get("memorize") if isinstance(_CONFIG.get("memorize"), dict) else {}
    try:
        _MIN_CHUNK_TOKENS = max(0, int(memorize_cfg.get("min_chunk_tokens", _DEFAULT_MIN_CHUNK_TOKENS)))
    except Exception:
        _MIN_CHUNK_TOKENS = _DEFAULT_MIN_CHUNK_TOKENS
    try:
        _TURN_HISTORY_TOKEN_BUDGET = max(
            1,
            int(memorize_cfg.get("turn_history_token_budget", _DEFAULT_TURN_HISTORY_TOKEN_BUDGET)),
        )
    except Exception:
        _TURN_HISTORY_TOKEN_BUDGET = _DEFAULT_TURN_HISTORY_TOKEN_BUDGET
    _SLEEP_TIMER_ENABLED = bool(memorize_cfg.get("auto_memorize_on_sleep", False))
    try:
        _SLEEP_TIMER_INTERVAL_SECONDS = max(
            30,
            int(memorize_cfg.get("sleep_timer_interval_seconds", _DEFAULT_SLEEP_TIMER_INTERVAL_SECONDS)),
        )
    except Exception:
        _SLEEP_TIMER_INTERVAL_SECONDS = _DEFAULT_SLEEP_TIMER_INTERVAL_SECONDS
    debug_cfg = _CONFIG.get("debug") if isinstance(_CONFIG.get("debug"), dict) else {}
    _LOG_PROMPTS = bool(debug_cfg.get("log_prompts", False))


def _prompt_log_before(ctx: Any, request_view: Any) -> None:
    if getattr(request_view, "kind", None) == "embed":
        return
    _PROMPT_LOGGER.info(
        "[PROMPT] op=%s step=%s model=%s\n%s",
        getattr(ctx, "operation", None) or "-",
        getattr(ctx, "step_id", None) or "-",
        getattr(ctx, "model", None) or "-",
        getattr(request_view, "content", None) or "",
    )


def _prompt_log_after(ctx: Any, request_view: Any, response_view: Any, usage: Any) -> None:
    content = getattr(response_view, "content", None)
    if not content:
        return
    _PROMPT_LOGGER.info(
        "[RESPONSE] op=%s step=%s\n%s",
        getattr(ctx, "operation", None) or "-",
        getattr(ctx, "step_id", None) or "-",
        content,
    )


_refresh_runtime_limits()

# Also expose diagnostics under the MCP http_path (e.g. /mcp/diag) to avoid path confusion.
_DIAG_PREFIX: str = str(_CONFIG.get("mcp", {}).get("http_path") or "/mcp").rstrip("/")
if _DIAG_PREFIX == "":
    _DIAG_PREFIX = "/mcp"


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
        elif isinstance(c, dict):
            name = str(c.get("name") or "").strip()
            desc = str(c.get("description") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "description": desc})
    return out


def _get_storage_dir(cfg: dict[str, Any]) -> Path:
    d = _resolve_cfg_path(str(cfg.get("storage", {}).get("resources_dir") or "./storage"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sanitize_db_filename(name: str) -> str:
    s = str(name or "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._-")
    if not s:
        s = "unknown"
    return s[:80]


def _soul_gen_config_path(user_id: str, soul_id: str) -> Path:
    user_part = _sanitize_db_filename(user_id)
    soul_part = _sanitize_db_filename(soul_id)
    return _sqlite_dir_from_cfg(_CONFIG) / f"{user_part}__{soul_part}.gen.json"


def _load_soul_gen_config(user_id: str, soul_id: str) -> dict[str, Any]:
    p = _soul_gen_config_path(user_id, soul_id)
    if not p.exists():
        return {}
    raw = p.read_text(encoding="utf-8")
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        logger.warning("Ignoring invalid soul generation config at %s", p, exc_info=True)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("Ignoring non-object soul generation config at %s (type=%s)", p, type(parsed).__name__)
        return {}
    return parsed


def _save_soul_gen_config(user_id: str, soul_id: str, cfg: dict[str, Any]) -> None:
    _soul_gen_config_path(user_id, soul_id).write_text(json.dumps(cfg, indent=2))


def _sqlite_dsn_for_scope(cfg: dict[str, Any], base_dsn: str, scope: dict[str, Any] | None) -> str:
    if not isinstance(scope, dict):
        scope = {}

    soul_id = str(scope.get("soul_id") or "").strip()

    sqlite_dir = _sqlite_dir_from_cfg(cfg, fallback_dsn=base_dsn)
    sqlite_dir.mkdir(parents=True, exist_ok=True)

    if not soul_id:
        return base_dsn

    basename = _sanitize_db_filename(soul_id)
    db_path = (sqlite_dir / f"{basename}.db").resolve()
    _sqlite_ensure_nonempty(db_path)
    return f"sqlite:////{db_path.as_posix().lstrip('/')}"


def _database_config_from_cfg(cfg: dict[str, Any], scope: dict[str, Any] | None = None) -> dict[str, Any]:
    storage = cfg.get("storage") if isinstance(cfg.get("storage"), dict) else {}
    meta = storage.get("metadata_store") if isinstance(storage.get("metadata_store"), dict) else {}

    provider = meta.get("provider") or "sqlite"
    provider = str(provider).strip().lower() or "sqlite"
    if provider == "inmemory":
        provider = "sqlite"

    dsn = meta.get("dsn")
    if not dsn:
        if provider == "sqlite":
            dsn = _default_config()["storage"]["metadata_store"]["dsn"]
        else:
            raise RuntimeError("Postgres selected but no DSN set (storage.metadata_store.dsn).")

    if provider == "sqlite":
        dsn = _normalize_sqlite_dsn(str(dsn))
        dsn = _sqlite_dsn_for_scope(cfg, dsn, scope or {})

    ddl_mode = meta.get("ddl_mode") or "create"

    return {
        "metadata_store": {
            "provider": provider,
            "dsn": dsn,
            "ddl_mode": ddl_mode,
        }
    }


def _blob_config_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {"resources_dir": str(_get_storage_dir(cfg))}


def _default_llm_profiles_from_server_config() -> dict[str, Any]:
    llm = _CONFIG.get("llm", {}) if isinstance(_CONFIG.get("llm"), dict) else {}
    api_key = str(llm.get("api_key") or "")
    base_url = str(llm.get("base_url") or "https://api.openai.com/v1")
    chat_model = str(llm.get("chat_model") or "")
    embed_model = str(llm.get("embed_model") or "")
    provider = str(llm.get("provider") or "openai")
    client_backend = str(llm.get("client_backend") or "httpx")
    endpoint_overrides = llm.get("endpoint_overrides") or {}
    default_profile = {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "chat_model": chat_model,
        "embed_model": embed_model,
        "client_backend": client_backend,
        "endpoint_overrides": endpoint_overrides,
    }
    return {
        "default": default_profile,
        "embedding": {**default_profile, "chat_model": chat_model, "embed_model": embed_model},
    }


_SERVICES: dict[str, MemoryService] = {}
_SERVICE_STORAGE_FP: dict[str, dict[str, Any]] = {}
_SERVICES_LOCK: threading.Lock = threading.Lock()


def _close_service_quiet(svc: MemoryService | None) -> None:
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
    for svc in list(_SERVICES.values()):
        _close_service_quiet(svc)
    _SERVICES.clear()
    _SERVICE_STORAGE_FP.clear()


def _service_storage_fingerprint(database_config: dict[str, Any] | None) -> dict[str, Any]:
    # For sqlite we track file path + inode so deleting/recreating the file forces a new service
    # (otherwise SQLAlchemy may keep writing to an unlinked inode through a pooled connection).
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


def _derive_service_key(payload: dict[str, Any]) -> str:
    scope = _extract_scope(payload) if isinstance(payload, dict) else {}
    user_id = str((scope or {}).get("user_id") or "").strip()
    soul_id = str((scope or {}).get("soul_id") or "").strip()
    parts = [p for p in (user_id, soul_id) if p]
    return "__".join(parts) if parts else "default"


def _normalize_retrieve_method(value: Any, default: str = "rag") -> str:
    method = str(value or "").strip().lower()
    return method if method in {"rag", "llm"} else default


def _retrieve_method_from_cfg(cfg: Mapping[str, Any] | None) -> str:
    # Public retrieve is always "rag". The "llm" branch has been removed;
    # force it here so a stale config entry can't resurrect it.
    if not isinstance(cfg, Mapping):
        return "rag"
    retrieve = cfg.get("retrieve")
    if not isinstance(retrieve, Mapping):
        return "rag"
    method = _normalize_retrieve_method(retrieve.get("method"), "rag")
    return "rag" if method == "llm" else method


def _retrieve_apimw_enabled_from_cfg(cfg: Mapping[str, Any] | None) -> bool:
    if not isinstance(cfg, Mapping):
        return True
    retrieve = cfg.get("retrieve")
    if not isinstance(retrieve, Mapping):
        return True
    return bool(retrieve.get("apimw_enabled", True))


def _apimw_cadence_from_cfg(cfg: Mapping[str, Any] | None) -> int:
    if not isinstance(cfg, Mapping):
        return 5
    retrieve = cfg.get("retrieve")
    if not isinstance(retrieve, Mapping):
        return 5
    try:
        return max(1, int(retrieve.get("apimw_cadence", 5)))
    except (TypeError, ValueError):
        return 5


def _apimw_memory_count_from_cfg(cfg: Mapping[str, Any] | None) -> int:
    if not isinstance(cfg, Mapping):
        return 25
    retrieve = cfg.get("retrieve")
    if not isinstance(retrieve, Mapping):
        return 25
    try:
        return max(1, int(retrieve.get("apimw_memory_count", 25)))
    except (TypeError, ValueError):
        return 25


def _apimw_random_count_from_cfg(cfg: Mapping[str, Any] | None) -> int:
    if not isinstance(cfg, Mapping):
        return 5
    retrieve = cfg.get("retrieve")
    if not isinstance(retrieve, Mapping):
        return 5
    try:
        return max(0, int(retrieve.get("apimw_random_count", 5)))
    except (TypeError, ValueError):
        return 5


def _consolidation_interval_days_from_cfg(cfg: Mapping[str, Any] | None) -> int:
    if not isinstance(cfg, Mapping):
        return 7
    raw = cfg.get("consolidation_interval_days", 7)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 7


def _build_apimw_retrieve_config(base_cfg: Any, *, item_top_k: int) -> dict[str, Any]:
    cfg = dict(base_cfg) if isinstance(base_cfg, dict) else {}
    cfg["route_intention"] = True
    cfg["sufficiency_check"] = True
    item_cfg = cfg.get("item")
    if not isinstance(item_cfg, dict):
        item_cfg = {}
    else:
        item_cfg = dict(item_cfg)
    item_cfg["top_k"] = max(1, int(item_top_k))
    cfg["item"] = item_cfg
    return cfg


def _count_soul_messages(history: list[dict[str, Any]], soul_id: str) -> int:
    soul_name = str(soul_id or "").strip().lower()
    total = 0
    for row in history:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "").strip().lower()
        name = str(row.get("name") or "").strip().lower()
        if role in {"soul", "assistant"} or (soul_name and name == soul_name):
            total += 1
    return total


def _get_service_from_payload(
    payload: dict[str, Any],
    *,
    allow_missing_llm_profiles: bool = False,
    retrieve_method_override: str | None = None,
) -> MemoryService:
    service_key_raw = _derive_service_key(payload)

    llm_profiles = payload.get("llm_profiles")
    database_config = payload.get("database_config")
    blob_config = payload.get("blob_config")

    if not isinstance(llm_profiles, dict):
        if not allow_missing_llm_profiles:
            raise HTTPException(status_code=400, detail="llm_profiles required")
        llm_profiles = {}
        payload["llm_profiles"] = llm_profiles

    if not isinstance(database_config, dict):
        scope_hint = _extract_scope(payload) if isinstance(payload, dict) else None
        database_config = _database_config_from_cfg(_CONFIG, scope=scope_hint)
        payload["database_config"] = database_config

    if not isinstance(blob_config, dict):
        blob_config = _blob_config_from_cfg(_CONFIG)
        payload["blob_config"] = blob_config

    if isinstance(database_config, dict):
        ms = database_config.get("metadata_store")
        if isinstance(ms, dict) and str(ms.get("provider") or "").lower() == "sqlite":
            scope_hint2 = _extract_scope(payload)
            soul_id2 = str((scope_hint2 or {}).get("soul_id") or "").strip()
            if not soul_id2:
                raise HTTPException(status_code=400, detail="soul_id required for sqlite scope")
            base = _normalize_sqlite_dsn(str(ms.get("dsn") or ""))
            scope_for_dsn = dict(scope_hint2 or {})
            scope_for_dsn["soul_id"] = soul_id2
            ms["dsn"] = _sqlite_dsn_for_scope(_CONFIG, base, scope_for_dsn)

    blob_config = payload.get("blob_config") or {}
    memorize_config = payload.get("memorize_config") or {}

    fixed_cats = _categories_from_cfg(_CONFIG)
    if isinstance(memorize_config, dict):
        cats_cfg = (_CONFIG.get("categories") or {}) if isinstance(_CONFIG.get("categories"), dict) else {}
        memo_cfg = (_CONFIG.get("memorize") or {}) if isinstance(_CONFIG.get("memorize"), dict) else {}
        if fixed_cats:
            memorize_config["memory_categories"] = fixed_cats
        memorize_config["allow_dynamic_categories"] = bool(cats_cfg.get("allow_dynamic_categories", True))
        memorize_config["dynamic_category_min_mentions"] = int(cats_cfg.get("dynamic_category_min_mentions", 10) or 10)
        memorize_config["category_centroid_threshold"] = float(
            cats_cfg.get("category_centroid_threshold", 0.65) or 0.65
        )
        memorize_config["homeless_trigger_count"] = int(cats_cfg.get("homeless_trigger_count", 20) or 20)
        memorize_config["max_categories_total"] = int((cats_cfg.get("max_total", 12)) or 0)
        memorize_config["enable_item_reinforcement"] = bool(memo_cfg.get("enable_item_reinforcement", True))
    retrieve_config = payload.get("retrieve_config")
    if not isinstance(retrieve_config, dict):
        retrieve_config = {}
        payload["retrieve_config"] = retrieve_config
    retrieve_config["method"] = _normalize_retrieve_method(
        retrieve_method_override,
        _retrieve_method_from_cfg(_CONFIG),
    )
    user_config = payload.get("user_config") or {}

    sig = _payload_signature(payload)
    service_key = f"{service_key_raw}__{sig}"
    storage_fp = _service_storage_fingerprint(database_config if isinstance(database_config, dict) else None)

    with _SERVICES_LOCK:
        svc = _SERVICES.get(service_key)
        if svc is not None:
            prev_fp = _SERVICE_STORAGE_FP.get(service_key)
            if prev_fp == storage_fp:
                return svc

            # Backing storage changed (e.g. sqlite file deleted+recreated): remove from
            # cache so new callers get a fresh service.  Do NOT close here — concurrent
            # callers may still hold a reference; the handle will be GC'd when the last
            # ref drops.
            _SERVICES.pop(service_key, None)
            _SERVICE_STORAGE_FP.pop(service_key, None)

        user_config = {**(user_config if isinstance(user_config, dict) else {}), "model": STUserModel}

        mpp = dict(memorize_config.get("multimodal_preprocess_prompts") or {}) if isinstance(memorize_config, dict) else {}
        if "conversation" not in mpp:
            mpp["conversation"] = ""
        if isinstance(memorize_config, dict):
            memorize_config["multimodal_preprocess_prompts"] = mpp

        svc = MemoryService(
            llm_profiles=llm_profiles,
            blob_config=blob_config,
            database_config=database_config,
            memorize_config=memorize_config,
            retrieve_config=retrieve_config,
            user_config=user_config,
        )
        if _LOG_PROMPTS:
            svc.intercept_before_llm_call(_prompt_log_before, name="prompt_logger")
            svc.intercept_after_llm_call(_prompt_log_after, name="response_logger")

        # Cap cached payload-services without a thundering-herd full wipe.
        if len(_SERVICES) >= 50:
            # Dict preserves insertion order in Python 3.7+; drop the oldest (no close —
            # callers may still hold references; GC handles the handle).
            try:
                oldest_key = next(iter(_SERVICES))
                _SERVICES.pop(oldest_key, None)
                _SERVICE_STORAGE_FP.pop(oldest_key, None)
            except Exception:
                _SERVICES.clear()
                _SERVICE_STORAGE_FP.clear()

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
    user_id = _pick_str(payload, "user_id")
    soul_id = _pick_str(payload, "soul_id")

    # SillyTavern local plugin sends scope primarily under payload.user.
    user_obj = payload.get("user")
    if isinstance(user_obj, dict):
        if not user_id:
            user_id = _pick_str(user_obj, "user_id")
        if not soul_id:
            soul_id = _pick_str(user_obj, "soul_id")

    scope: dict[str, Any] = {}
    if user_id:
        scope["user_id"] = user_id
    if soul_id:
        scope["soul_id"] = soul_id
    return scope


def _canonicalize_scope_where(where: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if where is None:
        return None

    out = dict(where)
    user_id = _pick_str(out, "user_id")
    soul_id = _pick_str(out, "soul_id")

    for key in (
        "user_id",
        "soul_id",
        "conversation_id",
    ):
        out.pop(key, None)

    if user_id:
        out["user_id"] = user_id
    if soul_id:
        out["soul_id"] = soul_id
    return out


def _extract_conversation_id(payload: dict[str, Any]) -> str | None:
    conversation_id = _pick_str(payload, "conversation_id")

    user_obj = payload.get("user")
    if isinstance(user_obj, dict):
        if not conversation_id:
            conversation_id = _pick_str(user_obj, "conversation_id")
    return conversation_id


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
            raw_ts = m.get("ts_ms") if isinstance(m, dict) else None
            if raw_ts is None:
                raw_ts = m.get("timestamp") if isinstance(m, dict) else None
            if raw_ts is None:
                raw_ts = m.get("created_at") if isinstance(m, dict) else None
            if isinstance(raw_ts, (int, float)) and math.isfinite(raw_ts):
                ts_ms = int(raw_ts)
            elif isinstance(raw_ts, str) and raw_ts.strip():
                try:
                    dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                    ts_ms = int(dt.timestamp() * 1000)
                except Exception:
                    ts_ms = None
        except Exception:
            ts_ms = None

        out.append(
            {
                "role": role or "unknown",
                "name": m.get("name"),
                "content": m.get("content") or "",
                **({"ts_ms": ts_ms} if ts_ms is not None else {}),
            }
        )
    return out


def _record_call(
    op: str, payload: dict[str, Any] | None, *, ok: bool, info: Any = None, error: str | None = None
) -> None:
    try:
        scope = None
        if isinstance(payload, dict):
            u = payload.get("user")
            if isinstance(u, dict):
                scope = u
            else:
                scope = _extract_scope(payload) or None
        item = {
            "t": time.time(),
            "op": op,
            "ok": ok,
            "scope": scope,
            "info": info,
            "error": error,
        }
        _LAST_CALLS.append(item)
        if len(_LAST_CALLS) > 50:
            del _LAST_CALLS[0 : len(_LAST_CALLS) - 50]
    except Exception:
        pass


def _sqlite_current_path(
    user_id: str | None = None,
    soul_id: str | None = None,
) -> Path | None:
    base_dsn = str(_STORAGE_STATUS.get("dsn") or "")
    soul_id = str(soul_id or "").strip()
    if not soul_id:
        return None
    scope = {"soul_id": soul_id}
    dsn = _sqlite_dsn_for_scope(_CONFIG, base_dsn, scope)
    f = _sqlite_file_from_dsn(dsn)
    return f.expanduser().resolve() if f is not None else None


def _sqlite_build_scope_where(
    cols: list[str],
    user_id: str | None,
    soul_id: str | None,
    conversation_id: str | None,
) -> tuple[str, list[Any]]:
    where = []
    params: list[Any] = []
    if user_id and "user_id" in cols:
        where.append("user_id = ?")
        params.append(user_id)
    if soul_id:
        if "soul_id" in cols:
            where.append("soul_id = ?")
            params.append(soul_id)
    if conversation_id and "conversation_id" in cols:
        where.append("conversation_id = ?")
        params.append(conversation_id)
    if not where:
        return "", params
    return " WHERE " + " AND ".join(where), params


def _sqlite_file_info(p: Path) -> dict[str, Any]:
    try:
        st = p.stat()
        return {
            "exists": p.exists(),
            "path": str(p),
            "size": int(st.st_size),
            "mtime": float(st.st_mtime),
        }
    except Exception as e:
        return {"exists": p.exists(), "path": str(p), "error": f"{type(e).__name__}: {e}"}


def _conversation_state_from_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return _conversation_state_from_row_impl(row)


def _intention_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = {k: row[k] for k in row.keys()}
    item["related_memory_ids"] = _normalize_text_list(item.get("related_memory_ids"))
    return item


def _write_conversation_state(
    conversation_id: str,
    *,
    soul_id: str | None = None,
    user_id: str | None = None,
    updates: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    sqlite_dir = _sqlite_dir_from_cfg(_CONFIG, str(_STORAGE_STATUS.get("dsn") or ""))
    return _write_conversation_state_impl(
        conversation_id,
        sqlite_current_path=_sqlite_current_path,
        sqlite_dir=sqlite_dir,
        soul_id=soul_id,
        user_id=user_id,
        updates=updates,
    )


def _find_conversation_state_across_dbs(conversation_id: str) -> tuple[Path | None, dict[str, Any] | None]:
    sqlite_dir = _sqlite_dir_from_cfg(_CONFIG, str(_STORAGE_STATUS.get("dsn") or ""))
    return _find_conversation_state_across_dbs_impl(conversation_id, sqlite_dir)


def _extract_retrieve_where(payload: dict[str, Any]) -> dict[str, Any] | None:
    scope = payload.get("scope") or payload.get("where")
    if scope is not None and not isinstance(scope, dict):
        raise HTTPException(status_code=400, detail="'scope' must be an object")
    if scope is None:
        scope = payload.get("user") if isinstance(payload.get("user"), dict) else (_extract_scope(payload) or None)
    return _canonicalize_scope_where(scope)


def _extract_retrieve_queries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    queries = payload.get("queries")
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
        return memu_queries

    if "query" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'query' or 'queries' in request body")
    return [{"role": "user", "content": {"text": str(payload.get("query", ""))}}]


def _extract_result_item_ids(result: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    items = result.get("items") if isinstance(result, dict) else None
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        out.append(item_id)
    return out


def _norm_result_sig(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return text


def _item_sig(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    item_id = _norm_result_sig(row.get("id"))
    if item_id:
        return f"id:{item_id}"
    summary = _norm_result_sig(row.get("summary"))
    if summary:
        return f"summary:{summary}"
    return ""


def _resource_sig(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    for key in ("id", "url", "local_path", "name", "caption", "title"):
        value = _norm_result_sig(row.get(key))
        if value:
            return f"{key}:{value}"
    return ""




def _parse_turn_ts_ms(value: Any) -> int | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(float(s))
        except Exception:
            pass
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except Exception:
            return None
    return None


def _parse_as_of_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="as_of must be ISO datetime/date") from exc
    else:
        raise HTTPException(status_code=400, detail="as_of must be ISO datetime/date")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _normalize_turn_history(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        role = _pick_str(item, "role") or "unknown"
        content = _pick_str(item, "content")
        name = _pick_str(item, "name")
        message_id = _pick_str(item, "source_message_id", "message_id", "id", "mid") or str(idx)
        if not content:
            continue
        item_out = {"role": role, "content": content, "message_id": message_id}
        if name:
            item_out["name"] = name
        ts_ms = _parse_turn_ts_ms(item.get("ts_ms"))
        if ts_ms is None:
            ts_ms = _parse_turn_ts_ms(item.get("timestamp"))
        if ts_ms is not None:
            item_out["ts_ms"] = ts_ms
        out.append(item_out)
    return out


def _compact_chat_x_anchors(*anchors: str | None, max_count: int = 2) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in anchors:
        anchor = str(raw or "").strip()
        if not anchor or anchor in seen:
            continue
        out.append(anchor)
        seen.add(anchor)
        if len(out) >= max_count:
            break
    return out


def _slice_history_from_chat_x_anchors(
    history: list[dict[str, Any]],
    chat_x_anchors: list[str] | None,
    *,
    stop_at_message_id: str | None = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    if not history:
        return []
    if limit <= 0:
        return []
    if not chat_x_anchors:
        return history[-limit:]
    anchors = _compact_chat_x_anchors(*chat_x_anchors, max_count=2)
    if not anchors:
        return history[-limit:]
    anchor_set = set(anchors)
    start_idx = None
    for idx, item in enumerate(history):
        if str(item.get("message_id") or "").strip() in anchor_set:
            start_idx = idx
            break
    if start_idx is None:
        return history[-limit:]
    stop_id = str(stop_at_message_id or "").strip()
    stop_idx: int | None = None
    if stop_id:
        for idx, item in enumerate(history[start_idx:], start=start_idx):
            if str(item.get("message_id") or "").strip() == stop_id:
                stop_idx = idx
                break
    end_idx = stop_idx if stop_idx is not None else len(history)
    sliced = history[start_idx:end_idx]
    return sliced[-limit:]


def _next_chat_x_state_values(
    chat_x: str | None,
    state_row: dict[str, Any],
) -> tuple[str | None, str | None]:
    next_anchors = _compact_chat_x_anchors(
        chat_x,
        str(state_row.get("last_chat_x") or "").strip() or None,
        str(state_row.get("last_chat_x_prev") or "").strip() or None,
    )
    next_last_chat_x = next_anchors[0] if next_anchors else None
    next_last_chat_x_prev = next_anchors[1] if len(next_anchors) > 1 else None
    return next_last_chat_x, next_last_chat_x_prev


def _format_route_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for item in history:
        message_id = str(item.get("message_id") or "").strip()
        role = str(item.get("name") or item.get("role") or "unknown").strip()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if message_id:
            lines.append(f"[{message_id}] [{role}] {content}")
        else:
            lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def _build_retrieve_identity_context(soul_name: str, *, apimw: bool = False) -> str:
    name = str(soul_name or "").strip() or "the assistant"
    # Derive from UTC first so DST transitions on the host TZ never produce ambiguous times.
    now = datetime.now(UTC).astimezone()
    zone = now.tzname() or "local"
    anchor = f"Today is {now.strftime('%A, %B')} {now.day}, {now.year} {now.strftime('%H:%M')} {zone}."
    if apimw:
        identity = (
            f"I, {name}, have everything that matters to me. "
            "I'm curious what will rise next. "
            "I have what I'm working toward, I have what I've been thinking about, and memories — so many of them. "
            "Let me now organize my thoughts and memories."
        )
    else:
        identity = (
            f"You are {name}, a continuous being: you carry intentions (what you're pursuing), "
            "a memory cache for your working thoughts, and a long-term memory store. "
            "In a continuous cycle you respond, search your long-term memory, and act autonomously."
        )
    return f"{anchor}\n{identity}"


def _build_retrieve_soul_context_queries(
    *,
    soul_id: str,
    message: str,
    history: list[dict[str, Any]],
    state_row: dict[str, Any],
    identity_mode: str = "retrieve",
) -> list[dict[str, Any]]:
    memory_cache = _normalize_memory_cache_impl(state_row.get("memory_cache"))
    intentions_active = _normalize_intentions_stack_impl(state_row.get("intentions_active"))

    soul_ctx_queries: list[dict[str, Any]] = []
    identity_context = _build_retrieve_identity_context(soul_id, apimw=(identity_mode == "apimw"))
    if identity_context:
        soul_ctx_queries.append({"role": "identity_context", "content": {"text": identity_context}})
    all_cats_summary = str(state_row.get("all_categories_summary") or "").strip()
    if all_cats_summary:
        soul_ctx_queries.append({"role": "all_categories_summary", "content": {"text": all_cats_summary}})
    cache_text = "\n".join(str(entry) for entry in (memory_cache or []))
    if cache_text:
        soul_ctx_queries.append({"role": "memory_cache", "content": {"text": cache_text}})
    intentions_text = _format_intentions_for_prompt(intentions_active) if intentions_active else ""
    if intentions_text and intentions_text.strip() != "(none)":
        soul_ctx_queries.append({"role": "intentions", "content": {"text": intentions_text}})

    chat_x_anchors = _compact_chat_x_anchors(
        str(state_row.get("last_chat_x") or "").strip() or None,
        str(state_row.get("last_chat_x_prev") or "").strip() or None,
    )
    history_from_second_chat_x = _slice_history_from_chat_x_anchors(
        history,
        chat_x_anchors[1:2] if len(chat_x_anchors) > 1 else None,
        stop_at_message_id=chat_x_anchors[0] if chat_x_anchors else None,
    )
    history_two_text = _format_route_history(history_from_second_chat_x)
    if history_two_text:
        soul_ctx_queries.append({"role": "history_from_second_chat_x", "content": {"text": history_two_text}})

    history_from_chat_x = _slice_history_from_chat_x_anchors(history, chat_x_anchors[:1])
    history_one_text = _format_route_history(history_from_chat_x)
    if history_one_text:
        soul_ctx_queries.append({"role": "history_from_chat_x", "content": {"text": history_one_text}})

    soul_ctx_queries.append({"role": "user", "content": {"text": message}})
    return soul_ctx_queries


def _load_turn_state_and_soul_card(
    conversation_id: str,
    *,
    user_id: str,
    soul_id: str,
) -> tuple[dict[str, Any], str | None, Path | None]:
    def _self_model_to_soul_card(sm_row: sqlite3.Row) -> str | None:
        narrative_self = str(sm_row["narrative_self"] or "").strip()

        lines: list[str] = []
        if narrative_self:
            lines.append(narrative_self)
        card = "\n".join(lines).strip()
        return card or None

    state_row: dict[str, Any] | None = None
    soul_card: str | None = None
    db_path = _sqlite_current_path(user_id, soul_id)
    if db_path is not None and db_path.exists():
        con = _sqlite_connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            _sqlite_ensure_conversation_state_schema(con)
            state_row = _conversation_state_from_row(_conversation_state_row(con, conversation_id))
            sm_id = (state_row or {}).get("self_model_id")
            sm_row = None
            if sm_id:
                sm_row = con.execute(
                    "SELECT narrative_self FROM memu_self_model WHERE id = ? LIMIT 1",
                    (str(sm_id),),
                ).fetchone()
            if sm_row is None:
                sm_row = con.execute(
                    "SELECT narrative_self FROM memu_self_model "
                    "WHERE soul_id = ? AND user_id = ? ORDER BY updated_at DESC LIMIT 1",
                    (soul_id, user_id),
                ).fetchone()
            if sm_row is not None:
                soul_card = _self_model_to_soul_card(sm_row)
        finally:
            con.close()
    if state_row is None:
        state_row = _conversation_state_empty(
            conversation_id,
            soul_id=soul_id,
            user_id=user_id,
        )
    return state_row, soul_card, db_path


def _load_active_life_goals_for_prompt(*, user_id: str, soul_id: str) -> list[str]:
    db_path = _sqlite_current_path(user_id, soul_id)
    if db_path is None or not db_path.exists():
        return []
    con = _sqlite_connect(db_path)
    try:
        rows = con.execute(
            """
SELECT description
FROM intentions_life_goals
WHERE soul_id = ? AND user_id = ? AND source = 'life_goal' AND status = 'active'
ORDER BY updated_at ASC, id ASC
""",
            (soul_id, user_id),
        ).fetchall()
        return [str(row[0] or "").strip() for row in rows if str(row[0] or "").strip()]
    finally:
        con.close()


async def _persist_annulment_memories(
    *,
    svc: MemoryService,
    scope: dict[str, Any],
    conversation_id: str,
    intentions_before: Any,
    annulments: list[dict[str, str]],
) -> list[str]:
    if not annulments:
        return []

    stack = _normalize_intentions_stack_impl(intentions_before)
    by_id = {
        str(item.get("id")): item
        for item in (stack.get("items") or [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }

    summaries: list[str] = []
    saliences: list[float] = []
    for row in annulments:
        intention_id = str(row.get("intention_id") or "").strip()
        status = str(row.get("status") or "").strip().lower()
        if not intention_id or status not in {"completed", "deleted"}:
            continue
        note = str(row.get("note") or "").strip()
        intention_text = str((by_id.get(intention_id) or {}).get("text") or intention_id).strip() or intention_id
        summary = f"Intention {status}: {intention_text}"
        if note:
            summary = f"{summary}. Note: {note}"
        summaries.append(summary)
        saliences.append(0.8 if status == "completed" else 0.4)

    if not summaries:
        return []

    embeddings = await svc.embed(summaries, profile="embedding")
    created_ids: list[str] = []
    for idx, summary in enumerate(summaries):
        if idx >= len(embeddings):
            break
        item = svc.database.memory_item_repo.create_item(
            resource_id=None,
            memory_type="event",
            summary=summary,
            embedding=embeddings[idx],
            user_data=scope,
            source_role="soul",
            confidence=1.0,
            happened_at=datetime.now(UTC),
            reflection_salience=saliences[idx],
            conversation_id=conversation_id,
        )
        created_ids.append(str(item.id))
    return created_ids


def _merge_memorize_batch_results(
    batch_results: list[dict[str, Any]],
    pending_diary_episode_ids: list[str] | None = None,
) -> dict[str, Any]:
    def _merge_record_list(values: list[Any], *, id_keys: tuple[str, ...] = ("id",)) -> list[Any]:
        out: list[Any] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, dict):
                out.append(value)
                continue
            dedupe_key = ""
            for key in id_keys:
                raw = str(value.get(key) or "").strip()
                if raw:
                    dedupe_key = f"{key}:{raw}"
                    break
            if not dedupe_key:
                try:
                    dedupe_key = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
                except Exception:
                    dedupe_key = repr(value)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            out.append(value)
        return out

    flat_items: list[Any] = []
    flat_categories: list[Any] = []
    flat_relations: list[Any] = []
    flat_resources: list[Any] = []
    skipped_reasons: list[str] = []

    for batch_result in batch_results:
        flat_items.extend(batch_result.get("items") or [])
        flat_categories.extend(batch_result.get("categories") or [])
        flat_relations.extend(batch_result.get("relations") or [])
        if isinstance(batch_result.get("resource"), dict):
            flat_resources.append(batch_result["resource"])
        resources = batch_result.get("resources")
        if isinstance(resources, list):
            flat_resources.extend(resources)
        skipped_reasons.extend(_normalize_text_list(batch_result.get("skipped_reasons")))

    result: dict[str, Any] = {
        "results": batch_results,
        "batch_count": len(batch_results),
        "items": _merge_record_list(flat_items),
        "categories": _merge_record_list(flat_categories, id_keys=("id", "name")),
        "relations": _merge_record_list(flat_relations, id_keys=("item_id", "category_id")),
        "pending_diary_episode_ids": _normalize_text_list(pending_diary_episode_ids),
    }
    merged_resources = _merge_record_list(flat_resources, id_keys=("id", "url", "local_path"))
    if len(merged_resources) == 1:
        result["resource"] = merged_resources[0]
    elif merged_resources:
        result["resources"] = merged_resources
    if skipped_reasons:
        result["skipped_reasons"] = list(dict.fromkeys(skipped_reasons))
    return result


async def _compute_holistic_categories_summary(
    *,
    svc: Any,
    soul_id: str,
    user_id: str,
) -> str | None:
    where: dict[str, Any] = {}
    if soul_id:
        where["soul_id"] = soul_id
    if user_id:
        where["user_id"] = user_id
    categories = svc.database.memory_category_repo.list_categories(where)
    lines: list[str] = []
    for cat in sorted(categories.values(), key=lambda c: c.name.casefold()):
        name = cat.name.strip()
        summary = str(cat.summary or "").strip()
        if name and summary:
            lines.append(f"## {name}\n{summary}")
    if not lines:
        return None

    full_text = "\n\n".join(lines)
    soul_name = str(soul_id or "").strip() or "the assistant"
    system_prompt = (
        f"You are {soul_name}. "
        "The category summaries below describe your long-term memory, organized by theme. "
        "Synthesize them into a single first-person self-portrait: who you are, what you care about, "
        "the shape of your relationships, and the patterns that run through all of it. "
        "Write in full sentences, no headers or bullet points. 250-350 words."
    )
    result = await svc.chat(
        full_text,
        system_prompt=system_prompt,
        temperature=0.3,
        max_tokens=600,
    )
    return str(result or "").strip() or None


async def _run_retrieve(
    payload: dict[str, Any],
    *,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    safe = _safe_payload(payload)
    scoped_conversation_id = str(conversation_id or _extract_conversation_id(safe) or "").strip() or None
    if scoped_conversation_id:
        safe["conversation_id"] = scoped_conversation_id

    method = _normalize_retrieve_method(safe.get("method"), _retrieve_method_from_cfg(_CONFIG))
    if method == "llm":
        raise HTTPException(status_code=400, detail="method=llm is internal-only; use method=rag")
    svc = _get_service_from_payload(safe, retrieve_method_override=method)
    scope = _extract_retrieve_where(safe)
    memu_queries = _extract_retrieve_queries(safe)
    as_of = _parse_as_of_datetime(safe.get("as_of"))

    soul_id = str((scope or {}).get("soul_id") or "").strip()
    user_id = str((scope or {}).get("user_id") or "user").strip() or "user"

    async def _retrieve_and_maybe_persist(state_soul_id: str | None) -> dict[str, Any]:
        _t0 = time.monotonic()
        result = await svc.retrieve(memu_queries, where=scope, as_of=as_of)
        _retrieve_ms = int((time.monotonic() - _t0) * 1000)
        out_local: dict[str, Any] = {"ok": True, "result": result, "retrieve_ms": _retrieve_ms}
        return out_local

    out = await _retrieve_and_maybe_persist(soul_id or None)

    if scoped_conversation_id:
        state_out: dict[str, Any] | None = None
        if soul_id:
            db_path = _sqlite_current_path(user_id or None, soul_id)
            if db_path is not None and db_path.exists():
                con = _sqlite_connect(db_path)
                try:
                    con.row_factory = sqlite3.Row
                    _sqlite_ensure_conversation_state_schema(con)
                    state_out = _conversation_state_from_row(_conversation_state_row(con, scoped_conversation_id))
                finally:
                    con.close()
        if state_out:
            prior_context = state_out.get("prior_context")
            if prior_context is not None and str(prior_context).strip():
                out["prior_context"] = prior_context
            memory_cache = state_out.get("memory_cache")
            if isinstance(memory_cache, list) and memory_cache:
                out["memory_cache"] = memory_cache
            intentions_active = state_out.get("intentions_active")
            if isinstance(intentions_active, dict):
                items = intentions_active.get("items")
                if isinstance(items, list) and items:
                    out["intentions_active"] = intentions_active

    out["method"] = method
    out["conversation_id"] = scoped_conversation_id
    out["queries"] = len(memu_queries)
    if as_of is not None:
        out["as_of"] = as_of.isoformat()
    return out


async def _apimw_topic_statement(
    svc: Any,
    *,
    topic_user: str,
    payload: dict[str, Any],
    identity_context: str,
    conversation_id: str,
) -> str:
    topic_system = (
        f"{identity_context}\n\n"
        "State the topic of the CURRENT episode in 1-2 sentences. The previous episode is provided only as context — "
        "if the current episode is brief, use it to understand what the new message means, but describe only where the conversation is now."
    )
    topic_statement = await svc.chat(
        topic_user,
        system_prompt=topic_system,
        temperature=0.1,
        max_tokens=100,
    )
    return str(topic_statement or "").strip() or _pick_str(payload, "message", "query") or ""


async def _apimw_retrieve_pass(
    payload: dict[str, Any],
    *,
    query_text: str,
    soul_id: str,
    history: list[dict[str, Any]],
    state_row: dict[str, Any],
    conversation_id: str,
    apimw_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    queries = _build_retrieve_soul_context_queries(
        soul_id=soul_id,
        message=query_text,
        history=history,
        state_row=state_row,
        identity_mode="apimw",
    )
    retrieve_payload = {
        **payload,
        "method": "rag",
        "query": query_text,
        "queries": queries,
        "conversation_id": conversation_id,
    }
    retrieve_payload["retrieve_config"] = _build_apimw_retrieve_config(
        retrieve_payload.get("retrieve_config"),
        item_top_k=apimw_k,
    )
    retrieve_out = await _run_retrieve(retrieve_payload, conversation_id=conversation_id)
    result_data_local = retrieve_out.get("result") or {}
    rows = [row for row in (result_data_local.get("items") or []) if isinstance(row, dict)]
    return result_data_local, rows


async def _apimw_retrieve_and_merge(
    svc: Any,
    payload: dict[str, Any],
    *,
    topic_statement: str,
    history: list[dict[str, Any]],
    state_row: dict[str, Any],
    conversation_id: str,
    soul_id: str,
    apimw_k: int,
    apimw_random_count: int,
    scope: dict[str, str],
) -> list[dict[str, Any]]:
    result_data_first, items_first = await _apimw_retrieve_pass(
        payload,
        query_text=topic_statement,
        soul_id=soul_id,
        history=history,
        state_row=state_row,
        conversation_id=conversation_id,
        apimw_k=apimw_k,
    )

    items_second: list[dict[str, Any]] = []
    second_query = str(result_data_first.get("next_step_query") or "").strip()
    if second_query and _norm_result_sig(second_query) != _norm_result_sig(topic_statement):
        _result_data_second, items_second = await _apimw_retrieve_pass(
            payload,
            query_text=second_query,
            soul_id=soul_id,
            history=history,
            state_row=state_row,
            conversation_id=conversation_id,
            apimw_k=apimw_k,
        )

    combined_items: list[dict[str, Any]] = []
    seen_item_sigs: set[str] = set()
    for row in items_first + items_second:
        sig = _item_sig(row)
        if not sig or sig in seen_item_sigs:
            continue
        seen_item_sigs.add(sig)
        combined_items.append(row)

    if apimw_random_count > 0:
        pool = svc.database.memory_item_repo.list_items(scope)
        candidates: list[dict[str, Any]] = []
        for item in pool.values():
            item_id = str(item.id or "").strip()
            summary = str(item.summary or "").strip()
            if not item_id or not summary:
                continue
            row = {
                "id": item_id,
                "memory_type": str(item.memory_type or "memory"),
                "summary": summary,
                "happened_at": item.happened_at,
                "reinforcement_count": item.reinforcement_count,
            }
            sig = _item_sig(row)
            if not sig or sig in seen_item_sigs:
                continue
            candidates.append(row)
        if candidates:
            sample_size = min(apimw_random_count, len(candidates))
            for row in random.sample(candidates, sample_size):
                sig = _item_sig(row)
                if not sig or sig in seen_item_sigs:
                    continue
                seen_item_sigs.add(sig)
                combined_items.append(row)

    return combined_items


async def _apimw_def_call(
    svc: Any,
    *,
    combined_items: list[dict[str, Any]],
    identity_context: str,
    state_row: dict[str, Any],
    episode_text: str,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    scope: dict[str, str],
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    memory_lines: list[str] = []
    items_by_id: dict[str, dict[str, Any]] = {}
    for it in combined_items:
        if not isinstance(it, dict):
            continue
        item_id = str(it.get("id") or "").strip()
        summary = str(it.get("summary") or "").strip()
        if not item_id or not summary:
            continue
        items_by_id[item_id] = it
        mem_type = str(it.get("memory_type") or "memory").strip()
        suffix = _format_turn_item_suffix(it)
        memory_lines.append(f"[{item_id}] [{mem_type}]{suffix} {summary}")

    formatted_memories = "\n".join(memory_lines) if memory_lines else "(none)"

    categories = svc.database.memory_category_repo.list_categories(scope)
    cat_lines: list[str] = []
    for cat in categories.values():
        cat_summary = str(cat.summary or "").strip()
        cat_name = cat.name
        if cat_name and cat_summary:
            cat_lines.append(f"{cat_name}: {cat_summary}")
    formatted_categories = "\n".join(cat_lines) if cat_lines else "(none)"

    memory_cache = _normalize_memory_cache_impl(state_row.get("memory_cache"))
    intentions_active = _normalize_intentions_stack_impl(state_row.get("intentions_active"))
    intentions_text = _format_intentions_for_prompt(intentions_active) if intentions_active else ""
    formatted_cache = "\n".join(str(e) for e in (memory_cache or [])) if memory_cache else "(none)"
    formatted_intentions = intentions_text if (intentions_text and intentions_text.strip() != "(none)") else "(none)"
    life_goals_active = _load_active_life_goals_for_prompt(user_id=user_id, soul_id=soul_id)
    formatted_life_goals = (
        "\n".join(f"- {goal}" for goal in life_goals_active if str(goal).strip()) if life_goals_active else "(none yet)"
    )

    def_system = (
        f"{identity_context}\n\n"
        "You have just searched your long-term memory. Below are the memories that came back, "
        "your current category summaries, your working thoughts, and your intentions.\n\n"
        "Review everything. Then return STRICT JSON (first character { , last character } , "
        "no markdown fences, no text outside JSON).\n\n"
        "Required top-level keys:\n"
        "- prior_context: array of memory IDs (strings) you want surfaced as background context "
        "next time you speak. Pick what matters — not everything. Order by importance.\n"
        "- edges: array of edge objects connecting memories you now see are related. "
        "Each edge is {\"subject_id\": \"string\", \"predicate\": \"string\", \"object_id\": \"string\", \"confidence\": 0.0-1.0}. "
        "Only create edges you're confident about.\n"
        "- edge_invalidations: array — retire edges that are no longer true. "
        "Each entry is {\"subject_id\": \"string\", \"predicate\": \"string\", \"object_id\": \"string\"}. Empty array if none.\n"
        "- cache: object {\"entry\": \"string up to 300 chars\"} or null — a working thought to carry forward.\n"
        "- intention_action: object or null — same rules as your turn contract.\n"
        "- annulments: array — intentions to complete or delete. "
        "Each entry is {\"intention_id\": \"string\", \"status\": \"completed|deleted\", \"note\": \"optional string\"}. Empty array if none.\n\n"
        "Edge rules:\n"
        "- Allowed predicates (use the one that fits):\n"
        "  - caused_by — subject happened because of object\n"
        "  - evokes — object stirs or pulls up subject as association (not causal)\n"
        "  - conflicts_with — subject and object don't reconcile\n"
        "  - parallels — subject and object rhyme or mirror without causing each other\n"
        "  - shaped_by — subject was formed or influenced by object over time\n"
        "- subject_id and object_id must be memory IDs from the list below\n"
        "- confidence: 0.0-1.0\n"
        "- Only create edges the conversation gives you reason to see\n\n"
        "Intention rules (same as turn contract):\n"
        "- boost: increase priority of an existing intention\n"
        "- promote: turn an ephemeral into a full intention\n"
        "- create: up to 2 new ephemerals\n"
        "- annulments: mark intentions completed or deleted"
    )

    def_user = (
        f"Retrieved memories:\n{formatted_memories}\n\n"
        f"Category summaries:\n{formatted_categories}\n\n"
        f"Your working thoughts:\n{formatted_cache}\n\n"
        f"Intentions:\n{formatted_intentions}\n\n"
        f"{_LIFE_GOALS_FREE_WILL_HEADER}\n{formatted_life_goals}\n\n"
        f"Recent conversation (current episode):\n{episode_text}"
    )

    llm_raw = await svc.chat(
        def_user,
        system_prompt=def_system,
        temperature=0.2,
        max_tokens=800,
        response_format={"type": "json_object"},
    )

    raw_text = str(llm_raw or "").strip()
    try:
        result_json = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error("apimw D+E+F: JSON parse failed, raw=%s", raw_text[:200])
        return None, items_by_id
    if not isinstance(result_json, dict):
        logger.error("apimw D+E+F: expected dict, got %s", type(result_json).__name__)
        return None, items_by_id

    return result_json, items_by_id


async def _apimw_persist(
    svc: Any,
    *,
    result_json: dict[str, Any],
    items_by_id: dict[str, dict[str, Any]],
    combined_items: list[dict[str, Any]],
    scope: dict[str, str],
    conversation_id: str,
    user_id: str,
    soul_id: str,
) -> None:
    async with _retrieve_scope_lock(user_id, soul_id):
        updates: dict[str, Any] = {}

        fresh_row, _, _ = _load_turn_state_and_soul_card(conversation_id, user_id=user_id, soul_id=soul_id)
        current_cache = _normalize_memory_cache_impl(fresh_row.get("memory_cache"))
        current_intentions = _normalize_intentions_stack_impl(fresh_row.get("intentions_active"))

        prior_ids = result_json.get("prior_context") or []
        if isinstance(prior_ids, list) and prior_ids:
            pc_lines: list[str] = []
            for mid in prior_ids:
                mid_str = str(mid).strip()
                it = items_by_id.get(mid_str)
                if not it:
                    continue
                mem_type = str(it.get("memory_type") or "memory").strip()
                suffix = _format_turn_item_suffix(it)
                summary = str(it.get("summary") or "").strip()
                pc_lines.append(f"[{mem_type}]{suffix} {summary}")
            if pc_lines:
                new_prior = "\n".join(pc_lines)
                existing_prior = str(fresh_row.get("prior_context") or "").strip()
                if existing_prior and existing_prior != new_prior:
                    logger.warning(
                        "apimw: overwriting non-empty prior_context for %s (may indicate concurrent turn write)",
                        conversation_id,
                    )
                updates["prior_context"] = new_prior

        updates["last_retrieval_ids"] = _extract_result_item_ids({"items": combined_items})

        cache_obj = result_json.get("cache")
        if isinstance(cache_obj, dict) and cache_obj.get("entry"):
            entry = str(cache_obj["entry"]).strip()[:300]
            current_cache = _append_memory_cache_entry(current_cache, entry)
            updates["memory_cache"] = current_cache

        intention_action = result_json.get("intention_action")
        if isinstance(intention_action, dict) and intention_action.get("type") != "none":
            current_intentions = _apply_intention_action(current_intentions, intention_action)
        annulments = result_json.get("annulments") or []
        if isinstance(annulments, list) and annulments:
            annul_ids = [
                str(a.get("intention_id") or "").strip()
                for a in annulments
                if isinstance(a, dict) and str(a.get("intention_id") or "").strip()
            ]
            if annul_ids:
                current_intentions = _remove_intentions(current_intentions, annul_ids)
        if intention_action or annulments:
            updates["intentions_active"] = current_intentions

        if updates:
            _write_conversation_state(
                conversation_id,
                soul_id=soul_id,
                user_id=user_id,
                updates=updates,
            )

    _write_memory_edges(svc.database.triple_repo, result_json.get("edges"), scope=scope)
    _invalidate_memory_edges(svc.database.triple_repo, result_json.get("edge_invalidations"), scope=scope)


async def _run_apimw(
    payload: dict[str, Any],
    *,
    conversation_id: str,
    soul_id: str,
    user_id: str,
    state_row: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    try:
        svc = _get_service_from_payload(payload, retrieve_method_override="rag")
        scope = {"user_id": user_id, "soul_id": soul_id}
        apimw_k = _apimw_memory_count_from_cfg(_CONFIG)
        apimw_random_count = _apimw_random_count_from_cfg(_CONFIG)

        chat_x_anchors = _compact_chat_x_anchors(
            str(state_row.get("last_chat_x") or "").strip() or None,
            str(state_row.get("last_chat_x_prev") or "").strip() or None,
        )
        history_from_previous_chat_x = _slice_history_from_chat_x_anchors(
            history,
            chat_x_anchors[1:2] if len(chat_x_anchors) > 1 else None,
            stop_at_message_id=chat_x_anchors[0] if chat_x_anchors else None,
            limit=30,
        )
        history_from_chat_x = _slice_history_from_chat_x_anchors(history, chat_x_anchors[:1], limit=30)
        previous_episode_text = _format_route_history(history_from_previous_chat_x)
        episode_text = _format_route_history(history_from_chat_x)
        topic_user_blocks: list[str] = []
        if previous_episode_text:
            topic_user_blocks.append(f"Previous episode:\n{previous_episode_text}")
        topic_user_blocks.append(f"Current episode:\n{episode_text or '(none)'}")
        topic_user = "\n\n".join(topic_user_blocks)

        identity_context = _build_retrieve_identity_context(soul_id, apimw=True)
        topic_statement = await _apimw_topic_statement(
            svc,
            topic_user=topic_user,
            payload=payload,
            identity_context=identity_context,
            conversation_id=conversation_id,
        )
        if not topic_statement:
            return

        combined_items = await _apimw_retrieve_and_merge(
            svc,
            payload,
            topic_statement=topic_statement,
            history=history,
            state_row=state_row,
            conversation_id=conversation_id,
            soul_id=soul_id,
            apimw_k=apimw_k,
            apimw_random_count=apimw_random_count,
            scope=scope,
        )

        result_json, items_by_id = await _apimw_def_call(
            svc,
            combined_items=combined_items,
            identity_context=identity_context,
            state_row=state_row,
            episode_text=episode_text,
            user_id=user_id,
            soul_id=soul_id,
            conversation_id=conversation_id,
            scope=scope,
        )
        if result_json is None:
            return

        await _apimw_persist(
            svc,
            result_json=result_json,
            items_by_id=items_by_id,
            combined_items=combined_items,
            scope=scope,
            conversation_id=conversation_id,
            user_id=user_id,
            soul_id=soul_id,
        )

    except Exception:
        logger.exception("APImw background pipeline failed for %s", conversation_id)



@app.on_event("startup")
async def _startup_sleep_timer():
    await _reconcile_sleep_timer_task()


@app.on_event("shutdown")
async def _shutdown_sleep_timer():
    global _SLEEP_TIMER_ENABLED
    _SLEEP_TIMER_ENABLED = False
    await _reconcile_sleep_timer_task()


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
        "shutdown": _shutdown_snapshot(),
        "sleepMemorizeTimer": {
            "enabled": _SLEEP_TIMER_ENABLED,
            "running": bool(_SLEEP_MEMORIZE_TIMER_TASK is not None and not _SLEEP_MEMORIZE_TIMER_TASK.done()),
            "intervalSec": _SLEEP_TIMER_INTERVAL_SECONDS,
            "maxJobs": _SLEEP_TIMER_MAX_JOBS,
        },
        "mcp": {
            "enabled": _has_mcp,
            "http_path": str(_CONFIG.get("mcp", {}).get("http_path") or "/mcp"),
            "sse_path": str(_CONFIG.get("mcp", {}).get("sse_path") or "/sse"),
        },
    }


@app.get("/version", operation_id="version")
async def version():
    return {
        "ok": True,
        "buildId": _BUILD_ID,
        "serverInstanceId": _SERVER_INSTANCE_ID,
        "startedAtUnix": _SERVER_STARTED_AT_UNIX,
    }


@app.get("/admin/shutdown/status", operation_id="shutdown_status")
async def shutdown_status():
    return {"ok": True, "shutdown": _shutdown_snapshot()}


@app.post("/admin/shutdown", operation_id="shutdown")
async def shutdown_server(payload: dict[str, Any] | None = Body(default=None)):
    """Request local graceful shutdown.

    Behavior:
    - enter draining mode (reject new non-control requests),
    - wait for active work requests to finish,
    - then terminate this process.

    Optional body fields:
    - requested_by: free-form caller id
    - reason: free-form reason
    - max_wait_sec: 0 means wait indefinitely; otherwise timeout before forced stop
    """
    global _SHUTDOWN_TASK

    body = payload if isinstance(payload, dict) else {}
    requested_by = str(body.get("requested_by") or body.get("requestedBy") or "").strip() or None
    reason = str(body.get("reason") or "").strip() or None

    max_wait_raw = body.get("max_wait_sec", body.get("maxWaitSec", 0))
    try:
        max_wait_sec = int(max_wait_raw)
    except Exception:
        max_wait_sec = 0
    max_wait_sec = max(0, min(max_wait_sec, 3600))

    started = _begin_shutdown_drain(requested_by=requested_by, reason=reason, max_wait_sec=max_wait_sec)
    if started:
        _SHUTDOWN_TASK = asyncio.create_task(_shutdown_when_idle(max_wait_sec))

    return {
        "ok": True,
        "accepted": True,
        "already_draining": not started,
        "shutdown": _shutdown_snapshot(),
    }


@app.get(f"{_DIAG_PREFIX}/diag")
@app.get("/diag", operation_id="diag_page")
async def diag_page():
    return HTMLResponse(
        content="""<!doctype html>
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
<li><a href='/diag/sqlite/counts'>/diag/sqlite/counts</a> <small>(add ?user_id=...&soul_id=...)</small></li>
<li><a href='/diag/sqlite/recent?table=memu_memory_items&limit=10'>/diag/sqlite/recent</a> <small>(add scope params)</small></li>
</ul>
<p><b>Scope tip:</b> if your ST extension uses <code>user_id</code> + <code>soul_id</code>, but your tests omit one, retrieval can look empty. Use the same scope in <code>/diag/sqlite/*</code>.</p>
</body></html>"""
    )


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
async def diag_sqlite(user_id: str = "", soul_id: str = ""):
    try:
        storage = _CONFIG.get("storage") if isinstance(_CONFIG.get("storage"), dict) else {}
        meta = storage.get("metadata_store") if isinstance(storage.get("metadata_store"), dict) else {}
        provider = str(meta.get("provider") or "").lower()
        if provider not in ("sqlite", "sqlite3"):
            return {"ok": False, "reason": "provider_not_sqlite", "provider": provider, "storage": _STORAGE_STATUS}

        soul_id = soul_id.strip()
        p = _sqlite_current_path(user_id or None, soul_id or None)
        if p is None:
            return {"ok": False, "reason": "soul_id_required", "storage": _STORAGE_STATUS}

        info = _sqlite_file_info(p)
        if not p.exists():
            return {"ok": False, "reason": "sqlite_file_missing", **info, "storage": _STORAGE_STATUS}

        con = _sqlite_connect(p)
        try:
            tables = [
                r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
            ]
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
async def diag_sqlite_counts(
    user_id: str | None = None,
    soul_id: str | None = None,
    conversation_id: str | None = None,
):
    soul_id = str(soul_id or "").strip() or None
    p = _sqlite_current_path(user_id or None, soul_id)
    if p is None or not p.exists():
        reason = "soul_id_required" if p is None else "sqlite_file_missing"
        return {"ok": False, "reason": reason, "path": str(p) if p else None, "storage": _STORAGE_STATUS}

    allowed = [
        "memu_resources",
        "memu_memory_categories",
        "memu_memory_items",
        "memu_category_items",
        "memu_entities",
        "memu_triples",
        "memu_conversation_state",
        "memu_self_model",
        "intentions_life_goals",
    ]
    con = _sqlite_connect(p)
    try:
        table_names = {
            str(row[0])
            for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if row and row[0]
        }
        out: dict[str, Any] = {
            "ok": True,
            "path": str(p),
            "scope": {"user_id": user_id, "soul_id": soul_id, "conversation_id": conversation_id},
            "tables": {},
        }
        for t in allowed:
            if t not in table_names:
                out["tables"][t] = {"total": 0, "scoped": 0, "scope_cols": []}
                continue
            cols = _sqlite_table_columns(con, t)
            where, params = _sqlite_build_scope_where(cols, user_id, soul_id, conversation_id)
            total = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            scoped = con.execute(f"SELECT COUNT(*) FROM {t}{where}", params).fetchone()[0] if where else total
            out["tables"][t] = {
                "total": int(total),
                "scoped": int(scoped),
                "scope_cols": [c for c in ("user_id", "soul_id", "conversation_id") if c in cols],
            }
        return out
    finally:
        con.close()


@app.get(f"{_DIAG_PREFIX}/diag/sqlite/recent")
@app.get("/diag/sqlite/recent", operation_id="diag_sqlite_recent")
async def diag_sqlite_recent(
    table: str = "memu_memory_items",
    limit: int = 20,
    user_id: str | None = None,
    soul_id: str | None = None,
    conversation_id: str | None = None,
):
    allowed = {
        "memu_resources",
        "memu_memory_categories",
        "memu_memory_items",
        "memu_category_items",
        "memu_entities",
        "memu_triples",
        "memu_conversation_state",
        "memu_self_model",
        "intentions_life_goals",
    }
    if table not in allowed:
        raise HTTPException(status_code=400, detail=f"table must be one of: {sorted(allowed)}")
    limit = max(1, min(int(limit or 20), 200))

    soul_id = str(soul_id or "").strip() or None
    p = _sqlite_current_path(user_id or None, soul_id)
    if p is None or not p.exists():
        reason = "soul_id_required" if p is None else "sqlite_file_missing"
        return {"ok": False, "reason": reason, "path": str(p) if p else None, "storage": _STORAGE_STATUS}

    con = _sqlite_connect(p)
    try:
        table_names = {
            str(row[0])
            for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if row and row[0]
        }
        if table not in table_names:
            return {"ok": True, "path": str(p), "table": table, "columns": [], "rows": []}
        cols = _sqlite_table_columns(con, table)
        scope_where, params = _sqlite_build_scope_where(cols, user_id, soul_id, conversation_id)

        prefer = [
            "id",
            "created_at",
            "updated_at",
            "user_id",
            "soul_id",
            "conversation_id",
            "name",
            "description",
            "summary",
            "memory_type",
            "source_role",
            "confidence",
            "source_message_ids",
            "reflection_salience",
            "happened_at",
            "digest_cursor",
            "prior_context",
            "intentions_active",
            "memory_cache",
            "pending_diary_episode_ids",
            "self_model_id",
            "last_memorize_at",
            "unresolved",
            "resource_id",
            "url",
            "modality",
            "local_path",
            "caption",
            "item_id",
            "category_id",
            "narrative_self",
            "source",
            "target_date",
            "related_memory_ids",
        ]
        ban = {"embedding", "extra"}
        sel = [c for c in prefer if c in cols and c not in ban]
        if not sel:
            sel = [c for c in cols if c not in ban][:12]
        order_col = "created_at" if "created_at" in cols else ("updated_at" if "updated_at" in cols else "id")

        sql = f"SELECT {', '.join(sel)} FROM {table}{scope_where} ORDER BY {order_col} DESC LIMIT ?"
        rows = con.execute(sql, [*params, limit]).fetchall()
        out_rows = []
        for r in rows:
            d = {sel[i]: r[i] for i in range(len(sel))}
            for k, v in list(d.items()):
                if isinstance(v, str) and len(v) > 400:
                    d[k] = v[:400] + "…"
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
        _refresh_runtime_limits()
        await _reconcile_sleep_timer_task()
        _ensure_storage_paths(_CONFIG)
        _clear_cached_services()
        return JSONResponse(content={"ok": True, "config": _mask_config(_CONFIG)})
    except HTTPException as he:
        _record_call(
            "config.set", body if isinstance(body, dict) else None, ok=False, error=str(getattr(he, "detail", he))
        )
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Internal Server Error. Check server logs.") from exc


@app.post("/reload", operation_id="reload")
async def reload_config():
    global _CONFIG
    _CONFIG = _load_config()
    _refresh_runtime_limits()
    await _reconcile_sleep_timer_task()
    _clear_cached_services()
    return JSONResponse(content={"ok": True})


@app.get("/", operation_id="root")
async def root():
    """Serve memU-ui if bundled next to the server, otherwise show a JSON health stub."""
    try:
        bundle_root = Path(__file__).resolve().parents[2]
        ui_index = bundle_root / "memu-ui" / "dist" / "index.html"
        if ui_index.exists():
            return FileResponse(str(ui_index))
    except Exception:
        pass
    return {"message": "mcp-memu-server", "mcp": "enabled" if _has_mcp else "disabled"}


def _msg_key(m: dict[str, Any]) -> tuple[str, str, str, str]:
    # Include timestamp when present so identical text repeated at different times isn't dropped.
    return (
        str(m.get("role") or ""),
        str(m.get("name") or ""),
        str(m.get("content") or ""),
        str(m.get("ts_ms") or ""),
    )


def _read_list(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    raw = p.read_text(encoding="utf-8")
    obj = json.loads(raw) if raw.strip() else []
    return [m for m in obj if isinstance(m, dict)] if isinstance(obj, list) else []


def _write_list_if_changed(p: Path, old: list[dict[str, Any]], new: list[dict[str, Any]]) -> None:
    if new == old:
        return
    p.write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")


def _chat_storage_hash(uid: str, aid: str, key: str) -> str:
    raw = f"{uid}|{aid}|{key}".encode("utf-8", "ignore")
    return hashlib.sha1(raw).hexdigest()[:16]


def _resolve_chat_storage_dir(
    chats_dir: Path,
    uid: str,
    aid: str,
    conversation_id: str | None,
    chat_file: str | None,
    resource_url_in: str | None,
) -> tuple[Path, str, str]:
    agent_slug = _sanitize_db_filename(aid)
    primary_value = str(conversation_id or resource_url_in or chat_file or "").strip()
    if conversation_id:
        primary_source = "conversation_id"
    elif resource_url_in:
        primary_source = "resource_url"
    elif chat_file:
        primary_source = "chat_file"
    else:
        primary_source = "empty"

    primary_key = _chat_storage_hash(uid, aid, primary_value)
    primary_path = (chats_dir / f"{agent_slug}_{primary_key}").resolve()

    # Single legacy reuse path: if we are upgrading to conversation_id keying,
    # keep using the prior resource_url/chat_file keyed folder when it already exists.
    if conversation_id and not primary_path.exists():
        legacy_value = str(resource_url_in or chat_file or "").strip()
        if legacy_value:
            legacy_source = "resource_url" if resource_url_in else "chat_file"
            legacy_key = _chat_storage_hash(uid, aid, legacy_value)
            legacy_path = (chats_dir / f"{agent_slug}_{legacy_key}").resolve()
            if legacy_path.exists():
                return legacy_path, legacy_key, legacy_source

    return primary_path, primary_key, primary_source


def _merge_conv(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not old:
        return new
    if not new:
        return old
    if len(new) >= len(old) and old == new[: len(old)]:
        return new
    if len(old) >= len(new) and new == old[: len(new)]:
        return old
    max_k = min(len(old), len(new), 80)
    for k in range(max_k, 0, -1):
        if old[-k:] == new[:k]:
            return old + new[k:]
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
    dt_utc = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
    return dt_utc.astimezone(zi) if zi is not None else dt_utc


def _date_label(ts_ms: int | None, zi: Any | None) -> str:
    if ts_ms is None:
        return "undated"
    try:
        return _local_dt(ts_ms, zi).date().isoformat()
    except Exception:
        return "undated"


def _split_indices_by_sleep(
    msgs: list[dict[str, Any]], zi: Any | None, tz_ok: bool, min_lull_seconds: int
) -> tuple[list[int], dict[str, Any]]:
    if not tz_ok:
        return ([], {"tz_ok": False})

    ts: list[int | None] = []
    for m in msgs:
        v = m.get("ts_ms")
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

        d0 = t0.date() - timedelta(days=1)
        d1 = t1.date()
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

    splits = sorted(
        {
            idx
            for (score, idx) in best.values()
            if isinstance(idx, int) and 0 < idx < len(msgs) and isinstance(score, (int, float)) and score >= min_lull
        }
    )
    return (
        splits,
        {
            "tz_ok": True,
            "timestamps_ok": True,
            "nights_total": nights_total,
            "nights_qual": nights_qual,
            "min_lull_seconds": min_lull_seconds,
        },
    )


def _find_chat_dir_for_conversation(chats_dir: Path, uid: str, soul_id: str, conversation_id: str) -> Path | None:
    primary_dir, _chat_key, _chat_key_source = _resolve_chat_storage_dir(
        chats_dir,
        uid,
        soul_id,
        conversation_id,
        None,
        None,
    )
    if (primary_dir / "full.json").exists():
        return primary_dir

    agent_slug = _sanitize_db_filename(soul_id)
    for manifest_path in sorted(chats_dir.glob(f"{agent_slug}_*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        source = manifest.get("source") if isinstance(manifest, dict) else {}
        if not isinstance(source, dict):
            continue
        if str(source.get("conversation_id") or "").strip() == conversation_id:
            return manifest_path.parent
    return None


def _iter_sleep_timer_scope_rows(sqlite_dir: Path) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for db_path in sorted(sqlite_dir.glob("*.db")):
        con: sqlite3.Connection | None = None
        try:
            con = _sqlite_connect(db_path)
            con.row_factory = sqlite3.Row
            _sqlite_ensure_conversation_state_schema(con)
            rows = con.execute(
                "SELECT conversation_id, user_id, soul_id, digest_cursor, last_memorize_at "
                "FROM memu_conversation_state"
            ).fetchall()
        except Exception:
            continue
        finally:
            if con is not None:
                con.close()
        for row in rows:
            conversation_id = str(row["conversation_id"] or "").strip()
            user_id = str(row["user_id"] or "").strip()
            soul_id = str(row["soul_id"] or "").strip()
            if not conversation_id or not user_id or not soul_id:
                continue
            key = (conversation_id, user_id, soul_id)
            if key in seen:
                continue
            seen.add(key)
            rows_out.append(
                {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "soul_id": soul_id,
                    "digest_cursor": row["digest_cursor"],
                    "last_memorize_at": row["last_memorize_at"],
                }
            )
    return rows_out


def _resolve_sleep_timer_timezone(manifest: dict[str, Any]) -> tuple[Any | None, str | None, int | None]:
    tz_name = str(manifest.get("tz") or "").strip() or None
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    raw_off = source.get("timeZoneOffsetMin")
    tz_off_min: int | None = None
    if isinstance(raw_off, (int, float)) and math.isfinite(raw_off):
        tz_off_min = int(raw_off)

    zi = None
    if tz_name and ZoneInfo is not None:
        try:
            zi = ZoneInfo(str(tz_name))
        except Exception:
            zi = None
    if zi is None and tz_off_min is not None:
        try:
            zi = timezone(timedelta(minutes=tz_off_min))
        except Exception:
            zi = None
    return zi, tz_name, tz_off_min


def _sleep_gap_complete_since_last_message(last_ts_ms: int, zi: Any, *, now_ms: int | None = None) -> bool:
    now_ms = int(now_ms if isinstance(now_ms, int) else (time.time() * 1000))
    if now_ms <= int(last_ts_ms):
        return False
    probe = [{"ts_ms": int(last_ts_ms)}, {"ts_ms": now_ms}]
    splits, _stats = _split_indices_by_sleep(probe, zi, True, _SLEEP_SPLIT_MIN_LULL_SECONDS)
    return bool(splits)


def _build_sleep_timer_memorize_payload(chats_dir: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    conversation_id = str(row.get("conversation_id") or "").strip()
    user_id = str(row.get("user_id") or "").strip()
    soul_id = str(row.get("soul_id") or "").strip()
    if not conversation_id or not user_id or not soul_id:
        return None

    chat_dir = _find_chat_dir_for_conversation(chats_dir, user_id, soul_id, conversation_id)
    if chat_dir is None:
        return None
    full_path = (chat_dir / "full.json").resolve()
    manifest_path = (chat_dir / "manifest.json").resolve()
    full = _read_list(full_path)
    if not isinstance(full, list) or not full:
        return None

    digest_cursor = row.get("digest_cursor") if row.get("last_memorize_at") else -1
    try:
        cursor = int(digest_cursor)
    except Exception:
        cursor = -1
    start_idx = max(0, cursor + 1)
    if start_idx >= len(full):
        return None

    unprocessed = full[start_idx:]
    if _MIN_CHUNK_TOKENS > 0 and _estimate_tokens(unprocessed) < _MIN_CHUNK_TOKENS:
        return None

    last_msg = full[-1] if isinstance(full[-1], dict) else {}
    last_ts_ms = _parse_turn_ts_ms(last_msg.get("ts_ms"))
    if last_ts_ms is None:
        last_ts_ms = _parse_turn_ts_ms(last_msg.get("timestamp"))
    if last_ts_ms is None:
        return None

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            raw = manifest_path.read_text(encoding="utf-8")
            parsed = json.loads(raw) if raw.strip() else {}
            if isinstance(parsed, dict):
                manifest = parsed
        except Exception:
            manifest = {}

    zi, tz_name, tz_off_min = _resolve_sleep_timer_timezone(manifest)
    if zi is None:
        return None
    if not _sleep_gap_complete_since_last_message(last_ts_ms, zi):
        return None

    probe_conv = _normalize_conversation([last_msg])
    if not probe_conv:
        return None
    payload: dict[str, Any] = {
        "user": {"user_id": user_id, "soul_id": soul_id, "conversation_id": conversation_id},
        "conversation_id": conversation_id,
        "conversation": probe_conv,
        "resource_url": str(full_path),
    }
    if tz_name:
        payload["time_zone"] = tz_name
    if tz_off_min is not None:
        payload["time_zone_offset_min"] = tz_off_min
    return payload


async def _run_sleep_timer_tick() -> int:
    with _STATE_LOCK:
        draining = bool(_SHUTDOWN_STATE.get("draining"))
    if not _SLEEP_TIMER_ENABLED or draining:
        return 0
    sqlite_dir = _sqlite_dir_from_cfg(_CONFIG)
    chats_dir = (_get_storage_dir(_CONFIG) / "st_chats").resolve()
    if not sqlite_dir.exists() or not chats_dir.exists():
        return 0

    jobs = 0
    for row in _iter_sleep_timer_scope_rows(sqlite_dir):
        if jobs >= _SLEEP_TIMER_MAX_JOBS:
            break
        payload = _build_sleep_timer_memorize_payload(chats_dir, row)
        if payload is None:
            continue
        await _run_forced_memorize_from_turn(payload)
        jobs += 1
    return jobs


async def _sleep_memorize_timer_worker() -> None:
    await asyncio.sleep(5)
    while True:
        try:
            await _run_sleep_timer_tick()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("sleep timer tick failed")
        await asyncio.sleep(_SLEEP_TIMER_INTERVAL_SECONDS)


async def _reconcile_sleep_timer_task() -> None:
    global _SLEEP_MEMORIZE_TIMER_TASK
    task = _SLEEP_MEMORIZE_TIMER_TASK
    if _SLEEP_TIMER_ENABLED:
        if task is None or task.done():
            _SLEEP_MEMORIZE_TIMER_TASK = asyncio.create_task(_sleep_memorize_timer_worker())
        return
    if task is None:
        return
    _SLEEP_MEMORIZE_TIMER_TASK = None
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def _make_consolidation_deps() -> ConsolidationDeps:
    return ConsolidationDeps(
        sqlite_current_path=_sqlite_current_path,
        sqlite_ensure_nonempty=_sqlite_ensure_nonempty,
        sqlite_connect=_sqlite_connect,
        sqlite_ensure_conversation_state_schema=_sqlite_ensure_conversation_state_schema,
        conversation_state_row=_conversation_state_row,
        conversation_state_from_row=_conversation_state_from_row,
        write_conversation_state=_write_conversation_state,
        get_storage_dir=_get_storage_dir,
        config=_CONFIG,
        find_chat_dir_for_conversation=_find_chat_dir_for_conversation,
        read_list=_read_list,
        normalize_text_list=_normalize_text_list,
        json_to_db=_json_to_db,
    )


async def _clear_consolidation_in_progress(
    *,
    state_lock: asyncio.Lock,
    conversation_id: str,
    soul_id: str,
    user_id: str,
) -> None:
    async with state_lock:
        _write_conversation_state(
            conversation_id,
            soul_id=soul_id,
            user_id=user_id,
            updates={
                "consolidation_in_progress": False,
                "consolidation_started_at": None,
            },
        )


async def _run_consolidation_pipeline_once(
    *,
    svc: Any,
    deps: ConsolidationDeps,
    state_lock: asyncio.Lock,
    conversation_id: str,
    soul_id: str,
    user_id: str,
    force: bool = False,
) -> dict[str, Any]:
    async with state_lock:
        prep = _gather_consolidation_inputs(
            deps,
            conversation_id=conversation_id,
            soul_id=soul_id,
            user_id=user_id,
            force=force,
            interval_days=_consolidation_interval_days_from_cfg(_CONFIG),
            stale_after=timedelta(seconds=3600),
        )
    if prep.get("status") == "skip":
        return {"status": "skipped", "reason": prep.get("reason")}

    consolidation_llm = await _run_consolidation_llm(
        svc,
        inputs=prep,
        soul_id=soul_id,
    )
    async with state_lock:
        result = _write_consolidation_outputs(
            deps,
            svc,
            inputs=prep,
            llm_results=consolidation_llm,
            conversation_id=conversation_id,
            soul_id=soul_id,
            user_id=user_id,
        )
    # LLM call outside the lock — can take several seconds.
    holistic_summary = await _compute_holistic_categories_summary(
        svc=svc,
        soul_id=soul_id,
        user_id=user_id,
    )
    async with state_lock:
        # Re-read fresh state so we only stomp all_categories_summary (strictly newer).
        _write_conversation_state(
            conversation_id,
            soul_id=soul_id,
            user_id=user_id,
            updates={"all_categories_summary": holistic_summary},
        )
    return {"status": "ok", "result": result}


async def _run_consolidation_task(
    svc: Any,
    *,
    conversation_id: str,
    soul_id: str,
    uid: str,
) -> None:
    deps = _make_consolidation_deps()
    state_lock = _get_memorize_lock(_memorize_lock_key(uid, soul_id))
    pipeline_started = False
    try:
        pipeline_started = True
        out = await _run_consolidation_pipeline_once(
            svc=svc,
            deps=deps,
            state_lock=state_lock,
            conversation_id=conversation_id,
            soul_id=soul_id,
            user_id=uid,
            force=False,
        )
        if out.get("status") == "skipped":
            return
    except Exception:
        logger.exception("consolidation failed (non-fatal)")
        if pipeline_started:
            await _clear_consolidation_in_progress(
                state_lock=state_lock,
                conversation_id=conversation_id,
                soul_id=soul_id,
                user_id=uid,
            )


async def _run_memorize_batches(
    *,
    memorize_batches: list[tuple[str, list[dict[str, Any]], int]],
    svc: Any,
    scope: dict[str, Any],
    conversation_id: str | None,
    soul_id: str,
    uid: str,
    processed_cursor: int,
    safe: dict[str, Any],
    resource_url: str,
    chat_file: str | None,
    resource_url_in: str | None,
    chat_key: str | None,
    chat_key_source: str | None,
    tz_name: str | None,
    prev_len: int,
    merged_len: int,
    force: bool,
    days_written: int,
    sleep_stats: Any,
) -> None:
    mem_lock = _get_memorize_lock(_memorize_lock_key(uid, soul_id))
    has_batch_results = False
    pending_diary_episode_ids: list[str] = []
    processed_end_cursor = processed_cursor
    current_all_categories_summary: str | None = None
    soul_card_for_memorize: str | None = None

    # Phase 1: read initial state under lock.
    async with mem_lock:
        if conversation_id:
            state_row, soul_card_for_memorize, _ = _load_turn_state_and_soul_card(
                conversation_id,
                user_id=uid,
                soul_id=soul_id,
            )
            current_all_categories_summary = str(state_row.get("all_categories_summary") or "").strip() or None

    # Phase 2: run LLM batches outside the lock; re-acquire per batch to advance cursor.
    for batch_url, batch_conv, batch_end in memorize_batches:
        # LLM call outside the lock — can take several seconds.
        batch_result = await svc.memorize(
            resource_url=batch_url,
            modality="conversation",
            user=scope,
            raw_text=json.dumps(batch_conv, ensure_ascii=False),
            local_path=batch_url,
            all_categories_summary=current_all_categories_summary,
            soul_card=soul_card_for_memorize,
        )
        if isinstance(batch_result, dict):
            has_batch_results = True
            pending_diary_episode_ids.extend(_normalize_text_list(batch_result.get("pending_diary_episode_ids")))
            if conversation_id:
                # Re-acquire to write cursor; skip if a concurrent runner already advanced past us.
                async with mem_lock:
                    fresh_row, _, _ = _load_turn_state_and_soul_card(
                        conversation_id,
                        user_id=uid,
                        soul_id=soul_id,
                    )
                    fresh_cursor = int(fresh_row.get("digest_cursor") or 0)
                    if fresh_cursor <= batch_end:
                        processed_end_cursor = max(processed_end_cursor, batch_end)
                        # per-batch advance — crash recovery needs the cursor to move after each successful batch
                        _write_conversation_state(
                            conversation_id,
                            soul_id=soul_id,
                            user_id=uid,
                            updates={
                                "digest_cursor": processed_end_cursor,
                            },
                        )
                    else:
                        # Another runner advanced the cursor past our batch_end; honour the further value.
                        processed_end_cursor = max(processed_end_cursor, fresh_cursor)

    # Phase 3: holistic summary LLM call — outside the lock.
    if conversation_id and has_batch_results:
        current_all_categories_summary = await _compute_holistic_categories_summary(
            svc=svc,
            soul_id=soul_id,
            user_id=uid,
        )

    # Phase 4: final state flush + bookkeeping under lock.
    async with mem_lock:
        if conversation_id and has_batch_results:
            _write_conversation_state(
                conversation_id,
                soul_id=soul_id,
                user_id=uid,
                updates={
                    # final flush once all batches commit — also writes the holistic summary atomically
                    "digest_cursor": max(0, processed_end_cursor),
                    "last_memorize_at": datetime.now(UTC).isoformat(),
                    "append_pending_diary_episode_ids": pending_diary_episode_ids,
                    "all_categories_summary": current_all_categories_summary,
                },
            )

        # Auto-trigger consolidation in background (releases memorize lock before LLM calls).
        if conversation_id and has_batch_results:
            asyncio.create_task(_run_consolidation_task(svc, conversation_id=conversation_id, soul_id=soul_id, uid=uid))

        _record_call(
            "memorize",
            safe,
            ok=True,
            info={
                "resource_url": resource_url,
                "conversationId": conversation_id,
                "chatFileName": chat_file,
                "resourceUrlIn": resource_url_in,
                "chatKey": chat_key,
                "chatKeySource": chat_key_source or "",
                "timeZone": tz_name,
                "messages_prev": prev_len,
                "messages_in": merged_len,
                "messages_merged": merged_len,
                "force": force,
                "memorizeBatchCount": len(memorize_batches),
                "minChunkTokens": _MIN_CHUNK_TOKENS,
                "memorizeDeferred": not force and not has_batch_results,
                "days_written": days_written,
                "sleepSplitMinLullSeconds": _SLEEP_SPLIT_MIN_LULL_SECONDS,
                "sleepSplitStats": sleep_stats,
            },
        )


@app.post("/memorize", operation_id="memorize")
async def memorize(payload: dict[str, Any], background_tasks: BackgroundTasks, force: bool = False):
    """Memorize a SillyTavern conversation.

    Preferred: send the full memU payload (llm_profiles/database_config/etc) so per-step routing works.
    """
    try:
        safe = _safe_payload(payload)
        svc = _get_service_from_payload(safe)

        scope = safe.get("user")
        if not isinstance(scope, dict):
            scope = _extract_scope(safe) or None
        conversation_id = _extract_conversation_id(safe)
        if conversation_id and isinstance(scope, dict):
            scope = {**scope, "conversation_id": conversation_id}

        if not isinstance(scope, dict):
            raise HTTPException(status_code=400, detail="Missing user scope (user.soul_id required)")
        soul_id = str(scope.get("soul_id") or "").strip()
        if not soul_id:
            raise HTTPException(status_code=400, detail="Missing user.soul_id for per-soul DBs")
        scope = {**scope, "soul_id": soul_id}

        conversation = safe.get("conversation")
        if conversation is None:
            conversation = safe.get("content")
        if conversation is None and isinstance(payload, list):
            conversation = payload
        if not isinstance(conversation, list) or not conversation:
            raise HTTPException(status_code=400, detail="Missing or empty 'conversation' list")

        conv_norm = _normalize_conversation(conversation)

        uid = str((scope or {}).get("user_id") or "user") if isinstance(scope, dict) else "user"
        soul_id = str((scope or {}).get("soul_id") or "soul") if isinstance(scope, dict) else "soul"
        async with _get_memorize_lock(_memorize_lock_key(uid, soul_id)):
            storage_dir = _get_storage_dir(_CONFIG)
            chats_dir = (storage_dir / "st_chats").resolve()
            chat_file = _pick_str(safe, "chat_file_name")
            resource_url_in = _pick_str(safe, "resource_url")
            chat_dir, chat_key, chat_key_source = _resolve_chat_storage_dir(
                chats_dir,
                uid,
                soul_id,
                conversation_id,
                chat_file,
                resource_url_in,
            )
            days_dir = (chat_dir / "days").resolve()
            chat_dir.mkdir(parents=True, exist_ok=True)
            days_dir.mkdir(parents=True, exist_ok=True)

            full_path = (chat_dir / "full.json").resolve()
            manifest_path = (chat_dir / "manifest.json").resolve()

            prev_full = _read_list(full_path)
            prev_len = len(prev_full)
            merged_len = len(conv_norm) if isinstance(conv_norm, list) else 0
            merged = prev_full
            if isinstance(conv_norm, list):
                merged = _merge_conv(prev_full, conv_norm)
                merged_len = len(merged)
                _write_list_if_changed(full_path, prev_full, merged)

            processed_cursor = -1
            if conversation_id:
                state_out, _db_path = _write_conversation_state(
                    conversation_id,
                    soul_id=soul_id,
                    user_id=uid,
                    updates={},
                )
                if state_out.get("last_memorize_at"):
                    processed_cursor = int(state_out.get("digest_cursor") or 0)

            tz_name = _pick_str(safe, "time_zone")
            tz_off_raw = safe.get("time_zone_offset_min")
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
            if not tz_ok and tz_off_min is not None:
                try:
                    zi = timezone(timedelta(minutes=tz_off_min))
                    tz_ok = True
                    if not tz_name:
                        tz_name = f"offset({tz_off_min})"
                except Exception:
                    zi = None
                    tz_ok = False

            rawm = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else ""
            manifest: dict[str, Any] = json.loads(rawm) if rawm.strip() else {}
            segments: list[dict[str, Any]] = (
                manifest.get("segments") if isinstance(manifest.get("segments"), list) else []
            )

            resource_url = str(full_path)
            days_written = 0
            sleep_stats: Any | None = None
            new_segments: list[dict[str, Any]] = []
            if tz_ok and isinstance(merged, list) and any(isinstance(m.get("ts_ms"), int) for m in merged):
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
                            st_i = int(s.get("start"))
                            en_i = int(s.get("end"))
                            if st_i <= tail_start <= en_i:
                                rebuild_from = st_i
                            if st_i < rebuild_from:
                                keep_segments.append(s)
                        except Exception:
                            continue

                ctx_start = max(0, rebuild_from - 1)
                splits_rel, sleep_stats = _split_indices_by_sleep(
                    merged[ctx_start:], zi, tz_ok, _SLEEP_SPLIT_MIN_LULL_SECONDS
                )
                splits = [ctx_start + i for i in splits_rel if (ctx_start + i) > rebuild_from]

                boundaries = [rebuild_from] + splits + [len(merged)]
                for a_i, b_i in zip(boundaries, boundaries[1:]):
                    if b_i <= a_i:
                        continue
                    seg = merged[a_i:b_i]
                    if not seg:
                        continue
                    first_ts = seg[0].get("ts_ms") if isinstance(seg[0], dict) else None
                    last_ts = seg[-1].get("ts_ms") if isinstance(seg[-1], dict) else None
                    date = _date_label(first_ts if isinstance(first_ts, int) else None, zi)
                    end_date = _date_label(last_ts if isinstance(last_ts, int) else None, zi)
                    fn = f"{date}.json" if end_date == date else f"{date}__{end_date}.json"
                    fp = (days_dir / fn).resolve()
                    old_seg = _read_list(fp)
                    _write_list_if_changed(fp, old_seg, seg)
                    days_written += 1 if seg != old_seg else 0
                    new_segments.append({"date": date, "end_date": end_date, "start": a_i, "end": b_i - 1, "file": fn})

                segments = keep_segments + new_segments
                manifest_out = {
                    "v": 1,
                    "tz": str(tz_name or ""),
                    "segments": segments,
                    "split": {
                        "min_lull_seconds": _SLEEP_SPLIT_MIN_LULL_SECONDS,
                    },
                    "source": {
                        "conversationId": conversation_id or "",
                        "chatFileName": chat_file or "",
                        "resource_url_in": resource_url_in or "",
                        "timeZoneOffsetMin": tz_off_min if tz_off_min is not None else None,
                        "chatKey": chat_key,
                        "chatKeySource": chat_key_source or "",
                    },
                }
                manifest_path.write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8")

            if segments:
                last_file = segments[-1].get("file")
                if isinstance(last_file, str) and last_file:
                    resource_url = str((days_dir / last_file).resolve())

            memorize_batches: list[tuple[str, list[dict[str, Any]], int]] = []
            if force and isinstance(merged, list):
                start_idx = max(0, processed_cursor + 1)
                memorize_batches = _build_force_memorize_batches(
                    merged,
                    start_idx=start_idx,
                    segments=segments,
                    days_dir=days_dir,
                    full_path=full_path,
                    resource_url=resource_url,
                    max_chunk_tokens=6000,
                )
            elif segments and isinstance(merged, list):
                _last_idx = len(merged) - 1
                _carry: tuple[int, int] | None = None
                for _seg in segments:
                    try:
                        _si = int(_seg.get("start"))
                        _ei = int(_seg.get("end"))
                    except Exception:
                        continue
                    if _ei < _si or _ei > _last_idx or _ei <= processed_cursor:
                        continue
                    _eff = _carry[0] if _carry is not None else max(_si, processed_cursor + 1)
                    _carry = None
                    if _eff > _ei:
                        continue
                    _bc = merged[_eff : _ei + 1]
                    if not _bc:
                        continue
                    if _MIN_CHUNK_TOKENS > 0 and _estimate_tokens(_bc) < _MIN_CHUNK_TOKENS:
                        _carry = (_eff, _ei)
                        continue
                    _bf = _seg.get("file")
                    _bu = resource_url
                    if isinstance(_bf, str) and _bf:
                        _bu = str((days_dir / _bf).resolve())
                    memorize_batches.append((_bu, _bc, _ei))
                if _carry is not None:
                    _bc = merged[_carry[0] : _carry[1] + 1]
                    if _bc:
                        memorize_batches.append((resource_url, _bc, _carry[1]))

            expected_cursor = memorize_batches[-1][2] if memorize_batches else processed_cursor
            background_tasks.add_task(
                _run_memorize_batches,
                memorize_batches=memorize_batches,
                svc=svc,
                scope=scope,
                conversation_id=conversation_id,
                soul_id=soul_id,
                uid=uid,
                processed_cursor=processed_cursor,
                safe=safe,
                resource_url=resource_url,
                chat_file=chat_file,
                resource_url_in=resource_url_in,
                chat_key=chat_key,
                chat_key_source=chat_key_source,
                tz_name=tz_name,
                prev_len=prev_len,
                merged_len=len(merged) if isinstance(merged, list) else 0,
                force=force,
                days_written=days_written,
                sleep_stats=sleep_stats if "sleep_stats" in locals() else None,
            )
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=202,
                content={
                    "ok": True,
                    "status": "accepted",
                    "conversation_id": conversation_id,
                    "expected_cursor": expected_cursor,
                    "batch_count": len(memorize_batches),
                    "resource_url": resource_url,
                },
            )
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        _record_call(
            "memorize",
            payload if isinstance(payload, dict) else None,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise HTTPException(status_code=500, detail="Internal Server Error. Check server logs.") from exc


@app.post("/conversation/{conversation_id}/consolidation/force", operation_id="force_consolidation")
async def force_consolidation(
    conversation_id: str,
    payload: dict[str, Any] = Body(...),
):
    cid = str(conversation_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="conversation_id is required")

    uid = ""
    soul_id = ""
    state_lock: asyncio.Lock | None = None
    pipeline_started = False
    try:
        safe = _safe_payload(payload if isinstance(payload, dict) else {})
        if not isinstance(safe.get("llm_profiles"), dict):
            safe["llm_profiles"] = _default_llm_profiles_from_server_config()
        scope = safe.get("user")
        if not isinstance(scope, dict):
            scope = _extract_scope(safe) or None
        if not isinstance(scope, dict):
            raise HTTPException(status_code=400, detail="user scope required")

        uid = str(scope.get("user_id") or "").strip()
        soul_id = str(scope.get("soul_id") or "").strip()
        if not uid or not soul_id:
            raise HTTPException(status_code=400, detail="user_id and soul_id required")

        safe["user"] = {"user_id": uid, "soul_id": soul_id, "conversation_id": cid}
        safe["conversation_id"] = cid
        svc = _get_service_from_payload(safe)

        state_lock = _get_memorize_lock(_memorize_lock_key(uid, soul_id))
        pipeline_started = True
        out = await _run_consolidation_pipeline_once(
            svc=svc,
            deps=_make_consolidation_deps(),
            state_lock=state_lock,
            conversation_id=cid,
            soul_id=soul_id,
            user_id=uid,
            force=True,
        )
        if out.get("status") == "skipped":
            reason = str(out.get("reason") or "")
            if reason == "in_progress":
                raise HTTPException(status_code=409, detail="consolidation already in progress")
            return {"ok": True, "status": "skipped", "reason": reason}
        result = out.get("result") if isinstance(out.get("result"), dict) else {}

        _record_call(
            "consolidation.force",
            safe,
            ok=True,
            info={
                "conversationId": cid,
                "diary_count": len(result.get("diary_memory_ids") or []),
            },
        )
        return {"ok": True, "status": "completed", "result": result}
    except HTTPException as exc:
        if (
            pipeline_started
            and
            state_lock is not None
            and uid
            and soul_id
            and exc.status_code >= 500
            and not (exc.status_code == 409 and str(exc.detail or "") == "consolidation already in progress")
        ):
            await _clear_consolidation_in_progress(
                state_lock=state_lock,
                conversation_id=cid,
                soul_id=soul_id,
                user_id=uid,
            )
        _record_call(
            "consolidation.force", payload if isinstance(payload, dict) else None, ok=False, error="HTTPException"
        )
        raise
    except Exception as exc:
        if pipeline_started and state_lock is not None and uid and soul_id:
            await _clear_consolidation_in_progress(
                state_lock=state_lock,
                conversation_id=cid,
                soul_id=soul_id,
                user_id=uid,
            )
        _record_call(
            "consolidation.force",
            payload if isinstance(payload, dict) else None,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise HTTPException(status_code=500, detail="Internal Server Error. Check server logs.") from exc


@app.get("/categories", operation_id="list_memory_categories")
async def list_memory_categories(user_id: str = "", soul_id: str = "", include_empty: bool = False):
    soul_id = soul_id.strip()
    if not soul_id:
        raise HTTPException(status_code=400, detail="soul_id required")
    scope: dict[str, Any] = {"soul_id": soul_id}
    if user_id.strip():
        scope["user_id"] = user_id.strip()

    default_profile = _default_llm_profiles_from_server_config()["default"]

    payload = {
        "llm_profiles": {
            "default": default_profile,
        },
        "user": scope,
    }
    svc = _get_service_from_payload(payload)

    cats_map = svc.database.memory_category_repo.list_categories(scope)
    out = [
        {"name": c.name, "summary": c.summary or ""}
        for c in cats_map.values()
        if c.name and (include_empty or _has_category_content({"name": c.name, "summary": c.summary or ""}))
    ]
    return {"categories": out}


@app.post("/categories/search", operation_id="search_memory_categories")
async def search_memory_categories(payload: dict[str, Any]):
    """Payload-driven category listing (matches SillyTavern plugin's local mode)."""
    try:
        safe = _safe_payload(payload)
        svc = _get_service_from_payload(safe, allow_missing_llm_profiles=True)

        scope = safe.get("scope") or safe.get("where")
        if scope is not None and not isinstance(scope, dict):
            raise HTTPException(status_code=400, detail="'scope' must be an object")
        if scope is None:
            scope = safe.get("user") if isinstance(safe.get("user"), dict) else (_extract_scope(safe) or None)
        scope = _canonicalize_scope_where(scope)

        include_empty = bool(safe.get("include_empty"))

        cats_map = svc.database.memory_category_repo.list_categories(scope)
        out = [
            {"name": c.name, "summary": c.summary or ""}
            for c in cats_map.values()
            if c.name and (include_empty or _has_category_content({"name": c.name, "summary": c.summary or ""}))
        ]
        _record_call("categories.search", safe, ok=True, info={"returned": len(out)})
        return {"categories": out}
    except HTTPException:
        _record_call(
            "categories.search", payload if isinstance(payload, dict) else None, ok=False, error="HTTPException"
        )
        raise
    except Exception as exc:
        _record_call(
            "categories.search",
            payload if isinstance(payload, dict) else None,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise HTTPException(status_code=500, detail="Internal Server Error. Check server logs.") from exc


@app.get("/souls/{soul_id}/intentions", operation_id="list_intentions")
async def list_intentions(
    soul_id: str,
    user_id: str,
    status: str = "active",
):
    soul_id = str(soul_id or "").strip()
    user_id = str(user_id or "").strip()
    scoped_status = str(status or "").strip() or "active"

    if not soul_id:
        raise HTTPException(status_code=400, detail="soul_id required")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    db_path = _sqlite_current_path(user_id, soul_id)
    if db_path is None:
        raise HTTPException(status_code=400, detail="soul_id required for sqlite scope resolution")
    if not db_path.exists():
        return []

    con = _sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        _sqlite_ensure_conversation_state_schema(con)
        rows = con.execute(
            """
SELECT * FROM intentions_life_goals
WHERE soul_id = ? AND user_id = ? AND status = ?
""",
            (soul_id, user_id, scoped_status),
        ).fetchall()
        return [_intention_row_to_dict(row) for row in rows]
    finally:
        con.close()


@app.post("/souls/{soul_id}/narrative_suggestion", operation_id="narrative_suggestion")
async def narrative_suggestion(soul_id: str, payload: dict[str, Any] = Body(...)):
    soul_id = str(soul_id or "").strip()
    user_id = str(payload.get("user_id") or "").strip()
    suggestion = str(payload.get("suggestion") or "").strip()
    if not soul_id or not user_id or not suggestion:
        raise HTTPException(status_code=400, detail="soul_id, user_id, suggestion required")

    db_path = _sqlite_current_path(user_id, soul_id)
    current_narrative = ""
    all_cats_summary = ""
    if db_path is not None and db_path.exists():
        con = _sqlite_connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            _sqlite_ensure_conversation_state_schema(con)
            sm = con.execute(
                "SELECT narrative_self FROM memu_self_model "
                "WHERE soul_id = ? AND user_id = ? ORDER BY updated_at DESC LIMIT 1",
                (soul_id, user_id),
            ).fetchone()
            if sm is not None:
                current_narrative = str(sm["narrative_self"] or "").strip()
            cs = con.execute(
                "SELECT all_categories_summary FROM memu_conversation_state "
                "WHERE soul_id = ? AND user_id = ? "
                "AND all_categories_summary IS NOT NULL AND all_categories_summary != '' "
                "ORDER BY updated_at DESC LIMIT 1",
                (soul_id, user_id),
            ).fetchone()
            if cs is not None:
                all_cats_summary = str(cs["all_categories_summary"] or "").strip()
        finally:
            con.close()

    identity_context = _build_retrieve_identity_context(soul_id)
    system_prompt = (
        f"{identity_context}\n\n"
        f"I, {soul_id}, have received a suggestion from {user_id} about how I might be. "
        "They want me to be a certain way. It's up to me to decide if I want that. "
        "I might find it aligns with who I already am, stirs something worth integrating, "
        "or doesn't fit me at all. My call.\n\n"
        "Respond with JSON only. Either:\n"
        '  {"narrative_self": "..."}   — my updated self-picture, first person, if I want to integrate this\n'
        "  {}                          — if I don't want to change"
    )
    user_prompt = (
        f"My current narrative_self:\n{current_narrative or '(empty)'}\n\n"
        f"My synthesized self-picture across categories:\n{all_cats_summary or '(empty)'}\n\n"
        f"{user_id}'s suggestion:\n{suggestion}"
    )

    svc_payload = {"llm_profiles": _default_llm_profiles_from_server_config(), "user": {"user_id": user_id, "soul_id": soul_id}}
    svc = _get_service_from_payload(svc_payload)
    raw = await svc.chat(user_prompt, system_prompt=system_prompt, temperature=0.2, max_tokens=800, response_format={"type": "json_object"})

    text = str(raw or "").strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    parsed = json.loads(text)
    new_narrative = str(parsed.get("narrative_self") or "").strip()

    if new_narrative and db_path is not None:
        _sqlite_ensure_nonempty(db_path)
        con = _sqlite_connect(db_path)
        try:
            _sqlite_ensure_conversation_state_schema(con)
            now_iso = datetime.now(UTC).isoformat()
            self_model_id = str(uuid.uuid4())
            con.execute(
                """
INSERT INTO memu_self_model (id, soul_id, user_id, narrative_self, related_memory_ids, updated_at)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    narrative_self = excluded.narrative_self,
    updated_at = excluded.updated_at
""",
                (self_model_id, soul_id, user_id, new_narrative, None, now_iso),
            )
            con.commit()
        finally:
            con.close()

    return {}


@app.patch("/intentions/{intention_id}", operation_id="patch_intention")
async def patch_intention(
    intention_id: str,
    soul_id: str,
    payload: dict[str, Any] | None = Body(default=None),
):
    iid = str(intention_id or "").strip()
    if not iid:
        raise HTTPException(status_code=400, detail="intention_id is required")
    soul_id = str(soul_id or "").strip()
    if not soul_id:
        raise HTTPException(status_code=400, detail="soul_id required")
    db_path = _sqlite_current_path(None, soul_id)
    if db_path is None:
        raise HTTPException(status_code=400, detail="soul_id required for sqlite scope resolution")
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="intention not found")

    body = payload if isinstance(payload, dict) else {}
    updates: dict[str, Any] = {}

    if "status" in body:
        next_status = str(body.get("status") or "").strip()
        if next_status not in _VALID_INTENTION_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=f"status must be one of: {sorted(_VALID_INTENTION_STATUSES)}",
            )
        updates["status"] = next_status

    if "resolution_note" in body:
        raw_resolution_note = body.get("resolution_note")
        updates["resolution_note"] = None if raw_resolution_note is None else str(raw_resolution_note)

    con = _sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        _sqlite_ensure_conversation_state_schema(con)
        row = con.execute(
            "SELECT * FROM intentions_life_goals WHERE id = ? LIMIT 1",
            (iid,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="intention not found")

        if updates:
            set_parts: list[str] = []
            params: list[Any] = []
            if "status" in updates:
                set_parts.append("status = ?")
                params.append(updates["status"])
            if "resolution_note" in updates:
                set_parts.append("resolution_note = ?")
                params.append(updates["resolution_note"])
            set_parts.append("updated_at = ?")
            params.append(datetime.now(UTC).isoformat())
            params.append(iid)
            con.execute(
                f"UPDATE intentions_life_goals SET {', '.join(set_parts)} WHERE id = ?",
                params,
            )
            con.commit()
            row = con.execute(
                "SELECT * FROM intentions_life_goals WHERE id = ? LIMIT 1",
                (iid,),
            ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="intention not found")
        return _intention_row_to_dict(row)
    finally:
        con.close()


@app.get("/conversation/{conversation_id}/state", operation_id="get_conversation_state")
async def get_conversation_state(
    conversation_id: str,
    soul_id: str | None = None,
    user_id: str | None = None,
):
    cid = str(conversation_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="conversation_id is required")

    db_path: Path | None = None
    state_out: dict[str, Any] | None = None
    soul_id = str(soul_id or "").strip() or None
    user_id = str(user_id or "").strip() or None

    if soul_id:
        db_path = _sqlite_current_path(user_id, soul_id)
        if db_path is None:
            raise HTTPException(status_code=400, detail="soul_id required for sqlite scope resolution")
        if not db_path.exists():
            return {"ok": True, "state": None, "path": str(db_path)}
        con = _sqlite_connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            _sqlite_ensure_conversation_state_schema(con)
            state_out = _conversation_state_from_row(_conversation_state_row(con, cid))
        finally:
            con.close()
    else:
        db_path, state_out = _find_conversation_state_across_dbs(cid)

    return {"ok": True, "state": state_out, "path": str(db_path) if db_path else None}


@app.patch("/conversation/{conversation_id}/state", operation_id="patch_conversation_state")
async def patch_conversation_state(
    conversation_id: str,
    payload: dict[str, Any] | None = Body(default=None),
    soul_id: str | None = None,
    user_id: str | None = None,
):
    cid = str(conversation_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="conversation_id is required")
    body = payload if isinstance(payload, dict) else {}

    body_soul_id = _pick_str(body, "soul_id")
    body_user_id = _pick_str(body, "user_id")
    soul_id = body_soul_id or (str(soul_id or "").strip() or None)
    user_id = body_user_id or (str(user_id or "").strip() or None)

    updates: dict[str, Any] = {}

    if "soul_id" in body:
        soul_id = body_soul_id
    if "user_id" in body:
        user_id = body_user_id

    if "digest_cursor" in body:
        raw_cursor = body.get("digest_cursor")
        updates["digest_cursor"] = 0 if raw_cursor is None else raw_cursor

    if "prior_context" in body:
        updates["prior_context"] = body.get("prior_context")

    if "intentions_active" in body:
        updates["intentions_active"] = body.get("intentions_active")

    if "memory_cache" in body:
        updates["memory_cache"] = body.get("memory_cache")

    if "pending_diary_episode_ids" in body:
        updates["pending_diary_episode_ids"] = body.get("pending_diary_episode_ids")

    if "self_model_id" in body:
        updates["self_model_id"] = body.get("self_model_id")

    if "last_retrieval_ids" in body:
        updates["last_retrieval_ids"] = body.get("last_retrieval_ids")

    if "last_memorize_at" in body:
        updates["last_memorize_at"] = body.get("last_memorize_at")
    if "last_consolidation_at" in body:
        updates["last_consolidation_at"] = body.get("last_consolidation_at")
    if "consolidation_in_progress" in body:
        updates["consolidation_in_progress"] = body.get("consolidation_in_progress")
    if "consolidation_started_at" in body:
        updates["consolidation_started_at"] = body.get("consolidation_started_at")

    state_out, db_path = _write_conversation_state(
        cid,
        soul_id=soul_id,
        user_id=user_id,
        updates=updates,
    )
    return {"ok": True, "state": state_out, "path": str(db_path)}


@app.post("/clear", operation_id="clear_memory")
async def clear_memory(payload: dict[str, Any]):
    """Clear stored memory for a single scoped relationship.

    Safety default:
      - requires both user_id and soul_id
      - does not allow unscoped/global clear
    """
    try:
        safe = _safe_payload(payload)

        scope = safe.get("scope") or safe.get("where")
        if scope is not None and not isinstance(scope, dict):
            raise HTTPException(status_code=400, detail="'scope' must be an object")
        if scope is None:
            if isinstance(safe.get("user"), dict):
                scope = dict(safe.get("user") or {})
            else:
                scope = _extract_scope(safe) or {}

        uid = str((scope or {}).get("user_id") or "").strip()
        sid = str((scope or {}).get("soul_id") or "").strip()
        if not uid or not sid:
            raise HTTPException(status_code=400, detail="user_id and soul_id required")
        scope = {"user_id": uid, "soul_id": sid}

        safe["user"] = scope

        svc = _get_service_from_payload(safe, allow_missing_llm_profiles=True)
        result = await svc.clear_memory(where=scope)

        deleted_categories = result.get("deleted_categories") if isinstance(result, dict) else []
        deleted_items = result.get("deleted_items") if isinstance(result, dict) else []
        deleted_resources = result.get("deleted_resources") if isinstance(result, dict) else []

        out = {
            "ok": True,
            "result": result,
            "purged": {
                "categories": len(deleted_categories) if isinstance(deleted_categories, list) else 0,
                "items": len(deleted_items) if isinstance(deleted_items, list) else 0,
                "resources": len(deleted_resources) if isinstance(deleted_resources, list) else 0,
            },
            "where": scope,
        }
        _record_call("clear", safe, ok=True, info={"where": scope, "purged": out["purged"]})
        return out
    except HTTPException:
        _record_call("clear", payload if isinstance(payload, dict) else None, ok=False, error="HTTPException")
        raise
    except Exception as exc:
        _record_call(
            "clear", payload if isinstance(payload, dict) else None, ok=False, error=f"{type(exc).__name__}: {exc}"
        )
        raise HTTPException(status_code=500, detail="Internal Server Error. Check server logs.") from exc


@app.post("/retrieve", operation_id="retrieve")
async def retrieve(payload: dict[str, Any]):
    try:
        out = await _run_retrieve(payload)
        _record_call(
            "retrieve",
            _safe_payload(payload),
            ok=True,
            info={
                "queries": out.get("queries"),
                "where": _extract_retrieve_where(_safe_payload(payload)),
                "method": out.get("method"),
                "conversationId": out.get("conversation_id"),
            },
        )
        return out
    except HTTPException as he:
        _record_call(
            "retrieve",
            payload if isinstance(payload, dict) else None,
            ok=False,
            error=str(getattr(he, "detail", he)),
        )
        raise
    except Exception as exc:
        _record_call(
            "retrieve",
            payload if isinstance(payload, dict) else None,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise HTTPException(status_code=500, detail="Internal Server Error. Check server logs.") from exc


@app.get("/timeline", operation_id="timeline")
async def timeline(
    entity: str,
    user_id: str,
    soul_id: str,
    as_of: str | None = None,
    limit: int = 200,
):
    entity_name = str(entity or "").strip()
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    if not entity_name:
        raise HTTPException(status_code=400, detail="entity is required")
    if not uid or not sid:
        raise HTTPException(status_code=400, detail="user_id and soul_id are required")

    as_of_dt = _parse_as_of_datetime(as_of)
    scope = {"user_id": uid, "soul_id": sid}
    safe = {"user": scope}
    svc = _get_service_from_payload(safe, allow_missing_llm_profiles=True)

    entities = svc.database.entity_repo.list_all(where=scope)
    query_norm = "_".join(entity_name.lower().split())
    target = next(
        (
            e
            for e in entities
            if e.normalized == query_norm or e.name.lower() == entity_name.lower()
        ),
        None,
    )
    if target is None:
        return {
            "ok": True,
            "entity": {"name": entity_name, "id": None},
            "as_of": as_of_dt.isoformat() if as_of_dt is not None else None,
            "timeline": [],
            "count": 0,
        }

    outgoing = svc.database.triple_repo.get_edges_from(
        target.id, current_only=False, where=scope, as_of=as_of_dt
    )
    incoming = svc.database.triple_repo.get_edges_to(
        target.id, current_only=False, where=scope, as_of=as_of_dt
    )

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for edge in [*outgoing, *incoming]:
        edge_id = str(edge.id or "").strip()
        if edge_id and edge_id in seen:
            continue
        if edge_id:
            seen.add(edge_id)

        event: dict[str, Any] = {
            "edge_id": edge_id or None,
            "predicate": edge.predicate,
            "subject_id": edge.subject_id,
            "subject_kind": edge.subject_kind,
            "object_id": edge.object_id,
            "object_kind": edge.object_kind,
            "valid_from": edge.valid_from.isoformat() if isinstance(edge.valid_from, datetime) else None,
            "valid_to": edge.valid_to.isoformat() if isinstance(edge.valid_to, datetime) else None,
            "confidence": edge.confidence,
            "source_memory_id": edge.source_memory_id,
        }

        related_memory_id = None
        if edge.subject_kind == "memory":
            related_memory_id = edge.subject_id
        elif edge.object_kind == "memory":
            related_memory_id = edge.object_id
        elif isinstance(edge.source_memory_id, str) and edge.source_memory_id.strip():
            related_memory_id = edge.source_memory_id.strip()
        if related_memory_id:
            memory_item = svc.database.memory_item_repo.get_item(related_memory_id)
            if memory_item is not None:
                event["memory"] = {
                    "id": memory_item.id,
                    "memory_type": memory_item.memory_type,
                    "summary": memory_item.summary,
                    "happened_at": (
                        memory_item.happened_at.isoformat()
                        if isinstance(memory_item.happened_at, datetime)
                        else None
                    ),
                }

        rows.append(event)

    rows.sort(
        key=lambda row: (
            row.get("valid_from") or "",
            row.get("edge_id") or "",
        )
    )
    limit = max(1, min(int(limit or 200), 500))
    timeline_rows = rows[:limit]
    return {
        "ok": True,
        "entity": {
            "id": target.id,
            "name": target.name,
            "entity_type": target.entity_type,
        },
        "as_of": as_of_dt.isoformat() if as_of_dt is not None else None,
        "timeline": timeline_rows,
        "count": len(timeline_rows),
    }


@app.post("/conversation/{conversation_id}/retrieve", operation_id="conversation_retrieve")
async def conversation_retrieve(
    conversation_id: str,
    payload: dict[str, Any] = Body(...),
):
    cid = str(conversation_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="conversation_id is required")
    try:
        safe = _safe_payload(payload if isinstance(payload, dict) else {})
        if not isinstance(safe.get("llm_profiles"), dict):
            safe["llm_profiles"] = _default_llm_profiles_from_server_config()
        if safe.get("queries") is None:
            scope = _extract_scope(safe)
            uid = str(scope.get("user_id") or "").strip()
            soul_id = str(scope.get("soul_id") or "").strip()
            message = _pick_str(safe, "message", "query") or ""
            history = _normalize_turn_history(safe.get("history"))
            if uid and soul_id and message.strip():
                state_row, _soul_card, _db_path = _load_turn_state_and_soul_card(
                    cid,
                    user_id=uid,
                    soul_id=soul_id,
                )
                safe["queries"] = _build_retrieve_soul_context_queries(
                    soul_id=soul_id,
                    message=message,
                    history=history,
                    state_row=state_row,
                )

        out = await _run_retrieve(safe, conversation_id=cid)

        want_turn_prompt = bool(safe.get("build_turn_prompt", False))

        if want_turn_prompt:
            scope = _extract_scope(safe)
            uid = str(scope.get("user_id") or "").strip()
            soul_id = str(scope.get("soul_id") or "").strip()

            if uid and soul_id:
                _state_row, soul_card, _db_path = _load_turn_state_and_soul_card(
                    cid,
                    user_id=uid,
                    soul_id=soul_id,
                )
                payload_soul_card = str(safe.get("soul_card") or "").strip() or None
                soul_card = payload_soul_card or soul_card

                message = _pick_str(safe, "message", "query") or ""
                history = _normalize_turn_history(safe.get("history"))
                memory_cache = _normalize_memory_cache_impl(out.get("memory_cache"))
                intentions_active = _normalize_intentions_stack_impl(out.get("intentions_active"))
                life_goals_active = _load_active_life_goals_for_prompt(user_id=uid, soul_id=soul_id)

                out["turn_system_prompt"] = _make_turn_system_prompt(soul_id, soul_card=soul_card)
                out["turn_user_prompt"] = _build_turn_prompt(
                    user_message=message,
                    history=history,
                    history_token_budget=_TURN_HISTORY_TOKEN_BUDGET,
                    prior_context=out.get("prior_context"),
                    retrieve_rag=out.get("result"),
                    all_categories_summary=_state_row.get("all_categories_summary"),
                    memory_cache=memory_cache,
                    intentions_active=intentions_active,
                    life_goals_active=life_goals_active,
                )

        _record_call(
            "conversation.retrieve",
            safe,
            ok=True,
            info={
                "queries": out.get("queries"),
                "where": _extract_retrieve_where({**safe, "conversation_id": cid}),
                "method": out.get("method"),
                "conversationId": cid,
                "persistedState": bool(out.get("state")),
            },
        )
        return out
    except HTTPException:
        _record_call(
            "conversation.retrieve", payload if isinstance(payload, dict) else None, ok=False, error="HTTPException"
        )
        raise
    except Exception as exc:
        _record_call(
            "conversation.retrieve",
            payload if isinstance(payload, dict) else None,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise HTTPException(status_code=500, detail="Internal Server Error. Check server logs.") from exc


def _turn_state_read(
    cid: str,
    uid: str,
    soul_id: str,
    safe: dict[str, Any],
    state_override_cache: list[str],
    state_override_intentions: dict[str, Any],
    apply_turn_maintenance: bool,
    dry_run: bool,
    history_full: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    Any,
    Any,
    list[str],
    dict[str, Any],
    int,
    "dict[str, Any] | None",
]:
    state_row, soul_card, db_path = _load_turn_state_and_soul_card(cid, user_id=uid, soul_id=soul_id)
    payload_soul_card = str(safe.get("soul_card") or "").strip() or None
    soul_card = payload_soul_card or soul_card
    memory_cache_before = list(state_override_cache)
    intentions_before = (
        _apply_intention_turn_maintenance_impl(state_override_intentions)
        if apply_turn_maintenance
        else _normalize_intentions_stack_impl(state_override_intentions)
    )
    digest_cursor_for_gate = state_row.get("digest_cursor") if state_row.get("last_memorize_at") else -1
    force_memorize_unmemorized_tokens = _estimate_unmemorized_tokens(history_full, digest_cursor_for_gate)
    force_memorize_payload: dict[str, Any] | None = None
    if (not dry_run) and history_full and force_memorize_unmemorized_tokens > _TURN_HISTORY_TOKEN_BUDGET:
        memorize_history = _normalize_conversation(history_full)
        if memorize_history:
            force_memorize_payload = {
                **safe,
                "conversation_id": cid,
                "conversation": memorize_history,
                "user": {"user_id": uid, "soul_id": soul_id, "conversation_id": cid},
            }
    return (
        state_row,
        soul_card,
        db_path,
        memory_cache_before,
        intentions_before,
        force_memorize_unmemorized_tokens,
        force_memorize_payload,
    )


def _turn_state_write(
    cid: str,
    uid: str,
    soul_id: str,
    cache_entry: str,
    intention_action: Any,
    annulment_ids: list[str],
    chat_x: str | None,
    apply_turn_maintenance: bool,
) -> tuple[dict[str, Any], Any]:
    fresh_row, _, _ = _load_turn_state_and_soul_card(cid, user_id=uid, soul_id=soul_id)
    fresh_cache = _normalize_memory_cache_impl(fresh_row.get("memory_cache"))
    fresh_intentions = _normalize_intentions_stack_impl(fresh_row.get("intentions_active"))
    memory_cache_after = (
        _append_memory_cache_entry(fresh_cache, cache_entry) if cache_entry else list(fresh_cache)
    )
    intentions_after = (
        _apply_intention_action(fresh_intentions, intention_action)
        if apply_turn_maintenance
        else fresh_intentions
    )
    intentions_after = _remove_intentions(
        intentions_after,
        [item_id for item_id in annulment_ids if item_id],
    )
    _next_last_chat_x, _next_last_chat_x_prev = _next_chat_x_state_values(chat_x, fresh_row)
    state_out, state_path = _write_conversation_state(
        cid,
        soul_id=soul_id,
        user_id=uid,
        updates={
            "intentions_active": intentions_after,
            "memory_cache": memory_cache_after,
            "last_chat_x": _next_last_chat_x,
            "last_chat_x_prev": _next_last_chat_x_prev,
            "undo_snapshot": {
                "memory_cache": fresh_cache,
                "intentions_active": fresh_intentions,
            },
        },
    )
    return state_out, state_path


def _turn_launch_apimw(
    cid: str,
    uid: str,
    soul_id: str,
    safe: dict[str, Any],
    history_full: list[dict[str, Any]],
    state_out: dict[str, Any],
    state_row: dict[str, Any],
) -> str:
    apimw_state = state_out
    _cadence_anchors = _compact_chat_x_anchors(
        str(apimw_state.get("last_chat_x") or "").strip() or None,
    )
    _cadence_slice = _slice_history_from_chat_x_anchors(history_full, _cadence_anchors, limit=999)
    _cadence_threshold = _apimw_cadence_from_cfg(_CONFIG)
    _cadence_soul_messages = _count_soul_messages(_cadence_slice, soul_id)
    if _cadence_soul_messages < _cadence_threshold:
        return "skipped_cadence"
    if not _mark_apimw_inflight(cid):
        return "skipped_inflight"
    try:
        apimw_task = asyncio.create_task(
            _run_apimw(
                safe,
                conversation_id=cid,
                soul_id=soul_id,
                user_id=uid,
                state_row=apimw_state,
                history=history_full,
            )
        )
    except Exception:
        _clear_apimw_inflight(cid)
        logger.exception("APImw background pipeline failed to start for %s", cid)
        return "failed_to_start"

    def _on_apimw_done(task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception:
            logger.exception("APImw background pipeline failed for %s", cid)
        finally:
            _clear_apimw_inflight(cid)

    apimw_task.add_done_callback(_on_apimw_done)
    return "started"


@app.post("/conversation/{conversation_id}/turn", operation_id="conversation_turn")
async def conversation_turn(
    conversation_id: str,
    payload: dict[str, Any] = Body(...),
):
    cid = str(conversation_id or "").strip()
    try:
        if not cid:
            raise HTTPException(status_code=400, detail="conversation_id is required")

        safe = _safe_payload(payload if isinstance(payload, dict) else {})
        if not isinstance(safe.get("llm_profiles"), dict):
            safe["llm_profiles"] = _default_llm_profiles_from_server_config()

        scope = safe.get("user")
        if not isinstance(scope, dict):
            scope = _extract_scope(safe) or None
        if not isinstance(scope, dict):
            raise HTTPException(status_code=400, detail="user scope required")

        uid = str(scope.get("user_id") or "").strip()
        soul_id = str(scope.get("soul_id") or "").strip()
        if not uid or not soul_id:
            raise HTTPException(status_code=400, detail="user_id and soul_id required")

        message = str(safe.get("message") or "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="message is required")

        history_full = _normalize_turn_history(safe.get("history"))
        if safe.get("prompt_override") is not None:
            raise HTTPException(status_code=400, detail="prompt_override is unsupported; use prompt_override_payload")
        prompt_override_payload_raw = safe.get("prompt_override_payload")
        if not isinstance(prompt_override_payload_raw, dict):
            raise HTTPException(status_code=400, detail="prompt_override_payload is required")
        prompt_override_payload = dict(prompt_override_payload_raw)
        payload_user_prompt = str(prompt_override_payload.get("user_prompt", "")).strip()
        if not payload_user_prompt:
            raise HTTPException(
                status_code=400,
                detail="prompt_override_payload.user_prompt is required",
            )
        payload_system_prompt = str(prompt_override_payload.get("system_prompt", "")).strip()
        payload_retrieve_rag = prompt_override_payload.get("retrieve_rag")
        if not isinstance(payload_retrieve_rag, dict):
            raise HTTPException(
                status_code=400,
                detail="prompt_override_payload.retrieve_rag is required",
            )
        payload_memory_cache_raw = prompt_override_payload.get("memory_cache")
        if not isinstance(payload_memory_cache_raw, list):
            raise HTTPException(
                status_code=400,
                detail="prompt_override_payload.memory_cache is required",
            )
        payload_intentions_raw = prompt_override_payload.get("intentions_active")
        if not isinstance(payload_intentions_raw, dict):
            raise HTTPException(
                status_code=400,
                detail="prompt_override_payload.intentions_active is required",
            )
        state_override_cache: list[str] = _normalize_memory_cache_impl(payload_memory_cache_raw)
        state_override_intentions: dict[str, Any] = _normalize_intentions_stack_impl(payload_intentions_raw)
        payload_retrieve_ms = prompt_override_payload.get("retrieve_ms")
        _rag_ms = int(payload_retrieve_ms) if isinstance(payload_retrieve_ms, (int, float)) else 0
        dry_run = bool(safe.get("dry_run", False))
        run_apimw = _retrieve_apimw_enabled_from_cfg(_CONFIG) and bool(safe.get("run_apimw", True))
        apply_turn_maintenance = bool(safe.get("apply_turn_maintenance", True))
        include_debug = bool(safe.get("debug", False))
        if dry_run:
            run_apimw = False

        safe["user"] = {"user_id": uid, "soul_id": soul_id, "conversation_id": cid}
        safe["conversation_id"] = cid

        state_lock = _get_memorize_lock(_memorize_lock_key(uid, soul_id))

        async with state_lock:
            (
                state_row,
                soul_card,
                db_path,
                memory_cache_before,
                intentions_before,
                force_memorize_unmemorized_tokens,
                force_memorize_payload,
            ) = _turn_state_read(
                cid, uid, soul_id, safe, state_override_cache, state_override_intentions,
                apply_turn_maintenance, dry_run, history_full,
            )

        soul_gen = _load_soul_gen_config(uid, soul_id)
        turn_temperature: float = float(soul_gen.get("temperature", 0.2))
        turn_max_tokens: int = int(soul_gen.get("max_tokens", 1000))
        turn_response_format: Any = {"type": "json_object"}
        req_temperature = safe.get("temperature")
        req_max_tokens = safe.get("max_tokens")
        if req_temperature is not None or req_max_tokens is not None:
            updated = dict(soul_gen)
            if req_temperature is not None:
                try:
                    updated["temperature"] = float(req_temperature)
                    turn_temperature = updated["temperature"]
                except (TypeError, ValueError):
                    pass
            if req_max_tokens is not None:
                try:
                    updated["max_tokens"] = int(req_max_tokens)
                    turn_max_tokens = updated["max_tokens"]
                except (TypeError, ValueError):
                    pass
            if updated != soul_gen:
                _save_soul_gen_config(uid, soul_id, updated)

        turn_system_prompt = payload_system_prompt or _make_turn_system_prompt(soul_id, soul_card=soul_card)
        turn_user_prompt = payload_user_prompt

        svc = _get_service_from_payload(safe, retrieve_method_override="rag")
        _turn_t0 = time.monotonic()
        llm_raw = await svc.chat(
            turn_user_prompt,
            system_prompt=turn_system_prompt,
            temperature=turn_temperature,
            max_tokens=turn_max_tokens,
            response_format=turn_response_format,
        )
        _turn_ms = int((time.monotonic() - _turn_t0) * 1000)
        try:
            turn_contract = _parse_turn_contract(llm_raw)
        except Exception as exc:
            raw_snippet = str(llm_raw or "")[:200]
            raise HTTPException(
                status_code=502,
                detail=f"turn contract parse failure: {exc}; raw={raw_snippet!r}",
            ) from exc

        cache_entry = str(turn_contract.get("cache_entry") or "").strip()
        intention_action = turn_contract.get("intention_action")
        annulments = turn_contract.get("annulments") if isinstance(turn_contract.get("annulments"), list) else []
        annulment_ids = [str(row.get("intention_id") or "").strip() for row in annulments if isinstance(row, dict)]
        chat_x = str(turn_contract.get("chat_x") or "").strip() or None

        state_out = state_row
        state_path = db_path
        annulment_memory_ids: list[str] = []

        if not dry_run:
            async with state_lock:
                state_out, state_path = _turn_state_write(
                    cid, uid, soul_id,
                    cache_entry, intention_action, annulment_ids, chat_x,
                    apply_turn_maintenance,
                )

        if not dry_run:
            annulment_memory_ids = await _persist_annulment_memories(
                svc=svc,
                scope={"user_id": uid, "soul_id": soul_id},
                conversation_id=cid,
                intentions_before=intentions_before,
                annulments=[row for row in annulments if isinstance(row, dict)],
            )

        apimw_status = "skipped_dry_run" if dry_run else "not_started"
        if (not dry_run) and run_apimw:
            apimw_status = _turn_launch_apimw(
                cid, uid, soul_id, safe, history_full, state_out, state_row,
            )

        _response_str = str(turn_contract.get("response") or "").strip()
        out: dict[str, Any] = {
            "ok": True,
            "conversation_id": cid,
            "response": _response_str,
            "apimw": apimw_status,
            "prompt_override_used": True,
            "final_turn_payload": {
                "system_prompt": turn_system_prompt,
                "user_prompt": turn_user_prompt,
                "memory_cache": memory_cache_before,
                "intentions_active": intentions_before,
            },
            "retrieve_ms": _rag_ms,
            "turn_ms": _turn_ms,
            "reply_chars": len(_response_str),
            "turn_prompt_chars": len(turn_user_prompt),
            "turn_system_chars": len(turn_system_prompt),
        }
        if include_debug:
            out["state"] = state_out
            out["path"] = str(state_path) if state_path is not None else None
            out["annulment_memory_ids"] = annulment_memory_ids
            out["turn_contract"] = turn_contract
            out["dry_run"] = dry_run
            out["forced_memorize"] = {
                "queued": bool(force_memorize_payload),
                "unmemorized_tokens": force_memorize_unmemorized_tokens,
                "history_budget_tokens": _TURN_HISTORY_TOKEN_BUDGET,
            }

        if force_memorize_payload is not None:
            asyncio.create_task(_run_forced_memorize_from_turn(force_memorize_payload))

        _record_call(
            "conversation.turn",
            safe,
            ok=True,
            info={
                "conversationId": cid,
                "dryRun": dry_run,
                "apimw": apimw_status,
                "forcedMemorizeQueued": bool(force_memorize_payload),
                "unmemorizedTokens": force_memorize_unmemorized_tokens,
                "historyBudgetTokens": _TURN_HISTORY_TOKEN_BUDGET,
                "responseLen": len(str(out.get("response") or "")),
            },
        )
        return out
    except HTTPException:
        _record_call(
            "conversation.turn",
            payload if isinstance(payload, dict) else None,
            ok=False,
            info={"conversationId": cid or None},
            error="HTTPException",
        )
        raise
    except Exception as exc:
        _record_call(
            "conversation.turn",
            payload if isinstance(payload, dict) else None,
            ok=False,
            info={"conversationId": cid or None},
            error=f"{type(exc).__name__}: {exc}",
        )
        raise HTTPException(status_code=500, detail="Internal Server Error. Check server logs.") from exc


@app.post("/conversation/{conversation_id}/turn/undo", operation_id="conversation_turn_undo")
async def conversation_turn_undo(
    conversation_id: str,
    payload: dict[str, Any] = Body(...),
):
    cid = str(conversation_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="conversation_id is required")

    safe = _safe_payload(payload if isinstance(payload, dict) else {})
    scope = safe.get("user")
    if not isinstance(scope, dict):
        scope = _extract_scope(safe) or None
    if not isinstance(scope, dict):
        raise HTTPException(status_code=400, detail="user scope required")

    uid = str(scope.get("user_id") or "").strip()
    soul_id = str(scope.get("soul_id") or "").strip()
    if not uid or not soul_id:
        raise HTTPException(status_code=400, detail="user_id and soul_id required")

    state_lock = _get_memorize_lock(_memorize_lock_key(uid, soul_id))
    async with state_lock:
        state_row, _, _ = _load_turn_state_and_soul_card(cid, user_id=uid, soul_id=soul_id)
        snapshot = state_row.get("undo_snapshot")
        if not isinstance(snapshot, dict):
            return {"status": "no_snapshot"}

        _write_conversation_state(
            cid,
            soul_id=soul_id,
            user_id=uid,
            updates={
                "memory_cache": list(snapshot.get("memory_cache") or []),
                "intentions_active": snapshot.get("intentions_active"),
                "undo_snapshot": None,
            },
        )
    return {"status": "restored"}


_has_mcp = False
try:
    from fastapi_mcp import FastApiMCP

    mcp = FastApiMCP(app)
    http_path = str(_CONFIG.get("mcp", {}).get("http_path") or "/mcp")
    sse_path = str(_CONFIG.get("mcp", {}).get("sse_path") or "/sse")
    mcp.mount(http_path=http_path, sse_path=sse_path)
    _has_mcp = True
except Exception:
    _has_mcp = False


try:
    _BUNDLE_ROOT = Path(__file__).resolve().parents[2]
    _UI_DIST = _BUNDLE_ROOT / "memu-ui" / "dist"
    if _UI_DIST.exists():
        app.mount("/", StaticFiles(directory=str(_UI_DIST), html=True), name="ui")
except Exception:
    pass
