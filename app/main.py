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
import warnings
from collections.abc import Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

try:
    import pwd
except ImportError:  # pragma: no cover
    pwd = None  # type: ignore

from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from memu.app import MemoryService
from pydantic import BaseModel

from app import procedural as _procedural
from app.config import (
    STARTUP_WARNINGS as _STARTUP_WARNINGS,
    STORAGE_STATUS as _STORAGE_STATUS,
    blob_config_from_cfg as _blob_config_from_cfg,
    categories_from_cfg as _categories_from_cfg,
    config_path as _config_path,
    database_config_from_cfg as _database_config_from_cfg,
    default_llm_profiles_from_server_config as _default_llm_profiles_from_server_config,
    ensure_storage_paths as _ensure_storage_paths,
    get_storage_dir as _get_storage_dir,
    is_ephemeral_db as _is_ephemeral_db,
    load_config as _load_config,
    load_soul_gen_config as _load_soul_gen_config,
    mask_config as _mask_config,
    normalize_sqlite_dsn as _normalize_sqlite_dsn,
    procedural_db_path as _procedural_db_path,
    procedural_should_ingest as _procedural_should_ingest,
    procedural_yaml_dir as _procedural_yaml_dir,
    sanitize_db_filename as _sanitize_db_filename,
    save_config as _save_config,
    save_soul_gen_config as _save_soul_gen_config,
    sqlite_dir_from_cfg as _sqlite_dir_from_cfg,
    sqlite_dsn_for_scope as _sqlite_dsn_for_scope,
    sqlite_file_from_dsn as _sqlite_file_from_dsn,
)
from app.db import (
    json_from_db as _json_from_db,
    json_to_db as _json_to_db,
    normalize_text_list as _normalize_text_list,
    sqlite_connect as _sqlite_connect,
    sqlite_ensure_conversation_state_schema as _sqlite_ensure_conversation_state_schema,
    sqlite_ensure_nonempty as _sqlite_ensure_nonempty,
    sqlite_pragmas as _sqlite_pragmas,
    sqlite_table_columns as _sqlite_table_columns,
)
from app.services.consolidation import (
    ConsolidationDeps,
    gather_consolidation_inputs as _gather_consolidation_inputs,
    run_consolidation_llm as _run_consolidation_llm,
    write_consolidation_outputs as _write_consolidation_outputs,
)
from app.services.graph_edges import (
    invalidate_memory_edges as _invalidate_memory_edges,
    write_memory_edges as _write_memory_edges,
)
from app.services.intention_state import (
    append_memory_cache_entry as _append_memory_cache_entry,
    apply_intention_action as _apply_intention_action,
    apply_intention_turn_maintenance as _apply_intention_turn_maintenance_impl,
    format_intentions_for_prompt as _format_intentions_for_prompt,
    normalize_intentions_stack as _normalize_intentions_stack_impl,
    normalize_memory_cache as _normalize_memory_cache_impl,
    remove_intentions as _remove_intentions,
)
from app.services import crud_endpoints as _crud_endpoints
from app.services import mcp_tools as _mcp_tools
from app.services import soul_state as _soul_state
from app.services.narrative_self import snapshot_previous_narrative_self
from app.services import memorize_endpoint as _memorize_endpoint
from app.services import sqlite_scope as _sqlite_scope
from app.services.state import (
    conversation_state_empty as _conversation_state_empty,
    conversation_state_from_row as _conversation_state_from_row_impl,
    conversation_state_row as _conversation_state_row,
    find_conversation_state_across_dbs as _find_conversation_state_across_dbs_impl,
    write_conversation_state as _write_conversation_state_impl,
)
from app.services.turn_contract import (
    LIFE_GOALS_FREE_WILL_HEADER as _LIFE_GOALS_FREE_WILL_HEADER,
    _format_item_suffix as _format_turn_item_suffix,
    build_turn_prompt as _build_turn_prompt,
    format_shaped_by_line as _format_shaped_by_line,
    make_turn_system_prompt as _make_turn_system_prompt,
    parse_turn_contract as _parse_turn_contract,
)


# ==== Module state & constants ====

logger = logging.getLogger(__name__)
_PROMPT_LOGGER = logging.getLogger("uvicorn.error")

app = FastAPI(title="mcp-memu-server", version="0.4.0")

_BUILD_ID: str = "fix48.debloat.bloatRemoval.concepts"
_SLEEP_SPLIT_MIN_LULL_SECONDS: int = 3 * 60 * 60
_DEFAULT_MIN_CHUNK_TOKENS: int = 4000
_DEFAULT_EPISODES_PER_SEGMENT: int = 3
_DEFAULT_HISTORY_TAIL_AFTER_MEMORIZE: int = 3000
_MIN_CHUNK_TOKENS: int = _DEFAULT_MIN_CHUNK_TOKENS
_EPISODES_PER_SEGMENT: int = _DEFAULT_EPISODES_PER_SEGMENT
_HISTORY_TAIL_AFTER_MEMORIZE: int = _DEFAULT_HISTORY_TAIL_AFTER_MEMORIZE
# Uniform runaway-protection caps for LLM calls. Not business logic —
# these are floors against pathological generation, not content limits.
# DB-storage limits are enforced separately at the storage write path.
PIPELINE_MAX_TOKENS: int = 4000  # turn/memorize/retrieve/apimw/consolidation
UTILITY_MAX_TOKENS: int = 1000   # holistic_summary/topic_statement/narrative_suggestion
_BACKGROUND_TASKS: set[asyncio.Task] = set()  # prevent GC of fire-and-forget tasks
_LOG_PROMPTS: bool = False
_VALID_INTENTION_STATUSES: set[str] = {"active", "resolved", "adapted", "deferred", "dissolved", "removed"}


# ==== Token estimation & segment planning ====

_estimate_tokens = _memorize_endpoint.estimate_tokens
_estimate_unmemorized_tokens = _memorize_endpoint.estimate_unmemorized_tokens


# ==== Memorize background orchestration ====

async def _run_forced_memorize_from_turn(payload: dict[str, Any]) -> None:
    await _memorize_endpoint.run_forced_memorize_from_turn(
        payload,
        memorize_handler=memorize,
        logger=logger,
        get_memorize_lock=_get_memorize_lock,
        memorize_lock_key=_memorize_lock_key,
        write_conversation_state=_write_conversation_state,
    )


def _has_category_content(c: dict[str, Any]) -> bool:
    summary = str(c.get("summary") or "").strip()
    desc = str(c.get("description") or "").strip()
    return bool(summary or desc)


# ==== Server state (locks, inflight, shutdown) ====

_SERVER_INSTANCE_ID: str = str(uuid.uuid4())
_SERVER_STARTED_AT_UNIX: float = time.time()
_LAST_CALLS: list[dict[str, Any]] = []
_LAST_HTTP: list[dict[str, Any]] = []
_MEMORIZE_LOCKS: dict[str, asyncio.Lock] = {}
_MEMORIZE_PROGRESS: dict[str, dict[str, Any]] = {}
_MEMORIZE_CANCEL: set[str] = set()


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
    except OSError:
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
    except OSError:
        os._exit(0)

    _SHUTDOWN_TASK = None


# ==== HTTP middleware ====

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
            logger.debug("request trace append failed", exc_info=True)


# ==== Config loading & storage paths ====


class STUserModel(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


_CONFIG: dict[str, Any] = _load_config()

def _refresh_runtime_limits() -> None:
    global _MIN_CHUNK_TOKENS, _EPISODES_PER_SEGMENT, _HISTORY_TAIL_AFTER_MEMORIZE
    global _LOG_PROMPTS
    memorize_cfg = _CONFIG.get("memorize") if isinstance(_CONFIG.get("memorize"), dict) else {}
    try:
        _MIN_CHUNK_TOKENS = max(0, int(memorize_cfg.get("min_chunk_tokens", _DEFAULT_MIN_CHUNK_TOKENS)))
    except (TypeError, ValueError, OverflowError):
        _MIN_CHUNK_TOKENS = _DEFAULT_MIN_CHUNK_TOKENS
    try:
        _EPISODES_PER_SEGMENT = max(1, int(memorize_cfg.get("episodes_per_segment", _DEFAULT_EPISODES_PER_SEGMENT)))
    except (TypeError, ValueError, OverflowError):
        _EPISODES_PER_SEGMENT = _DEFAULT_EPISODES_PER_SEGMENT
    try:
        _HISTORY_TAIL_AFTER_MEMORIZE = max(
            1,
            int(memorize_cfg.get("history_tail_after_memorize", _DEFAULT_HISTORY_TAIL_AFTER_MEMORIZE)),
        )
    except (TypeError, ValueError, OverflowError):
        _HISTORY_TAIL_AFTER_MEMORIZE = _DEFAULT_HISTORY_TAIL_AFTER_MEMORIZE
    debug_cfg = _CONFIG.get("debug") if isinstance(_CONFIG.get("debug"), dict) else {}
    _LOG_PROMPTS = bool(debug_cfg.get("log_prompts", False))


def _prompt_log_before(ctx: Any, request_view: Any) -> None:
    import time as _time
    ctx._llm_start = _time.monotonic()
    if getattr(request_view, "kind", None) == "embed":
        return
    op = getattr(ctx, "operation", None) or "-"
    step = getattr(ctx, "step_id", None) or "-"
    banner = f"===== {op.upper()} · {step} ".ljust(70, "=")
    meta = getattr(request_view, "metadata", None) or {}
    sys_prompt = meta.get("system_prompt", "")
    user_content = getattr(request_view, "content", None) or ""
    prompt_text = f"== SYSTEM ==\n{sys_prompt}\n\n== USER ==\n{user_content}" if sys_prompt else user_content
    _PROMPT_LOGGER.info(
        "\n\n\n%s\n\n[PROMPT] op=%s step=%s model=%s\n%s\n\n",
        banner,
        op,
        step,
        getattr(ctx, "model", None) or "-",
        prompt_text,
    )


def _prompt_log_after(ctx: Any, request_view: Any, response_view: Any, usage: Any) -> None:
    import time as _time
    elapsed = _time.monotonic() - getattr(ctx, "_llm_start", _time.monotonic())
    content = getattr(response_view, "content", None)
    kind = getattr(request_view, "kind", None)
    if kind == "embed":
        logger.info("[EMBED] elapsed=%.1fs", elapsed)
        return
    if not content:
        return
    finish_reason = getattr(usage, "finish_reason", None)
    in_tok = getattr(usage, "input_tokens", None)
    out_tok = getattr(usage, "output_tokens", None)
    total_tok = getattr(usage, "total_tokens", None)
    _PROMPT_LOGGER.info(
        "[RESPONSE] op=%s step=%s elapsed=%.1fs finish_reason=%s tokens=in:%s/out:%s/total:%s content_chars=%s\n\n%s",
        getattr(ctx, "operation", None) or "-",
        getattr(ctx, "step_id", None) or "-",
        elapsed,
        finish_reason,
        in_tok,
        out_tok,
        total_tok,
        len(content),
        content,
    )


_refresh_runtime_limits()

# Also expose diagnostics under the MCP http_path (e.g. /mcp/diag) to avoid path confusion.
_DIAG_PREFIX: str = str(_CONFIG.get("mcp", {}).get("http_path") or "/mcp").rstrip("/")
if _DIAG_PREFIX == "":
    _DIAG_PREFIX = "/mcp"


def _resolve_profile(svc: Any, name: str | None) -> str | None:
    """Return *name* if the service has that profile, else fall back to None (default)."""
    if not name:
        return None
    if hasattr(svc, "llm_profiles") and hasattr(svc.llm_profiles, "profiles"):
        if name in svc.llm_profiles.profiles:
            return name
    return None


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
        logger.debug("service close failed", exc_info=True)


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
    except OSError:
        logger.debug("service storage fingerprint stat failed for %s", p, exc_info=True)

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
    return "rag" if method == "rag" else default


def _retrieve_method_from_cfg(cfg: Mapping[str, Any] | None) -> str:
    # Single-path retrieve: always rag.
    if not isinstance(cfg, Mapping):
        return "rag"
    retrieve = cfg.get("retrieve")
    if not isinstance(retrieve, Mapping):
        return "rag"
    return _normalize_retrieve_method(retrieve.get("method"), "rag")


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
    retrieve_method_override: str | None = None,
) -> MemoryService:
    service_key_raw = _derive_service_key(payload)

    # Profile pool: server defaults from config.json are always loaded; client-supplied
    # profiles merge on top per-key. Clients can send nothing, one override, or the full
    # set. Same path for ST plugin, MCP clients, PicoClaw, test tools.
    client_profiles = payload.get("llm_profiles") if isinstance(payload.get("llm_profiles"), dict) else {}
    llm_profiles = {**_default_llm_profiles_from_server_config(_CONFIG), **client_profiles}
    payload["llm_profiles"] = llm_profiles
    database_config = payload.get("database_config")
    blob_config = payload.get("blob_config")

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
        if fixed_cats:
            memorize_config["memory_categories"] = fixed_cats
        memorize_config["dynamic_category_cluster_size"] = int(cats_cfg.get("dynamic_category_cluster_size", 3) or 3)
        memorize_config["max_categories_total"] = int((cats_cfg.get("max_total", 12)) or 0)
        memorize_config["episodes_per_segment"] = _EPISODES_PER_SEGMENT
        step_models = (_CONFIG.get("llm", {}) if isinstance(_CONFIG.get("llm"), dict) else {}).get("step_models", {})
        if isinstance(step_models, dict):
            for cfg_key, profile_field in (
                ("preprocess", "preprocess_llm_profile"),
                ("memory_extract", "memory_extract_llm_profile"),
                ("category_update", "category_update_llm_profile"),
            ):
                if profile_field not in memorize_config and str(step_models.get(cfg_key) or "").strip():
                    memorize_config[profile_field] = cfg_key
    retrieve_config = payload.get("retrieve_config")
    if not isinstance(retrieve_config, dict):
        retrieve_config = {}
        payload["retrieve_config"] = retrieve_config
    retrieve_config["method"] = _normalize_retrieve_method(
        retrieve_method_override,
        _retrieve_method_from_cfg(_CONFIG),
    )
    step_models_r = (_CONFIG.get("llm", {}) if isinstance(_CONFIG.get("llm"), dict) else {}).get("step_models", {})
    if isinstance(step_models_r, dict):
        for cfg_key, profile_field in (
            ("reflection", "sufficiency_check_llm_profile"),
            ("ranking", "llm_ranking_llm_profile"),
        ):
            if profile_field not in retrieve_config and str(step_models_r.get(cfg_key) or "").strip():
                retrieve_config[profile_field] = cfg_key
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
            oldest_key = next(iter(_SERVICES), None)
            if oldest_key is not None:
                _SERVICES.pop(oldest_key, None)
                _SERVICE_STORAGE_FP.pop(oldest_key, None)

        _SERVICES[service_key] = svc
        _SERVICE_STORAGE_FP[service_key] = storage_fp
        return svc


# ==== Payload & scope extraction ====

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
            except (ValueError, OverflowError):
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
        logger.debug("record_call telemetry append failed", exc_info=True)


# ==== SQLite scope helpers ====

def _sqlite_current_path(
    user_id: str | None = None,
    soul_id: str | None = None,
) -> Path | None:
    return _sqlite_scope.sqlite_current_path(
        user_id=user_id,
        soul_id=soul_id,
        storage_status=_STORAGE_STATUS,
        config=_CONFIG,
        sqlite_dsn_for_scope=_sqlite_dsn_for_scope,
        sqlite_file_from_dsn=_sqlite_file_from_dsn,
    )


def _sqlite_build_scope_where(
    cols: list[str],
    user_id: str | None,
    soul_id: str | None,
    conversation_id: str | None,
) -> tuple[str, list[Any]]:
    return _sqlite_scope.sqlite_build_scope_where(
        cols,
        user_id,
        soul_id,
        conversation_id,
    )


def _sqlite_file_info(p: Path) -> dict[str, Any]:
    return _sqlite_scope.sqlite_file_info(p)


def _conversation_state_from_row(row: sqlite3.Row | None, *, con: sqlite3.Connection | None = None) -> dict[str, Any] | None:
    state = _conversation_state_from_row_impl(row)
    if state is not None and con is not None:
        state.update(_soul_state.read(con))
    return state


def _intention_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return _sqlite_scope.intention_row_to_dict(
        row,
        normalize_text_list=_normalize_text_list,
    )


def _write_conversation_state(
    conversation_id: str,
    *,
    soul_id: str | None = None,
    user_id: str | None = None,
    updates: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    return _sqlite_scope.write_conversation_state(
        conversation_id,
        soul_id=soul_id,
        user_id=user_id,
        updates=updates,
        config=_CONFIG,
        storage_status=_STORAGE_STATUS,
        sqlite_dir_from_cfg=_sqlite_dir_from_cfg,
        write_conversation_state_impl=_write_conversation_state_impl,
        sqlite_current_path=_sqlite_current_path,
    )


def _find_conversation_state_across_dbs(conversation_id: str) -> tuple[Path | None, dict[str, Any] | None]:
    return _sqlite_scope.find_conversation_state_across_dbs(
        conversation_id,
        config=_CONFIG,
        storage_status=_STORAGE_STATUS,
        sqlite_dir_from_cfg=_sqlite_dir_from_cfg,
        find_conversation_state_across_dbs_impl=_find_conversation_state_across_dbs_impl,
    )


# ==== Retrieve payload helpers ====

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
            parsed_num = float(s)
        except (TypeError, ValueError, OverflowError):
            parsed_num = None
        if parsed_num is not None:
            return int(parsed_num)
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except (ValueError, OverflowError):
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
    # With stop_at_message_id set, the returned slice must never include or
    # extend past that message — the two slices (previous / current episode)
    # must not overlap.
    if not history or limit <= 0:
        return []
    stop_id = str(stop_at_message_id or "").strip()
    if not chat_x_anchors:
        return [] if stop_id else history[-limit:]
    anchors = _compact_chat_x_anchors(*chat_x_anchors, max_count=2)
    if not anchors:
        return [] if stop_id else history[-limit:]
    anchor_set = set(anchors)
    start_idx = None
    for idx, item in enumerate(history):
        if str(item.get("message_id") or "").strip() in anchor_set:
            start_idx = idx
            break
    if start_idx is None:
        return [] if stop_id else history[-limit:]
    stop_idx: int | None = None
    if stop_id:
        for idx, item in enumerate(history):
            if str(item.get("message_id") or "").strip() == stop_id:
                stop_idx = idx
                break
        if stop_idx is None or stop_idx <= start_idx:
            return []
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


# ==== Turn prompt context builders ====

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
    db_path = _sqlite_current_path(user_id, soul_id)
    state_row: dict[str, Any] | None = None
    soul = _soul_state.defaults()
    if db_path is not None and db_path.exists():
        con = _sqlite_connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            _sqlite_ensure_conversation_state_schema(con)
            state_row = _conversation_state_from_row(_conversation_state_row(con, conversation_id))
            soul = _soul_state.read(con)
        finally:
            con.close()
    if state_row is None:
        state_row = _conversation_state_empty(conversation_id, soul_id=soul_id, user_id=user_id)
    state_row.update(soul)
    narrative = str(soul.get("narrative_self") or "").strip()
    soul_card = narrative or None
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
FROM intentions
WHERE soul_id = ? AND user_id = ? AND source = 'life_goal' AND status = 'active'
ORDER BY updated_at ASC, id ASC
""",
            (soul_id, user_id),
        ).fetchall()
        return [str(row[0] or "").strip() for row in rows if str(row[0] or "").strip()]
    finally:
        con.close()


def _validate_relationship_speaker_id(raw: Any) -> str:
    return _crud_endpoints._validate_relationship_speaker_id(raw)


def _relationship_item_from_values(
    *,
    normalized: str,
    name: str,
    entity_type: str,
    properties: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    return _crud_endpoints._relationship_item_from_values(
        normalized=normalized,
        name=name,
        entity_type=entity_type,
        properties=properties,
    )


def _assert_user_declared_relationship(props: Mapping[str, Any] | None) -> None:
    _crud_endpoints._assert_user_declared_relationship(props)


# ==== Memorize execution helpers ====

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
    soul_label = str(scope.get("soul_id") or "").strip()
    soul_slug = re.sub(r"[^a-z0-9]+", "_", soul_label.lower()).strip("_") or "soul"
    created_ids: list[str] = []
    for idx, summary in enumerate(summaries):
        if idx >= len(embeddings):
            break
        item = svc.database.memory_item_repo.create_item(
            resource_id=None,
            memory_type="reflection",
            summary=summary,
            embedding=embeddings[idx],
            user_data=scope,
            source_role="soul",
            speaker_id=f"soul:{soul_slug}",
            speaker_label=soul_label or "soul",
            confidence=1.0,
            happened_at=datetime.now(UTC),
            reflection_salience=saliences[idx],
            conversation_id=conversation_id,
        )
        created_ids.append(str(item.id))
    return created_ids


def _merge_memorize_segment_results(
    segment_results: list[dict[str, Any]],
    pending_episode_ids: list[str] | None = None,
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
                except (TypeError, ValueError):
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

    for batch_result in segment_results:
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
        "results": segment_results,
        "segment_count": len(segment_results),
        "items": _merge_record_list(flat_items),
        "categories": _merge_record_list(flat_categories, id_keys=("id", "name")),
        "relations": _merge_record_list(flat_relations, id_keys=("item_id", "category_id")),
        "pending_episode_ids": _normalize_text_list(pending_episode_ids),
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
    system_prompt = (
        "Write a 250-350 word overview of your life from the summaries below, in your own voice."
    )
    result = await svc.chat(
        full_text,
        system_prompt=system_prompt,
        temperature=0.3,
        max_tokens=UTILITY_MAX_TOKENS,
        op="categories",
        step="holistic_summary",
    )
    return str(result or "").strip() or None


# ==== Retrieve & APIMW pipeline ====

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
    svc = _get_service_from_payload(safe, retrieve_method_override=method)
    scope = _extract_retrieve_where(safe)
    memu_queries = _extract_retrieve_queries(safe)
    as_of = _parse_as_of_datetime(safe.get("as_of"))

    soul_id = str((scope or {}).get("soul_id") or "").strip()
    user_id = str((scope or {}).get("user_id") or "user").strip() or "user"

    # Prompt-diversity: read the rotating rewrite angle so the router
    # uses a different lens (topic / relation / counterpoint-hint) on
    # consecutive RETRIEVE turns.
    retrieve_rewrite_angle = 0
    if scoped_conversation_id and soul_id:
        db_path = _sqlite_current_path(user_id or None, soul_id)
        if db_path is not None and db_path.exists():
            con = _sqlite_connect(db_path)
            try:
                con.row_factory = sqlite3.Row
                _sqlite_ensure_conversation_state_schema(con)
                pre_row = _conversation_state_from_row(_conversation_state_row(con, scoped_conversation_id), con=con)
                if pre_row:
                    retrieve_rewrite_angle = int(pre_row.get("retrieve_rewrite_angle") or 0)
            finally:
                con.close()

    retrieve_started_at = time.monotonic()
    retrieve_result = await svc.retrieve(
        memu_queries,
        where=scope,
        as_of=as_of,
        rewrite_angle=retrieve_rewrite_angle,
    )
    retrieve_ms = int((time.monotonic() - retrieve_started_at) * 1000)
    out: dict[str, Any] = {
        "ok": True,
        "result": retrieve_result,
        "retrieve_ms": retrieve_ms,
    }

    # Mental-health procedural sidecar. The router gated this by writing
    # mental_health_query only when the turn touches such a theme — if it
    # stayed empty, nothing fires.
    if bool(safe.get("mental_health_addon")) and isinstance(out.get("result"), dict):
        mh_query = str(out["result"].get("mental_health_query") or "").strip()
        if mh_query:
            yaml_dir = _procedural_yaml_dir(_CONFIG)
            procedural_db = _procedural_db_path(_CONFIG)
            if _procedural_should_ingest(procedural_db, yaml_dir):
                try:
                    await _procedural.ingest(svc, procedural_db, yaml_dir)
                except Exception:
                    logger.exception("procedural ingest failed")
            try:
                embeds = await svc.embed([mh_query], profile="embedding")
                qvec = list(embeds[0]) if embeds else []
                expected_embedding_model = str((_CONFIG.get("llm") or {}).get("embed_model") or "").strip() or None
                hits = _procedural.lookup(
                    procedural_db,
                    domain="mental_health",
                    query_vec=qvec,
                    expected_embedding_model=expected_embedding_model,
                    limit=1,
                )
                if hits:
                    row, score = hits[0]
                    out["result"].setdefault("items", []).append({
                        **row,
                        "memory_type": "procedural",
                        "score": float(score),
                    })
            except Exception:
                logger.exception("procedural lookup failed")

    # Advance the angle only on RETRIEVE verdicts (NO_RETRIEVE keeps it put).
    if (
        scoped_conversation_id
        and soul_id
        and isinstance(out.get("result"), dict)
        and out["result"].get("needs_retrieval")
    ):
        _write_conversation_state(
            scoped_conversation_id,
            soul_id=soul_id,
            user_id=user_id,
            updates={"retrieve_rewrite_angle": (retrieve_rewrite_angle + 1) % 3},
        )

    if scoped_conversation_id:
        state_out: dict[str, Any] | None = None
        if soul_id:
            db_path = _sqlite_current_path(user_id or None, soul_id)
            if db_path is not None and db_path.exists():
                con = _sqlite_connect(db_path)
                try:
                    con.row_factory = sqlite3.Row
                    _sqlite_ensure_conversation_state_schema(con)
                    state_out = _conversation_state_from_row(_conversation_state_row(con, scoped_conversation_id), con=con)
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
    logger.info("apimw step A: topic statement for %s", conversation_id)
    topic_system = (
        f"{identity_context}\n\n"
        "State the topic of the CURRENT episode in 1-2 sentences. The previous episode is provided only as context — "
        "if the current episode is brief, use it to understand what the new message means, but describe only where the conversation is now."
    )
    topic_statement = await svc.chat(
        topic_user,
        system_prompt=topic_system,
        temperature=0.1,
        max_tokens=UTILITY_MAX_TOKENS,
        op="apimw",
        step="topic_statement",
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
    retrieve_queries = _build_retrieve_soul_context_queries(
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
        "queries": retrieve_queries,
        "conversation_id": conversation_id,
    }
    retrieve_payload["retrieve_config"] = _build_apimw_retrieve_config(
        retrieve_payload.get("retrieve_config"),
        item_top_k=apimw_k,
    )
    logger.info("apimw retrieve for %s", conversation_id)
    retrieve_out = await _run_retrieve(retrieve_payload, conversation_id=conversation_id)
    retrieve_result_data = retrieve_out.get("result") or {}
    retrieved_items = [item for item in (retrieve_result_data.get("items") or []) if isinstance(item, dict)]
    logger.info("apimw retrieved %d items for %s", len(retrieved_items), conversation_id)
    return retrieve_result_data, retrieved_items


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
    first_pass_result, first_pass_items = await _apimw_retrieve_pass(
        payload,
        query_text=topic_statement,
        soul_id=soul_id,
        history=history,
        state_row=state_row,
        conversation_id=conversation_id,
        apimw_k=apimw_k,
    )

    second_pass_items: list[dict[str, Any]] = []
    second_query = str(first_pass_result.get("next_step_query") or "").strip()
    if second_query and _norm_result_sig(second_query) != _norm_result_sig(topic_statement):
        _second_pass_result, second_pass_items = await _apimw_retrieve_pass(
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
    for item in first_pass_items + second_pass_items:
        sig = _item_sig(item)
        if not sig or sig in seen_item_sigs:
            continue
        seen_item_sigs.add(sig)
        combined_items.append(item)

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
            }
            sig = _item_sig(row)
            if not sig or sig in seen_item_sigs:
                continue
            candidates.append(row)
        if candidates:
            sample_size = min(apimw_random_count, len(candidates))
            for candidate_item in random.sample(candidates, sample_size):
                sig = _item_sig(candidate_item)
                if not sig or sig in seen_item_sigs:
                    continue
                seen_item_sigs.add(sig)
                combined_items.append(candidate_item)

    logger.info("apimw combined pool %d items for %s", len(combined_items), conversation_id)
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
    llm_profile: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    logger.info("apimw step D+E+F: combined call for %s", conversation_id)
    formatted_memory_lines: list[str] = []
    items_by_id: dict[str, dict[str, Any]] = {}
    for item in combined_items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        summary = str(item.get("summary") or "").strip()
        if not item_id or not summary:
            continue
        items_by_id[item_id] = item
        memory_type = str(item.get("memory_type") or "memory").strip()
        suffix = _format_turn_item_suffix(item)
        formatted_memory_lines.append(f"[{item_id}] [{memory_type}]{suffix} {summary}")
        shaped_by = item.get("shaped_by")
        if isinstance(shaped_by, dict):
            formatted_memory_lines.append(_format_shaped_by_line(shaped_by, with_id=True))

    formatted_memories = "\n".join(formatted_memory_lines) if formatted_memory_lines else "(none)"

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

    apimw_system_prompt = (
        f"{identity_context}\n\n"
        "This is your subconscious — a background process that runs between your turns. "
        "You have just searched your long-term memory. Below are your summaries, your individual memories, "
        "your working thoughts, your intentions, and your life goals. "
        "Your working thoughts, intentions, and life goals are READ-ONLY here — you cannot change them from the subconscious. "
        "You can only observe them and let them inform your choices. "
        "The soul also has the full conversation history as context — you are adding depth, not providing it.\n\n"
        "Review everything. Then return STRICT JSON (first character { , last character } , "
        "no markdown fences, no text outside JSON).\n\n"
        "Required top-level keys:\n"
        "- prior_context: array of memory IDs (strings) you want surfaced as background context "
        "next time you speak. Pick what matters — not everything. Order by importance.\n"
        "- message_to_self: string or null — a brief thought you want to surface in your working memory for one turn. "
        "Use this when something you noticed in the background feels important enough to bring to your own attention next time you speak. One sentence."
    )

    apimw_user_prompt = (
        f"Summaries:\n{formatted_categories}\n\n"
        f"Individual memories:\n{formatted_memories}\n\n"
        f"Your working thoughts:\n{formatted_cache}\n\n"
        f"Intentions:\n{formatted_intentions}\n\n"
        f"{_LIFE_GOALS_FREE_WILL_HEADER}\n{formatted_life_goals}\n\n"
        f"Recent conversation:\n{episode_text}"
    )

    llm_raw = await svc.chat(
        apimw_user_prompt,
        profile=llm_profile,
        system_prompt=apimw_system_prompt,
        temperature=0.2,
        max_tokens=PIPELINE_MAX_TOKENS,
        response_format={"type": "json_object"},
        op="apimw",
        step="def_call",
    )

    apimw_response_text = str(llm_raw or "").strip()
    try:
        result_json = json.loads(apimw_response_text)
    except json.JSONDecodeError:
        logger.error("apimw D+E+F: JSON parse failed, raw=%s", apimw_response_text[:200])
        return None, items_by_id
    if not isinstance(result_json, dict):
        logger.error("apimw D+E+F: expected dict, got %s", type(result_json).__name__)
        return None, items_by_id

    logger.info("apimw D+E+F: parsed JSON with keys %s for %s", list(result_json.keys()), conversation_id)
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

        prior_context_ids_raw = result_json.get("prior_context") or []
        if isinstance(prior_context_ids_raw, list) and prior_context_ids_raw:
            prior_context_lines: list[str] = []
            for raw_memory_id in prior_context_ids_raw:
                memory_id = str(raw_memory_id).strip()
                item = items_by_id.get(memory_id)
                if not item:
                    continue
                memory_type = str(item.get("memory_type") or "memory").strip()
                suffix = _format_turn_item_suffix(item)
                summary = str(item.get("summary") or "").strip()
                prior_context_lines.append(f"[{memory_type}]{suffix} {summary}")
                shaped_by = item.get("shaped_by")
                if isinstance(shaped_by, dict):
                    prior_context_lines.append(_format_shaped_by_line(shaped_by))
            if prior_context_lines:
                new_prior = "\n".join(prior_context_lines)
                existing_prior = str(fresh_row.get("prior_context") or "").strip()
                if existing_prior and existing_prior != new_prior:
                    logger.warning(
                        "apimw: overwriting non-empty prior_context for %s (may indicate concurrent turn write)",
                        conversation_id,
                    )
                updates["prior_context"] = new_prior

        updates["last_retrieval_ids"] = _extract_result_item_ids({"items": combined_items})

        message_to_self = str(result_json.get("message_to_self") or "").strip()
        if message_to_self:
            updates["subconscious_message"] = f"[subconscious] {message_to_self[:300]}"
            try:
                sc_text = message_to_self[:300]
                sc_embedding = (await svc.embed([sc_text], profile="embedding"))[0]
                svc.database.memory_item_repo.create_item(
                    resource_id=None,
                    memory_type="subconscious",
                    source_role="soul",
                    summary=sc_text,
                    embedding=sc_embedding,
                    user_data={"user_id": user_id, "soul_id": soul_id, "conversation_id": conversation_id},
                    conversation_id=conversation_id,
                )
            except Exception:
                logger.warning("failed to persist subconscious memory item", exc_info=True)

        prior_context_ids = (
            [str(mid).strip() for mid in prior_context_ids_raw if isinstance(mid, str) and str(mid).strip()]
            if isinstance(prior_context_ids_raw, list)
            else []
        )
        if prior_context_ids:
            updates["append_prior_context_ids_since_consolidation"] = prior_context_ids

        if updates:
            _write_conversation_state(
                conversation_id,
                soul_id=soul_id,
                user_id=user_id,
                updates=updates,
            )
            logger.info("apimw state written for %s (keys: %s)", conversation_id, list(updates.keys()))


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
        apimw_item_top_k = _apimw_memory_count_from_cfg(_CONFIG)
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
            apimw_k=apimw_item_top_k,
            apimw_random_count=apimw_random_count,
            scope=scope,
        )

        apimw_heavy_profile = _resolve_profile(svc, "memory_extract")
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
            llm_profile=apimw_heavy_profile,
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



# =============================================================================
# HTTP ENDPOINTS
# =============================================================================


# ---- Health, version, diag, shutdown ----

@app.get("/health", operation_id="health")
async def health():
    storage_dir = _get_storage_dir(_CONFIG)
    uid = getattr(os, "geteuid", lambda: None)()
    gid = getattr(os, "getegid", lambda: None)()
    user = None
    try:
        if uid is not None and pwd is not None:
            user = pwd.getpwuid(uid).pw_name
    except (KeyError, OSError):
        logger.debug("health user lookup failed for uid=%s", uid, exc_info=True)
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
    # Body fields: requested_by, reason, max_wait_sec (0 = wait forever).
    global _SHUTDOWN_TASK

    body = payload if isinstance(payload, dict) else {}
    requested_by = str(body.get("requested_by") or body.get("requestedBy") or "").strip() or None
    reason = str(body.get("reason") or "").strip() or None

    max_wait_raw = body.get("max_wait_sec", body.get("maxWaitSec", 0))
    try:
        max_wait_sec = int(max_wait_raw)
    except (TypeError, ValueError, OverflowError):
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
<li><a href='/diag/sqlite/recent?table=memory_items&limit=10'>/diag/sqlite/recent</a> <small>(add scope params)</small></li>
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
        "resources",
        "categories",
        "memory_items",
        "category_items",
        "entities",
        "triples",
        "conversations",
        "narrative_history",
        "intentions",
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
    table: str = "memory_items",
    limit: int = 20,
    user_id: str | None = None,
    soul_id: str | None = None,
    conversation_id: str | None = None,
):
    allowed = {
        "resources",
        "categories",
        "memory_items",
        "category_items",
        "entities",
        "triples",
        "conversations",
        "narrative_history",
        "intentions",
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
            "pending_episode_ids",
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


# ---- Config endpoints ----

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
    except OSError:
        logger.debug("root UI index lookup failed", exc_info=True)
    return {"message": "mcp-memu-server", "mcp": "enabled" if _has_mcp else "disabled"}


# ---- Chat storage & sleep-gap helpers ----

def _read_list(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    raw = p.read_text(encoding="utf-8")
    obj = json.loads(raw) if raw.strip() else []
    return [m for m in obj if isinstance(m, dict)] if isinstance(obj, list) else []


def _resolve_chat_storage_dir(
    chats_dir: Path,
    uid: str,
    aid: str,
    conversation_id: str | None,
    chat_file: str | None,
    resource_url_in: str | None,
) -> tuple[Path, str, str]:
    return _memorize_endpoint.resolve_chat_storage_dir(
        chats_dir,
        uid,
        aid,
        conversation_id,
        chat_file,
        resource_url_in,
        _sanitize_db_filename,
    )


_date_label = _memorize_endpoint.date_label
_split_indices_by_sleep = _memorize_endpoint.split_indices_by_sleep


def _find_chat_dir_for_conversation(chats_dir: Path, uid: str, soul_id: str, conversation_id: str) -> Path | None:
    return _memorize_endpoint.find_chat_dir_for_conversation(
        chats_dir,
        uid,
        soul_id,
        conversation_id,
        _sanitize_db_filename,
    )


def _slice_history_after_last_memorized_segment(
    history: list[dict[str, Any]],
    *,
    chats_dir: Path,
    uid: str,
    soul_id: str,
    conversation_id: str,
) -> list[dict[str, Any]]:
    return _memorize_endpoint.slice_history_after_last_memorized_segment(
        history,
        chats_dir=chats_dir,
        uid=uid,
        soul_id=soul_id,
        conversation_id=conversation_id,
        sanitize_db_filename=_sanitize_db_filename,
    )


def _unmemorized_sleep_gap_detected(
    history: list[dict[str, Any]],
    digest_cursor: Any,
    safe: dict[str, Any],
) -> bool:
    return _memorize_endpoint.unmemorized_sleep_gap_detected(
        history,
        digest_cursor,
        safe,
        logger=logger,
        min_chunk_tokens=_MIN_CHUNK_TOKENS,
        sleep_split_min_lull_seconds=_SLEEP_SPLIT_MIN_LULL_SECONDS,
    )


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
        llm_profile=_resolve_profile(svc, "consolidation"),
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


def _make_memorize_context() -> _memorize_endpoint.MemorizeContext:
    return _memorize_endpoint.MemorizeContext(
        get_memorize_lock=_get_memorize_lock,
        memorize_lock_key=_memorize_lock_key,
        write_conversation_state=_write_conversation_state,
        memorize_progress=_MEMORIZE_PROGRESS,
        memorize_cancel=_MEMORIZE_CANCEL,
        record_call=_record_call,
        logger=logger,
        min_chunk_tokens=_MIN_CHUNK_TOKENS,
        sleep_split_min_lull_seconds=_SLEEP_SPLIT_MIN_LULL_SECONDS,
    )


def _make_memorize_run_context() -> _memorize_endpoint.MemorizeRunContext:
    return _memorize_endpoint.MemorizeRunContext(
        base=_make_memorize_context(),
        load_turn_state_and_soul_card=_load_turn_state_and_soul_card,
        normalize_text_list=_normalize_text_list,
        compute_holistic_categories_summary=_compute_holistic_categories_summary,
        run_consolidation_task=_run_consolidation_task,
        background_tasks_set=_BACKGROUND_TASKS,
    )


def _make_memorize_endpoint_context() -> _memorize_endpoint.MemorizeEndpointContext:
    return _memorize_endpoint.MemorizeEndpointContext(
        base=_make_memorize_context(),
        safe_payload=_safe_payload,
        get_service_from_payload=_get_service_from_payload,
        extract_scope=_extract_scope,
        extract_conversation_id=_extract_conversation_id,
        normalize_conversation=_normalize_conversation,
        pick_str=_pick_str,
        sqlite_current_path=_sqlite_current_path,
        clear_cached_services=_clear_cached_services,
        get_storage_dir=_get_storage_dir,
        run_memorize_episodes=_run_memorize_episodes,
        get_config=lambda: _CONFIG,
        sanitize_db_filename=_sanitize_db_filename,
    )


async def _run_memorize_episodes(
    *,
    memorize_segments: list[tuple[str, list[dict[str, Any]], int]],
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
    sleep_stats: Any,
    episodes_dir: Path,
    zi: Any = None,
) -> None:
    await _memorize_endpoint.run_memorize_episodes(
        memorize_segments=memorize_segments,
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
        merged_len=merged_len,
        force=force,
        sleep_stats=sleep_stats,
        run_ctx=_make_memorize_run_context(),
        episodes_dir=episodes_dir,
        zi=zi,
    )


# ---- Memorize endpoint ----

@app.post("/memorize", operation_id="memorize")
async def memorize(payload: dict[str, Any], background_tasks: BackgroundTasks, force: bool = False):
    return await _memorize_endpoint.memorize_endpoint(
        payload,
        background_tasks,
        force,
        endpoint_ctx=_make_memorize_endpoint_context(),
    )


@app.get("/memorize/progress", operation_id="memorize_progress")
async def memorize_progress(user_id: str = "", soul_id: str = ""):
    return _memorize_endpoint.memorize_progress_endpoint(
        user_id,
        soul_id,
        memorize_lock_key=_memorize_lock_key,
        memorize_progress=_MEMORIZE_PROGRESS,
    )


@app.post("/memorize/cancel", operation_id="memorize_cancel")
async def memorize_cancel(payload: dict[str, Any] = Body(...)):
    return _memorize_endpoint.memorize_cancel_endpoint(
        payload,
        memorize_lock_key=_memorize_lock_key,
        memorize_progress=_MEMORIZE_PROGRESS,
        memorize_cancel=_MEMORIZE_CANCEL,
    )


# ---- Consolidation endpoint ----

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
        safe = _safe_payload(payload)
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
            },
        )
        return {"ok": True, "status": "completed", "result": result}
    except HTTPException as exc:
        if pipeline_started and exc.status_code >= 500:
            await _clear_consolidation_in_progress(
                state_lock=state_lock,
                conversation_id=cid,
                soul_id=soul_id,
                user_id=uid,
            )
        _record_call(
            "consolidation.force", payload, ok=False, error="HTTPException"
        )
        raise
    except Exception as exc:
        logger.exception("consolidation.force failed: %s", exc)
        if pipeline_started:
            await _clear_consolidation_in_progress(
                state_lock=state_lock,
                conversation_id=cid,
                soul_id=soul_id,
                user_id=uid,
            )
        _record_call(
            "consolidation.force",
            payload,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise HTTPException(status_code=500, detail="Internal Server Error. Check server logs.") from exc


# ---- Categories endpoints ----

@app.get("/categories", operation_id="list_memory_categories")
async def list_memory_categories(user_id: str = "", soul_id: str = "", include_empty: bool = False):
    return await _crud_endpoints.list_memory_categories_endpoint(
        user_id=user_id,
        soul_id=soul_id,
        include_empty=include_empty,
        config=_CONFIG,
        default_llm_profiles_from_server_config=_default_llm_profiles_from_server_config,
        get_service_from_payload=_get_service_from_payload,
        has_category_content=_has_category_content,
    )


@app.post("/categories/search", operation_id="search_memory_categories")
async def search_memory_categories(payload: dict[str, Any]):
    """Payload-driven category listing (matches SillyTavern plugin's local mode)."""
    return await _crud_endpoints.search_memory_categories_endpoint(
        payload,
        safe_payload=_safe_payload,
        get_service_from_payload=_get_service_from_payload,
        extract_scope=_extract_scope,
        canonicalize_scope_where=_canonicalize_scope_where,
        has_category_content=_has_category_content,
        record_call=_record_call,
    )


# ---- Intentions & relationships endpoints ----

@app.get("/souls/{soul_id}/intentions", operation_id="list_intentions")
async def list_intentions(
    soul_id: str,
    user_id: str,
    status: str = "active",
):
    return await _crud_endpoints.list_intentions_endpoint(
        soul_id=soul_id,
        user_id=user_id,
        status=status,
        sqlite_current_path=_sqlite_current_path,
        sqlite_connect=_sqlite_connect,
        sqlite_ensure_conversation_state_schema=_sqlite_ensure_conversation_state_schema,
        intention_row_to_dict=_intention_row_to_dict,
    )


@app.get("/souls/{soul_id}/relationships", operation_id="list_relationships")
async def list_relationships(
    soul_id: str,
    user_id: str,
):
    return await _crud_endpoints.list_relationships_endpoint(
        soul_id=soul_id,
        user_id=user_id,
        get_service_from_payload=_get_service_from_payload,
        sqlite_current_path=_sqlite_current_path,
        sqlite_ensure_nonempty=_sqlite_ensure_nonempty,
        json_from_db=_json_from_db,
    )


@app.post("/souls/{soul_id}/relationships", operation_id="create_relationship")
async def create_relationship(
    soul_id: str,
    payload: dict[str, Any] = Body(...),
):
    return await _crud_endpoints.create_relationship_endpoint(
        soul_id=soul_id,
        payload=payload,
        get_service_from_payload=_get_service_from_payload,
        sqlite_current_path=_sqlite_current_path,
        sqlite_ensure_nonempty=_sqlite_ensure_nonempty,
        sqlite_connect=_sqlite_connect,
        json_to_db=_json_to_db,
        json_from_db=_json_from_db,
    )


@app.patch("/souls/{soul_id}/relationships/{speaker_id}", operation_id="update_relationship")
async def update_relationship(
    soul_id: str,
    speaker_id: str,
    payload: dict[str, Any] = Body(...),
):
    return await _crud_endpoints.update_relationship_endpoint(
        soul_id=soul_id,
        speaker_id=speaker_id,
        payload=payload,
        get_service_from_payload=_get_service_from_payload,
        sqlite_current_path=_sqlite_current_path,
        sqlite_ensure_nonempty=_sqlite_ensure_nonempty,
        sqlite_connect=_sqlite_connect,
        json_to_db=_json_to_db,
        json_from_db=_json_from_db,
    )


@app.delete("/souls/{soul_id}/relationships/{speaker_id}", operation_id="delete_relationship")
async def delete_relationship(
    soul_id: str,
    speaker_id: str,
    user_id: str,
):
    return await _crud_endpoints.delete_relationship_endpoint(
        soul_id=soul_id,
        speaker_id=speaker_id,
        user_id=user_id,
        get_service_from_payload=_get_service_from_payload,
        sqlite_current_path=_sqlite_current_path,
        sqlite_ensure_nonempty=_sqlite_ensure_nonempty,
        sqlite_connect=_sqlite_connect,
        json_to_db=_json_to_db,
        json_from_db=_json_from_db,
    )


# ---- Narrative suggestion endpoint ----

@app.post("/souls/{soul_id}/narrative_suggestion", operation_id="narrative_suggestion")
async def narrative_suggestion(soul_id: str, payload: dict[str, Any] = Body(...)):
    return await _crud_endpoints.narrative_suggestion_endpoint(
        soul_id=soul_id,
        payload=payload,
        sqlite_current_path=_sqlite_current_path,
        sqlite_connect=_sqlite_connect,
        sqlite_ensure_conversation_state_schema=_sqlite_ensure_conversation_state_schema,
        sqlite_ensure_nonempty=_sqlite_ensure_nonempty,
        get_service_from_payload=_get_service_from_payload,
        build_retrieve_identity_context=_build_retrieve_identity_context,
        snapshot_previous_narrative_self=snapshot_previous_narrative_self,
        utility_max_tokens=UTILITY_MAX_TOKENS,
    )


@app.patch("/intentions/{intention_id}", operation_id="patch_intention")
async def patch_intention(
    intention_id: str,
    soul_id: str,
    payload: dict[str, Any] | None = Body(default=None),
):
    return await _crud_endpoints.patch_intention_endpoint(
        intention_id=intention_id,
        soul_id=soul_id,
        payload=payload,
        valid_intention_statuses=_VALID_INTENTION_STATUSES,
        sqlite_current_path=_sqlite_current_path,
        sqlite_connect=_sqlite_connect,
        sqlite_ensure_conversation_state_schema=_sqlite_ensure_conversation_state_schema,
        intention_row_to_dict=_intention_row_to_dict,
    )


# ---- Conversation state endpoints ----

@app.get("/conversation/{conversation_id}/state", operation_id="get_conversation_state")
async def get_conversation_state(
    conversation_id: str,
    soul_id: str | None = None,
    user_id: str | None = None,
):
    return await _crud_endpoints.get_conversation_state_endpoint(
        conversation_id=conversation_id,
        soul_id=soul_id,
        user_id=user_id,
        sqlite_current_path=_sqlite_current_path,
        sqlite_connect=_sqlite_connect,
        sqlite_ensure_conversation_state_schema=_sqlite_ensure_conversation_state_schema,
        conversation_state_from_row=_conversation_state_from_row,
        conversation_state_row=_conversation_state_row,
        find_conversation_state_across_dbs=_find_conversation_state_across_dbs,
    )


@app.patch("/conversation/{conversation_id}/state", operation_id="patch_conversation_state")
async def patch_conversation_state(
    conversation_id: str,
    payload: dict[str, Any] | None = Body(default=None),
    soul_id: str | None = None,
    user_id: str | None = None,
):
    return await _crud_endpoints.patch_conversation_state_endpoint(
        conversation_id=conversation_id,
        payload=payload,
        soul_id=soul_id,
        user_id=user_id,
        pick_str=_pick_str,
        write_conversation_state=_write_conversation_state,
    )


# ---- Clear memory endpoint ----

@app.post("/clear", operation_id="clear_memory")
async def clear_memory(payload: dict[str, Any]):
    return await _crud_endpoints.clear_memory_endpoint(
        payload,
        safe_payload=_safe_payload,
        extract_scope=_extract_scope,
        get_service_from_payload=_get_service_from_payload,
        record_call=_record_call,
    )


# ---- Retrieve & timeline endpoints ----

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
            payload,
            ok=False,
            error=str(getattr(he, "detail", he)),
        )
        raise
    except Exception as exc:
        _record_call(
            "retrieve",
            payload,
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
    svc = _get_service_from_payload(safe)

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
        safe = _safe_payload(payload)
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
                # Tail = whatever hasn't been memorized yet. The most recent
                # memorized segment (per manifest.json) marks the boundary;
                # everything before it lives in segment files + category
                # summaries, and the soul reads those through the retrieved
                # memory context. history_tail_after_memorize is the token
                # ceiling applied inside _render_history as a safety cap.
                history = _slice_history_after_last_memorized_segment(
                    history,
                    chats_dir=(_get_storage_dir(_CONFIG) / "st_chats").resolve(),
                    uid=uid,
                    soul_id=soul_id,
                    conversation_id=cid,
                )
                memory_cache = _normalize_memory_cache_impl(out.get("memory_cache"))
                intentions_active = _normalize_intentions_stack_impl(out.get("intentions_active"))

                out["turn_system_prompt"] = _make_turn_system_prompt(
                    soul_id,
                    soul_card=soul_card,
                    response_sentences=int(_CONFIG.get("turn_response_sentences", 3)),
                    include_chat_x=(len(history) > 8 and len(history) % 6 == 0),
                )
                _subconscious_msg = str(_state_row.get("subconscious_message") or "").strip() or None
                out["turn_user_prompt"] = _build_turn_prompt(
                    user_message=message,
                    history=history,
                    history_tail_after_memorize=_HISTORY_TAIL_AFTER_MEMORIZE,
                    prior_context=out.get("prior_context"),
                    retrieve_rag=out.get("result"),
                    all_categories_summary=_state_row.get("all_categories_summary"),
                    memory_cache=memory_cache,
                    intentions_active=intentions_active,
                    subconscious_message=_subconscious_msg,
                )
                if _subconscious_msg:
                    _write_conversation_state(
                        cid, soul_id=soul_id, user_id=uid,
                        updates={"subconscious_message": None},
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
            "conversation.retrieve", payload, ok=False, error="HTTPException"
        )
        raise
    except Exception as exc:
        _record_call(
            "conversation.retrieve",
            payload,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise HTTPException(status_code=500, detail="Internal Server Error. Check server logs.") from exc


# ---- Turn state helpers + conversation turn endpoints ----

def _turn_state_read(
    cid: str,
    uid: str,
    soul_id: str,
    safe: dict[str, Any],
    state_override_cache: list[str],
    state_override_intentions: dict[str, Any],
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
    conversation_state, soul_card, db_path = _load_turn_state_and_soul_card(cid, user_id=uid, soul_id=soul_id)
    request_soul_card = str(safe.get("soul_card") or "").strip() or None
    soul_card = request_soul_card or soul_card
    memory_cache_before = list(state_override_cache)
    intentions_before = _normalize_intentions_stack_impl(state_override_intentions)
    unmemorized_digest_cursor = (
        conversation_state.get("digest_cursor")
        if conversation_state.get("last_memorize_at")
        else -1
    )
    unmemorized_tokens = _estimate_unmemorized_tokens(history_full, unmemorized_digest_cursor)
    queued_memorize_payload: dict[str, Any] | None = None
    if (not dry_run) and history_full and unmemorized_tokens >= _MIN_CHUNK_TOKENS:
        has_sleep_gap = _unmemorized_sleep_gap_detected(
            history_full,
            unmemorized_digest_cursor,
            safe,
        )
        if has_sleep_gap:
            normalized_history = _normalize_conversation(history_full)
            if normalized_history:
                queued_memorize_payload = {
                    **safe,
                    "conversation_id": cid,
                    "conversation": normalized_history,
                    "user": {"user_id": uid, "soul_id": soul_id, "conversation_id": cid},
                }
    return (
        conversation_state,
        soul_card,
        db_path,
        memory_cache_before,
        intentions_before,
        unmemorized_tokens,
        queued_memorize_payload,
    )


def _turn_state_write(
    cid: str,
    uid: str,
    soul_id: str,
    cache_entry: str,
    annulment_ids: list[str],
    chat_x: str | None,
    retrieval_ids_since_consolidation: list[str],
    apply_turn_maintenance: bool,
) -> tuple[dict[str, Any], Any]:
    latest_state_row, _, _ = _load_turn_state_and_soul_card(cid, user_id=uid, soul_id=soul_id)
    current_memory_cache = _normalize_memory_cache_impl(latest_state_row.get("memory_cache"))
    current_intentions = _normalize_intentions_stack_impl(latest_state_row.get("intentions_active"))
    intentions_snapshot = current_intentions
    if apply_turn_maintenance:
        current_intentions = _apply_intention_turn_maintenance_impl(current_intentions)
    next_memory_cache = (
        _append_memory_cache_entry(current_memory_cache, cache_entry)
        if cache_entry
        else list(current_memory_cache)
    )
    next_intentions = _remove_intentions(
        current_intentions,
        [item_id for item_id in annulment_ids if item_id],
    )
    next_chat_x, next_chat_x_prev = _next_chat_x_state_values(chat_x, latest_state_row)
    updates: dict[str, Any] = {
        "intentions_active": next_intentions,
        "memory_cache": next_memory_cache,
        "last_chat_x": next_chat_x,
        "last_chat_x_prev": next_chat_x_prev,
        "undo_snapshot": {
            "memory_cache": current_memory_cache,
            "intentions_active": intentions_snapshot,
        },
    }
    # Scene break resets the rewrite angle so the next RETRIEVE starts at the
    # topic lens instead of mid-rotation.
    previous_chat_x = str(latest_state_row.get("last_chat_x") or "").strip() or None
    if next_chat_x != previous_chat_x:
        updates["retrieve_rewrite_angle"] = 0
    if retrieval_ids_since_consolidation:
        updates["append_retrieval_ids_since_consolidation"] = retrieval_ids_since_consolidation
    state_out, state_path = _write_conversation_state(
        cid,
        soul_id=soul_id,
        user_id=uid,
        updates=updates,
    )
    return state_out, state_path


def _turn_launch_apimw(
    cid: str,
    uid: str,
    soul_id: str,
    safe: dict[str, Any],
    history_full: list[dict[str, Any]],
    state_out: dict[str, Any],
) -> str:
    cadence_anchors = _compact_chat_x_anchors(
        str(state_out.get("last_chat_x") or "").strip() or None,
    )
    cadence_slice = _slice_history_from_chat_x_anchors(history_full, cadence_anchors, limit=999)
    cadence_threshold = _apimw_cadence_from_cfg(_CONFIG)
    cadence_soul_messages = _count_soul_messages(cadence_slice, soul_id)
    if cadence_soul_messages < cadence_threshold:
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
                state_row=state_out,
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

        safe = _safe_payload(payload)

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
        override_user_prompt = str(prompt_override_payload.get("user_prompt", "")).strip()
        if not override_user_prompt:
            raise HTTPException(
                status_code=400,
                detail="prompt_override_payload.user_prompt is required",
            )
        override_system_prompt = str(prompt_override_payload.get("system_prompt", "")).strip()
        override_retrieve_rag = prompt_override_payload.get("retrieve_rag")
        if not isinstance(override_retrieve_rag, dict):
            raise HTTPException(
                status_code=400,
                detail="prompt_override_payload.retrieve_rag is required",
            )
        override_memory_cache_raw = prompt_override_payload.get("memory_cache")
        if not isinstance(override_memory_cache_raw, list):
            raise HTTPException(
                status_code=400,
                detail="prompt_override_payload.memory_cache is required",
            )
        override_intentions_raw = prompt_override_payload.get("intentions_active")
        if not isinstance(override_intentions_raw, dict):
            raise HTTPException(
                status_code=400,
                detail="prompt_override_payload.intentions_active is required",
            )
        override_memory_cache: list[str] = _normalize_memory_cache_impl(override_memory_cache_raw)
        override_intentions: dict[str, Any] = _normalize_intentions_stack_impl(override_intentions_raw)
        override_retrieve_ms = prompt_override_payload.get("retrieve_ms")
        retrieve_ms = int(override_retrieve_ms) if isinstance(override_retrieve_ms, (int, float)) else 0
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
                conversation_state_before,
                soul_card,
                db_path,
                memory_cache_before,
                intentions_before,
                unmemorized_tokens,
                queued_memorize_payload,
            ) = _turn_state_read(
                cid, uid, soul_id, safe, override_memory_cache, override_intentions,
                dry_run, history_full,
            )

        generation_config = _load_soul_gen_config(_CONFIG, uid, soul_id, logger=logger)
        turn_temperature: float = float(generation_config.get("temperature", 0.2))
        turn_response_format: Any = {"type": "json_object"}
        requested_temperature = safe.get("temperature")
        if requested_temperature is not None:
            try:
                updated_generation_config = dict(generation_config)
                updated_generation_config["temperature"] = float(requested_temperature)
                turn_temperature = updated_generation_config["temperature"]
                if updated_generation_config != generation_config:
                    _save_soul_gen_config(_CONFIG, uid, soul_id, updated_generation_config)
            except (TypeError, ValueError):
                logger.debug("ignoring invalid temperature override: %r", requested_temperature, exc_info=True)

        turn_system_prompt = override_system_prompt or _make_turn_system_prompt(
            soul_id,
            soul_card=soul_card,
            response_sentences=int(_CONFIG.get("turn_response_sentences", 3)),
        )
        turn_user_prompt = override_user_prompt

        memory_service = _get_service_from_payload(safe, retrieve_method_override="rag")
        turn_started_at = time.monotonic()
        turn_response_raw = await memory_service.chat(
            turn_user_prompt,
            system_prompt=turn_system_prompt,
            temperature=turn_temperature,
            max_tokens=PIPELINE_MAX_TOKENS,
            response_format=turn_response_format,
            op="turn",
            step="respond",
        )
        turn_ms = int((time.monotonic() - turn_started_at) * 1000)
        try:
            turn_contract = _parse_turn_contract(turn_response_raw)
        except Exception as exc:
            raw_snippet = str(turn_response_raw or "")[:200]
            raise HTTPException(
                status_code=502,
                detail=f"turn contract parse failure: {exc}; raw={raw_snippet!r}",
            ) from exc

        turn_cache_entry = str(turn_contract.get("cache_entry") or "").strip()
        turn_annulments = turn_contract.get("annulments") if isinstance(turn_contract.get("annulments"), list) else []
        normalized_annulments = [row for row in turn_annulments if isinstance(row, dict)]
        turn_annulment_ids = [
            str(row.get("intention_id") or "").strip()
            for row in normalized_annulments
        ]
        chat_x_anchor = str(turn_contract.get("chat_x") or "").strip() or None
        if chat_x_anchor:
            anchor_index: int | None = None
            for history_index, history_message in enumerate(history_full):
                if str(history_message.get("message_id") or "").strip() == chat_x_anchor:
                    anchor_index = history_index
                    break
            if anchor_index is None or (len(history_full) - anchor_index) < 6:
                chat_x_anchor = None

        conversation_state_after = conversation_state_before
        conversation_state_path = db_path
        annulment_memory_ids: list[str] = []

        if not dry_run:
            async with state_lock:
                retrieved_item_ids = _extract_result_item_ids(override_retrieve_rag)
                conversation_state_after, conversation_state_path = _turn_state_write(
                    cid, uid, soul_id,
                    turn_cache_entry, turn_annulment_ids, chat_x_anchor,
                    retrieved_item_ids,
                    apply_turn_maintenance,
                )

        if not dry_run:
            annulment_memory_ids = await _persist_annulment_memories(
                svc=memory_service,
                scope={"user_id": uid, "soul_id": soul_id},
                conversation_id=cid,
                intentions_before=intentions_before,
                annulments=normalized_annulments,
            )

        apimw_status = "skipped_dry_run" if dry_run else "not_started"
        if (not dry_run) and run_apimw:
            apimw_status = _turn_launch_apimw(
                cid, uid, soul_id, safe, history_full, conversation_state_after,
            )

        response_text = str(turn_contract.get("response") or "").strip()
        response_payload: dict[str, Any] = {
            "ok": True,
            "conversation_id": cid,
            "response": response_text,
            "apimw": apimw_status,
            "prompt_override_used": True,
            "final_turn_payload": {
                "system_prompt": turn_system_prompt,
                "user_prompt": turn_user_prompt,
                "memory_cache": memory_cache_before,
                "intentions_active": intentions_before,
            },
            "retrieve_ms": retrieve_ms,
            "turn_ms": turn_ms,
            "reply_chars": len(response_text),
            "turn_prompt_chars": len(turn_user_prompt),
            "turn_system_chars": len(turn_system_prompt),
        }
        if include_debug:
            response_payload["state"] = conversation_state_after
            response_payload["path"] = str(conversation_state_path) if conversation_state_path is not None else None
            response_payload["annulment_memory_ids"] = annulment_memory_ids
            response_payload["turn_contract"] = turn_contract
            response_payload["dry_run"] = dry_run
            response_payload["forced_memorize"] = {
                "queued": bool(queued_memorize_payload),
                "unmemorized_tokens": unmemorized_tokens,
                "min_chunk_tokens": _MIN_CHUNK_TOKENS,
            }

        if queued_memorize_payload is not None:
            _t = asyncio.create_task(_run_forced_memorize_from_turn(queued_memorize_payload))
            _BACKGROUND_TASKS.add(_t)
            _t.add_done_callback(_BACKGROUND_TASKS.discard)

        _record_call(
            "conversation.turn",
            safe,
            ok=True,
            info={
                "conversationId": cid,
                "dryRun": dry_run,
                "apimw": apimw_status,
                "forcedMemorizeQueued": bool(queued_memorize_payload),
                "unmemorizedTokens": unmemorized_tokens,
                "minChunkTokens": _MIN_CHUNK_TOKENS,
                "responseLen": len(str(response_payload.get("response") or "")),
            },
        )
        return response_payload
    except HTTPException:
        _record_call(
            "conversation.turn",
            payload,
            ok=False,
            info={"conversationId": cid or None},
            error="HTTPException",
        )
        raise
    except Exception as exc:
        _record_call(
            "conversation.turn",
            payload,
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

    safe = _safe_payload(payload)
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
        conversation_state, _, _ = _load_turn_state_and_soul_card(cid, user_id=uid, soul_id=soul_id)
        undo_snapshot = conversation_state.get("undo_snapshot")
        if not isinstance(undo_snapshot, dict):
            return {"status": "no_snapshot"}

        _write_conversation_state(
            cid,
            soul_id=soul_id,
            user_id=uid,
            updates={
                "memory_cache": list(undo_snapshot.get("memory_cache") or []),
                "intentions_active": undo_snapshot.get("intentions_active"),
                "undo_snapshot": None,
            },
        )
    return {"status": "restored"}


@app.post("/integration/memu/turn", operation_id="memu_turn", tags=["mcp_tools"])
async def mcp_memu_turn(req: _mcp_tools.MemuTurnRequest):
    return await _mcp_tools.memu_turn_endpoint(
        req,
        conversation_retrieve=conversation_retrieve,
        conversation_turn=conversation_turn,
    )


@app.post("/integration/memu/retrieve", operation_id="memu_retrieve", tags=["mcp_tools"])
async def mcp_memu_retrieve(req: _mcp_tools.MemuRetrieveRequest):
    return await _mcp_tools.memu_retrieve_endpoint(
        req,
        retrieve=retrieve,
    )


@app.post("/integration/memu/memorize", operation_id="memu_memorize", tags=["mcp_tools"])
async def mcp_memu_memorize(req: _mcp_tools.MemuMemorizeRequest):
    async def _memorize_call(payload: dict[str, Any], force: bool) -> dict[str, Any]:
        background_tasks = BackgroundTasks()
        response = await memorize(payload, background_tasks, force)
        # Internal endpoint call: FastAPI response hooks do not execute, so launch
        # background tasks explicitly and return immediately (fire-and-forget).
        try:
            runner = asyncio.create_task(background_tasks())
        except RuntimeError:
            logger.exception("failed to schedule memorize background tasks")
            return {"ok": False, "detail": "failed to schedule memorize background tasks"}
        _BACKGROUND_TASKS.add(runner)
        runner.add_done_callback(_BACKGROUND_TASKS.discard)
        if isinstance(response, dict):
            return response
        if not isinstance(response, JSONResponse):
            logger.error("unexpected memorize response type: %s", type(response).__name__)
            return {"ok": False, "detail": "unexpected memorize response type"}
        try:
            parsed = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.exception("failed to parse memorize response body")
            return {"ok": False, "detail": "invalid memorize response"}
        if not isinstance(parsed, dict):
            logger.error("unexpected memorize response payload type: %s", type(parsed).__name__)
            return {"ok": False, "detail": "unexpected memorize response payload type"}
        return parsed

    return await _mcp_tools.memu_memorize_endpoint(
        req,
        memorize=_memorize_call,
    )


@app.post("/integration/memu/consolidate", operation_id="memu_consolidate", tags=["mcp_tools"])
async def mcp_memu_consolidate(req: _mcp_tools.MemuConsolidateRequest):
    return await _mcp_tools.memu_consolidate_endpoint(
        req,
        force_consolidation=force_consolidation,
    )


@app.post("/integration/memu/intentions", operation_id="memu_intentions", tags=["mcp_tools"])
async def mcp_memu_intentions(req: _mcp_tools.MemuIntentionsRequest):
    return await _mcp_tools.memu_intentions_endpoint(
        req,
        list_intentions=list_intentions,
    )


@app.post("/integration/memu/state", operation_id="memu_state", tags=["mcp_tools"])
async def mcp_memu_state(req: _mcp_tools.MemuStateRequest):
    return await _mcp_tools.memu_state_endpoint(
        req,
        get_state=get_conversation_state,
        patch_state=patch_conversation_state,
    )


# =============================================================================
# Optional MCP mount & static UI
# =============================================================================

_has_mcp = False
try:
    # fastapi_mcp currently triggers a pydantic v2.11+ deprecation warning
    # at import-time; keep runtime clean while preserving optional MCP mount.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The `__get_pydantic_core_schema__` method of the `BaseModel` class is deprecated.*",
            category=DeprecationWarning,
            module=r"pydantic\._internal\._generate_schema",
        )
        from fastapi_mcp import FastApiMCP

    mcp = FastApiMCP(
        app,
        include_operations=[
            "memu_turn",
            "memu_retrieve",
            "memu_memorize",
            "memu_consolidate",
            "memu_intentions",
        ],
    )
    http_path = str(_CONFIG.get("mcp", {}).get("http_path") or "/mcp")
    sse_path = str(_CONFIG.get("mcp", {}).get("sse_path") or "/sse")
    mcp.mount_http(mount_path=http_path)
    mcp.mount_sse(mount_path=sse_path)
    _has_mcp = True
except (ImportError, TypeError):
    _has_mcp = False


try:
    _BUNDLE_ROOT = Path(__file__).resolve().parents[2]
    _UI_DIST = _BUNDLE_ROOT / "memu-ui" / "dist"
    if _UI_DIST.exists():
        app.mount("/", StaticFiles(directory=str(_UI_DIST), html=True), name="ui")
except (ImportError, OSError):
    logger.debug("static UI mount skipped", exc_info=True)
