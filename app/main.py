import asyncio
import json
import logging
import os
import random
import re
import signal
import sqlite3
import threading
import time
import uuid
import warnings
from collections import deque
from collections.abc import Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
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
from app.services import admin_routes as _admin_routes
from app.services.consolidation import (
    ConsolidationDeps,
    gather_consolidation_inputs as _gather_consolidation_inputs,
    run_consolidation_llm as _run_consolidation_llm,
    write_consolidation_outputs as _write_consolidation_outputs,
)
from app.services.intention_state import (
    append_memory_cache_entry as _append_memory_cache_entry,
    apply_intention_turn_maintenance as _apply_intention_turn_maintenance_impl,
    format_intentions_for_prompt as _format_intentions_for_prompt,
    normalize_intentions_stack as _normalize_intentions_stack_impl,
    normalize_memory_cache as _normalize_memory_cache_impl,
    remove_intentions as _remove_intentions,
)
from app.services import crud_endpoints as _crud_endpoints
from app.services import conversation_sources as _conversation_sources
from app.services import mcp_tools as _mcp_tools
from app.services import message_log as _message_log
from app.services import retrieve_orchestration as _retrieve_orchestration
from app.services import soul_state as _soul_state
from app.services.narrative_self import snapshot_previous_narrative_self
from app.services import memorize_endpoint as _memorize_endpoint
from app.services import service_factory as _service_factory
from app.services.payload import (
    _canonicalize_scope_where,
    _extract_conversation_id,
    _extract_result_item_ids,
    _extract_scope,
    _item_sig,
    _norm_result_sig,
    _normalize_conversation,
    _normalize_turn_history,
    _parse_as_of_datetime,
    _payload_signature,
    _parse_turn_ts_ms,
    _pick_str,
    _safe_payload,
)
from app.services import sqlite_scope as _sqlite_scope
from app.services.state import (
    conversation_state_empty as _conversation_state_empty,
    conversation_state_from_row as _conversation_state_from_row_impl,
    conversation_state_row as _conversation_state_row,
    write_conversation_state as _write_conversation_state_impl,
)
from app.services.turn_contract import (
    LIFE_GOALS_FREE_WILL_HEADER as _LIFE_GOALS_FREE_WILL_HEADER,
    build_turn_prompt as _build_turn_prompt,
    format_time_anchor as _format_time_anchor,
    format_memory_line as _format_memory_line,
    format_memory_legend as _format_memory_legend,
    format_shaped_by_line as _format_shaped_by_line,
    make_turn_system_prompt as _make_turn_system_prompt,
    parse_turn_contract as _parse_turn_contract,
    render_history as _render_history,
    _merge_current_into_conversations,
    _section_title_from_conversation_id,
)


# ==== Module state & constants ====

logger = logging.getLogger(__name__)
_PROMPT_LOGGER = logging.getLogger("uvicorn.error")

app = FastAPI(title="mcp-memu-server", version="0.4.0")

_BUILD_ID: str = "fix48.debloat.bloatRemoval.concepts"
_SLEEP_SPLIT_MIN_LULL_SECONDS: int = 3 * 60 * 60
_DEFAULT_MIN_CHUNK_TOKENS: int = 4000
_DEFAULT_EPISODES_PER_SEGMENT: int = 3
_DEFAULT_BACKGROUND_SUMMARY_TOKENS: int = 1000
_MIN_CHUNK_TOKENS: int = _DEFAULT_MIN_CHUNK_TOKENS
_EPISODES_PER_SEGMENT: int = _DEFAULT_EPISODES_PER_SEGMENT
_BACKGROUND_SUMMARY_TOKENS: int = _DEFAULT_BACKGROUND_SUMMARY_TOKENS
# Uniform runaway-protection caps for LLM calls. Not business logic —
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


def _background_sleep_gap_detected(
    *,
    history: list[dict[str, Any]],
    safe: dict[str, Any],
) -> bool:
    gap_safe = dict(safe)
    has_tz_name = bool(str(gap_safe.get("time_zone") or gap_safe.get("timeZone") or "").strip())
    has_tz_off = isinstance(gap_safe.get("time_zone_offset_min"), (int, float)) or isinstance(
        gap_safe.get("timeZoneOffsetMin"), (int, float)
    )
    if not has_tz_name and not has_tz_off:
        gap_safe["time_zone_offset_min"] = 0
    return _memorize_endpoint.unmemorized_sleep_gap_detected(
        history,
        digest_cursor=-1,
        safe=gap_safe,
        logger=logger,
        min_chunk_tokens=_BACKGROUND_SUMMARY_TOKENS,
        sleep_split_min_lull_seconds=_SLEEP_SPLIT_MIN_LULL_SECONDS,
    )


async def _run_background_rollup_for_conversation(
    *,
    conversation_id: str,
    user_id: str,
    soul_id: str,
    safe_payload: dict[str, Any],
    service: MemoryService | None = None,
) -> str:
    cid = str(conversation_id or "").strip()
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    if not cid or not uid or not sid:
        return "skipped_scope"

    state_lock = _get_memorize_lock(_memorize_lock_key(uid, sid))
    async with state_lock:
        state_row, _soul_card, db_path = _load_turn_state_and_soul_card(
            cid,
            user_id=uid,
            soul_id=sid,
        )
        if bool(state_row.get("memorize_chat", True)):
            return "skipped_primary_chat"
        if db_path is None or not db_path.exists():
            return "skipped_no_db"

        con = _sqlite_connect(db_path)
        rollup_error: Exception | None = None
        try:
            con.row_factory = sqlite3.Row
            _sqlite_ensure_conversation_state_schema(con)
            rolling_cursor_id = state_row.get("rolling_summary_cursor_id")
            tail = _message_log.read_tail_after_message_id(con, cid, rolling_cursor_id)
            if len(tail) < 2:
                return "skipped_short_tail"
            tail_end_row_id = int(tail[-1].get("id") or 0)
            if tail_end_row_id <= 0:
                return "skipped_short_tail"

            sleep_history: list[dict[str, Any]] = []
            tokenize_messages: list[dict[str, Any]] = []
            for msg in tail:
                content = str(msg.get("content") or "").strip()
                if not content:
                    continue
                ts_ms = _parse_turn_ts_ms(msg.get("received_at"))
                if ts_ms is None:
                    continue
                sleep_history.append({"content": content, "ts_ms": ts_ms})
                tokenize_messages.append({"content": content})
            if len(sleep_history) < 2:
                return "skipped_short_tail"

            token_estimate = _estimate_tokens(tokenize_messages)
            if token_estimate < _BACKGROUND_SUMMARY_TOKENS:
                return "skipped_tokens"
            if not _background_sleep_gap_detected(history=sleep_history, safe=safe_payload):
                return "skipped_lull"

            prior_summary = str(state_row.get("rolling_summary") or "").strip() or None
            llm_service = service or _get_service_from_payload(
                {
                    **safe_payload,
                    "user": {"user_id": uid, "soul_id": sid, "conversation_id": cid},
                    "conversation_id": cid,
                }
            )
            summary_input = [
                {
                    "role": str(msg.get("role") or "user"),
                    "name": str(msg.get("speaker") or "").strip() or None,
                    "content": str(msg.get("content") or ""),
                    "source_label": msg.get("source_label"),
                    "_message_index": int(msg.get("id") or 0),
                }
                for msg in tail
            ]
            new_summary = await llm_service.summarize_background_chat_rollup(
                prior_summary=prior_summary,
                messages=summary_input,
            )
            now_iso = datetime.now(UTC).isoformat()
            con.execute(
                "UPDATE conversations SET rolling_summary = ?, rolling_summary_cursor_id = ?, "
                "rolling_summary_updated_at = ?, updated_at = ? WHERE conversation_id = ?",
                (new_summary, tail_end_row_id, now_iso, now_iso, cid),
            )
            _message_log.delete_messages_through_id(con, cid, tail_end_row_id)
            con.commit()
            return "rolled_up"
        except Exception as exc:
            rollup_error = exc
            raise
        finally:
            con.close()
            if rollup_error is not None:
                try:
                    _set_background_error(
                        cid,
                        soul_id=sid,
                        user_id=uid,
                        code="background_rollup_failed",
                        detail=f"{type(rollup_error).__name__}: {str(rollup_error)[:220]}",
                    )
                except Exception:
                    logger.exception("failed to record background rollup error for %s", cid)


def _queue_background_rollup_task(
    *,
    conversation_id: str,
    user_id: str,
    soul_id: str,
    safe_payload: dict[str, Any],
    service: MemoryService | None = None,
) -> None:
    marker = f"{user_id}::{soul_id}::{conversation_id}"
    if not _mark_inflight(_BACKGROUND_ROLLUP_INFLIGHT, marker):
        return
    task = asyncio.create_task(
        _run_background_rollup_for_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            soul_id=soul_id,
            safe_payload=safe_payload,
            service=service,
        )
    )
    _BACKGROUND_TASKS.add(task)
    def _on_done(done_task: asyncio.Task) -> None:
        try:
            done_task.result()
        except Exception:
            logger.exception("background rollup task failed for %s", marker)
        finally:
            _BACKGROUND_TASKS.discard(done_task)
            _clear_inflight(_BACKGROUND_ROLLUP_INFLIGHT, marker)
    task.add_done_callback(_on_done)


# ==== Server state (locks, inflight, shutdown) ====

_SERVER_INSTANCE_ID: str = str(uuid.uuid4())
_SERVER_STARTED_AT_UNIX: float = time.time()
_LAST_CALLS: deque[dict[str, Any]] = deque(maxlen=50)
_LAST_HTTP: deque[dict[str, Any]] = deque(maxlen=200)
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
_BACKGROUND_ROLLUP_INFLIGHT: set[str] = set()
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
    return _mark_inflight(_APIMW_INFLIGHT, conversation_id)


def _clear_apimw_inflight(conversation_id: str) -> None:
    _clear_inflight(_APIMW_INFLIGHT, conversation_id)


def _mark_inflight(pool: set[str], marker: str) -> bool:
    key = str(marker or "").strip()
    if not key:
        return False
    with _STATE_LOCK:
        if key in pool:
            return False
        pool.add(key)
        return True


def _clear_inflight(pool: set[str], marker: str) -> None:
    key = str(marker or "").strip()
    if not key:
        return
    with _STATE_LOCK:
        pool.discard(key)


_CONTROL_PATHS: frozenset[str] = frozenset(
    ("/health", "/version", "/admin/shutdown", "/admin/shutdown/status", "/diag")
)


def _is_control_path(path: str) -> bool:
    p = str(path or "")
    if p in _CONTROL_PATHS or p.startswith("/diag/"):
        return True
    pref = str(_DIAG_PREFIX or "").rstrip("/")
    return bool(pref and (p == f"{pref}/diag" or p.startswith(f"{pref}/diag/")))


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


def _schedule_shutdown(max_wait_sec: int) -> None:
    global _SHUTDOWN_TASK
    _SHUTDOWN_TASK = asyncio.create_task(_shutdown_when_idle(max_wait_sec))


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


# ==== Config loading & storage paths ====


class STUserModel(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


_CONFIG: dict[str, Any] = _load_config()

def _refresh_runtime_limits() -> None:
    global _MIN_CHUNK_TOKENS, _EPISODES_PER_SEGMENT, _BACKGROUND_SUMMARY_TOKENS
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
        _BACKGROUND_SUMMARY_TOKENS = max(
            0,
            int(memorize_cfg.get("background_summary_tokens", _DEFAULT_BACKGROUND_SUMMARY_TOKENS)),
        )
    except (TypeError, ValueError, OverflowError):
        _BACKGROUND_SUMMARY_TOKENS = _DEFAULT_BACKGROUND_SUMMARY_TOKENS
    debug_cfg = _CONFIG.get("debug") if isinstance(_CONFIG.get("debug"), dict) else {}
    _LOG_PROMPTS = bool(debug_cfg.get("log_prompts", False))


def _prompt_log_before(ctx: Any, request_view: Any) -> None:
    import time as _time
    ctx._llm_start = _time.monotonic()
    return None


def _prompt_log_after(ctx: Any, request_view: Any, response_view: Any, usage: Any) -> None:
    import time as _time
    elapsed = _time.monotonic() - getattr(ctx, "_llm_start", _time.monotonic())
    content = getattr(response_view, "content", None)
    kind = getattr(request_view, "kind", None)
    if kind == "embed":
        logger.info("[EMBED] elapsed=%.1fs", elapsed)
        return
    meta = getattr(request_view, "metadata", None) or {}
    payload = meta.get("payload")
    op = getattr(ctx, "operation", None) or "-"
    step = getattr(ctx, "step_id", None) or "-"
    req_id = str(getattr(ctx, "request_id", "") or "").strip() or "-"
    trace_id = str(getattr(ctx, "trace_id", "") or "").strip() or "-"
    model = getattr(ctx, "model", None) or "-"
    banner = f"===== {str(op).upper()} · {step} ".ljust(70, "=")
    lines = [
        "",
        "",
        "",
        banner,
        "",
        f"[PROMPT] trace={trace_id} req={req_id} op={op} step={step} model={model}",
    ]
    if isinstance(payload, dict):
        payload_log_text = json.dumps(payload, ensure_ascii=False, indent=2).replace("\\n", "\n")
        lines.append(f"[PAYLOAD] trace={trace_id} req={req_id} op={op} step={step} kind={kind or '-'} model={model}")
        lines.append(payload_log_text)
    finish_reason = getattr(usage, "finish_reason", None)
    in_tok = getattr(usage, "input_tokens", None)
    out_tok = getattr(usage, "output_tokens", None)
    total_tok = getattr(usage, "total_tokens", None)
    lines.append(
        (
            f"[RESPONSE] trace={trace_id} req={req_id} op={op} step={step} elapsed={elapsed:.1f}s "
            f"finish_reason={finish_reason} tokens=in:{in_tok}/out:{out_tok}/total:{total_tok} "
            f"content_chars={len(content or '')}"
        )
    )
    if content:
        lines.extend(["", content, ""])
    _PROMPT_LOGGER.info(
        "\n".join(lines),
    )


def _prompt_log_on_error(ctx: Any, request_view: Any, error: Any, usage: Any) -> None:
    import time as _time
    elapsed = _time.monotonic() - getattr(ctx, "_llm_start", _time.monotonic())
    if getattr(request_view, "kind", None) == "embed":
        logger.error("[EMBED] elapsed=%.1fs error=%s", elapsed, type(error).__name__)
        return
    meta = getattr(request_view, "metadata", None) or {}
    payload = meta.get("payload")
    op = getattr(ctx, "operation", None) or "-"
    step = getattr(ctx, "step_id", None) or "-"
    req_id = str(getattr(ctx, "request_id", "") or "").strip() or "-"
    trace_id = str(getattr(ctx, "trace_id", "") or "").strip() or "-"
    model = getattr(ctx, "model", None) or "-"
    banner = f"===== {str(op).upper()} · {step} ".ljust(70, "=")
    lines = [
        "",
        "",
        "",
        banner,
        "",
        f"[PROMPT] trace={trace_id} req={req_id} op={op} step={step} model={model}",
    ]
    if isinstance(payload, dict):
        payload_log_text = json.dumps(payload, ensure_ascii=False, indent=2).replace("\\n", "\n")
        kind = getattr(request_view, "kind", None)
        lines.append(f"[PAYLOAD] trace={trace_id} req={req_id} op={op} step={step} kind={kind or '-'} model={model}")
        lines.append(payload_log_text)
    lines.append(
        (
            f"[ERROR] trace={trace_id} req={req_id} op={op} step={step} elapsed={elapsed:.1f}s "
            f"type={type(error).__name__} message={error}"
        )
    )
    _PROMPT_LOGGER.error("\n".join(lines))


_refresh_runtime_limits()

# Also expose diagnostics under the MCP http_path (e.g. /mcp/diag) to avoid path confusion.
_DIAG_PREFIX: str = str(_CONFIG.get("mcp", {}).get("http_path") or "/mcp").rstrip("/")
if _DIAG_PREFIX == "":
    _DIAG_PREFIX = "/mcp"


_resolve_profile = _service_factory._resolve_profile
_retrieve_apimw_enabled_from_cfg = _service_factory._retrieve_apimw_enabled_from_cfg
_cfg_int = _service_factory._cfg_int
_apimw_cadence_from_cfg = _service_factory._apimw_cadence_from_cfg
_apimw_memory_count_from_cfg = _service_factory._apimw_memory_count_from_cfg
_apimw_random_count_from_cfg = _service_factory._apimw_random_count_from_cfg
_consolidation_interval_days_from_cfg = _service_factory._consolidation_interval_days_from_cfg
_build_apimw_retrieve_config = _service_factory._build_apimw_retrieve_config
_count_soul_messages = _service_factory._count_soul_messages
_merge_llm_profiles = _service_factory._merge_llm_profiles
_clear_cached_services = _service_factory._clear_cached_services


def _get_service_from_payload(payload: dict[str, Any]):
    return _service_factory._get_service_from_payload(
        payload,
        config=_CONFIG,
        default_llm_profiles_from_server_config=_default_llm_profiles_from_server_config,
        database_config_from_cfg=_database_config_from_cfg,
        blob_config_from_cfg=_blob_config_from_cfg,
        categories_from_cfg=_categories_from_cfg,
        normalize_sqlite_dsn=_normalize_sqlite_dsn,
        sqlite_dsn_for_scope=_sqlite_dsn_for_scope,
        sqlite_file_from_dsn=_sqlite_file_from_dsn,
        extract_scope=_extract_scope,
        payload_signature=_payload_signature,
        episodes_per_segment=_EPISODES_PER_SEGMENT,
        log_prompts=_LOG_PROMPTS,
        prompt_log_before=_prompt_log_before,
        prompt_log_after=_prompt_log_after,
        prompt_log_on_error=_prompt_log_on_error,
        st_user_model=STUserModel,
        logger=logger,
    )


# ==== Payload & scope extraction ====


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
    except Exception:
        logger.debug("record_call telemetry append failed", exc_info=True)


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None:
        marker = id(current)
        if marker in seen:
            break
        seen.add(marker)
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _timeout_cause(exc: BaseException) -> BaseException | None:
    for err in _iter_exception_chain(exc):
        if isinstance(err, (TimeoutError, asyncio.TimeoutError)):
            return err
        name = type(err).__name__.lower()
        if "timeout" in name:
            return err
    return None


def _raise_upstream_http_error(exc: Exception, *, op: str) -> None:
    timeout = _timeout_cause(exc)
    if timeout is not None:
        detail = f"upstream timeout during {op}: {type(timeout).__name__}: {timeout}"
        raise HTTPException(status_code=504, detail=detail) from exc
    raise HTTPException(status_code=500, detail="Internal Server Error. Check server logs.") from exc


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
        write_conversation_state_impl=_write_conversation_state_impl,
        sqlite_current_path=_sqlite_current_path,
    )


# ==== Retrieve payload helpers ====

_extract_retrieve_where = _retrieve_orchestration._extract_retrieve_where
_extract_retrieve_queries = _retrieve_orchestration._extract_retrieve_queries


# ==== Turn prompt context builders ====

_build_retrieve_identity_context = _retrieve_orchestration._build_retrieve_identity_context
_build_retrieve_soul_context_queries = _retrieve_orchestration._build_retrieve_soul_context_queries


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
            happened_at=datetime.now(UTC),
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
            if summary.lstrip().startswith(f"# {name}"):
                lines.append(summary)
            else:
                lines.append(f"## {name}\n{summary}")
    if not lines:
        return None

    full_text = "\n\n".join(lines)
    system_prompt = (
        f"You are {soul_id}. Write a 250-350 word overview of your life from the summaries below, in your own voice."
    )
    result = await svc.chat(
        full_text,
        profile="holistic_summary",
        system_prompt=system_prompt,
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
    return await _retrieve_orchestration._run_retrieve(
        payload,
        conversation_id=conversation_id,
        safe_payload=_safe_payload,
        extract_conversation_id=_extract_conversation_id,
        get_service_from_payload=_get_service_from_payload,
        parse_as_of_datetime=_parse_as_of_datetime,
        sqlite_current_path=_sqlite_current_path,
        sqlite_connect=_sqlite_connect,
        sqlite_ensure_conversation_state_schema=_sqlite_ensure_conversation_state_schema,
        conversation_state_from_row=_conversation_state_from_row,
        conversation_state_row=_conversation_state_row,
        write_conversation_state=_write_conversation_state,
        procedural_module=_procedural,
        procedural_yaml_dir=_procedural_yaml_dir,
        procedural_db_path=_procedural_db_path,
        procedural_should_ingest=_procedural_should_ingest,
        config=_CONFIG,
        logger=logger,
    )


def _set_background_error(
    conversation_id: str,
    *,
    soul_id: str,
    user_id: str,
    code: str,
    detail: str,
) -> None:
    msg = f"{code}: {detail}".strip()[:300]
    _write_conversation_state(
        conversation_id,
        soul_id=soul_id,
        user_id=user_id,
        updates={
            "last_background_error": msg,
            "last_background_error_at": datetime.now(UTC).isoformat(),
        },
    )


_APIMW_BACKGROUND_ERROR_PREFIXES = (
    "apimw_topic_failed:",
    "apimw_def_parse_failed:",
    "apimw_failed:",
)


def _clear_background_error_if_apimw_owned(
    conversation_id: str,
    *,
    soul_id: str,
    user_id: str,
) -> None:
    state_row, _soul_card, _db_path = _load_turn_state_and_soul_card(
        conversation_id,
        user_id=user_id,
        soul_id=soul_id,
    )
    existing = str((state_row or {}).get("last_background_error") or "").strip()
    if not existing:
        return
    if not any(existing.startswith(prefix) for prefix in _APIMW_BACKGROUND_ERROR_PREFIXES):
        return
    _write_conversation_state(
        conversation_id,
        soul_id=soul_id,
        user_id=user_id,
        updates={
            "last_background_error": None,
            "last_background_error_at": None,
        },
    )


async def _apimw_topic_statement(
    svc: Any,
    *,
    topic_user: str,
    payload: dict[str, Any],
    identity_context: str,
    conversation_id: str,
) -> tuple[str, bool]:
    logger.info("apimw step A: topic statement for %s", conversation_id)
    topic_system = (
        f"{identity_context}\n\n"
        "State the topic of the CURRENT episode in 1-2 sentences. The previous episode is provided only as context — "
        "if the current episode is brief, use it to understand what the new message means, but describe only where the conversation is now."
    )
    topic_statement = await svc.chat(
        topic_user,
        profile="topic_statement",
        system_prompt=topic_system,
        op="apimw",
        step="topic_statement",
    )
    parsed_topic = str(topic_statement or "").strip()
    if parsed_topic:
        return parsed_topic, False
    fallback = _pick_str(payload, "message", "query") or ""
    if fallback:
        logger.warning(
            "apimw step A: empty topic statement for %s; falling back to message/query",
            conversation_id,
        )
        return fallback, True
    logger.warning("apimw step A: empty topic statement for %s; no message/query fallback", conversation_id)
    return "", True


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
        conversation_id=conversation_id,
    )
    retrieve_payload = {
        **payload,
        "query": query_text,
        "queries": retrieve_queries,
        "conversation_id": conversation_id,
        "force_retrieve": True,
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
                "created_at": item.created_at,
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
    id_map: dict[str, str] = {}
    counter = 1
    for item in combined_items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        summary = str(item.get("summary") or "").strip()
        if not item_id or not summary:
            continue
        items_by_id[item_id] = item
        id_map[str(counter)] = item_id
        formatted_memory_lines.append(_format_memory_line(item, show_id=True, item_id=str(counter)))
        counter += 1
        shaped_by = item.get("shaped_by")
        if isinstance(shaped_by, dict):
            formatted_memory_lines.append(_format_shaped_by_line(shaped_by))

    legend = _format_memory_legend({str(item.get("memory_type") or "") for item in combined_items if isinstance(item, dict)})
    if legend and formatted_memory_lines:
        formatted_memory_lines.insert(0, legend)
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
        response_format={"type": "json_object"},
        op="apimw",
        step="def_call",
    )

    apimw_response_text = str(llm_raw or "").strip()
    try:
        result_json = json.loads(apimw_response_text)
    except json.JSONDecodeError:
        logger.error("apimw D+E+F: JSON parse failed, raw=%s", apimw_response_text[:200])
        return None, items_by_id, id_map
    if not isinstance(result_json, dict):
        logger.error("apimw D+E+F: expected dict, got %s", type(result_json).__name__)
        return None, items_by_id, id_map

    logger.info("apimw D+E+F: parsed JSON with keys %s for %s", list(result_json.keys()), conversation_id)
    return result_json, items_by_id, id_map


async def _apimw_persist(
    svc: Any,
    *,
    result_json: dict[str, Any],
    items_by_id: dict[str, dict[str, Any]],
    id_map: dict[str, str],
    combined_items: list[dict[str, Any]],
    scope: dict[str, str],
    conversation_id: str,
    user_id: str,
    soul_id: str,
) -> None:
    async with _retrieve_scope_lock(user_id, soul_id):
        updates: dict[str, Any] = {}
        resolved_prior_context_ids: list[str] = []

        fresh_row, _, _ = _load_turn_state_and_soul_card(conversation_id, user_id=user_id, soul_id=soul_id)

        prior_context_ids_raw = result_json.get("prior_context") or []
        if isinstance(prior_context_ids_raw, list) and prior_context_ids_raw:
            prior_context_lines: list[str] = []
            for raw_memory_id in prior_context_ids_raw:
                numbered = str(raw_memory_id).strip()
                if not numbered:
                    continue
                memory_id = id_map.get(numbered, numbered)
                resolved_prior_context_ids.append(memory_id)
                item = items_by_id.get(memory_id)
                if not item:
                    continue
                prior_context_lines.append(_format_memory_line(item))
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

        message_to_self = str(result_json.get("message_to_self") or "").strip()
        if message_to_self:
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
            list(dict.fromkeys(resolved_prior_context_ids))
            if resolved_prior_context_ids
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
        svc = _get_service_from_payload(payload)
        scope = {"user_id": user_id, "soul_id": soul_id}
        apimw_item_top_k = _apimw_memory_count_from_cfg(_CONFIG)
        apimw_random_count = _apimw_random_count_from_cfg(_CONFIG)

        recent_history = history[-30:] if history else []
        episode_text = _render_history(recent_history)
        topic_user = f"Recent conversation:\n{episode_text or '(none)'}"

        identity_context = _build_retrieve_identity_context(soul_id, apimw=True)
        topic_statement, used_topic_fallback = await _apimw_topic_statement(
            svc,
            topic_user=topic_user,
            payload=payload,
            identity_context=identity_context,
            conversation_id=conversation_id,
        )
        if used_topic_fallback:
            detail = "topic_statement empty; used message/query fallback" if topic_statement else "topic_statement empty and no fallback"
            try:
                _set_background_error(
                    conversation_id,
                    soul_id=soul_id,
                    user_id=user_id,
                    code="apimw_topic_failed",
                    detail=detail,
                )
            except Exception:
                logger.exception("failed to record APImw topic fallback state for %s", conversation_id)
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
        result_json, items_by_id, apimw_id_map = await _apimw_def_call(
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
            try:
                _set_background_error(
                    conversation_id,
                    soul_id=soul_id,
                    user_id=user_id,
                    code="apimw_def_parse_failed",
                    detail="step D+E+F response was not valid JSON object",
                )
            except Exception:
                logger.exception("failed to record APImw def-parse failure for %s", conversation_id)
            return

        await _apimw_persist(
            svc,
            result_json=result_json,
            items_by_id=items_by_id,
            id_map=apimw_id_map,
            combined_items=combined_items,
            scope=scope,
            conversation_id=conversation_id,
            user_id=user_id,
            soul_id=soul_id,
        )
        try:
            _clear_background_error_if_apimw_owned(conversation_id, soul_id=soul_id, user_id=user_id)
        except Exception:
            logger.exception("failed to clear APImw background error state for %s", conversation_id)

    except Exception as exc:
        try:
            _set_background_error(
                conversation_id,
                soul_id=soul_id,
                user_id=user_id,
                code="apimw_failed",
                detail=f"{type(exc).__name__}: {str(exc)[:220]}",
            )
        except Exception:
            logger.exception("failed to record APImw failure state for %s", conversation_id)
        logger.exception("APImw background pipeline failed for %s", conversation_id)


# =============================================================================
# HTTP ENDPOINTS
# =============================================================================


# ---- Health, version, diag, shutdown ----
_admin_routes.register_admin_routes(
    app,
    diag_prefix=_DIAG_PREFIX,
    build_id=_BUILD_ID,
    server_instance_id=_SERVER_INSTANCE_ID,
    server_started_at_unix=_SERVER_STARTED_AT_UNIX,
    startup_warnings=_STARTUP_WARNINGS,
    storage_status=_STORAGE_STATUS,
    get_config=lambda: _CONFIG,
    get_storage_dir=_get_storage_dir,
    is_ephemeral_db=_is_ephemeral_db,
    config_path=_config_path,
    services_cached=_service_factory._services_cached,
    mcp_enabled=lambda: _has_mcp,
    shutdown_snapshot=_shutdown_snapshot,
    begin_shutdown_drain=_begin_shutdown_drain,
    schedule_shutdown=_schedule_shutdown,
    last_calls=_LAST_CALLS,
    last_http=_LAST_HTTP,
    sqlite_current_path=_sqlite_current_path,
    sqlite_file_info=_sqlite_file_info,
    sqlite_connect=_sqlite_connect,
    sqlite_pragmas=_sqlite_pragmas,
    sqlite_table_columns=_sqlite_table_columns,
    sqlite_build_scope_where=_sqlite_build_scope_where,
    logger=logger,
)


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
) -> tuple[Path, str, str]:
    return _memorize_endpoint.resolve_chat_storage_dir(
        chats_dir,
        uid,
        aid,
        conversation_id,
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


def _load_cross_tail_from_sources(
    con: sqlite3.Connection,
    *,
    exclude_conversation_id: str | None = None,
) -> list[dict[str, Any]]:
    hermes_cfg = _CONFIG.get("hermes") if isinstance(_CONFIG.get("hermes"), dict) else {}
    hermes_home_raw = str(hermes_cfg.get("home") or "").strip()
    sessions_index_raw = str(hermes_cfg.get("sessions_index_path") or "").strip()
    state_db_raw = str(hermes_cfg.get("state_db_path") or "").strip()
    hermes_home_path = Path(hermes_home_raw).expanduser().resolve() if hermes_home_raw else None
    sessions_index_path = Path(sessions_index_raw).expanduser().resolve() if sessions_index_raw else None
    state_db_path = Path(state_db_raw).expanduser().resolve() if state_db_raw else None
    excluded_id = str(exclude_conversation_id or "").strip()
    cursor_rows = con.execute(
        "SELECT conversation_id, digest_cursor, last_memorize_at FROM conversations"
    ).fetchall()
    all_messages: list[dict[str, Any]] = []
    for row in cursor_rows:
        cid = str(row["conversation_id"] or "").strip()
        if not cid or cid == excluded_id:
            continue
        if _message_log.derive_source_label(cid) != "whatsapp":
            continue
        cursor = int(row["digest_cursor"] or 0) if row["last_memorize_at"] else -1
        try:
            tail = _conversation_sources.load_whatsapp_tail(
                conversation_id=cid,
                since_cursor=cursor,
                recent_fallback_messages=_message_log.DEFAULT_CROSS_RECENT_FALLBACK_MESSAGES,
                hermes_home=hermes_home_path,
                sessions_index_path=sessions_index_path,
                state_db_path=state_db_path,
            )
        except Exception as exc:
            logger.error("cross-context source read failed for conversation_id=%s: %s", cid, exc)
            continue
        all_messages.extend(tail)
    all_messages.sort(
        key=lambda msg: (
            str(msg.get("received_at") or ""),
            str(msg.get("conversation_id") or ""),
            int(msg.get("source_conversation_index") or 0),
        )
    )
    return all_messages


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
    progress_key: str | None = None,
    memorize_progress: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if progress_key and memorize_progress is not None:
        _memorize_endpoint._set_memorize_progress(
            memorize_progress,
            progress_key,
            active=True,
            phase="consolidating",
            current=1,
            total=1,
        )
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
            if progress_key and memorize_progress is not None:
                _memorize_endpoint._set_memorize_progress(
                    memorize_progress,
                    progress_key,
                    active=False,
                    last_result="success",
                )
            return {"ok": True, "status": "skipped"}
        _write_conversation_state(
            conversation_id,
            soul_id=soul_id,
            user_id=uid,
            updates={
                "last_consolidation_error": None,
                "last_consolidation_error_at": None,
            },
        )
        if progress_key and memorize_progress is not None:
            _memorize_endpoint._set_memorize_progress(
                memorize_progress,
                progress_key,
                active=False,
                last_result="success",
            )
        return {"ok": True, "status": "ok"}
    except Exception as exc:
        logger.exception("consolidation failed (non-fatal)")
        if pipeline_started:
            await _clear_consolidation_in_progress(
                state_lock=state_lock,
                conversation_id=conversation_id,
                soul_id=soul_id,
                user_id=uid,
            )
        try:
            _write_conversation_state(
                conversation_id,
                soul_id=soul_id,
                user_id=uid,
                updates={
                    "last_consolidation_error": f"{type(exc).__name__}: {str(exc)[:260]}",
                    "last_consolidation_error_at": datetime.now(UTC).isoformat(),
                },
            )
        except Exception:
            logger.exception("failed to record consolidation error state for %s", conversation_id)
        if progress_key and memorize_progress is not None:
            _memorize_endpoint._set_memorize_progress(
                memorize_progress,
                progress_key,
                active=False,
                last_result="failure",
                error=f"{type(exc).__name__}: {exc}",
            )
        return {"ok": False, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


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
        run_consolidation_task=_run_consolidation_task,
        get_config=lambda: _CONFIG,
        sanitize_db_filename=_sanitize_db_filename,
    )


async def _run_memorize_episodes(
    *,
    memorize_segments: list[tuple[str, list[dict[str, Any]], int, int]],
    svc: Any,
    scope: dict[str, Any],
    conversation_id: str | None,
    soul_id: str,
    uid: str,
    processed_cursor: int,
    safe: dict[str, Any],
    resource_url: str,
    chat_key: str | None,
    tz_name: str | None,
    prev_len: int,
    merged_len: int,
    force: bool,
    sleep_stats: Any,
    segments_dir: Path,
    zi: Any = None,
    cross_memorize: bool = False,
    final_cursors: dict[str, int] | None = None,
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
        chat_key=chat_key,
        tz_name=tz_name,
        prev_len=prev_len,
        merged_len=merged_len,
        force=force,
        sleep_stats=sleep_stats,
        run_ctx=_make_memorize_run_context(),
        segments_dir=segments_dir,
        zi=zi,
        cross_memorize=cross_memorize,
        final_cursors=final_cursors,
    )


# ---- Memorize endpoint ----

@app.post("/memorize", operation_id="memorize")
async def memorize(payload: dict[str, Any], background_tasks: BackgroundTasks, force: bool = False, tail: bool = False):
    return await _memorize_endpoint.memorize_endpoint(
        payload,
        background_tasks,
        force,
        tail=tail,
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
        scope = _extract_scope(safe)

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
        result = out.get("result") or {}

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
        utility_max_tokens=None,
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
        _raise_upstream_http_error(exc, op="retrieve")


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
        scope = _extract_scope(safe)
        uid = str(scope.get("user_id") or "").strip()
        soul_id = str(scope.get("soul_id") or "").strip()
        message = _pick_str(safe, "message", "query") or ""
        history = _normalize_turn_history(safe.get("history"))
        state_row: dict[str, Any] | None = None
        cross_tail: list[dict[str, Any]] = []
        if uid and soul_id and message.strip():
            state_row, _soul_card, _db_path = _load_turn_state_and_soul_card(
                cid,
                user_id=uid,
                soul_id=soul_id,
            )
            if _db_path is not None and _db_path.exists():
                _con = _sqlite_connect(_db_path)
                try:
                    _con.row_factory = sqlite3.Row
                    _sqlite_ensure_conversation_state_schema(_con)
                    cross_tail = _load_cross_tail_from_sources(_con, exclude_conversation_id=cid)
                finally:
                    _con.close()

        chat_name_for_prompt = str(safe.get("chat_name") or "").strip()
        chat_type_for_prompt = str(safe.get("chat_type") or "").strip()
        if chat_name_for_prompt and chat_type_for_prompt:
            chat_label_for_prompt = f"[{chat_type_for_prompt}][{chat_name_for_prompt}]"
        elif chat_name_for_prompt:
            chat_label_for_prompt = f"[{chat_name_for_prompt}]"
        else:
            chat_label_for_prompt = None

        if safe.get("queries") is None and uid and soul_id and message.strip() and state_row is not None:
            safe["queries"] = _build_retrieve_soul_context_queries(
                soul_id=soul_id,
                message=message,
                history=history,
                state_row=state_row,
                conversation_id=cid,
                chat_label=chat_label_for_prompt,
            )

        if cross_tail:
            cross_text = _message_log.format_merged_history(cross_tail)
            safe["_cross_conversation_history"] = cross_text
            queries = safe.get("queries")
            if isinstance(queries, list):
                for i, query in enumerate(queries):
                    if isinstance(query, dict) and str(query.get("role") or "").strip() == "history":
                        current_history = str(query.get("content", {}).get("text", "") if isinstance(query.get("content"), dict) else "").strip()
                        section_header = _section_title_from_conversation_id(cid)
                        # Strip embedded section header — _merge expects a raw chat block
                        if current_history.startswith(section_header):
                            current_history = current_history[len(section_header):].strip()
                        merged = _merge_current_into_conversations(cross_text, current_history, section_header)
                        queries[i] = {"role": "history", "content": {"text": merged}}
                        break
                else:
                    queries.insert(-1, {"role": "history", "content": {"text": cross_text}})

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
                # Canonical current-chat history is assembled once above
                # (DB unmemorized tail + floor backfill) and reused here so
                # retrieve + turn see the same context.
                turn_history = history
                memory_cache = _normalize_memory_cache_impl(out.get("memory_cache"))
                intentions_active = _normalize_intentions_stack_impl(out.get("intentions_active"))

                out["turn_system_prompt"] = _make_turn_system_prompt(
                    soul_id,
                    soul_card=soul_card,
                    response_sentences=int(_CONFIG.get("turn_response_sentences", 3)),
                )
                out["turn_user_prompt"] = _build_turn_prompt(
                    user_message=message,
                    history=turn_history,
                    prior_context=out.get("prior_context"),
                    retrieve_rag=out.get("result"),
                    all_categories_summary=_state_row.get("all_categories_summary"),
                    memory_cache=memory_cache,
                    intentions_active=intentions_active,
                    cross_conversation_history=safe.get("_cross_conversation_history"),
                    chat_label=chat_label_for_prompt,
                    conversation_id=cid,
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
        _raise_upstream_http_error(exc, op="conversation.retrieve")


# ---- Turn state helpers + conversation turn endpoints ----

def _build_cross_conversation_payload(
    cid: str,
    uid: str,
    soul_id: str,
    safe: dict[str, Any],
    history_full: list[dict[str, Any]],
    digest_cursor: int,
    trigger_memorize_default: bool = True,
) -> dict[str, Any] | None:
    """Merge unmemorized tails from all conversations into one memorize payload."""
    db_path = _sqlite_current_path(uid, soul_id)
    if db_path is None or not db_path.exists():
        return None

    trigger_label = _message_log.derive_source_label(cid)
    trigger_memorize_raw = safe.get("memorize_chat")
    trigger_memorize = trigger_memorize_raw if isinstance(trigger_memorize_raw, bool) else trigger_memorize_default
    trigger_cursor = max(0, digest_cursor + 1)
    trigger_tail = _normalize_conversation(history_full[trigger_cursor:]) if trigger_cursor < len(history_full) else []
    if not trigger_tail:
        return None

    for i, msg in enumerate(trigger_tail):
        msg["source_label"] = trigger_label
        msg["source_conversation_id"] = cid
        msg["source_conversation_index"] = digest_cursor + 1 + i
        msg["memorize_chat"] = trigger_memorize
        ts = msg.get("ts_ms")
        if isinstance(ts, (int, float)) and "received_at" not in msg:
            msg["received_at"] = datetime.fromtimestamp(ts / 1000.0, tz=UTC).isoformat()

    final_cursors: dict[str, int] = {cid: digest_cursor + len(trigger_tail)}
    all_messages = list(trigger_tail)

    con = _sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        other_tails = _message_log.read_all_tails_for_memorize(con, exclude_conversation_id=cid)
        rolling_summaries = _message_log.read_background_rolling_summaries(
            con,
            exclude_conversation_id=cid,
        )
    finally:
        con.close()

    for other_cid, tail_msgs in other_tails.items():
        if not tail_msgs:
            continue
        final_cursors[other_cid] = tail_msgs[-1]["source_conversation_index"]
        all_messages.extend(tail_msgs)

    all_messages.sort(key=lambda m: m.get("received_at") or "")

    return {
        **safe,
        "conversation_id": cid,
        "conversation": all_messages,
        "user": {"user_id": uid, "soul_id": soul_id, "conversation_id": cid},
        "_cross_memorize": True,
        "_final_cursors": final_cursors,
        "_background_rolling_summaries": rolling_summaries,
    }


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
    chat_is_primary = bool(conversation_state.get("memorize_chat", True))
    primary_history = history_full if chat_is_primary else []
    unmemorized_tokens = _estimate_unmemorized_tokens(primary_history, unmemorized_digest_cursor)
    queued_memorize_payload: dict[str, Any] | None = None
    if (not dry_run) and primary_history and unmemorized_tokens >= _MIN_CHUNK_TOKENS:
        has_sleep_gap = _unmemorized_sleep_gap_detected(
            primary_history,
            unmemorized_digest_cursor,
            safe,
        )
        if has_sleep_gap:
            queued_memorize_payload = _build_cross_conversation_payload(
                cid,
                uid,
                soul_id,
                safe,
                primary_history,
                unmemorized_digest_cursor,
                True,
            )
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
    retrieval_ids_since_consolidation: list[str],
    memorize_chat: bool | None = None,
) -> tuple[dict[str, Any], Any]:
    latest_state_row, _, _ = _load_turn_state_and_soul_card(cid, user_id=uid, soul_id=soul_id)
    current_memory_cache = _normalize_memory_cache_impl(latest_state_row.get("memory_cache"))
    current_intentions = _normalize_intentions_stack_impl(latest_state_row.get("intentions_active"))
    intentions_snapshot = current_intentions
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
    updates: dict[str, Any] = {
        "intentions_active": next_intentions,
        "memory_cache": next_memory_cache,
        "undo_snapshot": {
            "memory_cache": current_memory_cache,
            "intentions_active": intentions_snapshot,
        },
    }
    if retrieval_ids_since_consolidation:
        updates["append_retrieval_ids_since_consolidation"] = retrieval_ids_since_consolidation
    if isinstance(memorize_chat, bool):
        updates["memorize_chat"] = memorize_chat
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
) -> str:
    cadence_threshold = _apimw_cadence_from_cfg(_CONFIG)
    cadence_soul_messages = _count_soul_messages(history_full, soul_id)
    if cadence_soul_messages < cadence_threshold:
        return "skipped_cadence"
    if cadence_threshold > 1 and (cadence_soul_messages % cadence_threshold) != 0:
        return "skipped_cadence"
    if not _mark_apimw_inflight(cid):
        return "skipped_inflight"
    apimw_state_row, _apimw_soul_card, _apimw_db_path = _load_turn_state_and_soul_card(
        cid,
        user_id=uid,
        soul_id=soul_id,
    )
    try:
        apimw_task = asyncio.create_task(
            _run_apimw(
                safe,
                conversation_id=cid,
                soul_id=soul_id,
                user_id=uid,
                state_row=apimw_state_row,
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


@app.post("/conversation/{conversation_id}/messages/append", operation_id="conversation_append_message")
async def conversation_append_message(
    conversation_id: str,
    payload: dict[str, Any] = Body(...),
):
    """Append a single message to the per-conversation messages table.

    Used by Hermes' listen-only policy: ingest the message into memU so it
    flows into memorize and cross-chat context, without engaging the soul
    for a response. No retrieve, no turn, no LLM calls — just a write.
    """
    cid = str(conversation_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="conversation_id is required")
    safe = _safe_payload(payload)
    scope = _extract_scope(safe)
    uid = str(scope.get("user_id") or "").strip()
    soul_id = str(scope.get("soul_id") or "").strip()
    if not uid or not soul_id:
        raise HTTPException(status_code=400, detail="user_id and soul_id required")
    message = _pick_str(safe, "message", "content") or ""
    if not message.strip():
        raise HTTPException(status_code=400, detail="message is required")
    user_name = _pick_str(safe, "user_name") or ""
    role = (_pick_str(safe, "role") or "user").strip()
    chat_name_for_append = _pick_str(safe, "chat_name") or None
    memorize_chat_raw = safe.get("memorize_chat")
    memorize_chat = memorize_chat_raw if isinstance(memorize_chat_raw, bool) else None

    write_updates: dict[str, Any] = {}
    if isinstance(memorize_chat, bool):
        write_updates["memorize_chat"] = memorize_chat
    _write_conversation_state(
        cid,
        soul_id=soul_id,
        user_id=uid,
        updates=write_updates,
    )

    _state_row, _soul_card, db_path = _load_turn_state_and_soul_card(
        cid, user_id=uid, soul_id=soul_id,
    )
    if db_path is None or not db_path.exists():
        raise HTTPException(status_code=404, detail="conversation state not found")
    _con = _sqlite_connect(db_path)
    try:
        _con.row_factory = sqlite3.Row
        _sqlite_ensure_conversation_state_schema(_con)
        msg: dict[str, Any] = {"role": role, "content": message}
        if user_name:
            msg["name"] = user_name
        ext_msg_id = _pick_str(safe, "external_message_id") or None
        if ext_msg_id:
            msg["external_message_id"] = ext_msg_id
        appended = _message_log.append_messages(_con, cid, [msg], chat_name=chat_name_for_append)
        _con.commit()
    finally:
        _con.close()
    if int(appended) > 0:
        _queue_background_rollup_task(
            conversation_id=cid,
            user_id=uid,
            soul_id=soul_id,
            safe_payload=safe,
        )
    return {"ok": True, "conversation_id": cid, "appended": int(appended)}


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

        scope = _extract_scope(safe)

        uid = str(scope.get("user_id") or "").strip()
        soul_id = str(scope.get("soul_id") or "").strip()
        if not uid or not soul_id:
            raise HTTPException(status_code=400, detail="user_id and soul_id required")

        message = str(safe.get("message") or "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="message is required")

        history_full = _normalize_turn_history(safe.get("history"))
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
        run_apimw = _retrieve_apimw_enabled_from_cfg(_CONFIG)
        include_debug = bool(safe.get("debug", False))
        memorize_chat_raw = safe.get("memorize_chat")
        memorize_chat = memorize_chat_raw if isinstance(memorize_chat_raw, bool) else None
        if dry_run:
            run_apimw = False

        safe["user"] = {"user_id": uid, "soul_id": soul_id, "conversation_id": cid}
        safe["conversation_id"] = cid
        trace_id_raw = safe.get("trace_id")
        if trace_id_raw is not None and not isinstance(trace_id_raw, str):
            raise HTTPException(status_code=400, detail="'trace_id' must be a string")
        trace_id = str(trace_id_raw or "").strip() or None

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

        turn_system_prompt = override_system_prompt or _make_turn_system_prompt(
            soul_id,
            soul_card=soul_card,
            response_sentences=int(_CONFIG.get("turn_response_sentences", 3)),
        )
        turn_user_prompt = override_user_prompt

        memory_service = _get_service_from_payload(safe)
        turn_started_at = time.monotonic()
        turn_contract: dict[str, Any] | None = None
        for attempt in (1, 2):
            turn_response_raw = await memory_service.chat(
                turn_user_prompt,
                system_prompt=turn_system_prompt,
                temperature=turn_temperature,
                response_format=turn_response_format,
                op="turn",
                step="respond" if attempt == 1 else "respond_retry",
                trace_id=trace_id,
            )
            try:
                turn_contract = _parse_turn_contract(turn_response_raw)
                break
            except Exception as exc:
                raw_snippet = str(turn_response_raw or "")[:200]
                if attempt == 1:
                    logger.warning(
                        "conversation_turn: turn contract parse failed on attempt 1; retrying once",
                    )
                    continue
                raise HTTPException(
                    status_code=502,
                    detail=f"turn contract parse failure: {exc}; raw={raw_snippet!r}",
                ) from exc
        if turn_contract is None:
            raise HTTPException(status_code=502, detail="turn contract parse failure: unknown")
        turn_ms = int((time.monotonic() - turn_started_at) * 1000)

        turn_cache_entry = str(turn_contract.get("cache_entry") or "").strip()
        turn_annulments = turn_contract.get("annulments") if isinstance(turn_contract.get("annulments"), list) else []
        normalized_annulments = [row for row in turn_annulments if isinstance(row, dict)]
        turn_annulment_ids = [
            str(row.get("intention_id") or "").strip()
            for row in normalized_annulments
        ]

        conversation_state_after = conversation_state_before
        conversation_state_path = db_path
        annulment_memory_ids: list[str] = []
        turn_appended_count = 0

        if not dry_run:
            async with state_lock:
                retrieved_item_ids = _extract_result_item_ids(override_retrieve_rag)
                conversation_state_after, conversation_state_path = _turn_state_write(
                    cid, uid, soul_id,
                    turn_cache_entry, turn_annulment_ids,
                    retrieved_item_ids,
                    memorize_chat=memorize_chat,
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
                cid, uid, soul_id, safe, history_full,
            )

        response_target = str(turn_contract.get("response_target") or "").strip().lower()
        if response_target not in {"respond", "listen", "private"}:
            raise HTTPException(status_code=502, detail="turn contract missing or invalid response_target")
        response_text = str(turn_contract.get("response") or "").strip()

        # Enforce response_target contract:
        # - listen: nothing is sent.
        # - respond: if chat_name is missing, proceed but log loudly.
        # - private: passes through; routing to the human's private chat is
        #   hermes-side (see HANDOFF for the wiring task).
        if response_target == "listen":
            response_text = ""
        elif response_target == "respond":
            chat_name = str(safe.get("chat_name") or "").strip()
            if not chat_name:
                logger.warning(
                    "conversation_turn: missing chat_name for respond; continuing without chat label"
                )
        if not dry_run and conversation_state_path is not None and conversation_state_path.exists():
            user_name = str(safe.get("user_name") or "").strip() or uid
            chat_name_for_append = str(safe.get("chat_name") or "").strip() or None
            current_user_msg: dict[str, Any] = {"role": "user", "content": message}
            if user_name:
                current_user_msg["name"] = user_name
            ext_msg_id = _pick_str(safe, "external_message_id") or None
            if ext_msg_id:
                current_user_msg["external_message_id"] = ext_msg_id
            append_rows: list[dict[str, Any]] = [current_user_msg]
            if response_text and response_target == "respond":
                append_rows.append(
                    {"role": "assistant", "name": soul_id, "content": response_text}
                )
            _con = _sqlite_connect(conversation_state_path)
            try:
                _con.row_factory = sqlite3.Row
                _sqlite_ensure_conversation_state_schema(_con)
                turn_appended_count = _message_log.append_messages(
                    _con,
                    cid,
                    append_rows,
                    chat_name=chat_name_for_append,
                )
                _con.commit()
            finally:
                _con.close()
        if (not dry_run) and turn_appended_count > 0:
            _queue_background_rollup_task(
                conversation_id=cid,
                user_id=uid,
                soul_id=soul_id,
                safe_payload=safe,
                service=memory_service,
            )

        response_payload: dict[str, Any] = {
            "ok": True,
            "conversation_id": cid,
            "response": response_text,
            "response_target": response_target,
            "apimw": apimw_status,
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
        if trace_id:
            response_payload["trace_id"] = trace_id
        background_error = str((conversation_state_after or {}).get("last_background_error") or "").strip()
        if background_error:
            response_payload["background_error"] = background_error
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
        _raise_upstream_http_error(exc, op="conversation.turn")


@app.post("/conversation/{conversation_id}/turn/undo", operation_id="conversation_turn_undo")
async def conversation_turn_undo(
    conversation_id: str,
    payload: dict[str, Any] = Body(...),
):
    cid = str(conversation_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="conversation_id is required")

    safe = _safe_payload(payload)
    scope = _extract_scope(safe)

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


def _persist_inbound_user_message(
    conversation_id: str,
    user_id: str,
    soul_id: str,
    message: str,
    user_name: str | None = None,
    chat_name: str | None = None,
    external_message_id: str | None = None,
) -> None:
    _write_conversation_state(conversation_id, soul_id=soul_id, user_id=user_id, updates={})
    _, _, db_path = _load_turn_state_and_soul_card(conversation_id, user_id=user_id, soul_id=soul_id)
    if db_path is None or not db_path.exists():
        return
    msg: dict[str, Any] = {"role": "user", "content": message}
    speaker = str(user_name or "").strip() or user_id
    if speaker:
        msg["name"] = speaker
    if external_message_id:
        msg["external_message_id"] = external_message_id
    con = _sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        _sqlite_ensure_conversation_state_schema(con)
        _message_log.append_messages(con, conversation_id, [msg], chat_name=chat_name)
        con.commit()
    finally:
        con.close()


@app.post("/integration/memu/turn", operation_id="memu_turn", tags=["mcp_tools"])
async def mcp_memu_turn(req: _mcp_tools.MemuTurnRequest):
    return await _mcp_tools.memu_turn_endpoint(
        req,
        conversation_retrieve=conversation_retrieve,
        conversation_turn=conversation_turn,
        persist_user_message=_persist_inbound_user_message,
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
