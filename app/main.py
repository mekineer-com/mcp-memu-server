import asyncio
import json
import logging
import os
import re
import signal
import sys
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
from pydantic import BaseModel, Field

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
from app.services import activity_messages as _activity_messages
from app.services import apimw as _apimw
from app.services import cross_history as _cross_history
from app.services import free_turn as _free_turn
from app.services import whatsapp_outbounds as _whatsapp_outbounds
from app.services.consolidation import (
    ConsolidationDeps,
    gather_consolidation_inputs as _gather_consolidation_inputs,
    run_consolidation_llm as _run_consolidation_llm,
    write_consolidation_outputs as _write_consolidation_outputs,
)
from app.services.intention_state import (
    append_memory_cache_entry as _append_memory_cache_entry,
    apply_intention_turn_maintenance as _apply_intention_turn_maintenance_impl,
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
    _normalize_conversation,
    _normalize_turn_history,
    _parse_as_of_datetime,
    _parse_turn_ts_ms,
    _payload_signature,
    _pick_str,
    _safe_payload,
)
from app.services import sqlite_scope as _sqlite_scope
from app.services.state import (
    conversation_state_empty as _conversation_state_empty,
    conversation_state_from_row as _conversation_state_from_row_impl,
    conversation_state_row as _conversation_state_row,
    effective_digest_cursor_from_row as _effective_digest_cursor_from_row,
    memorize_chat_from_row as _memorize_chat_from_row,
    write_conversation_state as _write_conversation_state_impl,
)
from app.services.turn_contract import (
    build_conversations_block as _build_conversations_block,
    build_turn_context_block as _build_turn_context_block,
    build_turn_prompt as _build_turn_prompt,
    format_memory_line as _format_memory_line,
    format_memory_legend as _format_memory_legend,
    format_shaped_by_line as _format_shaped_by_line,
    format_time_anchor as _format_time_anchor,
    make_turn_identity_prompt as _make_turn_identity_prompt,
    make_turn_system_prompt as _make_turn_system_prompt,
    parse_turn_contract as _parse_turn_contract,
)


# ==== Module state & constants ====

logger = logging.getLogger(__name__)
_apimw._main = sys.modules[__name__]
_cross_history._main = sys.modules[__name__]
_message_log._main = sys.modules[__name__]
_PROMPT_LOGGER = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    global _FREE_TURN_FOLLOW_UP_TASK
    if _FREE_TURN_FOLLOW_UP_TASK is None or _FREE_TURN_FOLLOW_UP_TASK.done():
        _FREE_TURN_FOLLOW_UP_TASK = asyncio.create_task(_free_turn_followup_scheduler())
    try:
        yield
    finally:
        task = _FREE_TURN_FOLLOW_UP_TASK
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        _FREE_TURN_FOLLOW_UP_TASK = None


app = FastAPI(title="mcp-memu-server", version="0.4.0", lifespan=_app_lifespan)


class AtomicSessionStartRequest(BaseModel):
    user_id: str
    soul_id: str
    conversation_id: str | None = None
    message: str | None = None
    soul_card: str | None = None
    debug: bool = False


class AtomicSessionEndRequest(BaseModel):
    user_id: str
    soul_id: str
    conversation_id: str
    activity_recap: str | None = None
    transcript: list[dict[str, Any]] = Field(default_factory=list)


class AtomicPromptLogRequest(BaseModel):
    conversation_id: str | None = None
    model: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)


_BUILD_ID: str = "fix48.debloat.bloatRemoval.concepts"
_SLEEP_SPLIT_MIN_LULL_SECONDS: int = 3 * 60 * 60
_DEFAULT_MIN_CHUNK_TOKENS: int = 8000
_DEFAULT_EPISODE_ITEMS_PER_SEGMENT: int = 3
_DEFAULT_BACKGROUND_SUMMARY_TOKENS: int = 1000
_MIN_CHUNK_TOKENS: int = _DEFAULT_MIN_CHUNK_TOKENS
_EPISODE_ITEMS_PER_SEGMENT: int = _DEFAULT_EPISODE_ITEMS_PER_SEGMENT
_BACKGROUND_SUMMARY_TOKENS: int = _DEFAULT_BACKGROUND_SUMMARY_TOKENS
# Uniform runaway-protection caps for LLM calls. Not business logic —
_BACKGROUND_TASKS: set[asyncio.Task] = set()  # prevent GC of fire-and-forget tasks
_LOG_PROMPTS: bool = False


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
    min_chunk_tokens: int,
) -> bool:
    return _memorize_endpoint.unmemorized_sleep_gap_detected(
        history,
        digest_cursor=-1,
        logger=logger,
        min_chunk_tokens=min_chunk_tokens,
        sleep_split_min_lull_seconds=_SLEEP_SPLIT_MIN_LULL_SECONDS,
    )


def _load_background_rollup_tail(
    *,
    conversation_id: str,
    user_id: str,
    soul_id: str,
    rolling_summary_cursor_id: int | None,
    min_timestamp: int | None = None,
) -> list[dict[str, Any]]:
    storage_dir, hermes_home_path, sessions_index_path, state_db_path = _resolve_cross_source_paths()
    source_label = _message_log.derive_source_label(conversation_id)
    if source_label.startswith("whatsapp:"):
        active_since = _load_soul_active_since(
            soul_id,
            hermes_home_path=hermes_home_path,
            state_db_path=state_db_path,
        )
        whatsapp_source, web_source_db_path, reply_prefix = _resolve_whatsapp_source_config()
        if whatsapp_source == "web_source":
            assistant_ids = _conversation_sources.load_whatsapp_assistant_source_message_ids(
                conversation_id=conversation_id,
                hermes_home=hermes_home_path,
                sessions_index_path=sessions_index_path,
                state_db_path=state_db_path,
            )
            tail = _conversation_sources.load_whatsapp_web_source_tail_after_rowid(
                conversation_id=conversation_id,
                after_rowid=rolling_summary_cursor_id,
                soul_id=soul_id,
                reply_prefix=reply_prefix,
                hermes_home=hermes_home_path,
                web_source_db_path=web_source_db_path,
                min_timestamp=max(
                    (value for value in (active_since, min_timestamp) if value is not None),
                    default=None,
                ),
                assistant_source_message_ids=assistant_ids,
            )
            _stamp_assistant_display_name(tail, soul_id)
            return tail
        tail = _conversation_sources.load_whatsapp_tail_after_message_id(
            conversation_id=conversation_id,
            after_message_id=rolling_summary_cursor_id,
            hermes_home=hermes_home_path,
            sessions_index_path=sessions_index_path,
            state_db_path=state_db_path,
            min_timestamp=active_since,
        )
        _stamp_assistant_display_name(tail, soul_id)
        return tail
    if source_label == "sillytavern":
        tail = _conversation_sources.load_sillytavern_tail(
            storage_dir=storage_dir,
            user_id=user_id,
            soul_id=soul_id,
            conversation_id=conversation_id,
            since_cursor=int(rolling_summary_cursor_id) if rolling_summary_cursor_id is not None else -1,
            recent_fallback_messages=0,
        )
        _stamp_assistant_display_name(tail, soul_id)
        return tail
    return []


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
        state_row, _, db_path = _load_turn_state_and_soul_card(
            cid,
            user_id=uid,
            soul_id=sid,
        )
        if bool(state_row.get("memorize_chat", True)):
            return "skipped_primary_chat"
        if db_path is None or not db_path.exists():
            return "skipped_no_db"

        rolling_cursor_id = state_row.get("rolling_summary_cursor_id")
        _, hermes_home_path, _, _ = _resolve_cross_source_paths()
        rolling_cursor_id, min_timestamp, web_source = _resolve_source_cursor(
            cid,
            int(rolling_cursor_id or 0),
            state_row.get("rolling_summary_cursor_source_message_id"),
            state_row.get("rolling_summary_cursor_ts"),
            rolling=True,
            hermes_home_path=hermes_home_path,
        )
        tail = _load_background_rollup_tail(
            conversation_id=cid,
            user_id=uid,
            soul_id=sid,
            rolling_summary_cursor_id=rolling_cursor_id,
            min_timestamp=min_timestamp,
        )
        if len(tail) < 2:
            return "skipped_short_tail"

        tail_end_cursor = int(tail[-1].get("source_conversation_index") or 0)
        if tail_end_cursor <= 0:
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
        if token_estimate < int(_BACKGROUND_SUMMARY_TOKENS):
            return "skipped_tokens"
        if not _background_sleep_gap_detected(
            history=sleep_history,
            safe=safe_payload,
            min_chunk_tokens=int(_BACKGROUND_SUMMARY_TOKENS),
        ):
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
                "_message_index": int(msg.get("source_conversation_index") or 0),
            }
            for msg in tail
        ]
        new_summary = str(
            await llm_service.summarize_background_chat_rollup(
                prior_summary=prior_summary,
                messages=summary_input,
                soul_name=sid,
            )
            or ""
        ).strip()
        if not new_summary:
            raise RuntimeError("background summarize returned empty summary")
        now_iso = datetime.now(UTC).isoformat()
        updates = {
            "rolling_summary": new_summary,
            "rolling_summary_cursor_id": tail_end_cursor,
            "rolling_summary_updated_at": now_iso,
            "updated_at": now_iso,
            "last_background_error": None,
            "last_background_error_at": None,
        }
        if web_source:
            checkpoint = _source_cursor_checkpoint(tail, web_source=True)
            if checkpoint is None:
                raise RuntimeError("WhatsApp web_source rollup has no checkpoint")
            updates.update(
                rolling_summary_cursor_id=checkpoint["cursor"],
                rolling_summary_cursor_source_message_id=checkpoint["source_message_id"],
                rolling_summary_cursor_ts=checkpoint["ts"],
            )
        _write_conversation_state(
            cid,
            soul_id=sid,
            user_id=uid,
            updates=updates,
        )
        return "rolled_up"


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
        except Exception as exc:
            _set_background_error(
                conversation_id,
                soul_id=soul_id,
                user_id=user_id,
                code="background_rollup_failed",
                detail=f"{type(exc).__name__}: {str(exc)[:220]}",
            )
            logger.exception("background rollup task failed for %s", marker)
        finally:
            _BACKGROUND_TASKS.discard(done_task)
            _clear_inflight(_BACKGROUND_ROLLUP_INFLIGHT, marker)

    task.add_done_callback(_on_done)


def _build_free_turn_prompt(
    *,
    reason: str,
    continuation_index: int,
    origin_conversation_id: str,
    previous_contract: dict[str, Any],
    allow_public_response: bool,
) -> str:
    return _free_turn._build_free_turn_prompt(
        reason=reason,
        continuation_index=continuation_index,
        origin_conversation_id=origin_conversation_id,
        previous_contract=previous_contract,
        allow_public_response=allow_public_response,
    )


def _attachment_workspace() -> str | None:
    return _free_turn._attachment_workspace(_CONFIG)


def _parse_free_turn_contract(raw: Any, *, allow_public_response: bool) -> dict[str, Any]:
    return _free_turn._parse_free_turn_contract(
        raw,
        allow_public_response=allow_public_response,
        config=_CONFIG,
        parse_turn_contract=_parse_turn_contract,
        logger=logger,
    )


def _turn_generation_metadata(payload: dict[str, Any]) -> dict[str, str]:
    return _free_turn._turn_generation_metadata(payload, config=_CONFIG)


async def _run_free_turn_chain(
    *,
    marker: str,
    service: MemoryService,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    session_id: str,
    initial_reason: str,
    initial_contract: dict[str, Any],
    system_prompt: str,
    allow_public_response: bool,
    safe_payload: dict[str, Any],
    soul_card: str | None,
    system_prompt_has_activity_recap: bool = False,
) -> None:
    await _free_turn._run_free_turn_chain(
        marker=marker,
        service=service,
        user_id=user_id,
        soul_id=soul_id,
        conversation_id=conversation_id,
        session_id=session_id,
        initial_reason=initial_reason,
        initial_contract=initial_contract,
        system_prompt=system_prompt,
        allow_public_response=allow_public_response,
        safe_payload=safe_payload,
        soul_card=soul_card,
        system_prompt_has_activity_recap=system_prompt_has_activity_recap,
        config=_CONFIG,
        make_turn_system_prompt=_make_turn_system_prompt,
        parse_free_turn_contract=_parse_free_turn_contract,
        record_activity_message=_record_activity_message,
        activity_recap_from_contract=_activity_recap_from_contract,
        insert_whatsapp_outbound=_insert_whatsapp_outbound,
        schedule_free_turn_follow_up=_schedule_free_turn_follow_up,
        clear_inflight=_clear_inflight,
        free_turn_inflight=_FREE_TURN_INFLIGHT,
        logger=logger,
    )


def _queue_free_turn_chain(
    *,
    service: MemoryService,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    session_id: str,
    initial_reason: str,
    initial_contract: dict[str, Any],
    system_prompt: str,
    allow_public_response: bool,
    safe_payload: dict[str, Any],
    soul_card: str | None,
    system_prompt_has_activity_recap: bool = False,
) -> bool:
    return _free_turn._queue_free_turn_chain(
        service=service,
        user_id=user_id,
        soul_id=soul_id,
        conversation_id=conversation_id,
        session_id=session_id,
        initial_reason=initial_reason,
        initial_contract=initial_contract,
        system_prompt=system_prompt,
        allow_public_response=allow_public_response,
        safe_payload=safe_payload,
        soul_card=soul_card,
        system_prompt_has_activity_recap=system_prompt_has_activity_recap,
        mark_inflight=_mark_inflight,
        free_turn_inflight=_FREE_TURN_INFLIGHT,
        background_tasks=_BACKGROUND_TASKS,
        run_free_turn_chain=_run_free_turn_chain,
        logger=logger,
    )


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
_FORCED_MEMORIZE_INFLIGHT: set[str] = set()
_FREE_TURN_INFLIGHT: set[str] = set()
_FREE_TURN_FOLLOW_UP_INFLIGHT: set[str] = set()
_FREE_TURN_FOLLOW_UP_TASK: asyncio.Task | None = None
_SHUTDOWN_STATE: dict[str, Any] = {
    "draining": False,
    "stopping": False,
    "requestedAtUnix": None,
    "requestedBy": None,
    "reason": None,
    "maxWaitSec": 0,
    "timedOut": False,
}



def _mark_apimw_inflight(*args: Any, **kwargs: Any) -> Any:
    return _apimw._mark_apimw_inflight(*args, **kwargs)

def _clear_apimw_inflight(*args: Any, **kwargs: Any) -> Any:
    return _apimw._clear_apimw_inflight(*args, **kwargs)

def _parse_apimw_json_response(*args: Any, **kwargs: Any) -> Any:
    return _apimw._parse_apimw_json_response(*args, **kwargs)

async def _apimw_retrieve_items(*args: Any, **kwargs: Any) -> Any:
    return await _apimw._apimw_retrieve_items(*args, **kwargs)

async def _apimw_collect_memory_items(*args: Any, **kwargs: Any) -> Any:
    return await _apimw._apimw_collect_memory_items(*args, **kwargs)

async def _apimw_synthesize(*args: Any, **kwargs: Any) -> Any:
    return await _apimw._apimw_synthesize(*args, **kwargs)

async def _apimw_persist(*args: Any, **kwargs: Any) -> Any:
    return await _apimw._apimw_persist(*args, **kwargs)

async def _run_apimw(*args: Any, **kwargs: Any) -> Any:
    return await _apimw._run_apimw(*args, **kwargs)

def _apimw_cadence_due(*args: Any, **kwargs: Any) -> Any:
    return _apimw._apimw_cadence_due(*args, **kwargs)

def _turn_launch_apimw(*args: Any, **kwargs: Any) -> Any:
    return _apimw._turn_launch_apimw(*args, **kwargs)



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
            "activeBackgroundTasks": _active_background_task_count(),
        }


def _active_background_task_count() -> int:
    return sum(1 for task in _BACKGROUND_TASKS if not task.done())


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
        active_background = _active_background_task_count()
        if active_work <= 0 and active_background <= 0:
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
    global _MIN_CHUNK_TOKENS, _EPISODE_ITEMS_PER_SEGMENT, _BACKGROUND_SUMMARY_TOKENS
    global _LOG_PROMPTS
    memorize_cfg = _CONFIG.get("memorize") if isinstance(_CONFIG.get("memorize"), dict) else {}
    try:
        _MIN_CHUNK_TOKENS = max(0, int(memorize_cfg.get("min_chunk_tokens", _DEFAULT_MIN_CHUNK_TOKENS)))
    except (TypeError, ValueError, OverflowError):
        _MIN_CHUNK_TOKENS = _DEFAULT_MIN_CHUNK_TOKENS
    try:
        _EPISODE_ITEMS_PER_SEGMENT = max(1, int(memorize_cfg.get("episode_items_per_segment", _DEFAULT_EPISODE_ITEMS_PER_SEGMENT)))
    except (TypeError, ValueError, OverflowError):
        _EPISODE_ITEMS_PER_SEGMENT = _DEFAULT_EPISODE_ITEMS_PER_SEGMENT
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


def _format_prompt_payload_for_log(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return json.dumps(payload, ensure_ascii=False, indent=2).replace("\\n", "\n")

    rest = {key: value for key, value in payload.items() if key != "messages"}
    lines: list[str] = []
    if rest:
        lines.append(json.dumps(rest, ensure_ascii=False, indent=2).replace("\\n", "\n"))
    lines.append("messages:")
    for idx, message in enumerate(messages, 1):
        if not isinstance(message, Mapping):
            lines.extend(["", f"[{idx}]", str(message)])
            continue
        lines.extend(["", f"[{idx}] role: {message.get('role') or '-'}"])
        if message.get("name") is not None:
            lines.append(f"name: {message['name']}")
        if message.get("content") is not None:
            lines.extend(["content:", str(message["content"])])
        extras = {key: value for key, value in message.items() if key not in {"role", "name", "content"}}
        if extras:
            lines.extend(["metadata:", json.dumps(extras, ensure_ascii=False, indent=2).replace("\\n", "\n")])
    return "\n".join(lines)


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
        payload_log_text = _format_prompt_payload_for_log(payload)
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
        payload_log_text = _format_prompt_payload_for_log(payload)
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


_resolve_profile_if_configured = _service_factory._resolve_profile_if_configured
_retrieve_apimw_enabled_from_cfg = _service_factory._retrieve_apimw_enabled_from_cfg
_apimw_cadence_from_cfg = _service_factory._apimw_cadence_from_cfg
_apimw_memory_count_from_cfg = _service_factory._apimw_memory_count_from_cfg
_apimw_random_count_from_cfg = _service_factory._apimw_random_count_from_cfg
_consolidation_interval_days_from_cfg = _service_factory._consolidation_interval_days_from_cfg
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
        episode_items_per_segment=_EPISODE_ITEMS_PER_SEGMENT,
        min_chunk_tokens=_MIN_CHUNK_TOKENS,
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
    return _sqlite_scope.intention_row_to_dict(row)


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


def _ensure_whatsapp_outbounds_schema(con: sqlite3.Connection) -> None:
    _whatsapp_outbounds._ensure_whatsapp_outbounds_schema(con)


def _whatsapp_outbound_row(row: sqlite3.Row) -> dict[str, Any]:
    return _whatsapp_outbounds._whatsapp_outbound_row(row)


def _sqlite_has_rows_quietly(
    db_path: Path,
    *,
    table: str,
    where_sql: str,
    params: tuple[Any, ...],
) -> bool:
    return _whatsapp_outbounds._sqlite_has_rows_quietly(
        db_path,
        table=table,
        where_sql=where_sql,
        params=params,
    )


def _poll_marker_path(db_path: Path, name: str) -> Path:
    return _whatsapp_outbounds._poll_marker_path(db_path, name)


def _touch_poll_marker(db_path: Path, name: str, value: str = "") -> None:
    _whatsapp_outbounds._touch_poll_marker(db_path, name, value)


def _remove_poll_marker(db_path: Path, name: str) -> None:
    _whatsapp_outbounds._remove_poll_marker(db_path, name)


def _poll_marker_due(db_path: Path, name: str, *, now: datetime) -> bool:
    return _whatsapp_outbounds._poll_marker_due(db_path, name, now=now)


def _insert_whatsapp_outbound(
    *,
    user_id: str,
    soul_id: str,
    origin_conversation_id: str,
    target: str,
    response_text: str,
    media_path: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    return _whatsapp_outbounds._insert_whatsapp_outbound(
        user_id=user_id,
        soul_id=soul_id,
        origin_conversation_id=origin_conversation_id,
        target=target,
        response_text=response_text,
        media_path=media_path,
        metadata=metadata,
        sqlite_current_path=_sqlite_current_path,
    )


def _activity_message_rows(
    con: sqlite3.Connection,
    *,
    user_id: str,
    soul_id: str,
    since_cursor: int,
    recent_fallback_messages: int,
) -> list[dict[str, Any]]:
    return _activity_messages.activity_message_rows(
        con,
        user_id=user_id,
        soul_id=soul_id,
        since_cursor=since_cursor,
        recent_fallback_messages=recent_fallback_messages,
    )


def _load_activity_tail_for_ai(
    con: sqlite3.Connection,
    *,
    user_id: str,
    soul_id: str,
) -> list[dict[str, Any]]:
    return _activity_messages.load_activity_tail_for_ai(
        con,
        user_id=user_id,
        soul_id=soul_id,
        recent_fallback_messages=TURN_HISTORY_WINDOW_MESSAGES,
    )


def _record_activity_message(
    *,
    user_id: str,
    soul_id: str,
    recap: str,
    happened_at: datetime | None = None,
) -> bool:
    return _activity_messages.record_activity_message(
        user_id=user_id,
        soul_id=soul_id,
        recap=recap,
        happened_at=happened_at,
        sqlite_current_path=_sqlite_current_path,
        logger=logger,
    )


def _activity_recap_from_contract(contract: dict[str, Any]) -> str:
    return _activity_messages.activity_recap_from_contract(contract)


def _claim_whatsapp_outbounds(
    *,
    user_id: str,
    soul_id: str,
    claimed_by: str,
    limit: int = 10,
    claim_timeout_seconds: int = 300,
) -> list[dict[str, Any]]:
    return _whatsapp_outbounds._claim_whatsapp_outbounds(
        user_id=user_id,
        soul_id=soul_id,
        claimed_by=claimed_by,
        limit=limit,
        claim_timeout_seconds=claim_timeout_seconds,
        sqlite_current_path=_sqlite_current_path,
    )


def _mark_whatsapp_outbound(
    *,
    user_id: str,
    soul_id: str,
    outbound_id: str,
    status: str,
    provider_message_id: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return _whatsapp_outbounds._mark_whatsapp_outbound(
        user_id=user_id,
        soul_id=soul_id,
        outbound_id=outbound_id,
        status=status,
        provider_message_id=provider_message_id,
        error=error,
        sqlite_current_path=_sqlite_current_path,
    )


def _ensure_free_turn_followups_schema(con: sqlite3.Connection) -> None:
    _free_turn._ensure_free_turn_followups_schema(con)


def _free_turn_followup_row(row: sqlite3.Row) -> dict[str, Any]:
    return _free_turn._free_turn_followup_row(row, json_from_db=_json_from_db)


def _parse_free_turn_follow_up_at(raw: str) -> datetime | None:
    return _free_turn._parse_free_turn_follow_up_at(
        raw,
        server_timezone=_memorize_endpoint.server_timezone,
    )


def _free_turn_followup_payload(safe: dict[str, Any]) -> dict[str, Any]:
    return _free_turn._free_turn_followup_payload(safe)


def _schedule_free_turn_follow_up(
    *,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    follow_up_at: str,
    follow_up_reason: str,
    safe_payload: dict[str, Any],
) -> str | None:
    return _free_turn._schedule_free_turn_follow_up(
        user_id=user_id,
        soul_id=soul_id,
        conversation_id=conversation_id,
        follow_up_at=follow_up_at,
        follow_up_reason=follow_up_reason,
        safe_payload=safe_payload,
        parse_free_turn_follow_up_at=_parse_free_turn_follow_up_at,
        sqlite_current_path=_sqlite_current_path,
        sqlite_ensure_nonempty=_sqlite_ensure_nonempty,
        sqlite_connect=_sqlite_connect,
        json_to_db=_json_to_db,
        touch_poll_marker=_touch_poll_marker,
        logger=logger,
    )


def _free_turn_followup_db_paths() -> list[Path]:
    return _free_turn._free_turn_followup_db_paths(
        storage_status=_STORAGE_STATUS,
        config=_CONFIG,
        sqlite_dir_from_cfg=_sqlite_dir_from_cfg,
        logger=logger,
    )


def _claim_due_free_turn_followups(
    db_path: Path,
    *,
    now: datetime,
    limit: int = 5,
    claim_timeout_seconds: int = 7200,
) -> list[dict[str, Any]]:
    return _free_turn._claim_due_free_turn_followups(
        db_path,
        now=now,
        limit=limit,
        claim_timeout_seconds=claim_timeout_seconds,
        json_from_db=_json_from_db,
        sqlite_connect=_sqlite_connect,
        poll_marker_due=_poll_marker_due,
        sqlite_has_rows_quietly=_sqlite_has_rows_quietly,
        remove_poll_marker=_remove_poll_marker,
    )


def _mark_free_turn_followup(
    db_path: Path,
    followup_id: str,
    *,
    status: str,
    error: str | None = None,
) -> None:
    _free_turn._mark_free_turn_followup(
        db_path,
        followup_id,
        status=status,
        error=error,
        sqlite_connect=_sqlite_connect,
    )


async def _run_free_turn_followup(row: dict[str, Any], db_path: Path) -> None:
    await _free_turn._run_free_turn_followup(
        row,
        db_path,
        mark_inflight=_mark_inflight,
        free_turn_follow_up_inflight=_FREE_TURN_FOLLOW_UP_INFLIGHT,
        conversation_retrieve=conversation_retrieve,
        conversation_turn=conversation_turn,
        build_prompt_override_payload=_mcp_tools.build_prompt_override_payload,
        insert_whatsapp_outbound=_insert_whatsapp_outbound,
        mark_free_turn_followup=_mark_free_turn_followup,
        clear_inflight=_clear_inflight,
        logger=logger,
    )


async def _run_due_free_turn_followups_once() -> int:
    return await _free_turn._run_due_free_turn_followups_once(
        free_turn_followup_db_paths=_free_turn_followup_db_paths,
        claim_due_free_turn_followups=_claim_due_free_turn_followups,
        run_free_turn_followup=_run_free_turn_followup,
    )


async def _free_turn_followup_scheduler() -> None:
    await _free_turn._free_turn_followup_scheduler(
        run_due_free_turn_followups_once=_run_due_free_turn_followups_once,
        logger=logger,
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

    event_at = datetime.now(UTC)
    summaries: list[str] = []
    for row in annulments:
        intention_id = str(row.get("intention_id") or "").strip()
        status = str(row.get("status") or "").strip().lower()
        if not intention_id or status not in {"completed", "deleted"}:
            continue
        note = str(row.get("note") or "").strip()
        intention_text = str((by_id.get(intention_id) or {}).get("text") or intention_id).strip() or intention_id
        summary = f'On {event_at.date().isoformat()}, I marked "{intention_text}" as {status}.'
        if note:
            summary = f"{summary} Note: {note}"
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
            happened_at=event_at,
            conversation_id=conversation_id,
        )
        created_ids.append(str(item.id))
    return created_ids


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
    "apimw_synthesis_parse_failed:",
    "apimw_failed:",
)


def _clear_background_error_if_apimw_owned(
    conversation_id: str,
    *,
    soul_id: str,
    user_id: str,
) -> None:
    state_row, _, _db_path = _load_turn_state_and_soul_card(
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
    _safe: dict[str, Any],
    *,
    min_chunk_tokens: int | None = None,
) -> bool:
    return _memorize_endpoint.unmemorized_sleep_gap_detected(
        history,
        digest_cursor,
        logger=logger,
        min_chunk_tokens=_MIN_CHUNK_TOKENS if min_chunk_tokens is None else int(min_chunk_tokens),
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


_ACTIVE_SINCE_UNSET = _cross_history._ACTIVE_SINCE_UNSET
TURN_HISTORY_WINDOW_MESSAGES = _cross_history.TURN_HISTORY_WINDOW_MESSAGES

def _resolve_cross_source_paths(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._resolve_cross_source_paths(*args, **kwargs)

def _resolve_whatsapp_source_config(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._resolve_whatsapp_source_config(*args, **kwargs)

def _resolve_whatsapp_history_limit(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._resolve_whatsapp_history_limit(*args, **kwargs)

def _load_soul_active_since(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._load_soul_active_since(*args, **kwargs)

def _current_whatsapp_active_since_for_soul(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._current_whatsapp_active_since_for_soul(*args, **kwargs)

def _filter_current_whatsapp_history_for_soul(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._filter_current_whatsapp_history_for_soul(*args, **kwargs)

def _degrade_live_whatsapp_history_after_filter_error(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._degrade_live_whatsapp_history_after_filter_error(*args, **kwargs)

def _source_id_matches_external(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._source_id_matches_external(*args, **kwargs)

def _filter_external_message_from_history(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._filter_external_message_from_history(*args, **kwargs)

def _load_current_whatsapp_history_from_source(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._load_current_whatsapp_history_from_source(*args, **kwargs)

def _prepare_current_whatsapp_history(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._prepare_current_whatsapp_history(*args, **kwargs)

def _stamp_assistant_display_name(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._stamp_assistant_display_name(*args, **kwargs)

def _persist_completed_sillytavern_turn_snapshot(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._persist_completed_sillytavern_turn_snapshot(*args, **kwargs)

def _load_tail_for_source_conversation(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._load_tail_for_source_conversation(*args, **kwargs)

def _resolve_source_cursor(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._resolve_source_cursor(*args, **kwargs)

def _latest_saved_segment_display_ranges(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._latest_saved_segment_display_ranges(*args, **kwargs)

def _load_cross_tail_from_sources(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._load_cross_tail_from_sources(*args, **kwargs)

def _clear_last_display_segments_for_nonparticipants(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._clear_last_display_segments_for_nonparticipants(*args, **kwargs)

def _chat_label_for_prompt(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._chat_label_for_prompt(*args, **kwargs)

def _chat_label_from_history(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._chat_label_from_history(*args, **kwargs)

def _format_cross_tail_for_ai(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._format_cross_tail_for_ai(*args, **kwargs)

def _format_all_chat_history_for_ai(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._format_all_chat_history_for_ai(*args, **kwargs)

def _load_cross_tail_for_ai(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._load_cross_tail_for_ai(*args, **kwargs)

def _load_cross_tail_with_activities_from_sources(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._load_cross_tail_with_activities_from_sources(*args, **kwargs)

def _load_cross_memorize_tails_from_sources(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._load_cross_memorize_tails_from_sources(*args, **kwargs)

def _read_background_rolling_summaries_from_conversations(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._read_background_rolling_summaries_from_conversations(*args, **kwargs)

def _turn_history_with_floor(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._turn_history_with_floor(*args, **kwargs)

def _source_cursor_checkpoint(*args: Any, **kwargs: Any) -> Any:
    return _cross_history._source_cursor_checkpoint(*args, **kwargs)


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
    current_chat_messages = [
        row for row in (prep.get("current_chat_messages") or [])
        if isinstance(row, dict)
    ]
    prep["all_chat_history"] = _format_all_chat_history_for_ai(
        current_history=current_chat_messages,
        cross_tail=_load_cross_tail_for_ai(
            user_id=user_id,
            soul_id=soul_id,
            conversation_id=conversation_id,
        ),
        conversation_id=conversation_id,
        soul_id=soul_id,
        mark_current_chat=False,
    )

    consolidation_llm = await _run_consolidation_llm(
        svc,
        inputs=prep,
        soul_id=soul_id,
        llm_profile=_resolve_profile_if_configured(svc, "consolidation"),
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
        completed = 0
        while True:
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
                if completed:
                    break
                if progress_key and memorize_progress is not None:
                    _memorize_endpoint._set_memorize_progress(
                        memorize_progress,
                        progress_key,
                        active=False,
                        last_result="success",
                    )
                return {"ok": True, "status": "skipped"}
            completed += 1
            result = out.get("result") or {}
            if not result.get("remaining_segment_ids"):
                break
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


def _resolve_web_source_checkpoint(
    conversation_id: str,
    source_message_id: str,
) -> int | None:
    source, db_path, _ = _resolve_whatsapp_source_config()
    if source != "web_source":
        raise RuntimeError("WhatsApp history source is no longer web_source")
    _, hermes_home_path, _, _ = _resolve_cross_source_paths()
    return _conversation_sources.whatsapp_web_source_message_rowid(
        conversation_id,
        source_message_id,
        hermes_home=hermes_home_path,
        web_source_db_path=db_path,
    )


def _make_memorize_run_context() -> _memorize_endpoint.MemorizeRunContext:
    return _memorize_endpoint.MemorizeRunContext(
        base=_make_memorize_context(),
        load_turn_state_and_soul_card=_load_turn_state_and_soul_card,
        normalize_text_list=_normalize_text_list,
        compute_holistic_categories_summary=_compute_holistic_categories_summary,
        run_consolidation_task=_run_consolidation_task,
        clear_last_display_segments_for_nonparticipants=_clear_last_display_segments_for_nonparticipants,
        resolve_web_source_checkpoint=_resolve_web_source_checkpoint,
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
        run_memorize_segments=_run_memorize_segments,
        run_consolidation_task=_run_consolidation_task,
        get_config=lambda: _CONFIG,
        sanitize_db_filename=_sanitize_db_filename,
    )


async def _run_memorize_segments(
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
    merged_len: int,
    force: bool,
    sleep_stats: Any,
    segments_dir: Path,
    zi: Any = None,
    cross_memorize: bool = False,
    final_cursors: dict[str, dict[str, Any]] | None = None,
) -> None:
    await _memorize_endpoint.run_memorize_segments(
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
async def memorize(payload: dict[str, Any], background_tasks: BackgroundTasks, force: bool = False, tail: bool = False, rebuild: bool = False):
    return await _memorize_endpoint.memorize_endpoint(
        payload,
        background_tasks,
        force,
        tail=tail,
        rebuild=rebuild,
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


@app.get("/graph", operation_id="memory_graph")
async def memory_graph(
    user_id: str,
    soul_id: str,
    limit: int = 200,
):
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    if not uid or not sid:
        raise HTTPException(status_code=400, detail="user_id and soul_id are required")
    scope = {"user_id": uid, "soul_id": sid}
    svc = _get_service_from_payload({"user": scope})
    return svc.graph_recent(where=scope, limit=limit)


@app.get("/integration/atomic/atoms", operation_id="atomic_memory_atoms", tags=["integration"])
async def atomic_memory_atoms(
    user_id: str,
    soul_id: str,
    limit: int = 50,
    offset: int = 0,
    category_id: str | None = None,
    tag_id: str | None = None,
    cursor: str | None = None,
    cursor_id: str | None = None,
):
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    if not uid or not sid:
        raise HTTPException(status_code=400, detail="user_id and soul_id are required")
    scope = {"user_id": uid, "soul_id": sid}
    svc = _get_service_from_payload({"user": scope})
    return svc.graph_atomic_atoms(
        where=scope,
        limit=limit,
        offset=offset,
        category_id=category_id or tag_id,
        cursor=cursor,
        cursor_id=cursor_id,
    )


@app.get("/integration/atomic/tags", operation_id="atomic_memory_tags", tags=["integration"])
async def atomic_memory_tags(
    user_id: str,
    soul_id: str,
    min_count: int = 0,
):
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    if not uid or not sid:
        raise HTTPException(status_code=400, detail="user_id and soul_id are required")
    scope = {"user_id": uid, "soul_id": sid}
    svc = _get_service_from_payload({"user": scope})
    return svc.graph_atomic_tags(where=scope, min_count=min_count)


@app.get("/integration/atomic/canvas-source", operation_id="atomic_memory_canvas_source", tags=["integration"])
async def atomic_memory_canvas_source(
    user_id: str,
    soul_id: str,
    limit: int = 500,
    atom_ids: str | None = None,
):
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    if not uid or not sid:
        raise HTTPException(status_code=400, detail="user_id and soul_id are required")
    scope = {"user_id": uid, "soul_id": sid}
    svc = _get_service_from_payload({"user": scope})
    requested = {part.strip() for part in (atom_ids or "").split(",") if part.strip()} or None
    if requested is not None:
        return svc.graph_atomic_canvas_source(where=scope, limit=limit, atom_ids=requested)
    return svc.graph_atomic_canvas_source(where=scope, limit=limit)


@app.post("/integration/atomic/canvas-source", operation_id="atomic_memory_canvas_source_post", tags=["integration"])
async def atomic_memory_canvas_source_post(
    payload: dict[str, Any],
):
    uid = str(payload.get("user_id") or "").strip()
    sid = str(payload.get("soul_id") or "").strip()
    if not uid or not sid:
        raise HTTPException(status_code=400, detail="user_id and soul_id are required")
    scope = {"user_id": uid, "soul_id": sid}
    limit = int(payload.get("limit") or 500)
    atom_ids = {str(atom_id).strip() for atom_id in payload.get("atom_ids") or [] if str(atom_id).strip()}
    svc = _get_service_from_payload({"user": scope})
    return svc.graph_atomic_canvas_source(where=scope, limit=limit, atom_ids=atom_ids or None)


@app.get("/integration/atomic/neighborhood/{item_id}", operation_id="atomic_memory_neighborhood", tags=["integration"])
async def atomic_memory_neighborhood(
    item_id: str,
    user_id: str,
    soul_id: str,
    depth: int = 1,
    min_similarity: float = 0.5,
    limit: int = 5,
):
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    if not uid or not sid:
        raise HTTPException(status_code=400, detail="user_id and soul_id are required")
    scope = {"user_id": uid, "soul_id": sid}
    svc = _get_service_from_payload({"user": scope})
    graph = svc.graph_atomic_neighborhood(
        item_id,
        where=scope,
        depth=depth,
        min_similarity=min_similarity,
        similarity_limit=limit,
    )
    if graph is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return graph


@app.get("/integration/atomic/similar/{item_id}", operation_id="atomic_memory_similar", tags=["integration"])
async def atomic_memory_similar(
    item_id: str,
    user_id: str,
    soul_id: str,
    limit: int = 5,
    min_similarity: float = 0.7,
):
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    if not uid or not sid:
        raise HTTPException(status_code=400, detail="user_id and soul_id are required")
    scope = {"user_id": uid, "soul_id": sid}
    svc = _get_service_from_payload({"user": scope})
    similar = svc.graph_atomic_similar(
        item_id,
        where=scope,
        limit=limit,
        min_similarity=min_similarity,
    )
    if similar is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return similar


@app.get("/integration/atomic/search", operation_id="atomic_memory_search", tags=["integration"])
async def atomic_memory_search(
    q: str,
    user_id: str,
    soul_id: str,
    limit: int = 5,
    mode: str = "hybrid",
    since_days: int | None = None,
):
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    query = str(q or "").strip()
    if not uid or not sid:
        raise HTTPException(status_code=400, detail="user_id and soul_id are required")
    if not query:
        raise HTTPException(status_code=400, detail="q is required")
    scope = {"user_id": uid, "soul_id": sid}
    svc = _get_service_from_payload({"user": scope})
    try:
        return await svc.graph_search(query, where=scope, limit=limit, mode=mode, since_days=since_days)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/memory/{item_id}", operation_id="memory_graph_item")
async def memory_graph_item(
    item_id: str,
    user_id: str,
    soul_id: str,
):
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    if not uid or not sid:
        raise HTTPException(status_code=400, detail="user_id and soul_id are required")
    scope = {"user_id": uid, "soul_id": sid}
    svc = _get_service_from_payload({"user": scope})
    item = svc.graph_memory(item_id, where=scope)
    if item is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return item


@app.get("/pending", operation_id="memory_graph_pending")
async def memory_graph_pending(
    user_id: str,
    soul_id: str,
):
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    if not uid or not sid:
        raise HTTPException(status_code=400, detail="user_id and soul_id are required")
    scope = {"user_id": uid, "soul_id": sid}
    svc = _get_service_from_payload({"user": scope})
    return svc.graph_list_pending(where=scope)


@app.patch("/memory/{item_id}", operation_id="memory_graph_item_update")
async def memory_graph_item_update(
    item_id: str,
    user_id: str,
    soul_id: str,
    payload: dict[str, Any] = Body(...),
):
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    summary = payload.get("summary")
    if not uid or not sid:
        raise HTTPException(status_code=400, detail="user_id and soul_id are required")
    if not isinstance(summary, str) or not summary.strip():
        raise HTTPException(status_code=400, detail="summary is required")
    scope = {"user_id": uid, "soul_id": sid}
    svc = _get_service_from_payload({"user": scope})
    try:
        item = await svc.graph_update_memory_summary(
            item_id,
            summary=summary,
            where=scope,
            edited_by=payload.get("edited_by") if isinstance(payload.get("edited_by"), str) else None,
            approved=payload.get("approved") is True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError:
        item = None
    if item is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return item


@app.post("/memory/{item_id}/approve", operation_id="memory_graph_item_approve")
async def memory_graph_item_approve(
    item_id: str,
    user_id: str,
    soul_id: str,
):
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    if not uid or not sid:
        raise HTTPException(status_code=400, detail="user_id and soul_id are required")
    scope = {"user_id": uid, "soul_id": sid}
    svc = _get_service_from_payload({"user": scope})
    try:
        item = svc.graph_approve_memory(item_id, where=scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if item is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return item


@app.delete("/memory/{item_id}", operation_id="memory_graph_item_delete")
async def memory_graph_item_delete(
    item_id: str,
    user_id: str,
    soul_id: str,
):
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    if not uid or not sid:
        raise HTTPException(status_code=400, detail="user_id and soul_id are required")
    scope = {"user_id": uid, "soul_id": sid}
    svc = _get_service_from_payload({"user": scope})
    try:
        item = svc.graph_delete_memory(item_id, where=scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if item is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return item


@app.patch("/category/{category_id}", operation_id="memory_graph_category_update")
async def memory_graph_category_update(
    category_id: str,
    user_id: str,
    soul_id: str,
    payload: dict[str, Any] = Body(...),
):
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    summary = payload.get("summary")
    if not uid or not sid:
        raise HTTPException(status_code=400, detail="user_id and soul_id are required")
    if not isinstance(summary, str) or not summary.strip():
        raise HTTPException(status_code=400, detail="summary is required")
    scope = {"user_id": uid, "soul_id": sid}
    svc = _get_service_from_payload({"user": scope})
    try:
        item = await svc.graph_update_category_summary(
            category_id,
            summary=summary,
            where=scope,
            edited_by=payload.get("edited_by") if isinstance(payload.get("edited_by"), str) else None,
            approved=payload.get("approved") is True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError:
        item = None
    if item is None:
        raise HTTPException(status_code=404, detail="category not found")
    return item


@app.post("/category/{category_id}/approve", operation_id="memory_graph_category_approve")
async def memory_graph_category_approve(
    category_id: str,
    user_id: str,
    soul_id: str,
):
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    if not uid or not sid:
        raise HTTPException(status_code=400, detail="user_id and soul_id are required")
    scope = {"user_id": uid, "soul_id": sid}
    svc = _get_service_from_payload({"user": scope})
    try:
        item = svc.graph_approve_category(category_id, where=scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if item is None:
        raise HTTPException(status_code=404, detail="category not found")
    return item


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
        build_atomic_snapshot = bool(safe.get("build_atomic_snapshot", False))
        if uid and soul_id:
            safe["user"] = {"user_id": uid, "soul_id": soul_id, "conversation_id": cid}
            safe["conversation_id"] = cid
        message = _pick_str(safe, "message", "query") or ""
        self_turn_directive = _pick_str(safe, "self_turn_directive") or ""
        self_turn_label = _pick_str(safe, "self_turn_label") or ""
        retrieve_focus = self_turn_directive or message
        if build_atomic_snapshot and not retrieve_focus:
            retrieve_focus = "Atomic memory workspace"
            if safe.get("queries") is None:
                safe["query"] = retrieve_focus
        display_current_user_text = message if build_atomic_snapshot else retrieve_focus
        current_whatsapp_active_since = _current_whatsapp_active_since_for_soul(cid, soul_id)
        is_live_turn = bool(safe.get("is_live_turn")) and _message_log.derive_source_label(cid).startswith("whatsapp:")
        history = _prepare_current_whatsapp_history(
            conversation_id=cid,
            soul_id=soul_id,
            raw_history=safe.get("history"),
            active_since=current_whatsapp_active_since,
            load_source_history=bool(safe.get("load_source_history")),
            external_message_id=safe.get("external_message_id"),
            is_live_turn=is_live_turn,
            op="conversation.retrieve",
            exclude_external_message=False,
        )
        history_without_external = _filter_external_message_from_history(
            history,
            safe.get("external_message_id"),
        )
        safe["history"] = history_without_external
        _stamp_assistant_display_name(history, soul_id)
        state_row: dict[str, Any] | None = None
        cross_tail: list[dict[str, Any]] = []
        if uid and soul_id and retrieve_focus.strip():
            if (
                _message_log.derive_source_label(cid) == "sillytavern"
                and history
                and message.strip()
                and not bool(safe.get("_read_only_retrieve", False))
            ):
                _conversation_sources.persist_sillytavern_history_snapshot(
                    storage_dir=_get_storage_dir(_CONFIG),
                    user_id=uid,
                    soul_id=soul_id,
                    conversation_id=cid,
                    history=history,
                    chat_name=_pick_str(safe, "chat_name") or None,
                )
            state_row, _, _db_path = _load_turn_state_and_soul_card(
                cid,
                user_id=uid,
                soul_id=soul_id,
            )
            if _db_path is not None and _db_path.exists():
                _con = _sqlite_connect(_db_path)
                try:
                    _con.row_factory = sqlite3.Row
                    _sqlite_ensure_conversation_state_schema(_con)
                    cross_tail = _load_cross_tail_with_activities_from_sources(
                        _con,
                        user_id=uid,
                        soul_id=soul_id,
                        exclude_conversation_id=cid,
                    )
                finally:
                    _con.close()

        chat_label_for_prompt = _chat_label_for_prompt(safe)
        digest_cursor, min_timestamp = -1, None
        if history:
            _, hermes_home_path, _, _ = _resolve_cross_source_paths()
            digest_cursor, min_timestamp, _ = _resolve_source_cursor(
                cid,
                _effective_digest_cursor_from_row(state_row),
                (state_row or {}).get("digest_cursor_source_message_id"),
                (state_row or {}).get("digest_cursor_ts"),
                rolling=False,
                hermes_home_path=hermes_home_path,
            )
        history_for_ai = _filter_external_message_from_history(
            _turn_history_with_floor(history, digest_cursor, min_timestamp),
            safe.get("external_message_id"),
        )
        cross_text = _format_cross_tail_for_ai(cross_tail, soul_id=soul_id) if cross_tail else ""
        if cross_text:
            safe["_cross_conversation_history"] = cross_text
        all_chat_history_for_ai = _format_all_chat_history_for_ai(
            current_history=history_for_ai,
            cross_tail=cross_tail,
            conversation_id=cid,
            soul_id=soul_id,
            chat_label=chat_label_for_prompt,
            current_user_text=display_current_user_text,
            current_user_name=_pick_str(safe, "user_name") or None,
            self_turn_directive=self_turn_directive or None,
        )

        should_build_default_queries = (
            uid
            and soul_id
            and retrieve_focus.strip()
            and state_row is not None
            and safe.get("queries") is None
        )
        should_rebuild_queries_for_cutoff = (
            current_whatsapp_active_since is not None
            and soul_id
            and retrieve_focus.strip()
        )
        if should_build_default_queries or should_rebuild_queries_for_cutoff:
            safe["queries"] = _build_retrieve_soul_context_queries(
                soul_id=soul_id,
                message=retrieve_focus,
                history=history_for_ai,
                state_row=state_row or {},
                conversation_id=cid,
                chat_label=chat_label_for_prompt,
                conversations_block=all_chat_history_for_ai or None,
                self_turn_directive=self_turn_directive or None,
                self_turn_label=self_turn_label or None,
            )

        out = await _run_retrieve(safe, conversation_id=cid)

        if build_atomic_snapshot:
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
                response_chat_history_for_ai = _format_all_chat_history_for_ai(
                    current_history=history_for_ai,
                    cross_tail=cross_tail,
                    conversation_id=cid,
                    soul_id=soul_id,
                    chat_label=chat_label_for_prompt,
                    current_user_text=message,
                    current_user_name=_pick_str(safe, "user_name") or None,
                    self_turn_directive=self_turn_directive,
                )
                if not response_chat_history_for_ai.strip() and cid.startswith("chat:atomic-"):
                    response_chat_history_for_ai = "My Atomic Conversations:"
                memory_cache = _normalize_memory_cache_impl(out.get("memory_cache"))
                intentions_active = _normalize_intentions_stack_impl(out.get("intentions_active"))
                atomic_retrieve_rag = out.get("result")
                if isinstance(atomic_retrieve_rag, dict):
                    atomic_retrieve_rag = {**atomic_retrieve_rag, "items": []}
                system_base = _make_turn_identity_prompt(
                    soul_id,
                    soul_card=soul_card,
                )
                context_block = _build_turn_context_block(
                    history=history_for_ai,
                    prior_context=out.get("prior_context"),
                    retrieve_rag=atomic_retrieve_rag,
                    all_categories_summary=_state_row.get("all_categories_summary"),
                    memory_cache=memory_cache,
                    intentions_active=intentions_active,
                    apimw_message_to_self=_state_row.get("apimw_message_to_self"),
                    conversations_block=response_chat_history_for_ai or None,
                    chat_label=chat_label_for_prompt,
                    conversation_id=cid,
                    soul_name=soul_id,
                    current_user_text=message,
                    self_turn_directive=self_turn_directive or None,
                    include_working_state=False,
                )
                out["atomic_snapshot_text"] = f"{system_base}\n\n{context_block}".strip()
                out["turn_prompt_source"] = "conversation_retrieve"

        want_turn_prompt = bool(safe.get("build_turn_prompt", False))

        if want_turn_prompt:
            scope = _extract_scope(safe)
            uid = str(scope.get("user_id") or "").strip()
            soul_id = str(scope.get("soul_id") or "").strip()
            turn_history = history_for_ai

            if uid and soul_id:
                _state_row, soul_card, _db_path = _load_turn_state_and_soul_card(
                    cid,
                    user_id=uid,
                    soul_id=soul_id,
                )
                payload_soul_card = str(safe.get("soul_card") or "").strip() or None
                soul_card = payload_soul_card or soul_card

                message = _pick_str(safe, "message", "query") or ""
                self_turn_directive = _pick_str(safe, "self_turn_directive") or ""
                self_turn_label = _pick_str(safe, "self_turn_label") or ""
                response_chat_history_for_ai = _format_all_chat_history_for_ai(
                    current_history=turn_history,
                    cross_tail=cross_tail,
                    conversation_id=cid,
                    soul_id=soul_id,
                    chat_label=chat_label_for_prompt,
                    current_user_text=message,
                    current_user_name=_pick_str(safe, "user_name") or None,
                    self_turn_directive=self_turn_directive,
                )
                memory_cache = _normalize_memory_cache_impl(out.get("memory_cache"))
                intentions_active = _normalize_intentions_stack_impl(out.get("intentions_active"))

                out["turn_system_prompt"] = _make_turn_system_prompt(
                    soul_id,
                    soul_card=soul_card,
                    response_sentences=int(_CONFIG.get("turn_response_sentences", 3)),
                    allow_public_response=bool(safe.get("allow_public_response", True)),
                    include_activity_recap=bool(self_turn_directive),
                )
                out["turn_user_prompt"] = _build_turn_prompt(
                    user_message=message,
                    history=turn_history,
                    prior_context=out.get("prior_context"),
                    retrieve_rag=out.get("result"),
                    all_categories_summary=_state_row.get("all_categories_summary"),
                    memory_cache=memory_cache,
                    intentions_active=intentions_active,
                    apimw_message_to_self=_state_row.get("apimw_message_to_self"),
                    conversations_block=response_chat_history_for_ai or None,
                    chat_label=chat_label_for_prompt,
                    conversation_id=cid,
                    self_turn_directive=self_turn_directive or None,
                    self_turn_label=self_turn_label or None,
                    response_sentences=int(_CONFIG.get("turn_response_sentences", 3)),
                    allow_public_response=bool(safe.get("allow_public_response", True)),
                    include_activity_recap=bool(self_turn_directive),
                )
                out["turn_prompt_source"] = "conversation_retrieve"
            if current_whatsapp_active_since is not None:
                out["turn_prompt_active_since"] = current_whatsapp_active_since
            if safe.get("_cross_conversation_history"):
                out["cross_conversation_history"] = safe.get("_cross_conversation_history")
            if is_live_turn:
                out["turn_history"] = turn_history

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
    trigger_history: list[dict[str, Any]],
    digest_cursor: int,
    trigger_memorize_default: bool = True,
    trigger_web_source: bool = False,
) -> dict[str, Any] | None:
    """Merge unmemorized tails from all conversations into one memorize payload."""
    db_path = _sqlite_current_path(uid, soul_id)
    if db_path is None or not db_path.exists():
        return None

    trigger_label = _message_log.derive_source_label(cid)
    trigger_memorize_raw = safe.get("memorize_chat")
    trigger_memorize = trigger_memorize_raw if isinstance(trigger_memorize_raw, bool) else trigger_memorize_default
    trigger_chat_name = str(safe.get("chat_name") or "").strip()
    trigger_checkpoint = (
        _source_cursor_checkpoint(trigger_history, web_source=True)
        if trigger_web_source
        else None
    )
    trigger_tail = _normalize_conversation(trigger_history)
    if not trigger_tail:
        return None

    for i, msg in enumerate(trigger_tail):
        msg["source_label"] = trigger_label
        msg["source_conversation_id"] = cid
        if trigger_web_source:
            raw = trigger_history[i]
            msg["source_conversation_index"] = raw.get("source_conversation_index")
            msg["source_message_id"] = raw.get("source_message_id")
        else:
            msg["source_conversation_index"] = digest_cursor + 1 + i
        msg["memorize_chat"] = trigger_memorize
        if trigger_chat_name and not str(msg.get("chat_name") or "").strip():
            msg["chat_name"] = trigger_chat_name
        ts = msg.get("ts_ms")
        if isinstance(ts, (int, float)) and "received_at" not in msg:
            msg["received_at"] = datetime.fromtimestamp(ts / 1000.0, tz=UTC).isoformat()

    if not trigger_web_source:
        trigger_checkpoint = {"cursor": digest_cursor + len(trigger_tail)}
    if trigger_checkpoint is None:
        raise RuntimeError(f"memorize tail has no checkpoint for {cid}")
    final_cursors: dict[str, dict[str, Any]] = {cid: trigger_checkpoint}
    all_messages = list(trigger_tail)

    con = _sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        other_tails = _load_cross_memorize_tails_from_sources(
            con,
            user_id=uid,
            soul_id=soul_id,
            exclude_conversation_id=cid,
        )
        rolling_summaries = _read_background_rolling_summaries_from_conversations(
            con,
            exclude_conversation_id=cid,
        )
    finally:
        con.close()

    whatsapp_source: str | None = None
    for other_cid, tail_msgs in other_tails.items():
        if not tail_msgs:
            continue
        source_label = _message_log.derive_source_label(other_cid)
        if source_label.startswith("whatsapp:") and whatsapp_source is None:
            whatsapp_source = _resolve_whatsapp_source_config()[0]
        web_source = source_label.startswith("whatsapp:") and whatsapp_source == "web_source"
        final_cursor = _source_cursor_checkpoint(tail_msgs, web_source=web_source)
        if final_cursor is not None:
            final_cursors[other_cid] = final_cursor
        all_messages.extend(tail_msgs)

    all_messages.sort(
        key=lambda m: (
            str(m.get("received_at") or ""),
            str(m.get("source_conversation_id") or m.get("conversation_id") or ""),
            int(m.get("source_conversation_index") or 0),
        )
    )

    return {
        **safe,
        "conversation_id": cid,
        "conversation": all_messages,
        "user": {"user_id": uid, "soul_id": soul_id, "conversation_id": cid},
        "_cross_memorize": True,
        "_final_cursors": final_cursors,
        "_background_rolling_summaries": rolling_summaries,
    }


def _estimate_primary_memorize_tokens(messages: list[dict[str, Any]]) -> int:
    primary = [
        {"content": str(msg.get("content") or "")}
        for msg in messages
        if bool(msg.get("memorize_chat", True))
    ]
    return _estimate_tokens(primary)


@app.get(f"{_DIAG_PREFIX}/diag/memorize/pending")
@app.get("/diag/memorize/pending", operation_id="diag_memorize_pending")
async def diag_memorize_pending(user_id: str = "", soul_id: str = ""):
    """Read-only snapshot of memorize pressure: summed unmemorized primary tokens vs threshold."""
    uid = user_id.strip()
    sid = soul_id.strip()
    db_path = _sqlite_current_path(uid or None, sid or None)
    if db_path is None:
        return {"ok": False, "reason": "soul_id_required"}
    if not db_path.exists():
        return {"ok": False, "reason": "sqlite_file_missing", "path": str(db_path)}
    con = _sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        if not uid:
            _sqlite_ensure_conversation_state_schema(con)
            rows = con.execute(
                "SELECT DISTINCT user_id FROM conversations WHERE user_id IS NOT NULL AND user_id != ''"
            ).fetchall()
            if len(rows) > 1:
                return {"ok": False, "reason": "user_id_ambiguous"}
            if rows:
                uid = str(rows[0]["user_id"])
        tails = _load_cross_memorize_tails_from_sources(con, user_id=uid, soul_id=sid)
    finally:
        con.close()
    merged = [msg for tail in tails.values() for msg in tail]
    merged.sort(
        key=lambda m: (
            str(m.get("received_at") or ""),
            str(m.get("source_conversation_id") or m.get("conversation_id") or ""),
            int(m.get("source_conversation_index") or 0),
        )
    )
    for msg in merged:
        if msg.get("ts_ms") is None:
            ts_ms = _parse_turn_ts_ms(msg.get("received_at"))
            if ts_ms is not None:
                msg["ts_ms"] = ts_ms
    summed = _estimate_primary_memorize_tokens(merged)
    threshold = _MIN_CHUNK_TOKENS
    return {
        "summed_unmemorized_tokens": summed,
        "threshold": threshold,
        "pct": round(summed * 100 / threshold) if threshold else 0,
        "sleep_gap_ready": _unmemorized_sleep_gap_detected(merged, -1, {}, min_chunk_tokens=0),
        "computed_at": datetime.now(UTC).isoformat(),
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
    unmemorized_digest_cursor = _effective_digest_cursor_from_row(conversation_state)
    _, hermes_home_path, _, _ = _resolve_cross_source_paths()
    resolved_cursor, min_timestamp, trigger_web_source = _resolve_source_cursor(
        cid,
        unmemorized_digest_cursor,
        conversation_state.get("digest_cursor_source_message_id"),
        conversation_state.get("digest_cursor_ts"),
        rolling=False,
        hermes_home_path=hermes_home_path,
    )
    chat_is_primary = bool(conversation_state.get("memorize_chat", True))
    primary_history = history_full if chat_is_primary else []
    if min_timestamp is not None:
        primary_history = [
            row for row in primary_history
            if int(row.get("ts_ms") or 0) >= min_timestamp * 1000
        ]
    unmemorized_history = _conversation_sources.slice_tail_with_floor(
        primary_history,
        since_cursor=resolved_cursor,
        recent_fallback_messages=0,
    )
    unmemorized_tokens = _estimate_unmemorized_tokens(unmemorized_history, -1)
    queued_memorize_payload: dict[str, Any] | None = None
    if (not dry_run) and unmemorized_history:
        has_sleep_gap = _unmemorized_sleep_gap_detected(
            unmemorized_history,
            -1,
            safe,
            min_chunk_tokens=0,
        )
        if has_sleep_gap:
            try:
                candidate_payload = _build_cross_conversation_payload(
                    cid,
                    uid,
                    soul_id,
                    safe,
                    unmemorized_history,
                    unmemorized_digest_cursor,
                    True,
                    trigger_web_source=trigger_web_source,
                )
                if candidate_payload is not None:
                    unmemorized_tokens = _estimate_primary_memorize_tokens(
                        list(candidate_payload.get("conversation") or [])
                    )
                    if unmemorized_tokens >= _MIN_CHUNK_TOKENS:
                        queued_memorize_payload = candidate_payload
            except Exception as exc:
                logger.error("forced memorize source assembly failed for conversation_id=%s: %s", cid, exc)
                _set_background_error(
                    cid,
                    soul_id=soul_id,
                    user_id=uid,
                    code="forced_memorize_source_failed",
                    detail=f"{type(exc).__name__}: {exc}",
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
    *,
    annulment_memory_ids: list[str] | None = None,
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
            "annulment_memory_ids": list(annulment_memory_ids or []),
        },
    }
    if retrieval_ids_since_consolidation:
        updates["append_retrieval_ids_since_consolidation"] = retrieval_ids_since_consolidation
    if isinstance(memorize_chat, bool):
        updates["memorize_chat"] = memorize_chat
    updates["prior_context"] = None
    updates["apimw_message_to_self"] = None
    state_out, state_path = _write_conversation_state(
        cid,
        soul_id=soul_id,
        user_id=uid,
        updates=updates,
    )
    return state_out, state_path




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
        self_turn_directive = str(safe.get("self_turn_directive") or "").strip()
        if not message and not self_turn_directive:
            raise HTTPException(status_code=400, detail="message or self_turn_directive is required")

        current_whatsapp_active_since = _current_whatsapp_active_since_for_soul(cid, soul_id)
        is_live_turn = bool(safe.get("is_live_turn")) and _message_log.derive_source_label(cid).startswith("whatsapp:")
        history_full = _prepare_current_whatsapp_history(
            conversation_id=cid,
            soul_id=soul_id,
            raw_history=safe.get("history"),
            active_since=current_whatsapp_active_since,
            load_source_history=bool(safe.get("load_source_history")),
            external_message_id=safe.get("external_message_id"),
            is_live_turn=is_live_turn,
            op="conversation.turn",
        )
        safe["history"] = history_full
        _stamp_assistant_display_name(history_full, soul_id)
        prompt_override_payload_raw = safe.get("prompt_override_payload")
        if not isinstance(prompt_override_payload_raw, dict):
            raise HTTPException(status_code=400, detail="prompt_override_payload is required")
        prompt_override_payload = dict(prompt_override_payload_raw)
        if (
            current_whatsapp_active_since is not None
            and str(prompt_override_payload.get("generated_by") or "").strip()
            != "conversation_retrieve"
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "prompt_override_payload must be generated by conversation_retrieve "
                    "when WhatsApp active_since cutoff is active"
                ),
            )
        if current_whatsapp_active_since is not None:
            try:
                prompt_active_since = float(prompt_override_payload.get("active_since"))
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "prompt_override_payload.active_since must match current "
                        "WhatsApp active_since cutoff"
                    ),
                ) from exc
            if prompt_active_since != current_whatsapp_active_since:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "prompt_override_payload.active_since must match current "
                        "WhatsApp active_since cutoff"
                    ),
                )
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
        allow_public_response = bool(safe.get("allow_public_response", True))
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
            allow_public_response=allow_public_response,
            include_activity_recap=bool(self_turn_directive),
        )
        turn_user_prompt = override_user_prompt

        memory_service = _get_service_from_payload(safe)
        generation_metadata = _turn_generation_metadata(safe)
        turn_started_at = time.monotonic()
        turn_contract: dict[str, Any] | None = None
        use_claude_session = bool(_CONFIG.get("claude_code", False))
        turn_session_id: str | None = None
        for attempt in (1, 2):
            attempt_session_id = str(uuid.uuid4()) if use_claude_session else None
            if attempt_session_id:
                logger.info("conversation_turn: claude session_id=%s conversation_id=%s", attempt_session_id, cid)
            chat_kwargs = {"session_id": attempt_session_id} if attempt_session_id else {}
            turn_response_raw = await memory_service.chat(
                turn_user_prompt,
                system_prompt=turn_system_prompt,
                temperature=turn_temperature,
                response_format=turn_response_format,
                op="turn",
                step="respond" if attempt == 1 else "respond_retry",
                trace_id=trace_id,
                **chat_kwargs,
            )
            try:
                turn_contract = _parse_turn_contract(
                    turn_response_raw,
                    allow_public_response=allow_public_response,
                    attachment_workspace=_attachment_workspace(),
                )
                turn_session_id = attempt_session_id
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

        if self_turn_directive and not dry_run:
            _record_activity_message(
                user_id=uid,
                soul_id=soul_id,
                recap=_activity_recap_from_contract(turn_contract),
            )

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

        if not dry_run:
            retrieved_item_ids = _extract_result_item_ids(override_retrieve_rag)
            annulment_memory_ids = await _persist_annulment_memories(
                svc=memory_service,
                scope={"user_id": uid, "soul_id": soul_id},
                conversation_id=cid,
                intentions_before=intentions_before,
                annulments=normalized_annulments,
            )
            async with state_lock:
                conversation_state_after, conversation_state_path = _turn_state_write(
                    cid, uid, soul_id,
                    turn_cache_entry, turn_annulment_ids,
                    retrieved_item_ids,
                    memorize_chat=memorize_chat,
                    annulment_memory_ids=annulment_memory_ids,
                )
            if not bool(conversation_state_after.get("memorize_chat", True)):
                _queue_background_rollup_task(
                    conversation_id=cid,
                    user_id=uid,
                    soul_id=soul_id,
                    safe_payload=safe,
                    service=memory_service,
                )

        apimw_status = "skipped_dry_run" if dry_run else "not_started"
        if (not dry_run) and run_apimw:
            apimw_status = _turn_launch_apimw(
                cid, uid, soul_id, safe, history_full,
            )

        response_target = str(turn_contract.get("response_target") or "").strip().lower()
        if response_target not in {"respond", "listen", "observe", "private"}:
            raise HTTPException(status_code=502, detail="turn contract missing or invalid response_target")
        response_text = str(turn_contract.get("response") or "").strip()
        attachment = turn_contract.get("attachment") or None
        continuation_reason = str(turn_contract.get("continue_reason") or "").strip().lower()
        continuation_queued = False
        if not dry_run and continuation_reason:
            if continuation_reason in {"task", "research", "diary"}:
                if turn_session_id:
                    continuation_queued = _queue_free_turn_chain(
                        service=memory_service,
                        user_id=uid,
                        soul_id=soul_id,
                        conversation_id=cid,
                        session_id=turn_session_id,
                        initial_reason=continuation_reason,
                        initial_contract=turn_contract,
                        system_prompt=turn_system_prompt,
                        allow_public_response=allow_public_response,
                        safe_payload=safe,
                        soul_card=soul_card,
                        system_prompt_has_activity_recap=bool(self_turn_directive),
                    )
                else:
                    logger.warning("free_turn: continuation requested but claude_code is disabled")
            elif continuation_reason == "follow_up":
                continuation_queued = bool(
                    _schedule_free_turn_follow_up(
                        user_id=uid,
                        soul_id=soul_id,
                        conversation_id=cid,
                        follow_up_at=str(turn_contract.get("follow_up_at") or ""),
                        follow_up_reason=str(turn_contract.get("follow_up_reason") or ""),
                        safe_payload=safe,
                    )
                )

        # Enforce response_target contract:
        # - listen/observe: nothing is sent.
        # - respond: if chat_name is missing, proceed but log loudly.
        # - private: passes through; routing to the human's private chat is
        #   hermes-side (see HANDOFF for the wiring task).
        if response_target in {"listen", "observe"}:
            response_text = ""
        elif response_target == "respond":
            chat_name = str(safe.get("chat_name") or "").strip()
            if not chat_name:
                logger.warning(
                    "conversation_turn: missing chat_name for respond; continuing without chat label"
                )

        if not dry_run:
            _persist_completed_sillytavern_turn_snapshot(
                safe=safe,
                history=history_full,
                conversation_id=cid,
                user_id=uid,
                soul_id=soul_id,
                message=message,
                response_text=response_text,
                response_target=response_target,
            )

        if attachment and not dry_run:
            if response_target in {"respond", "private"} and cid.startswith("whatsapp:"):
                out_id = _insert_whatsapp_outbound(
                    user_id=uid,
                    soul_id=soul_id,
                    origin_conversation_id=cid,
                    target=response_target,
                    response_text=response_text,
                    media_path=attachment,
                    metadata={"source": "turn_attachment"},
                )
                response_text = ""
                logger.info(
                    "conversation_turn: queued attachment outbound %s target=%s",
                    out_id,
                    response_target,
                )
            elif not cid.startswith("whatsapp:"):
                logger.error(
                    "conversation_turn: attachment dropped — not a WhatsApp conversation (cid=%s)", cid
                )
            else:
                logger.error(
                    "conversation_turn: attachment dropped — response_target=%s cannot carry attachment",
                    response_target,
                )

        forced_memorize_scheduled = False
        if queued_memorize_payload is not None:
            marker = _memorize_lock_key(uid, soul_id)
            if _mark_inflight(_FORCED_MEMORIZE_INFLIGHT, marker):
                try:
                    _t = asyncio.create_task(_run_forced_memorize_from_turn(queued_memorize_payload))
                except Exception:
                    _clear_inflight(_FORCED_MEMORIZE_INFLIGHT, marker)
                    raise
                _BACKGROUND_TASKS.add(_t)
                forced_memorize_scheduled = True

                def _on_forced_memorize_done(done_task: asyncio.Task) -> None:
                    _BACKGROUND_TASKS.discard(done_task)
                    _clear_inflight(_FORCED_MEMORIZE_INFLIGHT, marker)

                _t.add_done_callback(_on_forced_memorize_done)

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
            "continuation_queued": continuation_queued,
        }
        if generation_metadata:
            response_payload["generation_metadata"] = generation_metadata
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
                "queued": forced_memorize_scheduled,
                "unmemorized_tokens": unmemorized_tokens,
                "min_chunk_tokens": _MIN_CHUNK_TOKENS,
            }

        _record_call(
            "conversation.turn",
            safe,
            ok=True,
            info={
                "conversationId": cid,
                "dryRun": dry_run,
                "apimw": apimw_status,
                "forcedMemorizeQueued": forced_memorize_scheduled,
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

        annulment_memory_ids = [
            str(value or "").strip()
            for value in undo_snapshot.get("annulment_memory_ids") or []
            if str(value or "").strip()
        ]
        if annulment_memory_ids:
            svc = _get_service_from_payload({"user": {"user_id": uid, "soul_id": soul_id}})
            for item_id in annulment_memory_ids:
                svc.graph_delete_memory(
                    item_id,
                    where={"user_id": uid, "soul_id": soul_id},
                )
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


@app.post("/integration/atomic/session_start", operation_id="atomic_session_start", tags=["integration"])
async def atomic_session_start(req: AtomicSessionStartRequest):
    uid = str(req.user_id or "").strip()
    soul_id = str(req.soul_id or "").strip()
    if not uid or not soul_id:
        raise HTTPException(status_code=400, detail="user_id and soul_id are required")

    raw_cid = str(req.conversation_id or "").strip()
    conversation_id = raw_cid or "chat:memory-surfer"
    if raw_cid and not raw_cid.startswith("chat:"):
        conversation_id = f"chat:atomic-{raw_cid}"
    if conversation_id.startswith("chat:atomic-"):
        now_iso = datetime.now(UTC).isoformat()
        _write_conversation_state(
            conversation_id,
            soul_id=soul_id,
            user_id=uid,
            updates={
                "memorize_chat": True,
                "atomic_session_started_at": now_iso,
                "atomic_session_ended_at": None,
            },
        )
        _conversation_sources.persist_atomic_history_snapshot(
            storage_dir=_get_storage_dir(_CONFIG),
            user_id=uid,
            soul_id=soul_id,
            conversation_id=conversation_id,
            history=[],
            chat_name="Atomic",
        )

    message = str(req.message or "").strip()
    retrieve_payload: dict[str, Any] = {
        "user": {"user_id": uid, "soul_id": soul_id, "conversation_id": conversation_id},
        "build_atomic_snapshot": True,
        "_read_only_retrieve": True,
        "mental_health_addon": False,
        "debug": bool(req.debug),
        "chat_name": uid,
        "chat_type": "dm",
    }
    if message:
        retrieve_payload["message"] = message
        retrieve_payload["query"] = message
    if req.soul_card:
        retrieve_payload["soul_card"] = str(req.soul_card)

    retrieve_out = await conversation_retrieve(conversation_id, retrieve_payload)
    snapshot_text = str(retrieve_out.get("atomic_snapshot_text") or "").strip()
    if not snapshot_text:
        raise HTTPException(status_code=502, detail="conversation_retrieve returned empty atomic_snapshot_text")
    return {
        "ok": True,
        "conversation_id": conversation_id,
        "snapshot_text": snapshot_text,
        "retrieve_ms": retrieve_out.get("retrieve_ms"),
    }


def _atomic_transcript_rows(transcript: list[dict[str, Any]], *, user_id: str, soul_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in transcript or []:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "").strip().lower()
        if role == "system":
            continue
        if role == "tool":
            continue
        content = str(row.get("content") or row.get("message") or "").strip()
        if not role or not content:
            continue
        out = {
            "role": "assistant" if role == "soul" else role,
            "content": content,
            "name": str(row.get("name") or row.get("speaker") or "").strip(),
            "created_at": str(row.get("created_at") or row.get("received_at") or "").strip(),
        }
        if out["role"] == "assistant" and not out["name"]:
            out["name"] = soul_id
        elif out["role"] == "user" and not out["name"]:
            out["name"] = user_id
        rows.append(out)
    return rows


def _atomic_has_interchange(rows: list[dict[str, Any]]) -> bool:
    roles = {str(row.get("role") or "").strip().lower() for row in rows}
    return "user" in roles and "assistant" in roles


def _atomic_parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _atomic_latest_created_at(rows: list[dict[str, Any]]) -> datetime | None:
    latest: datetime | None = None
    for row in rows:
        created_at = _atomic_parse_dt(row.get("created_at") or row.get("received_at"))
        if created_at and (latest is None or created_at > latest):
            latest = created_at
    return latest


@app.post("/integration/atomic/session_end", operation_id="atomic_session_end", tags=["integration"])
async def atomic_session_end(req: AtomicSessionEndRequest):
    uid = str(req.user_id or "").strip()
    soul_id = str(req.soul_id or "").strip()
    conversation_id = str(req.conversation_id or "").strip()
    if not uid or not soul_id or not conversation_id:
        raise HTTPException(status_code=400, detail="user_id, soul_id, and conversation_id are required")
    if not conversation_id.startswith("chat:atomic-"):
        raise HTTPException(status_code=400, detail="conversation_id must start with chat:atomic-")

    rows = _atomic_transcript_rows(req.transcript, user_id=uid, soul_id=soul_id)
    state_row, _, _ = _load_turn_state_and_soul_card(conversation_id, user_id=uid, soul_id=soul_id)
    ended_at = _atomic_parse_dt(state_row.get("atomic_session_ended_at"))
    latest_created_at = _atomic_latest_created_at(rows)
    if ended_at and (latest_created_at is None or latest_created_at <= ended_at):
        return {"ok": True, "status": "already_ended", "conversation_id": conversation_id}

    recap = str(req.activity_recap or "").strip()
    if _atomic_has_interchange(rows) and not recap:
        raise HTTPException(status_code=400, detail="activity_recap is required when session has user/soul interchange")

    _conversation_sources.persist_atomic_history_snapshot(
        storage_dir=_get_storage_dir(_CONFIG),
        user_id=uid,
        soul_id=soul_id,
        conversation_id=conversation_id,
        history=rows,
        chat_name="Atomic",
    )
    if recap:
        _record_activity_message(user_id=uid, soul_id=soul_id, recap=recap)

    ended_dt = datetime.now(UTC)
    if latest_created_at and latest_created_at > ended_dt:
        ended_dt = latest_created_at
    ended_at = ended_dt.isoformat()
    _write_conversation_state(
        conversation_id,
        soul_id=soul_id,
        user_id=uid,
        updates={
            "memorize_chat": True,
            "atomic_session_ended_at": ended_at,
        },
    )
    return {
        "ok": True,
        "status": "ended",
        "conversation_id": conversation_id,
        "activity_recorded": bool(recap),
        "message_count": len(rows),
    }


def _atomic_chat_settings_from_config(cfg: dict[str, Any]) -> dict[str, str]:
    profile = _default_llm_profiles_from_server_config(cfg).get("default", {})
    provider = str(profile.get("provider") or "").strip().lower()
    base_url = str(profile.get("base_url") or "").strip()
    api_key = str(profile.get("api_key") or "")
    chat_model = str(profile.get("chat_model") or "").strip()
    try:
        response_sentences = max(1, int(cfg.get("turn_response_sentences", 3)))
    except (TypeError, ValueError):
        response_sentences = 3
    debug_cfg = cfg.get("debug") if isinstance(cfg.get("debug"), dict) else {}
    common = {
        "memu_response_sentences": str(response_sentences),
        "memu_log_prompts": "1" if bool(debug_cfg.get("log_prompts", False)) else "0",
    }

    if provider in {"openai", "openai_compat", "nanogpt"}:
        return common | {
            "provider": "openai_compat",
            "openai_compat_base_url": base_url,
            "openai_compat_api_key": api_key,
            "openai_compat_llm_model": chat_model,
        }
    if provider == "ollama":
        return common | {
            "provider": "ollama",
            "ollama_host": base_url,
            "ollama_llm_model": chat_model,
        }
    if provider == "openrouter":
        return common | {
            "provider": "openrouter",
            "openrouter_api_key": api_key,
            "chat_model": chat_model,
        }
    raise HTTPException(status_code=500, detail=f"Unsupported Atomic chat provider: {provider or '<empty>'}")


@app.get("/integration/atomic/chat_profile", operation_id="atomic_chat_profile", tags=["integration"])
async def atomic_chat_profile():
    return {"ok": True, "settings": _atomic_chat_settings_from_config(_CONFIG)}


@app.post("/integration/atomic/prompt_log", operation_id="atomic_prompt_log", tags=["integration"])
async def atomic_prompt_log(req: AtomicPromptLogRequest):
    if not _LOG_PROMPTS:
        return {"ok": True, "logged": False}
    cid = str(req.conversation_id or "").strip() or "-"
    model = str(req.model or "").strip() or "-"
    banner = "===== ATOMIC CHAT · prompt ".ljust(70, "=")
    lines = [
        "",
        "",
        "",
        banner,
        "",
        f"[PROMPT] op=atomic_chat conversation_id={cid} model={model}",
    ]
    for message in req.messages:
        role = str(message.get("role") or "-")
        content = message.get("content")
        lines.extend([
            "",
            f"role: {role}",
        ])
        if message.get("name") is not None:
            lines.append(f"name: {message['name']}")
        lines.extend(["content:", "" if content is None else str(content)])
        if message.get("tool_calls") is not None:
            lines.extend([
                "tool_calls:",
                json.dumps(message["tool_calls"], ensure_ascii=False, indent=2),
            ])
        if message.get("tool_call_id") is not None:
            lines.append(f"tool_call_id: {message['tool_call_id']}")
    _PROMPT_LOGGER.info("\n".join(lines))
    return {"ok": True, "logged": True}


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


@app.post("/integration/whatsapp/outbounds/claim", operation_id="whatsapp_outbounds_claim", tags=["integration"])
async def whatsapp_outbounds_claim(payload: dict[str, Any] = Body(...)):
    return {
        "ok": True,
        "outbounds": _claim_whatsapp_outbounds(
            user_id=str(payload.get("user_id") or ""),
            soul_id=str(payload.get("soul_id") or ""),
            claimed_by=str(payload.get("claimed_by") or payload.get("claimer") or "hermes"),
            limit=int(payload.get("limit", 10)),
            claim_timeout_seconds=int(payload.get("claim_timeout_seconds", 300)),
        ),
    }


@app.post("/integration/whatsapp/outbounds/mark", operation_id="whatsapp_outbounds_mark", tags=["integration"])
async def whatsapp_outbounds_mark(payload: dict[str, Any] = Body(...)):
    row = _mark_whatsapp_outbound(
        user_id=str(payload.get("user_id") or ""),
        soul_id=str(payload.get("soul_id") or ""),
        outbound_id=str(payload.get("outbound_id") or payload.get("id") or ""),
        status=str(payload.get("status") or ""),
        provider_message_id=str(payload.get("provider_message_id") or "").strip() or None,
        error=str(payload.get("error") or "").strip() or None,
    )
    return {"ok": True, "outbound": row}


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
