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

_BUILD_ID: str = "fix48.debloat.bloatRemoval.concepts"
_SLEEP_SPLIT_MIN_LULL_SECONDS: int = 3 * 60 * 60
_DEFAULT_MIN_CHUNK_TOKENS: int = 8000
_DEFAULT_EPISODES_PER_SEGMENT: int = 3
_DEFAULT_BACKGROUND_SUMMARY_TOKENS: int = 1000
_DEFAULT_BACKGROUND_SUMMARY_MIN_TOKENS: int = 100
_MIN_CHUNK_TOKENS: int = _DEFAULT_MIN_CHUNK_TOKENS
_EPISODES_PER_SEGMENT: int = _DEFAULT_EPISODES_PER_SEGMENT
_BACKGROUND_SUMMARY_TOKENS: int = _DEFAULT_BACKGROUND_SUMMARY_TOKENS
_BACKGROUND_SUMMARY_MIN_TOKENS: int = _DEFAULT_BACKGROUND_SUMMARY_MIN_TOKENS
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
                min_timestamp=active_since,
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
    trigger_min_tokens: int,
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

        rolling_cursor_id = state_row.get("rolling_summary_cursor_id")
        tail = _load_background_rollup_tail(
            conversation_id=cid,
            user_id=uid,
            soul_id=sid,
            rolling_summary_cursor_id=int(rolling_cursor_id) if rolling_cursor_id is not None else None,
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
        if token_estimate < int(trigger_min_tokens):
            return "skipped_tokens"
        if not _background_sleep_gap_detected(
            history=sleep_history,
            safe=safe_payload,
            min_chunk_tokens=int(trigger_min_tokens),
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
            )
            or ""
        ).strip()
        if not new_summary:
            raise RuntimeError("background summarize returned empty summary")
        now_iso = datetime.now(UTC).isoformat()
        _write_conversation_state(
            cid,
            soul_id=sid,
            user_id=uid,
            updates={
                "rolling_summary": new_summary,
                "rolling_summary_cursor_id": tail_end_cursor,
                "rolling_summary_updated_at": now_iso,
                "updated_at": now_iso,
                "last_background_error": None,
                "last_background_error_at": None,
            },
        )
        return "rolled_up"


def _queue_background_rollup_task(
    *,
    conversation_id: str,
    user_id: str,
    soul_id: str,
    safe_payload: dict[str, Any],
    trigger_min_tokens: int,
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
            trigger_min_tokens=trigger_min_tokens,
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
    response_target = str(previous_contract.get("response_target") or "").strip().lower()
    response = str(previous_contract.get("response") or "").strip()
    rehearsal = str(previous_contract.get("rehearsal") or "").strip()
    target_instruction = (
        "If you choose response_target respond/private, it can be sent through WhatsApp."
        if allow_public_response
        else "Valid response_target values here are observe/private. Do not use listen/respond."
    )
    return "\n".join(
        [
            f"You chose continue_reason={reason!r} after the live turn from {origin_conversation_id}.",
            f"This is continuation turn {continuation_index} of 3.",
            "Continue only the specific task/research/diary purpose you chose.",
            "Return the same strict turn-contract JSON. Do not invent a new user message.",
            target_instruction,
            "",
            "Previous turn outcome:",
            f"- response_target: {response_target or 'unknown'}",
            f"- response: {response or '(empty)'}",
            f"- rehearsal: {rehearsal or '(empty)'}",
        ]
    )


def _parse_free_turn_contract(raw: Any, *, allow_public_response: bool) -> dict[str, Any]:
    try:
        return _parse_turn_contract(raw, allow_public_response=allow_public_response)
    except (ValueError, json.JSONDecodeError):
        text = str(raw or "").strip()
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("No complete JSON object found")
            parsed = json.loads(text[start : end + 1])
        except (ValueError, json.JSONDecodeError):
            raise
        logger.warning("free_turn: Claude Code returned prose around JSON; extracted turn contract")
        return _parse_turn_contract(json.dumps(parsed), allow_public_response=allow_public_response)


async def _persist_free_turn_summary(
    *,
    service: MemoryService,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    reason: str,
    continuation_index: int,
    contract: dict[str, Any],
    soul_card: str | None,
) -> None:
    response_target = str(contract.get("response_target") or "").strip().lower()
    response = str(contract.get("response") or "").strip()
    rehearsal = str(contract.get("rehearsal") or "").strip()
    cache_entry = str(contract.get("cache_entry") or "").strip()
    summary_lines = [
        f"{soul_id} took an agentic continuation turn.",
        f"Purpose: {reason}.",
        f"Origin conversation: {conversation_id}.",
        f"Continuation index: {continuation_index}.",
    ]
    if rehearsal:
        summary_lines.append(f"What {soul_id} worked through: {rehearsal}")
    if cache_entry:
        summary_lines.append(f"Working note: {cache_entry}")
    if response:
        summary_lines.append(f"Message intent ({response_target}): {response}")
    await service.memorize(
        resource_url=f"agentic-continuation:{uuid.uuid4()}",
        modality="conversation",
        raw_text="\n".join(summary_lines),
        user={
            "user_id": user_id,
            "soul_id": soul_id,
            "conversation_id": f"agentic:{conversation_id}",
        },
        soul_card=soul_card,
    )


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
) -> None:
    reason = initial_reason
    previous_contract = initial_contract
    try:
        for continuation_index in range(1, 4):
            prompt = _build_free_turn_prompt(
                reason=reason,
                continuation_index=continuation_index,
                origin_conversation_id=conversation_id,
                previous_contract=previous_contract,
                allow_public_response=allow_public_response,
            )
            raw = await service.chat(
                prompt,
                system_prompt=system_prompt,
                response_format={"type": "json_object"},
                op="free_turn",
                step=f"continue_{continuation_index}",
                resume_session_id=session_id,
            )
            contract = _parse_free_turn_contract(raw, allow_public_response=allow_public_response)
            await _persist_free_turn_summary(
                service=service,
                user_id=user_id,
                soul_id=soul_id,
                conversation_id=conversation_id,
                reason=reason,
                continuation_index=continuation_index,
                contract=contract,
                soul_card=soul_card,
            )
            response_target = str(contract.get("response_target") or "").strip().lower()
            response = str(contract.get("response") or "").strip()
            if response_target in {"respond", "private"} and response:
                if conversation_id.startswith("whatsapp:"):
                    out_id = _insert_whatsapp_outbound(
                        user_id=user_id,
                        soul_id=soul_id,
                        origin_conversation_id=conversation_id,
                        target=response_target,
                        response_text=response,
                        metadata={
                            "reason": reason,
                            "continuation_index": continuation_index,
                            "source": "free_turn",
                        },
                    )
                    logger.info("free_turn: queued WhatsApp outbound %s target=%s", out_id, response_target)
                else:
                    logger.info("free_turn: response ignored for non-WhatsApp conversation")
            cache_entry = str(contract.get("cache_entry") or "").strip()
            annulments = contract.get("annulments") if isinstance(contract.get("annulments"), list) else []
            if cache_entry or annulments:
                logger.info("free_turn: cache_entry/annulments intentionally ignored for continuation state")
            next_reason = str(contract.get("continue_reason") or "").strip().lower()
            if next_reason not in {"task", "research", "diary"}:
                if next_reason == "follow_up":
                    _schedule_free_turn_follow_up(
                        user_id=user_id,
                        soul_id=soul_id,
                        conversation_id=conversation_id,
                        follow_up_at=str(contract.get("follow_up_at") or ""),
                        follow_up_reason=str(contract.get("follow_up_reason") or ""),
                        safe_payload=safe_payload,
                    )
                return
            reason = next_reason
            previous_contract = contract
    except Exception:
        logger.exception("free_turn: continuation chain failed for %s", marker)
    finally:
        _clear_inflight(_FREE_TURN_INFLIGHT, marker)


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
) -> bool:
    marker = f"{user_id}::{soul_id}"
    if not _mark_inflight(_FREE_TURN_INFLIGHT, marker):
        logger.info("free_turn: continuation skipped because one is already running for %s", marker)
        return False
    task = asyncio.create_task(
        _run_free_turn_chain(
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
        )
    )
    _BACKGROUND_TASKS.add(task)

    def _on_done(done_task: asyncio.Task) -> None:
        _BACKGROUND_TASKS.discard(done_task)

    task.add_done_callback(_on_done)
    return True


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
    global _MIN_CHUNK_TOKENS, _EPISODES_PER_SEGMENT, _BACKGROUND_SUMMARY_TOKENS, _BACKGROUND_SUMMARY_MIN_TOKENS
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
    try:
        _BACKGROUND_SUMMARY_MIN_TOKENS = max(
            0,
            int(memorize_cfg.get("background_summary_min_tokens", _DEFAULT_BACKGROUND_SUMMARY_MIN_TOKENS)),
        )
    except (TypeError, ValueError, OverflowError):
        _BACKGROUND_SUMMARY_MIN_TOKENS = _DEFAULT_BACKGROUND_SUMMARY_MIN_TOKENS
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


def _ensure_whatsapp_outbounds_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
CREATE TABLE IF NOT EXISTS whatsapp_pending_outbounds (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    soul_id TEXT NOT NULL,
    origin_conversation_id TEXT NOT NULL,
    target TEXT NOT NULL CHECK (target IN ('respond', 'private')),
    target_conversation_id TEXT,
    response_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'claimed', 'sent', 'failed')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    claimed_at TEXT,
    claimed_by TEXT,
    sent_at TEXT,
    failed_at TEXT,
    provider_message_id TEXT,
    last_error TEXT,
    metadata_json TEXT
)
"""
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_whatsapp_pending_outbounds_claim "
        "ON whatsapp_pending_outbounds(status, created_at)"
    )
    con.commit()


def _whatsapp_outbound_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "soul_id": row["soul_id"],
        "origin_conversation_id": row["origin_conversation_id"],
        "target": row["target"],
        "target_conversation_id": row["target_conversation_id"],
        "response_text": row["response_text"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "claimed_at": row["claimed_at"],
        "claimed_by": row["claimed_by"],
        "sent_at": row["sent_at"],
        "failed_at": row["failed_at"],
        "provider_message_id": row["provider_message_id"],
        "last_error": row["last_error"],
        "metadata": _json_from_db(row["metadata_json"]) or {},
    }


def _insert_whatsapp_outbound(
    *,
    user_id: str,
    soul_id: str,
    origin_conversation_id: str,
    target: str,
    response_text: str,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    cid = str(origin_conversation_id or "").strip()
    target_clean = str(target or "").strip().lower()
    text = str(response_text or "").strip()
    if not uid or not sid or not cid:
        raise ValueError("user_id, soul_id, and origin_conversation_id are required")
    if target_clean not in {"respond", "private"}:
        raise ValueError("target must be respond|private")
    if not text:
        raise ValueError("response_text is required")

    db_path = _sqlite_current_path(uid, sid)
    if db_path is None:
        raise ValueError("sqlite path unavailable for outbound scope")
    _sqlite_ensure_nonempty(db_path)
    out_id = f"waout_{uuid.uuid4().hex}"
    now_iso = datetime.now(UTC).isoformat()
    target_conversation_id = cid if target_clean == "respond" else None
    con = _sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        _ensure_whatsapp_outbounds_schema(con)
        con.execute(
            """
INSERT INTO whatsapp_pending_outbounds (
    id, user_id, soul_id, origin_conversation_id, target, target_conversation_id,
    response_text, status, created_at, updated_at, metadata_json
) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
""",
            (
                out_id,
                uid,
                sid,
                cid,
                target_clean,
                target_conversation_id,
                text,
                now_iso,
                now_iso,
                _json_to_db(dict(metadata or {})),
            ),
        )
        con.commit()
    finally:
        con.close()
    return out_id


def _claim_whatsapp_outbounds(
    *,
    user_id: str,
    soul_id: str,
    claimed_by: str,
    limit: int = 10,
    claim_timeout_seconds: int = 300,
) -> list[dict[str, Any]]:
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    claimer = str(claimed_by or "").strip() or "hermes"
    if not uid or not sid:
        raise HTTPException(status_code=400, detail="user_id and soul_id are required")
    db_path = _sqlite_current_path(uid, sid)
    if db_path is None:
        raise HTTPException(status_code=400, detail="sqlite path unavailable for outbound scope")
    _sqlite_ensure_nonempty(db_path)
    now = datetime.now(UTC)
    now_iso = now.isoformat()
    stale_before = (now - timedelta(seconds=max(1, int(claim_timeout_seconds)))).isoformat()
    claim_limit = max(1, min(50, int(limit)))
    claimed: list[dict[str, Any]] = []
    con = _sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        _ensure_whatsapp_outbounds_schema(con)
        rows = con.execute(
            """
SELECT id
FROM whatsapp_pending_outbounds
WHERE user_id = ? AND soul_id = ?
  AND (status = 'pending' OR (status = 'claimed' AND claimed_at < ?))
ORDER BY created_at ASC
LIMIT ?
""",
            (uid, sid, stale_before, claim_limit),
        ).fetchall()
        for row in rows:
            out_id = str(row["id"])
            cur = con.execute(
                """
UPDATE whatsapp_pending_outbounds
SET status = 'claimed', claimed_at = ?, claimed_by = ?, updated_at = ?
WHERE id = ? AND user_id = ? AND soul_id = ?
  AND (status = 'pending' OR (status = 'claimed' AND claimed_at < ?))
""",
                (now_iso, claimer, now_iso, out_id, uid, sid, stale_before),
            )
            if cur.rowcount != 1:
                continue
            claimed_row = con.execute(
                "SELECT * FROM whatsapp_pending_outbounds WHERE id = ? LIMIT 1",
                (out_id,),
            ).fetchone()
            if claimed_row is not None:
                claimed.append(_whatsapp_outbound_row(claimed_row))
        con.commit()
    finally:
        con.close()
    return claimed


def _mark_whatsapp_outbound(
    *,
    user_id: str,
    soul_id: str,
    outbound_id: str,
    status: str,
    provider_message_id: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    out_id = str(outbound_id or "").strip()
    final_status = str(status or "").strip().lower()
    if not uid or not sid or not out_id:
        raise HTTPException(status_code=400, detail="user_id, soul_id, and outbound_id are required")
    if final_status not in {"sent", "failed"}:
        raise HTTPException(status_code=400, detail="status must be sent|failed")
    db_path = _sqlite_current_path(uid, sid)
    if db_path is None:
        raise HTTPException(status_code=400, detail="sqlite path unavailable for outbound scope")
    _sqlite_ensure_nonempty(db_path)
    now_iso = datetime.now(UTC).isoformat()
    sent_at = now_iso if final_status == "sent" else None
    failed_at = now_iso if final_status == "failed" else None
    con = _sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        _ensure_whatsapp_outbounds_schema(con)
        cur = con.execute(
            """
UPDATE whatsapp_pending_outbounds
SET status = ?, updated_at = ?, sent_at = ?, failed_at = ?,
    provider_message_id = COALESCE(?, provider_message_id),
    last_error = ?
WHERE id = ? AND user_id = ? AND soul_id = ? AND status = 'claimed'
""",
            (
                final_status,
                now_iso,
                sent_at,
                failed_at,
                str(provider_message_id or "").strip() or None,
                str(error or "").strip() or None,
                out_id,
                uid,
                sid,
            ),
        )
        if cur.rowcount != 1:
            raise HTTPException(status_code=409, detail="outbound is not claimed or does not exist")
        row = con.execute(
            "SELECT * FROM whatsapp_pending_outbounds WHERE id = ? LIMIT 1",
            (out_id,),
        ).fetchone()
        con.commit()
    finally:
        con.close()
    if row is None:
        raise HTTPException(status_code=404, detail="outbound not found")
    return _whatsapp_outbound_row(row)


def _ensure_free_turn_followups_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
CREATE TABLE IF NOT EXISTS free_turn_followups (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    soul_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    follow_up_at TEXT NOT NULL,
    due_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    claimed_at TEXT,
    completed_at TEXT,
    failed_at TEXT,
    last_error TEXT,
    payload_json TEXT NOT NULL
)
"""
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_free_turn_followups_due "
        "ON free_turn_followups(status, due_at)"
    )
    con.commit()


def _free_turn_followup_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "soul_id": row["soul_id"],
        "conversation_id": row["conversation_id"],
        "follow_up_at": row["follow_up_at"],
        "due_at": row["due_at"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "claimed_at": row["claimed_at"],
        "completed_at": row["completed_at"],
        "failed_at": row["failed_at"],
        "last_error": row["last_error"],
        "payload": _json_from_db(row["payload_json"]) or {},
    }


def _parse_free_turn_follow_up_at(raw: str) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = _parse_as_of_datetime(text)
        return parsed.astimezone(UTC) if parsed is not None else None
    except HTTPException:
        pass

    zone = _memorize_endpoint.server_timezone()
    candidates = [text, text.rsplit(" ", 1)[0]]
    for candidate in candidates:
        for fmt in ("%A, %B %d, %Y %H:%M", "%B %d, %Y %H:%M"):
            try:
                dt = datetime.strptime(candidate, fmt)
            except ValueError:
                continue
            return dt.replace(tzinfo=zone).astimezone(UTC)
    return None


def _free_turn_followup_payload(safe: dict[str, Any]) -> dict[str, Any]:
    kept: dict[str, Any] = {}
    for key in (
        "user",
        "user_name",
        "chat_name",
        "chat_type",
        "channel_mode",
        "soul_card",
        "memorize_chat",
        "allow_public_response",
    ):
        if key in safe:
            kept[key] = safe[key]
    return kept


def _schedule_free_turn_follow_up(
    *,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    follow_up_at: str,
    follow_up_reason: str,
    safe_payload: dict[str, Any],
) -> str | None:
    reason = str(follow_up_reason or "").strip()
    if not reason:
        logger.warning("free_turn: follow_up ignored because follow_up_reason is missing")
        return None
    due_at = _parse_free_turn_follow_up_at(follow_up_at)
    if due_at is None:
        logger.warning("free_turn: invalid follow_up_at ignored: %r", follow_up_at)
        return None
    db_path = _sqlite_current_path(user_id, soul_id)
    if db_path is None:
        logger.warning("free_turn: follow_up ignored because sqlite path is unavailable")
        return None
    _sqlite_ensure_nonempty(db_path)
    now_iso = datetime.now(UTC).isoformat()
    followup_id = f"wafup_{uuid.uuid4().hex}"
    payload = _free_turn_followup_payload(safe_payload)
    payload["follow_up_reason"] = reason
    con = _sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        _ensure_free_turn_followups_schema(con)
        con.execute(
            """
INSERT INTO free_turn_followups (
    id, user_id, soul_id, conversation_id, follow_up_at, due_at,
    status, created_at, updated_at, payload_json
) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
""",
            (
                followup_id,
                user_id,
                soul_id,
                conversation_id,
                follow_up_at,
                due_at.isoformat(),
                now_iso,
                now_iso,
                _json_to_db(payload),
            ),
        )
        con.commit()
    finally:
        con.close()
    logger.info("free_turn: scheduled follow_up %s due_at=%s", followup_id, due_at.isoformat())
    return followup_id


def _free_turn_followup_db_paths() -> list[Path]:
    base_dsn = str(_STORAGE_STATUS.get("dsn") or "")
    sqlite_dir = _sqlite_dir_from_cfg(_CONFIG, fallback_dsn=base_dsn)
    try:
        return sorted(path for path in sqlite_dir.glob("*.db") if path.is_file())
    except OSError:
        logger.exception("free_turn: failed to scan follow_up sqlite dir %s", sqlite_dir)
        return []


def _claim_due_free_turn_followups(
    db_path: Path,
    *,
    now: datetime,
    limit: int = 5,
    claim_timeout_seconds: int = 7200,
) -> list[dict[str, Any]]:
    claimed: list[dict[str, Any]] = []
    con = _sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        table = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'free_turn_followups'"
        ).fetchone()
        if table is None:
            return []
        now_iso = now.astimezone(UTC).isoformat()
        stale_before = (now.astimezone(UTC) - timedelta(seconds=max(1, int(claim_timeout_seconds)))).isoformat()
        rows = con.execute(
            """
SELECT id
FROM free_turn_followups
WHERE (status = 'pending' AND due_at <= ?)
   OR (status = 'running' AND claimed_at < ?)
ORDER BY due_at ASC
LIMIT ?
""",
            (now_iso, stale_before, max(1, min(20, int(limit)))),
        ).fetchall()
        for row in rows:
            followup_id = str(row["id"])
            cur = con.execute(
                """
UPDATE free_turn_followups
SET status = 'running', claimed_at = ?, updated_at = ?
WHERE id = ?
  AND ((status = 'pending' AND due_at <= ?) OR (status = 'running' AND claimed_at < ?))
""",
                (now_iso, now_iso, followup_id, now_iso, stale_before),
            )
            if cur.rowcount != 1:
                continue
            claimed_row = con.execute(
                "SELECT * FROM free_turn_followups WHERE id = ? LIMIT 1",
                (followup_id,),
            ).fetchone()
            if claimed_row is not None:
                claimed.append(_free_turn_followup_row(claimed_row))
        con.commit()
    finally:
        con.close()
    return claimed


def _mark_free_turn_followup(
    db_path: Path,
    followup_id: str,
    *,
    status: str,
    error: str | None = None,
) -> None:
    final_status = str(status or "").strip().lower()
    if final_status not in {"completed", "failed"}:
        raise ValueError("followup status must be completed|failed")
    now_iso = datetime.now(UTC).isoformat()
    completed_at = now_iso if final_status == "completed" else None
    failed_at = now_iso if final_status == "failed" else None
    con = _sqlite_connect(db_path)
    try:
        con.execute(
            """
UPDATE free_turn_followups
SET status = ?, updated_at = ?, completed_at = ?, failed_at = ?, last_error = ?
WHERE id = ? AND status = 'running'
""",
            (
                final_status,
                now_iso,
                completed_at,
                failed_at,
                str(error or "").strip() or None,
                followup_id,
            ),
        )
        con.commit()
    finally:
        con.close()


async def _run_free_turn_followup(row: dict[str, Any], db_path: Path) -> None:
    followup_id = str(row.get("id") or "").strip()
    marker = f"followup::{followup_id}"
    if not _mark_inflight(_FREE_TURN_FOLLOW_UP_INFLIGHT, marker):
        return
    try:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        user_id = str(row.get("user_id") or "").strip()
        soul_id = str(row.get("soul_id") or "").strip()
        conversation_id = str(row.get("conversation_id") or "").strip()
        user_scope = {"user_id": user_id, "soul_id": soul_id, "conversation_id": conversation_id}
        follow_up_reason = str(payload.get("follow_up_reason") or "").strip()
        trace_id = uuid.uuid4().hex
        message = (
            f"Scheduled follow-up due now. You asked to wake at {row.get('follow_up_at')}. "
            f"Reason you gave: {follow_up_reason}. "
            "Use fresh memory and current chat history, then decide whether to send a WhatsApp message."
        )
        retrieve_payload = {
            **payload,
            "user": user_scope,
            "self_turn_directive": message,
            "self_turn_label": "Scheduled wake",
            "history": [],
            "build_turn_prompt": True,
            "load_source_history": conversation_id.startswith("whatsapp:"),
            "is_live_turn": False,
            "trace_id": trace_id,
        }
        retrieve_out = await conversation_retrieve(conversation_id, retrieve_payload)
        prompt_override_payload = _mcp_tools.build_prompt_override_payload(retrieve_out)
        if not str(prompt_override_payload.get("user_prompt") or "").strip():
            raise RuntimeError("conversation_retrieve returned empty turn_user_prompt")
        turn_payload = {
            **payload,
            "user": user_scope,
            "self_turn_directive": message,
            "self_turn_label": "Scheduled wake",
            "history": retrieve_out.get("turn_history") if isinstance(retrieve_out.get("turn_history"), list) else [],
            "prompt_override_payload": prompt_override_payload,
            "load_source_history": conversation_id.startswith("whatsapp:"),
            "is_live_turn": False,
            "trace_id": trace_id,
        }
        result = await conversation_turn(conversation_id, turn_payload)
        response_target = str(result.get("response_target") or "").strip().lower()
        response = str(result.get("response") or "").strip()
        if response_target in {"respond", "private"} and response:
            outbound_target = response_target if conversation_id.startswith("whatsapp:") else "private"
            _insert_whatsapp_outbound(
                user_id=user_id,
                soul_id=soul_id,
                origin_conversation_id=conversation_id,
                target=outbound_target,
                response_text=response,
                metadata={
                    "source": "free_turn_follow_up",
                    "followup_id": followup_id,
                    "requested_target": response_target,
                },
            )
        _mark_free_turn_followup(db_path, followup_id, status="completed")
    except Exception as exc:
        logger.exception("free_turn: follow_up failed id=%s", followup_id)
        _mark_free_turn_followup(db_path, followup_id, status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        _clear_inflight(_FREE_TURN_FOLLOW_UP_INFLIGHT, marker)


async def _run_due_free_turn_followups_once() -> int:
    count = 0
    for db_path in _free_turn_followup_db_paths():
        for row in _claim_due_free_turn_followups(db_path, now=datetime.now(UTC)):
            count += 1
            await _run_free_turn_followup(row, db_path)
    return count


async def _free_turn_followup_scheduler() -> None:
    while True:
        try:
            await _run_due_free_turn_followups_once()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("free_turn: follow_up scheduler pass failed")
        await asyncio.sleep(30)


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
    "apimw_synthesis_parse_failed:",
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


async def _apimw_retrieve_items(
    payload: dict[str, Any],
    *,
    focus_text: str,
    soul_id: str,
    history: list[dict[str, Any]],
    state_row: dict[str, Any],
    conversation_id: str,
    apimw_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    retrieve_queries = _build_retrieve_soul_context_queries(
        soul_id=soul_id,
        message=focus_text,
        history=history,
        state_row=state_row,
        identity_mode="apimw",
        conversation_id=conversation_id,
    )
    retrieve_config = dict(payload.get("retrieve_config")) if isinstance(payload.get("retrieve_config"), dict) else {}
    item_config = dict(retrieve_config.get("item")) if isinstance(retrieve_config.get("item"), dict) else {}
    item_config["top_k"] = max(1, int(apimw_k))
    retrieve_config["item"] = item_config
    retrieve_payload = {
        **payload,
        "query": focus_text,
        "queries": retrieve_queries,
        "conversation_id": conversation_id,
        "force_retrieve": True,
        "retrieve_config": retrieve_config,
        "trace_id": uuid.uuid4().hex,
    }
    logger.info("apimw retrieve for %s", conversation_id)
    retrieve_out = await _run_retrieve(retrieve_payload, conversation_id=conversation_id)
    retrieve_result_data = retrieve_out.get("result") or {}
    retrieved_items = [item for item in (retrieve_result_data.get("items") or []) if isinstance(item, dict)]
    logger.info("apimw retrieved %d items for %s", len(retrieved_items), conversation_id)
    return retrieve_result_data, retrieved_items


async def _apimw_collect_memory_items(
    svc: Any,
    payload: dict[str, Any],
    *,
    focus_text: str,
    history: list[dict[str, Any]],
    state_row: dict[str, Any],
    conversation_id: str,
    soul_id: str,
    apimw_k: int,
    apimw_random_count: int,
    scope: dict[str, str],
) -> list[dict[str, Any]]:
    _retrieve_result, retrieved_items = await _apimw_retrieve_items(
        payload,
        focus_text=focus_text,
        soul_id=soul_id,
        history=history,
        state_row=state_row,
        conversation_id=conversation_id,
        apimw_k=apimw_k,
    )

    combined_items: list[dict[str, Any]] = []
    seen_item_sigs: set[str] = set()
    for item in retrieved_items:
        sig = _item_sig(item)
        if not sig or sig in seen_item_sigs:
            continue
        seen_item_sigs.add(sig)
        combined_items.append(item)

    if apimw_random_count > 0:
        pool = svc.database.memory_item_repo.list_items(scope, include_superseded=False)
        candidates: list[dict[str, Any]] = []
        for item in pool.values():
            item_id = str(item.id or "").strip()
            memory_type = str(item.memory_type or "memory")
            summary = str(item.summary or "").strip()
            if not item_id or not summary or memory_type == "narrative_self":
                continue
            row = {
                "id": item_id,
                "memory_type": memory_type,
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


async def _apimw_synthesize(
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
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]], dict[str, str]]:
    logger.info("apimw synthesis for %s", conversation_id)
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
        "- message_to_self: string or null — a brief thought you want surfaced as the first My Memories item for one turn. "
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
        step="synthesis",
    )

    apimw_response_text = str(llm_raw or "").strip()
    try:
        result_json = json.loads(apimw_response_text)
    except json.JSONDecodeError:
        logger.error("apimw synthesis: JSON parse failed, raw=%s", apimw_response_text[:200])
        return None, items_by_id, id_map
    if not isinstance(result_json, dict):
        logger.error("apimw synthesis: expected dict, got %s", type(result_json).__name__)
        return None, items_by_id, id_map

    logger.info("apimw synthesis: parsed JSON with keys %s for %s", list(result_json.keys()), conversation_id)
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
    expected_prior_context: str = "",
) -> None:
    async with _retrieve_scope_lock(user_id, soul_id):
        updates: dict[str, Any] = {}
        resolved_prior_context_ids: list[str] = []

        fresh_row, _, _ = _load_turn_state_and_soul_card(conversation_id, user_id=user_id, soul_id=soul_id)
        existing_prior = str(fresh_row.get("prior_context") or "").strip()
        if existing_prior != str(expected_prior_context or "").strip():
            logger.warning("apimw: stale prior_context for %s; skipping APImw persist", conversation_id)
            return

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
                updates["prior_context"] = new_prior

        message_to_self = str(result_json.get("message_to_self") or "").strip()
        if message_to_self:
            sc_text = message_to_self[:300]
            updates["apimw_message_to_self"] = sc_text
            try:
                sc_embedding = (await svc.embed([sc_text], profile="embedding"))[0]
                svc.database.memory_item_repo.create_item(
                    resource_id=None,
                    memory_type="subconscious",
                    source_role="soul",
                    summary=sc_text,
                    embedding=sc_embedding,
                    extra={"apimw_message_to_self": True},
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
        identity_context = _build_retrieve_identity_context(soul_id, apimw=True)
        focus_text = episode_text.strip()
        if not focus_text:
            logger.info("apimw skipped for %s: no recent conversation text", conversation_id)
            return

        combined_items = await _apimw_collect_memory_items(
            svc,
            payload,
            focus_text=focus_text,
            history=history,
            state_row=state_row,
            conversation_id=conversation_id,
            soul_id=soul_id,
            apimw_k=apimw_item_top_k,
            apimw_random_count=apimw_random_count,
            scope=scope,
        )

        apimw_heavy_profile = _resolve_profile(svc, "memory_extract")
        result_json, items_by_id, apimw_id_map = await _apimw_synthesize(
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
                    code="apimw_synthesis_parse_failed",
                    detail="synthesis response was not valid JSON object",
                )
            except Exception:
                logger.exception("failed to record APImw synthesis-parse failure for %s", conversation_id)
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
            expected_prior_context=str(state_row.get("prior_context") or ""),
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


def _resolve_cross_source_paths() -> tuple[Path, Path | None, Path | None, Path | None]:
    hermes_cfg = _CONFIG.get("hermes") if isinstance(_CONFIG.get("hermes"), dict) else {}
    hermes_home_raw = str(hermes_cfg.get("home") or "").strip()
    sessions_index_raw = str(hermes_cfg.get("sessions_index_path") or "").strip()
    state_db_raw = str(hermes_cfg.get("state_db_path") or "").strip()
    hermes_home_path = Path(hermes_home_raw).expanduser().resolve() if hermes_home_raw else None
    sessions_index_path = Path(sessions_index_raw).expanduser().resolve() if sessions_index_raw else None
    state_db_path = Path(state_db_raw).expanduser().resolve() if state_db_raw else None
    return _get_storage_dir(_CONFIG), hermes_home_path, sessions_index_path, state_db_path


def _resolve_whatsapp_source_config() -> tuple[str, Path | None, str]:
    hermes_cfg = _CONFIG.get("hermes") if isinstance(_CONFIG.get("hermes"), dict) else {}
    source = str(hermes_cfg.get("whatsapp_history_source") or "hermes_state").strip().lower()
    if source in {"state_db", "hermes_state"}:
        source = "hermes_state"
    elif source != "web_source":
        raise RuntimeError(f"unsupported hermes.whatsapp_history_source: {source!r}")
    db_raw = str(hermes_cfg.get("whatsapp_web_source_db") or "").strip()
    db_path = Path(db_raw).expanduser().resolve() if db_raw else None
    reply_prefix = str(hermes_cfg.get("whatsapp_reply_prefix") or "")
    return source, db_path, reply_prefix


def _resolve_whatsapp_history_limit() -> int:
    hermes_cfg = _CONFIG.get("hermes") if isinstance(_CONFIG.get("hermes"), dict) else {}
    raw = hermes_cfg.get("whatsapp_history_limit", 250)
    try:
        return max(1, min(int(raw), 5000))
    except (TypeError, ValueError):
        return 250


def _load_soul_active_since(
    soul_id: str,
    *,
    hermes_home_path: Path | None,
    state_db_path: Path | None,
) -> float | None:
    return _conversation_sources.load_soul_active_since(
        soul_id=soul_id,
        hermes_home=hermes_home_path,
        state_db_path=state_db_path,
    )


_ACTIVE_SINCE_UNSET = object()


def _current_whatsapp_active_since_for_soul(
    conversation_id: str,
    soul_id: str,
) -> float | None:
    if not _message_log.derive_source_label(conversation_id).startswith("whatsapp:"):
        return None
    _storage_dir, hermes_home_path, _sessions_index_path, state_db_path = _resolve_cross_source_paths()
    return _load_soul_active_since(
        soul_id,
        hermes_home_path=hermes_home_path,
        state_db_path=state_db_path,
    )


def _filter_current_whatsapp_history_for_soul(
    conversation_id: str,
    soul_id: str,
    history: list[dict[str, Any]],
    *,
    active_since: Any = _ACTIVE_SINCE_UNSET,
) -> list[dict[str, Any]]:
    if not history or not _message_log.derive_source_label(conversation_id).startswith("whatsapp:"):
        return history
    if active_since is _ACTIVE_SINCE_UNSET:
        active_since = _current_whatsapp_active_since_for_soul(conversation_id, soul_id)
    if active_since is None:
        return history

    threshold_ms = active_since * 1000.0
    out: list[dict[str, Any]] = []
    for i, msg in enumerate(history):
        ts_ms = msg.get("ts_ms")
        if isinstance(ts_ms, bool) or not isinstance(ts_ms, (int, float)):
            raise HTTPException(
                status_code=400,
                detail=f"WhatsApp history row {i} is missing ts_ms for soul active_since cutoff",
            )
        if float(ts_ms) >= threshold_ms:
            out.append(msg)
    return out


def _degrade_live_whatsapp_history_after_filter_error(
    *,
    conversation_id: str,
    soul_id: str,
    history: list[dict[str, Any]],
    active_since: Any,
    exc: HTTPException,
    op: str,
) -> list[dict[str, Any]]:
    logger.warning(
        "%s: live WhatsApp history cutoff failed; continuing with valid in-scope rows only "
        "conversation_id=%s soul_id=%s detail=%s",
        op,
        conversation_id,
        soul_id,
        exc.detail,
    )
    if active_since is None:
        return history
    threshold_ms = active_since * 1000.0
    out: list[dict[str, Any]] = []
    dropped = 0
    for msg in history:
        ts_ms = msg.get("ts_ms")
        if isinstance(ts_ms, bool) or not isinstance(ts_ms, (int, float)):
            dropped += 1
            continue
        if float(ts_ms) >= threshold_ms:
            out.append(msg)
    if dropped:
        logger.warning(
            "%s: dropped %d live WhatsApp history row(s) without ts_ms conversation_id=%s soul_id=%s",
            op,
            dropped,
            conversation_id,
            soul_id,
        )
    return out


def _source_id_matches_external(source_id: Any, external_message_id: Any) -> bool:
    source = str(source_id or "").strip()
    external = str(external_message_id or "").strip()
    return bool(source and external and (source == external or external in source))


def _load_current_whatsapp_history_from_source(
    conversation_id: str,
    soul_id: str,
    *,
    active_since: float | None,
    external_message_id: Any = None,
) -> list[dict[str, Any]] | None:
    if not _message_log.derive_source_label(conversation_id).startswith("whatsapp:"):
        return None
    _storage_dir, hermes_home_path, sessions_index_path, state_db_path = _resolve_cross_source_paths()
    whatsapp_source, web_source_db_path, reply_prefix = _resolve_whatsapp_source_config()
    history_limit = _resolve_whatsapp_history_limit()
    if whatsapp_source == "web_source":
        assistant_ids = _conversation_sources.load_whatsapp_assistant_source_message_ids(
            conversation_id=conversation_id,
            hermes_home=hermes_home_path,
            sessions_index_path=sessions_index_path,
            state_db_path=state_db_path,
        )
        rows = _conversation_sources.load_whatsapp_web_source_tail(
            conversation_id=conversation_id,
            since_cursor=-1,
            recent_fallback_messages=0,
            soul_id=soul_id,
            reply_prefix=reply_prefix,
            hermes_home=hermes_home_path,
            web_source_db_path=web_source_db_path,
            min_timestamp=active_since,
            max_messages=history_limit,
            assistant_source_message_ids=assistant_ids,
        )
    else:
        rows = _conversation_sources.load_whatsapp_tail(
            conversation_id=conversation_id,
            since_cursor=-1,
            recent_fallback_messages=0,
            hermes_home=hermes_home_path,
            sessions_index_path=sessions_index_path,
            state_db_path=state_db_path,
            min_timestamp=active_since,
            max_messages=history_limit,
        )
    if not external_message_id:
        return rows
    return [
        row for row in rows
        if not _source_id_matches_external(row.get("source_message_id"), external_message_id)
    ]


def _prepare_current_whatsapp_history(
    *,
    conversation_id: str,
    soul_id: str,
    raw_history: Any,
    active_since: float | None,
    load_source_history: bool,
    external_message_id: Any,
    is_live_turn: bool,
    op: str,
) -> list[dict[str, Any]]:
    history = _normalize_turn_history(raw_history)
    if load_source_history:
        try:
            source_history = _load_current_whatsapp_history_from_source(
                conversation_id,
                soul_id,
                active_since=active_since,
                external_message_id=external_message_id,
            )
        except Exception as exc:
            if not is_live_turn:
                raise
            logger.warning(
                "%s: live WhatsApp source history load failed; continuing with payload history "
                "conversation_id=%s soul_id=%s: %s",
                op,
                conversation_id,
                soul_id,
                exc,
            )
            source_history = None
        if source_history is not None:
            history = _normalize_turn_history(source_history)
    try:
        return _filter_current_whatsapp_history_for_soul(
            conversation_id,
            soul_id,
            history,
            active_since=active_since,
        )
    except HTTPException as exc:
        if not is_live_turn:
            raise
        return _degrade_live_whatsapp_history_after_filter_error(
            conversation_id=conversation_id,
            soul_id=soul_id,
            history=history,
            active_since=active_since,
            exc=exc,
            op=op,
        )


def _stamp_assistant_display_name(messages: list[dict[str, Any]], soul_name: str) -> None:
    display = str(soul_name or "").strip()
    if not display:
        return
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip().lower()
        if role != "assistant":
            continue
        existing = str(msg.get("name") or msg.get("speaker") or "").strip()
        if existing:
            continue
        msg["name"] = display
        msg["speaker"] = display


def _load_tail_for_source_conversation(
    *,
    conversation_id: str,
    user_id: str,
    soul_id: str,
    since_cursor: int,
    recent_fallback_messages: int,
    storage_dir: Path,
    hermes_home_path: Path | None,
    sessions_index_path: Path | None,
    state_db_path: Path | None,
) -> list[dict[str, Any]]:
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
            return _conversation_sources.load_whatsapp_web_source_tail(
                conversation_id=conversation_id,
                since_cursor=since_cursor,
                recent_fallback_messages=recent_fallback_messages,
                soul_id=soul_id,
                reply_prefix=reply_prefix,
                hermes_home=hermes_home_path,
                web_source_db_path=web_source_db_path,
                min_timestamp=active_since,
                assistant_source_message_ids=assistant_ids,
            )
        return _conversation_sources.load_whatsapp_tail(
            conversation_id=conversation_id,
            since_cursor=since_cursor,
            recent_fallback_messages=recent_fallback_messages,
            hermes_home=hermes_home_path,
            sessions_index_path=sessions_index_path,
            state_db_path=state_db_path,
            min_timestamp=active_since,
        )
    if source_label == "sillytavern":
        return _conversation_sources.load_sillytavern_tail(
            storage_dir=storage_dir,
            user_id=user_id,
            soul_id=soul_id,
            conversation_id=conversation_id,
            since_cursor=since_cursor,
            recent_fallback_messages=recent_fallback_messages,
        )
    return []


def _load_cross_tail_from_sources(
    con: sqlite3.Connection,
    *,
    user_id: str,
    soul_id: str,
    exclude_conversation_id: str | None = None,
) -> list[dict[str, Any]]:
    storage_dir, hermes_home_path, sessions_index_path, state_db_path = _resolve_cross_source_paths()
    excluded_id = str(exclude_conversation_id or "").strip()
    cursor_rows = con.execute(
        "SELECT conversation_id, digest_cursor, last_memorize_at FROM conversations"
    ).fetchall()
    all_messages: list[dict[str, Any]] = []
    for row in cursor_rows:
        cid = str(row["conversation_id"] or "").strip()
        if not cid or cid == excluded_id:
            continue
        source_label = _message_log.derive_source_label(cid)
        cursor = int(row["digest_cursor"] or 0) if row["last_memorize_at"] else -1
        try:
            tail = _load_tail_for_source_conversation(
                conversation_id=cid,
                user_id=user_id,
                soul_id=soul_id,
                since_cursor=cursor,
                recent_fallback_messages=_message_log.DEFAULT_CROSS_RECENT_FALLBACK_MESSAGES,
                storage_dir=storage_dir,
                hermes_home_path=hermes_home_path,
                sessions_index_path=sessions_index_path,
                state_db_path=state_db_path,
            )
        except Exception as exc:
            logger.error("cross-context source read failed for conversation_id=%s: %s", cid, exc)
            is_web_source_whatsapp = (
                source_label.startswith("whatsapp:")
                and _resolve_whatsapp_source_config()[0] == "web_source"
            )
            if is_web_source_whatsapp:
                raise RuntimeError(f"WhatsApp web_source read failed for {cid}: {exc}") from exc
            continue
        _stamp_assistant_display_name(tail, soul_id)
        all_messages.extend(tail)
    all_messages.sort(
        key=lambda msg: (
            str(msg.get("received_at") or ""),
            str(msg.get("conversation_id") or ""),
            int(msg.get("source_conversation_index") or 0),
        )
    )
    return all_messages


def _load_cross_memorize_tails_from_sources(
    con: sqlite3.Connection,
    *,
    user_id: str,
    soul_id: str,
    exclude_conversation_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    storage_dir, hermes_home_path, sessions_index_path, state_db_path = _resolve_cross_source_paths()
    excluded_id = str(exclude_conversation_id or "").strip()
    rows = con.execute(
        "SELECT conversation_id, digest_cursor, last_memorize_at, memorize_chat, rolling_summary_cursor_id "
        "FROM conversations"
    ).fetchall()
    tails: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cid = str(row["conversation_id"] or "").strip()
        if not cid or cid == excluded_id:
            continue
        try:
            memorize_chat = True if row["memorize_chat"] is None else bool(int(row["memorize_chat"]))
            if memorize_chat:
                cursor = int(row["digest_cursor"] or 0) if row["last_memorize_at"] else -1
                tail = _load_tail_for_source_conversation(
                    conversation_id=cid,
                    user_id=user_id,
                    soul_id=soul_id,
                    since_cursor=cursor,
                    recent_fallback_messages=0,
                    storage_dir=storage_dir,
                    hermes_home_path=hermes_home_path,
                    sessions_index_path=sessions_index_path,
                    state_db_path=state_db_path,
                )
                _stamp_assistant_display_name(tail, soul_id)
                if not tail:
                    continue
                for i, msg in enumerate(tail):
                    msg["source_conversation_id"] = cid
                    if msg.get("source_conversation_index") is None:
                        msg["source_conversation_index"] = cursor + 1 + i
                    msg["memorize_chat"] = True
                tails[cid] = tail
                continue

            rolling_cursor_id = row["rolling_summary_cursor_id"]
            source_label = _message_log.derive_source_label(cid)
            if source_label.startswith("whatsapp:"):
                active_since = _load_soul_active_since(
                    soul_id,
                    hermes_home_path=hermes_home_path,
                    state_db_path=state_db_path,
                )
                whatsapp_source, web_source_db_path, reply_prefix = _resolve_whatsapp_source_config()
                if whatsapp_source == "web_source":
                    assistant_ids = _conversation_sources.load_whatsapp_assistant_source_message_ids(
                        conversation_id=cid,
                        hermes_home=hermes_home_path,
                        sessions_index_path=sessions_index_path,
                        state_db_path=state_db_path,
                    )
                    tail = _conversation_sources.load_whatsapp_web_source_tail_after_rowid(
                        conversation_id=cid,
                        after_rowid=int(rolling_cursor_id) if rolling_cursor_id is not None else None,
                        soul_id=soul_id,
                        reply_prefix=reply_prefix,
                        hermes_home=hermes_home_path,
                        web_source_db_path=web_source_db_path,
                        min_timestamp=active_since,
                        assistant_source_message_ids=assistant_ids,
                    )
                else:
                    tail = _conversation_sources.load_whatsapp_tail_after_message_id(
                        conversation_id=cid,
                        after_message_id=int(rolling_cursor_id) if rolling_cursor_id is not None else None,
                        hermes_home=hermes_home_path,
                        sessions_index_path=sessions_index_path,
                        state_db_path=state_db_path,
                        min_timestamp=active_since,
                    )
            elif source_label == "sillytavern":
                tail = _conversation_sources.load_sillytavern_tail(
                    storage_dir=storage_dir,
                    user_id=user_id,
                    soul_id=soul_id,
                    conversation_id=cid,
                    since_cursor=int(rolling_cursor_id) if rolling_cursor_id is not None else -1,
                    recent_fallback_messages=0,
                )
            else:
                continue
            _stamp_assistant_display_name(tail, soul_id)
            if not tail:
                continue
            for msg in tail:
                msg["source_conversation_id"] = cid
                if msg.get("source_conversation_index") is None:
                    raise RuntimeError(
                        f"cross-memorize listen-only tail missing source_conversation_index for {cid}"
                    )
                msg["source_conversation_index"] = int(msg["source_conversation_index"])
                msg["memorize_chat"] = False
            tails[cid] = tail
        except Exception as exc:
            logger.error("cross-memorize source read failed for conversation_id=%s: %s", cid, exc)
            is_web_source_whatsapp = (
                _message_log.derive_source_label(cid).startswith("whatsapp:")
                and _resolve_whatsapp_source_config()[0] == "web_source"
            )
            if is_web_source_whatsapp:
                raise RuntimeError(f"WhatsApp web_source read failed for {cid}: {exc}") from exc
            continue
    return tails


def _read_background_rolling_summaries_from_conversations(
    con: sqlite3.Connection,
    *,
    exclude_conversation_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    rows = con.execute(
        "SELECT conversation_id, memorize_chat, rolling_summary, rolling_summary_updated_at "
        "FROM conversations"
    ).fetchall()
    excluded_id = str(exclude_conversation_id or "").strip()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = str(row["conversation_id"] or "").strip()
        if not cid or cid == excluded_id:
            continue
        memorize_chat = True if row["memorize_chat"] is None else bool(int(row["memorize_chat"]))
        if memorize_chat:
            continue
        summary = str(row["rolling_summary"] or "").strip()
        if not summary:
            continue
        out[cid] = {
            "source_conversation_id": cid,
            "source_label": _message_log.derive_source_label(cid),
            "summary": summary,
            "updated_at": row["rolling_summary_updated_at"],
        }
    return out


TURN_HISTORY_WINDOW_MESSAGES = 8


def _sillytavern_turn_history_with_floor(
    history: list[dict[str, Any]],
    state_row: dict[str, Any],
) -> list[dict[str, Any]]:
    if not history:
        return []
    digest_cursor = int(state_row.get("digest_cursor") or 0) if state_row.get("last_memorize_at") else -1
    start = max(0, digest_cursor + 1)
    window = history[start:] if start < len(history) else []
    if len(window) < TURN_HISTORY_WINDOW_MESSAGES and len(history) > len(window):
        window = history[-TURN_HISTORY_WINDOW_MESSAGES:]
    return window


def _max_nonnegative_source_conversation_index(messages: list[dict[str, Any]]) -> int | None:
    max_index: int | None = None
    for msg in messages:
        try:
            idx = int(msg.get("source_conversation_index"))
        except (TypeError, ValueError, OverflowError):
            continue
        if idx < 0:
            continue
        if max_index is None or idx > max_index:
            max_index = idx
    return max_index


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
        self_turn_directive = _pick_str(safe, "self_turn_directive") or ""
        self_turn_label = _pick_str(safe, "self_turn_label") or ""
        retrieve_focus = self_turn_directive or message
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
        )
        safe["history"] = history
        _stamp_assistant_display_name(history, soul_id)
        state_row: dict[str, Any] | None = None
        cross_tail: list[dict[str, Any]] = []
        if uid and soul_id and retrieve_focus.strip():
            if _message_log.derive_source_label(cid) == "sillytavern" and history and message.strip():
                _conversation_sources.persist_sillytavern_history_snapshot(
                    storage_dir=_get_storage_dir(_CONFIG),
                    user_id=uid,
                    soul_id=soul_id,
                    conversation_id=cid,
                    history=history,
                    chat_name=_pick_str(safe, "chat_name") or None,
                )
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
                    cross_tail = _load_cross_tail_from_sources(
                        _con,
                        user_id=uid,
                        soul_id=soul_id,
                        exclude_conversation_id=cid,
                    )
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

        should_build_default_queries = (
            safe.get("queries") is None
            and uid
            and soul_id
            and retrieve_focus.strip()
            and state_row is not None
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
                history=history,
                state_row=state_row or {},
                conversation_id=cid,
                chat_label=chat_label_for_prompt,
                self_turn_directive=self_turn_directive or None,
                self_turn_label=self_turn_label or None,
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
                self_turn_directive = _pick_str(safe, "self_turn_directive") or ""
                self_turn_label = _pick_str(safe, "self_turn_label") or ""
                turn_history = history
                if _message_log.derive_source_label(cid) == "sillytavern":
                    turn_history = _sillytavern_turn_history_with_floor(history, _state_row)
                memory_cache = _normalize_memory_cache_impl(out.get("memory_cache"))
                intentions_active = _normalize_intentions_stack_impl(out.get("intentions_active"))

                out["turn_system_prompt"] = _make_turn_system_prompt(
                    soul_id,
                    soul_card=soul_card,
                    response_sentences=int(_CONFIG.get("turn_response_sentences", 3)),
                    allow_public_response=bool(safe.get("allow_public_response", True)),
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
                    cross_conversation_history=safe.get("_cross_conversation_history"),
                    chat_label=chat_label_for_prompt,
                    conversation_id=cid,
                    self_turn_directive=self_turn_directive or None,
                    self_turn_label=self_turn_label or None,
                )
                out["turn_prompt_source"] = "conversation_retrieve"
            if current_whatsapp_active_since is not None:
                out["turn_prompt_active_since"] = current_whatsapp_active_since
            if safe.get("_cross_conversation_history"):
                out["cross_conversation_history"] = safe.get("_cross_conversation_history")
            if is_live_turn:
                out["turn_history"] = history

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

    for other_cid, tail_msgs in other_tails.items():
        if not tail_msgs:
            continue
        final_cursor = _max_nonnegative_source_conversation_index(tail_msgs)
        if final_cursor is not None:
            final_cursors[other_cid] = final_cursor
        all_messages.extend(tail_msgs)
        if not bool(tail_msgs[0].get("memorize_chat", True)):
            token_estimate = _estimate_tokens(
                [{"content": str(msg.get("content") or "")} for msg in tail_msgs]
            )
            if token_estimate >= _BACKGROUND_SUMMARY_MIN_TOKENS:
                _queue_background_rollup_task(
                    conversation_id=other_cid,
                    user_id=uid,
                    soul_id=soul_id,
                    safe_payload=safe,
                    trigger_min_tokens=_BACKGROUND_SUMMARY_MIN_TOKENS,
                )

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
    if (not dry_run) and primary_history:
        has_sleep_gap = _unmemorized_sleep_gap_detected(
            primary_history,
            unmemorized_digest_cursor,
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
                    primary_history,
                    unmemorized_digest_cursor,
                    True,
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
    updates["apimw_message_to_self"] = None
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
        )
        turn_user_prompt = override_user_prompt

        memory_service = _get_service_from_payload(safe)
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
            async with state_lock:
                retrieved_item_ids = _extract_result_item_ids(override_retrieve_rag)
                conversation_state_after, conversation_state_path = _turn_state_write(
                    cid, uid, soul_id,
                    turn_cache_entry, turn_annulment_ids,
                    retrieved_item_ids,
                    memorize_chat=memorize_chat,
                )
            if not bool(conversation_state_after.get("memorize_chat", True)):
                _queue_background_rollup_task(
                    conversation_id=cid,
                    user_id=uid,
                    soul_id=soul_id,
                    safe_payload=safe,
                    trigger_min_tokens=_BACKGROUND_SUMMARY_TOKENS,
                    service=memory_service,
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
        if response_target not in {"respond", "listen", "observe", "private"}:
            raise HTTPException(status_code=502, detail="turn contract missing or invalid response_target")
        response_text = str(turn_contract.get("response") or "").strip()
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
