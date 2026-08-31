import asyncio
import hashlib
import json
import time as _time
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, TypedDict

from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from app.services.conversation_id import canonical_conversation_id
from app.services.payload import message_ts_ms
from app.services.state import effective_digest_cursor_from_row


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    words = sum(len(str(m.get("content") or m.get("mes") or "").split()) for m in messages)
    return int(words / 0.75)


_FORCE_MEMORIZE_MAX_CHUNK_TOKENS: int = 6000


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
        msg_tokens = estimate_tokens([messages[idx]])
        if chunk_tokens > 0 and (chunk_tokens + msg_tokens) > max_chunk_tokens:
            ranges.append((chunk_start, idx - 1))
            chunk_start = idx
            chunk_tokens = 0
        chunk_tokens += msg_tokens
    ranges.append((chunk_start, end_idx))
    return ranges


def estimate_unmemorized_tokens(messages: list[dict[str, Any]], digest_cursor: Any) -> int:
    if not messages:
        return 0
    try:
        cursor = int(digest_cursor)
    except (TypeError, ValueError, OverflowError):
        cursor = -1
    start = max(0, cursor + 1)
    if start >= len(messages):
        return 0
    return estimate_tokens(messages[start:])


def stamp_current_conversation_metadata(
    messages: list[dict[str, Any]],
    *,
    conversation_id: str | None,
    chat_name: str | None,
) -> None:
    cid = str(conversation_id or "").strip()
    name = str(chat_name or "").strip()
    if not cid and not name:
        return
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        source_cid = str(msg.get("source_conversation_id") or "").strip()
        if not source_cid and cid:
            msg["source_conversation_id"] = cid
            source_cid = cid
        if name and source_cid == cid and not str(msg.get("chat_name") or "").strip():
            msg["chat_name"] = name




class SegmentMemorizeJob(TypedDict):
    segment_payload: dict[str, Any]
    segment_messages: list[dict[str, Any]]
    segment_resource_url: str
    segment_raw_text: str
    segment_start_index: int
    segment_end_index: int
    segment_id: str | None
    memory_producing: bool


MemorizeSegment = tuple[
    str,
    list[dict[str, Any]],
    int,
    int,
    tuple[int, int] | None,
]


def _cursor_updates_for_unit(
    *,
    memory_producing: bool,
    cursor: int | None,
    now_iso: str,
    source_message_id: str = "",
    source_ts: Any = None,
    pending_segment_ids: list[str] | None = None,
) -> dict[str, Any]:
    pending = pending_segment_ids or []
    if not memory_producing and pending:
        raise RuntimeError("context-only memorize unit cannot append pending segment ids")

    updates: dict[str, Any] = {}
    if cursor is not None:
        prefix = "digest" if memory_producing else "rolling_summary"
        updates[f"{prefix}_cursor" if memory_producing else "rolling_summary_cursor_id"] = max(0, cursor)
        updates[f"{prefix}_cursor_source_message_id"] = source_message_id or None
        updates[f"{prefix}_cursor_ts"] = int(source_ts) if source_message_id and source_ts is not None else None
    if memory_producing:
        updates["last_memorize_at"] = now_iso
        if pending:
            updates["append_pending_segment_ids"] = pending
    return updates


@dataclass(slots=True)
class MemorizeContext:
    get_memorize_lock: Callable[[str], asyncio.Lock]
    memorize_lock_key: Callable[[str, str], str]
    write_conversation_state: Callable[..., tuple[dict[str, Any], Any]]
    memorize_progress: dict[str, dict[str, Any]]
    memorize_cancel: set[str]
    record_call: Callable[..., None]
    logger: Any
    min_chunk_tokens: int
    sleep_split_min_lull_seconds: int
    consolidation_due: Callable[[dict[str, Any]], bool]


@dataclass(slots=True)
class MemorizeRunContext:
    base: MemorizeContext
    load_turn_state_and_soul_card: Callable[..., tuple[dict[str, Any], str | None, str | None]]
    normalize_text_list: Callable[[Any], list[str]]
    run_consolidation_task: Callable[..., Awaitable[dict[str, Any]]]
    clear_last_display_segments_for_nonparticipants: Callable[..., None]
    resolve_web_source_checkpoint: Callable[[str, str], int | None]
    background_tasks_set: set[asyncio.Task]


@dataclass(slots=True)
class MemorizeEndpointContext:
    base: MemorizeContext
    safe_payload: Callable[[dict[str, Any]], dict[str, Any]]
    get_service_from_payload: Callable[[dict[str, Any]], Any]
    extract_scope: Callable[[dict[str, Any]], dict[str, Any] | None]
    extract_conversation_id: Callable[[dict[str, Any]], str | None]
    normalize_conversation: Callable[[Any], Any]
    pick_str: Callable[..., str | None]
    sqlite_current_path: Callable[[str | None, str], Path | None]
    clear_cached_services: Callable[[], None]
    get_storage_dir: Callable[[dict[str, Any]], Path]
    run_memorize_segments: Callable[..., Awaitable[None]]
    run_consolidation_task: Callable[..., Awaitable[dict[str, Any]]]
    get_config: Callable[[], dict[str, Any]]
    sanitize_db_filename: Callable[[str], str]


def _segment_display_ranges(
    segment_messages: list[dict[str, Any]],
) -> dict[str, tuple[int, int]]:
    ranges: dict[str, tuple[int, int]] = {}
    for fallback_idx, msg in enumerate(segment_messages):
        if not isinstance(msg, dict):
            continue
        if not bool(msg.get("memorize_chat", True)):
            continue
        cid = str(msg.get("source_conversation_id") or msg.get("conversation_id") or "").strip()
        if not cid:
            continue
        raw_index = msg.get("source_conversation_index")
        try:
            index = int(raw_index)
        except (TypeError, ValueError, OverflowError):
            index = fallback_idx
        current = ranges.get(cid)
        if current is None:
            ranges[cid] = (index, index)
        else:
            ranges[cid] = (min(current[0], index), max(current[1], index))
    return ranges


def _set_memorize_progress(
    memorize_progress: dict[str, dict[str, Any]],
    key: str,
    *,
    active: bool,
    phase: str | None = None,
    current: int | None = None,
    total: int | None = None,
    last_result: str | None = None,
    error: str | None = None,
) -> None:
    row: dict[str, Any] = {"active": bool(active)}
    if phase:
        row["phase"] = str(phase)
    if current is not None:
        row["current"] = int(current)
    if total is not None:
        row["total"] = int(total)
    if last_result:
        row["last_result"] = str(last_result)
    if error:
        row["error"] = str(error)
    row["updated_at"] = datetime.now(UTC).isoformat()
    memorize_progress[key] = row


async def run_forced_memorize_from_turn(
    payload: dict[str, Any],
    *,
    memorize_handler: Callable[[dict[str, Any], BackgroundTasks, bool], Awaitable[Any]],
    logger: Any,
    get_memorize_lock: Callable[[str], asyncio.Lock],
    memorize_lock_key: Callable[[str, str], str],
    write_conversation_state: Callable[..., tuple[dict[str, Any], Any]],
) -> bool:
    try:
        background_tasks = BackgroundTasks()
        await memorize_handler(payload, background_tasks, True)
        await background_tasks()
        return True
    except Exception as exc:
        logger.exception("forced memorize from turn failed")
        # Surface the failure on conversation state so operator + soul can see it.
        # Log-only failures silently lose episodes; the state field gives a trail.
        state_write_error: Exception | None = None
        try:
            scope = payload.get("user") if isinstance(payload.get("user"), dict) else {}
            conversation_id = str(scope.get("conversation_id") or payload.get("conversation_id") or "").strip()
            soul_id = str(scope.get("soul_id") or "").strip() or None
            user_id = str(scope.get("user_id") or "").strip() or None
            if conversation_id and soul_id:
                lock_user_id = user_id or "user"
                async with get_memorize_lock(memorize_lock_key(lock_user_id, soul_id)):
                    write_conversation_state(
                        conversation_id,
                        soul_id=soul_id,
                        user_id=user_id,
                        updates={
                            "last_background_error": f"forced_memorize: {type(exc).__name__}: {str(exc)[:300]}",
                            "last_background_error_at": datetime.now(UTC).isoformat(),
                        },
                    )
        except Exception as state_exc:
            state_write_error = state_exc
            logger.exception("failed to record background error on conversation state")
        if state_write_error is not None:
            msg = (
                "forced memorize failed and background error state write also failed; "
                "check logs for both exceptions"
            )
            raise RuntimeError(msg) from state_write_error
        return False


async def run_memorize_segments(
    *,
    memorize_segments: list[MemorizeSegment],
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
    run_ctx: MemorizeRunContext,
    segments_dir: Path,
    zi: Any | None = None,
    cross_memorize: bool = False,
    final_cursors: dict[str, dict[str, Any]] | None = None,
) -> None:
    ctx = run_ctx.base
    progress_key = ctx.memorize_lock_key(uid, soul_id)
    mem_lock = ctx.get_memorize_lock(progress_key)
    has_results = False
    has_memory_results = False
    pending_segment_ids: list[str] = []
    processed_end_cursor = processed_cursor
    soul_card_for_memorize: str | None = None
    cached_retrieval_ids: list[str] = []
    cached_prior_context_ids: list[str] = []
    conversation_rolling_summary: str | None = None
    consumed_background_summary_ids: set[str] = set()
    latest_display_ranges: dict[str, tuple[int, int]] = {}
    created_segment_paths: list[Path] = []
    reserved_manifest_ranges = [
        durable_range
        for _resource_url, _messages, _source_start, _source_end, durable_range in memorize_segments
        if durable_range is not None
    ]
    total_segments = 0
    had_existing_pending = False
    consolidation_started = False
    durable_segments_committed = False
    terminal_result: str = "success"
    rolling_summaries_raw = safe.get("_background_rolling_summaries")
    rolling_summaries: dict[str, dict[str, Any]] = (
        rolling_summaries_raw if isinstance(rolling_summaries_raw, dict) else {}
    )
    run_started_at = datetime.now(UTC).isoformat()
    try:
        # Phase 1: read initial state under lock.
        async with mem_lock:
            if conversation_id:
                state_row, soul_card_for_memorize, _ = run_ctx.load_turn_state_and_soul_card(
                    conversation_id,
                    user_id=uid,
                    soul_id=soul_id,
                )
                raw_ret_ids = state_row.get("retrieval_ids_since_consolidation")
                if isinstance(raw_ret_ids, list):
                    cached_retrieval_ids = [str(rid).strip() for rid in raw_ret_ids if str(rid).strip()]
                raw_pc_ids = state_row.get("prior_context_ids_since_consolidation")
                if isinstance(raw_pc_ids, list):
                    cached_prior_context_ids = [str(rid).strip() for rid in raw_pc_ids if str(rid).strip()]
                had_existing_pending = bool(run_ctx.normalize_text_list(state_row.get("pending_segment_ids")))
                conversation_rolling_summary = str(state_row.get("rolling_summary") or "").strip() or None

        # Phase 2: persist each memorize segment as a single file and feed one
        # synthetic segment payload per segment into batch memorize.
        segment_jobs: list[SegmentMemorizeJob] = []
        cancelled = False
        for _seg_idx, (
            _segment_resource_url,
            segment_messages,
            segment_start_index,
            segment_end_index,
            durable_range,
        ) in enumerate(memorize_segments):
            if progress_key in ctx.memorize_cancel:
                ctx.memorize_cancel.discard(progress_key)
                ctx.logger.info("memorize cancelled during segment prep after segment %d/%d", _seg_idx, len(memorize_segments))
                cancelled = True
                terminal_result = "cancelled"
                break
            segment_raw_text = json.dumps(segment_messages, ensure_ascii=False)
            first_ts = message_ts_ms(segment_messages[0]) if segment_messages and isinstance(segment_messages[0], dict) else None
            last_ts = message_ts_ms(segment_messages[-1]) if segment_messages and isinstance(segment_messages[-1], dict) else None
            d1 = date_label(first_ts, zi)
            d2 = date_label(last_ts, zi)
            fn = f"{d1}.json" if d1 == d2 else f"{d1}__{d2}.json"
            segment_path = (segments_dir / fn).resolve()
            n = 1
            while segment_path.exists():
                n += 1
                fn_base = f"{d1}" if d1 == d2 else f"{d1}__{d2}"
                segment_path = (segments_dir / f"{fn_base}_{n}.json").resolve()
            memory_producing = any(
                message.get("memorize_chat") is not False
                for message in segment_messages
                if isinstance(message, dict)
            )
            if memory_producing != (durable_range is not None):
                raise RuntimeError("memorize policy and durable segment range disagree")
            if memory_producing:
                segment_path.write_text(segment_raw_text, encoding="utf-8")
                created_segment_paths.append(segment_path)
            segment_id = (
                f"{conversation_id or 'cross'}:{durable_range[0]}-{durable_range[1]}"
                if durable_range is not None
                else None
            )
            segment_background_rows: list[dict[str, Any]] = []
            seen_background_sources: set[str] = set()
            if conversation_rolling_summary and conversation_id and not cross_memorize:
                segment_background_rows.append(
                    {
                        "summary": conversation_rolling_summary,
                        "source_label": "conversation-prehistory",
                        "source_conversation_id": conversation_id,
                        "rolled_up": True,
                    }
                )
                seen_background_sources.add(conversation_id)
            for idx, msg in enumerate(segment_messages):
                if not isinstance(msg, dict):
                    continue
                if bool(msg.get("memorize_chat", True)):
                    continue
                source_cid = str(msg.get("source_conversation_id") or "").strip()
                if not source_cid or source_cid in seen_background_sources:
                    continue
                seen_background_sources.add(source_cid)
                summary_row = rolling_summaries.get(source_cid)
                if not isinstance(summary_row, dict):
                    continue
                summary = str(summary_row.get("summary") or "").strip()
                if not summary:
                    continue
                source_label = str(summary_row.get("source_label") or msg.get("source_label") or "background").strip() or "background"
                segment_background_rows.append(
                    {
                        "summary": summary,
                        "source_label": source_label,
                        "source_conversation_id": source_cid,
                        "rolled_up": True,
                        "anchor_index": int(idx),
                    }
                )
                if memory_producing:
                    consumed_background_summary_ids.add(source_cid)
            segment_payload: dict[str, Any] = {
                "message_indices": list(range(len(segment_messages))),
                "segment_background_context_rows": segment_background_rows,
                "context_only": not memory_producing,
            }
            if segment_id is not None:
                segment_payload["segment_id"] = segment_id
            segment_jobs.append(
                {
                    "segment_payload": segment_payload,
                    "segment_messages": segment_messages,
                    "segment_resource_url": str(segment_path),
                    "segment_raw_text": segment_raw_text,
                    "segment_start_index": segment_start_index,
                    "segment_end_index": segment_end_index,
                    "segment_id": segment_id,
                    "memory_producing": memory_producing,
                }
            )

        total_segments = len(segment_jobs)
        _set_memorize_progress(
            ctx.memorize_progress,
            progress_key,
            active=True,
            phase="accepted",
            current=0,
            total=max(1, total_segments),
        )

        if not cancelled and segment_jobs:
            if progress_key in ctx.memorize_cancel:
                ctx.memorize_cancel.discard(progress_key)
                ctx.logger.info("memorize cancelled before batch extraction")
                terminal_result = "cancelled"
            else:
                _set_memorize_progress(
                    ctx.memorize_progress,
                    progress_key,
                    active=True,
                    phase="extracting",
                    current=0,
                    total=0,
                )
                ep_start = _time.monotonic()
                batch_results = await svc.memorize_segments_batch(
                    modality="conversation",
                    segments=[
                        {
                            "resource_url": job["segment_resource_url"],
                            "local_path": job["segment_resource_url"],
                            "raw_text": job["segment_raw_text"],
                            "segment": job["segment_payload"],
                        }
                        for job in segment_jobs
                    ],
                    user=scope,
                    soul_card=soul_card_for_memorize,
                    memory_retrieve_history=cached_retrieval_ids or None,
                    memory_prior_context=cached_prior_context_ids or None,
                    conversation_id=conversation_id,
                    on_extraction_progress=lambda current, total: _set_memorize_progress(
                        ctx.memorize_progress,
                        progress_key,
                        active=True,
                        phase="extracting",
                        current=current,
                        total=total,
                    ),
                )
                if len(batch_results) != len(segment_jobs):
                    error_message = (
                        f"memorize_segments_batch returned {len(batch_results)} results "
                        f"for {len(segment_jobs)} segment jobs"
                    )
                    raise RuntimeError(error_message)
                ctx.logger.info(
                    "memorize batch segments=%d elapsed=%.1fs",
                    total_segments,
                    _time.monotonic() - ep_start,
                )

                for seg_num, segment_job in enumerate(segment_jobs, 1):
                    _set_memorize_progress(
                        ctx.memorize_progress,
                        progress_key,
                        active=True,
                        phase="persist",
                        current=seg_num,
                        total=total_segments,
                    )
                    ep_result = batch_results[seg_num - 1] if seg_num - 1 < len(batch_results) else None
                    if isinstance(ep_result, dict):
                        has_results = True
                        result_pending = run_ctx.normalize_text_list(ep_result.get("pending_segment_ids"))
                        if not segment_job["memory_producing"] and result_pending:
                            raise RuntimeError("context-only memorize unit returned pending segment ids")
                        if segment_job["memory_producing"]:
                            has_memory_results = True
                            pending_segment_ids.extend(result_pending)
                        if cross_memorize and segment_job["memory_producing"]:
                            latest_display_ranges = _segment_display_ranges(segment_job["segment_messages"])
                    segment_end_index = segment_job["segment_end_index"]
                    if conversation_id and not cross_memorize:
                        # Re-acquire to write cursor; skip if a concurrent runner already advanced past us.
                        async with mem_lock:
                            fresh_row, _, _ = run_ctx.load_turn_state_and_soul_card(
                                conversation_id,
                                user_id=uid,
                                soul_id=soul_id,
                            )
                            memory_producing = segment_job["memory_producing"]
                            cursor_field = "digest_cursor" if memory_producing else "rolling_summary_cursor_id"
                            fresh_cursor = int(fresh_row.get(cursor_field) or 0)
                            if fresh_cursor <= segment_end_index:
                                processed_end_cursor = max(processed_end_cursor, segment_end_index)
                                # per-segment advance — crash recovery needs the cursor to move
                                # only after the whole segment completes.
                                updates = _cursor_updates_for_unit(
                                    memory_producing=memory_producing,
                                    cursor=processed_end_cursor,
                                    now_iso=datetime.now(UTC).isoformat(),
                                )
                                ctx.write_conversation_state(
                                    conversation_id,
                                    soul_id=soul_id,
                                    user_id=uid,
                                    updates=updates,
                                )
                            else:
                                # Another runner advanced the cursor past this segment; honour the further value.
                                processed_end_cursor = max(processed_end_cursor, fresh_cursor)

        # Phase 3: final state flush + bookkeeping under lock.
        async with mem_lock:
            if terminal_result == "cancelled":
                for segment_path in created_segment_paths:
                    segment_path.unlink(missing_ok=True)
                created_segment_paths.clear()
                if conversation_id:
                    _remove_manifest_ranges(segments_dir.parent / "manifest.json", reserved_manifest_ranges)
            if has_results and final_cursors:
                now_iso = datetime.now(UTC).isoformat()
                if cross_memorize:
                    run_ctx.clear_last_display_segments_for_nonparticipants(
                        user_id=uid,
                        soul_id=soul_id,
                        participant_conversation_ids=set(latest_display_ranges),
                        run_started_at=run_started_at,
                    )
                for fc_cid, checkpoint in final_cursors.items():
                    fc_cursor = max(0, int(checkpoint["cursor"]))
                    checkpoint_memory_producing = checkpoint.get("memory_producing")
                    if not isinstance(checkpoint_memory_producing, bool):
                        raise RuntimeError(f"memorize checkpoint missing captured policy for {fc_cid}")
                    source_id = str(checkpoint.get("source_message_id") or "").strip()
                    source_ts = checkpoint.get("ts")
                    web_source = bool(source_id and source_ts is not None)
                    fresh_row, _, _ = run_ctx.load_turn_state_and_soul_card(
                        fc_cid,
                        user_id=uid,
                        soul_id=soul_id,
                    )
                    if web_source:
                        payload_position = run_ctx.resolve_web_source_checkpoint(fc_cid, source_id)
                        if payload_position is None:
                            raise RuntimeError(
                                f"WhatsApp web_source checkpoint disappeared for {fc_cid}; "
                                "repair the conversation cursor before retrying"
                            )
                        fresh_id_field = (
                            "digest_cursor_source_message_id"
                            if checkpoint_memory_producing
                            else "rolling_summary_cursor_source_message_id"
                        )
                        fresh_cursor_field = (
                            "digest_cursor" if checkpoint_memory_producing else "rolling_summary_cursor_id"
                        )
                        fresh_source_id = str(fresh_row.get(fresh_id_field) or "").strip()
                        if fresh_source_id:
                            fresh_position = run_ctx.resolve_web_source_checkpoint(
                                fc_cid,
                                fresh_source_id,
                            )
                            if fresh_position is None:
                                fresh_position = -1 if checkpoint_memory_producing else 0
                        else:
                            fresh_position = int(fresh_row.get(fresh_cursor_field) or 0)
                        if payload_position <= fresh_position:
                            updates = _cursor_updates_for_unit(
                                memory_producing=checkpoint_memory_producing,
                                cursor=None,
                                now_iso=now_iso,
                                pending_segment_ids=(pending_segment_ids if fc_cid == conversation_id else None),
                            )
                            if updates:
                                ctx.write_conversation_state(
                                    fc_cid,
                                    soul_id=soul_id,
                                    user_id=uid,
                                    updates=updates,
                                )
                            continue
                        fc_cursor = payload_position
                    updates = _cursor_updates_for_unit(
                        memory_producing=checkpoint_memory_producing,
                        cursor=fc_cursor,
                        now_iso=now_iso,
                        source_message_id=source_id,
                        source_ts=source_ts,
                        pending_segment_ids=(pending_segment_ids if fc_cid == conversation_id else None),
                    )
                    display_range = latest_display_ranges.get(fc_cid)
                    if checkpoint_memory_producing and cross_memorize and display_range is not None:
                        updates["last_display_segment_start_index"] = display_range[0]
                        updates["last_display_segment_end_index"] = display_range[1]
                        updates["last_display_segment_at"] = now_iso
                    elif checkpoint_memory_producing and cross_memorize:
                        updates["last_display_segment_start_index"] = None
                        updates["last_display_segment_end_index"] = None
                        updates["last_display_segment_at"] = None
                    ctx.write_conversation_state(
                        fc_cid,
                        soul_id=soul_id,
                        user_id=uid,
                        updates=updates,
                    )
            elif conversation_id and has_results and pending_segment_ids:
                ctx.write_conversation_state(
                    conversation_id,
                    soul_id=soul_id,
                    user_id=uid,
                    updates=_cursor_updates_for_unit(
                        memory_producing=True,
                        cursor=None,
                        now_iso=datetime.now(UTC).isoformat(),
                        pending_segment_ids=pending_segment_ids,
                    ),
                )
            if conversation_id and has_memory_results:
                # Pending ids now reference these files; the failure path must not unlink them
                # or consolidation would reject the whole pending list as missing history.
                created_segment_paths.clear()
                durable_segments_committed = True
            if has_memory_results:
                for bg_cid in consumed_background_summary_ids:
                    ctx.write_conversation_state(
                        bg_cid,
                        soul_id=soul_id,
                        user_id=uid,
                        updates={
                            "rolling_summary": None,
                            "rolling_summary_updated_at": None,
                        },
                    )

            # Auto-trigger consolidation in background (releases memorize lock before LLM calls).
            should_consolidate = False
            if conversation_id and (has_memory_results or had_existing_pending):
                fresh_state, _, _ = run_ctx.load_turn_state_and_soul_card(
                    conversation_id,
                    user_id=uid,
                    soul_id=soul_id,
                )
                should_consolidate = ctx.consolidation_due(fresh_state)
            if should_consolidate:
                consolidation_started = True
                _set_memorize_progress(
                    ctx.memorize_progress,
                    progress_key,
                    active=True,
                    phase="consolidating",
                    current=1,
                    total=1,
                )
                _ct = asyncio.create_task(
                    run_ctx.run_consolidation_task(
                        svc,
                        conversation_id=conversation_id,
                        soul_id=soul_id,
                        uid=uid,
                        progress_key=progress_key,
                        memorize_progress=ctx.memorize_progress,
                    )
                )
                run_ctx.background_tasks_set.add(_ct)
                _ct.add_done_callback(run_ctx.background_tasks_set.discard)

            ctx.record_call(
                "memorize",
                safe,
                ok=True,
                info={
                    "resource_url": resource_url,
                    "conversationId": conversation_id,
                    "chatKey": chat_key,
                    "serverTimeZone": str(zi or ""),
                    "messages_in": merged_len,
                    "messages_merged": merged_len,
                    "force": force,
                    "memorizeSegmentCount": len(memorize_segments),
                    "memorizeSegmentPersistedCount": total_segments,
                    "minChunkTokens": ctx.min_chunk_tokens,
                    "memorizeDeferred": not force and not has_results,
                    "pendingSegmentRetryOnly": bool(had_existing_pending and not has_memory_results),
                    "sleepSplitMinLullSeconds": ctx.sleep_split_min_lull_seconds,
                    "sleepSplitStats": sleep_stats,
                },
            )
        if not consolidation_started:
            if terminal_result == "cancelled":
                _set_memorize_progress(
                    ctx.memorize_progress,
                    progress_key,
                    active=False,
                    last_result="cancelled",
                )
            elif not has_results and not had_existing_pending:
                _set_memorize_progress(
                    ctx.memorize_progress,
                    progress_key,
                    active=False,
                    last_result="nothing_to_memorize",
                )
            else:
                _set_memorize_progress(
                    ctx.memorize_progress,
                    progress_key,
                    active=False,
                    last_result="success",
                )
    except Exception as exc:
        for segment_path in created_segment_paths:
            try:
                segment_path.unlink(missing_ok=True)
            except OSError:
                ctx.logger.warning("failed to remove failed memorize segment file %s", segment_path)
        if conversation_id and not durable_segments_committed:
            try:
                async with mem_lock:
                    _remove_manifest_ranges(segments_dir.parent / "manifest.json", reserved_manifest_ranges)
            except (OSError, ValueError, json.JSONDecodeError):
                ctx.logger.exception("failed to release memorize manifest ranges")
        _set_memorize_progress(
            ctx.memorize_progress,
            progress_key,
            active=False,
            last_result="failure",
            error=f"{type(exc).__name__}: {exc}",
        )
        # pending_segment_ids stays untouched: any stored ids are from committed runs
        # (files intact) and clearing them would silently orphan those segments
        # from consolidation forever.
        raise
    finally:
        ctx.memorize_cancel.discard(progress_key)


def chat_storage_hash(uid: str, aid: str, key: str) -> str:
    raw = f"{uid}|{aid}|{key}".encode("utf-8", "ignore")
    return hashlib.sha1(raw).hexdigest()[:16]


def resolve_chat_storage_dir(
    chats_dir: Path,
    uid: str,
    aid: str,
    conversation_id: str | None,
    sanitize_db_filename: Callable[[str], str],
) -> tuple[Path, str, str]:
    agent_slug = sanitize_db_filename(aid)
    primary_value = str(conversation_id or "").strip()
    if conversation_id:
        primary_source = "conversation_id"
    else:
        primary_source = "empty"

    primary_key = chat_storage_hash(uid, aid, primary_value)
    primary_path = (chats_dir / f"{agent_slug}_{primary_key}").resolve()

    return primary_path, primary_key, primary_source


def _local_dt(ts_ms: int, zi: Any | None) -> datetime:
    dt_utc = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
    return dt_utc.astimezone(zi) if zi is not None else dt_utc


def server_timezone() -> Any:
    return datetime.now().astimezone().tzinfo or UTC


def date_label(ts_ms: int | None, zi: Any | None) -> str:
    if ts_ms is None:
        return "undated"
    try:
        return _local_dt(ts_ms, zi).date().isoformat()
    except (ValueError, OverflowError, OSError):
        return "undated"


def _merge_manifest_segments(
    existing: list[dict[str, Any]],
    memorize_segments: list[MemorizeSegment],
    *,
    rebuildable: bool = True,
) -> list[dict[str, Any]]:
    canonical: list[tuple[int, int]] = []
    non_rebuildable: list[dict[str, Any]] = []
    seen_non_rebuildable: set[tuple[int, int]] = set()

    def parsed_range(start: Any, end: Any) -> tuple[int, int] | None:
        try:
            st_i = int(start)
            en_i = int(end)
        except (TypeError, ValueError):
            return None
        if en_i < st_i:
            return None
        return st_i, en_i

    for segment in existing:
        if isinstance(segment, dict):
            parsed = parsed_range(segment.get("start"), segment.get("end"))
            if parsed is None:
                continue
            st_i, en_i = parsed
            if segment.get("rebuildable") is False:
                key = (st_i, en_i)
                if key not in seen_non_rebuildable:
                    seen_non_rebuildable.add(key)
                    non_rebuildable.append({"start": st_i, "end": en_i, "rebuildable": False})
            else:
                canonical.append((st_i, en_i))
    for _resource_url, _messages, _source_start, _source_end, durable_range in memorize_segments:
        if durable_range is None:
            continue
        parsed = parsed_range(*durable_range)
        if parsed is None:
            continue
        st_i, en_i = parsed
        if rebuildable:
            canonical.append((st_i, en_i))
        else:
            key = (st_i, en_i)
            if key not in seen_non_rebuildable:
                seen_non_rebuildable.add(key)
                non_rebuildable.append({"start": st_i, "end": en_i, "rebuildable": False})

    merged: list[dict[str, int]] = []
    for st_i, en_i in sorted(canonical):
        if merged and st_i <= int(merged[-1]["end"]) + 1:
            merged[-1]["end"] = max(int(merged[-1]["end"]), en_i)
        else:
            merged.append({"start": st_i, "end": en_i})
    non_rebuildable.sort(key=lambda row: (int(row["start"]), int(row["end"])))
    return [*merged, *non_rebuildable]


def _canonical_manifest_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        segment
        for segment in segments
        if isinstance(segment, dict) and segment.get("rebuildable") is not False
    ]


def _next_manifest_start(segments: list[dict[str, Any]]) -> int:
    max_end = -1
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        try:
            end = int(segment.get("end"))
        except (TypeError, ValueError):
            continue
        max_end = max(max_end, end)
    return max_end + 1


def _remove_manifest_ranges(path: Path, ranges: list[tuple[int, int]]) -> None:
    if not path.exists() or not ranges:
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    existing = manifest.get("segments") if isinstance(manifest.get("segments"), list) else []
    remaining: list[dict[str, Any]] = []
    for segment in existing:
        if not isinstance(segment, dict):
            continue
        try:
            pieces = [(int(segment["start"]), int(segment["end"]))]
        except (KeyError, TypeError, ValueError):
            continue
        for remove_start, remove_end in ranges:
            pieces = [
                piece
                for start, end in pieces
                for piece in (
                    ([(start, min(end, remove_start - 1))] if start < remove_start else [])
                    + ([(max(start, remove_end + 1), end)] if end > remove_end else [])
                )
            ]
        for start, end in pieces:
            remaining.append({**segment, "start": start, "end": end})
    manifest["segments"] = remaining
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _offset_memorize_segments(
    memorize_segments: list[tuple[str, list[dict[str, Any]], int, int]],
    *,
    start: int,
) -> list[MemorizeSegment]:
    out: list[MemorizeSegment] = []
    cursor = max(0, start)
    for resource_url, messages, source_start, source_end in memorize_segments:
        memory_producing = any(
            message.get("memorize_chat") is not False
            for message in messages
            if isinstance(message, dict)
        )
        durable_range = None
        if memory_producing:
            end = cursor + len(messages) - 1
            durable_range = (cursor, end)
            cursor = end + 1
        out.append((resource_url, messages, source_start, source_end, durable_range))
    return out


def split_indices_by_sleep(
    msgs: list[dict[str, Any]],
    zi: Any | None,
    min_lull_seconds: int,
) -> tuple[list[int], dict[str, Any]]:
    ts: list[int | None] = []
    for m in msgs:
        v = m.get("ts_ms")
        ts.append(int(v) if isinstance(v, int) else None)
    if sum(1 for x in ts if x is not None) < 2:
        return ([], {"timestamps_ok": False})

    best_gap_per_night: dict[Any, tuple[float, int]] = {}
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
            prev = best_gap_per_night.get(d)
            if prev is None or overlap > prev[0]:
                best_gap_per_night[d] = (overlap, i + 1)

    min_lull = float(min_lull_seconds)
    nights_total = len(best_gap_per_night)
    nights_qual = sum(
        1
        for (score, _idx) in best_gap_per_night.values()
        if isinstance(score, (int, float)) and score >= min_lull
    )

    raw_splits = sorted(
        {
            idx
            for (score, idx) in best_gap_per_night.values()
            if isinstance(idx, int) and 0 < idx < len(msgs) and isinstance(score, (int, float)) and score >= min_lull
        }
    )
    splits = list(raw_splits)
    return (
        splits,
        {
            "timestamps_ok": True,
            "nights_total": nights_total,
            "nights_qual": nights_qual,
            "min_lull_seconds": min_lull_seconds,
        },
    )


def select_sleep_splits_after_min_tokens(
    messages: list[dict[str, Any]],
    *,
    start_index: int,
    candidate_splits: list[int],
    min_chunk_tokens: int,
) -> list[int]:
    if min_chunk_tokens <= 0 or not candidate_splits:
        return [split_idx for split_idx in candidate_splits if split_idx > start_index]

    word_prefix: list[int] = [0]
    for message in messages:
        text = ""
        if isinstance(message, dict):
            text = str(message.get("content") or message.get("mes") or "")
        else:
            text = str(message or "")
        word_prefix.append(word_prefix[-1] + len(text.split()))

    gated_splits: list[int] = []
    chunk_start = max(0, start_index)
    for split_idx in candidate_splits:
        if split_idx <= chunk_start:
            continue
        words = max(0, word_prefix[split_idx] - word_prefix[chunk_start])
        token_estimate = int(words / 0.75)
        if token_estimate < min_chunk_tokens:
            continue
        gated_splits.append(split_idx)
        chunk_start = split_idx
    return gated_splits


def find_chat_dir_for_conversation(
    chats_dir: Path,
    uid: str,
    soul_id: str,
    conversation_id: str,
    sanitize_db_filename: Callable[[str], str],
) -> Path | None:
    primary_dir, _chat_key, _chat_key_source = resolve_chat_storage_dir(
        chats_dir,
        uid,
        soul_id,
        conversation_id,
        sanitize_db_filename,
    )
    if (primary_dir / "manifest.json").exists():
        return primary_dir

    agent_slug = sanitize_db_filename(soul_id)
    for manifest_path in sorted(chats_dir.glob(f"{agent_slug}_*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        source = manifest.get("source") if isinstance(manifest, dict) else {}
        if not isinstance(source, dict):
            continue
        source_conversation_id = str(
            source.get("conversation_id")
            or source.get("conversationId")
            or ""
        ).strip()
        if source_conversation_id == conversation_id:
            return manifest_path.parent
    return None


_MIN_HISTORY_MESSAGES_FOR_CONTINUITY = 8  # 4 turns × (user + soul)


def slice_history_after_last_memorized_segment(
    history: list[dict[str, Any]],
    *,
    chats_dir: Path,
    uid: str,
    soul_id: str,
    conversation_id: str,
    sanitize_db_filename: Callable[[str], str],
) -> list[dict[str, Any]]:
    if not isinstance(history, list) or not history:
        return history
    min_recent_start = max(0, len(history) - _MIN_HISTORY_MESSAGES_FOR_CONTINUITY)
    chat_dir = find_chat_dir_for_conversation(
        chats_dir,
        uid,
        soul_id,
        conversation_id,
        sanitize_db_filename,
    )
    if chat_dir is None:
        return history[min_recent_start:]
    manifest_path = (chat_dir / "manifest.json").resolve()
    if not manifest_path.exists():
        return history[min_recent_start:]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return history[min_recent_start:]
    segments = manifest.get("segments") if isinstance(manifest, dict) else None
    if not isinstance(segments, list) or not segments:
        return history[min_recent_start:]
    last = segments[-1]
    if not isinstance(last, dict):
        return history[min_recent_start:]
    try:
        tail_start = int(last.get("end", -1)) + 1
    except (TypeError, ValueError):
        return history[min_recent_start:]
    # Include the last memorized segment's successor + enough recent messages
    # for continuity, whichever reaches further back.
    tail_start = max(0, min(tail_start, min_recent_start))
    return history[tail_start:]


def unmemorized_sleep_gap_detected(
    history: list[dict[str, Any]],
    digest_cursor: Any,
    *,
    logger: Any,
    min_chunk_tokens: int,
    sleep_split_min_lull_seconds: int,
) -> bool:
    try:
        cursor = int(digest_cursor)
    except (TypeError, ValueError, OverflowError):
        cursor = -1
    start = max(0, cursor + 1)
    unproc = history[start:] if isinstance(history, list) else []
    if len(unproc) < 2:
        return False
    splits, _stats = split_indices_by_sleep(unproc, server_timezone(), sleep_split_min_lull_seconds)
    eligible_splits = select_sleep_splits_after_min_tokens(
        unproc,
        start_index=0,
        candidate_splits=splits,
        min_chunk_tokens=min_chunk_tokens,
    )
    return bool(eligible_splits)


async def memorize_endpoint(
    payload: dict[str, Any],
    background_tasks: BackgroundTasks,
    force: bool,
    tail: bool = False,
    rebuild: bool = False,
    *,
    endpoint_ctx: MemorizeEndpointContext,
) -> JSONResponse:
    """Memorize a SillyTavern conversation.

    force=True   — bypass sleep-split/min-token gating and process immediately.
    rebuild=True — archive the live DB, reset cursor, clear segments, then re-memorize
                   from scratch (implies force).

    Preferred: send the full memU payload (llm_profiles/database_config/etc) so per-step routing works.
    """
    if rebuild:
        force = True
    ctx = endpoint_ctx.base
    try:
        safe = endpoint_ctx.safe_payload(payload)
        is_cross = bool(safe.get("_cross_memorize"))

        scope = safe.get("user")
        if not isinstance(scope, dict):
            scope = endpoint_ctx.extract_scope(safe) or None
        conversation_id = canonical_conversation_id(endpoint_ctx.extract_conversation_id(safe))
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
        if not isinstance(conversation, list) or not conversation:
            raise HTTPException(status_code=400, detail="Missing or empty 'conversation' list")

        conv_norm = endpoint_ctx.normalize_conversation(conversation)

        # scope is validated dict with non-empty soul_id above; no need to re-guard.
        uid = str(scope.get("user_id") or "user")
        async with ctx.get_memorize_lock(ctx.memorize_lock_key(uid, soul_id)):
            if rebuild:
                db_path = endpoint_ctx.sqlite_current_path(uid, soul_id)
                if db_path is not None and db_path.exists():
                    ts = datetime.now(UTC).strftime("%y%m%d-%H%M%S")
                    archive_path = db_path.with_suffix(f".bak-{ts}")
                    db_path.rename(archive_path)
                    for wal_suffix in ("-wal", "-shm"):
                        wal_file = db_path.with_name(db_path.name + wal_suffix)
                        if wal_file.exists():
                            wal_file.rename(archive_path.with_name(archive_path.name + wal_suffix))
                    ctx.logger.info("re-memorize: archived %s → %s", db_path.name, archive_path.name)
                    endpoint_ctx.clear_cached_services()
            # Acquire (or re-acquire after archive) the service so schema creation runs against
            # the correct file. Must happen after clear_cached_services() in the rebuild path.
            svc = endpoint_ctx.get_service_from_payload(safe)
            storage_dir = endpoint_ctx.get_storage_dir(endpoint_ctx.get_config())
            chats_dir = (storage_dir / "st_chats").resolve()
            chat_dir, chat_key, chat_key_source = resolve_chat_storage_dir(
                chats_dir,
                uid,
                soul_id,
                conversation_id,
                endpoint_ctx.sanitize_db_filename,
            )
            segments_dir = (chat_dir / "segments").resolve()
            chat_dir.mkdir(parents=True, exist_ok=True)
            segments_dir.mkdir(parents=True, exist_ok=True)

            manifest_path = (chat_dir / "manifest.json").resolve()

            merged: list[dict[str, Any]] = conv_norm if isinstance(conv_norm, list) else []
            stamp_current_conversation_metadata(
                merged,
                conversation_id=conversation_id,
                chat_name=endpoint_ctx.pick_str(safe, "chat_name"),
            )

            processed_cursor = -1
            has_pending_segments = False
            if conversation_id:
                state_out, _db_path = ctx.write_conversation_state(
                    conversation_id,
                    soul_id=soul_id,
                    user_id=uid,
                    updates={},
                )
                processed_cursor = effective_digest_cursor_from_row(state_out)
                raw_pending_ids = state_out.get("pending_segment_ids")
                has_pending_segments = isinstance(raw_pending_ids, list) and any(
                    str(item).strip() for item in raw_pending_ids
                )

            zi = server_timezone()

            rawm = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else ""
            manifest: dict[str, Any] = json.loads(rawm) if rawm.strip() else {}
            raw_segments = manifest.get("segments") if isinstance(manifest.get("segments"), list) else []
            manifest_segments_existing: list[dict[str, Any]] = [
                segment for segment in raw_segments if isinstance(segment, dict)
            ]
            segments = _canonical_manifest_segments(manifest_segments_existing)

            resource_url = str(chat_dir)
            sleep_stats: Any | None = None
            new_segments: list[dict[str, Any]] = []
            if not tail and not is_cross and isinstance(merged, list) and any(isinstance(m.get("ts_ms"), int) for m in merged):
                tail_n = 2500
                if not segments:
                    rebuild_from = 0
                    keep_segments: list[dict[str, Any]] = []
                else:
                    # Rebuild manifest for the last ~2500 messages; keep earlier segments intact.
                    tail_start = max(0, len(merged) - tail_n)
                    rebuild_from = tail_start
                    keep_segments = []
                    for s in segments:
                        try:
                            st_i = int(s.get("start"))
                            en_i = int(s.get("end"))
                        except (TypeError, ValueError):
                            continue
                        if st_i <= tail_start <= en_i:
                            rebuild_from = st_i
                        if st_i < rebuild_from:
                            keep_segments.append(s)

                ctx_start = max(0, rebuild_from - 1)
                splits_rel, sleep_stats = split_indices_by_sleep(
                    merged[ctx_start:], zi, ctx.sleep_split_min_lull_seconds,
                )
                candidate_splits = [ctx_start + i for i in splits_rel if (ctx_start + i) > rebuild_from]
                splits = select_sleep_splits_after_min_tokens(
                    merged,
                    start_index=rebuild_from,
                    candidate_splits=candidate_splits,
                    min_chunk_tokens=ctx.min_chunk_tokens,
                )

                segment_start = max(0, rebuild_from)
                for split_idx in splits:
                    if split_idx <= segment_start:
                        continue
                    new_segments.append({"start": segment_start, "end": split_idx - 1})
                    segment_start = split_idx

                segments = keep_segments + new_segments

            raw_memorize_segments: list[tuple[str, list[dict[str, Any]], int, int]] = []
            if is_cross or tail:
                tail_start = 0 if is_cross else max(0, processed_cursor + 1)
                if tail_start < len(merged):
                    raw_memorize_segments = [(resource_url, merged[tail_start:], tail_start, len(merged) - 1)]
                if not raw_memorize_segments:
                    progress_key = ctx.memorize_lock_key(uid, soul_id)
                    if (
                        conversation_id
                        and has_pending_segments
                        and ctx.consolidation_due(state_out)
                    ):
                        _set_memorize_progress(
                            ctx.memorize_progress,
                            progress_key,
                            active=True,
                            phase="consolidating",
                            current=1,
                            total=1,
                        )
                        background_tasks.add_task(
                            endpoint_ctx.run_consolidation_task,
                            svc,
                            conversation_id=conversation_id,
                            soul_id=soul_id,
                            uid=uid,
                            progress_key=progress_key,
                            memorize_progress=ctx.memorize_progress,
                        )
                        return JSONResponse(
                            status_code=202,
                            content={
                                "ok": True,
                                "status": "accepted",
                                "conversation_id": conversation_id,
                                "segment_count": 0,
                                "pending_segment_retry": True,
                            },
                            background=background_tasks,
                        )
                    _set_memorize_progress(
                        ctx.memorize_progress,
                        progress_key,
                        active=False,
                        last_result="nothing_to_memorize",
                    )
                    return JSONResponse(
                        status_code=200,
                        content={"ok": True, "status": "nothing_to_memorize", "conversation_id": conversation_id},
                    )
            elif rebuild:
                processed_cursor = -1
                manifest_segments_existing = []
                if segments_dir and segments_dir.exists():
                    for old_seg in segments_dir.glob("*.json"):
                        old_seg.unlink(missing_ok=True)
            if not tail and not is_cross and segments and isinstance(merged, list):
                last_message_idx = len(merged) - 1
                for segment in segments:
                    try:
                        seg_start = int(segment.get("start"))
                        seg_end = int(segment.get("end"))
                    except (TypeError, ValueError):
                        continue
                    if seg_end < seg_start or seg_end > last_message_idx or seg_end <= processed_cursor:
                        continue
                    effective_start = max(seg_start, processed_cursor + 1)
                    if effective_start > seg_end:
                        continue
                    seg_messages = merged[effective_start : seg_end + 1]
                    if not seg_messages:
                        continue
                    raw_memorize_segments.append((resource_url, seg_messages, effective_start, seg_end))

            if force and not tail and not is_cross and not raw_memorize_segments:
                if isinstance(merged, list) and merged:
                    force_start = max(0, processed_cursor + 1)
                    if force_start < len(merged):
                        raw_memorize_segments = [
                            (resource_url, merged[chunk_start : chunk_end + 1], chunk_start, chunk_end)
                            for chunk_start, chunk_end in _chunk_index_ranges_by_token_budget(
                                merged,
                                start_idx=force_start,
                                end_idx=len(merged) - 1,
                                max_chunk_tokens=_FORCE_MEMORIZE_MAX_CHUNK_TOKENS,
                            )
                        ]
                if not raw_memorize_segments:
                    _set_memorize_progress(
                        ctx.memorize_progress,
                        ctx.memorize_lock_key(uid, soul_id),
                        active=False,
                        last_result="nothing_to_memorize",
                    )
                    return JSONResponse(
                        status_code=200,
                        content={"ok": True, "status": "nothing_to_memorize", "conversation_id": conversation_id},
                    )

            memorize_segments = _offset_memorize_segments(
                raw_memorize_segments,
                start=_next_manifest_start(manifest_segments_existing),
            )

            if conversation_id and memorize_segments:
                manifest_segments = _merge_manifest_segments(
                    manifest_segments_existing,
                    memorize_segments,
                    rebuildable=not is_cross,
                )
                manifest_out = {
                    "v": 1,
                    "tz": str(zi),
                    "segments": manifest_segments,
                    "split": {
                        "min_lull_seconds": ctx.sleep_split_min_lull_seconds,
                    },
                    "source": {
                        "conversation_id": conversation_id,
                        "conversationId": conversation_id,
                        "chatKey": chat_key,
                        "chatKeySource": chat_key_source or "",
                    },
                }
                manifest_path.write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8")

            expected_cursor = memorize_segments[-1][3] if memorize_segments else processed_cursor
            background_tasks.add_task(
                endpoint_ctx.run_memorize_segments,
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
                merged_len=len(merged) if isinstance(merged, list) else 0,
                force=force,
                sleep_stats=sleep_stats,
                segments_dir=segments_dir,
                zi=zi,
                cross_memorize=is_cross,
                final_cursors=safe.get("_final_cursors") if is_cross else None,
            )
            # background=background_tasks is REQUIRED: when an endpoint returns a
            # Response object directly, FastAPI does not auto-attach the tasks from
            # the injected BackgroundTasks parameter. Without this, add_task above
            # is silently a no-op and the segments never run.
            estimated_total_segments = max(1, len(memorize_segments))
            _set_memorize_progress(
                ctx.memorize_progress,
                ctx.memorize_lock_key(uid, soul_id),
                active=True,
                phase="accepted",
                current=0,
                total=estimated_total_segments,
            )
            return JSONResponse(
                status_code=202,
                content={
                    "ok": True,
                    "status": "accepted",
                    "conversation_id": conversation_id,
                    "expected_cursor": expected_cursor,
                    "segment_count": len(memorize_segments),
                    "progress_unit": "segment",
                    "resource_url": resource_url,
                },
                background=background_tasks,
            )
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        ctx.record_call(
            "memorize",
            payload,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise HTTPException(status_code=500, detail="Internal Server Error. Check server logs.") from exc


def memorize_progress_endpoint(
    user_id: str,
    soul_id: str,
    *,
    memorize_lock_key: Callable[[str, str], str],
    memorize_progress: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    key = memorize_lock_key(user_id, soul_id)
    progress = memorize_progress.get(key)
    if progress is None:
        return {"active": False}
    out = dict(progress)
    out["active"] = bool(out.get("active", False))
    return out


def memorize_cancel_endpoint(
    payload: dict[str, Any],
    *,
    memorize_lock_key: Callable[[str, str], str],
    memorize_progress: dict[str, dict[str, Any]],
    memorize_cancel: set[str],
) -> dict[str, Any]:
    uid = str(payload.get("user_id") or "").strip()
    sid = str(payload.get("soul_id") or "").strip()
    key = memorize_lock_key(uid, sid)
    row = memorize_progress.get(key) or {}
    if bool(row.get("active")):
        memorize_cancel.add(key)
        return {"ok": True, "status": "cancel_requested"}
    return {"ok": False, "status": "no_active_memorize"}
