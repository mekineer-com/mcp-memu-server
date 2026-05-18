import asyncio
import hashlib
import json
import math
import time as _time
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, TypedDict

from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    words = sum(len(str(m.get("content") or m.get("mes") or "").split()) for m in messages)
    return int(words / 0.75)


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




class EpisodeMemorizeJob(TypedDict):
    episode_payload: dict[str, Any]
    episode_resource_url: str
    segment_raw_text: str
    segment_end_index: int


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


@dataclass(slots=True)
class MemorizeRunContext:
    base: MemorizeContext
    load_turn_state_and_soul_card: Callable[..., tuple[dict[str, Any], str | None, str | None]]
    normalize_text_list: Callable[[Any], list[str]]
    compute_holistic_categories_summary: Callable[..., Awaitable[str | None]]
    run_consolidation_task: Callable[..., Awaitable[None]]
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
    run_memorize_episodes: Callable[..., Awaitable[None]]
    run_consolidation_task: Callable[..., Awaitable[None]]
    get_config: Callable[[], dict[str, Any]]
    sanitize_db_filename: Callable[[str], str]


async def run_forced_memorize_from_turn(
    payload: dict[str, Any],
    *,
    memorize_handler: Callable[[dict[str, Any], BackgroundTasks, bool], Awaitable[Any]],
    logger: Any,
    get_memorize_lock: Callable[[str], asyncio.Lock],
    memorize_lock_key: Callable[[str, str], str],
    write_conversation_state: Callable[..., tuple[dict[str, Any], Any]],
) -> None:
    try:
        background_tasks = BackgroundTasks()
        await memorize_handler(payload, background_tasks, True)
        await background_tasks()
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


async def run_memorize_episodes(
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
    chat_key: str | None,
    tz_name: str | None,
    prev_len: int,
    merged_len: int,
    force: bool,
    sleep_stats: Any,
    run_ctx: MemorizeRunContext,
    episodes_dir: Path,
    zi: Any | None = None,
    cross_memorize: bool = False,
    final_cursors: dict[str, int] | None = None,
) -> None:
    ctx = run_ctx.base
    progress_key = ctx.memorize_lock_key(uid, soul_id)
    mem_lock = ctx.get_memorize_lock(progress_key)
    has_results = False
    pending_episode_ids: list[str] = []
    processed_end_cursor = processed_cursor
    current_all_categories_summary: str | None = None
    soul_card_for_memorize: str | None = None
    cached_retrieval_ids: list[str] = []
    cached_prior_context_ids: list[str] = []
    total_episodes = 0
    had_existing_pending = False
    try:
        # Phase 1: read initial state under lock.
        async with mem_lock:
            if conversation_id:
                state_row, soul_card_for_memorize, _ = run_ctx.load_turn_state_and_soul_card(
                    conversation_id,
                    user_id=uid,
                    soul_id=soul_id,
                )
                current_all_categories_summary = str(state_row.get("all_categories_summary") or "").strip() or None
                raw_ret_ids = state_row.get("retrieval_ids_since_consolidation")
                if isinstance(raw_ret_ids, list):
                    cached_retrieval_ids = [str(rid).strip() for rid in raw_ret_ids if str(rid).strip()]
                raw_pc_ids = state_row.get("prior_context_ids_since_consolidation")
                if isinstance(raw_pc_ids, list):
                    cached_prior_context_ids = [str(rid).strip() for rid in raw_pc_ids if str(rid).strip()]
                had_existing_pending = bool(run_ctx.normalize_text_list(state_row.get("pending_episode_ids")))

        # Phase 2: split segments into episodes, write each episode as its own file.
        episode_jobs: list[EpisodeMemorizeJob] = []
        cancelled = False
        for _seg_idx, (segment_resource_url, segment_messages, segment_end_index) in enumerate(memorize_segments):
            if progress_key in ctx.memorize_cancel:
                ctx.memorize_cancel.discard(progress_key)
                ctx.logger.info("memorize cancelled during pre-split after segment %d/%d", _seg_idx, len(memorize_segments))
                cancelled = True
                break
            segment_raw_text = json.dumps(segment_messages, ensure_ascii=False)
            if cross_memorize:
                segment_episodes = await svc.split_cross_conversation_into_episodes(
                    raw_text=segment_raw_text,
                )
            else:
                segment_episodes = await svc.split_segment_into_episodes(
                    local_path=segment_resource_url,
                    raw_text=segment_raw_text,
                    modality="conversation",
                )
            if not segment_episodes:
                msg = (
                    f"preprocessor returned no episodes for segment "
                    f"{_seg_idx + 1}/{len(memorize_segments)}"
                )
                raise RuntimeError(msg)
            for episode_payload in segment_episodes:
                episode_indices = episode_payload.get("message_indices")
                if isinstance(episode_indices, list) and episode_indices:
                    episode_messages = [
                        segment_messages[i]
                        for i in episode_indices
                        if 0 <= i < len(segment_messages)
                    ]
                else:
                    episode_messages = segment_messages
                first_ts = (
                    episode_messages[0].get("ts_ms")
                    if episode_messages and isinstance(episode_messages[0], dict)
                    else None
                )
                last_ts = (
                    episode_messages[-1].get("ts_ms")
                    if episode_messages and isinstance(episode_messages[-1], dict)
                    else None
                )
                d1 = date_label(first_ts if isinstance(first_ts, int) else None, zi)
                d2 = date_label(last_ts if isinstance(last_ts, int) else None, zi)
                fn = f"{d1}.json" if d1 == d2 else f"{d1}__{d2}.json"
                episode_path = (episodes_dir / fn).resolve()
                n = 1
                while episode_path.exists():
                    n += 1
                    fn_base = f"{d1}" if d1 == d2 else f"{d1}__{d2}"
                    episode_path = (episodes_dir / f"{fn_base}_{n}.json").resolve()
                episode_path.write_text(json.dumps(episode_messages, ensure_ascii=False), encoding="utf-8")
                episode_jobs.append(
                    {
                        "episode_payload": episode_payload,
                        "episode_resource_url": str(episode_path),
                        "segment_raw_text": segment_raw_text,
                        "segment_end_index": segment_end_index,
                    }
                )

        total_episodes = len(episode_jobs)
        ctx.memorize_progress[progress_key] = {"current": 0, "total": total_episodes}

        if not cancelled and episode_jobs:
            if progress_key in ctx.memorize_cancel:
                ctx.memorize_cancel.discard(progress_key)
                ctx.logger.info("memorize cancelled before batch extraction")
            else:
                ctx.memorize_progress[progress_key] = {"current": 0, "total": total_episodes, "phase": "extracting"}
                ep_start = _time.monotonic()
                batch_results = await svc.memorize_episodes_batch(
                    modality="conversation",
                    episodes=[
                        {
                            "resource_url": job["episode_resource_url"],
                            "local_path": job["episode_resource_url"],
                            "raw_text": job["segment_raw_text"],
                            "episode": job["episode_payload"],
                        }
                        for job in episode_jobs
                    ],
                    user=scope,
                    all_categories_summary=current_all_categories_summary,
                    soul_card=soul_card_for_memorize,
                    memory_retrieve_history=cached_retrieval_ids or None,
                    memory_prior_context=cached_prior_context_ids or None,
                    conversation_id=conversation_id,
                )
                if len(batch_results) != len(episode_jobs):
                    msg = (
                        f"memorize_episodes_batch returned {len(batch_results)} results "
                        f"for {len(episode_jobs)} episode jobs"
                    )
                    raise RuntimeError(msg)
                ctx.logger.info(
                    "memorize batch episodes=%d elapsed=%.1fs",
                    total_episodes,
                    _time.monotonic() - ep_start,
                )

                for ep_num, episode_job in enumerate(episode_jobs, 1):
                    if progress_key in ctx.memorize_cancel:
                        ctx.memorize_cancel.discard(progress_key)
                        ctx.logger.info("memorize cancelled after batch result %d/%d", ep_num - 1, total_episodes)
                        break
                    ctx.memorize_progress[progress_key] = {"current": ep_num, "total": total_episodes, "phase": "persist"}
                    ep_result = batch_results[ep_num - 1] if ep_num - 1 < len(batch_results) else None
                    if isinstance(ep_result, dict):
                        has_results = True
                        pending_episode_ids.extend(run_ctx.normalize_text_list(ep_result.get("pending_episode_ids")))
                    next_segment_end_index = (
                        episode_jobs[ep_num]["segment_end_index"] if ep_num < total_episodes else None
                    )
                    segment_end_index = episode_job["segment_end_index"]
                    last_episode_of_segment = (next_segment_end_index != segment_end_index)
                    if last_episode_of_segment and conversation_id and not cross_memorize:
                        # Re-acquire to write cursor; skip if a concurrent runner already advanced past us.
                        async with mem_lock:
                            fresh_row, _, _ = run_ctx.load_turn_state_and_soul_card(
                                conversation_id,
                                user_id=uid,
                                soul_id=soul_id,
                            )
                            fresh_cursor = int(fresh_row.get("digest_cursor") or 0)
                            if fresh_cursor <= segment_end_index:
                                processed_end_cursor = max(processed_end_cursor, segment_end_index)
                                # per-segment advance — crash recovery needs the cursor to move
                                # only after all episodes in a segment complete.
                                ctx.write_conversation_state(
                                    conversation_id,
                                    soul_id=soul_id,
                                    user_id=uid,
                                    updates={
                                        "digest_cursor": processed_end_cursor,
                                    },
                                )
                            else:
                                # Another runner advanced the cursor past this segment; honour the further value.
                                processed_end_cursor = max(processed_end_cursor, fresh_cursor)

        # Phase 3: holistic summary LLM call — outside the lock.
        if conversation_id and has_results:
            current_all_categories_summary = await run_ctx.compute_holistic_categories_summary(
                svc=svc,
                soul_id=soul_id,
                user_id=uid,
            )

        # Phase 4: final state flush + bookkeeping under lock.
        async with mem_lock:
            if has_results and final_cursors:
                now_iso = datetime.now(UTC).isoformat()
                for fc_cid, fc_cursor in final_cursors.items():
                    updates: dict[str, Any] = {
                        "digest_cursor": max(0, fc_cursor),
                        "last_memorize_at": now_iso,
                    }
                    if fc_cid == conversation_id:
                        updates["append_pending_episode_ids"] = pending_episode_ids
                        updates["all_categories_summary"] = current_all_categories_summary
                    ctx.write_conversation_state(
                        fc_cid,
                        soul_id=soul_id,
                        user_id=uid,
                        updates=updates,
                    )
            elif conversation_id and has_results:
                ctx.write_conversation_state(
                    conversation_id,
                    soul_id=soul_id,
                    user_id=uid,
                    updates={
                        "digest_cursor": max(0, processed_end_cursor),
                        "last_memorize_at": datetime.now(UTC).isoformat(),
                        "append_pending_episode_ids": pending_episode_ids,
                        "all_categories_summary": current_all_categories_summary,
                    },
                )

            # Auto-trigger consolidation in background (releases memorize lock before LLM calls).
            if conversation_id and (has_results or had_existing_pending):
                _ct = asyncio.create_task(
                    run_ctx.run_consolidation_task(svc, conversation_id=conversation_id, soul_id=soul_id, uid=uid)
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
                    "timeZone": tz_name,
                    "messages_prev": prev_len,
                    "messages_in": merged_len,
                    "messages_merged": merged_len,
                    "force": force,
                    "memorizeSegmentCount": len(memorize_segments),
                    "memorizeEpisodeCount": total_episodes,
                    "minChunkTokens": ctx.min_chunk_tokens,
                    "memorizeDeferred": not force and not has_results,
                    "pendingEpisodeRetryOnly": bool(had_existing_pending and not has_results),
                    "sleepSplitMinLullSeconds": ctx.sleep_split_min_lull_seconds,
                    "sleepSplitStats": sleep_stats,
                },
            )
    finally:
        ctx.memorize_progress.pop(progress_key, None)
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


def date_label(ts_ms: int | None, zi: Any | None) -> str:
    if ts_ms is None:
        return "undated"
    try:
        return _local_dt(ts_ms, zi).date().isoformat()
    except (ValueError, OverflowError, OSError):
        return "undated"


def split_indices_by_sleep(
    msgs: list[dict[str, Any]],
    zi: Any | None,
    tz_ok: bool,
    min_lull_seconds: int,
) -> tuple[list[int], dict[str, Any]]:
    if not tz_ok:
        return ([], {"tz_ok": False})

    ts: list[int | None] = []
    for m in msgs:
        v = m.get("ts_ms")
        ts.append(int(v) if isinstance(v, int) else None)
    if sum(1 for x in ts if x is not None) < 2:
        return ([], {"tz_ok": True, "timestamps_ok": False})

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
            "tz_ok": True,
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


def resolve_turn_timezone(safe: dict[str, Any], logger: Any) -> Any | None:
    tz_name = str(safe.get("time_zone") or safe.get("timeZone") or "").strip() or None
    raw_off = safe.get("time_zone_offset_min")
    if raw_off is None:
        raw_off = safe.get("timeZoneOffsetMin")
    tz_off_min: int | None = None
    if isinstance(raw_off, (int, float)) and math.isfinite(raw_off):
        tz_off_min = int(raw_off)
    if tz_name and ZoneInfo is not None:
        try:
            return ZoneInfo(str(tz_name))
        except (KeyError, OSError, ValueError):
            logger.debug("invalid time zone name: %s", tz_name, exc_info=True)
    if tz_off_min is not None:
        try:
            return timezone(timedelta(minutes=tz_off_min))
        except (TypeError, ValueError, OverflowError):
            logger.debug("invalid time zone offset minutes: %s", tz_off_min, exc_info=True)
    return None


def unmemorized_sleep_gap_detected(
    history: list[dict[str, Any]],
    digest_cursor: Any,
    safe: dict[str, Any],
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
    zi = resolve_turn_timezone(safe, logger)
    if zi is None:
        return False
    splits, _stats = split_indices_by_sleep(unproc, zi, True, sleep_split_min_lull_seconds)
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
    *,
    endpoint_ctx: MemorizeEndpointContext,
) -> JSONResponse:
    """Memorize a SillyTavern conversation.

    Preferred: send the full memU payload (llm_profiles/database_config/etc) so per-step routing works.
    """
    ctx = endpoint_ctx.base
    try:
        safe = endpoint_ctx.safe_payload(payload)
        is_cross = bool(safe.get("_cross_memorize"))
        svc = endpoint_ctx.get_service_from_payload(safe)

        scope = safe.get("user")
        if not isinstance(scope, dict):
            scope = endpoint_ctx.extract_scope(safe) or None
        conversation_id = endpoint_ctx.extract_conversation_id(safe)
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
            if force:
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
            storage_dir = endpoint_ctx.get_storage_dir(endpoint_ctx.get_config())
            chats_dir = (storage_dir / "st_chats").resolve()
            chat_dir, chat_key, chat_key_source = resolve_chat_storage_dir(
                chats_dir,
                uid,
                soul_id,
                conversation_id,
                endpoint_ctx.sanitize_db_filename,
            )
            episodes_dir = (chat_dir / "episodes").resolve()
            chat_dir.mkdir(parents=True, exist_ok=True)
            episodes_dir.mkdir(parents=True, exist_ok=True)

            manifest_path = (chat_dir / "manifest.json").resolve()

            merged: list[dict[str, Any]] = conv_norm if isinstance(conv_norm, list) else []
            prev_len = 0

            processed_cursor = -1
            has_pending_episodes = False
            if conversation_id:
                state_out, _db_path = ctx.write_conversation_state(
                    conversation_id,
                    soul_id=soul_id,
                    user_id=uid,
                    updates={},
                )
                if state_out.get("last_memorize_at"):
                    processed_cursor = int(state_out.get("digest_cursor") or 0)
                raw_pending_ids = state_out.get("pending_episode_ids")
                has_pending_episodes = isinstance(raw_pending_ids, list) and any(
                    str(item).strip() for item in raw_pending_ids
                )

            tz_name = endpoint_ctx.pick_str(safe, "time_zone")
            tz_off_raw = safe.get("time_zone_offset_min")
            tz_off_min = int(tz_off_raw) if isinstance(tz_off_raw, (int, float)) and math.isfinite(tz_off_raw) else None

            tz_ok = False
            zi = None
            if tz_name and ZoneInfo is not None:
                try:
                    zi = ZoneInfo(str(tz_name))
                    tz_ok = True
                except (KeyError, ValueError):
                    zi = None
                    tz_ok = False
            if not tz_ok and tz_off_min is not None:
                try:
                    zi = timezone(timedelta(minutes=tz_off_min))
                    tz_ok = True
                    if not tz_name:
                        tz_name = f"offset({tz_off_min})"
                except (ValueError, OverflowError):
                    zi = None
                    tz_ok = False

            rawm = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else ""
            manifest: dict[str, Any] = json.loads(rawm) if rawm.strip() else {}
            segments: list[dict[str, Any]] = (
                manifest.get("segments") if isinstance(manifest.get("segments"), list) else []
            )

            resource_url = str(chat_dir)
            sleep_stats: Any | None = None
            new_segments: list[dict[str, Any]] = []
            if not tail and not is_cross and tz_ok and isinstance(merged, list) and any(isinstance(m.get("ts_ms"), int) for m in merged):
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
                    merged[ctx_start:], zi, tz_ok, ctx.sleep_split_min_lull_seconds,
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
                manifest_out = {
                    "v": 1,
                    "tz": str(tz_name or ""),
                    "segments": segments,
                    "split": {
                        "min_lull_seconds": ctx.sleep_split_min_lull_seconds,
                    },
                    "source": {
                        "conversation_id": conversation_id or "",
                        "conversationId": conversation_id or "",
                        "timeZoneOffsetMin": tz_off_min if tz_off_min is not None else None,
                        "chatKey": chat_key,
                        "chatKeySource": chat_key_source or "",
                    },
                }
                manifest_path.write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8")

            memorize_segments: list[tuple[str, list[dict[str, Any]], int]] = []
            if is_cross or tail:
                tail_start = 0 if is_cross else max(0, processed_cursor + 1)
                if tail_start < len(merged):
                    memorize_segments = [(resource_url, merged[tail_start:], len(merged) - 1)]
                if not memorize_segments:
                    if conversation_id and has_pending_episodes:
                        background_tasks.add_task(
                            endpoint_ctx.run_consolidation_task,
                            svc,
                            conversation_id=conversation_id,
                            soul_id=soul_id,
                            uid=uid,
                        )
                        return JSONResponse(
                            status_code=202,
                            content={
                                "ok": True,
                                "status": "accepted",
                                "conversation_id": conversation_id,
                                "segment_count": 0,
                                "pending_episode_retry": True,
                            },
                            background=background_tasks,
                        )
                    return JSONResponse(
                        status_code=200,
                        content={"ok": True, "status": "nothing_to_memorize", "conversation_id": conversation_id},
                    )
            elif force:
                processed_cursor = -1
                if episodes_dir and episodes_dir.exists():
                    for old_ep in episodes_dir.glob("*.json"):
                        old_ep.unlink(missing_ok=True)
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
                    memorize_segments.append((resource_url, seg_messages, seg_end))

            expected_cursor = memorize_segments[-1][2] if memorize_segments else processed_cursor
            background_tasks.add_task(
                endpoint_ctx.run_memorize_episodes,
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
                merged_len=len(merged) if isinstance(merged, list) else 0,
                force=force,
                sleep_stats=sleep_stats,
                episodes_dir=episodes_dir,
                zi=zi,
                cross_memorize=is_cross,
                final_cursors=safe.get("_final_cursors") if is_cross else None,
            )
            # background=background_tasks is REQUIRED: when an endpoint returns a
            # Response object directly, FastAPI does not auto-attach the tasks from
            # the injected BackgroundTasks parameter. Without this, add_task above
            # is silently a no-op and the segments never run.
            episodes_cap_raw = getattr(getattr(svc, "memorize_config", None), "episodes_per_segment", 3)
            try:
                episodes_cap = int(episodes_cap_raw)
            except (TypeError, ValueError):
                episodes_cap = 3
            episodes_cap = max(1, episodes_cap)
            estimated_total_episodes = max(1, len(memorize_segments) * episodes_cap)
            ctx.memorize_progress[ctx.memorize_lock_key(uid, soul_id)] = {
                "current": 0,
                "total": estimated_total_episodes,
            }
            return JSONResponse(
                status_code=202,
                content={
                    "ok": True,
                    "status": "accepted",
                    "conversation_id": conversation_id,
                    "expected_cursor": expected_cursor,
                    "segment_count": len(memorize_segments),
                    "progress_unit": "episode",
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
    return {"active": True, **progress}


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
    if key in memorize_progress:
        memorize_cancel.add(key)
        return {"ok": True, "status": "cancel_requested"}
    return {"ok": False, "status": "no_active_memorize"}
