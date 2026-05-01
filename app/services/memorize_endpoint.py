import asyncio
import hashlib
import json
import math
import traceback
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
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
        except Exception:
            logger.exception("failed to record background error on conversation state")


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
    get_memorize_lock: Callable[[str], asyncio.Lock],
    memorize_lock_key: Callable[[str, str], str],
    write_conversation_state: Callable[..., tuple[dict[str, Any], Any]],
    load_turn_state_and_soul_card: Callable[..., tuple[dict[str, Any], str | None, str | None]],
    normalize_text_list: Callable[[Any], list[str]],
    compute_holistic_categories_summary: Callable[..., Awaitable[str | None]],
    run_consolidation_task: Callable[..., Awaitable[None]],
    record_call: Callable[..., None],
    logger: Any,
    memorize_progress: dict[str, dict[str, Any]],
    memorize_cancel: set[str],
    min_chunk_tokens: int,
    sleep_split_min_lull_seconds: int,
) -> None:
    progress_key = memorize_lock_key(uid, soul_id)
    mem_lock = get_memorize_lock(progress_key)
    has_results = False
    pending_episode_ids: list[str] = []
    processed_end_cursor = processed_cursor
    current_all_categories_summary: str | None = None
    soul_card_for_memorize: str | None = None
    cached_retrieval_ids: list[str] = []
    cached_prior_context_ids: list[str] = []
    total_episodes = 0
    try:
        # Phase 1: read initial state under lock.
        async with mem_lock:
            if conversation_id:
                state_row, soul_card_for_memorize, _ = load_turn_state_and_soul_card(
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

        # Phase 2: split segments into episodes, build flat episode list.
        EpisodeWork = tuple[dict[str, Any], str, str, str, int]  # (episode, url, seg_url, seg_raw_text, seg_end)
        all_episodes: list[EpisodeWork] = []
        cancelled = False
        for _seg_idx, (seg_url, seg_conv, seg_end) in enumerate(memorize_segments):
            if progress_key in memorize_cancel:
                memorize_cancel.discard(progress_key)
                logger.info("memorize cancelled during pre-split after segment %d/%d", _seg_idx, len(memorize_segments))
                cancelled = True
                break
            seg_raw_text = json.dumps(seg_conv, ensure_ascii=False)
            episodes = await svc.split_segment_into_episodes(
                local_path=seg_url,
                raw_text=seg_raw_text,
                modality="conversation",
            )
            if not episodes:
                episodes = [{"text": seg_raw_text, "caption": None}]
            for ep_idx, episode in enumerate(episodes):
                ep_url = seg_url if len(episodes) == 1 else f"{seg_url}#episode:{ep_idx + 1}"
                all_episodes.append((episode, ep_url, seg_url, seg_raw_text, seg_end))

        total_episodes = len(all_episodes)
        memorize_progress[progress_key] = {"current": 0, "total": total_episodes}
        import time as _time

        if not cancelled:
            for ep_num, (episode, ep_url, seg_url, seg_raw_text, seg_end) in enumerate(all_episodes, 1):
                if progress_key in memorize_cancel:
                    memorize_cancel.discard(progress_key)
                    logger.info("memorize cancelled after episode %d/%d", ep_num - 1, total_episodes)
                    break
                memorize_progress[progress_key] = {"current": ep_num, "total": total_episodes}
                ep_start = _time.monotonic()
                ep_result = await svc.memorize_episode(
                    resource_url=ep_url,
                    modality="conversation",
                    episode=episode,
                    user=scope,
                    raw_text=seg_raw_text,
                    local_path=seg_url,
                    all_categories_summary=current_all_categories_summary,
                    soul_card=soul_card_for_memorize,
                    memory_retrieve_history=cached_retrieval_ids or None,
                    memory_prior_context=cached_prior_context_ids or None,
                    conversation_id=conversation_id,
                )
                logger.info("memorize episode %d/%d elapsed=%.1fs", ep_num, total_episodes, _time.monotonic() - ep_start)
                if isinstance(ep_result, dict):
                    has_results = True
                    pending_episode_ids.extend(normalize_text_list(ep_result.get("pending_episode_ids")))
                next_seg_end = all_episodes[ep_num][4] if ep_num < total_episodes else None
                segment_done = (next_seg_end != seg_end)
                if segment_done and conversation_id:
                    # Re-acquire to write cursor; skip if a concurrent runner already advanced past us.
                    async with mem_lock:
                        fresh_row, _, _ = load_turn_state_and_soul_card(
                            conversation_id,
                            user_id=uid,
                            soul_id=soul_id,
                        )
                        fresh_cursor = int(fresh_row.get("digest_cursor") or 0)
                        if fresh_cursor <= seg_end:
                            processed_end_cursor = max(processed_end_cursor, seg_end)
                            # per-segment advance — crash recovery needs the cursor to move
                            # only after all episodes in a segment complete.
                            write_conversation_state(
                                conversation_id,
                                soul_id=soul_id,
                                user_id=uid,
                                updates={
                                    "digest_cursor": processed_end_cursor,
                                },
                            )
                        else:
                            # Another runner advanced the cursor past our seg_end; honour the further value.
                            processed_end_cursor = max(processed_end_cursor, fresh_cursor)

        # Phase 3: holistic summary LLM call — outside the lock.
        if conversation_id and has_results:
            current_all_categories_summary = await compute_holistic_categories_summary(
                svc=svc,
                soul_id=soul_id,
                user_id=uid,
            )

        # Phase 4: final state flush + bookkeeping under lock.
        async with mem_lock:
            if conversation_id and has_results:
                write_conversation_state(
                    conversation_id,
                    soul_id=soul_id,
                    user_id=uid,
                    updates={
                        # final flush once all segments commit — also writes the holistic summary atomically
                        "digest_cursor": max(0, processed_end_cursor),
                        "last_memorize_at": datetime.now(UTC).isoformat(),
                        "append_pending_episode_ids": pending_episode_ids,
                        "all_categories_summary": current_all_categories_summary,
                    },
                )

            # Auto-trigger consolidation in background (releases memorize lock before LLM calls).
            if conversation_id and has_results:
                asyncio.create_task(run_consolidation_task(svc, conversation_id=conversation_id, soul_id=soul_id, uid=uid))

            record_call(
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
                    "memorizeSegmentCount": len(memorize_segments),
                    "memorizeEpisodeCount": total_episodes,
                    "minChunkTokens": min_chunk_tokens,
                    "memorizeDeferred": not force and not has_results,
                    "days_written": days_written,
                    "sleepSplitMinLullSeconds": sleep_split_min_lull_seconds,
                    "sleepSplitStats": sleep_stats,
                },
            )
    finally:
        memorize_progress.pop(progress_key, None)
        memorize_cancel.discard(progress_key)


def read_list(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    raw = p.read_text(encoding="utf-8")
    obj = json.loads(raw) if raw.strip() else []
    return [m for m in obj if isinstance(m, dict)] if isinstance(obj, list) else []


def write_list_if_changed(p: Path, old: list[dict[str, Any]], new: list[dict[str, Any]]) -> None:
    if new == old:
        return
    p.write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")


def chat_storage_hash(uid: str, aid: str, key: str) -> str:
    raw = f"{uid}|{aid}|{key}".encode("utf-8", "ignore")
    return hashlib.sha1(raw).hexdigest()[:16]


def resolve_chat_storage_dir(
    chats_dir: Path,
    uid: str,
    aid: str,
    conversation_id: str | None,
    chat_file: str | None,
    resource_url_in: str | None,
    sanitize_db_filename: Callable[[str], str],
) -> tuple[Path, str, str]:
    agent_slug = sanitize_db_filename(aid)
    primary_value = str(conversation_id or resource_url_in or chat_file or "").strip()
    if conversation_id:
        primary_source = "conversation_id"
    elif resource_url_in:
        primary_source = "resource_url"
    elif chat_file:
        primary_source = "chat_file"
    else:
        primary_source = "empty"

    primary_key = chat_storage_hash(uid, aid, primary_value)
    primary_path = (chats_dir / f"{agent_slug}_{primary_key}").resolve()

    # Single legacy reuse path: if we are upgrading to conversation_id keying,
    # keep using the prior resource_url/chat_file keyed folder when it already exists.
    if conversation_id and not primary_path.exists():
        legacy_value = str(resource_url_in or chat_file or "").strip()
        if legacy_value:
            legacy_source = "resource_url" if resource_url_in else "chat_file"
            legacy_key = chat_storage_hash(uid, aid, legacy_value)
            legacy_path = (chats_dir / f"{agent_slug}_{legacy_key}").resolve()
            if legacy_path.exists():
                return legacy_path, legacy_key, legacy_source

    return primary_path, primary_key, primary_source


def _local_dt(ts_ms: int, zi: Any | None) -> datetime:
    dt_utc = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
    return dt_utc.astimezone(zi) if zi is not None else dt_utc


def date_label(ts_ms: int | None, zi: Any | None) -> str:
    if ts_ms is None:
        return "undated"
    try:
        return _local_dt(ts_ms, zi).date().isoformat()
    except Exception:
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

    raw_splits = sorted(
        {
            idx
            for (score, idx) in best.values()
            if isinstance(idx, int) and 0 < idx < len(msgs) and isinstance(score, (int, float)) and score >= min_lull
        }
    )
    max_episodes = 3
    splits = list(raw_splits)
    if len(splits) >= max_episodes:
        splits = splits[:max_episodes - 1]
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
        None,
        None,
        sanitize_db_filename,
    )
    if (primary_dir / "manifest.json").exists():
        return primary_dir

    agent_slug = sanitize_db_filename(soul_id)
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
    except Exception:
        return history[min_recent_start:]
    segments = manifest.get("segments") if isinstance(manifest, dict) else None
    if not isinstance(segments, list) or not segments:
        return history[min_recent_start:]
    last = segments[-1]
    if not isinstance(last, dict):
        return history[min_recent_start:]
    try:
        tail_start = int(last.get("end", -1)) + 1
    except Exception:
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
    return bool(splits)


async def memorize_endpoint(
    payload: dict[str, Any],
    background_tasks: BackgroundTasks,
    force: bool,
    *,
    safe_payload: Callable[[dict[str, Any]], dict[str, Any]],
    get_service_from_payload: Callable[[dict[str, Any]], Any],
    extract_scope: Callable[[dict[str, Any]], dict[str, Any] | None],
    extract_conversation_id: Callable[[dict[str, Any]], str | None],
    normalize_conversation: Callable[[Any], Any],
    pick_str: Callable[..., str | None],
    get_memorize_lock: Callable[[str], asyncio.Lock],
    memorize_lock_key: Callable[[str, str], str],
    sqlite_current_path: Callable[[str | None, str], Path | None],
    clear_cached_services: Callable[[], None],
    get_storage_dir: Callable[[dict[str, Any]], Path],
    write_conversation_state: Callable[..., tuple[dict[str, Any], Any]],
    run_memorize_episodes: Callable[..., Awaitable[None]],
    get_config: Callable[[], dict[str, Any]],
    sanitize_db_filename: Callable[[str], str],
    min_chunk_tokens: int,
    sleep_split_min_lull_seconds: int,
    memorize_progress: dict[str, dict[str, Any]],
    record_call: Callable[..., None],
    logger: Any,
) -> JSONResponse:
    """Memorize a SillyTavern conversation.

    Preferred: send the full memU payload (llm_profiles/database_config/etc) so per-step routing works.
    """
    try:
        safe = safe_payload(payload)
        svc = get_service_from_payload(safe)

        scope = safe.get("user")
        if not isinstance(scope, dict):
            scope = extract_scope(safe) or None
        conversation_id = extract_conversation_id(safe)
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

        conv_norm = normalize_conversation(conversation)

        # scope is validated dict with non-empty soul_id above; no need to re-guard.
        uid = str(scope.get("user_id") or "user")
        async with get_memorize_lock(memorize_lock_key(uid, soul_id)):
            if force:
                db_path = sqlite_current_path(uid, soul_id)
                if db_path is not None and db_path.exists():
                    ts = datetime.now(UTC).strftime("%y%m%d-%H%M%S")
                    archive_path = db_path.with_suffix(f".bak-{ts}")
                    db_path.rename(archive_path)
                    for wal_suffix in ("-wal", "-shm"):
                        wal_file = db_path.with_name(db_path.name + wal_suffix)
                        if wal_file.exists():
                            wal_file.rename(archive_path.with_name(archive_path.name + wal_suffix))
                    logger.info("re-memorize: archived %s → %s", db_path.name, archive_path.name)
                    clear_cached_services()
            storage_dir = get_storage_dir(get_config())
            chats_dir = (storage_dir / "st_chats").resolve()
            chat_file = pick_str(safe, "chat_file_name")
            resource_url_in = pick_str(safe, "resource_url")
            chat_dir, chat_key, chat_key_source = resolve_chat_storage_dir(
                chats_dir,
                uid,
                soul_id,
                conversation_id,
                chat_file,
                resource_url_in,
                sanitize_db_filename,
            )
            days_dir = (chat_dir / "days").resolve()
            chat_dir.mkdir(parents=True, exist_ok=True)
            days_dir.mkdir(parents=True, exist_ok=True)

            manifest_path = (chat_dir / "manifest.json").resolve()

            merged: list[dict[str, Any]] = conv_norm if isinstance(conv_norm, list) else []
            prev_len = 0

            processed_cursor = -1
            if conversation_id:
                state_out, _db_path = write_conversation_state(
                    conversation_id,
                    soul_id=soul_id,
                    user_id=uid,
                    updates={},
                )
                if state_out.get("last_memorize_at"):
                    processed_cursor = int(state_out.get("digest_cursor") or 0)

            tz_name = pick_str(safe, "time_zone")
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

            resource_url = str(chat_dir)
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
                splits_rel, sleep_stats = split_indices_by_sleep(
                    merged[ctx_start:], zi, tz_ok, sleep_split_min_lull_seconds
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
                    date = date_label(first_ts if isinstance(first_ts, int) else None, zi)
                    end_date = date_label(last_ts if isinstance(last_ts, int) else None, zi)
                    fn = f"{date}.json" if end_date == date else f"{date}__{end_date}.json"
                    fp = (days_dir / fn).resolve()
                    old_seg = read_list(fp)
                    write_list_if_changed(fp, old_seg, seg)
                    days_written += 1 if seg != old_seg else 0
                    new_segments.append({"date": date, "end_date": end_date, "start": a_i, "end": b_i - 1, "file": fn})

                segments = keep_segments + new_segments
                manifest_out = {
                    "v": 1,
                    "tz": str(tz_name or ""),
                    "segments": segments,
                    "split": {
                        "min_lull_seconds": sleep_split_min_lull_seconds,
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

            memorize_segments: list[tuple[str, list[dict[str, Any]], int]] = []
            if force:
                processed_cursor = -1
            if not segments and force and isinstance(merged, list) and merged:
                memorize_segments.append((resource_url, merged, len(merged) - 1))
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
                    if min_chunk_tokens > 0 and estimate_tokens(_bc) < min_chunk_tokens:
                        _carry = (_eff, _ei)
                        continue
                    _bf = _seg.get("file")
                    _bu = resource_url
                    if isinstance(_bf, str) and _bf:
                        _bu = str((days_dir / _bf).resolve())
                    memorize_segments.append((_bu, _bc, _ei))
                if _carry is not None:
                    _cf, _ce = _carry
                    _cc = merged[_cf : _ce + 1]
                    if _cc:
                        memorize_segments.append((resource_url, _cc, _ce))

            expected_cursor = memorize_segments[-1][2] if memorize_segments else processed_cursor
            background_tasks.add_task(
                run_memorize_episodes,
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
                merged_len=len(merged) if isinstance(merged, list) else 0,
                force=force,
                days_written=days_written,
                sleep_stats=sleep_stats if "sleep_stats" in locals() else None,
            )
            # background=background_tasks is REQUIRED: when an endpoint returns a
            # Response object directly, FastAPI does not auto-attach the tasks from
            # the injected BackgroundTasks parameter. Without this, add_task above
            # is silently a no-op and the segments never run.
            memorize_progress[memorize_lock_key(uid, soul_id)] = {"current": 0, "total": len(memorize_segments)}
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
        record_call(
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
