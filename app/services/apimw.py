"""APImw background pipeline helpers."""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from typing import Any

_main: Any = None


def _m() -> Any:
    if _main is None:
        raise RuntimeError("apimw is not bound to app.main")
    return _main


def _mark_apimw_inflight(conversation_id: str) -> bool:
    return _m()._mark_inflight(_m()._APIMW_INFLIGHT, conversation_id)


def _clear_apimw_inflight(conversation_id: str) -> None:
    _m()._clear_inflight(_m()._APIMW_INFLIGHT, conversation_id)


def _parse_apimw_json_response(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        if start < 0:
            return None
        try:
            parsed, _end = json.JSONDecoder().raw_decode(raw[start:])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


async def _apimw_retrieve_items(
    payload: dict[str, Any],
    *,
    focus_text: str,
    conversations_block: str,
    soul_id: str,
    history: list[dict[str, Any]],
    state_row: dict[str, Any],
    conversation_id: str,
    apimw_k: int,
    trace_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    retrieve_queries = _m()._build_retrieve_soul_context_queries(
        soul_id=soul_id,
        message=focus_text,
        history=history,
        state_row=state_row,
        identity_mode="apimw",
        conversation_id=conversation_id,
        conversations_block=conversations_block,
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
        "trace_id": trace_id,
    }
    _m().logger.info("apimw retrieve for %s", conversation_id)
    retrieve_out = await _m()._run_retrieve(retrieve_payload, conversation_id=conversation_id)
    retrieve_result_data = retrieve_out.get("result") or {}
    retrieved_items = [item for item in (retrieve_result_data.get("items") or []) if isinstance(item, dict)]
    _m().logger.info("apimw retrieved %d items for %s", len(retrieved_items), conversation_id)
    return retrieve_result_data, retrieved_items


async def _apimw_collect_memory_items(
    svc: Any,
    payload: dict[str, Any],
    *,
    focus_text: str,
    conversations_block: str,
    history: list[dict[str, Any]],
    state_row: dict[str, Any],
    conversation_id: str,
    soul_id: str,
    apimw_k: int,
    apimw_random_count: int,
    scope: dict[str, str],
    trace_id: str,
) -> list[dict[str, Any]]:
    _retrieve_result, retrieved_items = await _m()._apimw_retrieve_items(
        payload,
        focus_text=focus_text,
        conversations_block=conversations_block,
        soul_id=soul_id,
        history=history,
        state_row=state_row,
        conversation_id=conversation_id,
        apimw_k=apimw_k,
        trace_id=trace_id,
    )

    combined_items: list[dict[str, Any]] = []
    seen_item_sigs: set[str] = set()
    for item in retrieved_items:
        sig = _m()._item_sig(item)
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
            sig = _m()._item_sig(row)
            if not sig or sig in seen_item_sigs:
                continue
            candidates.append(row)
        if candidates:
            sample_size = min(apimw_random_count, len(candidates))
            for candidate_item in random.sample(candidates, sample_size):
                sig = _m()._item_sig(candidate_item)
                if not sig or sig in seen_item_sigs:
                    continue
                seen_item_sigs.add(sig)
                combined_items.append(candidate_item)

    _m().logger.info("apimw combined pool %d items for %s", len(combined_items), conversation_id)
    return combined_items


async def _apimw_synthesize(
    svc: Any,
    *,
    combined_items: list[dict[str, Any]],
    state_row: dict[str, Any],
    segment_text: str,
    current_message_text: str,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    scope: dict[str, str],
    llm_profile: str | None = None,
    trace_id: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]], dict[str, str]]:
    _m().logger.info("apimw synthesis for %s", conversation_id)
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
        formatted_memory_lines.append(_m()._format_memory_line(item, show_id=True, item_id=str(counter)))
        counter += 1
        shaped_by = item.get("shaped_by")
        if isinstance(shaped_by, dict):
            formatted_memory_lines.append(_m()._format_shaped_by_line(shaped_by))

    legend = _m()._format_memory_legend({str(item.get("memory_type") or "") for item in combined_items if isinstance(item, dict)})
    if legend and formatted_memory_lines:
        formatted_memory_lines.insert(0, legend)
    formatted_memories = "\n".join(formatted_memory_lines) if formatted_memory_lines else "(none)"

    memory_cache = _m()._normalize_memory_cache_impl(state_row.get("memory_cache"))
    intentions_active = _m()._normalize_intentions_stack_impl(state_row.get("intentions_active"))
    context_block = _m()._build_turn_context_block(
        history=[],
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=state_row.get("all_categories_summary"),
        memory_cache=memory_cache,
        intentions_active=intentions_active,
        conversations_block=segment_text,
        current_user_text=current_message_text,
    )

    apimw_system_prompt = (
        f"Today is {_m()._format_time_anchor()}.\n\n"
        "You are your soul's subconscious: a background process that runs between your turns. "
        "You have just searched your long-term memory, and now you'll want to guide yourself by surfacing the memories most relevant to the conversation. "
        "When you respond, you'll have the same context: your summary, activities, conversations, working thoughts and intentions. "
        "You won't have the memories from the \"Memories List\", so surface what matters.\n\n"
        "In addition, you can send an optional message_to_self with a valuable higher-order insight like:\n"
        "- conceptual understanding\n"
        "- strategic foresight\n"
        "- penetrating discernment\n\n"
        "The message_to_self should NOT have anything obvious:\n"
        "- Do not restate anything from the conversation, or anything that's an easy conclusion from the conversation.\n"
        "- Do not duplicate anything from a memory you will surface to yourself.\n\n"
        "Return STRICT JSON (first character { , last character } , no markdown fences, no text outside JSON).\n\n"
        "Required top-level keys:\n"
        "- prior_context: array of memory IDs (strings) you want surfaced as background context "
        "next time you speak. Pick what matters — not everything. Order by importance.\n"
        "- message_to_self: string or null — a brief thought you want to surface to your conscious self. "
        "Use this when something you noticed in the background feels important enough to bring to your own attention next time you speak. One sentence."
    )

    apimw_user_prompt = "\n\n".join(
        part for part in (context_block, f"Memories List:\n{formatted_memories}")
        if str(part or "").strip()
    )

    llm_raw = await svc.chat(
        apimw_user_prompt,
        profile=llm_profile,
        system_prompt=apimw_system_prompt,
        response_format={"type": "json_object"},
        op="apimw",
        step="synthesis",
        trace_id=trace_id,
    )

    apimw_response_text = str(llm_raw or "").strip()
    result_json = _m()._parse_apimw_json_response(apimw_response_text)
    if result_json is None:
        _m().logger.error("apimw synthesis: JSON parse failed, raw=%s", apimw_response_text[:200])
        return None, items_by_id, id_map

    _m().logger.info("apimw synthesis: parsed JSON with keys %s for %s", list(result_json.keys()), conversation_id)
    return result_json, items_by_id, id_map


async def _apimw_persist(
    svc: Any,
    *,
    result_json: dict[str, Any],
    items_by_id: dict[str, dict[str, Any]],
    id_map: dict[str, str],
    scope: dict[str, str],
    conversation_id: str,
    user_id: str,
    soul_id: str,
    expected_prior_context: str = "",
) -> None:
    async with _m()._retrieve_scope_lock(user_id, soul_id):
        updates: dict[str, Any] = {}
        resolved_prior_context_ids: list[str] = []

        fresh_row, _, _ = _m()._load_turn_state_and_soul_card(conversation_id, user_id=user_id, soul_id=soul_id)
        existing_prior = str(fresh_row.get("prior_context") or "").strip()
        if existing_prior != str(expected_prior_context or "").strip():
            _m().logger.warning("apimw: stale prior_context for %s; skipping APImw persist", conversation_id)
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
                prior_context_lines.append(_m()._format_memory_line(item))
                shaped_by = item.get("shaped_by")
                if isinstance(shaped_by, dict):
                    prior_context_lines.append(_m()._format_shaped_by_line(shaped_by))
            if prior_context_lines:
                new_prior = "\n".join(prior_context_lines)
                updates["prior_context"] = new_prior

        message_to_self = str(result_json.get("message_to_self") or "").strip()
        if message_to_self:
            updates["apimw_message_to_self"] = f"[subconscious] {message_to_self}"
            try:
                sc_embedding = (await svc.embed([message_to_self], profile="embedding"))[0]
                svc.database.memory_item_repo.create_item(
                    resource_id=None,
                    memory_type="subconscious",
                    source_role="soul",
                    summary=message_to_self,
                    embedding=sc_embedding,
                    extra={"apimw_message_to_self": True},
                    user_data={"user_id": user_id, "soul_id": soul_id, "conversation_id": conversation_id},
                    conversation_id=conversation_id,
                )
            except Exception:
                _m().logger.warning("failed to persist subconscious memory item", exc_info=True)

        prior_context_ids = (
            list(dict.fromkeys(resolved_prior_context_ids))
            if resolved_prior_context_ids
            else []
        )
        if prior_context_ids:
            updates["append_prior_context_ids_since_consolidation"] = prior_context_ids

        if updates:
            _m()._write_conversation_state(
                conversation_id,
                soul_id=soul_id,
                user_id=user_id,
                updates=updates,
            )
            _m().logger.info("apimw state written for %s (keys: %s)", conversation_id, list(updates.keys()))


async def _run_apimw(
    payload: dict[str, Any],
    *,
    conversation_id: str,
    soul_id: str,
    user_id: str,
    state_row: dict[str, Any],
    current_history: list[dict[str, Any]],
) -> None:
    try:
        svc = _m()._get_service_from_payload(payload)
        apimw_trace_id = uuid.uuid4().hex
        scope = {"user_id": user_id, "soul_id": soul_id}
        apimw_item_top_k = _m()._apimw_memory_count_from_cfg(_m()._CONFIG)
        apimw_random_count = _m()._apimw_random_count_from_cfg(_m()._CONFIG)

        cross_tail = _m()._load_cross_tail_for_ai(
            user_id=user_id,
            soul_id=soul_id,
            conversation_id=conversation_id,
        )
        current_message_text = _m()._pick_str(payload, "message", "query") or ""
        segment_text = _m()._format_all_chat_history_for_ai(
            current_history=current_history,
            cross_tail=cross_tail,
            conversation_id=conversation_id,
            soul_id=soul_id,
            chat_label=_m()._chat_label_for_prompt(payload),
            current_user_text=current_message_text,
        )
        focus_text = current_message_text or segment_text.strip()
        if not focus_text and not segment_text.strip():
            _m().logger.info("apimw skipped for %s: no recent conversation text", conversation_id)
            return

        combined_items = await _m()._apimw_collect_memory_items(
            svc,
            payload,
            focus_text=focus_text,
            conversations_block=segment_text,
            history=current_history,
            state_row=state_row,
            conversation_id=conversation_id,
            soul_id=soul_id,
            apimw_k=apimw_item_top_k,
            apimw_random_count=apimw_random_count,
            scope=scope,
            trace_id=apimw_trace_id,
        )

        apimw_heavy_profile = _m()._resolve_profile_if_configured(svc, "memory_extract")
        result_json, items_by_id, apimw_id_map = await _m()._apimw_synthesize(
            svc,
            combined_items=combined_items,
            state_row=state_row,
            segment_text=segment_text,
            current_message_text=current_message_text,
            user_id=user_id,
            soul_id=soul_id,
            conversation_id=conversation_id,
            scope=scope,
            llm_profile=apimw_heavy_profile,
            trace_id=apimw_trace_id,
        )
        if result_json is None:
            try:
                _m()._set_background_error(
                    conversation_id,
                    soul_id=soul_id,
                    user_id=user_id,
                    code="apimw_synthesis_parse_failed",
                    detail="synthesis response was not valid JSON object",
                )
                # prior_context stays: it is durable until a successful turn
                # consumes it, so a transient synthesis failure must not erase
                # the context the next turn should still see.
            except Exception:
                _m().logger.exception("failed to record APImw synthesis-parse failure for %s", conversation_id)
            return

        await _m()._apimw_persist(
            svc,
            result_json=result_json,
            items_by_id=items_by_id,
            id_map=apimw_id_map,
            scope=scope,
            conversation_id=conversation_id,
            user_id=user_id,
            soul_id=soul_id,
            expected_prior_context=str(state_row.get("prior_context") or ""),
        )
        try:
            _m()._clear_background_error_if_apimw_owned(conversation_id, soul_id=soul_id, user_id=user_id)
        except Exception:
            _m().logger.exception("failed to clear APImw background error state for %s", conversation_id)

    except Exception as exc:
        try:
            _m()._set_background_error(
                conversation_id,
                soul_id=soul_id,
                user_id=user_id,
                code="apimw_failed",
                detail=f"{type(exc).__name__}: {str(exc)[:220]}",
            )
            # prior_context stays on failure too (see parse-failure branch).
        except Exception:
            _m().logger.exception("failed to record APImw failure state for %s", conversation_id)
        _m().logger.exception("APImw background pipeline failed for %s", conversation_id)

def _apimw_cadence_due(user_id: str, soul_id: str, cadence_threshold: int) -> bool:
    db_path = _m()._sqlite_current_path(user_id, soul_id)
    if db_path is None:
        return False
    _m()._sqlite_ensure_nonempty(db_path)
    con = _m()._sqlite_connect(db_path)
    try:
        con.execute(
            "CREATE TABLE IF NOT EXISTS apimw_cadence "
            "(id INTEGER PRIMARY KEY CHECK (id = 1), turn_count INTEGER NOT NULL DEFAULT 0)"
        )
        con.execute("INSERT OR IGNORE INTO apimw_cadence (id, turn_count) VALUES (1, 0)")
        con.execute("UPDATE apimw_cadence SET turn_count = turn_count + 1 WHERE id = 1")
        turn_count = int(con.execute("SELECT turn_count FROM apimw_cadence WHERE id = 1").fetchone()[0] or 0)
        con.commit()
    finally:
        con.close()
    return (turn_count % cadence_threshold) == 0


def _turn_launch_apimw(
    cid: str,
    uid: str,
    soul_id: str,
    safe: dict[str, Any],
    history_full: list[dict[str, Any]],
) -> str:
    cadence_threshold = _m()._apimw_cadence_from_cfg(_m()._CONFIG)
    if not _m()._apimw_cadence_due(uid, soul_id, cadence_threshold):
        return "skipped_cadence"
    if not _m()._mark_apimw_inflight(cid):
        return "skipped_inflight"
    apimw_task: asyncio.Task | None = None
    try:
        apimw_state_row, _, _apimw_db_path = _m()._load_turn_state_and_soul_card(
            cid,
            user_id=uid,
            soul_id=soul_id,
        )
        history_for_ai = _m()._turn_history_with_floor(history_full, apimw_state_row)
        apimw_task = asyncio.create_task(
            _m()._run_apimw(
                safe,
                conversation_id=cid,
                soul_id=soul_id,
                user_id=uid,
                state_row=apimw_state_row,
                current_history=history_for_ai,
            )
        )
        _m()._BACKGROUND_TASKS.add(apimw_task)

        def _on_apimw_done(task: asyncio.Task) -> None:
            try:
                task.result()
            except Exception:
                _m().logger.exception("APImw background pipeline failed for %s", cid)
            finally:
                _m()._BACKGROUND_TASKS.discard(task)
                _m()._clear_apimw_inflight(cid)

        apimw_task.add_done_callback(_on_apimw_done)
    except Exception:
        if apimw_task is None:
            _m()._clear_apimw_inflight(cid)
        _m().logger.exception("APImw background pipeline failed to start for %s", cid)
        return "failed_to_start"
    return "started"
