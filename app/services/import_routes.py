"""Admission and bounded historical processing for the client-owned import source."""
from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from memu.app.dossier_revision import estimate_prompt_tokens
from app.services import consolidation, conversation_sources, memorize_endpoint
from app.services.consolidation import consolidation_input_budget
from app.services.state import effective_digest_cursor_from_row
from app.services.segment import _message_happened_at


class ImportMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    name: str = ""
    timestamp: str
    source_day: date
    position: int = Field(ge=0)


class ImportScope(BaseModel):
    user_id: str
    soul_id: str
    label: str


class ImportProcess(ImportScope):
    continuous: bool = False


class ImportPreview(ImportScope):
    conversation_id: str
    title: str | None = None
    history_end_index: int = Field(default=0, ge=0)
    current_messages: list[ImportMessage]


async def recover_memorize(runtime: Any, *, scoped: dict, cid: str, historical: bool = False) -> None:
    state, _card, _path = runtime._load_turn_state_and_soul_card(cid, **scoped)
    record = state["import_state"] if historical else runtime._paid_work_state(**scoped).get("memorize_failure")
    work = (record or {}).get("segment_work", {})
    if not work:
        return
    if not historical:
        runtime._write_conversation_state(cid, **scoped, updates={"memorize_failure": {**record, "paused": True}})
    marker = runtime._memorize_lock_key(**scoped)
    svc = runtime._get_service_from_payload({"user": scoped})
    for index, (segment_id, phase) in enumerate(work.items(), start=1):
        memorize_endpoint._set_memorize_progress(
            runtime._MEMORIZE_PROGRESS, marker, active=True, phase="memorizing", current=index, total=len(work),
        )
        def deduped(session, segment_id=segment_id):
            con = session.connection().connection.driver_connection
            previous_factory = con.row_factory
            try:
                con.row_factory = runtime.sqlite3.Row
                runtime._write_conversation_state(cid, **scoped, connection=con, updates={
                    "import_segment_work" if historical else "memorize_segment_work": {segment_id: "review"},
                })
            finally:
                con.row_factory = previous_factory
        await svc.resume_memorize_segment(
            segment_id=segment_id, phase=phase, user=scoped, on_dedupe_complete=deduped,
            enforce_input_budget=historical,
        )
    runtime._write_conversation_state(cid, **scoped, updates={
        "import_segment_work" if historical else "memorize_segment_work": {}, "finish_segment_work": True,
    })


async def run_waiting_memorize(runtime: Any, *, scoped: dict, import_cid: str | None = None, retry: bool = False) -> None:
    uid, sid = scoped["user_id"], scoped["soul_id"]
    marker = runtime._memorize_lock_key(uid, sid)
    success = False
    payload = None
    try:
        state = runtime._paid_work_state(uid, sid)
        failure = state.get("memorize_failure")
        cid = runtime._soul_import_state(uid, sid)[1]["ordinary_waiting"] if import_cid else failure["conversation_id"]
        if failure and retry:
            await recover_memorize(runtime, scoped=scoped, cid=cid)
            state = runtime._paid_work_state(uid, sid)
            failure = state.get("memorize_failure")
        if (import_cid and runtime._soul_state.consolidation_failure(state)) or (failure and not retry):
            return
        if failure and runtime._memorize_targets_complete(uid, sid, failure["targets"]):
            runtime._write_conversation_state(cid, **scoped, updates={"memorize_failure": None})
            await runtime._run_consolidation_task(
                runtime._get_service_from_payload({"user": scoped}), conversation_id=cid,
                soul_id=sid, uid=uid, progress_key=marker, memorize_progress=runtime._MEMORIZE_PROGRESS,
            )
            success = True
        else:
            payload = runtime._saved_memorize_payload(cid, uid, sid, failure)
            if payload is None and failure:
                raise RuntimeError("Failed Memorize still has unfinished source checkpoints; restore the source and Retry.")
            tasks = BackgroundTasks()
            if payload is not None:
                await runtime._memorize_owned(payload, tasks, True, admitted=True, batch_owned=True,
                                              import_handoff=True, retry=bool(failure))
            success = True
            for task in tasks.tasks:
                if not await task.func(*task.args, **task.kwargs):
                    success = False
                    break
        if success and import_cid:
            runtime._write_conversation_state(import_cid, **scoped, updates={"import_ordinary_waiting": None})
    except (Exception, asyncio.CancelledError) as exc:
        success = False
        runtime.logger.exception("Saved-chat Memorize did not complete for %s", sid)
        failure = runtime._paid_work_state(uid, sid).get("memorize_failure")
        runtime._write_conversation_state(cid, **scoped, updates={"memorize_failure": {
                **(failure or {}),
                "conversation_id": cid, "paused": True, "error": f"Memorize failed: {exc}"[:300],
                "targets": (failure or {}).get("targets") or (payload or {}).get("_final_cursors") or {},
        }})
        if isinstance(exc, asyncio.CancelledError):
            raise
    finally:
        if runtime._MEMORIZE_PROGRESS.get(marker, {}).get("phase") != "consolidating":
            memorize_endpoint._set_memorize_progress(
                runtime._MEMORIZE_PROGRESS, marker, active=False,
                last_result="success" if success else "failure",
            )
        await runtime._finish_memorize_claim(marker, success)


async def run_import_batch(runtime: Any, *, scoped: dict, chat: dict, retry: bool = False) -> bool:
    cid = chat["conversation_id"]
    uid, sid = scoped["user_id"], scoped["soul_id"]
    marker = runtime._memorize_lock_key(uid, sid)
    lock = runtime._get_memorize_lock(marker)
    phase, success = "memorize", False
    def record():
        state, _card, _path = runtime._load_turn_state_and_soul_card(cid, **scoped)
        return state["import_state"]
    try:
        if retry:
            await recover_memorize(runtime, scoped=scoped, cid=cid, historical=True)
        current = record()
        pending_first = bool(current["pending_segment_ids"]) and not (
            retry and current["error"] and current["stage"] == "memorize"
        )
        phase = "consolidation" if pending_first else "memorize"
        svc = runtime._get_service_from_payload({"user": scoped})
        deps = runtime._make_consolidation_deps()
        profile = runtime._resolve_profile_if_configured(svc, "consolidation")
        days: set[date] = set()
        while not pending_first:
            phase = "memorize"
            rows = conversation_sources.load_import_tail(
                **scoped, conversation_id=cid, since_cursor=current["memorize_cursor"],
                recent_fallback_messages=0, import_state=current, historical=True,
            )
            if not rows:
                runtime._write_conversation_state(cid, **scoped, updates={
                    "import_memorize_cursor": current["history_end_index"] - 1, "import_error": None,
                })
                current = record()
                break
            tasks = BackgroundTasks()
            await runtime._memorize_owned(
                {"user": scoped, "conversation_id": cid, "chat_name": chat["title"] or chat["label"],
                 "conversation": rows},
                tasks, True, admitted=True, historical=True, batch_owned=True,
            )
            for task in tasks.tasks:
                if not await task.func(*task.args, **task.kwargs):
                    raise RuntimeError("Import extraction did not complete; Retry required")
            previous_cursor = current["memorize_cursor"]
            current = record()
            if current["memorize_cursor"] <= previous_cursor:
                raise RuntimeError("Import extraction did not advance its source checkpoint")
            days.update(date.fromisoformat(row["source_day"]) for row in rows)
            phase = "consolidation"
            pairs = {(cid, segment_id) for segment_id in current["pending_segment_ids"]}
            async with lock:
                prep = consolidation.gather_consolidation_inputs(
                    deps, conversation_id=cid, **scoped, force=True, historical=True, selected_segments=pairs,
                )
            if prep.get("status") == "skip":
                raise RuntimeError("Extracted import segments could not be prepared for consolidation")
            prep["all_chat_history"] = runtime._format_all_chat_history_for_ai(
                current_history=prep["current_chat_messages"], cross_tail=[],
                conversation_id=cid, soul_id=sid, mark_current_chat=False,
            )
            days.update(date.fromisoformat(row["source_day"]) for row in prep["current_chat_messages"])
            estimates = consolidation._prepare_dossier_consolidation_prompts(
                svc, inputs=prep, **scoped,
            )[3]
            if (max(days) - min(days)).days >= runtime._consolidation_interval_days_from_cfg(runtime._CONFIG) or consolidation.consolidation_size_due(
                svc, estimates, len(prep["segment_inputs"]), profile,
            ):
                break
        if current["pending_segment_ids"]:
            phase = "consolidation"
            memorize_endpoint._set_memorize_progress(
                runtime._MEMORIZE_PROGRESS, marker, active=True, phase="consolidating", current=1, total=1,
            )
            result = await runtime._run_consolidation_pipeline_once(
                svc=svc, deps=deps, state_lock=lock, running=runtime._CONSOLIDATION_RUNNING,
                load_cross_tail_for_ai=runtime._load_cross_tail_for_ai,
                format_all_chat_history_for_ai=runtime._format_all_chat_history_for_ai,
                conversation_id=cid, **scoped, force=True, historical=True,
                selected_segments={(cid, segment_id) for segment_id in current["pending_segment_ids"]},
            )
            if result.get("status") != "ok":
                raise RuntimeError(f"Import consolidation did not complete: {result.get('reason', 'skipped')}")
        success = True
        memorize_endpoint._set_memorize_progress(runtime._MEMORIZE_PROGRESS, marker, active=False, last_result="success")
    except (Exception, asyncio.CancelledError) as exc:
        error = "Import interrupted. Retry required." if isinstance(exc, asyncio.CancelledError) else f"{type(exc).__name__}: {exc}"
        runtime.logger.exception("Import batch failed for %s", cid)
        memorize_endpoint._set_memorize_progress(
            runtime._MEMORIZE_PROGRESS, marker, active=False, last_result="failure", error=error,
        )
        try:
            runtime._write_conversation_state(cid, **scoped, updates={"import_stage": phase, "import_error": error[:300]})
        except Exception:
            runtime.logger.exception("Failed to record import error for %s", cid)
        if isinstance(exc, asyncio.CancelledError):
            raise
    return success


def import_task(runtime: Any, marker: str) -> asyncio.Task | None:
    return next((task for task in tuple(runtime._BACKGROUND_TASKS)
                 if not task.done() and task.get_name() == f"import:{marker}"), None)


def import_retry_running(runtime: Any, marker: str) -> bool:
    task = import_task(runtime, marker)
    return bool(task and task.import_retry)


async def run_import(runtime: Any, *, scoped: dict, chat: dict, retry: bool) -> None:
    marker = runtime._memorize_lock_key(**scoped)
    success = False
    try:
        while True:
            success = False
            success = await run_import_batch(runtime, scoped=scoped, chat=chat, retry=retry)
            if success:
                asyncio.current_task().import_retry = False
            retry = False
            current = runtime._load_turn_state_and_soul_card(chat["conversation_id"], **scoped)[0]["import_state"]
            if (not success or current["stage"] == "complete" or
                    not asyncio.current_task().import_continuous or runtime._SHUTDOWN_STATE["draining"] or
                    marker in runtime._MEMORIZE_CANCEL):
                break
    finally:
        runtime._MEMORIZE_CANCEL.discard(marker)
        await runtime._finish_memorize_claim(marker, success)
    if not success or runtime._SHUTDOWN_STATE["draining"]:
        return
    current = runtime._load_turn_state_and_soul_card(chat["conversation_id"], **scoped)[0]["import_state"]
    if current["stage"] == "complete" and current.get("ordinary_waiting"):
        with runtime._STATE_LOCK:
            runtime._FORCED_MEMORIZE_INFLIGHT[marker] = False
        await run_waiting_memorize(runtime, scoped=scoped, import_cid=chat["conversation_id"])


def register_import_routes(app: FastAPI, *, runtime: Any) -> None:
    def scope(request: ImportScope) -> tuple[dict, str]:
        scoped = runtime._extract_scope(request.model_dump())
        if not scoped.get("user_id") or not scoped.get("soul_id") or len(request.label.split()) != 1:
            raise HTTPException(status_code=400, detail="Owner, Soul and a one-word chat-app label are required")
        return scoped, request.label.strip()

    def registered(request: ImportScope):
        scoped, label = scope(request)
        chat = conversation_sources.import_chat_info(**scoped, label=label)
        if chat is None:
            raise HTTPException(status_code=404, detail="Imported chat not found")
        state, _card, _path = runtime._load_turn_state_and_soul_card(chat["conversation_id"], **scoped)
        if state.get("import_state") is None:
            raise HTTPException(status_code=409, detail="Register the imported chat first")
        return scoped, chat, state["import_state"]

    def start(request: ImportProcess, *, retry: bool):
        scoped, chat, record = registered(request)
        if record["stage"] == "complete" and not (retry and record["error"]):
            raise HTTPException(status_code=409, detail="No eligible history to process")
        marker = runtime._memorize_lock_key(**scoped)
        with runtime._STATE_LOCK:
            if runtime._paid_work_state(**scoped).get("memorize_failure"):
                raise HTTPException(status_code=409, detail="Retry Memorize before processing an import")
            if marker in runtime._FORCED_MEMORIZE_INFLIGHT or (scoped["user_id"], scoped["soul_id"]) in runtime._CONSOLIDATION_RUNNING:
                raise HTTPException(status_code=409, detail="Memory work is still running")
            if bool(record["error"]) != retry:
                raise HTTPException(status_code=409, detail="Retry the failed import" if record["error"] else "No failed import to retry")
            runtime._FORCED_MEMORIZE_INFLIGHT[marker] = True
        task = asyncio.create_task(run_import(runtime, scoped=scoped, chat=chat, retry=retry), name=f"import:{marker}")
        task.import_continuous = request.continuous
        task.import_retry = retry
        runtime._BACKGROUND_TASKS.add(task)
        task.add_done_callback(runtime._BACKGROUND_TASKS.discard)
        return JSONResponse(status_code=202, content={"status": "accepted", "conversation_id": chat["conversation_id"]})

    @app.post("/imports/process", operation_id="process_import")
    async def process(request: ImportProcess):
        return start(request, retry=False)

    @app.post("/imports/retry", operation_id="retry_import")
    async def retry(request: ImportProcess):
        return start(request, retry=True)

    @app.post("/imports/continuation", operation_id="import_continuation")
    async def continuation(request: ImportProcess):
        scoped, _chat, _record = registered(request)
        task = import_task(runtime, runtime._memorize_lock_key(**scoped))
        if task is None:
            raise HTTPException(status_code=409, detail="No import is running")
        task.import_continuous = request.continuous
        return {"continuous": task.import_continuous}

    @app.get("/imports/status", operation_id="import_status")
    def status(user_id: str, soul_id: str, label: str):
        scoped, chat, record = registered(ImportScope(user_id=user_id, soul_id=soul_id, label=label))
        marker = runtime._memorize_lock_key(**scoped)
        task = import_task(runtime, marker)
        return {"conversation_id": chat["conversation_id"], "import_state": record,
                "deferred_history": chat["history_end_index"] > record["history_end_index"],
                "running": task is not None, "continuous": bool(task and task.import_continuous),
                "progress": runtime._MEMORIZE_PROGRESS.get(marker, {"active": False}) if task else {"active": False}}

    @app.post("/imports/register", operation_id="register_import")
    def register(request: ImportScope):
        scoped, label = scope(request)
        chat = conversation_sources.import_chat_info(**scoped, label=label)
        if chat is None:
            raise HTTPException(status_code=404, detail="Store the imported chat before registering it")
        cid = chat["conversation_id"]
        runtime._get_service_from_payload({"user": scoped})
        path = runtime._sqlite_current_path(**scoped)
        runtime._sqlite_ensure_nonempty(path)
        con = runtime._sqlite_connect(path)
        try:
            con.row_factory = runtime.sqlite3.Row
            runtime._sqlite_ensure_conversation_state_schema(con)
            with runtime._STATE_LOCK:
                con.execute("BEGIN IMMEDIATE")
                state = runtime._conversation_state_from_row(
                    runtime._conversation_state_row(con, cid, **scoped))
                record = state.get("import_state") if state else None
                if record is None:
                    if runtime._soul_state.read(con).get("memorize_failure"):
                        raise HTTPException(status_code=409, detail="Retry Memorize before registering an import")
                    if runtime._memorize_lock_key(**scoped) in runtime._FORCED_MEMORIZE_INFLIGHT or (
                            scoped["user_id"], scoped["soul_id"]) in runtime._CONSOLIDATION_RUNNING:
                        raise HTTPException(status_code=409, detail="Memory work is still running; register when it finishes")
                    prior_segments = con.execute(
                        "SELECT 1 FROM resources WHERE user_id = ? AND soul_id = ? AND modality = 'conversation' LIMIT 1",
                        (scoped["user_id"], scoped["soul_id"]),
                    ).fetchone()
                    end = 0 if prior_segments else chat["initial_history_end_index"]
                    record = {"history_end_index": end, "memorize_cursor": -1, "pending_segment_ids": [],
                              "stage": "memorize" if end else "complete", "error": None}
                result, _ = runtime._write_conversation_state(
                    cid, **scoped, updates={"import_state": record}, connection=con)
                con.commit()
            return {"conversation_id": cid, "import_state": result["import_state"]}
        finally:
            con.close()

    @app.post("/imports/validate", operation_id="validate_import")
    def validate(request: ImportPreview):
        scoped, label = scope(request)
        if not request.conversation_id.startswith("import:dm:") or not request.conversation_id.removeprefix("import:dm:"):
            raise HTTPException(status_code=400, detail="An imported-chat identity is required")
        chat = conversation_sources.import_chat_info(**scoped, label=label)
        cid = chat["conversation_id"] if chat else request.conversation_id
        label = chat["label"] if chat else label
        if cid != request.conversation_id:
            raise HTTPException(status_code=409, detail="Use the existing chat for this app label")
        svc = runtime._get_service_from_payload({"user": scoped})
        state, card, path = runtime._load_turn_state_and_soul_card(cid, **scoped)
        cursor = effective_digest_cursor_from_row(state)
        stored = conversation_sources.load_import_tail(
            **scoped, conversation_id=cid, since_cursor=cursor, recent_fallback_messages=8,
            include_floor_without_new=True, import_state=state.get("import_state"),
        ) if chat else []
        stored_pending = [row for row in stored if row["source_conversation_index"] > cursor]
        processed = conversation_sources.import_processed_days(
            **scoped, label=label, cursor=cursor, import_state=state.get("import_state"))
        guidance = {"pending_start_day": None,
                    "processed_start_day": processed[0], "processed_end_day": processed[1],
                    "deferred_history": False}
        if path is not None and path.exists():
            con = runtime._sqlite_connect(path)
            try:
                con.row_factory = runtime.sqlite3.Row
                tails = runtime._load_cross_memorize_tails_from_sources(con, **scoped, strict=True)
                guidance["pending_start_day"] = min(
                    (day.date().isoformat() for cid, tail in tails.items() if not cid.startswith("activity:")
                     for row in tail if (day := _message_happened_at(row)) is not None), default=None)
                history_end = max(request.history_end_index, chat["history_end_index"] if chat else 0)
                record = state.get("import_state")
                guidance["deferred_history"] = history_end > record["history_end_index"] if record is not None else (
                    history_end > 0 and bool(con.execute(
                        "SELECT 1 FROM resources WHERE user_id = ? AND soul_id = ? AND modality = 'conversation' LIMIT 1",
                        (scoped["user_id"], scoped["soul_id"]),
                    ).fetchone()))
            finally:
                con.close()
        if not request.current_messages:
            return {"ok": True, "estimated_tokens": 0, "input_budget": None, **guidance}
        proposed = []
        for item in request.current_messages:
            try:
                if len(item.timestamp) == 10:
                    parsed = datetime.combine(date.fromisoformat(item.timestamp), datetime.min.time(), UTC)
                else:
                    parsed = datetime.fromisoformat(item.timestamp.replace("Z", "+00:00"))
                    if parsed.tzinfo is None:
                        raise ValueError("Missing timestamp timezone")
            except ValueError:
                raise HTTPException(status_code=422, detail="Imported messages require valid timestamps")
            proposed.append({**item.model_dump(exclude={"position", "source_day"}),
                "source_day": item.source_day.isoformat(), "ts_ms": int(parsed.timestamp() * 1000),
                "received_at": item.timestamp, "source_conversation_index": item.position,
                "conversation_id": cid, "source_conversation_id": cid, "app_label": label,
                "chat_name": (chat["title"] if chat else request.title) or label})
        pending = stored_pending + proposed
        floor = [row for row in stored if row["source_conversation_index"] <= cursor]
        needed = max(0, 8 - len(pending))
        displayed = runtime._load_cross_tail_for_ai(**scoped, conversation_id="")
        displayed_current = [row for row in displayed if row.get("conversation_id") == cid]
        floor_by_index = {row["source_conversation_index"]: row for row in [
            *(floor[-needed:] if needed else []),
            *(row for row in displayed_current if row["source_conversation_index"] <= cursor),
        ]}
        history = [floor_by_index[index] for index in sorted(floor_by_index)] + pending
        cross = [row for row in displayed if row.get("conversation_id") != cid]
        block = runtime._format_all_chat_history_for_ai(
            current_history=history, cross_tail=cross, conversation_id=cid, soul_id=scoped["soul_id"])
        system = runtime._make_turn_system_prompt(
            scoped["soul_id"], soul_card=card, response_sentences=int(runtime._CONFIG.get("turn_response_sentences", 3)))
        user = runtime._build_turn_prompt(
            user_message="", history=history, prior_context=state.get("prior_context"), retrieve_rag=None,
            all_categories_summary=svc.build_dossier_index(scoped), memory_cache=state.get("memory_cache"),
            intentions_active=state.get("intentions_active"), apimw_message_to_self=state.get("apimw_message_to_self"),
            conversations_block=block, conversation_id=cid,
            response_sentences=int(runtime._CONFIG.get("turn_response_sentences", 3)))
        try:
            allowance = consolidation_input_budget(svc, "default")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        estimated = estimate_prompt_tokens(system) + estimate_prompt_tokens(user)
        if estimated > allowance:
            raise HTTPException(status_code=400, detail="The current chat suffix exceeds the model context; move more messages to history")
        return {"ok": True, "estimated_tokens": estimated, "input_budget": allowance, **guidance}
