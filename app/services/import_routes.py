"""Unpaid admission for the client-owned imported-chat source."""
from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from memu.app.dossier_revision import estimate_prompt_tokens
from app.services import conversation_sources
from app.services.consolidation import consolidation_input_budget
from app.services.state import effective_digest_cursor_from_row


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


class ImportPreview(ImportScope):
    conversation_id: str
    title: str | None = None
    current_messages: list[ImportMessage]


def register_import_routes(app: FastAPI, *, runtime: Any) -> None:
    def scope(request: ImportScope) -> tuple[dict, str]:
        scoped = runtime._extract_scope(request.model_dump())
        if not scoped.get("user_id") or not scoped.get("soul_id") or len(request.label.split()) != 1:
            raise HTTPException(status_code=400, detail="Owner, Soul and a one-word chat-app label are required")
        return scoped, request.label.strip()

    @app.post("/imports/register", operation_id="register_import")
    def register(request: ImportScope):
        scoped, label = scope(request)
        chat = conversation_sources.import_chat_info(**scoped, label=label)
        if chat is None:
            raise HTTPException(status_code=404, detail="Store the imported chat before registering it")
        cid = chat["conversation_id"]
        path = runtime._sqlite_current_path(**scoped)
        runtime._sqlite_ensure_nonempty(path)
        con = runtime._sqlite_connect(path)
        try:
            con.row_factory = runtime.sqlite3.Row
            runtime._sqlite_ensure_conversation_state_schema(con)
            con.execute("BEGIN IMMEDIATE")
            state = runtime._conversation_state_from_row(
                runtime._conversation_state_row(con, cid, **scoped))
            record = state.get("import_state") if state else None
            if record is None:
                end = chat["history_end_index"]
                record = {"history_end_index": end, "memorize_cursor": -1, "pending_segment_ids": [],
                          "stage": "memorize" if end else "complete", "error": None}
            elif chat["history_end_index"] > record["history_end_index"]:
                record = {**record, "history_end_index": chat["history_end_index"],
                          "stage": "memorize" if record["stage"] == "complete" else record["stage"]}
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
        if cid != request.conversation_id:
            raise HTTPException(status_code=409, detail="Use the existing chat for this app label")
        if not request.current_messages:
            return {"ok": True, "estimated_tokens": 0, "input_budget": None}
        state, card, _ = runtime._load_turn_state_and_soul_card(cid, **scoped)
        cursor = effective_digest_cursor_from_row(state)
        stored = conversation_sources.load_import_tail(
            **scoped, conversation_id=cid, since_cursor=cursor, recent_fallback_messages=8,
            include_floor_without_new=True, import_state=state.get("import_state"),
        ) if chat else []
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
        pending = [row for row in stored if row["source_conversation_index"] > cursor] + proposed
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
        svc = runtime._get_service_from_payload({"user": scoped})
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
        return {"ok": True, "estimated_tokens": estimated, "input_budget": allowance}
