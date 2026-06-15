from __future__ import annotations

from collections.abc import Callable
import sqlite3
import time
from typing import Any

from fastapi import HTTPException

from app.services.intention_state import (
    format_intentions_for_prompt as _format_intentions_for_prompt,
    normalize_intentions_stack as _normalize_intentions_stack_impl,
    normalize_memory_cache as _normalize_memory_cache_impl,
)
from app.services.payload import _canonicalize_scope_where, _extract_scope
from app.services.turn_contract import (
    build_conversations_block,
    format_time_anchor as _format_time_anchor,
    format_working_thoughts_lines as _format_working_thoughts_lines,
)


def _extract_force_retrieve(payload: dict[str, Any]) -> bool:
    value = payload.get("force_retrieve")
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    raise HTTPException(status_code=400, detail="'force_retrieve' must be a boolean")


def _extract_trace_id(payload: dict[str, Any]) -> str | None:
    value = payload.get("trace_id")
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="'trace_id' must be a string")
    trace_id = value.strip()
    return trace_id or None


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


def _build_retrieve_identity_context(
    soul_name: str,
    *,
    apimw: bool = False,
) -> str:
    name = str(soul_name or "").strip() or "the soul"
    anchor = f"Today is {_format_time_anchor()}."
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
            "In a continuous cycle you respond, search your long-term memory, and act autonomously. "
            "The first-person voice that follows is yours."
        )
    return f"{anchor}\n{identity}"


def _build_retrieve_soul_context_queries(
    *,
    soul_id: str,
    message: str,
    history: list[dict[str, Any]],
    state_row: dict[str, Any],
    identity_mode: str = "retrieve",
    conversation_id: str | None = None,
    chat_label: str | None = None,
    conversations_block: str | None = None,
    self_turn_directive: str | None = None,
    self_turn_label: str | None = None,
) -> list[dict[str, Any]]:
    memory_cache = _normalize_memory_cache_impl(state_row.get("memory_cache"))
    intentions_active = _normalize_intentions_stack_impl(state_row.get("intentions_active"))
    current_user_text = str(message or "").strip()
    directive_text = str(self_turn_directive or "").strip()

    soul_context_for_retrieve: list[dict[str, Any]] = []
    identity_context = _build_retrieve_identity_context(soul_id, apimw=(identity_mode == "apimw"))
    if identity_context:
        soul_context_for_retrieve.append({"role": "identity_context", "content": {"text": identity_context}})
    all_cats_summary = str(state_row.get("all_categories_summary") or "").strip()
    if all_cats_summary:
        soul_context_for_retrieve.append({"role": "all_categories_summary", "content": {"text": all_cats_summary}})

    if conversations_block is not None:
        history_text = str(conversations_block or "").strip()
    else:
        history_text = build_conversations_block(
            history=history or [],
            conversation_id=conversation_id,
            chat_label=chat_label,
            soul_name=soul_id,
            include_empty_current=False,
            current_user_text=current_user_text,
            self_turn_directive=directive_text,
        )
    if history_text:
        soul_context_for_retrieve.append(
            {
                "role": "history",
                "content": {"text": history_text},
            }
        )
    cache_text = "\n".join(_format_working_thoughts_lines(memory_cache))
    if cache_text:
        soul_context_for_retrieve.append({"role": "memory_cache", "content": {"text": cache_text}})
    intentions_text = _format_intentions_for_prompt(intentions_active) if intentions_active else ""
    if intentions_text and intentions_text.strip() != "(none)":
        soul_context_for_retrieve.append({"role": "intentions", "content": {"text": intentions_text}})

    if directive_text:
        label = str(self_turn_label or "").strip() or "Self-turn directive"
        soul_context_for_retrieve.append(
            {"role": "self_turn", "content": {"text": f"{label}:\n{directive_text}"}}
        )
    else:
        soul_context_for_retrieve.append({"role": "user", "content": {"text": message}})
    return soul_context_for_retrieve


async def _run_retrieve(
    payload: dict[str, Any],
    *,
    conversation_id: str | None = None,
    safe_payload: Callable[[dict[str, Any]], dict[str, Any]],
    extract_conversation_id: Callable[[dict[str, Any]], str | None],
    get_service_from_payload: Callable[[dict[str, Any]], Any],
    parse_as_of_datetime: Callable[[Any], Any],
    sqlite_current_path: Callable[[str | None, str | None], Any],
    sqlite_connect: Callable[[Any], Any],
    sqlite_ensure_conversation_state_schema: Callable[[Any], None],
    conversation_state_from_row: Callable[..., dict[str, Any] | None],
    conversation_state_row: Callable[[Any, str], Any],
    write_conversation_state: Callable[..., Any],
    procedural_module: Any,
    procedural_yaml_dir: Callable[[dict[str, Any]], Any],
    procedural_db_path: Callable[[dict[str, Any]], Any],
    procedural_should_ingest: Callable[[Any, Any], bool],
    config: dict[str, Any],
    logger: Any,
) -> dict[str, Any]:
    safe = safe_payload(payload)
    scoped_conversation_id = str(conversation_id or extract_conversation_id(safe) or "").strip() or None
    if scoped_conversation_id:
        safe["conversation_id"] = scoped_conversation_id

    svc = get_service_from_payload(safe)
    scope = _extract_retrieve_where(safe)
    memu_queries = _extract_retrieve_queries(safe)
    force_retrieve = _extract_force_retrieve(safe)
    trace_id = _extract_trace_id(safe)
    as_of = parse_as_of_datetime(safe.get("as_of"))

    soul_id = str((scope or {}).get("soul_id") or "").strip()
    user_id = str((scope or {}).get("user_id") or "user").strip() or "user"

    retrieve_rewrite_angle = 0
    if scoped_conversation_id and soul_id:
        db_path = sqlite_current_path(user_id or None, soul_id)
        if db_path is not None and db_path.exists():
            con = sqlite_connect(db_path)
            try:
                con.row_factory = sqlite3.Row
                sqlite_ensure_conversation_state_schema(con)
                pre_row = conversation_state_from_row(conversation_state_row(con, scoped_conversation_id), con=con)
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
        mental_health_enabled=bool(safe.get("mental_health_addon")),
        force_retrieve=force_retrieve,
        trace_id=trace_id,
    )
    retrieve_ms = int((time.monotonic() - retrieve_started_at) * 1000)
    out: dict[str, Any] = {
        "ok": True,
        "result": retrieve_result,
        "retrieve_ms": retrieve_ms,
    }

    if safe.get("mental_health_addon"):
        mh_query = str(retrieve_result.get("mental_health_query") or "").strip()
        if mh_query:
            yaml_dir = procedural_yaml_dir(config)
            procedural_db = procedural_db_path(config)
            if procedural_should_ingest(procedural_db, yaml_dir):
                try:
                    await procedural_module.ingest(svc, procedural_db, yaml_dir)
                except Exception:
                    logger.exception("procedural ingest failed")
            try:
                embeds = await svc.embed([mh_query], profile="embedding")
                qvec = list(embeds[0]) if embeds else []
                expected_embedding_model = str((config.get("llm") or {}).get("embed_model") or "").strip() or None
                hits = procedural_module.lookup(
                    procedural_db,
                    domain="mental_health",
                    query_vec=qvec,
                    expected_embedding_model=expected_embedding_model,
                    limit=1,
                )
                if hits:
                    row, score = hits[0]
                    retrieve_result.setdefault("items", []).append({
                        **row,
                        "memory_type": "procedural",
                        "score": float(score),
                    })
            except Exception:
                logger.exception("procedural lookup failed")

    if scoped_conversation_id and soul_id and retrieve_result.get("needs_retrieval"):
        write_conversation_state(
            scoped_conversation_id,
            soul_id=soul_id,
            user_id=user_id,
            updates={"retrieve_rewrite_angle": (retrieve_rewrite_angle + 1) % 3},
        )

    if scoped_conversation_id:
        state_out: dict[str, Any] | None = None
        if soul_id:
            db_path = sqlite_current_path(user_id or None, soul_id)
            if db_path is not None and db_path.exists():
                con = sqlite_connect(db_path)
                try:
                    con.row_factory = sqlite3.Row
                    sqlite_ensure_conversation_state_schema(con)
                    state_out = conversation_state_from_row(conversation_state_row(con, scoped_conversation_id), con=con)
                finally:
                    con.close()
        if state_out:
            prior_context = state_out.get("prior_context")
            if prior_context is not None and str(prior_context).strip():
                out["prior_context"] = prior_context
            memory_cache = state_out.get("memory_cache") or []
            if memory_cache:
                out["memory_cache"] = memory_cache
            intentions_active = state_out.get("intentions_active") or {}
            if intentions_active.get("items"):
                out["intentions_active"] = intentions_active

    out["method"] = "rag"
    out["conversation_id"] = scoped_conversation_id
    out["queries"] = len(memu_queries)
    if as_of is not None:
        out["as_of"] = as_of.isoformat()
    return out
