from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import HTTPException

from app.services.intention_state import (
    format_intentions_for_prompt as _format_intentions_for_prompt,
    normalize_intentions_stack as _normalize_intentions_stack_impl,
    normalize_memory_cache as _normalize_memory_cache_impl,
)
from app.services.payload import _canonicalize_scope_where, _extract_scope
from app.services.turn_contract import format_time_anchor as _format_time_anchor, render_history as _render_history


RETRIEVE_REWRITE_HISTORY_MESSAGES = 8
APIMW_RETRIEVE_REWRITE_HISTORY_MESSAGES = 12


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
    format_time_anchor: Callable[[], str] | None = None,
) -> str:
    name = str(soul_name or "").strip() or "the assistant"
    time_anchor = format_time_anchor or _format_time_anchor
    anchor = f"Today is {time_anchor()}."
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
) -> list[dict[str, Any]]:
    memory_cache = _normalize_memory_cache_impl(state_row.get("memory_cache"))
    intentions_active = _normalize_intentions_stack_impl(state_row.get("intentions_active"))

    soul_ctx_queries: list[dict[str, Any]] = []
    identity_context = _build_retrieve_identity_context(soul_id, apimw=(identity_mode == "apimw"))
    if identity_context:
        soul_ctx_queries.append({"role": "identity_context", "content": {"text": identity_context}})
    all_cats_summary = str(state_row.get("all_categories_summary") or "").strip()
    if all_cats_summary:
        soul_ctx_queries.append({"role": "all_categories_summary", "content": {"text": all_cats_summary}})
    cache_text = "\n".join(str(entry) for entry in (memory_cache or []))
    if cache_text:
        soul_ctx_queries.append({"role": "memory_cache", "content": {"text": cache_text}})
    intentions_text = _format_intentions_for_prompt(intentions_active) if intentions_active else ""
    if intentions_text and intentions_text.strip() != "(none)":
        soul_ctx_queries.append({"role": "intentions", "content": {"text": intentions_text}})

    history_limit = (
        APIMW_RETRIEVE_REWRITE_HISTORY_MESSAGES
        if identity_mode == "apimw"
        else RETRIEVE_REWRITE_HISTORY_MESSAGES
    )
    history_slice = history[-history_limit:] if history_limit > 0 else history
    history_text = _render_history(history_slice)
    if history_text:
        soul_ctx_queries.append({"role": "history", "content": {"text": history_text}})

    soul_ctx_queries.append({"role": "user", "content": {"text": message}})
    return soul_ctx_queries
