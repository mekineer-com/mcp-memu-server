from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException


def register_graph_routes(app: FastAPI, *, get_service: Callable[[dict[str, Any]], Any]) -> None:
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
        svc = get_service({"user": scope})
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
        svc = get_service({"user": scope})
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
        svc = get_service({"user": scope})
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
        svc = get_service({"user": scope})
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
        svc = get_service({"user": scope})
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
        svc = get_service({"user": scope})
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
        svc = get_service({"user": scope})
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
        memory_only: bool = False,
        exclude_entity_id: str | None = None,
        exclude_category_id: str | None = None,
    ):
        uid = str(user_id or "").strip()
        sid = str(soul_id or "").strip()
        query = str(q or "").strip()
        if not uid or not sid:
            raise HTTPException(status_code=400, detail="user_id and soul_id are required")
        if not query:
            raise HTTPException(status_code=400, detail="q is required")
        scope = {"user_id": uid, "soul_id": sid}
        svc = get_service({"user": scope})
        try:
            return await svc.graph_search(
                query,
                where=scope,
                limit=limit,
                mode=mode,
                since_days=since_days,
                memory_only=memory_only,
                exclude_entity_id=exclude_entity_id,
                exclude_category_id=exclude_category_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
