from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from memu.app.graph import EntityActionConflictError, EntityMergeConflictError


def register_entity_routes(app: FastAPI, *, get_service: Callable[[dict[str, Any]], Any]) -> None:
    @app.get("/integration/atomic/entities", operation_id="atomic_memory_entities", tags=["integration"])
    async def atomic_memory_entities(user_id: str, soul_id: str):
        uid = str(user_id or "").strip()
        sid = str(soul_id or "").strip()
        if not uid or not sid:
            raise HTTPException(status_code=400, detail="user_id and soul_id are required")
        scope = {"user_id": uid, "soul_id": sid}
        return get_service({"user": scope}).graph_atomic_entities(where=scope)


    def _atomic_entity_aliases(payload: dict[str, Any]) -> list[str] | None:
        aliases = payload.get("aliases")
        if aliases is None:
            return None
        if not isinstance(aliases, list):
            raise HTTPException(status_code=400, detail="aliases must be a list")
        return [str(alias or "").strip() for alias in aliases if str(alias or "").strip()]


    @app.post("/integration/atomic/entities", operation_id="atomic_create_entity", tags=["integration"])
    async def atomic_create_entity(user_id: str, soul_id: str, payload: dict[str, Any] = Body(...)):
        uid = str(user_id or "").strip()
        sid = str(soul_id or "").strip()
        if not uid or not sid:
            raise HTTPException(status_code=400, detail="user_id and soul_id are required")
        scope = {"user_id": uid, "soul_id": sid}
        try:
            return get_service({"user": scope}).graph_create_entity(
                str(payload.get("name") or ""),
                str(payload.get("entity_type") or ""),
                aliases=_atomic_entity_aliases(payload),
                where=scope,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @app.get(
        "/integration/atomic/entities/{entity_id}",
        operation_id="atomic_memory_entity",
        tags=["integration"],
    )
    async def atomic_memory_entity(entity_id: str, user_id: str, soul_id: str):
        uid = str(user_id or "").strip()
        sid = str(soul_id or "").strip()
        if not uid or not sid:
            raise HTTPException(status_code=400, detail="user_id and soul_id are required")
        scope = {"user_id": uid, "soul_id": sid}
        entity = get_service({"user": scope}).graph_atomic_entity(entity_id, where=scope)
        if entity is None:
            raise HTTPException(status_code=404, detail="entity not found")
        return entity


    @app.patch(
        "/integration/atomic/entities/{entity_id}",
        operation_id="atomic_update_entity",
        tags=["integration"],
    )
    async def atomic_update_entity(
        entity_id: str,
        user_id: str,
        soul_id: str,
        payload: dict[str, Any] = Body(...),
    ):
        uid = str(user_id or "").strip()
        sid = str(soul_id or "").strip()
        if not uid or not sid:
            raise HTTPException(status_code=400, detail="user_id and soul_id are required")
        if "description" in payload:
            raise HTTPException(status_code=400, detail="description is not supported for entities")
        if not any(key in payload for key in ("name", "entity_type", "aliases")):
            raise HTTPException(status_code=400, detail="name, entity_type, or aliases required")
        scope = {"user_id": uid, "soul_id": sid}
        try:
            entity = get_service({"user": scope}).graph_update_entity(
                entity_id,
                name=str(payload["name"]) if "name" in payload else None,
                entity_type=str(payload["entity_type"]) if "entity_type" in payload else None,
                aliases=_atomic_entity_aliases(payload),
                where=scope,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if entity is None:
            raise HTTPException(status_code=404, detail="entity not found")
        return entity


    async def _atomic_set_entity_ignored(
        entity_id: str,
        user_id: str,
        soul_id: str,
        *,
        ignored: bool,
    ):
        scope = {"user_id": str(user_id or "").strip(), "soul_id": str(soul_id or "").strip()}
        if not all(scope.values()):
            raise HTTPException(status_code=400, detail="user_id and soul_id are required")
        try:
            return get_service({"user": scope}).graph_set_entity_ignored(
                entity_id, ignored=ignored, where=scope
            )
        except EntityActionConflictError as exc:
            raise HTTPException(status_code=409, detail={"conflicts": exc.conflicts}) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


    @app.post("/integration/atomic/entities/{entity_id}/ignore", tags=["integration"])
    async def atomic_ignore_entity(entity_id: str, user_id: str, soul_id: str):
        return await _atomic_set_entity_ignored(entity_id, user_id, soul_id, ignored=True)


    @app.post("/integration/atomic/entities/{entity_id}/restore", tags=["integration"])
    async def atomic_restore_entity(entity_id: str, user_id: str, soul_id: str):
        return await _atomic_set_entity_ignored(entity_id, user_id, soul_id, ignored=False)


    @app.delete("/integration/atomic/entities/{entity_id}", tags=["integration"])
    async def atomic_delete_entity(entity_id: str, user_id: str, soul_id: str):
        scope = {"user_id": str(user_id or "").strip(), "soul_id": str(soul_id or "").strip()}
        if not all(scope.values()):
            raise HTTPException(status_code=400, detail="user_id and soul_id are required")
        try:
            get_service({"user": scope}).graph_delete_entity(entity_id, where=scope)
        except EntityActionConflictError as exc:
            raise HTTPException(status_code=409, detail={"conflicts": exc.conflicts}) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"ok": True, "entity_id": entity_id}


    @app.get(
        "/integration/atomic/entities/{entity_id}/merge-preview",
        operation_id="atomic_preview_entity_merge",
        tags=["integration"],
    )
    async def atomic_preview_entity_merge(
        entity_id: str,
        duplicate_entity_id: str,
        user_id: str,
        soul_id: str,
    ):
        scope = {"user_id": str(user_id or "").strip(), "soul_id": str(soul_id or "").strip()}
        if not all(scope.values()):
            raise HTTPException(status_code=400, detail="user_id and soul_id are required")
        try:
            return get_service({"user": scope}).graph_preview_entity_merge(
                entity_id, duplicate_entity_id, where=scope
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @app.post(
        "/integration/atomic/entities/{entity_id}/merge",
        operation_id="atomic_merge_entities",
        tags=["integration"],
    )
    async def atomic_merge_entities(
        entity_id: str,
        user_id: str,
        soul_id: str,
        payload: dict[str, Any] = Body(...),
    ):
        scope = {"user_id": str(user_id or "").strip(), "soul_id": str(soul_id or "").strip()}
        duplicate_id = str(payload.get("duplicate_entity_id") or "").strip()
        if not all(scope.values()) or not duplicate_id:
            raise HTTPException(status_code=400, detail="user_id, soul_id, and duplicate_entity_id are required")
        try:
            return get_service({"user": scope}).graph_merge_entities(
                entity_id, duplicate_id, where=scope
            )
        except EntityMergeConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail={"conflicts": exc.conflicts, "canonical": exc.canonical, "duplicate": exc.duplicate},
            ) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    async def _atomic_set_memory_entity(
        memory_id: str,
        entity_id: str,
        user_id: str,
        soul_id: str,
        *,
        attached: bool,
    ) -> dict[str, Any]:
        uid = str(user_id or "").strip()
        sid = str(soul_id or "").strip()
        if not uid or not sid:
            raise HTTPException(status_code=400, detail="user_id and soul_id are required")
        scope = {"user_id": uid, "soul_id": sid}
        svc = get_service({"user": scope})
        try:
            memory = (
                svc.graph_attach_entity(memory_id, entity_id, where=scope)
                if attached
                else svc.graph_detach_entity(memory_id, entity_id, where=scope)
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except EntityActionConflictError as exc:
            raise HTTPException(status_code=409, detail={"conflicts": exc.conflicts}) from exc
        if memory is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return memory


    @app.put(
        "/integration/atomic/memories/{memory_id}/entities/{entity_id}",
        operation_id="atomic_attach_entity",
        tags=["integration"],
    )
    async def atomic_attach_entity(memory_id: str, entity_id: str, user_id: str, soul_id: str):
        return await _atomic_set_memory_entity(memory_id, entity_id, user_id, soul_id, attached=True)


    @app.delete(
        "/integration/atomic/memories/{memory_id}/entities/{entity_id}",
        operation_id="atomic_detach_entity",
        tags=["integration"],
    )
    async def atomic_detach_entity(memory_id: str, entity_id: str, user_id: str, soul_id: str):
        return await _atomic_set_memory_entity(memory_id, entity_id, user_id, soul_id, attached=False)
