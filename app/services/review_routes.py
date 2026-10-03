from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from memu.app.dossier import DossierRevisionStaleError
from memu.app.graph import DossierMembershipConflictError, MemoryCitationConflictError

from app.services import soul_state as _soul_state
from app.services import soul_summaries as _soul_summaries


def register_review_routes(
    app: FastAPI,
    *,
    get_service: Callable[[dict[str, Any]], Any],
    sqlite_current_path: Callable[[str, str], Path | None],
    sqlite_connect: Callable[[Path], sqlite3.Connection],
) -> None:
    @app.get("/memory/{item_id}", operation_id="memory_graph_item")
    async def memory_graph_item(
        item_id: str,
        user_id: str,
        soul_id: str,
    ):
        uid = str(user_id or "").strip()
        sid = str(soul_id or "").strip()
        if not uid or not sid:
            raise HTTPException(status_code=400, detail="user_id and soul_id are required")
        scope = {"user_id": uid, "soul_id": sid}
        svc = get_service({"user": scope})
        item = svc.graph_memory(item_id, where=scope)
        if item is None:
            raise HTTPException(status_code=404, detail="memory not found")
        if item_id.startswith("category:"):
            db_path = sqlite_current_path(uid, sid)
            if db_path is not None and db_path.exists():
                con = sqlite_connect(db_path)
                try:
                    item = {**item, "summaries_revision": _soul_summaries.current_revision(con)}
                finally:
                    con.close()
        return item


    @app.get("/pending", operation_id="memory_graph_pending")
    async def memory_graph_pending(
        user_id: str,
        soul_id: str,
    ):
        uid = str(user_id or "").strip()
        sid = str(soul_id or "").strip()
        if not uid or not sid:
            raise HTTPException(status_code=400, detail="user_id and soul_id are required")
        scope = {"user_id": uid, "soul_id": sid}
        svc = get_service({"user": scope})
        pending = svc.graph_list_pending(where=scope)
        db_path = sqlite_current_path(uid, sid)
        pending["soul_summaries"] = []
        pending["summaries_revision"] = 0
        if db_path is not None and db_path.exists():
            con = sqlite_connect(db_path)
            try:
                con.row_factory = sqlite3.Row
                pending["soul_summaries"] = [
                    row for row in _soul_summaries.list_for_review(con) if row["pending"]
                ]
                pending["summaries_revision"] = _soul_summaries.current_revision(con)
            finally:
                con.close()
        return pending


    def _summary_snapshot(payload: Mapping[str, Any]) -> tuple[int, str] | None:
        has_revision = "summaries_revision" in payload
        has_displayed = "displayed_summary" in payload
        if has_revision != has_displayed:
            raise HTTPException(status_code=400, detail="summaries_revision and displayed_summary are required together")
        if not has_revision:
            return None
        revision = payload["summaries_revision"]
        displayed = payload["displayed_summary"]
        if isinstance(revision, bool) or not isinstance(revision, int) or not isinstance(displayed, str):
            raise HTTPException(status_code=400, detail="invalid summary snapshot")
        return revision, displayed


    def _category_snapshot(payload: Mapping[str, Any]) -> tuple[int, str, str, str] | None:
        snapshot = _summary_snapshot(payload)
        fields = ("displayed_title", "displayed_description")
        if snapshot is None and not any(field in payload for field in fields):
            return None
        if snapshot is None or any(not isinstance(payload.get(field), str) for field in fields):
            raise HTTPException(status_code=400, detail="complete category snapshot is required")
        return (snapshot[0], snapshot[1].strip(), payload["displayed_title"].strip(), payload["displayed_description"].strip())


    def _summary_db(user_id: str, soul_id: str) -> tuple[sqlite3.Connection, dict[str, str]]:
        db_path = sqlite_current_path(user_id, soul_id)
        if db_path is None or not db_path.exists():
            raise HTTPException(status_code=404, detail="soul state not found")
        con = sqlite_connect(db_path)
        con.row_factory = sqlite3.Row
        _soul_state.ensure_schema(con)
        return con, {"user_id": user_id, "soul_id": soul_id}


    def _soul_summary_response(con: sqlite3.Connection, kind: str) -> dict[str, Any]:
        item = next((row for row in _soul_summaries.list_for_review(con) if row["kind"] == kind), None)
        if item is None:
            raise HTTPException(status_code=404, detail="soul summary not found")
        return {**item, "summaries_revision": _soul_summaries.current_revision(con)}


    @contextmanager
    def _category_review_write(svc: Any, category_id: str, scope: dict[str, str], snapshot: tuple[int, str, str, str] | None):
        raw_id = category_id.removeprefix("category:")
        session_cm = svc._sqlite_write_session(svc.database)
        if session_cm is None:
            raise RuntimeError("Category review requires SQLite")
        with session_cm as session:
            connection = session.connection()
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            con = connection.connection.driver_connection
            previous_row_factory = con.row_factory
            try:
                con.row_factory = sqlite3.Row
                _soul_state.ensure_schema(con)
                current = svc.database.memory_category_repo.list_categories(
                    {**scope, "id": raw_id}, session=session,
                ).get(raw_id)
                if current is None:
                    raise HTTPException(status_code=404, detail="category not found")
                revision = None
                if snapshot is not None:
                    if (
                        str(current.summary or current.description or "") != snapshot[1]
                        or current.name != snapshot[2]
                        or (current.description or "") != snapshot[3]
                    ):
                        raise HTTPException(status_code=409, detail="summary_snapshot_stale")
                    try:
                        revision = _soul_summaries.reserve_revision(con, snapshot[0])
                    except ValueError as exc:
                        raise HTTPException(status_code=409, detail="summary_snapshot_stale") from exc
                yield session, revision
                con.row_factory = previous_row_factory
                session.commit()
            except Exception:
                con.row_factory = previous_row_factory
                session.rollback()
                raise
            finally:
                con.row_factory = previous_row_factory


    @app.patch("/soul-summary/{kind}", operation_id="soul_summary_update")
    async def soul_summary_update(
        kind: str,
        user_id: str,
        soul_id: str,
        payload: dict[str, Any] = Body(...),
    ):
        if kind == "all_categories_summary":
            raise HTTPException(status_code=409, detail="all_categories_summary is read-only")
        uid = str(user_id or "").strip()
        sid = str(soul_id or "").strip()
        summary = payload.get("summary")
        if not uid or not sid:
            raise HTTPException(status_code=400, detail="user_id and soul_id are required")
        if not isinstance(summary, str) or not summary.strip():
            raise HTTPException(status_code=400, detail="summary is required")
        snapshot = _summary_snapshot(payload)
        if snapshot is None:
            raise HTTPException(status_code=400, detail="summary snapshot is required")
        con, scope = _summary_db(uid, sid)
        try:
            con.execute("BEGIN IMMEDIATE")
            _soul_summaries.write_live(
                con,
                kind=kind,
                summary=summary,
                approve=True,
                expected_revision=snapshot[0],
                displayed_summary=snapshot[1],
            )
            con.commit()
            response = _soul_summary_response(con, kind)
        except ValueError as exc:
            con.rollback()
            status = 409 if str(exc) == "summary_snapshot_stale" else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        finally:
            con.close()
        # The successful write has verified this snapshot against the locked row.
        _soul_summaries.journal_committed_write(
            kind=kind, before=snapshot[1], after=summary.strip(), scope=scope, edited_by="atomic:user",
        )
        return response


    @app.post("/soul-summary/{kind}/approve", operation_id="soul_summary_approve")
    async def soul_summary_approve(
        kind: str,
        user_id: str,
        soul_id: str,
        payload: dict[str, Any] = Body(...),
    ):
        if kind == "all_categories_summary":
            raise HTTPException(status_code=409, detail="all_categories_summary is read-only")
        uid = str(user_id or "").strip()
        sid = str(soul_id or "").strip()
        if not uid or not sid:
            raise HTTPException(status_code=400, detail="user_id and soul_id are required")
        snapshot = _summary_snapshot(payload)
        if snapshot is None:
            raise HTTPException(status_code=400, detail="summary snapshot is required")
        con, _scope = _summary_db(uid, sid)
        try:
            con.execute("BEGIN IMMEDIATE")
            _soul_summaries.approve(
                con,
                kind=kind,
                expected_revision=snapshot[0],
                displayed_summary=snapshot[1],
            )
            con.commit()
            return _soul_summary_response(con, kind)
        except ValueError as exc:
            con.rollback()
            status = 409 if str(exc) == "summary_snapshot_stale" else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        finally:
            con.close()


    @app.patch("/memory/{item_id}", operation_id="memory_graph_item_update")
    async def memory_graph_item_update(
        item_id: str,
        user_id: str,
        soul_id: str,
        payload: dict[str, Any] = Body(...),
    ):
        uid = str(user_id or "").strip()
        sid = str(soul_id or "").strip()
        summary = payload.get("summary")
        if not uid or not sid:
            raise HTTPException(status_code=400, detail="user_id and soul_id are required")
        if not isinstance(summary, str) or not summary.strip():
            raise HTTPException(status_code=400, detail="summary is required")
        displayed_summary = payload.get("displayed_summary")
        if not isinstance(displayed_summary, str):
            raise HTTPException(status_code=400, detail="displayed_summary is required")
        scope = {"user_id": uid, "soul_id": sid}
        svc = get_service({"user": scope})
        try:
            item = await svc.graph_update_memory_summary(
                item_id,
                summary=summary,
                where=scope,
                edited_by=payload.get("edited_by") if isinstance(payload.get("edited_by"), str) else None,
                approved=payload.get("approved") is True,
                expected_summary=displayed_summary,
            )
        except ValueError as exc:
            status = 409 if str(exc) == "summary_snapshot_stale" else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        except KeyError:
            item = None
        if item is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return item


    @app.post("/memory/{item_id}/approve", operation_id="memory_graph_item_approve")
    async def memory_graph_item_approve(
        item_id: str,
        user_id: str,
        soul_id: str,
        payload: dict[str, Any] = Body(...),
    ):
        uid = str(user_id or "").strip()
        sid = str(soul_id or "").strip()
        if not uid or not sid:
            raise HTTPException(status_code=400, detail="user_id and soul_id are required")
        displayed_summary = payload.get("displayed_summary")
        if not isinstance(displayed_summary, str):
            raise HTTPException(status_code=400, detail="displayed_summary is required")
        scope = {"user_id": uid, "soul_id": sid}
        svc = get_service({"user": scope})
        try:
            item = svc.graph_approve_memory(item_id, where=scope, expected_summary=displayed_summary)
        except ValueError as exc:
            status = 409 if str(exc) == "summary_snapshot_stale" else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return item


    @app.delete("/memory/{item_id}", operation_id="memory_graph_item_delete")
    async def memory_graph_item_delete(
        item_id: str,
        user_id: str,
        soul_id: str,
        displayed_summary: str,
    ):
        uid = str(user_id or "").strip()
        sid = str(soul_id or "").strip()
        if not uid or not sid:
            raise HTTPException(status_code=400, detail="user_id and soul_id are required")
        scope = {"user_id": uid, "soul_id": sid}
        svc = get_service({"user": scope})
        try:
            item = svc.graph_delete_memory(item_id, where=scope, expected_summary=displayed_summary)
        except MemoryCitationConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail={"message": str(exc), "dossiers": exc.usages},
            ) from exc
        except ValueError as exc:
            status = 409 if str(exc) == "summary_snapshot_stale" else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return item


    @app.patch("/category/{category_id}", operation_id="memory_graph_category_update")
    async def memory_graph_category_update(
        category_id: str,
        user_id: str,
        soul_id: str,
        payload: dict[str, Any] = Body(...),
    ):
        uid = str(user_id or "").strip()
        sid = str(soul_id or "").strip()
        summary = payload.get("summary")
        title = payload.get("title")
        description = payload.get("description")
        category_kind = payload.get("kind")
        if not uid or not sid:
            raise HTTPException(status_code=400, detail="user_id and soul_id are required")
        supplied = (summary, title, description, category_kind)
        if not any(value is not None for value in supplied):
            raise HTTPException(status_code=400, detail="title, description, summary, or kind is required")
        if any(value is not None and (not isinstance(value, str) or not value.strip()) for value in (summary, title, description)):
            raise HTTPException(status_code=400, detail="dossier text fields must be non-empty strings")
        if category_kind is not None and category_kind not in {"lore", "topic", "goal"}:
            raise HTTPException(status_code=400, detail="kind must be lore, topic, or goal")
        scope = {"user_id": uid, "soul_id": sid}
        svc = get_service({"user": scope})
        snapshot = _category_snapshot(payload)
        try:
            prepared = await svc.prepare_graph_category_update(
                category_id, summary=summary, title=title, description=description,
                category_kind=category_kind, where=scope,
                edited_by=payload.get("edited_by") if isinstance(payload.get("edited_by"), str) else None,
                approved=payload.get("approved") is True,
            )
            with _category_review_write(svc, prepared["category_id"], scope, snapshot) as (session, revision):
                category, journal = svc.write_graph_category_update(prepared, scope, session=session)
            item = svc.finish_category_review(category, scope, journal=journal)
        except DossierRevisionStaleError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KeyError:
            item = None
        if item is None:
            raise HTTPException(status_code=404, detail="category not found")
        if revision is not None:
            item = {**item, "summaries_revision": revision}
        return item


    async def _memory_graph_category_membership(
        category_id: str,
        memory_id: str,
        user_id: str,
        soul_id: str,
        payload: dict[str, Any],
        *,
        attached: bool,
    ):
        uid = str(user_id or "").strip()
        sid = str(soul_id or "").strip()
        if not uid or not sid:
            raise HTTPException(status_code=400, detail="user_id and soul_id are required")
        snapshot = _category_snapshot(payload)
        if snapshot is None:
            raise HTTPException(status_code=400, detail="summary snapshot is required")
        scope = {"user_id": uid, "soul_id": sid}
        svc = get_service({"user": scope})
        raw_category_id = category_id.removeprefix("category:")
        raw_memory_id = memory_id.removeprefix("memory:")
        try:
            with _category_review_write(svc, raw_category_id, scope, snapshot) as (session, revision):
                category = svc.graph_set_category_membership(
                    raw_category_id, raw_memory_id, attached=attached,
                    expected_displayed_summary=snapshot[1], where=scope, session=session,
                )
            item = svc.finish_category_review(category, scope, refresh_memberships=True)
        except (DossierRevisionStaleError, DossierMembershipConflictError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {**item, "summaries_revision": revision}


    @app.put(
        "/category/{category_id}/memory/{memory_id}",
        operation_id="memory_graph_category_memory_attach",
    )
    async def memory_graph_category_memory_attach(
        category_id: str,
        memory_id: str,
        user_id: str,
        soul_id: str,
        payload: dict[str, Any] = Body(...),
    ):
        return await _memory_graph_category_membership(
            category_id, memory_id, user_id, soul_id, payload, attached=True
        )


    @app.delete(
        "/category/{category_id}/memory/{memory_id}",
        operation_id="memory_graph_category_memory_detach",
    )
    async def memory_graph_category_memory_detach(
        category_id: str,
        memory_id: str,
        user_id: str,
        soul_id: str,
        payload: dict[str, Any] = Body(...),
    ):
        return await _memory_graph_category_membership(
            category_id, memory_id, user_id, soul_id, payload, attached=False
        )


    @app.post("/category/{category_id}/approve", operation_id="memory_graph_category_approve")
    async def memory_graph_category_approve(
        category_id: str,
        user_id: str,
        soul_id: str,
        payload: dict[str, Any] | None = None,
    ):
        uid = str(user_id or "").strip()
        sid = str(soul_id or "").strip()
        if not uid or not sid:
            raise HTTPException(status_code=400, detail="user_id and soul_id are required")
        scope = {"user_id": uid, "soul_id": sid}
        svc = get_service({"user": scope})
        snapshot = _category_snapshot(payload or {})
        try:
            kind, _, raw_id = str(category_id or "").partition(":")
            if not raw_id:
                kind, raw_id = "category", kind
            if kind != "category" or not raw_id:
                raise ValueError("only category approvals are supported")
            with _category_review_write(svc, raw_id, scope, snapshot) as (session, revision):
                category = svc.graph_approve_category(category_id, where=scope, session=session)
                if category is None:
                    raise HTTPException(status_code=404, detail="category not found")
            item = svc.finish_category_review(category, scope)
        except DossierRevisionStaleError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="category not found")
        if revision is not None:
            item = {**item, "summaries_revision": revision}
        return item
