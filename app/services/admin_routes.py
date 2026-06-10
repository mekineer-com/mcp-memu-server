from __future__ import annotations

import logging
import os
import sqlite3
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

try:
    import pwd
except ImportError:  # pragma: no cover
    pwd = None  # type: ignore


def register_admin_routes(
    app: FastAPI,
    *,
    diag_prefix: str,
    build_id: str,
    server_instance_id: str,
    server_started_at_unix: float,
    startup_warnings: list[str],
    storage_status: Mapping[str, Any],
    get_config: Callable[[], dict[str, Any]],
    get_storage_dir: Callable[[dict[str, Any]], Path],
    is_ephemeral_db: Callable[[dict[str, Any]], bool],
    config_path: Callable[[], Path],
    services_cached: Callable[[], int],
    mcp_enabled: Callable[[], bool],
    shutdown_snapshot: Callable[[], dict[str, Any]],
    begin_shutdown_drain: Callable[[str | None, str | None, int], bool],
    schedule_shutdown: Callable[[int], None],
    last_calls: deque[dict[str, Any]],
    last_http: deque[dict[str, Any]],
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    sqlite_file_info: Callable[[Path], dict[str, Any]],
    sqlite_connect: Callable[[Path], sqlite3.Connection],
    sqlite_pragmas: Callable[[sqlite3.Connection], dict[str, Any]],
    sqlite_table_columns: Callable[[sqlite3.Connection, str], list[str]],
    sqlite_build_scope_where: Callable[
        [list[str], str | None, str | None, str | None], tuple[str, list[Any]]
    ],
    logger: logging.Logger,
) -> None:
    @app.get("/health", operation_id="health")
    async def health():
        cfg = get_config()
        storage_dir = get_storage_dir(cfg)
        uid = getattr(os, "geteuid", lambda: None)()
        gid = getattr(os, "getegid", lambda: None)()
        user = None
        try:
            if uid is not None and pwd is not None:
                user = pwd.getpwuid(uid).pw_name
        except (KeyError, OSError):
            logger.debug("health user lookup failed for uid=%s", uid, exc_info=True)
        return {
            "ok": (storage_status.get("ok") is not False),
            "serverInstanceId": server_instance_id,
            "buildId": build_id,
            "startedAtUnix": server_started_at_unix,
            "process": {"uid": uid, "gid": gid, "user": user},
            "ephemeralDb": is_ephemeral_db(cfg),
            "config_path": str(config_path()),
            "storage_dir": str(storage_dir),
            "storage": storage_status,
            "services_cached": services_cached(),
            "startup_warnings": startup_warnings,
            "shutdown": shutdown_snapshot(),
            "mcp": {
                "enabled": mcp_enabled(),
                "http_path": str(cfg.get("mcp", {}).get("http_path") or "/mcp"),
                "sse_path": str(cfg.get("mcp", {}).get("sse_path") or "/sse"),
            },
        }

    @app.get("/version", operation_id="version")
    async def version():
        return {
            "ok": True,
            "buildId": build_id,
            "serverInstanceId": server_instance_id,
            "startedAtUnix": server_started_at_unix,
        }

    @app.get("/admin/shutdown/status", operation_id="shutdown_status")
    async def shutdown_status():
        return {"ok": True, "shutdown": shutdown_snapshot()}

    @app.post("/admin/shutdown", operation_id="shutdown")
    async def shutdown_server(payload: dict[str, Any] | None = Body(default=None)):
        body = payload if isinstance(payload, dict) else {}
        requested_by = str(body.get("requested_by") or body.get("requestedBy") or "").strip() or None
        reason = str(body.get("reason") or "").strip() or None

        max_wait_raw = body.get("max_wait_sec", body.get("maxWaitSec", 0))
        try:
            max_wait_sec = int(max_wait_raw)
        except (TypeError, ValueError, OverflowError):
            max_wait_sec = 0
        max_wait_sec = max(0, min(max_wait_sec, 3600))

        started = begin_shutdown_drain(
            requested_by=requested_by,
            reason=reason,
            max_wait_sec=max_wait_sec,
        )
        if started:
            schedule_shutdown(max_wait_sec)

        return {
            "ok": True,
            "accepted": True,
            "already_draining": not started,
            "shutdown": shutdown_snapshot(),
        }

    @app.get(f"{diag_prefix}/diag")
    @app.get("/diag", operation_id="diag_page")
    async def diag_page():
        return HTMLResponse(
            content="""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>mcp-memu-server diagnostics</title>
<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:20px;line-height:1.4;max-width:980px}code,pre{background:#f2f2f2;border-radius:10px}code{padding:2px 6px}pre{padding:12px;overflow:auto}</style>
</head><body>
<h2>mcp-memu-server diagnostics</h2>
<ul>
<li><a href='/health'>/health</a></li>
<li><a href='/diag/http'>/diag/http</a> <small>(last 200 HTTP requests seen by this server)</small></li>
<li><a href='/diag/calls'>/diag/calls</a> <small>(last 50 memorize/retrieve calls)</small></li>
<li><a href='/diag/sqlite'>/diag/sqlite</a></li>
<li><a href='/diag/sqlite/counts'>/diag/sqlite/counts</a> <small>(add ?user_id=...&soul_id=...)</small></li>
<li><a href='/diag/sqlite/recent?table=memory_items&limit=10'>/diag/sqlite/recent</a> <small>(add scope params)</small></li>
</ul>
<p><b>Scope tip:</b> if your ST extension uses <code>user_id</code> + <code>soul_id</code>, but your tests omit one, retrieval can look empty. Use the same scope in <code>/diag/sqlite/*</code>.</p>
</body></html>"""
        )

    @app.get(f"{diag_prefix}/diag/calls")
    @app.get("/diag/calls", operation_id="diag_calls")
    async def diag_calls():
        return {"ok": True, "calls": last_calls}

    @app.get(f"{diag_prefix}/diag/http")
    @app.get("/diag/http", operation_id="diag_http")
    async def diag_http():
        return {"ok": True, "http": last_http}

    @app.get(f"{diag_prefix}/diag/sqlite")
    @app.get("/diag/sqlite", operation_id="diag_sqlite")
    async def diag_sqlite(user_id: str = "", soul_id: str = ""):
        cfg = get_config()
        try:
            storage = cfg.get("storage") if isinstance(cfg.get("storage"), dict) else {}
            meta = storage.get("metadata_store") if isinstance(storage.get("metadata_store"), dict) else {}
            provider = str(meta.get("provider") or "").lower()
            if provider not in ("sqlite", "sqlite3"):
                return {
                    "ok": False,
                    "reason": "provider_not_sqlite",
                    "provider": provider,
                    "storage": storage_status,
                }

            soul_id = soul_id.strip()
            p = sqlite_current_path(user_id or None, soul_id or None)
            if p is None:
                return {"ok": False, "reason": "soul_id_required", "storage": storage_status}

            info = sqlite_file_info(p)
            if not p.exists():
                return {"ok": False, "reason": "sqlite_file_missing", **info, "storage": storage_status}

            con = sqlite_connect(p)
            try:
                tables = [
                    r[0]
                    for r in con.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                    ).fetchall()
                ]
                return {
                    "ok": True,
                    "storage": storage_status,
                    "file": info,
                    "tables": tables,
                    "pragmas": sqlite_pragmas(con),
                }
            finally:
                con.close()
        except Exception as e:
            return {
                "ok": False,
                "reason": "exception",
                "error": f"{type(e).__name__}: {e}",
                "storage": storage_status,
            }

    @app.get(f"{diag_prefix}/diag/sqlite/counts")
    @app.get("/diag/sqlite/counts", operation_id="diag_sqlite_counts")
    async def diag_sqlite_counts(
        user_id: str | None = None,
        soul_id: str | None = None,
        conversation_id: str | None = None,
    ):
        soul_id = str(soul_id or "").strip() or None
        p = sqlite_current_path(user_id or None, soul_id)
        if p is None or not p.exists():
            reason = "soul_id_required" if p is None else "sqlite_file_missing"
            return {"ok": False, "reason": reason, "path": str(p) if p else None, "storage": storage_status}

        allowed = [
            "resources",
            "categories",
            "memory_items",
            "category_items",
            "entities",
            "triples",
            "conversations",
            "narrative_history",
            "intentions",
        ]
        con = sqlite_connect(p)
        try:
            table_names = {
                str(row[0])
                for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                if row and row[0]
            }
            out: dict[str, Any] = {
                "ok": True,
                "path": str(p),
                "scope": {"user_id": user_id, "soul_id": soul_id, "conversation_id": conversation_id},
                "tables": {},
            }
            for t in allowed:
                if t not in table_names:
                    out["tables"][t] = {"total": 0, "scoped": 0, "scope_cols": []}
                    continue
                cols = sqlite_table_columns(con, t)
                where, params = sqlite_build_scope_where(cols, user_id, soul_id, conversation_id)
                total = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                scoped = con.execute(f"SELECT COUNT(*) FROM {t}{where}", params).fetchone()[0] if where else total
                out["tables"][t] = {
                    "total": int(total),
                    "scoped": int(scoped),
                    "scope_cols": [c for c in ("user_id", "soul_id", "conversation_id") if c in cols],
                }
            return out
        finally:
            con.close()

    @app.get(f"{diag_prefix}/diag/sqlite/recent")
    @app.get("/diag/sqlite/recent", operation_id="diag_sqlite_recent")
    async def diag_sqlite_recent(
        table: str = "memory_items",
        limit: int = 20,
        user_id: str | None = None,
        soul_id: str | None = None,
        conversation_id: str | None = None,
    ):
        allowed = {
            "resources",
            "categories",
            "memory_items",
            "category_items",
            "entities",
            "triples",
            "conversations",
            "narrative_history",
            "intentions",
        }
        if table not in allowed:
            raise HTTPException(status_code=400, detail=f"table must be one of: {sorted(allowed)}")
        limit = max(1, min(int(limit or 20), 200))

        soul_id = str(soul_id or "").strip() or None
        p = sqlite_current_path(user_id or None, soul_id)
        if p is None or not p.exists():
            reason = "soul_id_required" if p is None else "sqlite_file_missing"
            return {"ok": False, "reason": reason, "path": str(p) if p else None, "storage": storage_status}

        con = sqlite_connect(p)
        try:
            table_names = {
                str(row[0])
                for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                if row and row[0]
            }
            if table not in table_names:
                return {"ok": True, "path": str(p), "table": table, "columns": [], "rows": []}
            cols = sqlite_table_columns(con, table)
            scope_where, params = sqlite_build_scope_where(cols, user_id, soul_id, conversation_id)

            prefer = [
                "id",
                "created_at",
                "updated_at",
                "user_id",
                "soul_id",
                "conversation_id",
                "name",
                "description",
                "summary",
                "memory_type",
                "source_role",
                "confidence",
                "source_message_ids",
                "reflection_salience",
                "happened_at",
                "digest_cursor",
                "prior_context",
                "intentions_active",
                "memory_cache",
                "pending_episode_ids",
                "last_memorize_at",
                "unresolved",
                "resource_id",
                "url",
                "modality",
                "local_path",
                "caption",
                "item_id",
                "category_id",
                "narrative_self",
                "source",
                "related_memory_ids",
            ]
            ban = {"embedding", "extra"}
            sel = [c for c in prefer if c in cols and c not in ban]
            if not sel:
                sel = [c for c in cols if c not in ban][:12]
            order_col = "created_at" if "created_at" in cols else ("updated_at" if "updated_at" in cols else "id")

            sql = f"SELECT {', '.join(sel)} FROM {table}{scope_where} ORDER BY {order_col} DESC LIMIT ?"
            rows = con.execute(sql, [*params, limit]).fetchall()
            out_rows = []
            for r in rows:
                d = {sel[i]: r[i] for i in range(len(sel))}
                for k, v in list(d.items()):
                    if isinstance(v, str) and len(v) > 400:
                        d[k] = v[:400] + "…"
                out_rows.append(d)
            return {"ok": True, "path": str(p), "table": table, "columns": sel, "rows": out_rows}
        finally:
            con.close()
