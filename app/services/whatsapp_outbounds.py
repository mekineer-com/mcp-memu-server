"""WhatsApp pending outbound queue helpers."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.db import (
    json_from_db,
    json_to_db,
    sqlite_connect,
    sqlite_ensure_nonempty,
)


def _ensure_whatsapp_outbounds_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
CREATE TABLE IF NOT EXISTS whatsapp_pending_outbounds (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    soul_id TEXT NOT NULL,
    origin_conversation_id TEXT NOT NULL,
    target TEXT NOT NULL CHECK (target IN ('respond', 'private')),
    target_conversation_id TEXT,
    response_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'claimed', 'sent', 'failed')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    claimed_at TEXT,
    claimed_by TEXT,
    sent_at TEXT,
    failed_at TEXT,
    provider_message_id TEXT,
    last_error TEXT,
    metadata_json TEXT,
    media_path TEXT
)
"""
    )
    try:
        con.execute("ALTER TABLE whatsapp_pending_outbounds ADD COLUMN media_path TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_whatsapp_pending_outbounds_claim "
        "ON whatsapp_pending_outbounds(status, created_at)"
    )
    con.commit()


def _whatsapp_outbound_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "soul_id": row["soul_id"],
        "origin_conversation_id": row["origin_conversation_id"],
        "target": row["target"],
        "target_conversation_id": row["target_conversation_id"],
        "response_text": row["response_text"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "claimed_at": row["claimed_at"],
        "claimed_by": row["claimed_by"],
        "sent_at": row["sent_at"],
        "failed_at": row["failed_at"],
        "provider_message_id": row["provider_message_id"],
        "last_error": row["last_error"],
        "metadata": json_from_db(row["metadata_json"]) or {},
        "media_path": row["media_path"],
    }


def _sqlite_has_rows_quietly(
    db_path: Path,
    *,
    table: str,
    where_sql: str,
    params: tuple[Any, ...],
) -> bool:
    con = sqlite3.connect(str(db_path), timeout=5.0)
    try:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout=3000")
        table_row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        if table_row is None:
            return False
        row = con.execute(f"SELECT 1 FROM {table} WHERE {where_sql} LIMIT 1", params).fetchone()
        return row is not None
    finally:
        con.close()


def _poll_marker_path(db_path: Path, name: str) -> Path:
    return db_path.parent / ".poll-markers" / f"{db_path.name}.{name}"


def _touch_poll_marker(db_path: Path, name: str, value: str = "") -> None:
    marker = _poll_marker_path(db_path, name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(value, encoding="utf-8")


def _remove_poll_marker(db_path: Path, name: str) -> None:
    _poll_marker_path(db_path, name).unlink(missing_ok=True)


def _poll_marker_due(db_path: Path, name: str, *, now: datetime) -> bool:
    marker = _poll_marker_path(db_path, name)
    if not marker.exists():
        return False
    raw = marker.read_text(encoding="utf-8").strip()
    if not raw:
        return True
    try:
        return float(raw) <= now.timestamp()
    except ValueError:
        return True


def _insert_whatsapp_outbound(
    *,
    user_id: str,
    soul_id: str,
    origin_conversation_id: str,
    target: str,
    response_text: str,
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    media_path: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    cid = str(origin_conversation_id or "").strip()
    target_clean = str(target or "").strip().lower()
    text = str(response_text or "").strip()
    mpath = str(media_path or "").strip() or None
    if not uid or not sid or not cid:
        raise ValueError("user_id, soul_id, and origin_conversation_id are required")
    if target_clean not in {"respond", "private"}:
        raise ValueError("target must be respond|private")
    if not text and not mpath:
        raise ValueError("response_text or media_path is required")

    db_path = sqlite_current_path(uid, sid)
    if db_path is None:
        raise ValueError("sqlite path unavailable for outbound scope")
    sqlite_ensure_nonempty(db_path)
    out_id = f"waout_{uuid.uuid4().hex}"
    now_iso = datetime.now(UTC).isoformat()
    target_conversation_id = cid if target_clean == "respond" else None
    con = sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        _ensure_whatsapp_outbounds_schema(con)
        con.execute(
            """
INSERT INTO whatsapp_pending_outbounds (
    id, user_id, soul_id, origin_conversation_id, target, target_conversation_id,
    response_text, status, created_at, updated_at, metadata_json, media_path
) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
""",
            (
                out_id,
                uid,
                sid,
                cid,
                target_clean,
                target_conversation_id,
                text,
                now_iso,
                now_iso,
                json_to_db(dict(metadata or {})),
                mpath,
            ),
        )
        con.commit()
    finally:
        con.close()
    _touch_poll_marker(db_path, "whatsapp-outbounds")
    return out_id


def _claim_whatsapp_outbounds(
    *,
    user_id: str,
    soul_id: str,
    claimed_by: str,
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    limit: int = 10,
    claim_timeout_seconds: int = 300,
) -> list[dict[str, Any]]:
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    claimer = str(claimed_by or "").strip() or "hermes"
    if not uid or not sid:
        raise HTTPException(status_code=400, detail="user_id and soul_id are required")
    db_path = sqlite_current_path(uid, sid)
    if db_path is None:
        raise HTTPException(status_code=400, detail="sqlite path unavailable for outbound scope")
    if not _poll_marker_path(db_path, "whatsapp-outbounds").exists():
        return []
    sqlite_ensure_nonempty(db_path)
    now = datetime.now(UTC)
    now_iso = now.isoformat()
    stale_before = (now - timedelta(seconds=max(1, int(claim_timeout_seconds)))).isoformat()
    claim_limit = max(1, min(50, int(limit)))
    if not _sqlite_has_rows_quietly(
        db_path,
        table="whatsapp_pending_outbounds",
        where_sql="user_id = ? AND soul_id = ? AND (status = 'pending' OR (status = 'claimed' AND claimed_at < ?))",
        params=(uid, sid, stale_before),
    ):
        _remove_poll_marker(db_path, "whatsapp-outbounds")
        return []
    claimed: list[dict[str, Any]] = []
    con = sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        _ensure_whatsapp_outbounds_schema(con)
        rows = con.execute(
            """
SELECT id
FROM whatsapp_pending_outbounds
WHERE user_id = ? AND soul_id = ?
  AND (status = 'pending' OR (status = 'claimed' AND claimed_at < ?))
ORDER BY created_at ASC
LIMIT ?
""",
            (uid, sid, stale_before, claim_limit),
        ).fetchall()
        for row in rows:
            out_id = str(row["id"])
            cur = con.execute(
                """
UPDATE whatsapp_pending_outbounds
SET status = 'claimed', claimed_at = ?, claimed_by = ?, updated_at = ?
WHERE id = ? AND user_id = ? AND soul_id = ?
  AND (status = 'pending' OR (status = 'claimed' AND claimed_at < ?))
""",
                (now_iso, claimer, now_iso, out_id, uid, sid, stale_before),
            )
            if cur.rowcount != 1:
                continue
            claimed_row = con.execute(
                "SELECT * FROM whatsapp_pending_outbounds WHERE id = ? LIMIT 1",
                (out_id,),
            ).fetchone()
            if claimed_row is not None:
                claimed.append(_whatsapp_outbound_row(claimed_row))
        con.commit()
    finally:
        con.close()
    if not claimed:
        _remove_poll_marker(db_path, "whatsapp-outbounds")
    return claimed


def _mark_whatsapp_outbound(
    *,
    user_id: str,
    soul_id: str,
    outbound_id: str,
    status: str,
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    provider_message_id: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    uid = str(user_id or "").strip()
    sid = str(soul_id or "").strip()
    out_id = str(outbound_id or "").strip()
    final_status = str(status or "").strip().lower()
    if not uid or not sid or not out_id:
        raise HTTPException(status_code=400, detail="user_id, soul_id, and outbound_id are required")
    if final_status not in {"sent", "failed"}:
        raise HTTPException(status_code=400, detail="status must be sent|failed")
    db_path = sqlite_current_path(uid, sid)
    if db_path is None:
        raise HTTPException(status_code=400, detail="sqlite path unavailable for outbound scope")
    sqlite_ensure_nonempty(db_path)
    now_iso = datetime.now(UTC).isoformat()
    sent_at = now_iso if final_status == "sent" else None
    failed_at = now_iso if final_status == "failed" else None
    con = sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        _ensure_whatsapp_outbounds_schema(con)
        cur = con.execute(
            """
UPDATE whatsapp_pending_outbounds
SET status = ?, updated_at = ?, sent_at = ?, failed_at = ?,
    provider_message_id = COALESCE(?, provider_message_id),
    last_error = ?
WHERE id = ? AND user_id = ? AND soul_id = ? AND status = 'claimed'
""",
            (
                final_status,
                now_iso,
                sent_at,
                failed_at,
                str(provider_message_id or "").strip() or None,
                str(error or "").strip() or None,
                out_id,
                uid,
                sid,
            ),
        )
        if cur.rowcount != 1:
            raise HTTPException(status_code=409, detail="outbound is not claimed or does not exist")
        row = con.execute(
            "SELECT * FROM whatsapp_pending_outbounds WHERE id = ? LIMIT 1",
            (out_id,),
        ).fetchone()
        con.commit()
    finally:
        con.close()
    if row is None:
        raise HTTPException(status_code=404, detail="outbound not found")
    return _whatsapp_outbound_row(row)
