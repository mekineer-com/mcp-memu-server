from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from typing import Any

from memu.app.category_summary_journal import append_summary_journal

from app.services import soul_state


_KINDS = {
    "narrative_self": ("Narrative Self", "soul-summary:narrative_self"),
    "all_categories_summary": ("All Categories Summary", "soul-summary:all_categories_summary"),
}


def _kind(kind: str) -> tuple[str, str]:
    try:
        return _KINDS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown soul summary kind: {kind}") from exc


def current_revision(con: sqlite3.Connection) -> int:
    soul_state.ensure_schema(con)
    row = con.execute("SELECT summaries_revision FROM soul_state WHERE id = 1").fetchone()
    return int(row[0] or 0)


def reserve_revision(con: sqlite3.Connection, expected_revision: int) -> int:
    soul_state.ensure_schema(con)
    result = con.execute(
        "UPDATE soul_state SET summaries_revision = summaries_revision + 1 WHERE id = 1 AND summaries_revision = ?",
        (expected_revision,),
    )
    if result.rowcount != 1:
        raise ValueError("summary_snapshot_stale")
    return expected_revision + 1


def write_live(
    con: sqlite3.Connection,
    *,
    kind: str,
    summary: str | None,
    scope: Mapping[str, Any],
    edited_by: str,
    approve: bool = False,
    advance_revision_on_noop: bool = False,
) -> dict[str, Any]:
    _label, summary_id = _kind(kind)
    clean = str(summary or "").strip()
    soul_state.ensure_schema(con)
    row = con.execute(
        f"SELECT {kind}, {kind}_approved FROM soul_state WHERE id = 1"
    ).fetchone()
    before_value = row[0]
    before = str(before_value or "")
    if not clean:
        if advance_revision_on_noop:
            con.execute("UPDATE soul_state SET summaries_revision = summaries_revision + 1 WHERE id = 1")
            return soul_state.read(con)
        raise ValueError("summary is required")
    if before == clean:
        if approve and str(row[1] or "") != clean:
            con.execute(
                f"UPDATE soul_state SET {kind}_approved = ?, summaries_revision = summaries_revision + 1 WHERE id = 1",
                (clean,),
            )
        elif advance_revision_on_noop:
            con.execute("UPDATE soul_state SET summaries_revision = summaries_revision + 1 WHERE id = 1")
        return soul_state.read(con)

    append_summary_journal(
        kind=kind,
        summary_id=summary_id,
        summary_before=before,
        summary_after=clean,
        scope=scope,
        edited_by=edited_by,
    )
    approved = f", {kind}_approved = ?" if approve else ""
    params: list[Any] = [before_value, clean]
    if approve:
        params.append(clean)
    con.execute(
        f"UPDATE soul_state SET {kind}_previous = ?, {kind} = ?{approved}, "
        "summaries_revision = summaries_revision + 1 WHERE id = 1",
        tuple(params),
    )
    return soul_state.read(con)


def approve(con: sqlite3.Connection, *, kind: str) -> dict[str, Any]:
    _kind(kind)
    soul_state.ensure_schema(con)
    row = con.execute(f"SELECT {kind} FROM soul_state WHERE id = 1").fetchone()
    current = str(row[0] or "").strip()
    if not current:
        raise ValueError("summary is required")
    con.execute(
        f"UPDATE soul_state SET {kind}_approved = ?, summaries_revision = summaries_revision + 1 WHERE id = 1",
        (current,),
    )
    return soul_state.read(con)


def list_for_review(con: sqlite3.Connection) -> list[dict[str, Any]]:
    state = soul_state.read(con)
    rows = []
    for kind, (label, summary_id) in _KINDS.items():
        summary = str(state.get(kind) or "").strip()
        if not summary:
            continue
        approved = state.get(f"{kind}_approved")
        rows.append(
            {
                "id": summary_id,
                "kind": kind,
                "label": label,
                "summary": summary,
                "approved_summary": approved,
                "previous_summary": state.get(f"{kind}_previous"),
                "pending": summary != str(approved or "").strip(),
            }
        )
    return rows
