from __future__ import annotations

import json
import sqlite3
import asyncio

import pytest

from app.services import soul_state, soul_summaries
from app import main


def _connection() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    soul_state.ensure_schema(con)
    return con


def test_legacy_migration_persists_approved_baseline_once(tmp_path) -> None:
    path = tmp_path / "soul.db"
    con = sqlite3.connect(path)
    con.execute(
        """
CREATE TABLE soul_state (
    id INTEGER PRIMARY KEY,
    narrative_self TEXT,
    all_categories_summary TEXT,
    memory_cache JSON DEFAULT '[]',
    intentions_active JSON,
    retrieve_rewrite_angle INTEGER DEFAULT 0,
    retrieval_ids_since_consolidation JSON DEFAULT '[]',
    prior_context_ids_since_consolidation JSON DEFAULT '[]',
    last_consolidation_at DATETIME,
    consolidation_in_progress BOOLEAN DEFAULT 0,
    consolidation_started_at DATETIME,
    updated_at DATETIME
)
"""
    )
    con.execute(
        "INSERT INTO soul_state (id, narrative_self, all_categories_summary) VALUES (1, 'old self', 'old cats')"
    )
    con.commit()
    soul_state.ensure_schema(con)
    con.close()

    check = sqlite3.connect(path)
    check.row_factory = sqlite3.Row
    state = soul_state.read(check)
    assert state["narrative_self_approved"] == "old self"
    assert state["all_categories_summary_approved"] == "old cats"
    check.execute("UPDATE soul_state SET narrative_self = 'new self' WHERE id = 1")
    check.commit()
    soul_state.ensure_schema(check)
    assert soul_state.read(check)["narrative_self_approved"] == "old self"
    check.close()


def test_soul_summary_write_approve_and_journal(monkeypatch, tmp_path) -> None:
    journal = tmp_path / "journal.jsonl"

    def append(**entry):
        journal.write_text(json.dumps(entry), encoding="utf-8")

    monkeypatch.setattr(soul_summaries, "append_summary_journal", append)
    con = _connection()
    scope = {"user_id": "u", "soul_id": "s"}

    state = soul_summaries.write_live(
        con,
        kind="narrative_self",
        summary="first self",
        scope=scope,
        edited_by="consolidation",
    )
    assert state["narrative_self"] == "first self"
    assert state["narrative_self_previous"] is None
    assert state["narrative_self_approved"] is None
    assert state["summaries_revision"] == 1
    assert json.loads(journal.read_text())["summary_id"] == "soul-summary:narrative_self"

    state = soul_summaries.approve(con, kind="narrative_self")
    assert state["narrative_self_approved"] == "first self"
    assert state["summaries_revision"] == 2
    assert soul_summaries.list_for_review(con)[0]["pending"] is False


def test_journal_failure_leaves_soul_summary_unchanged(monkeypatch) -> None:
    con = _connection()

    def fail(**_kwargs):
        raise OSError("blocked")

    monkeypatch.setattr(soul_summaries, "append_summary_journal", fail)
    with pytest.raises(OSError, match="blocked"):
        soul_summaries.write_live(
            con,
            kind="all_categories_summary",
            summary="new",
            scope={"user_id": "u", "soul_id": "s"},
            edited_by="pipeline",
        )
    assert soul_state.read(con)["all_categories_summary"] is None


def test_soul_summary_route_rejects_stale_snapshot(monkeypatch, tmp_path) -> None:
    path = tmp_path / "soul.db"
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    soul_state.ensure_schema(con)
    con.execute(
        "UPDATE soul_state SET narrative_self = 'current', narrative_self_approved = 'approved', summaries_revision = 4"
    )
    con.commit()
    con.close()
    monkeypatch.setattr(main, "_sqlite_current_path", lambda _uid, _sid: path)
    monkeypatch.setattr(soul_summaries, "append_summary_journal", lambda **_kwargs: None)

    with pytest.raises(main.HTTPException) as exc_info:
        asyncio.run(
            main.soul_summary_update(
                kind="narrative_self",
                user_id="u",
                soul_id="s",
                payload={
                    "summary": "edited",
                    "displayed_summary": "current",
                    "summaries_revision": 3,
                },
            )
        )
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "summary_snapshot_stale"


def test_soul_summary_route_updates_displayed_snapshot(monkeypatch, tmp_path) -> None:
    path = tmp_path / "soul.db"
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    soul_state.ensure_schema(con)
    con.execute(
        "UPDATE soul_state SET narrative_self = 'current', narrative_self_approved = 'approved', summaries_revision = 4"
    )
    con.commit()
    con.close()
    monkeypatch.setattr(main, "_sqlite_current_path", lambda _uid, _sid: path)
    monkeypatch.setattr(soul_summaries, "append_summary_journal", lambda **_kwargs: None)

    out = asyncio.run(
        main.soul_summary_update(
            kind="narrative_self",
            user_id="u",
            soul_id="s",
            payload={
                "summary": "edited",
                "displayed_summary": "current",
                "summaries_revision": 4,
            },
        )
    )
    assert out["summary"] == "edited"
    assert out["approved_summary"] == "edited"
    assert out["previous_summary"] == "current"
    assert out["summaries_revision"] == 5
