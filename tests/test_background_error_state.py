"""State round-trip for last_background_error fields.

Covers the minimal contract: `last_background_error` and
`last_background_error_at` can be written via `write_conversation_state`
and read back via `conversation_state_row` → `conversation_state_from_row`.

The actual wire-up in `_run_forced_memorize_from_turn` uses the same
writer, so if this test passes, the state surface is intact.
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from app.db import sqlite_ensure_conversation_state_schema
from app.services.state import (
    conversation_state_from_row,
    conversation_state_row,
    write_conversation_state,
)


def _tmp_sqlite_setup(tmp_dir: Path, soul_id: str) -> tuple[Path, Path]:
    db_path = tmp_dir / f"{soul_id}.db"
    con = sqlite3.connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        sqlite_ensure_conversation_state_schema(con)
    finally:
        con.close()
    return db_path, tmp_dir


def test_background_error_fields_round_trip_through_state() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        db_path, sqlite_dir = _tmp_sqlite_setup(tmp_dir, soul_id="SoulA")

        cid = "conv-xyz"
        now_iso = datetime.now(UTC).isoformat()
        state, _ = write_conversation_state(
            cid,
            sqlite_current_path=lambda _user, _soul: db_path,
            sqlite_dir=sqlite_dir,
            soul_id="SoulA",
            user_id="UserA",
            updates={
                "last_background_error": "forced_memorize: RuntimeError: LLM refused",
                "last_background_error_at": now_iso,
            },
        )

        assert state["last_background_error"].startswith("forced_memorize:")
        assert state["last_background_error_at"] == now_iso

        # Round-trip via a fresh read
        con = sqlite3.connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            row = conversation_state_row(con, cid)
            assert row is not None
            loaded = conversation_state_from_row(row)
        finally:
            con.close()

        assert loaded is not None
        assert loaded["last_background_error"] == state["last_background_error"]
        assert loaded["last_background_error_at"] == now_iso


def test_empty_background_error_state_defaults_to_none() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        db_path, sqlite_dir = _tmp_sqlite_setup(tmp_dir, soul_id="SoulB")

        cid = "conv-new"
        state, _ = write_conversation_state(
            cid,
            sqlite_current_path=lambda _user, _soul: db_path,
            sqlite_dir=sqlite_dir,
            soul_id="SoulB",
            user_id="UserB",
            updates={},
        )
        assert state["last_background_error"] is None
        assert state["last_background_error_at"] is None


def test_subconscious_message_round_trip_through_state() -> None:
    from app.services import soul_state as _soul_state

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        db_path, sqlite_dir = _tmp_sqlite_setup(tmp_dir, soul_id="SoulC")

        cid = "conv-subconscious"
        write_conversation_state(
            cid,
            sqlite_current_path=lambda _user, _soul: db_path,
            sqlite_dir=sqlite_dir,
            soul_id="SoulC",
            user_id="UserC",
            updates={"subconscious_message": "[subconscious] remember this"},
        )

        con = sqlite3.connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            soul = _soul_state.read(con)
        finally:
            con.close()

        assert soul["subconscious_message"] == "[subconscious] remember this"
