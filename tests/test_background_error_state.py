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

import pytest

from app.db import sqlite_ensure_conversation_state_schema
from app.services import memorize_endpoint
from app.services import soul_state as _soul_state
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
            soul_id="SoulB",
            user_id="UserB",
            updates={},
        )
        assert state["last_background_error"] is None
        assert state["last_background_error_at"] is None
        assert state["last_consolidation_error"] is None
        assert state["last_consolidation_error_at"] is None
        assert state["apimw_message_to_self"] is None


def test_apimw_message_to_self_is_soul_scoped() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        db_path, sqlite_dir = _tmp_sqlite_setup(tmp_dir, soul_id="SoulC")

        state, _ = write_conversation_state(
            "conv-apimw",
            sqlite_current_path=lambda _user, _soul: db_path,
            soul_id="SoulC",
            user_id="UserC",
            updates={"apimw_message_to_self": "notice the quiet signal"},
        )

        assert state["apimw_message_to_self"] == "notice the quiet signal"

        con = sqlite3.connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            soul = _soul_state.read(con)
        finally:
            con.close()

        assert soul["apimw_message_to_self"] == "notice the quiet signal"

        other_state, _ = write_conversation_state(
            "conv-other",
            sqlite_current_path=lambda _user, _soul: db_path,
            soul_id="SoulC",
            user_id="UserC",
            updates={},
        )
        assert other_state["apimw_message_to_self"] == "notice the quiet signal"

        cleared_state, _ = write_conversation_state(
            "conv-other",
            sqlite_current_path=lambda _user, _soul: db_path,
            soul_id="SoulC",
            user_id="UserC",
            updates={"apimw_message_to_self": None},
        )
        assert cleared_state["apimw_message_to_self"] is None


def test_consolidation_error_fields_round_trip_through_state() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        db_path, sqlite_dir = _tmp_sqlite_setup(tmp_dir, soul_id="SoulD")

        cid = "conv-consolidation-err"
        now_iso = datetime.now(UTC).isoformat()
        state, _ = write_conversation_state(
            cid,
            sqlite_current_path=lambda _user, _soul: db_path,
            soul_id="SoulD",
            user_id="UserD",
            updates={
                "last_consolidation_error": "RuntimeError: consolidation llm timeout",
                "last_consolidation_error_at": now_iso,
            },
        )

        assert state["last_consolidation_error"].startswith("RuntimeError:")
        assert state["last_consolidation_error_at"] == now_iso

        con = sqlite3.connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            row = conversation_state_row(con, cid)
            assert row is not None
            loaded = conversation_state_from_row(row)
        finally:
            con.close()

        assert loaded is not None
        assert loaded["last_consolidation_error"] == state["last_consolidation_error"]
        assert loaded["last_consolidation_error_at"] == now_iso


def test_apimw_failure_clears_prior_context() -> None:
    """APImw failure path must clear prior_context so a stale value doesn't linger."""
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        db_path, _ = _tmp_sqlite_setup(tmp_dir, soul_id="SoulE")

        cid = "conv-apimw-fail"
        # Seed a prior_context value (as if a previous successful APImw run wrote it).
        write_conversation_state(
            cid,
            sqlite_current_path=lambda _user, _soul: db_path,
            soul_id="SoulE",
            user_id="UserE",
            updates={"prior_context": "some stale context"},
        )

        # Simulate APImw failure: write error + clear prior_context (same as main.py does).
        write_conversation_state(
            cid,
            sqlite_current_path=lambda _user, _soul: db_path,
            soul_id="SoulE",
            user_id="UserE",
            updates={
                "last_background_error": "apimw_failed: RuntimeError: boom",
                "last_background_error_at": datetime.now(UTC).isoformat(),
            },
        )
        state, _ = write_conversation_state(
            cid,
            sqlite_current_path=lambda _user, _soul: db_path,
            soul_id="SoulE",
            user_id="UserE",
            updates={"prior_context": None},
        )

        assert state["prior_context"] is None
        assert state["last_background_error"].startswith("apimw_failed:")

        # Verify via fresh read.
        con = sqlite3.connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            row = conversation_state_row(con, cid)
            loaded = conversation_state_from_row(row)
        finally:
            con.close()

        assert loaded["prior_context"] is None


def test_apimw_success_writes_prior_context() -> None:
    """APImw success path writes prior_context; absence after failure is not permanent."""
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        db_path, _ = _tmp_sqlite_setup(tmp_dir, soul_id="SoulF")

        cid = "conv-apimw-ok"
        # Simulate a failure that cleared prior_context.
        write_conversation_state(
            cid,
            sqlite_current_path=lambda _user, _soul: db_path,
            soul_id="SoulF",
            user_id="UserF",
            updates={"prior_context": None},
        )

        # Simulate next-turn APImw success: writes a fresh prior_context.
        state, _ = write_conversation_state(
            cid,
            sqlite_current_path=lambda _user, _soul: db_path,
            soul_id="SoulF",
            user_id="UserF",
            updates={"prior_context": "fresh context after recovery"},
        )

        assert state["prior_context"] == "fresh context after recovery"


@pytest.mark.asyncio
async def test_forced_memorize_raises_if_state_error_write_fails() -> None:
    class _Logger:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def exception(self, message: str) -> None:
            self.messages.append(message)

    class _NoopLock:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, _exc_type, _exc, _tb) -> bool:
            return False

    async def _failing_memorize_handler(_payload, _background_tasks, _force) -> None:
        raise RuntimeError("llm refused")

    def _get_memorize_lock(_key: str) -> _NoopLock:
        return _NoopLock()

    def _memorize_lock_key(user_id: str, soul_id: str) -> str:
        return f"{user_id}:{soul_id}"

    def _write_state(*_args, **_kwargs):
        raise RuntimeError("sqlite write failed")

    logger = _Logger()
    payload = {
        "user": {
            "conversation_id": "conv-1",
            "soul_id": "Echo",
            "user_id": "Marcos",
        },
    }

    with pytest.raises(RuntimeError, match="background error state write also failed"):
        await memorize_endpoint.run_forced_memorize_from_turn(
            payload,
            memorize_handler=_failing_memorize_handler,
            logger=logger,
            get_memorize_lock=_get_memorize_lock,
            memorize_lock_key=_memorize_lock_key,
            write_conversation_state=_write_state,
        )

    assert "forced memorize from turn failed" in logger.messages
    assert "failed to record background error on conversation state" in logger.messages
