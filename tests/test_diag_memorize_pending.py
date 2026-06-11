"""Tests for the read-only /diag/memorize/pending endpoint."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app import main


def _iso(y: int, mo: int, d: int, hh: int, mm: int = 0) -> str:
    return datetime(y, mo, d, hh, mm, tzinfo=UTC).isoformat()


def _msg(cid: str, idx: int, content: str, received_at: str, memorize_chat: bool) -> dict:
    return {
        "role": "user",
        "content": content,
        "received_at": received_at,
        "source_conversation_id": cid,
        "source_conversation_index": idx,
        "memorize_chat": memorize_chat,
    }


def _fixture_tails() -> dict[str, list[dict]]:
    return {
        "whatsapp:dm:1": [
            _msg("whatsapp:dm:1", 0, "w " * 300, _iso(2026, 1, 1, 10), True),
            _msg("whatsapp:dm:1", 1, "w " * 300, _iso(2026, 1, 1, 10, 5), True),
        ],
        "st-chat": [
            _msg("st-chat", 0, "w " * 200, _iso(2026, 1, 1, 11), True),
        ],
        "whatsapp:group:listen": [
            _msg("whatsapp:group:listen", 0, "w " * 5000, _iso(2026, 1, 1, 12), False),
        ],
    }


def _patch_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tails: dict) -> None:
    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    con.close()
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_k: db_path)
    monkeypatch.setattr(
        main,
        "_load_cross_memorize_tails_from_sources",
        lambda _con, **_k: tails,
    )


@pytest.mark.asyncio
async def test_diag_pending_sum_matches_turn_path_estimate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tails = _fixture_tails()
    _patch_sources(tmp_path, monkeypatch, tails)
    monkeypatch.setattr(main, "_MIN_CHUNK_TOKENS", 500)

    out = await main.diag_memorize_pending(user_id="u1", soul_id="Echo")

    merged = [msg for tail in tails.values() for msg in tail]
    expected = main._estimate_primary_memorize_tokens(merged)
    assert expected > 0
    assert out["summed_unmemorized_tokens"] == expected
    assert out["threshold"] == 500
    assert out["pct"] == round(expected * 100 / 500)
    assert out["computed_at"]


@pytest.mark.asyncio
async def test_diag_pending_threshold_tracks_min_chunk_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sources(tmp_path, monkeypatch, _fixture_tails())
    monkeypatch.setattr(main, "_MIN_CHUNK_TOKENS", 12345)

    out = await main.diag_memorize_pending(user_id="u1", soul_id="Echo")

    assert out["threshold"] == 12345


@pytest.mark.asyncio
async def test_diag_pending_sleep_gap_matches_detector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A multi-day lull spans full 22:00-08:00 windows in any host timezone.
    gap_tails = {
        "whatsapp:dm:1": [
            _msg("whatsapp:dm:1", 0, "w " * 100, _iso(2026, 1, 1, 10), True),
            _msg("whatsapp:dm:1", 1, "w " * 100, _iso(2026, 1, 4, 10), True),
        ],
    }
    _patch_sources(tmp_path, monkeypatch, gap_tails)

    out = await main.diag_memorize_pending(user_id="u1", soul_id="Echo")

    merged = [msg for tail in gap_tails.values() for msg in tail]
    assert out["sleep_gap_ready"] is True
    assert out["sleep_gap_ready"] == main._unmemorized_sleep_gap_detected(
        merged, -1, {}, min_chunk_tokens=0
    )


@pytest.mark.asyncio
async def test_diag_pending_no_sleep_gap_for_tight_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tight_tails = {
        "whatsapp:dm:1": [
            _msg("whatsapp:dm:1", 0, "w " * 100, _iso(2026, 1, 1, 10), True),
            _msg("whatsapp:dm:1", 1, "w " * 100, _iso(2026, 1, 1, 10, 1), True),
        ],
    }
    _patch_sources(tmp_path, monkeypatch, tight_tails)

    out = await main.diag_memorize_pending(user_id="u1", soul_id="Echo")

    assert out["sleep_gap_ready"] is False


@pytest.mark.asyncio
async def test_diag_pending_requires_soul_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_k: None)

    out = await main.diag_memorize_pending()

    assert out == {"ok": False, "reason": "soul_id_required"}


def _seed_conversations(tmp_path: Path, user_ids: list[str]) -> Path:
    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    main._sqlite_ensure_conversation_state_schema(con)
    con.executemany(
        "INSERT INTO conversations (conversation_id, soul_id, user_id) VALUES (?, 'Echo', ?)",
        [(f"c{i}", uid) for i, uid in enumerate(user_ids)],
    )
    con.commit()
    con.close()
    return db_path


@pytest.mark.asyncio
async def test_diag_pending_resolves_user_id_from_conversations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tails = _fixture_tails()
    db_path = _seed_conversations(tmp_path, ["u1", "u1"])
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_k: db_path)

    def _loader(_con: sqlite3.Connection, **kwargs: object) -> dict[str, list[dict]]:
        # Mirror production: a wrong/empty uid loses the SillyTavern tail.
        if kwargs.get("user_id") != "u1":
            return {k: v for k, v in tails.items() if k != "st-chat"}
        return tails

    monkeypatch.setattr(main, "_load_cross_memorize_tails_from_sources", _loader)
    monkeypatch.setattr(main, "_MIN_CHUNK_TOKENS", 500)

    out = await main.diag_memorize_pending(soul_id="Echo")

    merged = [msg for tail in tails.values() for msg in tail]
    assert out["summed_unmemorized_tokens"] == main._estimate_primary_memorize_tokens(merged)


@pytest.mark.asyncio
async def test_diag_pending_ambiguous_user_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = _seed_conversations(tmp_path, ["u1", "u2"])
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_k: db_path)

    out = await main.diag_memorize_pending(soul_id="Echo")

    assert out == {"ok": False, "reason": "user_id_ambiguous"}
