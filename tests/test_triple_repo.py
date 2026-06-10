"""Tests for SQLiteTripleRepo time-travel methods (as_of)."""

from datetime import UTC, datetime

import pytest

try:
    from memu.app.settings import DefaultUserModel
    from memu.database.models import Triple
    from memu.database.sqlite.sqlite import SQLiteStore
    _AVAILABLE = True
except Exception:
    _AVAILABLE = False


@pytest.fixture()
def repo():
    if not _AVAILABLE:
        pytest.skip("memu not importable")
    # DefaultUserModel (user_id: str | None) is the minimal scope model that
    # avoids the MRO conflict in build_sqlite_table_model.
    return SQLiteStore(dsn="sqlite:///:memory:", scope_model=DefaultUserModel).triple_repo


def _ts(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def _triple(subject: str, obj: str, predicate: str = "parallels", valid_from=None, valid_to=None) -> Triple:
    return Triple(
        subject_id=subject,
        subject_kind="memory",
        predicate=predicate,
        object_id=obj,
        object_kind="memory",
        valid_from=valid_from,
        valid_to=valid_to,
    )


def test_as_of_excludes_expired_edges(repo):
    """as_of should not return an edge whose valid_to < as_of."""
    active = _triple("m1", "m2", valid_from=_ts(2026, 1, 1), valid_to=None)
    expired = _triple("m1", "m3", valid_from=_ts(2026, 1, 1), valid_to=_ts(2026, 3, 1))

    repo.add(active)
    repo.add(expired)

    # Query point after expiry of second edge
    as_of = _ts(2026, 4, 1)
    edges = repo.get_edges_from("m1", as_of=as_of)

    object_ids = {e.object_id for e in edges}
    assert "m2" in object_ids
    assert "m3" not in object_ids


def test_as_of_includes_edge_valid_at_point(repo):
    """as_of should include an edge whose valid window contains the query point."""
    t = _triple("a1", "b1", valid_from=_ts(2026, 1, 1), valid_to=_ts(2026, 6, 1))
    repo.add(t)

    in_window = repo.get_edges_from("a1", as_of=_ts(2026, 3, 1))
    assert any(e.object_id == "b1" for e in in_window)

    before_start = repo.get_edges_from("a1", as_of=_ts(2025, 12, 31))
    assert all(e.object_id != "b1" for e in before_start)

    after_end = repo.get_edges_from("a1", as_of=_ts(2026, 7, 1))
    assert all(e.object_id != "b1" for e in after_end)
