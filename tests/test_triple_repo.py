"""Tests for SQLiteTripleRepo time-travel methods (as_of + timeline)."""

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


def test_timeline_orders_chronologically(repo):
    """timeline() must return edges in ascending valid_from order."""
    repo.add(_triple("ent1", "m_c", valid_from=_ts(2026, 3, 1)))
    repo.add(_triple("ent1", "m_a", valid_from=_ts(2026, 1, 1)))
    repo.add(_triple("ent1", "m_b", valid_from=_ts(2026, 2, 1)))

    rows = repo.timeline("ent1")
    dates = [r.valid_from for r in rows]
    assert dates == sorted(dates)


def test_timeline_as_of_excludes_expired(repo):
    """timeline() with as_of should exclude edges expired before as_of."""
    repo.add(_triple("ent2", "m1", valid_from=_ts(2026, 1, 1), valid_to=None))
    repo.add(_triple("ent2", "m2", valid_from=_ts(2026, 1, 1), valid_to=_ts(2026, 2, 1)))

    rows = repo.timeline("ent2", as_of=_ts(2026, 4, 1))
    object_ids = {r.object_id for r in rows}
    assert "m1" in object_ids
    assert "m2" not in object_ids


def test_timeline_respects_limit(repo):
    """timeline() should cap results at limit."""
    for i in range(10):
        repo.add(_triple("ent3", f"m{i}", valid_from=_ts(2026, 1, i + 1)))

    rows = repo.timeline("ent3", limit=3)
    assert len(rows) == 3


def test_query_entity_outgoing(repo):
    """query_entity direction='outgoing' returns subject edges only."""
    repo.add(_triple("subj", "obj1"))
    repo.add(_triple("other", "subj"))  # incoming, should NOT appear

    rows = repo.query_entity("subj", direction="outgoing")
    assert all(r.subject_id == "subj" for r in rows)
    assert not any(r.object_id == "subj" for r in rows)


def test_query_entity_incoming(repo):
    """query_entity direction='incoming' returns object edges only."""
    repo.add(_triple("other1", "subj2"))
    repo.add(_triple("subj2", "other2"))  # outgoing, should NOT appear

    rows = repo.query_entity("subj2", direction="incoming")
    assert all(r.object_id == "subj2" for r in rows)
    assert not any(r.subject_id == "subj2" for r in rows)


def test_query_entity_both(repo):
    """query_entity direction='both' returns all edges touching entity.

    Uses an asymmetric predicate so endpoint order is preserved — symmetric
    predicates (parallels, conflicts_with) canonicalize endpoints on write
    and don't retain "incoming vs outgoing" distinction.
    """
    repo.add(_triple("nodeX", "nodeY", predicate="shaped_by"))
    repo.add(_triple("nodeZ", "nodeX", predicate="shaped_by"))

    rows = repo.query_entity("nodeX", direction="both")
    ids = [(r.subject_id, r.object_id) for r in rows]
    assert ("nodeX", "nodeY") in ids
    assert ("nodeZ", "nodeX") in ids


def test_query_entity_as_of_excludes_expired(repo):
    """query_entity with as_of excludes edges expired before query point."""
    repo.add(_triple("e1", "e2", valid_from=_ts(2026, 1, 1), valid_to=_ts(2026, 2, 1)))
    repo.add(_triple("e1", "e3", valid_from=_ts(2026, 1, 1), valid_to=None))

    rows = repo.query_entity("e1", as_of=_ts(2026, 4, 1))
    object_ids = {r.object_id for r in rows}
    assert "e3" in object_ids
    assert "e2" not in object_ids
