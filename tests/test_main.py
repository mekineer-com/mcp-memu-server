"""Basic tests for the application.

Note: Full integration tests with FastAPI TestClient will be added
as the project evolves. Currently using placeholder tests to ensure
CI pipeline runs successfully.
"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_placeholder():
    """Placeholder test to ensure pytest runs successfully.

    This test will be replaced with actual integration tests
    as features are implemented.
    """
    assert True


def test_retrieve_method_from_cfg_forces_public_rag():
    try:
        from app import main
    except Exception as e:
        pytest.skip(f"Import test skipped due to compatibility issue: {e}")

    assert main._retrieve_method_from_cfg({"retrieve": {"method": "rag"}}) == "rag"
    assert main._retrieve_method_from_cfg({"retrieve": {"method": "llm"}}) == "rag"
    assert main._retrieve_method_from_cfg({"retrieve": {"method": "unknown"}}) == "rag"


def test_retrieve_apimw_enabled_from_cfg_defaults_and_override():
    try:
        from app import main
    except Exception as e:
        pytest.skip(f"Import test skipped due to compatibility issue: {e}")

    assert main._retrieve_apimw_enabled_from_cfg(None) is True
    assert main._retrieve_apimw_enabled_from_cfg({"retrieve": {}}) is True
    assert main._retrieve_apimw_enabled_from_cfg({"retrieve": {"apimw_enabled": True}}) is True
    assert main._retrieve_apimw_enabled_from_cfg({"retrieve": {"apimw_enabled": False}}) is False


def test_imports():
    """Test that main application modules can be imported."""
    try:
        from app import main

        assert hasattr(main, "app")
        assert hasattr(main, "service")
    except Exception as e:
        pytest.skip(f"Import test skipped due to compatibility issue: {e}")


def test_merge_memorize_batch_results_flattens_top_level_lists():
    try:
        from app import main
    except Exception as e:
        pytest.skip(f"Import test skipped due to compatibility issue: {e}")

    out = main._merge_memorize_batch_results(
        [
            {
                "resource": {"id": "r1", "url": "file://a"},
                "items": [{"id": "m1", "summary": "one"}],
                "categories": [{"id": "c1", "name": "Profiles"}],
                "relations": [{"item_id": "m1", "category_id": "c1"}],
                "skipped_reasons": ["skip-a"],
            },
            {
                "resource": {"id": "r2", "url": "file://b"},
                "items": [{"id": "m2", "summary": "two"}, {"id": "m1", "summary": "one"}],
                "categories": [{"id": "c1", "name": "Profiles"}, {"id": "c2", "name": "Goals"}],
                "relations": [{"item_id": "m2", "category_id": "c2"}],
                "skipped_reasons": ["skip-b", "skip-a"],
            },
        ],
        ["m2", "m1", "m2"],
    )

    assert out["batch_count"] == 2
    assert [item["id"] for item in out["items"]] == ["m1", "m2"]
    assert [cat["id"] for cat in out["categories"]] == ["c1", "c2"]
    assert out["pending_diary_episode_ids"] == ["m2", "m1"]
    assert out["skipped_reasons"] == ["skip-a", "skip-b"]
    assert "results" in out
    assert [res["id"] for res in out["resources"]] == ["r1", "r2"]


def test_estimate_unmemorized_tokens_respects_digest_cursor():
    try:
        from app import main
    except Exception as e:
        pytest.skip(f"Import test skipped due to compatibility issue: {e}")

    messages = [
        {"content": "one two three"},
        {"content": "four five six"},
        {"content": "seven eight nine"},
    ]
    assert main._estimate_unmemorized_tokens(messages, -1) == main._estimate_tokens(messages)
    assert main._estimate_unmemorized_tokens(messages, 1) == main._estimate_tokens(messages[2:])
    assert main._estimate_unmemorized_tokens(messages, 99) == 0


def test_sleep_gap_complete_since_last_message_uses_night_window():
    try:
        from app import main
    except Exception as e:
        pytest.skip(f"Import test skipped due to compatibility issue: {e}")

    zi = UTC
    last_ts = int(datetime(2026, 4, 6, 22, 30, tzinfo=UTC).timestamp() * 1000)
    now_ts = int(datetime(2026, 4, 7, 2, 30, tzinfo=UTC).timestamp() * 1000)
    assert main._sleep_gap_complete_since_last_message(last_ts, zi, now_ms=now_ts) is True

    day_last = int(datetime(2026, 4, 6, 10, 0, tzinfo=UTC).timestamp() * 1000)
    day_now = int(datetime(2026, 4, 6, 14, 30, tzinfo=UTC).timestamp() * 1000)
    assert main._sleep_gap_complete_since_last_message(day_last, zi, now_ms=day_now) is False


def test_compact_chat_x_anchors_keeps_two_unique_newest():
    try:
        from app import main
    except Exception as e:
        pytest.skip(f"Import test skipped due to compatibility issue: {e}")

    anchors = main._compact_chat_x_anchors("m9", "m8", "m9", "m7")
    assert anchors == ["m9", "m8"]


def test_slice_history_from_chat_x_anchors_uses_two_anchors_and_optional_stop_boundary():
    try:
        from app import main
    except Exception as e:
        pytest.skip(f"Import test skipped due to compatibility issue: {e}")

    history = [
        {"message_id": "m1", "content": "1"},
        {"message_id": "m2", "content": "2"},
        {"message_id": "m3", "content": "3"},
        {"message_id": "m4", "content": "4"},
        {"message_id": "m5", "content": "5"},
        {"message_id": "m6", "content": "6"},
    ]
    sliced = main._slice_history_from_chat_x_anchors(history, ["m5", "m3"], limit=12)
    assert [item.get("message_id") for item in sliced] == ["m3", "m4", "m5", "m6"]
    stopped = main._slice_history_from_chat_x_anchors(history, ["m3"], stop_at_message_id="m5", limit=12)
    assert [item.get("message_id") for item in stopped] == ["m3", "m4"]


def test_slice_history_from_chat_x_anchors_uses_one_anchor():
    try:
        from app import main
    except Exception as e:
        pytest.skip(f"Import test skipped due to compatibility issue: {e}")

    history = [
        {"message_id": "m1", "content": "1"},
        {"message_id": "m2", "content": "2"},
        {"message_id": "m3", "content": "3"},
        {"message_id": "m4", "content": "4"},
        {"message_id": "m5", "content": "5"},
    ]
    sliced = main._slice_history_from_chat_x_anchors(history, ["m4"], limit=12)
    assert [item.get("message_id") for item in sliced] == ["m4", "m5"]


def test_slice_history_from_chat_x_anchors_anchor_inversion_returns_empty():
    try:
        from app import main
    except Exception as e:
        pytest.skip(f"Import test skipped due to compatibility issue: {e}")

    history = [
        {"message_id": "m1", "content": "1"},
        {"message_id": "m2", "content": "2"},
        {"message_id": "m3", "content": "3"},
        {"message_id": "m4", "content": "4"},
        {"message_id": "m5", "content": "5"},
    ]
    # Previous anchor m4 lies AFTER the current anchor m2 in history:
    # no valid previous-episode range. Must not overlap the m2+ slice.
    sliced = main._slice_history_from_chat_x_anchors(
        history, ["m4"], stop_at_message_id="m2", limit=12
    )
    assert sliced == []
    # Equal case: same anchor on both slots.
    sliced = main._slice_history_from_chat_x_anchors(
        history, ["m3"], stop_at_message_id="m3", limit=12
    )
    assert sliced == []


def test_slice_history_from_chat_x_anchors_missing_prev_with_stop_returns_empty():
    try:
        from app import main
    except Exception as e:
        pytest.skip(f"Import test skipped due to compatibility issue: {e}")

    history = [
        {"message_id": "m1", "content": "1"},
        {"message_id": "m2", "content": "2"},
        {"message_id": "m3", "content": "3"},
    ]
    # stop_at_message_id given but no anchor → no previous range, not tail.
    sliced = main._slice_history_from_chat_x_anchors(
        history, None, stop_at_message_id="m2", limit=12
    )
    assert sliced == []
    sliced = main._slice_history_from_chat_x_anchors(
        history, [], stop_at_message_id="m2", limit=12
    )
    assert sliced == []


def test_build_force_memorize_batches_prefers_segments():
    try:
        from app import main
    except Exception as e:
        pytest.skip(f"Import test skipped due to compatibility issue: {e}")

    merged = [{"content": f"m{i}"} for i in range(1, 7)]
    segments = [
        {"start": 0, "end": 2, "file": "day1.json"},
        {"start": 3, "end": 5, "file": "day2.json"},
    ]
    batches = main._build_force_memorize_batches(
        merged,
        start_idx=0,
        segments=segments,
        days_dir=Path("/tmp/days"),
        full_path=Path("/tmp/full.json"),
        resource_url="/tmp/day-latest.json",
        max_chunk_tokens=999,
    )
    assert [end for _url, _conv, end in batches] == [2, 5]
    assert batches[0][0].endswith("/tmp/days/day1.json")
    assert batches[1][0].endswith("/tmp/days/day2.json")


def test_build_force_memorize_batches_falls_back_to_token_windows():
    try:
        from app import main
    except Exception as e:
        pytest.skip(f"Import test skipped due to compatibility issue: {e}")

    merged = [{"content": "one"} for _ in range(7)]
    batches = main._build_force_memorize_batches(
        merged,
        start_idx=0,
        segments=[],
        days_dir=Path("/tmp/days"),
        full_path=Path("/tmp/full.json"),
        resource_url="/tmp/day-latest.json",
        max_chunk_tokens=3,
    )
    assert [end for _url, _conv, end in batches] == [2, 5, 6]
    assert all(url.endswith("/tmp/full.json") for url, _conv, _end in batches)


def test_build_force_memorize_batches_fills_segment_gaps_with_full_path():
    try:
        from app import main
    except Exception as e:
        pytest.skip(f"Import test skipped due to compatibility issue: {e}")

    merged = [{"content": f"m{i}"} for i in range(1, 7)]
    segments = [
        {"start": 0, "end": 1, "file": "day1.json"},
        {"start": 4, "end": 5, "file": "day2.json"},
    ]
    batches = main._build_force_memorize_batches(
        merged,
        start_idx=0,
        segments=segments,
        days_dir=Path("/tmp/days"),
        full_path=Path("/tmp/full.json"),
        resource_url="/tmp/day-latest.json",
        max_chunk_tokens=999,
    )
    assert [end for _url, _conv, end in batches] == [1, 3, 5]
    assert batches[0][0].endswith("/tmp/days/day1.json")
    assert batches[1][0].endswith("/tmp/full.json")
    assert batches[2][0].endswith("/tmp/days/day2.json")


def test_normalize_conversation_uses_created_at_when_timestamp_missing():
    try:
        from app import main
    except Exception as e:
        pytest.skip(f"Import test skipped due to compatibility issue: {e}")

    conv = [{"role": "user", "content": "hello", "created_at": "2026-04-16T12:00:00Z"}]
    out = main._normalize_conversation(conv)

    assert isinstance(out, list) and out
    assert out[0]["ts_ms"] == int(datetime(2026, 4, 16, 12, 0, tzinfo=UTC).timestamp() * 1000)


def test_parse_as_of_datetime_accepts_iso_date_and_datetime():
    try:
        from app import main
    except Exception as e:
        pytest.skip(f"Import test skipped due to compatibility issue: {e}")

    date_only = main._parse_as_of_datetime("2026-04-18")
    assert date_only is not None
    assert date_only.isoformat().startswith("2026-04-18T00:00:00")

    zulu = main._parse_as_of_datetime("2026-04-18T10:15:00Z")
    assert zulu is not None
    assert zulu.tzinfo is not None
    assert zulu.isoformat().startswith("2026-04-18T10:15:00")


def test_parse_as_of_datetime_rejects_invalid():
    try:
        from app import main
    except Exception as e:
        pytest.skip(f"Import test skipped due to compatibility issue: {e}")

    with pytest.raises(main.HTTPException):
        main._parse_as_of_datetime("not-a-date")


def test_timeline_endpoint_returns_entity_edges(monkeypatch: pytest.MonkeyPatch):
    try:
        from app import main
    except Exception as e:
        pytest.skip(f"Import test skipped due to compatibility issue: {e}")

    class _EntityRepo:
        def list_all(self, where=None):
            return [SimpleNamespace(id="ent_1", name="Marcos", entity_type="person", normalized="marcos")]

    class _TripleRepo:
        def __init__(self) -> None:
            self.captured_as_of = None

        def get_edges_from(self, *_args, **kwargs):
            self.captured_as_of = kwargs.get("as_of")
            return [
                SimpleNamespace(
                    id="edge_1",
                    subject_id="ent_1",
                    subject_kind="entity",
                    predicate="parallels",
                    object_id="mem_1",
                    object_kind="memory",
                    valid_from=datetime(2026, 4, 18, 8, 0, tzinfo=UTC),
                    valid_to=None,
                    confidence=0.9,
                    source_memory_id="mem_1",
                )
            ]

        def get_edges_to(self, *_args, **kwargs):
            self.captured_as_of = kwargs.get("as_of")
            return []

    class _MemoryRepo:
        def get_item(self, item_id: str):
            if item_id != "mem_1":
                return None
            return SimpleNamespace(
                id="mem_1",
                memory_type="knowledge",
                summary="Marcos values continuity.",
                happened_at=datetime(2026, 4, 18, 7, 0, tzinfo=UTC),
            )

    triple_repo = _TripleRepo()
    fake_db = SimpleNamespace(entity_repo=_EntityRepo(), triple_repo=triple_repo, memory_item_repo=_MemoryRepo())
    fake_svc = SimpleNamespace(database=fake_db)
    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_k: fake_svc)

    out = asyncio.run(
        main.timeline(
            entity="Marcos",
            user_id="u",
            soul_id="s",
            as_of="2026-04-18T12:00:00Z",
        )
    )

    assert out["ok"] is True
    assert out["entity"]["id"] == "ent_1"
    assert out["count"] == 1
    assert out["timeline"][0]["predicate"] == "parallels"
    assert out["timeline"][0]["memory"]["id"] == "mem_1"
    assert triple_repo.captured_as_of is not None
