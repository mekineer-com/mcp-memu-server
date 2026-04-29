"""Basic tests for the application."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import main


def test_placeholder():
    assert True


def test_retrieve_method_from_cfg_forces_public_rag():
    assert main._retrieve_method_from_cfg({"retrieve": {"method": "rag"}}) == "rag"
    assert main._retrieve_method_from_cfg({"retrieve": {"method": "llm"}}) == "rag"
    assert main._retrieve_method_from_cfg({"retrieve": {"method": "unknown"}}) == "rag"


def test_retrieve_apimw_enabled_from_cfg_defaults_and_override():
    assert main._retrieve_apimw_enabled_from_cfg(None) is True
    assert main._retrieve_apimw_enabled_from_cfg({"retrieve": {}}) is True
    assert main._retrieve_apimw_enabled_from_cfg({"retrieve": {"apimw_enabled": True}}) is True
    assert main._retrieve_apimw_enabled_from_cfg({"retrieve": {"apimw_enabled": False}}) is False


def test_imports():
    assert hasattr(main, "app")


def test_merge_memorize_batch_results_flattens_top_level_lists():
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
    assert out["pending_episode_ids"] == ["m2", "m1"]
    assert out["skipped_reasons"] == ["skip-a", "skip-b"]
    assert "results" in out
    assert [res["id"] for res in out["resources"]] == ["r1", "r2"]


def test_estimate_unmemorized_tokens_respects_digest_cursor():
    messages = [
        {"content": "one two three"},
        {"content": "four five six"},
        {"content": "seven eight nine"},
    ]
    assert main._estimate_unmemorized_tokens(messages, -1) == main._estimate_tokens(messages)
    assert main._estimate_unmemorized_tokens(messages, 1) == main._estimate_tokens(messages[2:])
    assert main._estimate_unmemorized_tokens(messages, 99) == 0


def test_compact_chat_x_anchors_keeps_two_unique_newest():
    anchors = main._compact_chat_x_anchors("m9", "m8", "m9", "m7")
    assert anchors == ["m9", "m8"]


def test_slice_history_from_chat_x_anchors_uses_two_anchors_and_optional_stop_boundary():
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
    history = [
        {"message_id": "m1", "content": "1"},
        {"message_id": "m2", "content": "2"},
        {"message_id": "m3", "content": "3"},
        {"message_id": "m4", "content": "4"},
        {"message_id": "m5", "content": "5"},
    ]
    sliced = main._slice_history_from_chat_x_anchors(
        history, ["m4"], stop_at_message_id="m2", limit=12
    )
    assert sliced == []
    sliced = main._slice_history_from_chat_x_anchors(
        history, ["m3"], stop_at_message_id="m3", limit=12
    )
    assert sliced == []


def test_slice_history_from_chat_x_anchors_missing_prev_with_stop_returns_empty():
    history = [
        {"message_id": "m1", "content": "1"},
        {"message_id": "m2", "content": "2"},
        {"message_id": "m3", "content": "3"},
    ]
    sliced = main._slice_history_from_chat_x_anchors(
        history, None, stop_at_message_id="m2", limit=12
    )
    assert sliced == []
    sliced = main._slice_history_from_chat_x_anchors(
        history, [], stop_at_message_id="m2", limit=12
    )
    assert sliced == []


def test_build_force_memorize_batches_prefers_segments():
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
        resource_url="/tmp/day-latest.json",
        max_chunk_tokens=999,
    )
    assert [end for _url, _conv, end in batches] == [2, 5]
    assert batches[0][0].endswith("/tmp/days/day1.json")
    assert batches[1][0].endswith("/tmp/days/day2.json")


def test_build_force_memorize_batches_falls_back_to_token_windows():
    merged = [{"content": "one"} for _ in range(7)]
    batches = main._build_force_memorize_batches(
        merged,
        start_idx=0,
        segments=[],
        days_dir=Path("/tmp/days"),
        resource_url="/tmp/day-latest.json",
        max_chunk_tokens=3,
    )
    assert [end for _url, _conv, end in batches] == [2, 5, 6]
    assert all(url.endswith("/tmp/day-latest.json") for url, _conv, _end in batches)


def test_build_force_memorize_batches_fills_segment_gaps_with_resource_url():
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
        resource_url="/tmp/day-latest.json",
        max_chunk_tokens=999,
    )
    assert [end for _url, _conv, end in batches] == [1, 3, 5]
    assert batches[0][0].endswith("/tmp/days/day1.json")
    assert batches[1][0].endswith("/tmp/day-latest.json")
    assert batches[2][0].endswith("/tmp/days/day2.json")


def test_normalize_conversation_uses_created_at_when_timestamp_missing():
    conv = [{"role": "user", "content": "hello", "created_at": "2026-04-16T12:00:00Z"}]
    out = main._normalize_conversation(conv)

    assert isinstance(out, list) and out
    assert out[0]["ts_ms"] == int(datetime(2026, 4, 16, 12, 0, tzinfo=UTC).timestamp() * 1000)


def test_parse_as_of_datetime_accepts_iso_date_and_datetime():
    date_only = main._parse_as_of_datetime("2026-04-18")
    assert date_only is not None
    assert date_only.isoformat().startswith("2026-04-18T00:00:00")

    zulu = main._parse_as_of_datetime("2026-04-18T10:15:00Z")
    assert zulu is not None
    assert zulu.tzinfo is not None
    assert zulu.isoformat().startswith("2026-04-18T10:15:00")


def test_parse_as_of_datetime_rejects_invalid():
    with pytest.raises(main.HTTPException):
        main._parse_as_of_datetime("not-a-date")


@pytest.mark.asyncio
async def test_run_memorize_batches_clears_progress_on_exception():
    class _FailingService:
        async def memorize(self, **_kwargs):
            raise RuntimeError("boom")

    user_id = "u"
    soul_id = "s"
    key = main._memorize_lock_key(user_id, soul_id)
    main._MEMORIZE_PROGRESS.pop(key, None)
    main._MEMORIZE_CANCEL.discard(key)

    with pytest.raises(RuntimeError):
        await main._run_memorize_batches(
            memorize_batches=[("/tmp/day.json", [{"role": "user", "content": "x"}], 0)],
            svc=_FailingService(),
            scope={"user_id": user_id, "soul_id": soul_id},
            conversation_id=None,
            soul_id=soul_id,
            uid=user_id,
            processed_cursor=-1,
            safe={},
            resource_url="/tmp/day.json",
            chat_file=None,
            resource_url_in=None,
            chat_key=None,
            chat_key_source=None,
            tz_name=None,
            prev_len=0,
            merged_len=1,
            force=True,
            days_written=0,
            sleep_stats=None,
        )

    assert key not in main._MEMORIZE_PROGRESS
    assert key not in main._MEMORIZE_CANCEL


def test_timeline_endpoint_returns_entity_edges(monkeypatch: pytest.MonkeyPatch):
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


def test_validate_relationship_speaker_id_rejects_reserved_prefixes():
    assert main._validate_relationship_speaker_id("entity:brother") == "brother"
    with pytest.raises(main.HTTPException):
        main._validate_relationship_speaker_id("user:marcos")
    with pytest.raises(main.HTTPException):
        main._validate_relationship_speaker_id("entity:brother!")


def test_relationship_item_from_values_filters_non_declared_or_inactive():
    item = main._relationship_item_from_values(
        normalized="brother",
        name="Brother",
        entity_type="person",
        properties={"origin": "user_declared", "relationship": "sibling", "active": True},
    )
    assert item is not None
    assert item["speaker_id"] == "entity:brother"
    assert item["relationship"] == "sibling"

    assert main._relationship_item_from_values(
        normalized="brother",
        name="Brother",
        entity_type="person",
        properties={"origin": "extracted", "relationship": "sibling", "active": True},
    ) is None
    assert main._relationship_item_from_values(
        normalized="brother",
        name="Brother",
        entity_type="person",
        properties={"origin": "user_declared", "active": False},
    ) is None


def test_assert_user_declared_relationship_is_strict():
    main._assert_user_declared_relationship({"origin": "user_declared"})
    with pytest.raises(main.HTTPException):
        main._assert_user_declared_relationship({"origin": ""})
    with pytest.raises(main.HTTPException):
        main._assert_user_declared_relationship({"origin": "extracted"})
