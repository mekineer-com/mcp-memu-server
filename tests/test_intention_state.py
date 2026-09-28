import pytest

from app.services.intention_state import (
    merge_consolidated_intentions,
    normalize_memory_cache,
    remove_intentions,
    restore_memory_cache_after_undo,
    restore_intentions,
    validate_intention_replacement,
    validate_intentions,
)


def test_validate_intentions_preserves_order_and_duplicate_text():
    assert validate_intentions([
        {"id": "first", "text": " Shared words "},
        {"id": "second", "text": "Shared words"},
    ]) == [
        {"id": "first", "text": "Shared words"},
        {"id": "second", "text": "Shared words"},
    ]


@pytest.mark.parametrize(
    "value",
    [
        None,
        {"items": []},
        ["not-an-object"],
        [{"id": "only-id"}],
        [{"id": "same", "text": "One"}, {"id": "same", "text": "Two"}],
    ],
)
def test_validate_intentions_rejects_noncanonical_state(value):
    with pytest.raises(ValueError):
        validate_intentions(value)


def test_validate_replacement_requires_slug_ids_and_caps_fresh_output():
    with pytest.raises(ValueError, match="Invalid intention id"):
        validate_intention_replacement([{"id": "Not: a slug", "text": "A pursuit"}])
    with pytest.raises(ValueError, match="cannot exceed 5"):
        validate_intention_replacement([
            {"id": f"item-{index}", "text": f"Pursuit {index}"}
            for index in range(6)
        ])


def test_remove_and_restore_intentions_use_exact_positions():
    intentions = [
        {"id": "a", "text": "A"},
        {"id": "b", "text": "B"},
        {"id": "c", "text": "C"},
        {"id": "d", "text": "D"},
    ]
    remaining, removed = remove_intentions(intentions, ["b", "d", "missing"])
    assert remaining == [{"id": "a", "text": "A"}, {"id": "c", "text": "C"}]
    assert removed == [
        {"item": {"id": "b", "text": "B"}, "position": 1},
        {"item": {"id": "d", "text": "D"}, "position": 3},
    ]
    assert restore_intentions(remaining, removed) == intentions


def test_restore_skips_id_already_present_in_replacement():
    current = [{"id": "restored", "text": "Model wording"}]
    removed = [{"item": {"id": "restored", "text": "Old wording"}, "position": 0}]
    assert restore_intentions(current, removed) == current


def test_merge_removes_intention_annulled_while_consolidation_ran():
    snapshot = [{"id": "keep", "text": "Keep"}, {"id": "done", "text": "Done"}]
    current = [{"id": "keep", "text": "Keep"}]
    replacement = [
        {"id": "done", "text": "Done"},
        {"id": "new", "text": "New"},
    ]
    assert merge_consolidated_intentions(snapshot, current, replacement) == [
        {"id": "new", "text": "New"}
    ]


def test_merge_preserves_concurrent_undo_without_duplicate_id():
    snapshot = [{"id": "old", "text": "Old"}]
    current = [
        {"id": "old", "text": "Old"},
        {"id": "restored", "text": "Restored"},
    ]
    replacement = [
        {"id": "fresh", "text": "Fresh"},
        {"id": "restored", "text": "Model wording"},
    ]
    assert merge_consolidated_intentions(snapshot, current, replacement) == replacement


def test_merge_may_temporarily_exceed_cap_only_for_concurrent_restore():
    replacement = [
        {"id": f"item-{index}", "text": f"Pursuit {index}"}
        for index in range(5)
    ]
    current = [{"id": "restored", "text": "Restored"}]
    merged = merge_consolidated_intentions([], current, replacement)
    assert len(merged) == 6
    assert merged[0] == {"id": "restored", "text": "Restored"}


def test_normalize_memory_cache_caps_size_and_entry_length():
    cache = normalize_memory_cache(["x" * 700 for _ in range(12)])
    assert len(cache) == 7
    assert all(len(entry) == 600 for entry in cache)


def test_cache_undo_preserves_later_entries_and_restores_the_right_eviction():
    before = ["a", "b", "c", "d", "e", "f", "g"]
    after_turn = ["b", "c", "d", "e", "f", "g", "turn"]
    after_wake = ["c", "d", "e", "f", "g", "turn", "wake"]

    assert restore_memory_cache_after_undo(before, after_wake, "turn") == [
        "b", "c", "d", "e", "f", "g", "wake"
    ]
    assert restore_memory_cache_after_undo(before, after_turn, "turn") == before
