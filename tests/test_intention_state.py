import pytest

from app.services.intention_state import (
    apply_intention_action,
    apply_intention_turn_maintenance,
    drop_unpromoted_ephemeral_intentions,
    format_intentions_for_prompt,
    normalize_intentions_stack,
    normalize_memory_cache,
    remove_intentions,
    upsert_intentions_stack_entries,
)


def _items_by_id(stack: dict):
    return {item["id"]: item for item in stack.get("items") or [] if isinstance(item, dict)}


def test_normalize_intentions_stack_from_legacy_id_list():
    stack = normalize_intentions_stack(["i-1", "i-2"])
    items = _items_by_id(stack)

    assert stack["version"] == 1
    assert "relax" in items
    assert items["relax"]["priority"] == 5.0
    assert items["i-1"]["priority"] == 10.0
    assert items["i-2"]["ephemeral"] is False


def test_apply_maintenance_keeps_priority_and_ephemeral_until_consolidation():
    stack = normalize_intentions_stack(
        {
            "turn_index": 0,
            "items": [
                {"id": "task-a", "text": "Task A", "priority": 8.0, "ephemeral": False},
                {"id": "temp-b", "text": "Temp B", "priority": 9.0, "ephemeral": True},
            ],
        }
    )

    maintained_turn_1 = apply_intention_turn_maintenance(stack)
    items_turn_1 = _items_by_id(maintained_turn_1)

    assert maintained_turn_1["turn_index"] == 1
    assert "temp-b" in items_turn_1
    assert items_turn_1["task-a"]["priority"] == 8.0

    maintained_turn_2 = apply_intention_turn_maintenance(maintained_turn_1)
    items_turn_2 = _items_by_id(maintained_turn_2)
    assert maintained_turn_2["turn_index"] == 2
    assert "temp-b" in items_turn_2


def test_upsert_intention_entries_keeps_highest_priority():
    base = normalize_intentions_stack(
        {
            "items": [
                {"id": "task-a", "text": "Task A", "priority": 8.0, "ephemeral": False},
            ]
        }
    )
    updated = upsert_intentions_stack_entries(
        base,
        [
            {"id": "task-a", "text": "Task A refined", "priority": 7.0},
            {"id": "task-b", "text": "Task B", "priority": 10.0},
        ],
    )
    items = _items_by_id(updated)

    assert items["task-a"]["priority"] == 8.0
    assert items["task-a"]["text"] == "Task A refined"
    assert items["task-b"]["priority"] == 10.0


def test_normalize_memory_cache_caps_size_and_entry_length():
    cache = normalize_memory_cache(["x" * 700 for _ in range(12)])
    assert len(cache) == 5
    assert all(len(entry) == 600 for entry in cache)


def test_apply_intention_action_create_then_promote():
    base = normalize_intentions_stack({"turn_index": 3, "items": []})
    created = apply_intention_action(
        base,
        {
            "type": "create",
            "entries": [{"id": "ep-1", "text": "maybe ask follow-up"}],
        },
    )
    created_items = _items_by_id(created)
    assert created_items["ep-1"]["ephemeral"] is True

    promoted = apply_intention_action(created, {"type": "promote", "target_id": "ep-1"})
    promoted_items = _items_by_id(promoted)
    assert promoted_items["ep-1"]["ephemeral"] is False
    assert promoted_items["ep-1"]["priority"] >= 10.0
    assert "expires_at_turn" not in promoted_items["ep-1"]


def test_drop_unpromoted_ephemeral_intentions_keeps_promoted_targets_only():
    stack = normalize_intentions_stack(
        {
            "items": [
                {"id": "stable", "text": "Stable", "priority": 8.0, "ephemeral": False},
                {"id": "keep-me", "text": "Keep me", "ephemeral": True},
                {"id": "drop-me", "text": "Drop me", "ephemeral": True},
            ]
        }
    )
    updated = drop_unpromoted_ephemeral_intentions(stack, {"keep-me"})
    items = _items_by_id(updated)
    assert "stable" in items
    assert "keep-me" in items
    assert "drop-me" not in items


def test_remove_intentions():
    stack = normalize_intentions_stack(
        {
            "items": [
                {"id": "a", "text": "a", "priority": 9},
                {"id": "b", "text": "b", "priority": 8},
            ]
        }
    )
    updated = remove_intentions(stack, ["b"])
    items = _items_by_id(updated)
    assert "b" not in items
    assert "a" in items


def test_maintenance_keeps_low_priority_while_decay_is_disabled():
    stack = normalize_intentions_stack(
        {
            "turn_index": 0,
            "decay_per_turn": 0.1,
            "items": [
                {"id": "low", "text": "fading", "priority": 0.05, "ephemeral": False},
                {"id": "high", "text": "strong", "priority": 5.0, "ephemeral": False},
            ],
        }
    )
    result = apply_intention_turn_maintenance(stack)
    items = _items_by_id(result)
    assert "low" in items
    assert "high" in items
    assert items["low"]["priority"] == pytest.approx(0.05)
    assert items["high"]["priority"] == pytest.approx(5.0)


def test_format_intentions_for_prompt_default_max_is_7():
    stack = normalize_intentions_stack(
        {
            "items": [
                {"id": f"t-{i}", "text": f"task {i}", "priority": float(10 - i), "ephemeral": False}
                for i in range(10)
            ]
        }
    )
    text = format_intentions_for_prompt(stack)
    # relax is always prepended by normalize; the non-relax items are t-0..t-9 (10 items)
    # default max_items=7 means at most 7 lines total (including relax if present)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) <= 7
