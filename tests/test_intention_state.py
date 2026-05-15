from app.services.intention_state import (
    apply_intention_action,
    apply_intention_turn_maintenance,
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


def test_apply_maintenance_decays_and_ephemeral_expires_next_turn():
    stack = normalize_intentions_stack(
        {
            "turn_index": 0,
            "items": [
                {"id": "task-a", "text": "Task A", "priority": 8.0, "ephemeral": False},
                {"id": "temp-b", "text": "Temp B", "priority": 9.0, "ephemeral": True, "expires_at_turn": 1},
            ],
        }
    )

    maintained_turn_1 = apply_intention_turn_maintenance(stack)
    items_turn_1 = _items_by_id(maintained_turn_1)

    assert maintained_turn_1["turn_index"] == 1
    assert "temp-b" in items_turn_1
    assert items_turn_1["task-a"]["priority"] == 7.9

    maintained_turn_2 = apply_intention_turn_maintenance(maintained_turn_1)
    items_turn_2 = _items_by_id(maintained_turn_2)
    assert maintained_turn_2["turn_index"] == 2
    assert "temp-b" not in items_turn_2


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
    cache = normalize_memory_cache(["x" * 500 for _ in range(12)])
    assert len(cache) == 5
    assert all(len(entry) == 300 for entry in cache)


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
    assert created_items["ep-1"]["expires_at_turn"] == 4

    promoted = apply_intention_action(created, {"type": "promote", "target_id": "ep-1"})
    promoted_items = _items_by_id(promoted)
    assert promoted_items["ep-1"]["ephemeral"] is False
    assert promoted_items["ep-1"]["priority"] >= 10.0
    assert "expires_at_turn" not in promoted_items["ep-1"]


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
