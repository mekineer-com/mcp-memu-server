from app.services.intention_state import (
    apply_intention_turn_maintenance,
    normalize_intention_stack,
    normalize_memory_cache,
    upsert_intention_stack_entries,
)


def _items_by_id(stack: dict):
    return {item["id"]: item for item in stack.get("items") or [] if isinstance(item, dict)}


def test_normalize_intention_stack_from_legacy_id_list():
    stack = normalize_intention_stack(["i-1", "i-2"])
    items = _items_by_id(stack)

    assert stack["version"] == 1
    assert "relax" in items
    assert items["relax"]["priority"] == 5.0
    assert items["i-1"]["priority"] == 10.0
    assert items["i-2"]["ephemeral"] is False


def test_apply_maintenance_decays_and_drops_ephemeral():
    stack = normalize_intention_stack(
        {
            "items": [
                {"id": "task-a", "text": "Task A", "priority": 8.0, "ephemeral": False},
                {"id": "temp-b", "text": "Temp B", "priority": 9.0, "ephemeral": True},
            ]
        }
    )

    maintained = apply_intention_turn_maintenance(stack)
    items = _items_by_id(maintained)

    assert "temp-b" not in items
    assert items["task-a"]["priority"] == 7.9


def test_upsert_intention_entries_keeps_highest_priority():
    base = normalize_intention_stack(
        {
            "items": [
                {"id": "task-a", "text": "Task A", "priority": 8.0, "ephemeral": False},
            ]
        }
    )
    updated = upsert_intention_stack_entries(
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
    assert len(cache) == 7
    assert all(len(entry) == 300 for entry in cache)
