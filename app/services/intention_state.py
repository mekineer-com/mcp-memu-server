from __future__ import annotations

import re
from typing import Any

MAX_INTENTIONS = 5
MAX_MEMORY_CACHE_ENTRIES = 7
MAX_MEMORY_CACHE_ENTRY_CHARS = 600
_INTENTION_ID_SANITIZE_RE = re.compile(r"[^a-z0-9]+")


def _text(value: Any) -> str:
    return str(value or "").strip()


def normalize_memory_cache(
    value: Any,
    *,
    max_entries: int = MAX_MEMORY_CACHE_ENTRIES,
    max_chars: int = MAX_MEMORY_CACHE_ENTRY_CHARS,
) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, dict):
            text = _text(item.get("text"))
        else:
            text = _text(item)
        if not text:
            continue
        out.append(text[:max_chars])
    if max_entries <= 0:
        return []
    return out[-max_entries:]


def append_memory_cache_entry(
    cache: Any,
    entry: Any,
    *,
    max_entries: int = MAX_MEMORY_CACHE_ENTRIES,
    max_chars: int = MAX_MEMORY_CACHE_ENTRY_CHARS,
) -> list[str]:
    items = normalize_memory_cache(cache, max_entries=max_entries, max_chars=max_chars)
    text = _text(entry)
    if not text:
        return items
    items.append(text[:max_chars])
    return items[-max_entries:]


def slugify_intention_id(value: Any) -> str:
    slug = _INTENTION_ID_SANITIZE_RE.sub("-", _text(value).lower()).strip("-")
    return slug[:64]


def validate_intentions(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("Intentions must be a list")
    intentions: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != {"id", "text"}:
            raise ValueError("Each intention must contain only id and text")
        item_id = _text(raw.get("id"))
        text = _text(raw.get("text"))
        if not item_id or not text:
            raise ValueError("Intention id and text are required")
        if item_id in seen_ids:
            raise ValueError(f"Duplicate intention id: {item_id}")
        seen_ids.add(item_id)
        intentions.append({"id": item_id, "text": text})
    return intentions


def validate_intention_replacement(value: Any) -> list[dict[str, str]]:
    intentions = validate_intentions(value)
    if len(intentions) > MAX_INTENTIONS:
        raise ValueError(f"Intentions cannot exceed {MAX_INTENTIONS}")
    for item in intentions:
        if item["id"] != slugify_intention_id(item["id"]):
            raise ValueError(f"Invalid intention id: {item['id']}")
    return intentions


def remove_intentions(
    value: Any,
    intention_ids: list[str],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    intentions = validate_intentions(value)
    remove_ids = {_text(item_id) for item_id in intention_ids if _text(item_id)}
    removed = [
        {"item": item, "position": position}
        for position, item in enumerate(intentions)
        if item["id"] in remove_ids
    ]
    return [item for item in intentions if item["id"] not in remove_ids], removed


def restore_intentions(value: Any, removed: Any) -> list[dict[str, str]]:
    intentions = validate_intentions(value)
    if not isinstance(removed, list):
        raise ValueError("Removed intentions must be a list")
    existing_ids = {item["id"] for item in intentions}
    records: list[tuple[int, dict[str, str]]] = []
    for raw in removed:
        if not isinstance(raw, dict) or set(raw) != {"item", "position"}:
            raise ValueError("Removed intention requires item and position")
        position = raw.get("position")
        if not isinstance(position, int) or isinstance(position, bool) or position < 0:
            raise ValueError("Removed intention position must be a non-negative integer")
        item = validate_intentions([raw.get("item")])[0]
        if item["id"] in existing_ids:
            continue
        existing_ids.add(item["id"])
        records.append((position, item))
    for position, item in sorted(records, key=lambda record: record[0]):
        intentions.insert(min(position, len(intentions)), item)
    return intentions


def merge_consolidated_intentions(
    snapshot: Any,
    current: Any,
    replacement: Any,
) -> list[dict[str, str]]:
    snapshot_items = validate_intentions(snapshot)
    current_items = validate_intentions(current)
    merged = validate_intention_replacement(replacement)
    snapshot_ids = {item["id"] for item in snapshot_items}
    current_ids = {item["id"] for item in current_items}
    merged = [item for item in merged if item["id"] not in snapshot_ids - current_ids]
    restored = [
        {"item": item, "position": position}
        for position, item in enumerate(current_items)
        if item["id"] not in snapshot_ids
    ]
    return restore_intentions(merged, restored)


def format_intentions_for_prompt(value: Any) -> str:
    return "\n".join(
        f"- {item['id']}: {item['text']}" for item in validate_intentions(value)
    )
