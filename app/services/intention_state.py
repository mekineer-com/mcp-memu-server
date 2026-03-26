from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

RELAX_INTENTION_ID = "relax"
RELAX_INTENTION_TEXT = "Relax"
DEFAULT_RELAX_PRIORITY = 5.0
DEFAULT_DECAY_PER_TURN = 0.1
DEFAULT_INTENTION_PRIORITY = 10.0
MAX_MEMORY_CACHE_ENTRIES = 7
MAX_MEMORY_CACHE_ENTRY_CHARS = 300


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _text(value: Any) -> str:
    return str(value or "").strip()


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _build_relax_item(relax_priority: float) -> dict[str, Any]:
    return {
        "id": RELAX_INTENTION_ID,
        "text": RELAX_INTENTION_TEXT,
        "priority": float(relax_priority),
        "ephemeral": False,
        "kind": "relax",
        "status": "active",
        "active": True,
    }


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


def _normalize_stack_item(raw: Any, *, default_priority: float, now_iso: str) -> dict[str, Any] | None:
    if isinstance(raw, str):
        text = _text(raw)
        if not text:
            return None
        return {
            "id": text,
            "text": text,
            "priority": float(default_priority),
            "ephemeral": False,
            "kind": "intention",
            "status": "active",
            "source_intention_id": text,
            "created_at": now_iso,
            "updated_at": now_iso,
        }

    if not isinstance(raw, dict):
        return None

    item_id = _text(raw.get("id") or raw.get("intention_id") or raw.get("source_intention_id") or raw.get("text"))
    if not item_id:
        return None

    text = _text(raw.get("text") or raw.get("description") or raw.get("name") or item_id)
    kind = _text(raw.get("kind") or "intention") or "intention"
    is_relax = item_id.lower() == RELAX_INTENTION_ID or kind == "relax"

    item: dict[str, Any] = {
        "id": RELAX_INTENTION_ID if is_relax else item_id,
        "text": RELAX_INTENTION_TEXT if is_relax else text,
        "priority": _float(raw.get("priority"), default_priority),
        "ephemeral": bool(raw.get("ephemeral") is True),
        "kind": "relax" if is_relax else "intention",
        "status": _text(raw.get("status") or "active") or "active",
    }

    source_intention_id = _text(raw.get("source_intention_id") or raw.get("intention_id"))
    if not is_relax and source_intention_id:
        item["source_intention_id"] = source_intention_id

    created_at = _text(raw.get("created_at"))
    updated_at = _text(raw.get("updated_at"))
    if not is_relax:
        item["created_at"] = created_at or now_iso
        item["updated_at"] = updated_at or now_iso

    return item


def normalize_intention_stack(
    value: Any,
    *,
    default_priority: float = DEFAULT_INTENTION_PRIORITY,
    default_decay_per_turn: float = DEFAULT_DECAY_PER_TURN,
    default_relax_priority: float = DEFAULT_RELAX_PRIORITY,
    now_iso: str | None = None,
) -> dict[str, Any]:
    now = now_iso or _now_iso()

    if isinstance(value, dict):
        raw_items = value.get("items") if isinstance(value.get("items"), list) else []
        decay_per_turn = _float(value.get("decay_per_turn"), default_decay_per_turn)
        relax_priority = _float(value.get("relax_priority"), default_relax_priority)
    elif isinstance(value, list):
        raw_items = value
        decay_per_turn = float(default_decay_per_turn)
        relax_priority = float(default_relax_priority)
    else:
        raw_items = []
        decay_per_turn = float(default_decay_per_turn)
        relax_priority = float(default_relax_priority)

    by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_items:
        item = _normalize_stack_item(raw, default_priority=default_priority, now_iso=now)
        if item is None:
            continue
        by_id[item["id"]] = item

    by_id[RELAX_INTENTION_ID] = _build_relax_item(relax_priority)

    relax_item = by_id[RELAX_INTENTION_ID]
    others = [item for key, item in by_id.items() if key != RELAX_INTENTION_ID]
    others.sort(key=lambda item: _float(item.get("priority"), 0.0), reverse=True)

    for item in others:
        item["active"] = _float(item.get("priority"), 0.0) >= _float(relax_item.get("priority"), default_relax_priority)

    return {
        "version": 1,
        "decay_per_turn": float(decay_per_turn),
        "relax_priority": _float(relax_item.get("priority"), default_relax_priority),
        "items": [relax_item, *others],
    }


def apply_intention_turn_maintenance(
    value: Any,
    *,
    decay_per_turn: float | None = None,
) -> dict[str, Any]:
    stack = normalize_intention_stack(value)
    relax_priority = _float(stack.get("relax_priority"), DEFAULT_RELAX_PRIORITY)
    decay = _float(decay_per_turn, _float(stack.get("decay_per_turn"), DEFAULT_DECAY_PER_TURN))
    now = _now_iso()

    kept: list[dict[str, Any]] = []
    for item in stack.get("items") or []:
        if not isinstance(item, dict):
            continue
        if _text(item.get("id")) == RELAX_INTENTION_ID or _text(item.get("kind")) == "relax":
            continue
        if item.get("ephemeral") is True:
            continue
        next_item = dict(item)
        next_item["priority"] = max(0.0, _float(item.get("priority"), 0.0) - decay)
        next_item["updated_at"] = now
        kept.append(next_item)

    return normalize_intention_stack(
        {
            "decay_per_turn": decay,
            "relax_priority": relax_priority,
            "items": kept,
        },
        now_iso=now,
    )


def upsert_intention_stack_entries(
    stack_value: Any,
    entries: list[dict[str, Any]],
    *,
    default_priority: float = DEFAULT_INTENTION_PRIORITY,
) -> dict[str, Any]:
    stack = normalize_intention_stack(stack_value)
    now = _now_iso()
    by_id: dict[str, dict[str, Any]] = {
        _text(item.get("id")): dict(item)
        for item in stack.get("items") or []
        if isinstance(item, dict) and _text(item.get("id")) and _text(item.get("id")) != RELAX_INTENTION_ID
    }

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        item_id = _text(entry.get("id") or entry.get("intention_id"))
        if not item_id:
            continue
        text = _text(entry.get("text") or entry.get("description") or item_id)
        priority = _float(entry.get("priority"), default_priority)
        ephemeral = bool(entry.get("ephemeral") is True)

        current = by_id.get(item_id)
        if current is None:
            current = {
                "id": item_id,
                "text": text,
                "priority": priority,
                "ephemeral": ephemeral,
                "kind": "intention",
                "status": "active",
                "source_intention_id": item_id,
                "created_at": now,
                "updated_at": now,
            }
        else:
            current["text"] = text or _text(current.get("text")) or item_id
            current["priority"] = max(_float(current.get("priority"), 0.0), priority)
            current["ephemeral"] = ephemeral
            current["updated_at"] = now
        by_id[item_id] = current

    return normalize_intention_stack(
        {
            "decay_per_turn": stack.get("decay_per_turn"),
            "relax_priority": stack.get("relax_priority"),
            "items": [*by_id.values()],
        },
        now_iso=now,
    )
