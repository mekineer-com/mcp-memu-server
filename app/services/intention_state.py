from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

RELAX_INTENTION_ID = "relax"
RELAX_INTENTION_TEXT = "Relax"
DEFAULT_RELAX_PRIORITY = 5.0
DEFAULT_DECAY_PER_TURN = 0.1
DEFAULT_INTENTION_PRIORITY = 10.0
DEFAULT_EPHEMERAL_TTL_TURNS = 1
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


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


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


def _normalize_stack_item(
    raw: Any,
    *,
    default_priority: float,
    now_iso: str,
    current_turn: int,
) -> dict[str, Any] | None:
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

    item_id = _text(raw.get("id") or "")
    if not item_id:
        return None

    text = _text(raw.get("text") or item_id)
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

    if is_relax:
        return item

    source_intention_id = _text(raw.get("source_intention_id") or raw.get("intention_id"))
    if source_intention_id:
        item["source_intention_id"] = source_intention_id

    created_at = _text(raw.get("created_at"))
    updated_at = _text(raw.get("updated_at"))
    item["created_at"] = created_at or now_iso
    item["updated_at"] = updated_at or now_iso

    if item.get("ephemeral") is True:
        # Ephemerals survive exactly one subsequent turn unless promoted.
        item["expires_at_turn"] = _int(raw.get("expires_at_turn"), current_turn)

    return item


def normalize_intentions_stack(
    value: Any,
    *,
    default_priority: float = DEFAULT_INTENTION_PRIORITY,
    default_decay_per_turn: float = DEFAULT_DECAY_PER_TURN,
    default_relax_priority: float = DEFAULT_RELAX_PRIORITY,
    now_iso: str | None = None,
) -> dict[str, Any]:
    now = now_iso or _now_iso()

    decay_per_turn = float(default_decay_per_turn)
    relax_priority = float(default_relax_priority)
    turn_index = 0
    if isinstance(value, dict):
        raw_items = value.get("items") if isinstance(value.get("items"), list) else []
        decay_per_turn = _float(value.get("decay_per_turn"), default_decay_per_turn)
        relax_priority = _float(value.get("relax_priority"), default_relax_priority)
        turn_index = max(0, _int(value.get("turn_index"), 0))
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = []

    by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_items:
        item = _normalize_stack_item(raw, default_priority=default_priority, now_iso=now, current_turn=turn_index)
        if item is None:
            continue
        by_id[item["id"]] = item

    by_id[RELAX_INTENTION_ID] = _build_relax_item(relax_priority)

    relax_item = by_id[RELAX_INTENTION_ID]
    others = [item for key, item in by_id.items() if key != RELAX_INTENTION_ID]
    others.sort(key=lambda item: _float(item.get("priority"), 0.0), reverse=True)

    relax_priority_value = _float(relax_item.get("priority"), default_relax_priority)
    for item in others:
        item["active"] = _float(item.get("priority"), 0.0) >= relax_priority_value

    ordered_items = [relax_item, *others]
    ordered_items.sort(key=lambda item: _float(item.get("priority"), 0.0), reverse=True)

    return {
        "version": 1,
        "turn_index": turn_index,
        "decay_per_turn": float(decay_per_turn),
        "relax_priority": relax_priority_value,
        "items": ordered_items,
    }


def apply_intention_turn_maintenance(
    value: Any,
    *,
    decay_per_turn: float | None = None,
) -> dict[str, Any]:
    stack = normalize_intentions_stack(value)
    relax_priority = _float(stack.get("relax_priority"), DEFAULT_RELAX_PRIORITY)
    decay = _float(decay_per_turn, _float(stack.get("decay_per_turn"), DEFAULT_DECAY_PER_TURN))
    now = _now_iso()
    current_turn = max(0, _int(stack.get("turn_index"), 0))
    next_turn = current_turn + 1

    kept: list[dict[str, Any]] = []
    for item in stack.get("items") or []:
        if not isinstance(item, dict):
            continue
        if _text(item.get("id")) == RELAX_INTENTION_ID or _text(item.get("kind")) == "relax":
            continue

        next_item = dict(item)
        if next_item.get("ephemeral") is True:
            expires_at_turn = _int(next_item.get("expires_at_turn"), current_turn)
            if expires_at_turn < next_turn:
                continue
            next_item["expires_at_turn"] = expires_at_turn
            next_item["updated_at"] = now
            kept.append(next_item)
            continue

        next_item["priority"] = max(0.0, _float(item.get("priority"), 0.0) - decay)
        next_item["updated_at"] = now
        kept.append(next_item)

    return normalize_intentions_stack(
        {
            "turn_index": next_turn,
            "decay_per_turn": decay,
            "relax_priority": relax_priority,
            "items": kept,
        },
        now_iso=now,
    )


def upsert_intentions_stack_entries(
    stack_value: Any,
    entries: list[dict[str, Any]],
    *,
    default_priority: float = DEFAULT_INTENTION_PRIORITY,
) -> dict[str, Any]:
    stack = normalize_intentions_stack(stack_value)
    now = _now_iso()
    current_turn = max(0, _int(stack.get("turn_index"), 0))
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

        if ephemeral:
            current["expires_at_turn"] = _int(
                entry.get("expires_at_turn"),
                current_turn + DEFAULT_EPHEMERAL_TTL_TURNS,
            )
        else:
            current.pop("expires_at_turn", None)

        by_id[item_id] = current

    return normalize_intentions_stack(
        {
            "turn_index": current_turn,
            "decay_per_turn": stack.get("decay_per_turn"),
            "relax_priority": stack.get("relax_priority"),
            "items": [*by_id.values()],
        },
        now_iso=now,
    )


def remove_intentions(stack_value: Any, intention_ids: list[str]) -> dict[str, Any]:
    stack = normalize_intentions_stack(stack_value)
    remove_ids = {_text(item_id) for item_id in intention_ids if _text(item_id)}
    if not remove_ids:
        return stack
    kept = [
        item
        for item in (stack.get("items") or [])
        if isinstance(item, dict) and _text(item.get("id")) not in remove_ids
    ]
    return normalize_intentions_stack(
        {
            "turn_index": stack.get("turn_index"),
            "decay_per_turn": stack.get("decay_per_turn"),
            "relax_priority": stack.get("relax_priority"),
            "items": kept,
        }
    )


def apply_intention_action(stack_value: Any, action: Any) -> dict[str, Any]:
    stack = normalize_intentions_stack(stack_value)
    if not isinstance(action, dict):
        return stack

    action_type = _text(action.get("type") or "none").lower()
    if action_type in {"", "none"}:
        return stack

    now = _now_iso()
    current_turn = max(0, _int(stack.get("turn_index"), 0))
    by_id: dict[str, dict[str, Any]] = {
        _text(item.get("id")): dict(item)
        for item in stack.get("items") or []
        if isinstance(item, dict) and _text(item.get("id")) and _text(item.get("id")) != RELAX_INTENTION_ID
    }

    if action_type == "boost":
        target_id = _text(action.get("target_id"))
        if target_id and target_id in by_id:
            amount = _float(action.get("amount"), 1.0)
            current = by_id[target_id]
            current["priority"] = max(0.0, _float(current.get("priority"), 0.0) + amount)
            current["updated_at"] = now
            by_id[target_id] = current

    elif action_type == "promote":
        target_id = _text(action.get("target_id"))
        explicit_text = _text(action.get("text"))
        if target_id:
            current = by_id.get(target_id)
            if current is None:
                current = {
                    "id": target_id,
                    "text": explicit_text or target_id,
                    "priority": DEFAULT_INTENTION_PRIORITY,
                    "ephemeral": False,
                    "kind": "intention",
                    "status": "active",
                    "source_intention_id": target_id,
                    "created_at": now,
                    "updated_at": now,
                }
            else:
                current["ephemeral"] = False
                current["priority"] = max(_float(current.get("priority"), 0.0), DEFAULT_INTENTION_PRIORITY)
                current["updated_at"] = now
                current.pop("expires_at_turn", None)
            if explicit_text:
                current["text"] = explicit_text
            by_id[target_id] = current

    elif action_type == "create":
        entries = action.get("entries")
        if not isinstance(entries, list):
            entries = []
        for raw in entries[:2]:
            if not isinstance(raw, dict):
                continue
            text = _text(raw.get("text"))
            if not text:
                continue
            item_id = _text(raw.get("id")) or f"ephem-{uuid.uuid4().hex[:8]}"
            current = by_id.get(item_id) or {
                "id": item_id,
                "text": text,
                "priority": _float(raw.get("priority"), DEFAULT_INTENTION_PRIORITY),
                "ephemeral": True,
                "kind": "intention",
                "status": "active",
                "source_intention_id": item_id,
                "created_at": now,
                "updated_at": now,
            }
            current["text"] = text
            current["ephemeral"] = True
            current["updated_at"] = now
            current["expires_at_turn"] = current_turn + DEFAULT_EPHEMERAL_TTL_TURNS
            by_id[item_id] = current

    return normalize_intentions_stack(
        {
            "turn_index": current_turn,
            "decay_per_turn": stack.get("decay_per_turn"),
            "relax_priority": stack.get("relax_priority"),
            "items": [*by_id.values()],
        },
        now_iso=now,
    )


def format_intentions_for_prompt(stack_value: Any, *, max_items: int = 12) -> str:
    stack = normalize_intentions_stack(stack_value)
    lines: list[str] = []
    for item in (stack.get("items") or [])[: max(1, int(max_items))]:
        if not isinstance(item, dict):
            continue
        item_id = _text(item.get("id"))
        text = _text(item.get("text"))
        priority = _float(item.get("priority"), 0.0)
        ephemeral = bool(item.get("ephemeral") is True)
        active = bool(item.get("active") is True)
        tags = []
        if item_id == RELAX_INTENTION_ID:
            tags.append("threshold")
        elif ephemeral:
            tags.append("ephemeral")
        else:
            tags.append("active" if active else "inactive")
        tag_suffix = f" [{', '.join(tags)}]" if tags else ""
        lines.append(f"- {item_id}: {text} (p={priority:.1f}){tag_suffix}")
    return "\n".join(lines)
