from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from app.services.conversation_id import canonical_conversation_id
from app.services.payload import message_ts_ms

if TYPE_CHECKING:
    from memu.app import MemoryService


def parse_segment_range(segment_id: str) -> tuple[int, int]:
    text = str(segment_id or "").strip()
    if not text or ":" not in text:
        raise ValueError(f"invalid segment_id: {segment_id}")
    range_part = text.rsplit(":", 1)[1]
    if "-" not in range_part:
        raise ValueError(f"invalid segment_id: {segment_id}")
    start_text, end_text = range_part.split("-", 1)
    try:
        start_idx = int(start_text)
        end_idx = int(end_text)
    except (TypeError, ValueError):
        raise ValueError(f"invalid segment_id: {segment_id}") from None
    if end_idx < start_idx:
        raise ValueError(f"invalid segment_id: {segment_id}")
    return start_idx, end_idx


def _message_happened_at(msg: dict[str, Any]) -> datetime | None:
    ts_ms = message_ts_ms(msg)
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000.0, UTC)


def build_segment_inputs(
    messages: list[dict[str, Any]],
    segment_ids: list[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for segment_id in segment_ids:
        start_idx, end_idx = parse_segment_range(segment_id)
        if not messages:
            continue
        start = max(0, start_idx)
        end = min(len(messages) - 1, end_idx)
        if start > end:
            continue
        msg = messages[start]
        happened_at = _message_happened_at(msg) if isinstance(msg, dict) else None
        out.append(
            {
                "segment_id": segment_id,
                "start_idx": start,
                "end_idx": end,
                "happened_at": happened_at,
            }
        )
    out.sort(key=lambda row: (int(row.get("start_idx") or 0), int(row.get("end_idx") or 0), str(row.get("segment_id") or "")))
    return out


def create_companion_memory(
    svc: MemoryService,
    *,
    user_id: str,
    soul_id: str,
    conversation_id: str,
    summary: str,
    embedding: list[float],
    happened_at: datetime | None,
) -> str:
    conversation_id = canonical_conversation_id(conversation_id)
    item = svc.database.memory_item_repo.create_item(
        resource_id=None,
        memory_type="reflection",
        source_role="soul",
        summary=summary,
        embedding=embedding,
        user_data={"user_id": user_id, "soul_id": soul_id, "conversation_id": conversation_id},
        conversation_id=conversation_id,
        happened_at=happened_at,
    )
    return str(item.id)
