from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from memu.utils.conversation import format_grouped_chat_history, format_relative_time_label

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
    ts_ms = msg.get("ts_ms")
    if not isinstance(ts_ms, (int, float)):
        return None
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, UTC)


def _segment_conversation_id(segment_id: str) -> str:
    text = str(segment_id or "").strip()
    return text.rsplit(":", 1)[0] if ":" in text else text


def format_segment_excerpt(
    messages: list[dict[str, Any]],
    *,
    segment_id: str,
    start_idx: int,
    end_idx: int,
    soul_name: str | None = None,
) -> str:
    if not messages:
        return ""
    start = max(0, start_idx)
    end = min(len(messages) - 1, end_idx)
    if start > end:
        return ""

    excerpt_messages: list[dict[str, Any]] = []
    fallback_conversation_id = _segment_conversation_id(segment_id)
    for idx in range(start, end + 1):
        msg = messages[idx]
        if not isinstance(msg, dict):
            continue
        content = " ".join(str(msg.get("content") or "").splitlines()).strip()
        if not content:
            continue
        row = {**msg, "content": content}
        conversation_id = str(row.get("source_conversation_id") or row.get("conversation_id") or fallback_conversation_id).strip()
        if conversation_id:
            row["conversation_id"] = conversation_id
        timestamp = row.get("received_at") or row.get("ts_ms") or row.get("created_at")
        if timestamp and "received_at" not in row:
            row["received_at"] = timestamp
        excerpt_messages.append(row)
    return format_grouped_chat_history(
        excerpt_messages,
        time_label_resolver=format_relative_time_label,
        soul_name=soul_name,
    )


def build_segment_inputs(
    messages: list[dict[str, Any]],
    segment_ids: list[str],
    *,
    soul_name: str | None = None,
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
        excerpt = format_segment_excerpt(
            messages,
            segment_id=segment_id,
            start_idx=start,
            end_idx=end,
            soul_name=soul_name,
        )
        if not excerpt:
            continue
        msg = messages[start]
        happened_at = _message_happened_at(msg) if isinstance(msg, dict) else None
        out.append(
            {
                "segment_id": segment_id,
                "start_idx": start,
                "end_idx": end,
                "excerpt": excerpt,
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
