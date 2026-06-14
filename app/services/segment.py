from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from memu.utils.conversation import render_grouped_chat_messages

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


def _format_time_range(start_msg: dict[str, Any], end_msg: dict[str, Any]) -> str | None:
    start_dt = _message_happened_at(start_msg) if isinstance(start_msg, dict) else None
    end_dt = _message_happened_at(end_msg) if isinstance(end_msg, dict) else None
    if not start_dt:
        return None
    start_str = start_dt.strftime("%b %d, %H:%M")
    if end_dt and end_dt != start_dt:
        if start_dt.date() == end_dt.date():
            return f"{start_str}\u2013{end_dt.strftime('%H:%M')}"
        return f"{start_str} \u2013 {end_dt.strftime('%b %d, %H:%M')}"
    return start_str


def format_segment_excerpt(
    messages: list[dict[str, Any]],
    *,
    segment_id: str,
    start_idx: int,
    end_idx: int,
) -> str:
    if not messages:
        return ""
    start = max(0, start_idx)
    end = min(len(messages) - 1, end_idx)
    if start > end:
        return ""

    time_range = _format_time_range(messages[start], messages[end])
    conversation_id = str(segment_id or "").rsplit(":", 1)[0]
    excerpt_messages: list[dict[str, Any]] = []
    for idx in range(start, end + 1):
        msg = messages[idx]
        if not isinstance(msg, dict):
            continue
        content = " ".join(str(msg.get("content") or "").splitlines()).strip()
        if not content:
            continue
        excerpt_messages.append(
            {
                **msg,
                "content": content,
                "conversation_id": msg.get("conversation_id") or conversation_id,
            }
        )

    chat_block = render_grouped_chat_messages(
        excerpt_messages,
        default_conversation_id=conversation_id,
        current_conversation_id=conversation_id,
        current_marker="",
        default_role="unknown",
    )
    return "\n".join(part for part in (time_range, chat_block) if part).strip()


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
        excerpt = format_segment_excerpt(
            messages,
            segment_id=segment_id,
            start_idx=start,
            end_idx=end,
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
