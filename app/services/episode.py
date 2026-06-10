from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memu.app import MemoryService


def parse_episode_range(episode_id: str) -> tuple[int, int]:
    text = str(episode_id or "").strip()
    if not text or ":" not in text:
        raise ValueError(f"invalid episode_id: {episode_id}")
    range_part = text.rsplit(":", 1)[1]
    if "-" not in range_part:
        raise ValueError(f"invalid episode_id: {episode_id}")
    start_text, end_text = range_part.split("-", 1)
    try:
        start_idx = int(start_text)
        end_idx = int(end_text)
    except (TypeError, ValueError):
        raise ValueError(f"invalid episode_id: {episode_id}") from None
    if end_idx < start_idx:
        raise ValueError(f"invalid episode_id: {episode_id}")
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


def format_episode_excerpt(
    messages: list[dict[str, Any]],
    *,
    episode_id: str,
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
    lines: list[str] = []
    if time_range:
        lines.append(time_range)
    for idx in range(start, end + 1):
        msg = messages[idx]
        if not isinstance(msg, dict):
            continue
        speaker = str(msg.get("name") or msg.get("role") or "unknown").strip() or "unknown"
        content = " ".join(str(msg.get("content") or "").splitlines()).strip()
        if not content:
            continue
        lines.append(f"[{speaker}] {content}")

    return "\n".join(lines).strip()


def build_episode_inputs(
    messages: list[dict[str, Any]],
    episode_ids: list[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        start_idx, end_idx = parse_episode_range(episode_id)
        if not messages:
            continue
        start = max(0, start_idx)
        end = min(len(messages) - 1, end_idx)
        if start > end:
            continue
        excerpt = format_episode_excerpt(
            messages,
            episode_id=episode_id,
            start_idx=start,
            end_idx=end,
        )
        if not excerpt:
            continue
        msg = messages[start]
        happened_at = _message_happened_at(msg) if isinstance(msg, dict) else None
        out.append(
            {
                "episode_id": episode_id,
                "start_idx": start,
                "end_idx": end,
                "excerpt": excerpt,
                "happened_at": happened_at,
            }
        )
    out.sort(key=lambda row: (int(row.get("start_idx") or 0), int(row.get("end_idx") or 0), str(row.get("episode_id") or "")))
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
