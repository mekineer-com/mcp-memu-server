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


def _message_utc_iso(msg: dict[str, Any]) -> str | None:
    happened_at = _message_happened_at(msg)
    return happened_at.isoformat() if happened_at is not None else None


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

    start_ts = _message_utc_iso(messages[start]) if isinstance(messages[start], dict) else None
    end_ts = _message_utc_iso(messages[end]) if isinstance(messages[end], dict) else None

    header = f"episode_id={episode_id} | message_index={start}-{end}"
    if start_ts and end_ts:
        header = f"{header} | time_utc={start_ts}..{end_ts}"

    lines: list[str] = [header]
    for idx in range(start, end + 1):
        msg = messages[idx]
        if not isinstance(msg, dict):
            continue
        speaker = str(msg.get("name") or msg.get("role") or "unknown").strip() or "unknown"
        content = " ".join(str(msg.get("content") or "").splitlines()).strip()
        if not content:
            continue
        lines.append(f"[{idx}] [{speaker}]: {content}")

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
