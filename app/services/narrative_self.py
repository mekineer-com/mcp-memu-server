from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from memu.database.models import Triple

if TYPE_CHECKING:
    from memu.app import MemoryService


def snapshot_previous_narrative_self(
    svc: MemoryService,
    *,
    scope: dict[str, str],
    old_text: str,
    old_embedding: list[float],
) -> str:
    prev_snapshots = svc.database.memory_item_repo.list_items(
        {"memory_type": "narrative_self", **scope}
    )
    prev_snapshot_id = next(iter(prev_snapshots.keys()), None)
    item = svc.database.memory_item_repo.create_item(
        resource_id=None,
        memory_type="narrative_self",
        summary=old_text,
        embedding=old_embedding,
        user_data=scope,
        source_role="soul",
        happened_at=datetime.now(UTC),
    )
    new_id = str(item.id)
    if prev_snapshot_id:
        svc.database.triple_repo.add(
            Triple(
                subject_id=prev_snapshot_id,
                subject_kind="memory",
                predicate="evolved_into",
                object_id=new_id,
                object_kind="memory",
                source_memory_id=new_id,
            ),
            user_data=scope,
        )
    return new_id
