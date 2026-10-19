"""Consolidation write/invalidate helpers for already-validated edges.

Confidence convention:
- If no AI judged the confidence, the field is left NULL.  No fallback.  Mechanical
  triples (``mentions``, ``evolved_into``) and LLM-judgment triples (the five
  ALLOWED_EDGE_PREDICATES below) where the model omitted ``confidence`` both land
  here.  Absence is signal: ``NULL`` means ``not assigned by AI``.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memu.database.models import Triple

ALLOWED_EDGE_PREDICATES = {"caused_by", "evokes", "conflicts_with", "parallels", "shaped_by"}


def write_memory_edges(triple_repo: Any, payload: Any, *, scope: Mapping[str, Any], session: Any | None = None) -> int:
    for edge in payload:
        subject_id = edge["subject_id"]
        triple_repo.add(
            Triple(
                subject_id=subject_id,
                subject_kind="memory",
                predicate=edge["predicate"],
                object_id=edge["object_id"],
                object_kind="memory",
                confidence=edge.get("confidence"),
                source_memory_id=subject_id,
            ),
            user_data=scope,
            session=session,
        )
    return len(payload)


def invalidate_memory_edges(triple_repo: Any, payload: Any, *, scope: Mapping[str, Any], session: Any | None = None) -> int:
    for edge in payload:
        triple_repo.invalidate(edge["subject_id"], edge["predicate"], edge["object_id"], scope=scope, session=session)
    return len(payload)
