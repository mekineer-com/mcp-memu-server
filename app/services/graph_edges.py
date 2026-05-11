"""Shared edge normalization + write/invalidate helpers for APImw and consolidation.

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


def _normalize_edges(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        subject_id = str(entry.get("subject_id") or "").strip()
        predicate = str(entry.get("predicate") or "").strip()
        object_id = str(entry.get("object_id") or "").strip()
        if not subject_id or not object_id or predicate not in ALLOWED_EDGE_PREDICATES:
            continue
        confidence_raw = entry.get("confidence")
        if confidence_raw is None:
            confidence: float | None = None
        else:
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                confidence = None
        out.append(
            {
                "subject_id": subject_id,
                "predicate": predicate,
                "object_id": object_id,
                "confidence": confidence,
            }
        )
    return out


def write_memory_edges(triple_repo: Any, payload: Any, *, scope: Mapping[str, Any]) -> int:
    edges = _normalize_edges(payload)
    for edge in edges:
        subject_id = edge["subject_id"]
        triple_repo.add(
            Triple(
                subject_id=subject_id,
                subject_kind="memory",
                predicate=edge["predicate"],
                object_id=edge["object_id"],
                object_kind="memory",
                confidence=edge["confidence"],
                source_memory_id=subject_id,
            ),
            user_data=scope,
        )
    return len(edges)


def invalidate_memory_edges(triple_repo: Any, payload: Any, *, scope: Mapping[str, Any]) -> int:
    edges = _normalize_edges(payload)
    for edge in edges:
        triple_repo.invalidate(edge["subject_id"], edge["predicate"], edge["object_id"], scope=scope)
    return len(edges)
