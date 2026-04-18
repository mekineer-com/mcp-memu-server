"""Shared edge normalization + write/invalidate helpers for APImw and consolidation.

Confidence convention:
- Mechanical triples (``mentions``, ``evolved_into``) are written directly by memorize.py
  without an explicit confidence value, so they land at Triple's model default of 1.0
  (certain by construction — no LLM judgment involved).
- LLM-judgment triples (the five ALLOWED_EDGE_PREDICATES below) are written via
  ``_normalize_edges``.  When the LLM omits confidence the fallback is 0.8, reflecting
  that the relationship is inferred rather than directly observed.
The two tiers should stay separate; do NOT unify the defaults.
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
        confidence_raw = entry.get("confidence", 0.8)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.8
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
