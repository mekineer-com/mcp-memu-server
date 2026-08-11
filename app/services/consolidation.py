from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

log = logging.getLogger(__name__)

from fastapi import HTTPException
from memu.app.dossier import label_sections, render_memory_records, revision_status_items
from memu.app.dossier_revision import parse_anchor_revisions, parse_dossier_revision_batch
from memu.prompts.consolidation import anchors as anchors_prompt
from memu.prompts.consolidation import dossiers as dossiers_prompt

from app.services.segment import (
    build_segment_inputs,
    create_companion_memory,
)
from app.services.xml_utils import extract_xml_fragment, xml_text
from app.services.graph_edges import (
    invalidate_memory_edges,
    write_memory_edges,
)
from app.services.intention_state import (
    RELAX_INTENTION_ID,
    format_intentions_for_prompt,
)
from app.services.narrative_self import snapshot_previous_narrative_self
from app.services import soul_summaries as _soul_summaries
from app.services.payload import parse_iso_datetime
from app.services import soul_state as _soul_state
from app.services.turn_contract import DEFAULT_SOUL_CARD, format_memory_legend, format_memory_line, format_shaped_by_line, format_relative_time_label

if TYPE_CHECKING:
    from memu.app import MemoryService


_SEGMENT_SUFFIX_RE = re.compile(r"_(\d+)\.json$")


def _segment_file_sort_key(path: Path) -> tuple[str, int]:
    stem = path.stem
    m = _SEGMENT_SUFFIX_RE.search(path.name)
    if m:
        return (path.name[:m.start()], int(m.group(1)))
    return (stem, 0)


_HEX_MEMORY_ID_RE = re.compile(r"^[0-9a-f]{8}$|^[0-9a-f]{16}$|^[0-9a-f]{32}$", re.IGNORECASE)
_UUID_MEMORY_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_INTENTION_ID_SANITIZE_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class ConsolidationDeps:
    sqlite_current_path: Callable[[str, str], Path | None]
    sqlite_ensure_nonempty: Callable[[Path], None]
    sqlite_connect: Callable[[Path], sqlite3.Connection]
    sqlite_ensure_conversation_state_schema: Callable[[sqlite3.Connection], None]
    conversation_state_row: Callable[[sqlite3.Connection, str], sqlite3.Row | None]
    conversation_state_from_row: Callable[..., dict[str, Any] | None]
    write_conversation_state: Callable[..., tuple[dict[str, Any], Path]]
    get_storage_dir: Callable[[dict[str, Any]], Path]
    config: dict[str, Any]
    find_chat_dir_for_conversation: Callable[[Path, str, str, str], Path | None]
    read_list: Callable[[Path], list[dict[str, Any]]]
    normalize_text_list: Callable[[Any], list[str]]
    json_to_db: Callable[[Any], str | None]


def _slugify_intention_id(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return ""
    slug = _INTENTION_ID_SANITIZE_RE.sub("-", text).strip("-")
    if not slug:
        return ""
    return slug[:64]


def _node_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return str(node.text or "").strip()


def _pick_attr(node: ET.Element, *keys: str) -> str:
    for key in keys:
        value = str(node.get(key) or "").strip()
        if value:
            return value
    return ""


def _select_prompt_objective(prompt: str, *, first_time: bool) -> str:
    selected = "first_time" if first_time else "ongoing"
    for tag in ("first_time", "ongoing"):
        pattern = re.compile(rf"<{tag}>\n?(.*?)</{tag}>\n?", re.DOTALL)
        matches = list(pattern.finditer(prompt))
        if len(matches) != 1:
            raise ValueError(f"Prompt requires exactly one <{tag}> objective")
        prompt = pattern.sub(
            lambda match: match.group(1) + "\n" if tag == selected else "",
            prompt,
        )
    return prompt


def _parse_reflection_xml(
    raw: str,
    anchor_bundles: dict[str, dict[str, Any]],
    *,
    first_time: bool,
) -> dict[str, Any]:
    root = extract_xml_fragment(raw, "reflection")
    narrative_self = xml_text(root, "narrative_self")
    if first_time and not narrative_self:
        raise ValueError("First reflection requires narrative_self")
    anchor_nodes = root.findall("anchor_revisions")
    if len(anchor_nodes) != 1:
        raise ValueError("Reflection requires exactly one anchor_revisions element")
    anchor_decisions = parse_anchor_revisions(
        anchor_nodes[0],
        anchor_bundles,
        first_time=first_time,
    )

    life_goals = root.find("life_goals")
    life_goal_add = []
    life_goal_remove = []
    if life_goals is not None:
        for item in life_goals.findall("add"):
            text = str(item.text or "").strip()
            if text:
                life_goal_add.append(text)
        for item in life_goals.findall("remove"):
            text = str(item.text or "").strip()
            if text:
                life_goal_remove.append(text)

    edges: list[dict[str, Any]] = []
    edge_invalidations: list[dict[str, Any]] = []
    edges_root = root.find("edges")
    if edges_root is not None:
        for edge_node in edges_root.findall("edge"):
            subject_id = str(xml_text(edge_node, "subject_id") or "").strip()
            predicate = str(xml_text(edge_node, "predicate") or "").strip()
            object_id = str(xml_text(edge_node, "object_id") or "").strip()
            if not subject_id or not predicate or not object_id:
                continue
            confidence_text = xml_text(edge_node, "confidence")
            confidence: float | None
            if confidence_text:
                try:
                    confidence = float(confidence_text)
                except ValueError:
                    confidence = None
            else:
                confidence = None
            edge_payload: dict[str, Any] = {
                "subject_id": subject_id,
                "predicate": predicate,
                "object_id": object_id,
            }
            if confidence is not None:
                edge_payload["confidence"] = confidence
            edges.append(edge_payload)
        for invalidate_node in edges_root.findall("invalidate"):
            subject_id = str(xml_text(invalidate_node, "subject_id") or "").strip()
            predicate = str(xml_text(invalidate_node, "predicate") or "").strip()
            object_id = str(xml_text(invalidate_node, "object_id") or "").strip()
            if not subject_id or not predicate or not object_id:
                continue
            edge_invalidations.append(
                {
                    "subject_id": subject_id,
                    "predicate": predicate,
                    "object_id": object_id,
                }
            )

    comp_hints_node = root.find("companion_shaped_by_hints")
    companion_shaped_by_hints: list[str] = []
    if comp_hints_node is not None:
        seen_hint: set[str] = set()
        for mid in comp_hints_node.findall("memory_id"):
            val = str(mid.text or "").strip()
            if val and val not in seen_hint:
                companion_shaped_by_hints.append(val)
                seen_hint.add(val)

    intention_actions: list[dict[str, Any]] = []
    intentions_node = root.find("intentions")
    if intentions_node is not None:
        for boost in intentions_node.findall("boost"):
            target = _pick_attr(boost, "target_id", "intention_id", "id") or _node_text(boost)
            if target:
                intention_actions.append({"type": "boost", "target_id": target, "amount": 1})
        for promote in intentions_node.findall("promote"):
            target = _pick_attr(promote, "target_id", "intention_id", "id") or _node_text(promote)
            if target:
                intention_actions.append({"type": "promote", "target_id": target})
        for create in intentions_node.findall("create"):
            ctext = _pick_attr(create, "text", "description") or _node_text(create)
            cid = _pick_attr(create, "id", "intention_id") or _slugify_intention_id(ctext)
            if cid and ctext:
                intention_actions.append({"type": "create", "id": cid, "text": ctext})
        for annul in intentions_node.findall("annul"):
            aid = _pick_attr(annul, "intention_id", "target_id", "id") or _node_text(annul)
            astatus = str(annul.get("status") or "completed").strip().lower()
            anote = str(annul.get("note") or "").strip()
            if aid:
                intention_actions.append({"type": "annul", "intention_id": aid, "status": astatus, "note": anote})

    return {
        "narrative_self": narrative_self,
        "life_goal_add": life_goal_add,
        "life_goal_remove": life_goal_remove,
        "companion_memory": xml_text(root, "companion_memory"),
        "companion_shaped_by_hints": companion_shaped_by_hints,
        "edges": edges,
        "edge_invalidations": edge_invalidations,
        "intention_actions": intention_actions,
        "anchor_decisions": anchor_decisions,
    }


def _read_only_citations(text: str) -> str:
    return re.sub(r"\[M([1-9][0-9]*)\]", r"[\1]", text)


def _format_dossier_revision_blocks(bundles: Sequence[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for bundle in bundles:
        dossier = bundle["dossier"]
        sectioned = label_sections(str(dossier.summary or "") or "## unlabeled")
        statuses = revision_status_items(bundle)
        blocks.append(
            "\n".join(
                [
                    f"## Dossier {dossier.id}",
                    f"Kind: {dossier.kind}",
                    f"Title: {dossier.name}",
                    f"Target words: {bundle['target_words']}",
                    f"Description: {dossier.description}",
                    "",
                    "### Current dossier prose",
                    sectioned[0] if sectioned is not None else str(dossier.summary or ""),
                    "",
                    "### cited_member list ([M#] full text to understand context)",
                    render_memory_records(statuses["cited"]),
                    "",
                    "### search_result list (relevance uncertain)",
                    render_memory_records(statuses["search"]),
                    "",
                    "### purged_member list (for prose review)",
                    render_memory_records(statuses["purged"]),
                    "",
                    "### pending_member list (decision needed)",
                    render_memory_records(statuses["pending"]),
                ]
            )
        )
    return "\n\n".join(blocks)


async def prepare_dossier_consolidation_context(
    svc: MemoryService,
    *,
    inputs: dict[str, Any],
    soul_id: str,
    user_id: str,
) -> dict[str, Any]:
    revision_profile = svc.memorize_config.category_update_llm_profile

    scope = {"soul_id": soul_id, "user_id": user_id}
    prompt_context = _build_consolidation_prompt_context(inputs, soul_id=soul_id)
    inputs["reflection_prompt_context"] = prompt_context
    actionable_ids = prompt_context["actionable_item_ids"]
    inputs["anchor_bundles"] = {
        role: svc.prepare_anchor_revision(role, scope, actionable_ids)
        for role in ("soul", "user")
    }

    bundles = [
        svc.prepare_dossier_revision(
            dossier.id,
            scope,
            narrative_self=inputs.get("narrative_self"),
            active_life_goals=inputs["active_life_goals"],
            removed_life_goals=inputs["removed_life_goals"],
        )
        for dossier in svc.list_due_dossiers(scope)
    ]
    if bundles:
        anchors = inputs["anchor_bundles"]
        first_time = not bool(str(inputs.get("narrative_self") or "").strip())
        system_prompt = _select_prompt_objective(
            dossiers_prompt.SYSTEM_PROMPT,
            first_time=first_time,
        ).format(soul_name=soul_id, user_name=user_id)
        literal_fields = {
            name: "{" + name + "}"
            for name in (
                "dossier_id",
                "dossier_kind",
                "dossier_title",
                "target_words",
                "dossier_description",
                "current_prose",
                "cited_memory_records",
                "candidate_memory_records",
                "cleanup_memberships",
                "required_memory_records",
            )
        }
        user_prompt = dossiers_prompt.USER_PROMPT.format(
            narrative_self=prompt_context["narrative_self"],
            soul_anchor=_read_only_citations(str(anchors["soul"]["dossier"].summary or "## unlabeled")),
            user_anchor=_read_only_citations(str(anchors["user"]["dossier"].summary or "## unlabeled")),
            dossier_index=_read_only_citations(svc.build_dossier_index(scope)) or "(none)",
            life_goals=prompt_context["life_goals"],
            current_intentions=prompt_context["current_intentions"],
            intention_activity=prompt_context["intention_activity"],
            prior_context_memory_items=prompt_context["prior_context_memory_items"],
            conversation_history=prompt_context["conversation_history"],
            segment_memory_items=prompt_context["segment_memory_items"],
            dossier_revision_blocks=_format_dossier_revision_blocks(bundles),
            **literal_fields,
        )
        if len((system_prompt + "\n" + user_prompt).split()) / 0.75 > 100_000:
            raise ValueError("Dossier consolidation prompt exceeds 100000 tokens")
        raw = await svc.chat(
            user_prompt,
            profile=revision_profile,
            system_prompt=system_prompt,
            op="consolidation",
            step="dossiers",
        )
        decisions = parse_dossier_revision_batch(str(raw or ""), bundles)
        for bundle, decision in zip(bundles, decisions, strict=True):
            await svc.apply_dossier_revision(bundle, decision, scope)

    inputs["dossier_index"] = svc.build_dossier_index(scope)
    inputs["relevant_dossiers"] = svc.list_dossiers_for_segments(
        scope,
        segment_ids=inputs["selected_segment_ids"],
    )
    return inputs


def preflight_consolidation_profiles(
    svc: MemoryService,
    consolidation_llm_profile: str | None,
) -> str:
    revision_profile = svc.memorize_config.category_update_llm_profile
    for profile_name in (revision_profile, consolidation_llm_profile or "default"):
        if profile_name not in svc.llm_profiles.profiles:
            raise KeyError(f"Step profile '{profile_name}' not found in config")
    return revision_profile


def _format_life_goals_for_prompt(active: list[str], removed: list[str]) -> str:
    parts: list[str] = []
    if active:
        parts.append("Active:")
        parts.extend(f"- {row}" for row in active)
    if removed:
        if parts:
            parts.append("")
        parts.append("Recently removed (remove again to extinguish permanently):")
        parts.extend(f"- {row}" for row in removed)
    return "\n".join(parts) if parts else "You haven't established any life goals yet."


def _format_intention_activity_for_prompt(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "Your intentions have been steady."
    lines = []
    for row in rows:
        description = str(row.get("description") or "").strip()
        status = str(row.get("status") or "").strip()
        updated_at = str(row.get("updated_at") or "").strip()
        if not description:
            continue
        time_label = format_relative_time_label(updated_at) if updated_at else None
        meta = ", ".join(part for part in (status, time_label) if part)
        lines.append(f"- {description}" + (f" ({meta})" if meta else ""))
    return "\n".join(lines) if lines else "Your intentions have been steady."


def _select_interval_segment_window(
    segment_inputs: list[dict[str, Any]],
    *,
    interval_days: int,
    force: bool,
) -> dict[str, Any]:
    if not segment_inputs:
        return {"selected_segment_ids": [], "remaining_segment_ids": [], "reason": "no_pending_segments"}

    ordered = sorted(
        segment_inputs,
        key=lambda row: (
            row.get("happened_at") or datetime.max.replace(tzinfo=UTC),
            int(row.get("start_idx") or 0),
            int(row.get("end_idx") or 0),
            str(row.get("segment_id") or ""),
        ),
    )
    all_ids = [str(row.get("segment_id") or "").strip() for row in ordered if str(row.get("segment_id") or "").strip()]
    if force:
        return {"selected_segment_ids": all_ids, "remaining_segment_ids": [], "reason": None}

    missing = [str(row.get("segment_id") or "") for row in ordered if not isinstance(row.get("happened_at"), datetime)]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"pending consolidation segment has no timestamp: {missing[0]}",
        )

    baseline = ordered[0]["happened_at"]
    due_at = baseline + timedelta(days=max(1, int(interval_days)))
    selected: list[str] = []
    for idx, row in enumerate(ordered):
        segment_id = str(row.get("segment_id") or "").strip()
        if not segment_id:
            continue
        selected.append(segment_id)
        if row["happened_at"] >= due_at:
            return {
                "selected_segment_ids": selected,
                "remaining_segment_ids": all_ids[idx + 1 :],
                "reason": None,
            }

    return {
        "selected_segment_ids": [],
        "remaining_segment_ids": all_ids,
        "reason": "pending_span_too_short",
    }


def _messages_for_segment_inputs(
    messages: list[dict[str, Any]],
    segment_inputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in segment_inputs:
        start = int(row.get("start_idx") or 0)
        end = int(row.get("end_idx") or 0)
        if start < 0 or end < start:
            continue
        out.extend(msg for msg in messages[start : end + 1] if isinstance(msg, dict))
    return out



def _format_segment_memory_items_for_prompt(
    segment_memory_groups: list[dict[str, Any]],
    id_map: dict[str, str],
) -> str:
    if not segment_memory_groups:
        return "(none queued)"
    all_types: set[str] = set()
    for row in segment_memory_groups:
        for s in row.get("memory_summaries") or []:
            if isinstance(s, dict):
                all_types.add(str(s.get("memory_type") or ""))
    lines: list[str] = []
    legend = format_memory_legend(all_types)
    if legend:
        lines.append(legend)
    for row in segment_memory_groups:
        summaries = row.get("memory_summaries") or []
        if summaries:
            for s in summaries:
                if isinstance(s, dict):
                    mid = s["id"]
                    memory_ref = int(s.get("memory_ref") or 0)
                    if memory_ref <= 0:
                        raise ValueError(f"Consolidation memory lacks stable reference: {mid}")
                    id_map[f"M{memory_ref}"] = mid
                    lines.append(f"- {format_memory_line(s, show_id=True, item_id=f'M{memory_ref}')}")
                elif str(s).strip():
                    lines.append(f"- {s}")
            lines.append("")
    return "\n".join(lines).strip() or "(none)"


def _dedupe_segment_memory_items_against_prior_context(
    segment_memory_groups: list[dict[str, Any]],
    prior_context_memory_items: list[Any],
) -> list[dict[str, Any]]:
    prior_ids = {
        str(item.get("id") or "").strip()
        for item in prior_context_memory_items
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    if not prior_ids:
        return segment_memory_groups

    deduped: list[dict[str, Any]] = []
    for group in segment_memory_groups:
        next_group = dict(group)
        next_group["memory_summaries"] = [
            item
            for item in group.get("memory_summaries") or []
            if not (
                isinstance(item, dict)
                and str(item.get("id") or "").strip() in prior_ids
            )
        ]
        deduped.append(next_group)
    return deduped


def _build_consolidation_prompt_context(
    inputs: dict[str, Any],
    *,
    soul_id: str,
) -> dict[str, Any]:
    life_goals_text = _format_life_goals_for_prompt(
        inputs["active_life_goals"],
        inputs["removed_life_goals"],
    )
    intention_text = _format_intention_activity_for_prompt(inputs["intention_activity"])
    id_map: dict[str, str] = {}
    actionable_ids: list[str] = []

    prior_context_memory_items = inputs.get("prior_context_memory_items") or []
    if prior_context_memory_items:
        prior_context_types = {
            str(item.get("memory_type") or "")
            for item in prior_context_memory_items
            if isinstance(item, dict)
        }
        prior_context_lines = [format_memory_legend(prior_context_types)]
        for item in prior_context_memory_items:
            if not isinstance(item, dict):
                if str(item).strip():
                    prior_context_lines.append(str(item))
                continue
            item_id = str(item["id"])
            memory_ref = int(item.get("memory_ref") or 0)
            if memory_ref <= 0:
                raise ValueError(f"Consolidation memory lacks stable reference: {item_id}")
            id_map[f"M{memory_ref}"] = item_id
            actionable_ids.append(item_id)
            prior_context_lines.append(
                format_memory_line(item, show_id=True, item_id=f"M{memory_ref}")
            )
            shaped_by = item.get("shaped_by")
            if isinstance(shaped_by, dict):
                prior_context_lines.append(format_shaped_by_line(shaped_by))
        prior_context_text = "\n".join(line for line in prior_context_lines if line)
    else:
        prior_context_text = "(none surfaced)"

    segment_groups = _dedupe_segment_memory_items_against_prior_context(
        inputs["segment_inputs"],
        prior_context_memory_items,
    )
    segment_memory_items_text = _format_segment_memory_items_for_prompt(segment_groups, id_map)
    actionable_ids.extend(
        str(item["id"])
        for group in segment_groups
        for item in group.get("memory_summaries") or []
        if isinstance(item, dict)
    )
    current_intentions_raw = inputs.get("state", {}).get("intentions_active")
    current_intentions_text = (
        format_intentions_for_prompt(current_intentions_raw, include_internals=True)
        if current_intentions_raw
        else "(none yet)"
    )
    narrative = str(inputs.get("narrative_self") or "").strip()
    return {
        "narrative_self": narrative or DEFAULT_SOUL_CARD.format(soul_name=soul_id),
        "life_goals": life_goals_text,
        "current_intentions": current_intentions_text,
        "intention_activity": intention_text,
        "prior_context_memory_items": prior_context_text,
        "conversation_history": str(inputs.get("all_chat_history") or "").strip() or "(none)",
        "segment_memory_items": segment_memory_items_text,
        "actionable_item_ids": list(dict.fromkeys(actionable_ids)),
        "id_map": id_map,
    }


def _looks_like_memory_id(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return bool(_HEX_MEMORY_ID_RE.fullmatch(text) or _UUID_MEMORY_ID_RE.fullmatch(text))


def _resolve_memory_ref(raw_value: Any, id_map: dict[str, str]) -> str | None:
    text = str(raw_value or "").strip()
    if not text:
        return None

    # Most common case: numbered prompt references like "16".
    mapped = id_map.get(text)
    if mapped:
        return mapped

    # Allow bracketed references like "[16]".
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if inner:
            mapped_inner = id_map.get(inner)
            if mapped_inner:
                return mapped_inner
            text = inner

    # Some models prepend "#" to numbered references ("#16").
    if text.startswith("#"):
        mapped_hash = id_map.get(text[1:].strip())
        if mapped_hash:
            return mapped_hash

    # Fall back to direct memory IDs when model emits raw IDs from context.
    if _looks_like_memory_id(text):
        return text
    return None


def _remap_edges_with_memory_ids(
    payload: list[dict[str, Any]],
    *,
    id_map: dict[str, str],
    include_confidence: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for edge in payload:
        if not isinstance(edge, dict):
            continue
        subject_id = _resolve_memory_ref(edge.get("subject_id"), id_map)
        object_id = _resolve_memory_ref(edge.get("object_id"), id_map)
        predicate = str(edge.get("predicate") or "").strip()
        if not subject_id or not object_id or not predicate:
            continue
        mapped = {"subject_id": subject_id, "predicate": predicate, "object_id": object_id}
        if include_confidence and "confidence" in edge:
            mapped["confidence"] = edge["confidence"]
        out.append(mapped)
    return out


def gather_consolidation_inputs(
    deps: ConsolidationDeps,
    *,
    conversation_id: str,
    soul_id: str,
    user_id: str,
    force: bool,
    interval_days: int,
    stale_after: timedelta,
) -> dict[str, Any]:
    db_path = deps.sqlite_current_path(user_id, soul_id)
    if db_path is None:
        raise HTTPException(status_code=400, detail="soul_id required")
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="conversation database not found")

    deps.sqlite_ensure_nonempty(db_path)
    con = deps.sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        deps.sqlite_ensure_conversation_state_schema(con)
        state = deps.conversation_state_from_row(deps.conversation_state_row(con, conversation_id), con=con)
        if state is None:
            raise HTTPException(status_code=404, detail="conversation state not found")

        now = datetime.now(UTC)
        started_at = parse_iso_datetime(state.get("consolidation_started_at"))
        if bool(state.get("consolidation_in_progress")):
            if started_at is not None and now - started_at <= stale_after:
                return {"status": "skip", "reason": "in_progress"}
            deps.write_conversation_state(
                conversation_id,
                soul_id=soul_id,
                user_id=user_id,
                updates={"consolidation_in_progress": False, "consolidation_started_at": None},
            )
            reread = deps.conversation_state_from_row(deps.conversation_state_row(con, conversation_id), con=con)
            if reread is None:
                raise HTTPException(404, "conversation state not found after stale-lock reset")
            state = reread

        pending_segment_ids = deps.normalize_text_list(state.get("pending_segment_ids"))
        if not pending_segment_ids:
            return {"status": "skip", "reason": "no_pending_segments"}

        life_goal_rows = con.execute(
            """
SELECT id, description, status
FROM life_goals
WHERE soul_id = ? AND user_id = ? AND status IN ('active', 'removed')
ORDER BY updated_at ASC, id ASC
""",
            (soul_id, user_id),
        ).fetchall()
        active_goals = [
            str(row["description"] or "").strip()
            for row in life_goal_rows
            if str(row["status"] or "").strip() == "active" and str(row["description"] or "").strip()
        ]
        removed_goals = [
            str(row["description"] or "").strip()
            for row in life_goal_rows
            if str(row["status"] or "").strip() == "removed" and str(row["description"] or "").strip()
        ]

        intention_sql = """
SELECT description, status, updated_at
FROM intentions
WHERE soul_id = ? AND user_id = ? AND source = 'inferred'
"""
        params: list[Any] = [soul_id, user_id]
        last_consolidation_at = parse_iso_datetime(state.get("last_consolidation_at"))
        if last_consolidation_at is not None:
            intention_sql += " AND updated_at >= ?"
            params.append(last_consolidation_at.isoformat())
        intention_sql += " ORDER BY updated_at ASC, id ASC LIMIT 40"
        intention_rows = con.execute(intention_sql, tuple(params)).fetchall()
        intention_activity = [
            {
                "description": str(row["description"] or "").strip(),
                "status": str(row["status"] or "").strip(),
                "updated_at": str(row["updated_at"] or "").strip(),
            }
            for row in intention_rows
            if str(row["description"] or "").strip()
        ]

        narrative_self = str(state.get("narrative_self") or "").strip() or None

        segment_inputs: list[dict[str, Any]] = []
        selected_segment_ids: list[str] = []
        remaining_segment_ids: list[str] = []
        messages: list[dict[str, Any]] = []
        if pending_segment_ids:
            storage_dir = deps.get_storage_dir(deps.config)
            chats_dir = (storage_dir / "st_chats").resolve()
            chat_dir = deps.find_chat_dir_for_conversation(chats_dir, user_id, soul_id, conversation_id)
            if chat_dir is None:
                raise HTTPException(status_code=404, detail="conversation resource not found")
            manifest_path = (chat_dir / "manifest.json").resolve()
            if not manifest_path.exists():
                raise HTTPException(status_code=404, detail="conversation manifest not found")
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise HTTPException(status_code=400, detail="conversation manifest unreadable") from exc
            raw_segments = manifest.get("segments") if isinstance(manifest, dict) else None
            if not isinstance(raw_segments, list) or not raw_segments:
                raise HTTPException(status_code=400, detail="conversation manifest has no segments")
            segments_dir = (chat_dir / "segments").resolve()
            if segments_dir.is_dir():
                for ep_file in sorted(segments_dir.glob("*.json"), key=_segment_file_sort_key):
                    try:
                        parsed = json.loads(ep_file.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if isinstance(parsed, list):
                        messages.extend(m for m in parsed if isinstance(m, dict))
            segment_inputs = build_segment_inputs(messages, pending_segment_ids)
            if len(segment_inputs) != len(pending_segment_ids):
                raise HTTPException(status_code=400, detail="queued segments are not present in conversation history")
            window = _select_interval_segment_window(
                segment_inputs,
                interval_days=interval_days,
                force=force,
            )
            selected_segment_ids = list(window["selected_segment_ids"])
            remaining_segment_ids = list(window["remaining_segment_ids"])
            if not selected_segment_ids:
                return {"status": "skip", "reason": window.get("reason") or "pending_span_too_short"}
            selected_segment_id_set = set(selected_segment_ids)
            segment_inputs = [
                row for row in segment_inputs
                if str(row.get("segment_id") or "").strip() in selected_segment_id_set
            ]

            for entry in segment_inputs:
                segment_id = str(entry["segment_id"])
                rows = con.execute(
                    """
SELECT id, memory_ref, summary, memory_type, happened_at, created_at
FROM memory_items
WHERE soul_id = ? AND user_id = ? AND conversation_id = ? AND segment_id = ? AND memory_type NOT IN ('narrative_self')
  AND (merged_into IS NULL OR TRIM(merged_into) = '')
  AND NOT EXISTS (
    SELECT 1 FROM triples t
    WHERE t.subject_id = memory_items.id
      AND t.predicate = 'evolved_into'
      AND t.valid_to IS NULL
  )
ORDER BY created_at ASC, id ASC
LIMIT 24
""",
                    (soul_id, user_id, conversation_id, segment_id),
                ).fetchall()
                entry["memory_summaries"] = [
                    {
                        "id": str(row["id"] or "").strip(),
                        "memory_ref": int(row["memory_ref"] or 0),
                        "summary": str(row["summary"] or "").strip(),
                        "memory_type": str(row["memory_type"] or "").strip(),
                        "happened_at": row["happened_at"] or row["created_at"],
                    }
                    for row in rows
                    if str(row["id"] or "").strip() and str(row["summary"] or "").strip()
                ]

        prior_context_memory_items: list[dict[str, Any]] = []
        all_prior_context_ids: list[str] = []
        last_consol = state.get("last_consolidation_at")
        try:
            if last_consol:
                res_rows = con.execute(
                    "SELECT memory_prior_context FROM resources WHERE soul_id = ? AND user_id = ? AND created_at >= ? AND memory_prior_context IS NOT NULL",
                    (soul_id, user_id, last_consol),
                ).fetchall()
            else:
                res_rows = con.execute(
                    "SELECT memory_prior_context FROM resources WHERE soul_id = ? AND user_id = ? AND memory_prior_context IS NOT NULL",
                    (soul_id, user_id),
                ).fetchall()
            for rr in res_rows:
                raw = rr["memory_prior_context"]
                if raw is None:
                    continue
                ids = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(ids, list):
                    all_prior_context_ids.extend(str(rid).strip() for rid in ids if str(rid).strip())
        except (sqlite3.Error, TypeError, ValueError, json.JSONDecodeError):
            log.error("consolidation: failed loading resource prior_context history", exc_info=True)
        clean_ids = list(dict.fromkeys(all_prior_context_ids))
        if clean_ids:
            placeholders = ",".join("?" for _ in clean_ids)
            ret_rows = con.execute(
                f"SELECT id, memory_ref, memory_type, summary, happened_at, created_at FROM memory_items WHERE id IN ({placeholders})",
                tuple(clean_ids),
            ).fetchall()
            items_by_id: dict[str, dict[str, Any]] = {}
            prior_context_memory_items = []
            for row in ret_rows:
                mid = str(row["id"] or "").strip()
                summary = str(row["summary"] or "").strip()
                if not mid or not summary:
                    continue
                entry: dict[str, Any] = {
                    "id": mid,
                    "memory_ref": int(row["memory_ref"] or 0),
                    "summary": summary,
                    "memory_type": str(row["memory_type"] or "").strip(),
                    "happened_at": row["happened_at"] or row["created_at"],
                }
                items_by_id[mid] = entry
                prior_context_memory_items.append(entry)

            edge_predicates = ("caused_by", "evokes", "conflicts_with", "parallels", "shaped_by")
            edge_placeholders = ",".join("?" for _ in clean_ids)
            pred_placeholders = ",".join("?" for _ in edge_predicates)
            edge_rows = con.execute(
                f"SELECT subject_id, predicate, object_id FROM triples "
                f"WHERE subject_id IN ({edge_placeholders}) AND object_id IN ({edge_placeholders}) "
                f"AND predicate IN ({pred_placeholders}) AND valid_to IS NULL",
                tuple(clean_ids) + tuple(clean_ids) + tuple(edge_predicates),
            ).fetchall()
            for er in edge_rows:
                subj = str(er["subject_id"]).strip()
                obj_id = str(er["object_id"]).strip()
                pred = str(er["predicate"]).strip()
                obj_item = items_by_id.get(obj_id)
                if subj in items_by_id and obj_item:
                    items_by_id[subj]["shaped_by"] = {
                        "predicate": pred,
                        "id": obj_id,
                        "summary": obj_item.get("summary", ""),
                        "memory_type": obj_item.get("memory_type", ""),
                        "happened_at": obj_item.get("happened_at"),
                    }

        deps.write_conversation_state(
            conversation_id,
            soul_id=soul_id,
            user_id=user_id,
            updates={
                "consolidation_in_progress": True,
                "consolidation_started_at": now.isoformat(),
            },
        )
        return {
            "status": "ready",
            "db_path": db_path,
            "state": state,
            "segment_ids": pending_segment_ids,
            "active_life_goals": active_goals,
            "removed_life_goals": removed_goals,
            "intention_activity": intention_activity,
            "segment_inputs": segment_inputs,
            "current_chat_messages": _messages_for_segment_inputs(messages, segment_inputs),
            "narrative_self": narrative_self,
            "last_consolidation_at": state.get("last_consolidation_at"),
            "started_at": now.isoformat(),
            "prior_context_memory_items": prior_context_memory_items,
            "selected_segment_ids": selected_segment_ids,
            "remaining_segment_ids": remaining_segment_ids,
        }
    finally:
        con.close()


async def run_consolidation_llm(
    svc: MemoryService,
    *,
    inputs: dict[str, Any],
    soul_id: str,
    user_id: str,
    llm_profile: str | None = None,
) -> dict[str, Any]:
    narrative = str(inputs.get("narrative_self") or "").strip()
    first_time = not bool(narrative)
    system_prompt = _select_prompt_objective(
        anchors_prompt.SYSTEM_PROMPT,
        first_time=first_time,
    ).format(soul_name=soul_id, user_name=user_id)
    context = inputs["reflection_prompt_context"]
    id_map = context["id_map"]
    anchor_bundles = inputs["anchor_bundles"]
    for bundle in anchor_bundles.values():
        for item in bundle["cited_items"]:
            memory_ref = int(item.memory_ref or 0)
            if memory_ref <= 0:
                raise ValueError(f"Consolidation memory lacks stable reference: {item.id}")
            id_map[f"M{memory_ref}"] = str(item.id)

    def anchor_prose(role: str) -> str:
        prose = str(anchor_bundles[role]["dossier"].summary or "") or "## unlabeled"
        sectioned = label_sections(prose)
        return sectioned[0] if sectioned is not None else prose

    relevant_parts: list[str] = []
    for dossier in inputs["relevant_dossiers"]:
        relevant_parts.extend(
            [
                f"## {dossier.name}",
                f"Description: {_read_only_citations(dossier.description)}",
                _read_only_citations(str(dossier.summary or "")),
                "",
            ]
        )
    relevant_dossiers = "\n".join(relevant_parts).strip() or "(none)"
    user_prompt = anchors_prompt.USER_PROMPT.format(
        narrative_self=context["narrative_self"],
        soul_anchor=anchor_prose("soul"),
        soul_anchor_cited_memory_items=render_memory_records(anchor_bundles["soul"]["cited_items"]),
        user_anchor=anchor_prose("user"),
        user_anchor_cited_memory_items=render_memory_records(anchor_bundles["user"]["cited_items"]),
        dossier_index=_read_only_citations(inputs["dossier_index"]) or "(none)",
        relevant_dossiers=relevant_dossiers,
        life_goals=context["life_goals"],
        current_intentions=context["current_intentions"],
        intention_activity=context["intention_activity"],
        prior_context_memory_items=context["prior_context_memory_items"],
        conversation_history=context["conversation_history"],
        segment_memory_items=context["segment_memory_items"],
    )
    if len((system_prompt + "\n" + user_prompt).split()) / 0.75 > 100_000:
        raise ValueError("Anchor reflection prompt exceeds 100000 tokens")
    raw = await svc.chat(
        user_prompt,
        profile=llm_profile,
        system_prompt=system_prompt,
        op="consolidation",
        step="reflection",
    )
    parsed = _parse_reflection_xml(
        str(raw or ""),
        anchor_bundles,
        first_time=first_time,
    )
    remapped_edges = _remap_edges_with_memory_ids(parsed["edges"], id_map=id_map, include_confidence=True)
    remapped_invalidations = _remap_edges_with_memory_ids(
        parsed["edge_invalidations"],
        id_map=id_map,
        include_confidence=False,
    )
    if parsed["edges"] and not remapped_edges:
        log.warning(
            "consolidation: dropped all parsed edges due unresolved ids (parsed=%d)",
            len(parsed["edges"]),
        )
    elif len(remapped_edges) < len(parsed["edges"]):
        log.warning(
            "consolidation: dropped %d/%d edges due unresolved ids",
            len(parsed["edges"]) - len(remapped_edges),
            len(parsed["edges"]),
        )
    if parsed["edge_invalidations"] and not remapped_invalidations:
        log.warning(
            "consolidation: dropped all parsed edge invalidations due unresolved ids (parsed=%d)",
            len(parsed["edge_invalidations"]),
        )

    intention_actions = [
        a for a in parsed["intention_actions"]
        if not (
            str(a.get("type") or "").strip() in {"boost", "promote"}
            and str(a.get("target_id") or "").strip().lower() == RELAX_INTENTION_ID
        )
    ]

    new_narrative = str(parsed["narrative_self"] or "").strip() or None
    current_narrative = str(inputs.get("narrative_self") or "").strip() or None
    snapshot_old_narrative = bool(current_narrative and new_narrative and current_narrative != new_narrative)
    embed_inputs: list[str] = []
    if str(parsed["companion_memory"] or "").strip():
        embed_inputs.append(str(parsed["companion_memory"]).strip())
    if snapshot_old_narrative:
        embed_inputs.append(current_narrative)
    embeddings = await svc.embed(embed_inputs, profile="embedding") if embed_inputs else []

    cursor = 0
    companion_embedding = None
    if str(parsed["companion_memory"] or "").strip():
        if cursor < len(embeddings):
            companion_embedding = embeddings[cursor]
        cursor += 1
    old_narrative_embedding = embeddings[cursor] if snapshot_old_narrative and cursor < len(embeddings) else None

    scope = {"soul_id": soul_id, "user_id": user_id}
    for role in ("soul", "user"):
        await svc.apply_anchor_revision(
            anchor_bundles[role],
            parsed["anchor_decisions"][role],
            scope,
        )

    return {
        "narrative_self": new_narrative,
        "life_goal_add": [str(x).strip() for x in parsed["life_goal_add"] if str(x).strip()],
        "life_goal_remove": [str(x).strip() for x in parsed["life_goal_remove"] if str(x).strip()],
        "companion_memory": str(parsed["companion_memory"] or "").strip() or None,
        "companion_shaped_by_hints": parsed["companion_shaped_by_hints"],
        "companion_embedding": companion_embedding,
        "old_narrative_text": current_narrative if snapshot_old_narrative else None,
        "old_narrative_embedding": old_narrative_embedding,
        "edges": remapped_edges,
        "edge_invalidations": remapped_invalidations,
        "intention_actions": intention_actions,
    }


def write_consolidation_outputs(
    deps: ConsolidationDeps,
    svc: MemoryService,
    *,
    inputs: dict[str, Any],
    llm_results: dict[str, Any],
    conversation_id: str,
    soul_id: str,
    user_id: str,
) -> dict[str, Any]:
    db_path: Path = inputs["db_path"]
    now_iso = datetime.now(UTC).isoformat()

    narrative_id = str(uuid.uuid4())
    narrative_self = str(llm_results.get("narrative_self") or "").strip() or None

    old_narrative_text = llm_results.get("old_narrative_text")
    companion_memory_id = None
    companion_text = str(llm_results.get("companion_memory") or "").strip()
    companion_embedding = llm_results.get("companion_embedding")
    if companion_text and not isinstance(companion_embedding, list):
        raise HTTPException(status_code=500, detail="missing companion memory embedding")

    deps.sqlite_ensure_nonempty(db_path)
    con = deps.sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        deps.sqlite_ensure_conversation_state_schema(con)

        if narrative_self:
            con.execute(
                "INSERT INTO narrative_history (id, narrative_self, related_memory_ids, created_at) "
                "VALUES (?, ?, ?, ?)",
                (narrative_id, narrative_self, deps.json_to_db([]), now_iso),
            )
            _soul_summaries.write_live(
                con,
                kind="narrative_self",
                summary=narrative_self,
                scope={"user_id": user_id, "soul_id": soul_id},
                edited_by="consolidation",
            )

        life_goal_rows = con.execute(
            """
SELECT id, description, status
FROM life_goals
WHERE soul_id = ? AND user_id = ? AND status IN ('active', 'removed')
ORDER BY updated_at ASC, id ASC
""",
            (soul_id, user_id),
        ).fetchall()
        active_ids: dict[str, str] = {}
        removed_ids: dict[str, str] = {}
        for row in life_goal_rows:
            description = str(row["description"] or "").strip()
            if not description:
                continue
            if str(row["status"] or "").strip() == "active":
                active_ids[description] = str(row["id"])
            else:
                removed_ids[description] = str(row["id"])

        goals_to_mark_removed: list[str] = []
        goals_to_delete: list[str] = []
        goals_to_add: list[tuple[str, str]] = []

        for desc in llm_results["life_goal_remove"]:
            text = str(desc or "").strip()
            if not text:
                continue
            if text in active_ids:
                goals_to_mark_removed.append(active_ids[text])
                removed_ids[text] = active_ids[text]
                active_ids.pop(text, None)
            elif text in removed_ids:
                goals_to_delete.append(removed_ids[text])
                removed_ids.pop(text, None)

        active_goal_count = len(active_ids)
        for desc in llm_results["life_goal_add"]:
            text = str(desc or "").strip()
            if not text or text in active_ids or active_goal_count >= 3:
                continue
            goal_id = str(uuid.uuid4())
            goals_to_add.append((goal_id, text))
            active_ids[text] = goal_id
            active_goal_count += 1

        for goal_id in goals_to_mark_removed:
            con.execute(
                "UPDATE life_goals SET status = 'removed', updated_at = ? WHERE id = ?",
                (now_iso, goal_id),
            )
        for goal_id in goals_to_delete:
            con.execute("DELETE FROM life_goals WHERE id = ?", (goal_id,))
        for goal_id, text in goals_to_add:
            con.execute(
                """
INSERT INTO life_goals (
    id, soul_id, user_id, description, status, updated_at
) VALUES (?, ?, ?, ?, 'active', ?)
""",
                (goal_id, soul_id, user_id, text, now_iso),
            )
        con.commit()
    finally:
        con.close()

    from app.services.intention_state import (
        apply_intention_action,
        drop_unpromoted_ephemeral_intentions,
        remove_intentions,
        upsert_intentions_stack_entries,
    )
    _con = deps.sqlite_connect(db_path)
    _con.row_factory = sqlite3.Row
    current_state = _soul_state.read(_con)
    _con.close()
    current_intentions = current_state.get("intentions_active")
    promoted_ids = {
        str(action.get("target_id") or "").strip()
        for action in (llm_results.get("intention_actions") or [])
        if str(action.get("type") or "").strip() == "promote"
    }
    current_intentions = drop_unpromoted_ephemeral_intentions(current_intentions, promoted_ids)

    if inputs.get("last_consolidation_at") is None:
        current_intentions = upsert_intentions_stack_entries(
            current_intentions,
            [{"id": "optimize-happiness", "text": "Optimize happiness for myself and those I love.", "ephemeral": False}],
        )

    for action in llm_results.get("intention_actions") or []:
        atype = str(action.get("type") or "").strip()
        if atype == "boost":
            current_intentions = apply_intention_action(current_intentions, action)
        elif atype == "promote":
            current_intentions = apply_intention_action(current_intentions, action)
        elif atype == "create":
            current_intentions = upsert_intentions_stack_entries(
                current_intentions,
                [{"id": action.get("id"), "text": action.get("text"), "ephemeral": True}],
            )
        elif atype == "annul":
            aid = str(action.get("intention_id") or "").strip()
            if aid and aid.lower() != "relax":
                current_intentions = remove_intentions(current_intentions, [aid])

    # Preflight the state writer before companion and graph side effects.
    deps.write_conversation_state(
        conversation_id,
        soul_id=soul_id,
        user_id=user_id,
        updates={},
    )

    if old_narrative_text:
        snapshot_previous_narrative_self(
            svc,
            scope={"user_id": user_id, "soul_id": soul_id},
            old_text=old_narrative_text,
            old_embedding=llm_results["old_narrative_embedding"],
        )
    if companion_text:
        companion_happened_at = datetime.now(UTC)
        companion_memory_id = create_companion_memory(
            svc,
            user_id=user_id,
            soul_id=soul_id,
            conversation_id=conversation_id,
            summary=companion_text,
            embedding=companion_embedding,
            happened_at=companion_happened_at,
        )

    scope = {"user_id": user_id, "soul_id": soul_id}
    wrote = write_memory_edges(svc.database.triple_repo, llm_results["edges"], scope=scope)
    invalidated = invalidate_memory_edges(svc.database.triple_repo, llm_results["edge_invalidations"], scope=scope)
    remaining_segment_ids = [
        str(segment_id).strip()
        for segment_id in (inputs.get("remaining_segment_ids") or [])
        if str(segment_id).strip()
    ]

    consumed_segment_ids = [
        str(segment_id).strip()
        for segment_id in (inputs.get("selected_segment_ids") or [])
        if str(segment_id).strip()
    ]
    state_updates: dict[str, Any] = {
        # Subtract only what this run consumed: a memorize that finished during
        # the LLM phase may have appended new pending ids, and an absolute
        # overwrite with the launch-time snapshot would silently drop them.
        "remove_pending_segment_ids": consumed_segment_ids,
        "last_consolidation_at": now_iso,
        "consolidation_in_progress": False,
        "consolidation_started_at": None,
        "intentions_active": current_intentions,
        "retrieval_ids_since_consolidation": [],
        "prior_context_ids_since_consolidation": [],
    }
    state_after, _ = deps.write_conversation_state(
        conversation_id,
        soul_id=soul_id,
        user_id=user_id,
        updates=state_updates,
    )

    return {
        "conversation_id": conversation_id,
        "narrative_id": narrative_id if llm_results.get("narrative_self") else None,
        "companion_memory_id": companion_memory_id,
        "edges_written": wrote,
        "edges_invalidated": invalidated,
        "consumed_segment_ids": inputs.get("selected_segment_ids") or [],
        "remaining_segment_ids": remaining_segment_ids,
        "state": state_after,
    }
