from __future__ import annotations

import asyncio
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
from typing import TYPE_CHECKING, Any, cast

log = logging.getLogger(__name__)

from fastapi import HTTPException
from memu.app.dossier import label_sections, render_memory_records, revision_status_items
from memu.app.dossier_revision import (
    estimate_prompt_tokens,
    parse_anchor_revisions,
    parse_dossier_revision_batch,
)
from memu.prompts.consolidation import anchors as anchors_prompt
from memu.prompts.consolidation import dossiers as dossiers_prompt
from memu.prompts.consolidation import weekly as weekly_prompt

from app.services.segment import (
    build_segment_inputs,
    create_companion_memory,
)
from app.services.xml_utils import extract_xml_fragment, xml_text
from app.services.graph_edges import (
    ALLOWED_EDGE_PREDICATES,
    invalidate_memory_edges,
    write_memory_edges,
)
from app.services.intention_state import (
    format_intentions_for_prompt,
    merge_consolidated_intentions,
    validate_intention_replacement,
)
from app.services.narrative_self import snapshot_previous_narrative_self
from app.services import soul_summaries as _soul_summaries
from app.services.conversation_id import canonical_conversation_id
from app.services.payload import parse_iso_datetime
from app.services import soul_state as _soul_state
from app.services import service_factory as _service_factory
from app.services.turn_contract import format_memory_legend, format_memory_line, format_shaped_by_line

if TYPE_CHECKING:
    from memu.app import MemoryService


_SEGMENT_SUFFIX_RE = re.compile(r"_(\d+)\.json$")


def _segment_file_sort_key(path: Path) -> tuple[str, int]:
    stem = path.stem
    m = _SEGMENT_SUFFIX_RE.search(path.name)
    if m:
        return (path.name[:m.start()], int(m.group(1)))
    return (stem, 0)


# Current beta reflection models have 1M-token contexts; keep 200k for output,
# provider framing, and estimator error. Revisit when the beta profile changes.
CONSOLIDATION_PROMPT_TOKEN_LIMIT = 800_000


@dataclass(frozen=True)
class ConsolidationDeps:
    sqlite_current_path: Callable[[str, str], Path | None]
    sqlite_ensure_nonempty: Callable[[Path], None]
    sqlite_connect: Callable[[Path], sqlite3.Connection]
    sqlite_ensure_conversation_state_schema: Callable[[sqlite3.Connection], None]
    conversation_state_row: Callable[..., sqlite3.Row | None]
    conversation_state_from_row: Callable[..., dict[str, Any] | None]
    write_conversation_state: Callable[..., tuple[dict[str, Any], Path]]
    get_storage_dir: Callable[[dict[str, Any]], Path]
    config: dict[str, Any]
    find_chat_dir_for_conversation: Callable[[Path, str, str, str], Path | None]
    read_list: Callable[[Path], list[dict[str, Any]]]
    normalize_text_list: Callable[[Any], list[str]]
    json_to_db: Callable[[Any], str | None]


def _node_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return str(node.text or "").strip()


def _parse_identity_maintenance_xml(
    raw: str,
    anchor_bundles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    root = extract_xml_fragment(raw, "identity_maintenance")
    if root.attrib or (root.text or "").strip():
        raise ValueError("Expected exact identity_maintenance element")
    if [child.tag for child in root] != ["narrative_self", "anchor_revisions", "life_goals"]:
        raise ValueError("Identity maintenance requires narrative_self, anchor_revisions, then life_goals")

    narrative_node = root.find("narrative_self")
    if narrative_node is None or set(narrative_node.attrib) != {"action"} or list(narrative_node):
        raise ValueError("Identity maintenance requires exact narrative_self action")
    narrative_action = str(narrative_node.get("action") or "").strip()
    narrative_text = _node_text(narrative_node)
    if narrative_action == "keep":
        if narrative_text:
            raise ValueError("Kept narrative_self must be empty")
        narrative_self = None
    elif narrative_action == "replace" and narrative_text:
        narrative_self = narrative_text
    else:
        raise ValueError("narrative_self action must be keep or nonblank replace")

    anchor_nodes = root.findall("anchor_revisions")
    if len(anchor_nodes) != 1:
        raise ValueError("Identity maintenance requires exactly one anchor_revisions element")
    anchor_decisions = parse_anchor_revisions(
        anchor_nodes[0],
        anchor_bundles,
    )

    life_goals = root.find("life_goals")
    if life_goals is None or set(life_goals.attrib) != {"action"}:
        raise ValueError("Identity maintenance requires exact life_goals action")
    life_goal_action = str(life_goals.get("action") or "").strip()
    if life_goal_action not in {"keep", "update"}:
        raise ValueError("life_goals action must be keep or update")
    life_goal_add: list[str] = []
    life_goal_remove: list[str] = []
    for item in life_goals:
        if item.tag not in {"add", "remove"} or item.attrib or list(item):
            raise ValueError("life_goals may contain only plain add/remove elements")
        text = _node_text(item)
        if text:
            (life_goal_add if item.tag == "add" else life_goal_remove).append(text)
    if life_goal_action == "keep" and (life_goal_add or life_goal_remove):
        raise ValueError("Kept life_goals cannot add or remove goals")

    return {
        "narrative_self": narrative_self,
        "life_goal_add": life_goal_add,
        "life_goal_remove": life_goal_remove,
        "anchor_decisions": anchor_decisions,
    }


def _parse_edges(root: ET.Element) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    edges: list[dict[str, Any]] = []
    edge_invalidations: list[dict[str, Any]] = []
    edges_root = root.find("edges")
    if edges_root is None or edges_root.attrib or (edges_root.text or "").strip():
        raise ValueError("Weekly reflection requires exact edges element")
    for child in edges_root:
        if child.tag not in {"edge", "invalidate"} or child.attrib or (child.tail or "").strip():
            raise ValueError("Edges may contain only edge or invalidate elements")
        required = ["subject_id", "predicate", "object_id"]
        tags = [node.tag for node in child]
        if tags not in (required, [*required, "confidence"]):
            raise ValueError(f"Invalid {child.tag} fields")
        if child.tag == "invalidate" and tags != required:
            raise ValueError("Edge invalidation cannot contain confidence")
    for edge_node in edges_root.findall("edge"):
        subject_id = str(xml_text(edge_node, "subject_id") or "").strip()
        predicate = str(xml_text(edge_node, "predicate") or "").strip()
        object_id = str(xml_text(edge_node, "object_id") or "").strip()
        if not subject_id or not predicate or not object_id:
            raise ValueError("Edge requires subject_id, predicate, and object_id")
        if predicate not in ALLOWED_EDGE_PREDICATES:
            raise ValueError(f"Invalid edge predicate: {predicate}")
        confidence_text = xml_text(edge_node, "confidence")
        confidence: float | None
        if confidence_text:
            try:
                confidence = float(confidence_text)
            except ValueError:
                raise ValueError("Edge confidence must be a number") from None
            if not 0 <= confidence <= 1:
                raise ValueError("Edge confidence must be between 0 and 1")
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
            raise ValueError("Edge invalidation requires subject_id, predicate, and object_id")
        if predicate not in ALLOWED_EDGE_PREDICATES:
            raise ValueError(f"Invalid edge predicate: {predicate}")
        edge_invalidations.append(
            {
                "subject_id": subject_id,
                "predicate": predicate,
                "object_id": object_id,
            }
        )
    return edges, edge_invalidations


def _parse_weekly_reflection_xml(raw: str) -> dict[str, Any]:
    root = extract_xml_fragment(raw, "weekly_reflection")
    if root.attrib or (root.text or "").strip():
        raise ValueError("Expected exact weekly_reflection element")
    if [child.tag for child in root] != ["intentions", "edges", "companion_memory"]:
        raise ValueError("Weekly reflection requires intentions, edges, then companion_memory")

    intentions_node = root.find("intentions")
    if intentions_node is None or intentions_node.attrib or (intentions_node.text or "").strip():
        raise ValueError("Weekly reflection requires exact intentions element")
    intentions: list[dict[str, str]] = []
    for node in intentions_node:
        if node.tag != "intention" or set(node.attrib) != {"id"} or list(node):
            raise ValueError("Intentions may contain only exact intention rows")
        intentions.append({"id": str(node.get("id") or "").strip(), "text": _node_text(node)})
    intentions = validate_intention_replacement(intentions)

    edges, edge_invalidations = _parse_edges(root)
    companion_memory = xml_text(root, "companion_memory")
    if not companion_memory:
        raise ValueError("Weekly reflection requires companion_memory")

    return {
        "intentions": intentions,
        "companion_memory": companion_memory,
        "edges": edges,
        "edge_invalidations": edge_invalidations,
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
                    "### cited_member list (already members; remove only if one no longer belongs)",
                    render_memory_records(statuses["cited"]),
                    "",
                    "### search_result list (add only what clearly belongs)",
                    render_memory_records(statuses["search"]),
                    "",
                    "### purged_member list (mend the prose and citations around these)",
                    render_memory_records(statuses["purged"]),
                    "",
                    "### pending_member list (decide add or remove for each)",
                    render_memory_records(statuses["pending"]),
                ]
            )
        )
    return "\n\n".join(blocks)


def _format_current_intentions_section(value: Any, heading: str) -> str:
    text = format_intentions_for_prompt(value)
    return f"# {heading}\n{text}" if text else ""


def _format_active_dossiers(dossiers: Sequence[Any]) -> str:
    blocks = []
    for dossier in dossiers:
        blocks.append(
            "\n".join(
                (
                    f"--- Dossier: {dossier.name} ---",
                    f"Description: {_read_only_citations(str(dossier.description or ''))}",
                    _read_only_citations(str(dossier.summary or "")),
                )
            ).strip()
        )
    return "\n\n".join(block for block in blocks if block) or "(none)"


def _anchor_prose(bundle: dict[str, Any]) -> str:
    prose = str(bundle["dossier"].summary or "") or "## unlabeled"
    sectioned = label_sections(prose)
    return sectioned[0] if sectioned is not None else prose


def _anchor_context(bundle: dict[str, Any], *, section_labels: bool) -> str:
    dossier = bundle["dossier"]
    prose = _anchor_prose(bundle) if section_labels else str(dossier.summary or "## unlabeled")
    return f"Description: {dossier.description}\n\n{prose}"


def _format_episode_memories(episodes: Sequence[Any]) -> str:
    lines = ["Key: [episode] episodic memory"] if episodes else []
    for item in episodes:
        extra = item.extra if isinstance(item.extra, dict) else {}
        summary = str(extra.get("episode_summary") or item.summary or "").strip()
        lines.append(
            "- " + format_memory_line(
                {
                    "memory_type": "episode",
                    "summary": summary,
                    "happened_at": item.happened_at or item.created_at,
                },
                show_id=True,
                item_id=f"M{item.memory_ref}",
            )
        )
    return "\n".join(lines) or "(none)"


def _render_identity_prompt(
    inputs: dict[str, Any], *, soul_id: str, user_id: str
) -> tuple[str, str]:
    context = inputs["reflection_prompt_context"]
    bundles = inputs["anchor_bundles"]
    statuses = {role: revision_status_items(bundle) for role, bundle in bundles.items()}
    system_prompt = anchors_prompt.SYSTEM_PROMPT.format(soul_name=soul_id, user_name=user_id)
    user_prompt = anchors_prompt.USER_PROMPT.format(
        narrative_self=context["narrative_self"],
        life_goals=context["life_goals"],
        current_intentions_section=_format_current_intentions_section(
            inputs["state"]["intentions_active"], "Current intentions (read-only)"
        ),
        dossier_index=_read_only_citations(inputs["dossier_index"]) or "(none)",
        active_dossiers=_format_active_dossiers(inputs["active_dossiers"]),
        episode_memories=_format_episode_memories(inputs["continuity_episodes"]),
        soul_anchor=_anchor_context(bundles["soul"], section_labels=True),
        soul_anchor_cited_refs="\n".join(
            f"[M{item.memory_ref}]" for item in statuses["soul"]["cited"]
        ) or "(none)",
        soul_anchor_inactive_linked_memory_items=render_memory_records(
            statuses["soul"]["purged"]
        ),
        user_anchor=_anchor_context(bundles["user"], section_labels=True),
        user_anchor_cited_refs="\n".join(
            f"[M{item.memory_ref}]" for item in statuses["user"]["cited"]
        ) or "(none)",
        user_anchor_inactive_linked_memory_items=render_memory_records(
            statuses["user"]["purged"]
        ),
        user_name=user_id,
    )
    return system_prompt, user_prompt


def _project_life_goals(
    active: list[str], additions: list[str], removals: list[str]
) -> list[str]:
    result = [goal for goal in active if goal not in removals]
    for goal in additions:
        if goal not in result and len(result) < 3:
            result.append(goal)
    return result


def _render_weekly_prompt(
    inputs: dict[str, Any],
    identity: dict[str, Any],
    *,
    soul_id: str,
    user_id: str,
) -> tuple[str, str]:
    context = inputs["reflection_prompt_context"]
    decisions = identity["anchor_decisions"]
    active_goals = _project_life_goals(
        inputs["active_life_goals"], identity["life_goal_add"], identity["life_goal_remove"]
    )
    system_prompt = weekly_prompt.SYSTEM_PROMPT.format(soul_name=soul_id, user_name=user_id)
    user_prompt = weekly_prompt.USER_PROMPT.format(
        narrative_self=identity["narrative_self"] or context["narrative_self"],
        soul_anchor=(
            f"Description: {decisions['soul']['description']}\n\n"
            + _read_only_citations(decisions["soul"]["resulting_prose"])
        ),
        user_anchor=(
            f"Description: {decisions['user']['description']}\n\n"
            + _read_only_citations(decisions["user"]["resulting_prose"])
        ),
        life_goals=_format_life_goals_for_prompt(active_goals, []),
        current_intentions_section=_format_current_intentions_section(
            inputs["state"]["intentions_active"],
            "Current intentions (keep, reword, or leave out by ID)",
        ),
        dossier_index=_read_only_citations(inputs["dossier_index"]) or "(none)",
        prior_context_memory_items=context["prior_context_memory_items"],
        conversation_history=context["conversation_history"],
        segment_memory_items=context["segment_memory_items"],
        existing_memory_edges=context["existing_memory_edges"],
    )
    return system_prompt, user_prompt


# Marcos' reminder: a category is a dossier.
async def prepare_dossier_consolidation_context(
    svc: MemoryService,
    *,
    inputs: dict[str, Any],
    soul_id: str,
    user_id: str,
    llm_profile: str | None = None,
) -> dict[str, Any]:
    revision_profile = svc.memorize_config.category_update_llm_profile
    scope = {"soul_id": soul_id, "user_id": user_id}
    prompt_context = _build_consolidation_prompt_context(inputs, soul_id=soul_id)
    inputs["reflection_prompt_context"] = prompt_context
    bundles = [
        svc.prepare_dossier_revision(
            dossier.id,
            scope,
            narrative_self=inputs.get("narrative_self"),
            active_life_goals=inputs["active_life_goals"],
            removed_life_goals=inputs["removed_life_goals"],
        )
        for dossier in svc.list_due_dossiers(
            scope,
            segment_ids=inputs["selected_segment_ids"],
        )
    ]
    inputs["dossier_index"] = svc.build_dossier_index(scope)
    continuity = svc.prepare_anchor_continuity_context(scope)
    inputs["active_dossiers"] = continuity["dossiers"]
    inputs["continuity_episodes"] = continuity["episodes"]
    episode_ids = [item.id for item in continuity["episodes"]]
    inputs["anchor_bundles"] = {
        role: svc.prepare_anchor_revision(role, scope, episode_ids)
        for role in ("soul", "user")
    }

    anchors = inputs["anchor_bundles"]
    system_prompt = dossiers_prompt.SYSTEM_PROMPT.format(soul_name=soul_id, user_name=user_id)
    literal_fields = {
        name: "{" + name + "}"
        for name in (
            "dossier_id", "dossier_kind", "dossier_title", "target_words",
            "dossier_description", "current_prose", "cited_memory_records",
            "candidate_memory_records", "cleanup_memberships", "required_memory_records",
        )
    }
    user_prompt = dossiers_prompt.USER_PROMPT.format(
        narrative_self=prompt_context["narrative_self"],
        soul_anchor=_read_only_citations(_anchor_context(anchors["soul"], section_labels=False)),
        user_anchor=_read_only_citations(_anchor_context(anchors["user"], section_labels=False)),
        life_goals=_format_life_goals_for_prompt(inputs["active_life_goals"], []),
        current_intentions_section=_format_current_intentions_section(
            inputs["state"]["intentions_active"], "Current intentions (read-only)"
        ),
        dossier_index=_read_only_citations(inputs["dossier_index"]) or "(none)",
        active_dossiers=_format_active_dossiers(inputs["active_dossiers"]),
        prior_context_memory_items=prompt_context["prior_context_memory_items"],
        conversation_history=prompt_context["conversation_history"],
        segment_memory_items=prompt_context["segment_memory_items"],
        dossier_revision_blocks=_format_dossier_revision_blocks(bundles),
        **literal_fields,
    )

    identity_system, identity_user = _render_identity_prompt(
        inputs, soul_id=soul_id, user_id=user_id
    )
    current_identity: dict[str, Any] = {
        "narrative_self": inputs.get("narrative_self"),
        "life_goal_add": [],
        "life_goal_remove": [],
        "anchor_decisions": {
            role: {
                "description": bundle["dossier"].description,
                "resulting_prose": _anchor_prose(bundle),
            }
            for role, bundle in anchors.items()
        },
    }
    weekly_system, weekly_user = _render_weekly_prompt(
        inputs, current_identity, soul_id=soul_id, user_id=user_id
    )
    estimates = {
        "dossiers": estimate_prompt_tokens(system_prompt + "\n" + user_prompt) if bundles else 0,
        "anchors": estimate_prompt_tokens(identity_system + "\n" + identity_user),
        "weekly": estimate_prompt_tokens(weekly_system + "\n" + weekly_user),
    }
    for stage, tokens in estimates.items():
        if tokens > CONSOLIDATION_PROMPT_TOKEN_LIMIT:
            raise ValueError(f"{stage} consolidation prompt exceeds provider-safe token limit")
    log.info(
        "consolidation prompt estimates: dossiers=%d anchors=%d weekly=%d combined=%d",
        estimates["dossiers"], estimates["anchors"], estimates["weekly"], sum(estimates.values()),
    )

    if bundles:
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
    continuity = svc.prepare_anchor_continuity_context(scope)
    inputs["active_dossiers"] = continuity["dossiers"]
    inputs["continuity_episodes"] = continuity["episodes"]
    episode_ids = [item.id for item in continuity["episodes"]]
    inputs["anchor_bundles"] = {
        role: svc.prepare_anchor_revision(role, scope, episode_ids)
        for role in ("soul", "user")
    }
    return inputs


def preflight_consolidation_profiles(
    svc: MemoryService,
    consolidation_llm_profile: str | None,
) -> str:
    revision_profile = str(svc.memorize_config.category_update_llm_profile)
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


def consolidation_due(
    last_consolidation_at: Any,
    *,
    interval_days: int,
    now: datetime | None = None,
) -> bool:
    last = parse_iso_datetime(last_consolidation_at)
    return last is None or (now or datetime.now(UTC)) >= last + timedelta(days=max(1, int(interval_days)))


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
    id_map: dict[str, str] = {}

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
    refs_by_id = {item_id: ref for ref, item_id in id_map.items()}
    existing_edges = "\n".join(
        f"- [{refs_by_id[row['subject_id']]}] {row['predicate']} [{refs_by_id[row['object_id']]}]"
        for row in inputs.get("existing_memory_edges") or []
        if row["subject_id"] in refs_by_id and row["object_id"] in refs_by_id
    ) or "(none)"
    narrative = str(inputs.get("narrative_self") or "").strip()
    return {
        "narrative_self": narrative or "(none)",
        "life_goals": life_goals_text,
        "prior_context_memory_items": prior_context_text,
        "conversation_history": str(inputs.get("all_chat_history") or "").strip() or "(none)",
        "segment_memory_items": segment_memory_items_text,
        "existing_memory_edges": existing_edges,
        "id_map": id_map,
    }


def _resolve_memory_ref(raw_value: Any, id_map: dict[str, str]) -> str | None:
    text = str(raw_value or "").strip()
    match = re.fullmatch(r"\[M([1-9][0-9]*)\]", text)
    if match is None:
        return None
    return id_map.get(f"M{match.group(1)}")


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
    force: bool = False,
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
        state = deps.conversation_state_from_row(
            deps.conversation_state_row(
                con, conversation_id, user_id=user_id, soul_id=soul_id
            ),
            con=con,
        )
        if state is None:
            raise HTTPException(status_code=404, detail="conversation state not found")

        pending_by_conversation: dict[str, list[str]] = {}
        for row in con.execute(
            "SELECT conversation_id, pending_segment_ids FROM conversations "
            "WHERE soul_id = ? AND user_id = ? ORDER BY conversation_id",
            (soul_id, user_id),
        ).fetchall():
            pending_ids = deps.normalize_text_list(row["pending_segment_ids"])
            if not pending_ids:
                continue
            owner = str(row["conversation_id"] or "").strip()
            if canonical_conversation_id(owner) != owner:
                raise HTTPException(
                    status_code=400,
                    detail=f"pending consolidation owner is not canonical: {owner}",
                )
            pending_by_conversation[owner] = pending_ids
        if not pending_by_conversation:
            return {"status": "skip", "reason": "no_pending_segments"}
        now = datetime.now(UTC)
        soul_state = _soul_state.read(con)
        last_error_at = parse_iso_datetime(soul_state.get("last_consolidation_error_at"))
        last_success_at = parse_iso_datetime(soul_state.get("last_consolidation_at"))
        if (
            not force
            and last_error_at is not None
            and (last_success_at is None or last_error_at > last_success_at)
        ):
            return {"status": "skip", "reason": "failure_requires_retry"}

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

        narrative_self = str(state.get("narrative_self") or "").strip() or None

        segment_inputs: list[dict[str, Any]] = []
        current_chat_messages: list[dict[str, Any]] = []
        selected_by_conversation: dict[str, list[str]] = {}
        storage_dir = deps.get_storage_dir(deps.config)
        chats_dir = (storage_dir / "st_chats").resolve()
        for pending_conversation_id, pending_segment_ids in pending_by_conversation.items():
            chat_dir = deps.find_chat_dir_for_conversation(
                chats_dir, user_id, soul_id, pending_conversation_id
            )
            if chat_dir is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"conversation resource not found: {pending_conversation_id}",
                )
            manifest_path = (chat_dir / "manifest.json").resolve()
            if not manifest_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"conversation manifest not found: {pending_conversation_id}",
                )
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"conversation manifest unreadable: {pending_conversation_id}",
                ) from exc
            raw_segments = manifest.get("segments") if isinstance(manifest, dict) else None
            if not isinstance(raw_segments, list) or not raw_segments:
                raise HTTPException(
                    status_code=400,
                    detail=f"conversation manifest has no segments: {pending_conversation_id}",
                )
            messages: list[dict[str, Any]] = []
            segments_dir = (chat_dir / "segments").resolve()
            if segments_dir.is_dir():
                for ep_file in sorted(segments_dir.glob("*.json"), key=_segment_file_sort_key):
                    try:
                        parsed = json.loads(ep_file.read_text(encoding="utf-8"))
                    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                        raise HTTPException(
                            status_code=400,
                            detail=f"segment history unreadable: {ep_file}",
                        ) from exc
                    if not isinstance(parsed, list):
                        raise HTTPException(
                            status_code=400,
                            detail=f"segment history is not a message list: {ep_file}",
                        )
                    if any(not isinstance(message, dict) for message in parsed):
                        raise HTTPException(
                            status_code=400,
                            detail=f"segment history contains a non-message row: {ep_file}",
                        )
                    messages.extend(parsed)
            try:
                conversation_segments = build_segment_inputs(messages, pending_segment_ids)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if len(conversation_segments) != len(pending_segment_ids):
                raise HTTPException(
                    status_code=400,
                    detail=f"queued segments are not present in conversation history: {pending_conversation_id}",
                )
            selected_by_conversation[pending_conversation_id] = list(pending_segment_ids)
            current_chat_messages.extend(_messages_for_segment_inputs(messages, conversation_segments))
            for entry in conversation_segments:
                segment_id = str(entry["segment_id"])
                entry["conversation_id"] = pending_conversation_id
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
""",
                    (soul_id, user_id, pending_conversation_id, segment_id),
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
            segment_inputs.extend(conversation_segments)

        selected_segment_ids = [
            segment_id
            for pending_ids in selected_by_conversation.values()
            for segment_id in pending_ids
        ]

        prior_context_memory_items: list[dict[str, Any]] = []
        all_prior_context_ids: list[str] = []
        last_consol = state.get("last_consolidation_at")
        try:
            if last_consol:
                res_rows = con.execute(
                    "SELECT memory_prior_context FROM resources WHERE soul_id = ? AND user_id = ? "
                    "AND julianday(created_at) >= julianday(?) AND memory_prior_context IS NOT NULL",
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
                f"""
SELECT id, memory_ref, memory_type, summary, happened_at, created_at
FROM memory_items
WHERE id IN ({placeholders}) AND soul_id = ? AND user_id = ?
  AND (merged_into IS NULL OR TRIM(merged_into) = '')
  AND NOT EXISTS (
    SELECT 1 FROM triples t
    WHERE t.subject_id = memory_items.id
      AND t.predicate = 'evolved_into'
      AND t.valid_to IS NULL
  )
""",
                tuple(clean_ids) + (soul_id, user_id),
            ).fetchall()
            items_by_id: dict[str, dict[str, Any]] = {}
            prior_context_memory_items = []
            for row in ret_rows:
                mid = str(row["id"] or "").strip()
                summary = str(row["summary"] or "").strip()
                if not mid or not summary:
                    continue
                prior_entry: dict[str, Any] = {
                    "id": mid,
                    "memory_ref": int(row["memory_ref"] or 0),
                    "summary": summary,
                    "memory_type": str(row["memory_type"] or "").strip(),
                    "happened_at": row["happened_at"] or row["created_at"],
                }
                items_by_id[mid] = prior_entry
                prior_context_memory_items.append(prior_entry)

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

        evidence_ids = sorted({
            str(item["id"])
            for group in segment_inputs
            for item in group.get("memory_summaries") or []
            if isinstance(item, dict)
        } | {str(item["id"]) for item in prior_context_memory_items})
        existing_memory_edges: list[dict[str, str]] = []
        if evidence_ids:
            evidence_id_set = set(evidence_ids)
            edge_rows = con.execute(
                "SELECT subject_id, predicate, object_id FROM triples "
                "WHERE predicate IN ('caused_by', 'evokes', 'conflicts_with', 'parallels', 'shaped_by') "
                "AND valid_to IS NULL ORDER BY subject_id, predicate, object_id"
            ).fetchall()
            existing_memory_edges = [
                {
                    "subject_id": str(row["subject_id"]),
                    "predicate": str(row["predicate"]),
                    "object_id": str(row["object_id"]),
                }
                for row in edge_rows
                if str(row["subject_id"]) in evidence_id_set
                and str(row["object_id"]) in evidence_id_set
            ]

        return {
            "status": "ready",
            "db_path": db_path,
            "state": state,
            "active_life_goals": active_goals,
            "removed_life_goals": removed_goals,
            "segment_inputs": segment_inputs,
            "current_chat_messages": current_chat_messages,
            "narrative_self": narrative_self,
            "last_consolidation_at": state.get("last_consolidation_at"),
            "started_at": now.isoformat(),
            "prior_context_memory_items": prior_context_memory_items,
            "existing_memory_edges": existing_memory_edges,
            "selected_segment_ids": selected_segment_ids,
            "selected_segment_ids_by_conversation": selected_by_conversation,
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
    context = inputs["reflection_prompt_context"]
    id_map = context["id_map"]
    anchor_bundles = inputs["anchor_bundles"]

    identity_system, identity_user = _render_identity_prompt(
        inputs, soul_id=soul_id, user_id=user_id
    )
    if estimate_prompt_tokens(identity_system + "\n" + identity_user) > CONSOLIDATION_PROMPT_TOKEN_LIMIT:
        raise ValueError("anchors consolidation prompt exceeds provider-safe token limit")
    identity_raw = await svc.chat(
        identity_user,
        profile=llm_profile,
        system_prompt=identity_system,
        op="consolidation",
        step="anchors",
    )
    identity = _parse_identity_maintenance_xml(
        str(identity_raw or ""), anchor_bundles
    )
    if not str(inputs.get("narrative_self") or "").strip() and not identity["narrative_self"]:
        raise ValueError("First identity maintenance requires narrative_self replacement")

    weekly_system, weekly_user = _render_weekly_prompt(
        inputs, identity, soul_id=soul_id, user_id=user_id
    )
    if estimate_prompt_tokens(weekly_system + "\n" + weekly_user) > CONSOLIDATION_PROMPT_TOKEN_LIMIT:
        raise ValueError("weekly consolidation prompt exceeds provider-safe token limit")
    weekly_raw = await svc.chat(
        weekly_user,
        profile=llm_profile,
        system_prompt=weekly_system,
        op="consolidation",
        step="weekly",
    )
    weekly = _parse_weekly_reflection_xml(str(weekly_raw or ""))
    remapped_edges = _remap_edges_with_memory_ids(weekly["edges"], id_map=id_map, include_confidence=True)
    remapped_invalidations = _remap_edges_with_memory_ids(
        weekly["edge_invalidations"],
        id_map=id_map,
        include_confidence=False,
    )
    if len(remapped_edges) != len(weekly["edges"]):
        raise ValueError("Weekly reflection edge references memory outside supplied evidence")
    if len(remapped_invalidations) != len(weekly["edge_invalidations"]):
        raise ValueError("Weekly reflection invalidation references memory outside supplied evidence")

    new_narrative = str(identity["narrative_self"] or "").strip() or None
    current_narrative = str(inputs.get("narrative_self") or "").strip() or None
    snapshot_old_narrative = bool(current_narrative and new_narrative and current_narrative != new_narrative)
    embed_inputs: list[str] = []
    if weekly["companion_memory"]:
        embed_inputs.append(weekly["companion_memory"])
    if snapshot_old_narrative:
        assert current_narrative is not None
        embed_inputs.append(current_narrative)
    embeddings = await svc.embed(embed_inputs, profile="embedding") if embed_inputs else []
    if len(embeddings) != len(embed_inputs):
        raise ValueError("Consolidation embedding count does not match inputs")

    cursor = 0
    companion_embedding = None
    if weekly["companion_memory"]:
        companion_embedding = embeddings[cursor]
        cursor += 1
    old_narrative_embedding = embeddings[cursor] if snapshot_old_narrative else None

    scope = {"soul_id": soul_id, "user_id": user_id}
    for role in ("soul", "user"):
        await svc.apply_anchor_revision(
            anchor_bundles[role],
            identity["anchor_decisions"][role],
            scope,
        )

    return {
        "narrative_self": new_narrative,
        "life_goal_add": identity["life_goal_add"],
        "life_goal_remove": identity["life_goal_remove"],
        "companion_memory": weekly["companion_memory"],
        "companion_embedding": companion_embedding,
        "old_narrative_text": current_narrative if snapshot_old_narrative else None,
        "old_narrative_embedding": old_narrative_embedding,
        "edges": remapped_edges,
        "edge_invalidations": remapped_invalidations,
        "intentions_snapshot": inputs["state"]["intentions_active"],
        "intentions_replacement": weekly["intentions"],
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
    # ponytail: rare partial-write retries may duplicate outputs; add idempotency if observed.
    db_path: Path = inputs["db_path"]
    now_iso = datetime.now(UTC).isoformat()
    started_at = str(inputs.get("started_at") or "").strip() or now_iso

    narrative_id = str(uuid.uuid4())
    narrative_self = str(llm_results.get("narrative_self") or "").strip() or None

    old_narrative_text = llm_results.get("old_narrative_text")
    companion_memory_id = None
    companion_text = str(llm_results.get("companion_memory") or "").strip()
    companion_embedding = llm_results.get("companion_embedding")

    deps.sqlite_ensure_nonempty(db_path)
    con = deps.sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        deps.sqlite_ensure_conversation_state_schema(con)

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
        goals_to_restore: list[str] = []
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
            goal_id = removed_ids.pop(text, None)
            if goal_id is None:
                goal_id = str(uuid.uuid4())
                goals_to_add.append((goal_id, text))
            else:
                goals_to_restore.append(goal_id)
            active_ids[text] = goal_id
            active_goal_count += 1

    finally:
        con.close()

    selected_by_conversation = {
        str(cid): [str(segment_id) for segment_id in segment_ids]
        for cid, segment_ids in (inputs.get("selected_segment_ids_by_conversation") or {}).items()
    }
    if not selected_by_conversation:
        selected_by_conversation = {
            conversation_id: [
                str(segment_id)
                for segment_id in (inputs.get("selected_segment_ids") or [])
            ]
        }
    # Preflight every state row before companion and graph side effects.
    for pending_conversation_id in selected_by_conversation:
        deps.write_conversation_state(
            pending_conversation_id,
            soul_id=soul_id,
            user_id=user_id,
            updates={},
        )

    if old_narrative_text:
        check = deps.sqlite_connect(db_path)
        try:
            check.row_factory = sqlite3.Row
            current = _soul_state.read(check)
        finally:
            check.close()
        if (
            int(current["summaries_revision"]) != int(inputs["state"]["summaries_revision"])
            or str(current["narrative_self"] or "") != str(inputs.get("narrative_self") or "")
        ):
            raise ValueError("summary_snapshot_stale")
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
            embedding=cast(list[float], companion_embedding),
            happened_at=companion_happened_at,
        )

    scope = {"user_id": user_id, "soul_id": soul_id}
    wrote = write_memory_edges(svc.database.triple_repo, llm_results["edges"], scope=scope)
    invalidated = invalidate_memory_edges(svc.database.triple_repo, llm_results["edge_invalidations"], scope=scope)
    consumed_segment_ids = [
        str(segment_id).strip()
        for segment_id in (inputs.get("selected_segment_ids") or [])
        if str(segment_id).strip()
    ]
    con = deps.sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        con.execute("BEGIN IMMEDIATE")
        current_intentions = _soul_state.read(con)["intentions_active"]
        merged_intentions = merge_consolidated_intentions(
            llm_results["intentions_snapshot"],
            current_intentions,
            llm_results["intentions_replacement"],
        )
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
                scope=scope,
                edited_by="consolidation",
                expected_revision=int(inputs["state"]["summaries_revision"]),
                displayed_summary=str(inputs.get("narrative_self") or ""),
                journal=False,
            )

        for goal_id in goals_to_mark_removed:
            con.execute(
                "UPDATE life_goals SET status = 'removed', updated_at = ? WHERE id = ?",
                (now_iso, goal_id),
            )
        for goal_id in goals_to_delete:
            con.execute("DELETE FROM life_goals WHERE id = ?", (goal_id,))
        for goal_id in goals_to_restore:
            con.execute(
                "UPDATE life_goals SET status = 'active', updated_at = ? WHERE id = ?",
                (now_iso, goal_id),
            )
        for goal_id, text in goals_to_add:
            con.execute(
                """
INSERT INTO life_goals (
    id, soul_id, user_id, description, status, updated_at
) VALUES (?, ?, ?, ?, 'active', ?)
""",
                (goal_id, soul_id, user_id, text, now_iso),
            )

        for pending_conversation_id, pending_ids in selected_by_conversation.items():
            if pending_conversation_id == conversation_id:
                continue
            deps.write_conversation_state(
                pending_conversation_id,
                soul_id=soul_id,
                user_id=user_id,
                updates={"remove_pending_segment_ids": pending_ids},
                connection=con,
            )

        state_after, _ = deps.write_conversation_state(
            conversation_id,
            soul_id=soul_id,
            user_id=user_id,
            updates={
                # Subtract only what this run consumed: a memorize that finished
                # during the LLM phase may have appended new pending ids.
                "remove_pending_segment_ids": selected_by_conversation.get(
                    conversation_id, []
                ),
                "last_consolidation_at": started_at,
                "last_consolidation_error": None,
                "last_consolidation_error_at": None,
                "intentions_active": merged_intentions,
                "remove_retrieval_ids_since_consolidation": inputs.get("state", {}).get(
                    "retrieval_ids_since_consolidation", []
                ),
                "remove_prior_context_ids_since_consolidation": inputs.get("state", {}).get(
                    "prior_context_ids_since_consolidation", []
                ),
            },
            connection=con,
        )
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    previous_narrative = str(inputs.get("narrative_self") or "")
    if narrative_self and narrative_self != previous_narrative:
        try:
            _soul_summaries.append_summary_journal(
                kind="narrative_self",
                summary_id="soul-summary:narrative_self",
                summary_before=previous_narrative,
                summary_after=narrative_self,
                scope=scope,
                edited_by="consolidation",
            )
        except Exception:
            log.exception("Failed to journal committed consolidation narrative")

    return {
        "conversation_id": conversation_id,
        "narrative_id": narrative_id if llm_results.get("narrative_self") else None,
        "companion_memory_id": companion_memory_id,
        "edges_written": wrote,
        "edges_invalidated": invalidated,
        "consumed_segment_ids": consumed_segment_ids,
        "state": state_after,
    }


def _record_consolidation_failure(
    *,
    deps: ConsolidationDeps,
    soul_id: str,
    user_id: str,
    exc: Exception,
) -> None:
    now_iso = datetime.now(UTC).isoformat()
    error = f"{type(exc).__name__}: {str(exc)[:260]}"
    db_path = deps.sqlite_current_path(user_id, soul_id)
    if db_path is None or not db_path.exists():
        raise FileNotFoundError(f"soul database not found: {soul_id}")
    con = deps.sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        _soul_state.ensure_schema(con)
        _soul_state.write(
            con,
            {
                "last_consolidation_error": error,
                "last_consolidation_error_at": now_iso,
            },
        )
        con.commit()
    finally:
        con.close()


async def _run_consolidation_pipeline_once(
    *,
    svc: Any,
    deps: ConsolidationDeps,
    state_lock: asyncio.Lock,
    running: set[tuple[str, str]],
    load_cross_tail_for_ai: Callable[..., Any],
    format_all_chat_history_for_ai: Callable[..., str],
    conversation_id: str,
    soul_id: str,
    user_id: str,
    force: bool = False,
) -> dict[str, Any]:
    run_key = (user_id, soul_id)
    if run_key in running:
        return {"status": "skipped", "reason": "in_progress"}
    running.add(run_key)
    try:
        async with state_lock:
            prep = gather_consolidation_inputs(
                deps,
                conversation_id=conversation_id,
                soul_id=soul_id,
                user_id=user_id,
                force=force,
            )
        if prep.get("status") == "skip":
            return {"status": "skipped", "reason": prep.get("reason")}
        consolidation_profile = _service_factory._resolve_profile_if_configured(svc, "consolidation")
        preflight_consolidation_profiles(svc, consolidation_profile)
        current_chat_messages = [
            row for row in (prep.get("current_chat_messages") or [])
            if isinstance(row, dict)
        ]
        prep["all_chat_history"] = format_all_chat_history_for_ai(
            current_history=current_chat_messages,
            cross_tail=load_cross_tail_for_ai(
                user_id=user_id,
                soul_id=soul_id,
                conversation_id=conversation_id,
            ),
            conversation_id=conversation_id,
            soul_id=soul_id,
            mark_current_chat=False,
        )
        await prepare_dossier_consolidation_context(
            svc,
            inputs=prep,
            soul_id=soul_id,
            user_id=user_id,
            llm_profile=consolidation_profile,
        )

        consolidation_llm = await run_consolidation_llm(
            svc,
            inputs=prep,
            soul_id=soul_id,
            user_id=user_id,
            llm_profile=consolidation_profile,
        )
        async with state_lock:
            result = write_consolidation_outputs(
                deps,
                svc,
                inputs=prep,
                llm_results=consolidation_llm,
                conversation_id=conversation_id,
                soul_id=soul_id,
                user_id=user_id,
            )
        return {"status": "ok", "result": result}
    except Exception as exc:
        db_path = deps.sqlite_current_path(user_id, soul_id)
        if db_path is not None and db_path.exists():
            try:
                async with state_lock:
                    _record_consolidation_failure(
                        deps=deps,
                        soul_id=soul_id,
                        user_id=user_id,
                        exc=exc,
                    )
            except Exception:
                log.exception(
                    "failed to record consolidation error state for %s",
                    conversation_id,
                )
        raise
    finally:
        running.discard(run_key)
