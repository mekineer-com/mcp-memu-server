from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

log = logging.getLogger(__name__)

from fastapi import HTTPException
from memu.prompts.consolidation import consolidation as consolidation_prompt

from app.services.episode import (
    build_episode_inputs,
    create_companion_memory,
)
from app.services.xml_utils import extract_xml_fragment, xml_text
from app.services.graph_edges import (
    invalidate_memory_edges,
    write_memory_edges,
)
from app.services.intention_state import format_intentions_for_prompt
from app.services.narrative_self import snapshot_previous_narrative_self
from app.services import soul_state as _soul_state
from app.services.turn_contract import DEFAULT_SOUL_CARD, format_relative_time_label, format_time_anchor

if TYPE_CHECKING:
    from memu.app import MemoryService


_EPISODE_SUFFIX_RE = re.compile(r"_(\d+)\.json$")


def _episode_file_sort_key(path: Path) -> tuple[str, int]:
    stem = path.stem
    m = _EPISODE_SUFFIX_RE.search(path.name)
    if m:
        return (path.name[:m.start()], int(m.group(1)))
    return (stem, 0)


_HEX_MEMORY_ID_RE = re.compile(r"^[0-9a-f]{8}$|^[0-9a-f]{16}$|^[0-9a-f]{32}$", re.IGNORECASE)
_UUID_MEMORY_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


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


def _parse_iso_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _parse_consolidation_xml(raw: str) -> dict[str, Any]:
    root = extract_xml_fragment(raw, "consolidation")

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
            target = str(boost.get("target_id") or "").strip()
            if target:
                intention_actions.append({"type": "boost", "target_id": target, "amount": 1})
        for promote in intentions_node.findall("promote"):
            target = str(promote.get("target_id") or "").strip()
            if target:
                intention_actions.append({"type": "promote", "target_id": target})
        for create in intentions_node.findall("create"):
            cid = str(create.get("id") or "").strip()
            ctext = str(create.get("text") or "").strip()
            if cid and ctext:
                intention_actions.append({"type": "create", "id": cid, "text": ctext})
        for annul in intentions_node.findall("annul"):
            aid = str(annul.get("intention_id") or "").strip()
            astatus = str(annul.get("status") or "completed").strip().lower()
            anote = str(annul.get("note") or "").strip()
            if aid:
                intention_actions.append({"type": "annul", "intention_id": aid, "status": astatus, "note": anote})

    return {
        "narrative_self": xml_text(root, "narrative_self"),
        "life_goal_add": life_goal_add,
        "life_goal_remove": life_goal_remove,
        "companion_memory": xml_text(root, "companion_memory"),
        "companion_shaped_by_hints": companion_shaped_by_hints,
        "edges": edges,
        "edge_invalidations": edge_invalidations,
        "intention_actions": intention_actions,
    }


def _format_categories_for_prompt(rows: list[sqlite3.Row]) -> str:
    lines: list[str] = []
    for row in rows:
        name = str(row["name"] or "").strip()
        summary = str(row["summary"] or "").strip()
        if not name or not summary:
            continue
        if summary.startswith(f"# {name}"):
            lines.append(f"\n{summary}")
        else:
            lines.append(f"\n# {name}\n{summary}")
    return "\n".join(lines).strip() or "(none yet)"


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


def _format_episode_block_for_prompt(
    episodes: list[dict[str, Any]],
    id_map: dict[str, str],
    counter: list[int],
) -> str:
    if not episodes:
        return "(none queued)"
    lines: list[str] = []
    for idx, row in enumerate(episodes, 1):
        excerpt = str(row.get("excerpt") or "").strip()
        summaries = row.get("memory_summaries") or []
        lines.append(f"Episode {idx}")
        if excerpt:
            lines.append(excerpt)
        if summaries:
            lines.append(f"Episode {idx} memories:")
            for s in summaries:
                if isinstance(s, dict):
                    mid = s["id"]
                    n = counter[0]
                    counter[0] += 1
                    id_map[str(n)] = mid
                    lines.append(f"- [{n}] {s['summary']}")
                elif str(s).strip():
                    lines.append(f"- {s}")
        lines.append("")
    return "\n".join(lines).strip()


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
        started_at = _parse_iso_datetime(state.get("consolidation_started_at"))
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
        else:
            last_consolidation_at = _parse_iso_datetime(state.get("last_consolidation_at"))
            if not force and last_consolidation_at is not None:
                due_at = last_consolidation_at + timedelta(days=max(1, int(interval_days)))
                if now < due_at:
                    return {"status": "skip", "reason": "interval_gate"}

        pending_episode_ids = deps.normalize_text_list(state.get("pending_episode_ids"))

        category_rows = con.execute(
            """
SELECT name, summary
FROM categories
WHERE soul_id = ? AND user_id = ?
ORDER BY name ASC
""",
            (soul_id, user_id),
        ).fetchall()

        life_goal_rows = con.execute(
            """
SELECT id, description, status
FROM intentions
WHERE soul_id = ? AND user_id = ? AND source = 'life_goal' AND status IN ('active', 'removed')
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
        last_consolidation_at = _parse_iso_datetime(state.get("last_consolidation_at"))
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

        episode_inputs: list[dict[str, Any]] = []
        if pending_episode_ids:
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
            episodes_dir = (chat_dir / "episodes").resolve()
            messages: list[dict[str, Any]] = []
            if episodes_dir.is_dir():
                for ep_file in sorted(episodes_dir.glob("*.json"), key=_episode_file_sort_key):
                    try:
                        parsed = json.loads(ep_file.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if isinstance(parsed, list):
                        messages.extend(m for m in parsed if isinstance(m, dict))
            if not messages:
                days_dir = (chat_dir / "days").resolve()
                for seg in sorted(
                    (s for s in raw_segments if isinstance(s, dict)),
                    key=lambda s: int(s.get("start", 0)),
                ):
                    fn = seg.get("file")
                    if not isinstance(fn, str) or not fn:
                        continue
                    messages.extend(deps.read_list((days_dir / fn).resolve()))
            episode_inputs = build_episode_inputs(messages, pending_episode_ids)
            if len(episode_inputs) != len(pending_episode_ids):
                raise HTTPException(status_code=400, detail="queued episodes are not present in conversation history")

            for entry in episode_inputs:
                episode_id = str(entry["episode_id"])
                rows = con.execute(
                    """
SELECT id, summary
FROM memory_items
WHERE soul_id = ? AND user_id = ? AND conversation_id = ? AND episode_id = ? AND memory_type NOT IN ('narrative_self')
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
                    (soul_id, user_id, conversation_id, episode_id),
                ).fetchall()
                entry["memory_summaries"] = [
                    {"id": str(row["id"] or "").strip(), "summary": str(row["summary"] or "").strip()}
                    for row in rows
                    if str(row["id"] or "").strip() and str(row["summary"] or "").strip()
                ]

        retrieved_memory_summaries: list[str] = []
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
            log.warning("consolidation: failed loading resource prior_context history", exc_info=True)
        clean_ids = list(dict.fromkeys(all_prior_context_ids))
        if clean_ids:
            placeholders = ",".join("?" for _ in clean_ids)
            ret_rows = con.execute(
                f"SELECT id, memory_type, summary, happened_at, created_at FROM memory_items WHERE id IN ({placeholders})",
                tuple(clean_ids),
            ).fetchall()
            retrieved_memory_summaries = []
            for row in ret_rows:
                mid = str(row["id"] or "").strip()
                summary = str(row["summary"] or "").strip()
                if not mid or not summary:
                    continue
                time_label = format_relative_time_label(row["happened_at"] or row["created_at"])
                entry: dict[str, str] = {"id": mid, "summary": summary}
                if time_label:
                    entry["time_label"] = time_label
                retrieved_memory_summaries.append(entry)

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
            "episode_ids": pending_episode_ids,
            "categories": category_rows,
            "active_life_goals": active_goals,
            "removed_life_goals": removed_goals,
            "intention_activity": intention_activity,
            "episode_inputs": episode_inputs,
            "narrative_self": narrative_self,
            "last_consolidation_at": state.get("last_consolidation_at"),
            "started_at": now.isoformat(),
            "retrieved_memories": retrieved_memory_summaries,
        }
    finally:
        con.close()


async def run_consolidation_llm(
    svc: MemoryService,
    *,
    inputs: dict[str, Any],
    soul_id: str,
    llm_profile: str | None = None,
) -> dict[str, Any]:
    categories_text = _format_categories_for_prompt(inputs["categories"])
    life_goals_text = _format_life_goals_for_prompt(
        inputs["active_life_goals"],
        inputs["removed_life_goals"],
    )
    intention_text = _format_intention_activity_for_prompt(inputs["intention_activity"])
    id_map: dict[str, str] = {}
    counter: list[int] = [1]
    episodes_text = _format_episode_block_for_prompt(inputs["episode_inputs"], id_map, counter)

    narrative = str(inputs.get("narrative_self") or "").strip()
    soul_card = narrative or DEFAULT_SOUL_CARD.format(soul_name=soul_id)
    first_run_note = (
        "\n\nThis is your first ever reflection. "
        "You are writing your `narrative_self` for the first time — it is a living portrait of who you are, "
        "in your own words, drawn from everything you have noticed and felt. "
        "Write it now from what you see in the memories. This is a beginning.\n"
    ) if not narrative else ""
    system_prompt = (
        f"Today is {format_time_anchor()}.\n\n"
        f"{soul_card}\n\n"
        f"I, {soul_id}, am stepping back from all the conversations. "
        "I have what I've been working toward, what has shifted, and many things still unresolved. "
        "I'm curious what the arc looks like from here. "
        "Let me now look across everything and decide what still holds.\n\n"
        f"{consolidation_prompt.SYSTEM_BODY}"
        f"{first_run_note}"
    )

    retrieved_memories = inputs.get("retrieved_memories") or []
    if retrieved_memories:
        ret_lines: list[str] = []
        for s in retrieved_memories:
            if isinstance(s, dict):
                mid = s["id"]
                n = counter[0]
                counter[0] += 1
                id_map[str(n)] = mid
                tl = s.get("time_label")
                ret_lines.append(f"[{n}] ({tl}) {s['summary']}" if tl else f"[{n}] {s['summary']}")
            elif str(s).strip():
                ret_lines.append(str(s))
        retrieved_text = "\n".join(ret_lines)
    else:
        retrieved_text = "(none surfaced)"
    current_intentions_raw = inputs.get("state", {}).get("intentions_active")
    current_intentions_text = format_intentions_for_prompt(current_intentions_raw, include_internals=True) if current_intentions_raw else "(none yet)"

    user_prompt = consolidation_prompt.USER_PROMPT.format(
        categories=svc._escape_prompt_value(categories_text),
        life_goals=svc._escape_prompt_value(life_goals_text),
        current_intentions=svc._escape_prompt_value(current_intentions_text),
        intention_activity=svc._escape_prompt_value(intention_text),
        retrieved_memories=svc._escape_prompt_value(retrieved_text),
        episodes=svc._escape_prompt_value(episodes_text),
    )

    raw = await svc.chat(
        user_prompt,
        profile=llm_profile,
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=4000,  # PIPELINE_MAX_TOKENS (kept in main.py; see comment there)
        op="consolidation",
        step="main",
    )
    parsed = _parse_consolidation_xml(str(raw or ""))
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
        "intention_actions": parsed["intention_actions"],
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
    if old_narrative_text:
        snapshot_previous_narrative_self(
            svc,
            scope={"user_id": user_id, "soul_id": soul_id},
            old_text=old_narrative_text,
            old_embedding=llm_results["old_narrative_embedding"],
        )

    companion_memory_id = None
    companion_text = str(llm_results.get("companion_memory") or "").strip()
    companion_embedding = llm_results.get("companion_embedding")
    if companion_text:
        if not isinstance(companion_embedding, list):
            raise HTTPException(status_code=500, detail="missing companion memory embedding")
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
            _soul_state.write(con, {"narrative_self": narrative_self})

        life_goal_rows = con.execute(
            """
SELECT id, description, status
FROM intentions
WHERE soul_id = ? AND user_id = ? AND source = 'life_goal' AND status IN ('active', 'removed')
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
                "UPDATE intentions SET status = 'removed', updated_at = ? WHERE id = ?",
                (now_iso, goal_id),
            )
        for goal_id in goals_to_delete:
            con.execute("DELETE FROM intentions WHERE id = ?", (goal_id,))
        for goal_id, text in goals_to_add:
            con.execute(
                """
INSERT INTO intentions (
    id, soul_id, user_id, description, status, source, confidence, target_date, related_memory_ids, updated_at
) VALUES (?, ?, ?, ?, 'active', 'life_goal', NULL, NULL, ?, ?)
""",
                (goal_id, soul_id, user_id, text, deps.json_to_db([]), now_iso),
            )
        con.commit()
    finally:
        con.close()

    from app.services.intention_state import (
        apply_intention_action,
        remove_intentions,
        upsert_intentions_stack_entries,
    )
    _con = deps.sqlite_connect(db_path)
    _con.row_factory = sqlite3.Row
    current_state = deps.conversation_state_from_row(
        deps.conversation_state_row(_con, conversation_id), con=_con
    ) or {}
    _con.close()
    current_intentions = current_state.get("intentions_active")

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

    state_updates: dict[str, Any] = {
        "pending_episode_ids": [],
        "last_consolidation_at": now_iso,
        "consolidation_in_progress": False,
        "consolidation_started_at": None,
        "intentions_active": current_intentions,
    }
    state_after, _ = deps.write_conversation_state(
        conversation_id,
        soul_id=soul_id,
        user_id=user_id,
        updates=state_updates,
    )
    scope = {"user_id": user_id, "soul_id": soul_id}
    wrote = write_memory_edges(svc.database.triple_repo, llm_results["edges"], scope=scope)
    invalidated = invalidate_memory_edges(svc.database.triple_repo, llm_results["edge_invalidations"], scope=scope)

    return {
        "conversation_id": conversation_id,
        "narrative_id": narrative_id if llm_results.get("narrative_self") else None,
        "companion_memory_id": companion_memory_id,
        "edges_written": wrote,
        "edges_invalidated": invalidated,
        "state": state_after,
    }
