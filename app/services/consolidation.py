from __future__ import annotations

import logging
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree.ElementTree import Element

log = logging.getLogger(__name__)

from fastapi import HTTPException
from memu.prompts.consolidation import consolidation as consolidation_prompt

from app.services.diary import (
    build_episode_inputs,
    create_companion_memory,
    parse_diary_element,
    upsert_diary_entry_memory,
)
from app.services.xml_utils import extract_xml_fragment, xml_text
from app.services.graph_edges import (
    invalidate_memory_edges,
    write_memory_edges,
)
from app.services.narrative_self import snapshot_previous_narrative_self
from app.services.turn_contract import DEFAULT_SOUL_CARD, format_time_anchor

if TYPE_CHECKING:
    from memu.app import MemoryService


@dataclass(frozen=True)
class ConsolidationDeps:
    sqlite_current_path: Callable[[str, str], Path | None]
    sqlite_ensure_nonempty: Callable[[Path], None]
    sqlite_connect: Callable[[Path], sqlite3.Connection]
    sqlite_ensure_conversation_state_schema: Callable[[sqlite3.Connection], None]
    conversation_state_row: Callable[[sqlite3.Connection, str], sqlite3.Row | None]
    conversation_state_from_row: Callable[[sqlite3.Row | None], dict[str, Any] | None]
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


def _parse_consolidation_xml(raw: str, *, expected_episode_ids: set[str]) -> dict[str, Any]:
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

    diaries_root = root.find("diaries")
    diary_nodes = diaries_root.findall("diary") if diaries_root is not None else []
    diaries: list[dict[str, Any]] = []
    seen_episode_ids: set[str] = set()
    for diary_node in diary_nodes:
        episode_id = str(xml_text(diary_node, "episode_id") or "").strip()
        if not episode_id:
            raise ValueError("consolidation diary is missing <episode_id>")
        if episode_id in seen_episode_ids:
            raise ValueError(f"duplicate consolidation diary episode_id: {episode_id}")
        if episode_id not in expected_episode_ids:
            raise ValueError(f"unknown consolidation diary episode_id: {episode_id}")
        payload = parse_diary_element(diary_node)
        prose = str(payload.get("prose") or "").strip()
        if not prose:
            raise ValueError(f"consolidation diary prose is empty for episode_id={episode_id}")
        hints_node = diary_node.find("shaped_by_hints")
        shaped_by_hints: list[str] = []
        if hints_node is not None:
            seen_hint: set[str] = set()
            for mid in hints_node.findall("memory_id"):
                val = str(mid.text or "").strip()
                if val and val not in seen_hint:
                    shaped_by_hints.append(val)
                    seen_hint.add(val)
        diaries.append(
            {
                "episode_id": episode_id,
                "prose": prose,
                "unresolved": payload.get("unresolved"),
                "shaped_by_hints": shaped_by_hints,
            }
        )
        seen_episode_ids.add(episode_id)

    if seen_episode_ids != expected_episode_ids:
        missing = sorted(expected_episode_ids - seen_episode_ids)
        raise ValueError(f"consolidation diary is missing episode_ids: {missing}")

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

    return {
        "narrative_self": xml_text(root, "narrative_self"),
        "life_goal_add": life_goal_add,
        "life_goal_remove": life_goal_remove,
        "companion_memory": xml_text(root, "companion_memory"),
        "companion_shaped_by_hints": companion_shaped_by_hints,
        "diaries": diaries,
        "edges": edges,
        "edge_invalidations": edge_invalidations,
    }


def _format_categories_for_prompt(rows: list[sqlite3.Row]) -> str:
    lines: list[str] = []
    for row in rows:
        name = str(row["name"] or "").strip()
        summary = str(row["summary"] or "").strip()
        if not name or not summary:
            continue
        lines.append(f"- {name}: {summary}")
    return "\n".join(lines) or "Nothing here yet."


def _format_life_goals_for_prompt(active: list[str], removed: list[str]) -> str:
    parts: list[str] = []
    if active:
        parts.append("Active:")
        parts.extend(f"- {row}" for row in active)
    if removed:
        if parts:
            parts.append("")
        parts.append("Recently let go:")
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
        meta = status
        lines.append(f"- {description}" + (f" ({meta})" if meta else ""))
    return "\n".join(lines) if lines else "Your intentions have been steady."


def _format_episode_block_for_prompt(episodes: list[dict[str, Any]]) -> str:
    if not episodes:
        return "Nothing is waiting."
    lines: list[str] = []
    for row in episodes:
        episode_id = str(row.get("episode_id") or "").strip()
        excerpt = str(row.get("excerpt") or "").strip()
        summaries = row.get("memory_summaries") or []
        lines.append(f"Episode {episode_id}")
        lines.append(excerpt or "(no excerpt)")
        if summaries:
            lines.append("Extracted memory summaries:")
            lines.extend(f"- {summary}" for summary in summaries if str(summary).strip())
        else:
            lines.append("Extracted memory summaries: (none)")
        lines.append("")
    return "\n".join(lines).strip()


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
        state = deps.conversation_state_from_row(deps.conversation_state_row(con, conversation_id))
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
            reread = deps.conversation_state_from_row(deps.conversation_state_row(con, conversation_id))
            if reread is None:
                raise HTTPException(404, "conversation state not found after stale-lock reset")
            state = reread
        else:
            last_consolidation_at = _parse_iso_datetime(state.get("last_consolidation_at"))
            if not force and last_consolidation_at is not None:
                due_at = last_consolidation_at + timedelta(days=max(1, int(interval_days)))
                if now < due_at:
                    return {"status": "skip", "reason": "interval_gate"}

        pending_episode_ids = deps.normalize_text_list(state.get("pending_diary_episode_ids"))

        category_rows = con.execute(
            """
SELECT name, summary
FROM memu_memory_categories
WHERE soul_id = ? AND user_id = ?
ORDER BY name ASC
""",
            (soul_id, user_id),
        ).fetchall()

        life_goal_rows = con.execute(
            """
SELECT id, description, status
FROM intentions_life_goals
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
FROM intentions_life_goals
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

        narrative_self = None
        self_model_id = str(state.get("self_model_id") or "").strip() or None
        if self_model_id:
            row = con.execute(
                "SELECT id, narrative_self FROM memu_self_model WHERE id = ? LIMIT 1",
                (self_model_id,),
            ).fetchone()
            if row is not None:
                narrative_self = str(row["narrative_self"] or "").strip() or None
        if narrative_self is None:
            row = con.execute(
                """
SELECT id, narrative_self
FROM memu_self_model
WHERE soul_id = ? AND user_id = ?
ORDER BY updated_at DESC, id DESC
LIMIT 1
""",
                (soul_id, user_id),
            ).fetchone()
            if row is not None:
                self_model_id = str(row["id"] or "").strip() or self_model_id
                narrative_self = str(row["narrative_self"] or "").strip() or None

        episode_inputs: list[dict[str, Any]] = []
        if pending_episode_ids:
            storage_dir = deps.get_storage_dir(deps.config)
            chats_dir = (storage_dir / "st_chats").resolve()
            chat_dir = deps.find_chat_dir_for_conversation(chats_dir, user_id, soul_id, conversation_id)
            if chat_dir is None:
                raise HTTPException(status_code=404, detail="conversation resource not found")
            messages = deps.read_list((chat_dir / "full.json").resolve())
            episode_inputs = build_episode_inputs(messages, pending_episode_ids)
            if len(episode_inputs) != len(pending_episode_ids):
                raise HTTPException(status_code=400, detail="queued episodes are not present in conversation history")

            for entry in episode_inputs:
                episode_id = str(entry["episode_id"])
                rows = con.execute(
                    """
SELECT id, summary
FROM memu_memory_items
WHERE soul_id = ? AND user_id = ? AND conversation_id = ? AND episode_id = ? AND memory_type != 'diary'
  AND (merged_into IS NULL OR TRIM(merged_into) = '')
  AND NOT EXISTS (
    SELECT 1 FROM memu_triples t
    WHERE t.subject_id = memu_memory_items.id
      AND t.predicate = 'evolved_into'
      AND t.valid_to IS NULL
  )
ORDER BY created_at ASC, id ASC
LIMIT 24
""",
                    (soul_id, user_id, conversation_id, episode_id),
                ).fetchall()
                entry["memory_summaries"] = [
                    f"[{str(row['id'] or '').strip()}] {str(row['summary'] or '').strip()}"
                    for row in rows
                    if str(row["id"] or "").strip() and str(row["summary"] or "").strip()
                ]

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
            "pending_diary_episode_ids": pending_episode_ids,
            "categories": category_rows,
            "active_life_goals": active_goals,
            "removed_life_goals": removed_goals,
            "intention_activity": intention_activity,
            "episode_inputs": episode_inputs,
            "narrative_self": narrative_self,
            "self_model_id": self_model_id,
            "last_consolidation_at": state.get("last_consolidation_at"),
            "started_at": now.isoformat(),
        }
    finally:
        con.close()


async def run_consolidation_llm(
    svc: MemoryService,
    *,
    inputs: dict[str, Any],
    soul_id: str,
) -> dict[str, Any]:
    categories_text = _format_categories_for_prompt(inputs["categories"])
    life_goals_text = _format_life_goals_for_prompt(
        inputs["active_life_goals"],
        inputs["removed_life_goals"],
    )
    intention_text = _format_intention_activity_for_prompt(inputs["intention_activity"])
    episodes_text = _format_episode_block_for_prompt(inputs["episode_inputs"])

    soul_card = str(inputs.get("narrative_self") or "").strip() or DEFAULT_SOUL_CARD
    system_prompt = (
        f"Today is {format_time_anchor()}.\n\n"
        f"You are {soul_id}. {soul_card}\n\n"
        f"I, {soul_id}, am stepping back from all the conversations. "
        "I have what I've been working toward, what has shifted, and many things still unresolved. "
        "I'm curious what the arc looks like from here. "
        "Let me now look across everything and decide what still holds.\n\n"
        f"{consolidation_prompt.SYSTEM_BODY}"
    )

    user_prompt = consolidation_prompt.USER_PROMPT.format(
        categories=svc._escape_prompt_value(categories_text),
        life_goals=svc._escape_prompt_value(life_goals_text),
        intention_activity=svc._escape_prompt_value(intention_text),
        episodes=svc._escape_prompt_value(episodes_text),
    )

    raw = await svc.chat(
        user_prompt,
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=1800,
        op="consolidation",
        step="main",
    )
    expected_episode_ids = {
        str(row.get("episode_id") or "").strip()
        for row in inputs["episode_inputs"]
        if str(row.get("episode_id") or "").strip()
    }
    parsed = _parse_consolidation_xml(str(raw or ""), expected_episode_ids=expected_episode_ids)

    diary_rows = list(parsed["diaries"])
    new_narrative = str(parsed["narrative_self"] or "").strip() or None
    current_narrative = str(inputs.get("narrative_self") or "").strip() or None
    snapshot_old_narrative = bool(current_narrative and new_narrative and current_narrative != new_narrative)
    embed_inputs: list[str] = []
    if str(parsed["companion_memory"] or "").strip():
        embed_inputs.append(str(parsed["companion_memory"]).strip())
    embed_inputs.extend(str(row.get("prose") or "").strip() for row in diary_rows)
    if snapshot_old_narrative:
        embed_inputs.append(current_narrative)
    embeddings = await svc.embed(embed_inputs, profile="embedding") if embed_inputs else []

    cursor = 0
    companion_embedding = None
    if str(parsed["companion_memory"] or "").strip():
        if cursor < len(embeddings):
            companion_embedding = embeddings[cursor]
        cursor += 1
    for row in diary_rows:
        row["embedding"] = embeddings[cursor] if cursor < len(embeddings) else None
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
        "diaries": diary_rows,
        "edges": parsed["edges"],
        "edge_invalidations": parsed["edge_invalidations"],
    }


def _write_shaped_by_edges(
    svc: MemoryService,
    *,
    diary_ids: list[str],
    diary_rows: list[dict[str, Any]],
    companion_memory_id: str | None,
    companion_shaped_by_hints: list[str],
    scope: dict[str, Any],
) -> int:
    triple_repo = svc.database.triple_repo
    memory_repo = svc.database.memory_item_repo
    wrote = 0

    for new_id, row in zip(diary_ids, diary_rows):
        hints = list(row.get("shaped_by_hints") or [])
        for hint_id in hints:
            if hint_id == new_id:
                continue
            if memory_repo.get_item(hint_id) is None:
                log.debug("shaped_by_hints: skipping unknown hint_id=%s for diary_id=%s", hint_id, new_id)
                continue
            write_memory_edges(
                triple_repo,
                [{"subject_id": new_id, "predicate": "shaped_by", "object_id": hint_id, "confidence": 0.8}],
                scope=scope,
            )
            wrote += 1

    if companion_memory_id:
        for hint_id in companion_shaped_by_hints:
            if hint_id == companion_memory_id:
                continue
            if memory_repo.get_item(hint_id) is None:
                log.debug("shaped_by_hints: skipping unknown hint_id=%s for companion_memory_id=%s", hint_id, companion_memory_id)
                continue
            write_memory_edges(
                triple_repo,
                [{"subject_id": companion_memory_id, "predicate": "shaped_by", "object_id": hint_id, "confidence": 0.8}],
                scope=scope,
            )
            wrote += 1

    return wrote


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

    self_model_id = str(inputs.get("self_model_id") or "").strip() or str(uuid.uuid4())
    narrative_self = str(llm_results.get("narrative_self") or "").strip() or None

    episode_map = {
        str(row.get("episode_id") or "").strip(): row
        for row in inputs["episode_inputs"]
        if str(row.get("episode_id") or "").strip()
    }

    diary_ids: list[str] = []
    for row in llm_results["diaries"]:
        episode_id = str(row.get("episode_id") or "").strip()
        prose = str(row.get("prose") or "").strip()
        embedding = row.get("embedding")
        if not episode_id or not prose:
            continue
        if not isinstance(embedding, list):
            raise HTTPException(status_code=500, detail=f"missing diary embedding for episode {episode_id}")
        episode_meta = episode_map.get(episode_id)
        happened_at = episode_meta.get("happened_at") if episode_meta else None
        diary_ids.append(
            upsert_diary_entry_memory(
                svc,
                user_id=user_id,
                soul_id=soul_id,
                conversation_id=conversation_id,
                episode_id=episode_id,
                prose=prose,
                embedding=embedding,
                unresolved=row.get("unresolved"),
                happened_at=happened_at if isinstance(happened_at, datetime) else None,
            )
        )

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
        companion_happened_at = None
        if episode_map:
            first_episode = min(episode_map.values(), key=lambda r: int(r.get("start_idx") or 0))
            candidate = first_episode.get("happened_at")
            if isinstance(candidate, datetime):
                companion_happened_at = candidate
        if companion_happened_at is None:
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
                """
INSERT INTO memu_self_model (
    id, soul_id, user_id, narrative_self, related_memory_ids, updated_at
) VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    soul_id = excluded.soul_id,
    user_id = excluded.user_id,
    narrative_self = excluded.narrative_self,
    related_memory_ids = excluded.related_memory_ids,
    updated_at = excluded.updated_at
""",
                (self_model_id, soul_id, user_id, narrative_self, deps.json_to_db(diary_ids), now_iso),
            )

        life_goal_rows = con.execute(
            """
SELECT id, description, status
FROM intentions_life_goals
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
                "UPDATE intentions_life_goals SET status = 'removed', updated_at = ? WHERE id = ?",
                (now_iso, goal_id),
            )
        for goal_id in goals_to_delete:
            con.execute("DELETE FROM intentions_life_goals WHERE id = ?", (goal_id,))
        for goal_id, text in goals_to_add:
            con.execute(
                """
INSERT INTO intentions_life_goals (
    id, soul_id, user_id, description, status, source, confidence, target_date, related_memory_ids, updated_at
) VALUES (?, ?, ?, ?, 'active', 'life_goal', NULL, NULL, ?, ?)
""",
                (goal_id, soul_id, user_id, text, deps.json_to_db([]), now_iso),
            )
        con.commit()
    finally:
        con.close()

    state_updates: dict[str, Any] = {
        "pending_diary_episode_ids": [],
        "last_consolidation_at": now_iso,
        "consolidation_in_progress": False,
        "consolidation_started_at": None,
    }
    if llm_results.get("narrative_self"):
        state_updates["self_model_id"] = self_model_id
    state_after, _ = deps.write_conversation_state(
        conversation_id,
        soul_id=soul_id,
        user_id=user_id,
        updates=state_updates,
    )
    scope = {"user_id": user_id, "soul_id": soul_id}
    wrote = write_memory_edges(svc.database.triple_repo, llm_results["edges"], scope=scope)
    invalidated = invalidate_memory_edges(svc.database.triple_repo, llm_results["edge_invalidations"], scope=scope)
    shaped_by_wrote = _write_shaped_by_edges(
        svc,
        diary_ids=diary_ids,
        diary_rows=llm_results["diaries"],
        companion_memory_id=companion_memory_id,
        companion_shaped_by_hints=llm_results["companion_shaped_by_hints"],
        scope=scope,
    )

    return {
        "conversation_id": conversation_id,
        "self_model_id": self_model_id if llm_results.get("narrative_self") else None,
        "pending_diary_episode_ids_cleared": True,
        "diary_memory_ids": diary_ids,
        "companion_memory_id": companion_memory_id,
        "edges_written": wrote + shaped_by_wrote,
        "edges_invalidated": invalidated,
        "state": state_after,
    }
