from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree.ElementTree import Element

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
from app.services.turn_contract import DEFAULT_SOUL_CARD, format_time_anchor

if TYPE_CHECKING:
    from memu.app import MemoryService

# calibrated for ≤5 queued episodes + broad-block output
_CONSOLIDATION_MAX_TOKENS = 1800
# Design invariant: SOUL_CYCLE_PLAN caps active goals at 3 to preserve prompt signal
_ACTIVE_LIFE_GOAL_CAP = 3


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
        diaries.append(
            {
                "episode_id": episode_id,
                "prose": prose,
                "affective_tags": payload.get("affective_tags"),
                "unresolved": payload.get("unresolved"),
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

    return {
        "narrative_self": xml_text(root, "narrative_self"),
        "life_goal_add": life_goal_add,
        "life_goal_remove": life_goal_remove,
        "companion_memory": xml_text(root, "companion_memory"),
        "diaries": diaries,
        "edges": edges,
        "edge_invalidations": edge_invalidations,
    }


def _format_categories_for_prompt(rows: list[sqlite3.Row]) -> str:
    lines: list[str] = []
    for row in rows:
        name = str(row["name"] or "").strip() if "name" in row.keys() else ""
        summary = str(row["summary"] or "").strip() if "summary" in row.keys() else ""
        if not name or not summary:
            continue
        lines.append(f"- {name}: {summary}")
    return "\n".join(lines) or "(none yet)"


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
    return "\n".join(parts) if parts else "(none)"


def _format_intention_activity_for_prompt(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "(none)"
    lines = []
    for row in rows:
        description = str(row.get("description") or "").strip()
        status = str(row.get("status") or "").strip()
        updated_at = str(row.get("updated_at") or "").strip()
        if not description:
            continue
        meta = ", ".join(part for part in (status, updated_at) if part)
        lines.append(f"- {description}" + (f" ({meta})" if meta else ""))
    return "\n".join(lines) if lines else "(none)"


def _format_episode_block_for_prompt(episodes: list[dict[str, Any]]) -> str:
    if not episodes:
        return "(none queued)"
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
                updates={
                    "consolidation_in_progress": False,
                    "consolidation_started_at": None,
                },
            )
            state = deps.conversation_state_from_row(deps.conversation_state_row(con, conversation_id)) or state

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
    categories_text = _format_categories_for_prompt(list(inputs.get("categories") or []))
    life_goals_text = _format_life_goals_for_prompt(
        list(inputs.get("active_life_goals") or []),
        list(inputs.get("removed_life_goals") or []),
    )
    intention_text = _format_intention_activity_for_prompt(list(inputs.get("intention_activity") or []))
    episodes_text = _format_episode_block_for_prompt(list(inputs.get("episode_inputs") or []))

    # narrative_self (DB column name, populated by consolidation writes) and soul_card
    # (in-prompt label used by turn_contract and memorize) refer to the same entity —
    # the soul's self-description paragraph. Two names exist because the storage layer
    # predates the prompt-layer rename; neither has been migrated to avoid cascade churn.
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
        max_tokens=_CONSOLIDATION_MAX_TOKENS,
    )
    expected_episode_ids = {
        str(row.get("episode_id") or "").strip()
        for row in (inputs.get("episode_inputs") or [])
        if str(row.get("episode_id") or "").strip()
    }
    parsed = _parse_consolidation_xml(str(raw or ""), expected_episode_ids=expected_episode_ids)

    diary_rows = list(parsed.get("diaries") or [])
    embed_inputs: list[str] = []
    if str(parsed.get("companion_memory") or "").strip():
        embed_inputs.append(str(parsed["companion_memory"]).strip())
    embed_inputs.extend(str(row.get("prose") or "").strip() for row in diary_rows)
    embeddings = await svc.embed(embed_inputs, profile="embedding") if embed_inputs else []

    cursor = 0
    companion_embedding = None
    if str(parsed.get("companion_memory") or "").strip():
        if cursor < len(embeddings):
            companion_embedding = embeddings[cursor]
        cursor += 1
    for row in diary_rows:
        row["embedding"] = embeddings[cursor] if cursor < len(embeddings) else None
        cursor += 1

    return {
        "narrative_self": str(parsed.get("narrative_self") or "").strip() or None,
        "life_goal_add": [str(x).strip() for x in (parsed.get("life_goal_add") or []) if str(x).strip()],
        "life_goal_remove": [str(x).strip() for x in (parsed.get("life_goal_remove") or []) if str(x).strip()],
        "companion_memory": str(parsed.get("companion_memory") or "").strip() or None,
        "companion_embedding": companion_embedding,
        "diaries": diary_rows,
        "edges": parsed.get("edges") or [],
        "edge_invalidations": parsed.get("edge_invalidations") or [],
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

    deps.sqlite_ensure_nonempty(db_path)
    con = deps.sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        deps.sqlite_ensure_conversation_state_schema(con)

        self_model_id = str(inputs.get("self_model_id") or "").strip() or str(uuid.uuid4())
        narrative_self = str(llm_results.get("narrative_self") or "").strip() or None

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

        for desc in llm_results.get("life_goal_remove") or []:
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
        for desc in llm_results.get("life_goal_add") or []:
            text = str(desc or "").strip()
            if not text or text in active_ids or active_goal_count >= _ACTIVE_LIFE_GOAL_CAP:
                continue
            goal_id = str(uuid.uuid4())
            goals_to_add.append((goal_id, text))
            active_ids[text] = goal_id
            active_goal_count += 1

        episode_map = {
            str(row.get("episode_id") or "").strip(): row
            for row in (inputs.get("episode_inputs") or [])
            if str(row.get("episode_id") or "").strip()
        }

        diary_ids: list[str] = []
        for row in llm_results.get("diaries") or []:
            episode_id = str(row.get("episode_id") or "").strip()
            prose = str(row.get("prose") or "").strip()
            embedding = row.get("embedding")
            if not episode_id or not prose:
                continue
            if not isinstance(embedding, list):
                raise HTTPException(status_code=500, detail=f"missing diary embedding for episode {episode_id}")
            happened_at = None
            episode_meta = episode_map.get(episode_id)
            if isinstance(episode_meta, dict):
                happened_at = episode_meta.get("happened_at")
            diary_ids.append(
                upsert_diary_entry_memory(
                    svc,
                    user_id=user_id,
                    soul_id=soul_id,
                    conversation_id=conversation_id,
                    episode_id=episode_id,
                    prose=prose,
                    embedding=embedding,
                    affective_tags=row.get("affective_tags"),
                    unresolved=row.get("unresolved"),
                    happened_at=happened_at if isinstance(happened_at, datetime) else None,
                )
            )

        companion_memory_id = None
        companion_text = str(llm_results.get("companion_memory") or "").strip()
        companion_embedding = llm_results.get("companion_embedding")
        if companion_text:
            if not isinstance(companion_embedding, list):
                raise HTTPException(status_code=500, detail="missing companion memory embedding")
            companion_happened_at = None
            if episode_map:
                first_episode = min(
                    episode_map.values(),
                    key=lambda row: int(row.get("start_idx") or 0),
                )
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
                (
                    self_model_id,
                    soul_id,
                    user_id,
                    narrative_self,
                    deps.json_to_db(diary_ids),
                    now_iso,
                ),
            )
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
    wrote = write_memory_edges(svc.database.triple_repo, llm_results.get("edges"), scope=scope)
    invalidated = invalidate_memory_edges(svc.database.triple_repo, llm_results.get("edge_invalidations"), scope=scope)

    return {
        "conversation_id": conversation_id,
        "self_model_id": self_model_id if llm_results.get("narrative_self") else None,
        "pending_diary_episode_ids_cleared": True,
        "diary_memory_ids": diary_ids,
        "companion_memory_id": companion_memory_id,
        "edges_written": wrote,
        "edges_invalidated": invalidated,
        "state": state_after,
    }
