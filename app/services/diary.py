from __future__ import annotations

import hashlib
import re
import sqlite3
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from memu.prompts.diary import self_model_update as diary_self_model_update_prompt
from memu.prompts.memory_type import diary as diary_memory_prompt

from app.services.intention_state import (
    normalize_intentions_stack,
    normalize_memory_cache,
    upsert_intentions_stack_entries,
)
from app.services.self_model_merge import _apply_tension_updates

if TYPE_CHECKING:
    from memu.app import MemoryService


@dataclass(frozen=True)
class DiaryDeps:
    sqlite_current_path: Callable[[str, str], Path | None]
    sqlite_ensure_nonempty: Callable[[Path], None]
    sqlite_connect: Callable[[Path], sqlite3.Connection]
    sqlite_ensure_conversation_state_schema: Callable[[sqlite3.Connection], None]
    conversation_state_row: Callable[[sqlite3.Connection, str], sqlite3.Row | None]
    conversation_state_from_row: Callable[[sqlite3.Row | None], dict[str, Any] | None]
    get_storage_dir: Callable[[dict[str, Any]], Path]
    config: dict[str, Any]
    find_chat_dir_for_conversation: Callable[[Path, str, str, str], Path | None]
    read_list: Callable[[Path], list[dict[str, Any]]]
    normalize_text_list: Callable[[Any], list[str]]
    normalize_trait_invariants: Callable[[Any], list[dict[str, Any]]]
    normalize_trait_strength: Callable[[Any], float]
    json_to_db: Callable[[Any], str | None]


def _parse_episode_range(episode_id: str) -> tuple[int, int]:
    text = str(episode_id or "").strip()
    if not text or ":" not in text:
        raise ValueError(f"invalid episode_id: {episode_id}")
    range_part = text.split(":", 1)[1]
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


def _message_ts_utc(msg: dict[str, Any]) -> str | None:
    ts_ms = msg.get("ts_ms")
    if not isinstance(ts_ms, (int, float)):
        return None
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, UTC).isoformat()


def _format_messages_for_diary(
    messages: list[dict[str, Any]],
    episode_ranges: list[tuple[str, int, int]],
) -> str:
    lines: list[str] = []
    prev_end: int | None = None
    for episode_num, (episode_id, start_idx, end_idx) in enumerate(episode_ranges, start=1):
        if not messages:
            continue
        start = max(0, start_idx)
        end = min(len(messages) - 1, end_idx)
        if start > end:
            continue
        if prev_end is not None and start > prev_end + 1:
            lines.append(f"[gap] message_index={prev_end + 1}-{start - 1}")
            lines.append("")
        start_ts = _message_ts_utc(messages[start]) if isinstance(messages[start], dict) else None
        end_ts = _message_ts_utc(messages[end]) if isinstance(messages[end], dict) else None
        header = f"Episode {episode_num} | message_index={start}-{end}"
        if start_ts and end_ts:
            header = f"{header} | time_utc={start_ts}..{end_ts}"
        lines.append(header)
        lines.append(f"episode_id={episode_id}")
        for idx in range(start, end + 1):
            msg = messages[idx]
            if not isinstance(msg, dict):
                continue
            speaker = str(msg.get("name") or msg.get("role") or "unknown").strip() or "unknown"
            content = " ".join(str(msg.get("content") or "").splitlines()).strip()
            if not content:
                continue
            lines.append(f"[{idx}] [{speaker}]: {content}")
        lines.append("")
        prev_end = end
    return "\n".join(lines).strip()


def _format_categories_for_diary(rows: Sequence[sqlite3.Row]) -> str:
    lines: list[str] = []
    for row in rows:
        name = str(row["name"] or "").strip() if "name" in row.keys() else ""
        if not name:
            continue
        summary = str(row["summary"] or "").strip() if "summary" in row.keys() else ""
        lines.append(f"- {name}: {summary or '(no summary yet)'}")
    return "\n".join(lines)


def _format_memory_rows_for_diary(rows: Sequence[sqlite3.Row]) -> str:
    lines: list[str] = []
    for row in rows:
        memory_type = str(row["memory_type"] or "").strip() if "memory_type" in row.keys() else ""
        source_role = str(row["source_role"] or "").strip() if "source_role" in row.keys() else ""
        summary = str(row["summary"] or "").strip() if "summary" in row.keys() else ""
        if not summary:
            continue
        meta_parts = [part for part in (memory_type, source_role) if part]
        confidence = row["confidence"] if "confidence" in row.keys() else None
        reflection_salience = row["reflection_salience"] if "reflection_salience" in row.keys() else None
        if isinstance(confidence, (int, float)):
            meta_parts.append(f"confidence={float(confidence):.2f}")
        if isinstance(reflection_salience, (int, float)):
            meta_parts.append(f"reflection_salience={float(reflection_salience):.2f}")
        meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
        lines.append(f"- {summary}{meta}")
    return "\n".join(lines)


def _format_memory_rows_for_search(rows: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        memory_id = str(row.get("id") or "").strip()
        summary = str(row.get("summary") or "").strip()
        if not memory_id or not summary:
            continue
        memory_type = str(row.get("memory_type") or "").strip()
        source_role = str(row.get("source_role") or "").strip()
        episode_id = str(row.get("episode_id") or "").strip()
        meta_parts = [part for part in (memory_type, source_role) if part]
        if episode_id:
            meta_parts.append(f"episode_id={episode_id}")
        confidence = row.get("confidence")
        reflection_salience = row.get("reflection_salience")
        if isinstance(confidence, (int, float)):
            meta_parts.append(f"confidence={float(confidence):.2f}")
        if isinstance(reflection_salience, (int, float)):
            meta_parts.append(f"reflection_salience={float(reflection_salience):.2f}")
        meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
        lines.append(f"- id={memory_id}{meta}: {summary}")
    return "\n".join(lines)


def _extract_xml_fragment(raw: str, root_tag: str) -> ET.Element:
    text = str(raw or "").strip()
    match = re.search(rf"<{root_tag}>(.*)</{root_tag}>", text, re.DOTALL)
    if match is None:
        raise ValueError(f"Missing <{root_tag}> root")
    return ET.fromstring(f"<{root_tag}>{match.group(1)}</{root_tag}>")


def _xml_text(element: ET.Element | None, path: str) -> str | None:
    if element is None:
        return None
    node = element.find(path)
    if node is None or node.text is None:
        return None
    text = str(node.text).strip()
    return text or None


def _xml_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_related_memory_ids_xml(raw: str) -> list[str]:
    root = _extract_xml_fragment(raw, "related_memory_ids")
    ids: list[str] = []
    for node in root.findall("id"):
        text = str(node.text or "").strip()
        if text:
            ids.append(text)
    return ids


def _parse_diary_xml(raw: str) -> dict[str, Any]:
    root = _extract_xml_fragment(raw, "diary")
    affect = root.find("affect")
    intentions: list[str] = []
    intentions_root = root.find("intentions")
    if intentions_root is not None:
        for item in intentions_root.findall("intention"):
            text = str(item.text or "").strip()
            if text:
                intentions.append(text)
    return {
        "prose": _xml_text(root, "prose"),
        "affective_tags": {
            "emotion": _xml_text(affect, "emotion"),
            "trigger": _xml_text(affect, "trigger"),
            "valence": _xml_float(_xml_text(affect, "valence")),
            "intensity": _xml_float(_xml_text(affect, "intensity")),
            "what_helped": _xml_text(affect, "what_helped"),
        },
        "unresolved": _xml_text(root, "unresolved"),
        "intentions": intentions,
        "companion_memory": _xml_text(root, "companion_memory"),
    }


def _parse_self_model_update_xml(raw: str, normalize_trait_strength: Callable[[Any], float]) -> dict[str, Any]:
    root = _extract_xml_fragment(raw, "self_model_update")
    trait_root = root.find("trait_invariants")
    adds: list[dict[str, Any]] = []
    removes: list[str] = []
    if trait_root is not None:
        for item in trait_root.findall("add"):
            tendency = _xml_text(item, "tendency") or str(item.text or "").strip()
            if tendency:
                adds.append(
                    {
                        "tendency": tendency,
                        "strength": normalize_trait_strength(_xml_text(item, "strength")),
                    }
                )
        for item in trait_root.findall("remove"):
            text = str(item.text or "").strip()
            if text:
                removes.append(text)
    # Tension pair parsing + merge logic
    tension_root = root.find("tensions")
    tension_adds: list[dict[str, Any]] = []
    tension_removes: list[str] = []
    if tension_root is not None:
        for item in tension_root.findall("add"):
            between = _xml_text(item, "between") or str(item.text or "").strip()
            if between:
                tension_adds.append(
                    {
                        "type": "tension",
                        "between": between,
                        "root": _xml_text(item, "root") or "",
                        "implication": _xml_text(item, "implication") or "",
                        "strength": normalize_trait_strength(_xml_text(item, "strength")),
                    }
                )
        for item in tension_root.findall("remove"):
            text = str(item.text or "").strip()
            if text:
                tension_removes.append(text)
    life_goal_root = root.find("life_goals")
    life_goal_add: list[str] = []
    life_goal_remove: list[str] = []
    if life_goal_root is not None:
        for item in life_goal_root.findall("add"):
            text = str(item.text or "").strip()
            if text:
                life_goal_add.append(text)
        for item in life_goal_root.findall("remove"):
            text = str(item.text or "").strip()
            if text:
                life_goal_remove.append(text)
    return {
        "trait_add": adds,
        "trait_remove": removes,
        "tension_add": tension_adds,
        "tension_remove": tension_removes,
        "life_goal_add": life_goal_add,
        "life_goal_remove": life_goal_remove,
        "narrative_self": _xml_text(root, "narrative_self"),
        "contextual_state": _xml_text(root, "contextual_state"),
    }


def _load_current_self_model(
    con: sqlite3.Connection,
    *,
    self_model_id: str | None,
    soul_id: str,
    user_id: str,
) -> sqlite3.Row | None:
    if self_model_id:
        row = con.execute(
            "SELECT * FROM memu_self_model WHERE id = ? LIMIT 1",
            (self_model_id,),
        ).fetchone()
        if row is not None:
            return row
    return con.execute(
        """
SELECT * FROM memu_self_model
WHERE soul_id = ? AND user_id = ?
ORDER BY updated_at DESC, id DESC
LIMIT 1
""",
        (soul_id, user_id),
    ).fetchone()


def _format_self_model_for_prompt(
    row: sqlite3.Row | None,
    *,
    normalize_trait_invariants: Callable[[Any], list[dict[str, Any]]],
) -> str:
    if row is None:
        return ""
    all_items = normalize_trait_invariants(row["trait_invariants"] if "trait_invariants" in row.keys() else None)
    tendencies = [item for item in all_items if item.get("type") != "tension"]
    tensions = [item for item in all_items if item.get("type") == "tension"]
    lines = ["Trait invariants (tendencies):"]
    if tendencies:
        lines.extend(f"- {t['tendency']} (strength: {t['strength']:.1f})" for t in tendencies)
    else:
        lines.append("-")
    lines.append("")
    lines.append("Trait invariants (tensions):")
    if tensions:
        for t in tensions:
            between = t.get("between", "")
            root = t.get("root", "")
            implication = t.get("implication", "")
            strength = t.get("strength", 0.3)
            lines.append(f"- {between} (strength: {strength:.1f})")
            if root:
                lines.append(f"  Root: {root}")
            if implication:
                lines.append(f"  Implication: {implication}")
    else:
        lines.append("-")
    lines.append("")
    lines.append("Narrative self:")
    lines.append(str(row["narrative_self"] or "").strip() or "-")
    lines.append("")
    lines.append("Contextual state:")
    lines.append(str(row["contextual_state"] or "").strip() or "-")
    return "\n".join(lines)


def gather_diary_inputs(
    deps: DiaryDeps,
    *,
    conversation_id: str,
    soul_id: str,
    user_id: str,
) -> dict[str, Any]:
    """Phase 1: read all diary inputs; clear pending_diary_episode_ids immediately.
    Returns plain-Python dict. Caller must hold the memorize lock."""
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

        state_row = deps.conversation_state_row(con, conversation_id)
        state = deps.conversation_state_from_row(state_row)
        if state is None:
            raise HTTPException(status_code=404, detail="conversation state not found")

        pending_diary_episode_ids = deps.normalize_text_list(state.get("pending_diary_episode_ids"))
        if not pending_diary_episode_ids:
            raise HTTPException(status_code=400, detail="no diary-worthy episodes queued")

        storage_dir = deps.get_storage_dir(deps.config)
        chats_dir = (storage_dir / "st_chats").resolve()
        chat_dir = deps.find_chat_dir_for_conversation(chats_dir, user_id, soul_id, conversation_id)
        if chat_dir is None:
            raise HTTPException(status_code=404, detail="conversation resource not found")

        memory_rows_raw = con.execute(
            """
SELECT id, memory_type, summary, source_role, confidence, episode_id, reflection_salience, updated_at, created_at
FROM memu_memory_items
WHERE soul_id = ? AND user_id = ? AND summary IS NOT NULL AND TRIM(summary) != ''
ORDER BY updated_at DESC, created_at DESC, id DESC
LIMIT 240
""",
            (soul_id, user_id),
        ).fetchall()
        pending_episode_set = {str(ep or "").strip() for ep in pending_diary_episode_ids if str(ep or "").strip()}
        memory_search_candidates: list[dict[str, Any]] = [
            {k: row[k] for k in row.keys()}
            for row in memory_rows_raw
            if str(row["episode_id"] or "").strip() not in pending_episode_set
        ]

        category_rows = con.execute(
            """
SELECT name, summary
FROM memu_memory_categories
WHERE soul_id = ? AND user_id = ?
ORDER BY name ASC
""",
            (soul_id, user_id),
        ).fetchall()

        full_messages = deps.read_list((chat_dir / "full.json").resolve())
        episode_ranges: list[tuple[str, int, int]] = []
        for ep in pending_diary_episode_ids:
            start_idx, end_idx = _parse_episode_range(str(ep))
            episode_ranges.append((str(ep), start_idx, end_idx))
        episode_ranges.sort(key=lambda item: (item[1], item[2], item[0]))
        excerpt = _format_messages_for_diary(full_messages, episode_ranges)
        if not excerpt.strip():
            raise HTTPException(status_code=400, detail="diary source messages not found in resource")

        current_self_model_row = _load_current_self_model(
            con,
            self_model_id=str(state.get("self_model_id") or "").strip() or None,
            soul_id=soul_id,
            user_id=user_id,
        )
        current_self_model: dict[str, Any] | None = None
        if current_self_model_row is not None:
            current_self_model = {
                "id": current_self_model_row["id"],
                "trait_invariants": current_self_model_row["trait_invariants"],
                "narrative_self": current_self_model_row["narrative_self"],
            }
        life_goal_rows = con.execute(
            "SELECT id, description FROM intentions_life_goals WHERE soul_id = ? AND user_id = ? AND status = 'active' AND source = 'life_goal' ORDER BY updated_at ASC",
            (soul_id, user_id),
        ).fetchall()
        existing_life_goals = [
            {"id": str(row["id"]), "description": str(row["description"] or "").strip()}
            for row in life_goal_rows
            if str(row["description"] or "").strip()
        ]

        context_parts: list[str] = []
        category_block = _format_categories_for_diary(category_rows)
        if category_block:
            context_parts.append(f"Current categories:\n{category_block}")
        if existing_life_goals:
            goals_text = "\n".join(f"- {g['description']}" for g in existing_life_goals)
            context_parts.append(f"Current life goals:\n{goals_text}")
        cache_entries = normalize_memory_cache(state.get("memory_cache"))
        if cache_entries:
            cache_text = "\n".join(f"- {entry}" for entry in cache_entries)
            context_parts.append(f"Recent internal cache:\n{cache_text}")
        intentions_stack = normalize_intentions_stack(state.get("intentions_active"))
        intention_lines = [
            f"- {text}"
            for item in (intentions_stack.get("items") or [])
            if isinstance(item, dict)
            and (text := str(item.get("text") or "").strip())
        ]
        if intention_lines:
            context_parts.append(f"Current intentions:\n{'\n'.join(intention_lines)}")

        existing_self_model_text = _format_self_model_for_prompt(
            current_self_model,
            normalize_trait_invariants=deps.normalize_trait_invariants,
        )

        # Clear only after all inputs validated — failures before here leave IDs intact for retry.
        con.execute(
            "UPDATE memu_conversation_state SET pending_diary_episode_ids = ?, updated_at = ? WHERE conversation_id = ?",
            (deps.json_to_db([]), datetime.now(UTC).isoformat(), conversation_id),
        )
        con.commit()

        run_hash = hashlib.sha1("|".join(sorted(pending_diary_episode_ids)).encode()).hexdigest()[:12]
        return {
            "db_path": db_path,
            "pending_diary_episode_ids": pending_diary_episode_ids,
            "memory_search_candidates": memory_search_candidates,
            "current_self_model": current_self_model,
            "existing_life_goals": existing_life_goals,
            "existing_self_model_text": existing_self_model_text,
            "context_parts": context_parts,
            "excerpt": excerpt,
            "diary_run_episode_id": f"diary-run:{run_hash}",
        }
    finally:
        con.close()


async def run_diary_llm(
    svc: MemoryService,
    deps: DiaryDeps,
    *,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Phase 2: run all LLM calls. No DB access. Safe to run outside memorize lock."""
    context_parts: list[str] = list(inputs["context_parts"])
    excerpt: str = inputs["excerpt"]
    existing_self_model_text: str = inputs.get("existing_self_model_text", "")
    existing_life_goals = inputs.get("existing_life_goals") or []
    memory_search_candidates: list[dict[str, Any]] = inputs.get("memory_search_candidates") or []
    related_memory_ids: list[str] = []
    if memory_search_candidates:
        search_pool_text = _format_memory_rows_for_search(memory_search_candidates)
        if search_pool_text:
            related_search_prompt = "\n\n".join(
                [
                    "# Task Objective",
                    "Select background memory IDs that are relevant to the anchor episodes.",
                    "The anchors are the source of truth. Selected memories are background context only.",
                    "# Rules",
                    "- Use only IDs from the candidate list.",
                    "- Return at most 24 IDs.",
                    "- If none are relevant, return an empty list.",
                    "# Output Format (XML)",
                    "<related_memory_ids>",
                    "  <id>memory-id</id>",
                    "</related_memory_ids>",
                    "# Anchor Episodes",
                    f"<anchor_episodes>\n{svc._escape_prompt_value(excerpt)}\n</anchor_episodes>",
                    "# Candidate Memories",
                    f"<candidate_memories>\n{svc._escape_prompt_value(search_pool_text)}\n</candidate_memories>",
                ]
            )
            related_raw = await svc.chat(related_search_prompt, temperature=0.0, max_tokens=800)
            selected_ids = _parse_related_memory_ids_xml(related_raw)
            by_id = {str(row.get("id") or "").strip(): row for row in memory_search_candidates}
            selected_rows: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for item_id in selected_ids:
                key = str(item_id or "").strip()
                row = by_id.get(key)
                if not key or row is None or key in seen_ids:
                    continue
                selected_rows.append(row)
                seen_ids.add(key)
                related_memory_ids.append(key)
            related_block = _format_memory_rows_for_diary(selected_rows)
            if related_block:
                context_parts.append(f"Related background memories:\n{related_block}")

    if context_parts:
        diary_prompt = diary_memory_prompt.PROMPT_WITH_CONTEXT.format(
            context=svc._escape_prompt_value("\n\n".join(context_parts)),
            conversation=svc._escape_prompt_value(excerpt),
        )
    else:
        diary_prompt = diary_memory_prompt.PROMPT.format(
            conversation=svc._escape_prompt_value(excerpt),
        )

    diary_raw = await svc.chat(diary_prompt, temperature=0.2, max_tokens=800)
    diary_data = _parse_diary_xml(diary_raw)
    prose = str(diary_data.get("prose") or "").strip()
    companion_memory = str(diary_data.get("companion_memory") or "").strip()
    if not prose:
        raise HTTPException(status_code=500, detail="diary generation returned empty prose")
    if not companion_memory:
        raise HTTPException(status_code=500, detail="diary generation returned empty companion_memory")

    diary_embedding, companion_embedding = await svc.embed([prose, companion_memory], profile="embedding")

    if existing_life_goals:
        goals_lines = "\n".join(f"- {g['description']}" for g in existing_life_goals)
        existing_self_model_text = (existing_self_model_text + f"\n\nCurrent life goals:\n{goals_lines}").strip()

    if existing_self_model_text:
        self_model_prompt = diary_self_model_update_prompt.PROMPT_WITH_EXISTING.format(
            existing_self_model=svc._escape_prompt_value(existing_self_model_text),
            diary_entry=svc._escape_prompt_value(prose),
        )
    else:
        self_model_prompt = diary_self_model_update_prompt.PROMPT.format(
            diary_entry=svc._escape_prompt_value(prose),
        )

    self_model_raw = await svc.chat(self_model_prompt, temperature=0.2, max_tokens=1200)
    self_model_update = _parse_self_model_update_xml(self_model_raw, deps.normalize_trait_strength)
    life_goal_remove = self_model_update.get("life_goal_remove") or []
    if life_goal_remove:
        dropped_embeddings = await svc.embed(life_goal_remove, profile="embedding")
        dropped_goal_embeddings = list(zip(life_goal_remove, dropped_embeddings))
    else:
        dropped_goal_embeddings = []

    return {
        "diary_data": {
            "affective_tags": diary_data.get("affective_tags"),
            "unresolved": diary_data.get("unresolved"),
            "intentions": diary_data.get("intentions") or [],
        },
        "related_memory_ids": related_memory_ids,
        "prose": prose,
        "companion_memory": companion_memory,
        "diary_embedding": diary_embedding,
        "companion_embedding": companion_embedding,
        "self_model_update": self_model_update,
        "dropped_goal_embeddings": dropped_goal_embeddings,
    }


def write_diary_outputs(
    deps: DiaryDeps,
    svc: MemoryService,
    *,
    inputs: dict[str, Any],
    llm_results: dict[str, Any],
    conversation_id: str,
    soul_id: str,
    user_id: str,
) -> dict[str, Any]:
    """Phase 3: write diary item, self_model, intentions, update state.
    Caller must hold the memorize lock."""
    db_path: Path = inputs["db_path"]
    current_self_model: dict[str, Any] | None = inputs.get("current_self_model")
    existing_life_goals = inputs.get("existing_life_goals") or []

    diary_data: dict[str, Any] = llm_results["diary_data"]
    prose: str = llm_results["prose"]
    companion_memory: str = llm_results["companion_memory"]
    diary_embedding = llm_results["diary_embedding"]
    companion_embedding = llm_results["companion_embedding"]
    self_model_update: dict[str, Any] = llm_results["self_model_update"]
    memory_ids: list[str] = [str(x) for x in llm_results["related_memory_ids"] if str(x).strip()]

    diary_run_episode_id: str | None = inputs.get("diary_run_episode_id")
    companion_episode_id = f"companion:{diary_run_episode_id}" if diary_run_episode_id else None

    existing_diary = (
        svc.database.memory_item_repo.list_items({"episode_id": diary_run_episode_id, "memory_type": "diary"})
        if diary_run_episode_id
        else {}
    )
    if existing_diary:
        diary_item = next(iter(existing_diary.values()))
    else:
        diary_item = svc.database.memory_item_repo.create_item(
            resource_id=None,
            memory_type="diary",
            summary=prose,
            embedding=diary_embedding,
            user_data={"user_id": user_id, "soul_id": soul_id, "conversation_id": conversation_id},
            conversation_id=conversation_id,
            affective_tags=diary_data.get("affective_tags"),
            unresolved=diary_data.get("unresolved"),
            episode_id=diary_run_episode_id,
        )

    existing_companion = (
        svc.database.memory_item_repo.list_items({"episode_id": companion_episode_id, "memory_type": "event"})
        if companion_episode_id
        else {}
    )
    if not existing_companion:
        svc.database.memory_item_repo.create_item(
            resource_id=None,
            memory_type="event",
            summary=companion_memory,
            embedding=companion_embedding,
            user_data={"user_id": user_id, "soul_id": soul_id, "conversation_id": conversation_id},
            source_role="soul",
            conversation_id=conversation_id,
            episode_id=companion_episode_id,
        )

    all_items = deps.normalize_trait_invariants(
        current_self_model["trait_invariants"]
        if current_self_model is not None and "trait_invariants" in current_self_model
        else None
    )
    existing_traits = [item for item in all_items if item.get("type") != "tension"]
    existing_tensions = [item for item in all_items if item.get("type") == "tension"]
    for text in self_model_update["trait_remove"]:
        existing_traits = [item for item in existing_traits if item.get("tendency") != text]
    for trait in self_model_update["trait_add"]:
        tendency = str(trait.get("tendency") or "").strip()
        if not tendency:
            continue
        for existing_trait in existing_traits:
            if existing_trait.get("tendency") == tendency:
                existing_trait["strength"] = deps.normalize_trait_strength(trait.get("strength"))
                break
        else:
            existing_traits.append(
                {
                    "tendency": tendency,
                    "strength": deps.normalize_trait_strength(trait.get("strength")),
                }
            )
    existing_tensions = _apply_tension_updates(
        existing_tensions,
        tension_remove=self_model_update.get("tension_remove", []),
        tension_add=self_model_update.get("tension_add", []),
        normalize_trait_strength=deps.normalize_trait_strength,
    )
    existing_traits = existing_traits + existing_tensions

    narrative_self = (
        str(self_model_update.get("narrative_self") or "").strip()
        or (
            str(current_self_model["narrative_self"] or "").strip()
            if current_self_model is not None and "narrative_self" in current_self_model
            else ""
        )
        or None
    )
    contextual_state = str(self_model_update.get("contextual_state") or "").strip() or None
    self_model_id = (
        str(current_self_model["id"]).strip()
        if current_self_model is not None and "id" in current_self_model
        else str(uuid.uuid4())
    )
    related_memory_ids = memory_ids
    now_iso = datetime.now(UTC).isoformat()
    dropped_goal_memories: list[tuple[str, list[float]]] = []

    deps.sqlite_ensure_nonempty(db_path)
    con = deps.sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        deps.sqlite_ensure_conversation_state_schema(con)
        where: dict[str, Any] = {}
        if soul_id:
            where["soul_id"] = soul_id
        if user_id:
            where["user_id"] = user_id
        categories = svc.database.memory_category_repo.list_categories(where)
        all_categories_summary = (
            "\n".join(
                f"{name}: {summary}"
                for cat in categories.values()
                for name, summary in [
                    (
                        str(getattr(cat, "name", "") or "").strip(),
                        str(getattr(cat, "summary", "") or "").strip(),
                    )
                ]
                if name and summary
            )
            or None
        )

        con.execute(
            """
INSERT INTO memu_self_model (
    id, soul_id, user_id, trait_invariants, narrative_self, contextual_state, related_memory_ids, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    soul_id = excluded.soul_id,
    user_id = excluded.user_id,
    trait_invariants = excluded.trait_invariants,
    narrative_self = excluded.narrative_self,
    contextual_state = excluded.contextual_state,
    related_memory_ids = excluded.related_memory_ids,
    updated_at = excluded.updated_at
""",
            (
                self_model_id,
                soul_id,
                user_id,
                deps.json_to_db(existing_traits),
                narrative_self,
                contextual_state,
                deps.json_to_db(related_memory_ids),
                now_iso,
            ),
        )

        goal_id_by_desc = {g["description"]: g["id"] for g in existing_life_goals}
        removed_goal_ids: set[str] = set()
        for desc, embedding in llm_results.get("dropped_goal_embeddings") or []:
            goal_id = goal_id_by_desc.get(desc)
            if goal_id and goal_id not in removed_goal_ids:
                con.execute(
                    "UPDATE intentions_life_goals SET status = 'removed', updated_at = ? WHERE id = ?",
                    (now_iso, goal_id),
                )
                removed_goal_ids.add(goal_id)
                dropped_goal_memories.append((str(desc), embedding))

        active_life_goal_count = len(existing_life_goals) - len(removed_goal_ids)
        for desc in self_model_update.get("life_goal_add") or []:
            text = str(desc or "").strip()
            if not text or active_life_goal_count >= 3:
                continue
            goal_id = str(uuid.uuid4())
            con.execute(
                """
INSERT INTO intentions_life_goals (
    id, soul_id, user_id, description, status, source, confidence, target_date, related_memory_ids, updated_at
) VALUES (?, ?, ?, ?, 'active', 'life_goal', NULL, NULL, ?, ?)
""",
                (goal_id, soul_id, user_id, text, deps.json_to_db([diary_item.id]), now_iso),
            )
            active_life_goal_count += 1

        intention_ids: list[str] = []
        intention_stack_entries: list[dict[str, Any]] = []
        for description in (diary_data.get("intentions") or [])[:2]:
            text = str(description or "").strip()
            if not text:
                continue
            intention_id = str(uuid.uuid4())
            con.execute(
                """
INSERT INTO intentions_life_goals (
    id, soul_id, user_id, description, status, source, confidence, target_date, related_memory_ids, updated_at
) VALUES (?, ?, ?, ?, 'active', 'inferred', NULL, NULL, ?, ?)
""",
                (intention_id, soul_id, user_id, text, deps.json_to_db([diary_item.id]), now_iso),
            )
            intention_ids.append(intention_id)
            intention_stack_entries.append(
                {
                    "id": intention_id,
                    "text": text,
                    "priority": 10.0,
                    "ephemeral": False,
                }
            )

        assignments = ["self_model_id = ?", "all_categories_summary = ?", "updated_at = ?"]
        params: list[Any] = [str(self_model_id).strip() or None, all_categories_summary, datetime.now(UTC).isoformat()]
        if intention_ids:
            current_state = deps.conversation_state_from_row(deps.conversation_state_row(con, conversation_id)) or {}
            merged_intentions_active = upsert_intentions_stack_entries(
                normalize_intentions_stack(current_state.get("intentions_active")),
                intention_stack_entries,
            )
            assignments.insert(2, "intentions_active = ?")
            params.insert(2, deps.json_to_db(merged_intentions_active))
        con.execute(
            f"UPDATE memu_conversation_state SET {', '.join(assignments)} WHERE conversation_id = ?",
            (*params, conversation_id),
        )
        con.commit()
        updated_state = deps.conversation_state_from_row(deps.conversation_state_row(con, conversation_id)) or {}
        result = {
            "conversation_id": conversation_id,
            "memory_id": diary_item.id,
            "self_model_id": self_model_id,
            "intention_ids": intention_ids,
            "pending_diary_episode_ids_cleared": True,
            "all_categories_summary": all_categories_summary,
            "state": updated_state,
        }
    finally:
        con.close()

    for desc, embedding in dropped_goal_memories:
        svc.database.memory_item_repo.create_item(
            resource_id=None,
            memory_type="event",
            source_role="soul",
            summary=f"I used to want: {desc}",
            embedding=embedding,
            user_data={"user_id": user_id, "soul_id": soul_id, "conversation_id": conversation_id},
            conversation_id=conversation_id,
        )
    return result
