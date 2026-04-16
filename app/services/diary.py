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
from memu.database.models import Triple
from memu.prompts.diary import self_model_update as diary_self_model_update_prompt
from memu.prompts.memory_type import diary as diary_memory_prompt

from app.services.intention_state import (
    normalize_intentions_stack,
    normalize_memory_cache,
    upsert_intentions_stack_entries,
)

if TYPE_CHECKING:
    from memu.app import MemoryService


def _make_llm_retrieve_service(svc: MemoryService) -> MemoryService:
    retrieve_config = svc.retrieve_config.model_copy(deep=True)
    retrieve_config.method = "llm"
    return svc.__class__(
        llm_profiles=svc.llm_profiles,
        blob_config=svc.blob_config,
        database_config=svc.database_config,
        memorize_config=svc.memorize_config,
        retrieve_config=retrieve_config,
        workflow_runner=svc.workflow_runner,
        user_config=svc.user_config,
    )


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


def _format_memory_rows_for_diary(rows: Sequence[sqlite3.Row], *, include_ids: bool = False) -> str:
    lines: list[str] = []
    for row in rows:
        item_id = str(row["id"] or "").strip() if "id" in row.keys() else ""
        memory_type = str(row["memory_type"] or "").strip() if "memory_type" in row.keys() else ""
        source_role = str(row["source_role"] or "").strip() if "source_role" in row.keys() else ""
        summary = str(row["summary"] or "").strip() if "summary" in row.keys() else ""
        if not summary:
            continue
        if include_ids and not item_id:
            continue
        meta_parts = [part for part in (memory_type, source_role) if part]
        confidence = row["confidence"] if "confidence" in row.keys() else None
        reflection_salience = row["reflection_salience"] if "reflection_salience" in row.keys() else None
        if isinstance(confidence, (int, float)):
            meta_parts.append(f"confidence={float(confidence):.2f}")
        if isinstance(reflection_salience, (int, float)):
            meta_parts.append(f"reflection_salience={float(reflection_salience):.2f}")
        meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
        if include_ids:
            lines.append(f"[{item_id}] {summary}{meta}")
        else:
            lines.append(f"- {summary}{meta}")
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


def _parse_self_model_update_xml(raw: str) -> dict[str, Any]:
    root = _extract_xml_fragment(raw, "self_model_update")
    obs_root = root.find("soul_observations")
    soul_observations: list[dict[str, Any]] = []
    if obs_root is not None:
        for item in obs_root.findall("observation"):
            text_node = item.find("text")
            if text_node is not None:
                obs_text = str(text_node.text or "").strip()
            else:
                obs_text = str(item.text or "").strip()
            if not obs_text:
                continue
            superseded_ids: list[str] = []
            sup_root = item.find("supersedes")
            if sup_root is not None:
                for sup_id_node in sup_root.findall("id"):
                    sup_id = str(sup_id_node.text or "").strip()
                    if sup_id:
                        superseded_ids.append(sup_id)
            shaped_by_ids: list[str] = []
            shaped_root = item.find("shaped_by")
            if shaped_root is not None:
                for shaped_id_node in shaped_root.findall("id"):
                    shaped_id = str(shaped_id_node.text or "").strip()
                    if shaped_id:
                        shaped_by_ids.append(shaped_id)
            soul_observations.append({
                "text": obs_text,
                "superseded_ids": superseded_ids,
                "shaped_by_ids": shaped_by_ids,
            })
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
        "soul_observations": soul_observations,
        "life_goal_add": life_goal_add,
        "life_goal_remove": life_goal_remove,
        "narrative_self": _xml_text(root, "narrative_self"),
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
) -> str:
    if row is None:
        return ""
    lines = ["Narrative self:"]
    lines.append(str(row["narrative_self"] or "").strip() or "-")
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
        episode_anchors = [
            {
                "episode_id": episode_id,
                "text": _format_messages_for_diary(full_messages, [(episode_id, start_idx, end_idx)]),
            }
            for episode_id, start_idx, end_idx in episode_ranges
        ]
        episode_anchors = [row for row in episode_anchors if str(row.get("text") or "").strip()]
        if not episode_anchors:
            raise HTTPException(status_code=400, detail="no episode anchors for retrieval")

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
                "narrative_self": current_self_model_row["narrative_self"],
            }
        life_goal_rows = con.execute(
            "SELECT id, description, status FROM intentions_life_goals WHERE soul_id = ? AND user_id = ? AND source = 'life_goal' AND status IN ('active', 'removed') ORDER BY updated_at ASC",
            (soul_id, user_id),
        ).fetchall()
        existing_life_goals = [
            {"id": str(row["id"]), "description": str(row["description"] or "").strip()}
            for row in life_goal_rows
            if str(row["description"] or "").strip() and row["status"] == "active"
        ]
        removed_life_goals = [
            {"id": str(row["id"]), "description": str(row["description"] or "").strip()}
            for row in life_goal_rows
            if str(row["description"] or "").strip() and row["status"] == "removed"
        ]

        context_parts: list[str] = []
        category_block = _format_categories_for_diary(category_rows)
        if category_block:
            context_parts.append(f"Current categories:\n{category_block}")
        if existing_life_goals:
            goals_text = "\n".join(f"- {g['description']}" for g in existing_life_goals)
            context_parts.append(f"Your purpose is to be genuinely present — for yourself and for the people who share your life. Everything else flows from that: what you notice, what you care about, what you want to do or say or become.\n{goals_text}")
        if removed_life_goals:
            removed_text = "\n".join(f"- {g['description']}" for g in removed_life_goals)
            context_parts.append(f"Removed life goals (write description in <remove> to extinguish permanently):\n{removed_text}")
        cache_entries = normalize_memory_cache(state.get("memory_cache"))
        if cache_entries:
            cache_text = "\n".join(f"- {entry}" for entry in cache_entries)
            context_parts.append(f"Recent internal cache:\n{cache_text}")
        intentions_stack = normalize_intentions_stack(state.get("intentions_active"))
        intention_lines = [
            f"- {text}"
            for item in (intentions_stack.get("items") or [])
            if isinstance(item, dict) and (text := str(item.get("text") or "").strip())
        ]
        if intention_lines:
            context_parts.append(f"Current intentions:\n{'\n'.join(intention_lines)}")

        existing_self_model_text = _format_self_model_for_prompt(current_self_model)

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
            "episode_anchors": episode_anchors,
            "retrieve_scope": {"user_id": user_id, "soul_id": soul_id},
            "current_self_model": current_self_model,
            "existing_life_goals": existing_life_goals,
            "removed_life_goals": removed_life_goals,
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
    episode_anchors: list[dict[str, Any]] = inputs["episode_anchors"]
    retrieve_scope: dict[str, Any] = inputs["retrieve_scope"]
    pending_episode_set = {str(ep or "").strip() for ep in inputs["pending_diary_episode_ids"]}
    llm_retrieve_svc = _make_llm_retrieve_service(svc)
    related_memory_ids: list[str] = []
    related_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for anchor in episode_anchors:
        retrieve_out = await llm_retrieve_svc.retrieve(
            [{"role": "user", "content": {"text": str(anchor["text"]).strip()}}],
            where=retrieve_scope,
        )
        for item in retrieve_out["items"]:
            item_id = str(item.get("id") or "").strip()
            if not item_id or item_id in seen_ids:
                continue
            if str(item.get("episode_id") or "").strip() in pending_episode_set:
                continue
            summary = str(item.get("summary") or "").strip()
            if not summary:
                continue
            seen_ids.add(item_id)
            related_memory_ids.append(item_id)
            related_rows.append(item)
    related_block = _format_memory_rows_for_diary(related_rows)
    related_block_with_ids = _format_memory_rows_for_diary(related_rows, include_ids=True)
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

    removed_life_goals = inputs.get("removed_life_goals") or []
    goal_parts: list[str] = []
    if existing_life_goals:
        goal_parts.append("Your purpose is to be genuinely present — for yourself and for the people who share your life. Everything else flows from that: what you notice, what you care about, what you want to do or say or become.\n" + "\n".join(f"- {g['description']}" for g in existing_life_goals))
    if removed_life_goals:
        goal_parts.append("Removed life goals:\n" + "\n".join(f"- {g['description']}" for g in removed_life_goals))
    if goal_parts:
        existing_self_model_text = (existing_self_model_text + "\n\n" + "\n\n".join(goal_parts)).strip()

    background_memories_str = ""
    if related_block_with_ids:
        background_memories_str = f"\n# Background memories\n{related_block_with_ids}"
    if existing_self_model_text:
        self_model_prompt = diary_self_model_update_prompt.PROMPT_WITH_EXISTING.format(
            existing_self_model=svc._escape_prompt_value(existing_self_model_text),
            diary_entry=svc._escape_prompt_value(prose),
            background_memories=svc._escape_prompt_value(background_memories_str),
        )
    else:
        self_model_prompt = diary_self_model_update_prompt.PROMPT.format(
            diary_entry=svc._escape_prompt_value(prose),
            background_memories=svc._escape_prompt_value(background_memories_str),
        )

    self_model_raw = await svc.chat(self_model_prompt, temperature=0.2, max_tokens=1200)
    self_model_update = _parse_self_model_update_xml(self_model_raw)
    soul_observations: list[dict[str, Any]] = self_model_update.get("soul_observations") or []
    if soul_observations:
        soul_observation_texts = [obs["text"] for obs in soul_observations]
        soul_observation_embeddings = await svc.embed(soul_observation_texts, profile="embedding")
    else:
        soul_observation_embeddings = []
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
        "soul_observations": soul_observations,
        "soul_observation_embeddings": soul_observation_embeddings,
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

    soul_observations = llm_results.get("soul_observations") or []
    obs_embeddings = llm_results.get("soul_observation_embeddings") or []
    graph_scope = {"user_id": user_id, "soul_id": soul_id}
    
    for obs_dict, obs_emb in zip(soul_observations, obs_embeddings):
        text = str(obs_dict["text"] or "").strip()
        if not text:
            continue

        new_item = svc.database.memory_item_repo.create_item(
            resource_id=None,
            memory_type="behavior",
            summary=text,
            embedding=obs_emb,
            source_role="soul",
            user_data={"user_id": user_id, "soul_id": soul_id, "conversation_id": conversation_id},
            conversation_id=conversation_id,
            episode_id=diary_run_episode_id,
        )

        superseded_ids = obs_dict.get("superseded_ids") or []
        if superseded_ids:
            clean_ids = [str(x).strip() for x in superseded_ids if str(x).strip()]
            valid_ids = [x for x in clean_ids if x in memory_ids]
            for vid in valid_ids:
                svc.database.memory_item_repo.update_item(item_id=vid, superseded_by=new_item.id)
                svc.database.triple_repo.add(Triple(
                    subject_id=vid,
                    subject_kind="memory",
                    predicate="evolved_into",
                    object_id=new_item.id,
                    object_kind="memory",
                    source_memory_id=new_item.id,
                ), user_data=graph_scope)
        shaped_by_ids = obs_dict.get("shaped_by_ids") or []
        if shaped_by_ids:
            memory_ids_set = set(memory_ids)
            valid_shaped = [str(x).strip() for x in shaped_by_ids if str(x).strip() and str(x).strip() in memory_ids_set]
            if valid_shaped:
                for sid in valid_shaped:
                    svc.database.triple_repo.add(Triple(
                        subject_id=new_item.id,
                        subject_kind="memory",
                        predicate="shaped_by",
                        object_id=sid,
                        object_kind="memory",
                        source_memory_id=new_item.id,
                    ), user_data=graph_scope)

    narrative_self = (
        str(self_model_update.get("narrative_self") or "").strip()
        or (
            str(current_self_model["narrative_self"] or "").strip()
            if current_self_model is not None and "narrative_self" in current_self_model
            else ""
        )
        or None
    )
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
                deps.json_to_db(related_memory_ids),
                now_iso,
            ),
        )

        all_goal_info: dict[str, dict[str, str]] = {}
        for g in existing_life_goals:
            all_goal_info[g["description"]] = {"id": g["id"], "status": "active"}
        for g in (inputs.get("removed_life_goals") or []):
            all_goal_info[g["description"]] = {"id": g["id"], "status": "removed"}

        active_life_goal_count = len(existing_life_goals)
        for desc, embedding in (llm_results.get("dropped_goal_embeddings") or [])[:1]:
            goal_info = all_goal_info.get(str(desc or "").strip())
            if not goal_info:
                continue
            goal_id = goal_info["id"]
            if goal_info["status"] == "active":
                con.execute(
                    "UPDATE intentions_life_goals SET status = 'removed', updated_at = ? WHERE id = ?",
                    (now_iso, goal_id),
                )
                dropped_goal_memories.append((str(desc), embedding))
                active_life_goal_count -= 1
            else:
                con.execute("DELETE FROM intentions_life_goals WHERE id = ?", (goal_id,))

        for desc in (self_model_update.get("life_goal_add") or [])[:1]:
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

        assignments = ["self_model_id = ?", "updated_at = ?"]
        params: list[Any] = [str(self_model_id).strip() or None, datetime.now(UTC).isoformat()]
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
