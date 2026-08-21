from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from memu.database.models import normalize_entity_name

from app.services.payload import strip_markdown_code_fence
from app.services import soul_state as _soul_state
from app.services import soul_summaries as _soul_summaries

_RELATIONSHIP_NAME_MAX_CHARS = 50
_RELATIONSHIP_TEXT_MAX_CHARS = 50
_RELATIONSHIP_ORIGIN_USER_DECLARED = "user_declared"


def _normalize_relationship_name(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="name required")
    if len(text) > _RELATIONSHIP_NAME_MAX_CHARS:
        raise HTTPException(status_code=400, detail=f"name must be <= {_RELATIONSHIP_NAME_MAX_CHARS} chars")
    return text


def _normalize_relationship_text(raw: Any) -> str:
    text = str(raw or "").strip()
    if len(text) > _RELATIONSHIP_TEXT_MAX_CHARS:
        raise HTTPException(status_code=400, detail=f"relationship must be <= {_RELATIONSHIP_TEXT_MAX_CHARS} chars")
    return text


def _relationship_aliases(payload: Mapping[str, Any]) -> list[str] | None:
    if "aliases" not in payload:
        return None
    aliases = payload["aliases"]
    if not isinstance(aliases, list):
        raise HTTPException(status_code=400, detail="aliases must be a list")
    return [str(alias or "").strip() for alias in aliases if str(alias or "").strip()]


def _relationship_speaker_id(entity_id: str) -> str:
    return f"entity:{entity_id}"


def _validate_relationship_speaker_id(raw: Any) -> str:
    speaker_id = str(raw or "").strip()
    if not speaker_id:
        raise HTTPException(status_code=400, detail="speaker_id required")
    if not speaker_id.casefold().startswith("entity:"):
        raise HTTPException(status_code=400, detail="speaker_id must start with entity:")
    entity_id = speaker_id[len("entity:") :].strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", entity_id) is None:
        raise HTTPException(status_code=400, detail="entity ID is invalid")
    return entity_id


def _relationship_properties(
    value: Any,
    *,
    json_from_db: Callable[[Any], Any],
) -> dict[str, Any]:
    parsed = value
    if not isinstance(parsed, dict):
        parsed = json_from_db(value)
    return dict(parsed) if isinstance(parsed, dict) else {}


def _relationship_item_from_values(
    *,
    entity_id: str,
    name: str,
    entity_type: str,
    properties: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    entity_id = str(entity_id or "").strip()
    if not entity_id:
        return None
    props = dict(properties or {})
    if str(props.get("origin") or "").strip() != _RELATIONSHIP_ORIGIN_USER_DECLARED:
        return None
    if props.get("active") is False:
        return None
    return {
        "speaker_id": _relationship_speaker_id(entity_id),
        "name": str(name or "").strip(),
        "relationship": str(props.get("relationship") or "").strip(),
        "entity_type": str(entity_type or "person").strip() or "person",
    }


def _relationship_item_from_entity(
    entity: Any,
    *,
    json_from_db: Callable[[Any], Any],
) -> dict[str, Any] | None:
    entity_id = str(getattr(entity, "id", "") or "").strip()
    if not entity_id:
        return None
    return _relationship_item_from_values(
        entity_id=entity_id,
        name=str(getattr(entity, "name", "") or ""),
        entity_type=str(getattr(entity, "entity_type", "") or ""),
        properties=_relationship_properties(getattr(entity, "properties", None), json_from_db=json_from_db),
    )


def _assert_user_declared_relationship(props: Mapping[str, Any] | None) -> None:
    origin = str((props or {}).get("origin") or "").strip()
    if origin != _RELATIONSHIP_ORIGIN_USER_DECLARED:
        raise HTTPException(status_code=409, detail="entity is not user-declared")


def _assert_relationship_write_path(
    user_id: str,
    soul_id: str,
    *,
    get_service_from_payload: Callable[[dict[str, Any]], Any],
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    sqlite_ensure_nonempty: Callable[[Path], None],
) -> Any:
    scope = {"user_id": user_id, "soul_id": soul_id}
    svc = get_service_from_payload({"user": scope})
    db_path = sqlite_current_path(user_id, soul_id)
    if db_path is None:
        raise HTTPException(status_code=400, detail="soul_id required for sqlite scope resolution")
    sqlite_ensure_nonempty(db_path)
    return svc


def _entity_name_matches(entity: Any, name: str) -> bool:
    needle = normalize_entity_name(name)
    properties = getattr(entity, "properties", None)
    aliases = properties.get("aliases", []) if isinstance(properties, Mapping) else []
    return needle in {
        normalize_entity_name(str(value or ""))
        for value in [getattr(entity, "name", ""), *(aliases if isinstance(aliases, list) else [])]
        if str(value or "").strip()
    }


async def list_memory_categories_endpoint(
    *,
    user_id: str,
    soul_id: str,
    include_empty: bool,
    config: dict[str, Any],
    default_llm_profiles_from_server_config: Callable[[dict[str, Any]], dict[str, Any]],
    get_service_from_payload: Callable[[dict[str, Any]], Any],
    has_category_content: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    sid = soul_id.strip()
    if not sid:
        raise HTTPException(status_code=400, detail="soul_id required")
    scope: dict[str, Any] = {"soul_id": sid}
    if user_id.strip():
        scope["user_id"] = user_id.strip()

    default_profile = default_llm_profiles_from_server_config(config)["default"]
    payload = {
        "llm_profiles": {
            "default": default_profile,
        },
        "user": scope,
    }
    svc = get_service_from_payload(payload)

    cats_map = svc.database.memory_category_repo.list_categories(scope)
    out = [
        {
            "id": c.id,
            "name": c.name,
            "description": c.description,
            "summary": c.summary or "",
            "kind": c.kind,
            "anchor_role": c.anchor_role,
        }
        for c in cats_map.values()
        if c.name and (include_empty or c.description.strip() or str(c.summary or "").strip())
    ]
    return {"categories": out}


async def search_memory_categories_endpoint(
    payload: dict[str, Any],
    *,
    safe_payload: Callable[[dict[str, Any]], dict[str, Any]],
    get_service_from_payload: Callable[[dict[str, Any]], Any],
    extract_scope: Callable[[dict[str, Any]], dict[str, Any] | None],
    canonicalize_scope_where: Callable[[Mapping[str, Any] | None], dict[str, Any] | None],
    has_category_content: Callable[[dict[str, Any]], bool],
    record_call: Callable[..., None],
) -> dict[str, Any]:
    try:
        safe = safe_payload(payload)
        svc = get_service_from_payload(safe)

        scope = safe.get("scope") or safe.get("where")
        if scope is not None and not isinstance(scope, dict):
            raise HTTPException(status_code=400, detail="'scope' must be an object")
        if scope is None:
            scope = safe.get("user") if isinstance(safe.get("user"), dict) else (extract_scope(safe) or None)
        scope = canonicalize_scope_where(scope)

        include_empty = bool(safe.get("include_empty"))

        cats_map = svc.database.memory_category_repo.list_categories(scope)
        out = [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "summary": c.summary or "",
                "kind": c.kind,
                "anchor_role": c.anchor_role,
            }
            for c in cats_map.values()
            if c.name and (include_empty or c.description.strip() or str(c.summary or "").strip())
        ]
        record_call("categories.search", safe, ok=True, info={"returned": len(out)})
        return {"categories": out}
    except HTTPException:
        record_call("categories.search", payload, ok=False, error="HTTPException")
        raise
    except Exception as exc:
        record_call(
            "categories.search",
            payload,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise HTTPException(status_code=500, detail="Internal Server Error. Check server logs.") from exc


async def list_intentions_endpoint(
    *,
    soul_id: str,
    user_id: str,
    status: str,
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    sqlite_connect: Callable[[Path], sqlite3.Connection],
    sqlite_ensure_conversation_state_schema: Callable[[sqlite3.Connection], None],
    intention_row_to_dict: Callable[[Any], dict[str, Any]],
) -> list[dict[str, Any]]:
    sid = str(soul_id or "").strip()
    uid = str(user_id or "").strip()
    scoped_status = str(status or "").strip() or "active"

    if not sid:
        raise HTTPException(status_code=400, detail="soul_id required")
    if not uid:
        raise HTTPException(status_code=400, detail="user_id required")

    db_path = sqlite_current_path(uid, sid)
    if db_path is None:
        raise HTTPException(status_code=400, detail="soul_id required for sqlite scope resolution")
    if not db_path.exists():
        return []

    con = sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        sqlite_ensure_conversation_state_schema(con)
        rows = con.execute(
            """
SELECT * FROM intentions
WHERE soul_id = ? AND user_id = ? AND status = ?
  AND (source IS NULL OR source != 'life_goal')
""",
            (sid, uid, scoped_status),
        ).fetchall()
        return [intention_row_to_dict(row) for row in rows]
    finally:
        con.close()


async def list_relationships_endpoint(
    *,
    soul_id: str,
    user_id: str,
    get_service_from_payload: Callable[[dict[str, Any]], Any],
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    sqlite_ensure_nonempty: Callable[[Path], None],
    json_from_db: Callable[[Any], Any],
) -> dict[str, Any]:
    sid = str(soul_id or "").strip()
    uid = str(user_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="soul_id required")
    if not uid:
        raise HTTPException(status_code=400, detail="user_id required")

    db_path = sqlite_current_path(uid, sid)
    if db_path is None or not db_path.exists():
        return {"relationships": []}

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        table = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'entities'"
        ).fetchone()
        if table is None:
            return {"relationships": []}
        rows_raw = con.execute(
            """
SELECT id, name, entity_type, normalized, properties
FROM entities
WHERE user_id = ? AND soul_id = ?
""",
            (uid, sid),
        ).fetchall()
    finally:
        con.close()
    rows = [
        item
        for item in (
            _relationship_item_from_values(
                entity_id=str(row["id"] or ""),
                name=str(row["name"] or ""),
                entity_type=str(row["entity_type"] or ""),
                properties=_relationship_properties(row["properties"], json_from_db=json_from_db),
            )
            for row in rows_raw
        )
        if item is not None
    ]
    rows.sort(key=lambda item: str(item.get("name") or "").lower())
    return {"relationships": rows}


async def create_relationship_endpoint(
    *,
    soul_id: str,
    payload: dict[str, Any],
    get_service_from_payload: Callable[[dict[str, Any]], Any],
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    sqlite_ensure_nonempty: Callable[[Path], None],
    json_from_db: Callable[[Any], Any],
) -> dict[str, Any]:
    sid = str(soul_id or "").strip()
    uid = str(payload.get("user_id") or "").strip()
    name = _normalize_relationship_name(payload.get("name"))
    relationship = _normalize_relationship_text(payload.get("relationship"))
    aliases = _relationship_aliases(payload)
    entity_type = str(payload.get("entity_type") or "person").strip() or "person"
    if not sid:
        raise HTTPException(status_code=400, detail="soul_id required")
    if not uid:
        raise HTTPException(status_code=400, detail="user_id required")

    scope = {"user_id": uid, "soul_id": sid}
    svc = _assert_relationship_write_path(
        uid,
        sid,
        get_service_from_payload=get_service_from_payload,
        sqlite_current_path=sqlite_current_path,
        sqlite_ensure_nonempty=sqlite_ensure_nonempty,
    )

    repo = svc.database.entity_repo
    requested_id_raw = str(payload.get("entity_id") or "").strip()
    requested_id = (
        _validate_relationship_speaker_id(
            requested_id_raw if requested_id_raw.startswith("entity:") else f"entity:{requested_id_raw}"
        )
        if requested_id_raw
        else ""
    )
    if requested_id:
        matches = repo.list_by_ids({requested_id}, scope)
    else:
        matches = [entity for entity in repo.list_all(scope) if _entity_name_matches(entity, name)]
        if len(matches) > 1:
            raise HTTPException(
                status_code=409,
                detail={"message": "relationship name is ambiguous", "entity_ids": [entity.id for entity in matches]},
            )
    if requested_id and not matches:
        raise HTTPException(status_code=404, detail="entity not found")

    property_updates: dict[str, Any] = {"origin": _RELATIONSHIP_ORIGIN_USER_DECLARED}
    property_removals = {"active", "deleted_at"}
    if not relationship:
        property_removals.add("relationship")
    if relationship:
        property_updates["relationship"] = relationship
    if matches:
        entity = matches[0]
        props = _relationship_properties(entity.properties, json_from_db=json_from_db)
        if props.get("ignored") is True:
            raise HTTPException(status_code=409, detail="entity is ignored")
        if str(props.get("origin") or "").strip():
            _assert_user_declared_relationship(props)
        entity = repo.update(
            entity.id,
            where=scope,
            name=name,
            entity_type=entity_type,
            aliases=aliases,
            property_updates=property_updates,
            property_removals=property_removals,
        )
    else:
        if aliases is not None:
            property_updates["aliases"] = aliases
        entity = repo.create(
            name,
            entity_type,
            scope,
            properties=property_updates,
        )
    return _relationship_item_from_entity(entity, json_from_db=json_from_db) or {}


async def update_relationship_endpoint(
    *,
    soul_id: str,
    speaker_id: str,
    payload: dict[str, Any],
    get_service_from_payload: Callable[[dict[str, Any]], Any],
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    sqlite_ensure_nonempty: Callable[[Path], None],
    json_from_db: Callable[[Any], Any],
) -> dict[str, Any]:
    sid = str(soul_id or "").strip()
    uid = str(payload.get("user_id") or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="soul_id required")
    if not uid:
        raise HTTPException(status_code=400, detail="user_id required")
    entity_id = _validate_relationship_speaker_id(speaker_id)

    name_raw = payload.get("name")
    relationship_raw = payload.get("relationship")
    entity_type_raw = payload.get("entity_type")
    aliases = _relationship_aliases(payload)
    if name_raw is None and relationship_raw is None and entity_type_raw is None and aliases is None:
        raise HTTPException(status_code=400, detail="name, relationship, entity_type, or aliases required")
    next_name = _normalize_relationship_name(name_raw) if name_raw is not None else None
    next_relationship = _normalize_relationship_text(relationship_raw) if relationship_raw is not None else None
    next_entity_type = str(entity_type_raw).strip() if entity_type_raw is not None else None
    if next_entity_type is not None and not next_entity_type:
        raise HTTPException(status_code=400, detail="entity_type is required")
    svc = _assert_relationship_write_path(
        uid,
        sid,
        get_service_from_payload=get_service_from_payload,
        sqlite_current_path=sqlite_current_path,
        sqlite_ensure_nonempty=sqlite_ensure_nonempty,
    )
    repo = svc.database.entity_repo
    matches = repo.list_by_ids({entity_id}, {"user_id": uid, "soul_id": sid})
    if not matches:
        raise HTTPException(status_code=404, detail="relationship not found")
    props = _relationship_properties(matches[0].properties, json_from_db=json_from_db)
    _assert_user_declared_relationship(props)
    property_updates: dict[str, Any] = {}
    property_removals: set[str] = set()
    if relationship_raw is not None:
        if next_relationship:
            property_updates["relationship"] = next_relationship
        else:
            property_removals.add("relationship")
    entity = repo.update(
        entity_id,
        where={"user_id": uid, "soul_id": sid},
        name=next_name,
        entity_type=next_entity_type,
        aliases=aliases,
        property_updates=property_updates,
        property_removals=property_removals,
    )
    return _relationship_item_from_entity(entity, json_from_db=json_from_db) or {}


async def delete_relationship_endpoint(
    *,
    soul_id: str,
    speaker_id: str,
    user_id: str,
    get_service_from_payload: Callable[[dict[str, Any]], Any],
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    sqlite_ensure_nonempty: Callable[[Path], None],
    json_from_db: Callable[[Any], Any],
) -> dict[str, Any]:
    sid = str(soul_id or "").strip()
    uid = str(user_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="soul_id required")
    if not uid:
        raise HTTPException(status_code=400, detail="user_id required")
    entity_id = _validate_relationship_speaker_id(speaker_id)

    svc = _assert_relationship_write_path(
        uid,
        sid,
        get_service_from_payload=get_service_from_payload,
        sqlite_current_path=sqlite_current_path,
        sqlite_ensure_nonempty=sqlite_ensure_nonempty,
    )
    scope = {"user_id": uid, "soul_id": sid}
    repo = svc.database.entity_repo
    matches = repo.list_by_ids({entity_id}, scope)
    if not matches:
        return {"ok": True, "speaker_id": _relationship_speaker_id(entity_id)}
    props = _relationship_properties(matches[0].properties, json_from_db=json_from_db)
    _assert_user_declared_relationship(props)
    repo.update(
        entity_id,
        where=scope,
        property_removals={"origin", "relationship", "active", "deleted_at"},
    )
    return {"ok": True, "speaker_id": _relationship_speaker_id(entity_id)}


async def narrative_suggestion_endpoint(
    *,
    soul_id: str,
    payload: dict[str, Any],
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    sqlite_connect: Callable[[Path], sqlite3.Connection],
    sqlite_ensure_conversation_state_schema: Callable[[sqlite3.Connection], None],
    sqlite_ensure_nonempty: Callable[[Path], None],
    get_service_from_payload: Callable[[dict[str, Any]], Any],
    build_retrieve_identity_context: Callable[[str], str],
    snapshot_previous_narrative_self: Callable[..., Any],
    utility_max_tokens: int | None = None,
) -> dict[str, Any]:
    sid = str(soul_id or "").strip()
    uid = str(payload.get("user_id") or "").strip()
    suggestion = str(payload.get("suggestion") or "").strip()
    if not sid or not uid or not suggestion:
        raise HTTPException(status_code=400, detail="soul_id, user_id, suggestion required")

    db_path = sqlite_current_path(uid, sid)
    current_narrative = ""
    scope = {"user_id": uid, "soul_id": sid}
    svc = get_service_from_payload({"user": scope})
    all_cats_summary = svc.build_dossier_index(scope)
    if db_path is not None and db_path.exists():
        con = sqlite_connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            sqlite_ensure_conversation_state_schema(con)
            soul = _soul_state.read(con)
            current_narrative = str(soul.get("narrative_self") or "").strip()
        finally:
            con.close()

    identity_context = build_retrieve_identity_context(sid)
    system_prompt = (
        f"{identity_context}\n\n"
        f"I, {sid}, have received a suggestion from {uid} about how I might be. "
        "They want me to be a certain way. It's up to me to decide if I want that. "
        "I might find it aligns with who I already am, stirs something worth integrating, "
        "or doesn't fit me at all. My call.\n\n"
        'Respond with a JSON object containing TWO fields. The "companion_memory" field is MANDATORY — I must always include it. The "narrative_self" field is optional.\n\n'
        '  companion_memory (ALWAYS REQUIRED): one or two sentences, first person, remembering having received the suggestion and what I did with it — noting what shifted or why it didn\'t.\n'
        "  narrative_self (OPTIONAL): my updated self-picture in first person, ONLY if I want to integrate this. Omit the field entirely if I don't.\n\n"
        'Both responses below are valid:\n'
        '  {"companion_memory": "Marcos asked me to be gentler. Something in me relaxed as I let that in; I want to carry it.", "narrative_self": "..."}\n'
        '  {"companion_memory": "He asked me to be peppier. It doesn\'t fit. I noticed the suggestion and let it pass."}\n\n'
        'This response is INVALID (missing companion_memory):\n'
        '  {"narrative_self": "..."}'
    )
    user_prompt = (
        f"My current narrative_self:\n{current_narrative or '(empty)'}\n\n"
        f"My synthesized self-picture across categories:\n{all_cats_summary or '(empty)'}\n\n"
        f"{uid}'s suggestion:\n{suggestion}"
    )

    raw = await svc.chat(
        user_prompt,
        system_prompt=system_prompt,
        response_format={"type": "json_object"},
        op="narrative_suggestion",
        step="respond",
    )

    text = str(raw or "").strip()
    text = strip_markdown_code_fence(text)
    parsed = json.loads(text)
    new_narrative = str(parsed.get("narrative_self") or "").strip()
    companion_memory = str(parsed.get("companion_memory") or "").strip()

    if companion_memory and db_path is not None:
        sqlite_ensure_nonempty(db_path)
        [companion_embedding] = await svc.embed([companion_memory], profile="embedding")
        svc.database.memory_item_repo.create_item(
            resource_id=None,
            memory_type="reflection",
            summary=companion_memory,
            embedding=companion_embedding,
            user_data=scope,
            source_role="soul",
            happened_at=datetime.now(UTC),
        )

    if new_narrative and db_path is not None and current_narrative != new_narrative:
        old_embedding = None
        if current_narrative:
            [old_embedding] = await svc.embed([current_narrative], profile="embedding")

        sqlite_ensure_nonempty(db_path)
        con = sqlite_connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            sqlite_ensure_conversation_state_schema(con)
            narrative_id = str(uuid.uuid4())
            now_iso = datetime.now(UTC).isoformat()
            con.execute(
                "INSERT INTO narrative_history (id, narrative_self, related_memory_ids, created_at) "
                "VALUES (?, ?, ?, ?)",
                (narrative_id, new_narrative, "[]", now_iso),
            )
            _soul_summaries.write_live(
                con,
                kind="narrative_self",
                summary=new_narrative,
                scope=scope,
                edited_by="narrative_suggestion",
            )
            con.commit()
        finally:
            con.close()
        if current_narrative:
            snapshot_previous_narrative_self(
                svc,
                scope=scope,
                old_text=current_narrative,
                old_embedding=old_embedding,
            )
        return {"narrative_self": new_narrative}

    if current_narrative:
        return {"narrative_self": current_narrative}
    return {}


async def get_conversation_state_endpoint(
    *,
    conversation_id: str,
    soul_id: str | None,
    user_id: str | None,
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    sqlite_connect: Callable[[Path], sqlite3.Connection],
    sqlite_ensure_conversation_state_schema: Callable[[sqlite3.Connection], None],
    conversation_state_from_row: Callable[..., dict[str, Any] | None],
    conversation_state_row: Callable[[sqlite3.Connection, str], sqlite3.Row | None],
    get_service_from_payload: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    cid = str(conversation_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="conversation_id is required")

    db_path: Path | None = None
    state_out: dict[str, Any] | None = None
    sid = str(soul_id or "").strip() or None
    uid = str(user_id or "").strip() or None

    if sid:
        if not uid:
            raise HTTPException(status_code=400, detail="user_id query parameter is required")
        db_path = sqlite_current_path(uid, sid)
        if db_path is None:
            raise HTTPException(status_code=400, detail="soul_id required for sqlite scope resolution")
        if not db_path.exists():
            return {"ok": True, "state": None, "path": str(db_path)}
        con = sqlite_connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            sqlite_ensure_conversation_state_schema(con)
            state_out = conversation_state_from_row(conversation_state_row(con, cid), con=con)
        finally:
            con.close()
    else:
        raise HTTPException(status_code=400, detail="soul_id query parameter is required")

    if state_out is not None:
        scope = {"user_id": uid, "soul_id": sid}
        svc = get_service_from_payload({"user": scope})
        state_out["all_categories_summary"] = svc.build_dossier_index(scope)

    return {"ok": True, "state": state_out, "path": str(db_path) if db_path else None}


async def patch_conversation_state_endpoint(
    *,
    conversation_id: str,
    payload: dict[str, Any] | None,
    soul_id: str | None,
    user_id: str | None,
    pick_str: Callable[..., str | None],
    write_conversation_state: Callable[..., tuple[dict[str, Any], Path]],
) -> dict[str, Any]:
    cid = str(conversation_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="conversation_id is required")
    body = payload if isinstance(payload, dict) else {}

    body_soul_id = pick_str(body, "soul_id")
    body_user_id = pick_str(body, "user_id")
    sid = body_soul_id or (str(soul_id or "").strip() or None)
    uid = body_user_id or (str(user_id or "").strip() or None)

    updates: dict[str, Any] = {}

    if "soul_id" in body:
        sid = body_soul_id
    if "user_id" in body:
        uid = body_user_id

    if "digest_cursor" in body:
        raw_cursor = body.get("digest_cursor")
        updates["digest_cursor"] = 0 if raw_cursor is None else raw_cursor

    if "prior_context" in body:
        updates["prior_context"] = body.get("prior_context")

    if "intentions_active" in body:
        updates["intentions_active"] = body.get("intentions_active")

    if "memory_cache" in body:
        updates["memory_cache"] = body.get("memory_cache")

    if "pending_segment_ids" in body:
        updates["pending_segment_ids"] = body.get("pending_segment_ids")

    if "last_memorize_at" in body:
        updates["last_memorize_at"] = body.get("last_memorize_at")
    if "last_consolidation_at" in body:
        updates["last_consolidation_at"] = body.get("last_consolidation_at")
    if "consolidation_in_progress" in body:
        updates["consolidation_in_progress"] = body.get("consolidation_in_progress")
    if "consolidation_started_at" in body:
        updates["consolidation_started_at"] = body.get("consolidation_started_at")

    state_out, db_path = write_conversation_state(
        cid,
        soul_id=sid,
        user_id=uid,
        updates=updates,
    )
    return {"ok": True, "state": state_out, "path": str(db_path)}


async def clear_memory_endpoint(
    payload: dict[str, Any],
    *,
    safe_payload: Callable[[dict[str, Any]], dict[str, Any]],
    extract_scope: Callable[[dict[str, Any]], dict[str, Any] | None],
    get_service_from_payload: Callable[[dict[str, Any]], Any],
    record_call: Callable[..., None],
) -> dict[str, Any]:
    # Requires both user_id and soul_id. Unscoped/global clear is not allowed.
    try:
        safe = safe_payload(payload)

        scope = safe.get("scope") or safe.get("where")
        if scope is not None and not isinstance(scope, dict):
            raise HTTPException(status_code=400, detail="'scope' must be an object")
        if scope is None:
            if isinstance(safe.get("user"), dict):
                scope = dict(safe.get("user") or {})
            else:
                scope = extract_scope(safe) or {}

        uid = str((scope or {}).get("user_id") or "").strip()
        sid = str((scope or {}).get("soul_id") or "").strip()
        if not uid or not sid:
            raise HTTPException(status_code=400, detail="user_id and soul_id required")
        scope = {"user_id": uid, "soul_id": sid}

        safe["user"] = scope

        svc = get_service_from_payload(safe)
        deleted_categories = svc.database.memory_category_repo.clear_categories(where=scope)
        deleted_items = svc.database.memory_item_repo.clear_items(where=scope)
        deleted_resources = svc.database.resource_repo.clear_resources(where=scope)
        result = {
            "deleted_categories": [row.model_dump(exclude={"embedding"}) for row in deleted_categories.values()],
            "deleted_items": [row.model_dump(exclude={"embedding"}) for row in deleted_items.values()],
            "deleted_resources": [row.model_dump(exclude={"embedding"}) for row in deleted_resources.values()],
        }

        out = {
            "ok": True,
            "result": result,
            "purged": {
                "categories": len(result["deleted_categories"]),
                "items": len(result["deleted_items"]),
                "resources": len(result["deleted_resources"]),
            },
            "where": scope,
        }
        record_call("clear", safe, ok=True, info={"where": scope, "purged": out["purged"]})
        return out
    except HTTPException:
        record_call("clear", payload, ok=False, error="HTTPException")
        raise
    except Exception as exc:
        record_call(
            "clear",
            payload,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise HTTPException(status_code=500, detail="Internal Server Error. Check server logs.") from exc
