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

from app.services import soul_state as _soul_state

_RELATIONSHIP_NAME_MAX_CHARS = 50
_RELATIONSHIP_TEXT_MAX_CHARS = 50
_RELATIONSHIP_RESERVED_PREFIXES = ("user:", "soul:", "peer:", "environment:")
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


def _slugify_relationship_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower()).strip("_")
    if not slug:
        raise HTTPException(status_code=400, detail="name does not contain usable slug characters")
    return slug


def _relationship_speaker_id_from_normalized(normalized: str) -> str:
    return f"entity:{normalized}"


def _validate_relationship_speaker_id(raw: Any) -> str:
    speaker_id = str(raw or "").strip().lower()
    if not speaker_id:
        raise HTTPException(status_code=400, detail="speaker_id required")
    if any(speaker_id.startswith(prefix) for prefix in _RELATIONSHIP_RESERVED_PREFIXES):
        raise HTTPException(status_code=400, detail="reserved speaker prefix")
    if not speaker_id.startswith("entity:"):
        raise HTTPException(status_code=400, detail="speaker_id must start with entity:")
    normalized = speaker_id[len("entity:") :].strip()
    if not normalized or re.search(r"[^a-z0-9_]", normalized):
        raise HTTPException(status_code=400, detail="speaker_id slug is invalid")
    return normalized


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
    normalized: str,
    name: str,
    entity_type: str,
    properties: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    props = dict(properties or {})
    if str(props.get("origin") or "").strip() != _RELATIONSHIP_ORIGIN_USER_DECLARED:
        return None
    if props.get("active") is False:
        return None
    return {
        "speaker_id": _relationship_speaker_id_from_normalized(normalized),
        "name": str(name or "").strip(),
        "relationship": str(props.get("relationship") or "").strip(),
        "entity_type": str(entity_type or "person").strip() or "person",
    }


def _relationship_item_from_entity(
    entity: Any,
    *,
    json_from_db: Callable[[Any], Any],
) -> dict[str, Any] | None:
    normalized = str(getattr(entity, "normalized", "") or "").strip()
    if not normalized:
        return None
    return _relationship_item_from_values(
        normalized=normalized,
        name=str(getattr(entity, "name", "") or ""),
        entity_type=str(getattr(entity, "entity_type", "") or ""),
        properties=_relationship_properties(getattr(entity, "properties", None), json_from_db=json_from_db),
    )


def _assert_user_declared_relationship(props: Mapping[str, Any] | None) -> None:
    origin = str((props or {}).get("origin") or "").strip()
    if origin != _RELATIONSHIP_ORIGIN_USER_DECLARED:
        raise HTTPException(status_code=409, detail="entity is not user-declared")


def _select_relationship_row(
    con: sqlite3.Connection,
    *,
    normalized: str,
    user_id: str,
    soul_id: str,
) -> sqlite3.Row | None:
    return con.execute(
        """
SELECT id, name, entity_type, normalized, properties
FROM memu_entities
WHERE normalized = ? AND user_id = ? AND soul_id = ?
LIMIT 1
""",
        (normalized, user_id, soul_id),
    ).fetchone()


def _assert_relationship_write_path(
    user_id: str,
    soul_id: str,
    *,
    get_service_from_payload: Callable[[dict[str, Any]], Any],
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    sqlite_ensure_nonempty: Callable[[Path], None],
) -> tuple[Any, Path]:
    scope = {"user_id": user_id, "soul_id": soul_id}
    svc = get_service_from_payload({"user": scope})
    db_path = sqlite_current_path(user_id, soul_id)
    if db_path is None:
        raise HTTPException(status_code=400, detail="soul_id required for sqlite scope resolution")
    sqlite_ensure_nonempty(db_path)
    return svc, db_path


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
        {"name": c.name, "summary": c.summary or ""}
        for c in cats_map.values()
        if c.name and (include_empty or has_category_content({"name": c.name, "summary": c.summary or ""}))
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
            {"name": c.name, "summary": c.summary or ""}
            for c in cats_map.values()
            if c.name and (include_empty or has_category_content({"name": c.name, "summary": c.summary or ""}))
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
SELECT * FROM intentions_life_goals
WHERE soul_id = ? AND user_id = ? AND status = ?
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

    svc, _db_path = _assert_relationship_write_path(
        uid,
        sid,
        get_service_from_payload=get_service_from_payload,
        sqlite_current_path=sqlite_current_path,
        sqlite_ensure_nonempty=sqlite_ensure_nonempty,
    )
    scope = {"user_id": uid, "soul_id": sid}
    entities = svc.database.entity_repo.list_all(where=scope)
    rows = [
        item
        for item in (_relationship_item_from_entity(entity, json_from_db=json_from_db) for entity in entities)
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
    sqlite_connect: Callable[[Path], sqlite3.Connection],
    json_to_db: Callable[[Any], str | None],
    json_from_db: Callable[[Any], Any],
) -> dict[str, Any]:
    sid = str(soul_id or "").strip()
    uid = str(payload.get("user_id") or "").strip()
    name = _normalize_relationship_name(payload.get("name"))
    relationship = _normalize_relationship_text(payload.get("relationship"))
    entity_type = str(payload.get("entity_type") or "person").strip().lower() or "person"
    if entity_type not in {"person", "topic", "place", "project"}:
        raise HTTPException(status_code=400, detail="entity_type must be person/topic/place/project")
    if not sid:
        raise HTTPException(status_code=400, detail="soul_id required")
    if not uid:
        raise HTTPException(status_code=400, detail="user_id required")

    normalized = _slugify_relationship_name(name)
    _validate_relationship_speaker_id(_relationship_speaker_id_from_normalized(normalized))
    scope = {"user_id": uid, "soul_id": sid}
    svc, db_path = _assert_relationship_write_path(
        uid,
        sid,
        get_service_from_payload=get_service_from_payload,
        sqlite_current_path=sqlite_current_path,
        sqlite_ensure_nonempty=sqlite_ensure_nonempty,
    )

    con = sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        row = _select_relationship_row(
            con,
            normalized=normalized,
            user_id=uid,
            soul_id=sid,
        )
        if row is None:
            svc.database.entity_repo.get_or_create(
                name=name,
                entity_type=entity_type,
                user_data=scope,
            )
            row = _select_relationship_row(
                con,
                normalized=normalized,
                user_id=uid,
                soul_id=sid,
            )
            if row is None:
                raise HTTPException(status_code=500, detail="failed to create relationship")
        props = _relationship_properties(row["properties"], json_from_db=json_from_db)
        if str(props.get("origin") or "").strip():
            _assert_user_declared_relationship(props)
        props["origin"] = _RELATIONSHIP_ORIGIN_USER_DECLARED
        props["active"] = True
        if relationship:
            props["relationship"] = relationship
        else:
            props.pop("relationship", None)
        con.execute(
            """
UPDATE memu_entities
SET name = ?, entity_type = ?, properties = ?, updated_at = ?
WHERE id = ?
""",
            (
                name,
                entity_type,
                json_to_db(props),
                datetime.now(UTC).isoformat(),
                str(row["id"]),
            ),
        )
        con.commit()
        return _relationship_item_from_values(
            normalized=normalized,
            name=name,
            entity_type=entity_type,
            properties=props,
        ) or {}
    finally:
        con.close()


async def update_relationship_endpoint(
    *,
    soul_id: str,
    speaker_id: str,
    payload: dict[str, Any],
    get_service_from_payload: Callable[[dict[str, Any]], Any],
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    sqlite_ensure_nonempty: Callable[[Path], None],
    sqlite_connect: Callable[[Path], sqlite3.Connection],
    json_to_db: Callable[[Any], str | None],
    json_from_db: Callable[[Any], Any],
) -> dict[str, Any]:
    sid = str(soul_id or "").strip()
    uid = str(payload.get("user_id") or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="soul_id required")
    if not uid:
        raise HTTPException(status_code=400, detail="user_id required")
    normalized = _validate_relationship_speaker_id(speaker_id)

    name_raw = payload.get("name")
    relationship_raw = payload.get("relationship")
    if name_raw is None and relationship_raw is None:
        raise HTTPException(status_code=400, detail="name or relationship required")
    next_name = _normalize_relationship_name(name_raw) if name_raw is not None else None
    next_relationship = _normalize_relationship_text(relationship_raw) if relationship_raw is not None else None
    _svc, db_path = _assert_relationship_write_path(
        uid,
        sid,
        get_service_from_payload=get_service_from_payload,
        sqlite_current_path=sqlite_current_path,
        sqlite_ensure_nonempty=sqlite_ensure_nonempty,
    )
    con = sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        row = _select_relationship_row(
            con,
            normalized=normalized,
            user_id=uid,
            soul_id=sid,
        )
        if row is None:
            raise HTTPException(status_code=404, detail="relationship not found")

        props = _relationship_properties(row["properties"], json_from_db=json_from_db)
        _assert_user_declared_relationship(props)
        props["origin"] = _RELATIONSHIP_ORIGIN_USER_DECLARED
        props["active"] = True
        if relationship_raw is not None:
            if next_relationship:
                props["relationship"] = next_relationship
            else:
                props.pop("relationship", None)
        final_name = next_name or str(row["name"] or "").strip()
        con.execute(
            """
UPDATE memu_entities
SET name = ?, properties = ?, updated_at = ?
WHERE id = ?
""",
            (
                final_name,
                json_to_db(props),
                datetime.now(UTC).isoformat(),
                str(row["id"]),
            ),
        )
        con.commit()
        return _relationship_item_from_values(
            normalized=normalized,
            name=final_name,
            entity_type=str(row["entity_type"] or "").strip() or "person",
            properties=props,
        ) or {}
    finally:
        con.close()


async def delete_relationship_endpoint(
    *,
    soul_id: str,
    speaker_id: str,
    user_id: str,
    get_service_from_payload: Callable[[dict[str, Any]], Any],
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    sqlite_ensure_nonempty: Callable[[Path], None],
    sqlite_connect: Callable[[Path], sqlite3.Connection],
    json_to_db: Callable[[Any], str | None],
    json_from_db: Callable[[Any], Any],
) -> dict[str, Any]:
    sid = str(soul_id or "").strip()
    uid = str(user_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="soul_id required")
    if not uid:
        raise HTTPException(status_code=400, detail="user_id required")
    normalized = _validate_relationship_speaker_id(speaker_id)

    _svc, db_path = _assert_relationship_write_path(
        uid,
        sid,
        get_service_from_payload=get_service_from_payload,
        sqlite_current_path=sqlite_current_path,
        sqlite_ensure_nonempty=sqlite_ensure_nonempty,
    )
    con = sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        row = _select_relationship_row(
            con,
            normalized=normalized,
            user_id=uid,
            soul_id=sid,
        )
        if row is None:
            return {"ok": True, "speaker_id": _relationship_speaker_id_from_normalized(normalized)}
        props = _relationship_properties(row["properties"], json_from_db=json_from_db)
        _assert_user_declared_relationship(props)
        props["origin"] = _RELATIONSHIP_ORIGIN_USER_DECLARED
        props["active"] = False
        now_iso = datetime.now(UTC).isoformat()
        props["deleted_at"] = now_iso
        con.execute(
            """
UPDATE memu_entities
SET properties = ?, updated_at = ?
WHERE id = ?
""",
            (
                json_to_db(props),
                now_iso,
                str(row["id"]),
            ),
        )
        con.commit()
        return {"ok": True, "speaker_id": _relationship_speaker_id_from_normalized(normalized)}
    finally:
        con.close()


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
    utility_max_tokens: int,
) -> dict[str, Any]:
    sid = str(soul_id or "").strip()
    uid = str(payload.get("user_id") or "").strip()
    suggestion = str(payload.get("suggestion") or "").strip()
    if not sid or not uid or not suggestion:
        raise HTTPException(status_code=400, detail="soul_id, user_id, suggestion required")

    db_path = sqlite_current_path(uid, sid)
    current_narrative = ""
    all_cats_summary = ""
    if db_path is not None and db_path.exists():
        con = sqlite_connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            sqlite_ensure_conversation_state_schema(con)
            soul = _soul_state.read(con)
            current_narrative = str(soul.get("narrative_self") or "").strip()
            all_cats_summary = str(soul.get("all_categories_summary") or "").strip()
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

    svc = get_service_from_payload({"user": {"user_id": uid, "soul_id": sid}})
    raw = await svc.chat(
        user_prompt,
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=utility_max_tokens,
        response_format={"type": "json_object"},
        op="narrative_suggestion",
        step="respond",
    )

    text = str(raw or "").strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    parsed = json.loads(text)
    new_narrative = str(parsed.get("narrative_self") or "").strip()
    companion_memory = str(parsed.get("companion_memory") or "").strip()

    scope = {"user_id": uid, "soul_id": sid}
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
        if current_narrative:
            [old_embedding] = await svc.embed([current_narrative], profile="embedding")
            snapshot_previous_narrative_self(
                svc,
                scope=scope,
                old_text=current_narrative,
                old_embedding=old_embedding,
            )

        sqlite_ensure_nonempty(db_path)
        con = sqlite_connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            sqlite_ensure_conversation_state_schema(con)
            self_model_id = str(uuid.uuid4())
            now_iso = datetime.now(UTC).isoformat()
            con.execute(
                "INSERT INTO narrative_history (id, narrative_self, related_memory_ids, created_at) "
                "VALUES (?, ?, ?, ?)",
                (self_model_id, new_narrative, "[]", now_iso),
            )
            _soul_state.write(con, {"narrative_self": new_narrative, "self_model_id": self_model_id})
            con.commit()
        finally:
            con.close()
        return {"narrative_self": new_narrative}

    if current_narrative:
        return {"narrative_self": current_narrative}
    return {}


async def patch_intention_endpoint(
    *,
    intention_id: str,
    soul_id: str,
    payload: dict[str, Any] | None,
    valid_intention_statuses: set[str],
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
    sqlite_connect: Callable[[Path], sqlite3.Connection],
    sqlite_ensure_conversation_state_schema: Callable[[sqlite3.Connection], None],
    intention_row_to_dict: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    iid = str(intention_id or "").strip()
    if not iid:
        raise HTTPException(status_code=400, detail="intention_id is required")
    sid = str(soul_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="soul_id required")
    db_path = sqlite_current_path(None, sid)
    if db_path is None:
        raise HTTPException(status_code=400, detail="soul_id required for sqlite scope resolution")
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="intention not found")

    body = payload if isinstance(payload, dict) else {}
    updates: dict[str, Any] = {}

    if "status" in body:
        next_status = str(body.get("status") or "").strip()
        if next_status not in valid_intention_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"status must be one of: {sorted(valid_intention_statuses)}",
            )
        updates["status"] = next_status

    if "resolution_note" in body:
        raw_resolution_note = body.get("resolution_note")
        updates["resolution_note"] = None if raw_resolution_note is None else str(raw_resolution_note)

    con = sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        sqlite_ensure_conversation_state_schema(con)
        row = con.execute(
            "SELECT * FROM intentions_life_goals WHERE id = ? LIMIT 1",
            (iid,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="intention not found")

        if updates:
            set_parts: list[str] = []
            params: list[Any] = []
            if "status" in updates:
                set_parts.append("status = ?")
                params.append(updates["status"])
            if "resolution_note" in updates:
                set_parts.append("resolution_note = ?")
                params.append(updates["resolution_note"])
            set_parts.append("updated_at = ?")
            params.append(datetime.now(UTC).isoformat())
            params.append(iid)
            con.execute(
                f"UPDATE intentions_life_goals SET {', '.join(set_parts)} WHERE id = ?",
                params,
            )
            con.commit()
            row = con.execute(
                "SELECT * FROM intentions_life_goals WHERE id = ? LIMIT 1",
                (iid,),
            ).fetchone()

        return intention_row_to_dict(row)
    finally:
        con.close()


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
    find_conversation_state_across_dbs: Callable[[str], tuple[Path | None, dict[str, Any] | None]],
) -> dict[str, Any]:
    cid = str(conversation_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="conversation_id is required")

    db_path: Path | None = None
    state_out: dict[str, Any] | None = None
    sid = str(soul_id or "").strip() or None
    uid = str(user_id or "").strip() or None

    if sid:
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
        db_path, state_out = find_conversation_state_across_dbs(cid)

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

    if "pending_episode_ids" in body:
        updates["pending_episode_ids"] = body.get("pending_episode_ids")

    if "self_model_id" in body:
        updates["self_model_id"] = body.get("self_model_id")

    if "last_retrieval_ids" in body:
        updates["last_retrieval_ids"] = body.get("last_retrieval_ids")

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
