from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlite3
from fastapi import HTTPException

from app.db import json_from_db, json_to_db


@dataclass(frozen=True)
class StateDeps:
    sqlite_current_path: Callable[[str | None, str | None], Path | None]
    sqlite_connect: Callable[[Path], sqlite3.Connection]
    sqlite_ensure_nonempty: Callable[[Path], None]
    sqlite_ensure_conversation_state_schema: Callable[[sqlite3.Connection], None]
    sqlite_dir_from_cfg: Callable[[dict[str, Any], str | None], Path]
    config: dict[str, Any]
    storage_status: Mapping[str, Any]
    normalize_text_list: Callable[[Any], list[str]]
    merge_unique_text_lists: Callable[[Any, Any], list[str]]
    normalize_intentions_stack: Callable[[Any], dict[str, Any]]
    normalize_memory_cache: Callable[[Any], list[str]]


def conversation_state_from_row(
    row: sqlite3.Row | None,
    *,
    normalize_text_list: Callable[[Any], list[str]],
    normalize_intentions_stack: Callable[[Any], dict[str, Any]],
    normalize_memory_cache: Callable[[Any], list[str]],
) -> dict[str, Any] | None:
    if row is None:
        return None
    try:
        digest_cursor = int(row['digest_cursor'] or 0)
    except Exception:
        digest_cursor = 0
    prior_context = row['prior_context']
    return {
        'conversation_id': row['conversation_id'],
        'soul_id': row['soul_id'],
        'user_id': row['user_id'],
        'digest_cursor': max(0, digest_cursor),
        'prior_context': None if prior_context is None else str(prior_context),
        'intentions_active': normalize_intentions_stack(json_from_db(row['intentions_active'])),
        'memory_cache': normalize_memory_cache(json_from_db(row['memory_cache'])),
        'pending_diary_memory_ids': normalize_text_list(row['pending_diary_memory_ids']),
        'self_model_id': row['self_model_id'],
        'last_retrieval_ids': json_from_db(row['last_retrieval_ids']),
        'last_memorize_at': row['last_memorize_at'],
        'updated_at': row['updated_at'],
    }


def conversation_state_row(con: sqlite3.Connection, conversation_id: str) -> sqlite3.Row | None:
    return con.execute(
        "SELECT conversation_id, soul_id, user_id, digest_cursor, prior_context, "
        "intentions_active, memory_cache, pending_diary_memory_ids, self_model_id, "
        "last_retrieval_ids, last_memorize_at, updated_at "
        "FROM memu_conversation_state WHERE conversation_id = ? LIMIT 1",
        (conversation_id,),
    ).fetchone()


def conversation_state_empty(
    conversation_id: str,
    soul_id: str | None = None,
    user_id: str | None = None,
    normalize_intentions_stack: Callable[[Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if normalize_intentions_stack is None:
        default_stack: dict[str, Any] = {
            'version': 1,
            'decay_per_turn': 0.1,
            'relax_priority': 5.0,
            'items': [
                {
                    'id': 'relax',
                    'text': 'Relax',
                    'priority': 5.0,
                    'ephemeral': False,
                    'kind': 'relax',
                    'status': 'active',
                    'active': True,
                }
            ],
        }
    else:
        default_stack = normalize_intentions_stack(None)
    return {
        'conversation_id': conversation_id,
        'soul_id': soul_id,
        'user_id': user_id,
        'digest_cursor': 0,
        'prior_context': None,
        'intentions_active': default_stack,
        'memory_cache': [],
        'pending_diary_memory_ids': [],
        'self_model_id': None,
        'last_retrieval_ids': None,
        'last_memorize_at': None,
        'updated_at': None,
    }


def sqlite_agent_db_paths(*, deps: StateDeps) -> list[Path]:
    sqlite_dir = deps.sqlite_dir_from_cfg(deps.config, str(deps.storage_status.get('dsn') or ''))
    if not sqlite_dir.exists():
        return []
    return sorted([p.resolve() for p in sqlite_dir.glob('*.db') if p.is_file()])


def find_conversation_state_across_dbs(
    conversation_id: str,
    *,
    deps: StateDeps,
) -> tuple[Path | None, dict[str, Any] | None]:
    for db_path in sqlite_agent_db_paths(deps=deps):
        con = deps.sqlite_connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            deps.sqlite_ensure_conversation_state_schema(con)
            row = conversation_state_row(con, conversation_id)
            if row is not None:
                return db_path, conversation_state_from_row(
                    row,
                    normalize_text_list=deps.normalize_text_list,
                    normalize_intentions_stack=deps.normalize_intentions_stack,
                    normalize_memory_cache=deps.normalize_memory_cache,
                )
        except Exception:
            continue
        finally:
            con.close()
    return None, None


def write_conversation_state(
    conversation_id: str,
    *,
    deps: StateDeps,
    soul_id: str | None = None,
    user_id: str | None = None,
    updates: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    cid = str(conversation_id or '').strip()
    if not cid:
        raise HTTPException(status_code=400, detail='conversation_id is required')

    scoped_soul = str(soul_id or '').strip() or None
    scoped_user = str(user_id or '').strip() or None

    db_path: Path | None = deps.sqlite_current_path(scoped_user, scoped_soul) if scoped_soul else None
    existing_state: dict[str, Any] | None = None
    if db_path is None:
        db_path, existing_state = find_conversation_state_across_dbs(cid, deps=deps)
        if db_path is None:
            raise HTTPException(status_code=400, detail='soul_id is required when creating new conversation state')

    deps.sqlite_ensure_nonempty(db_path)
    con = deps.sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        deps.sqlite_ensure_conversation_state_schema(con)

        if existing_state is None:
            existing_state = conversation_state_from_row(
                conversation_state_row(con, cid),
                normalize_text_list=deps.normalize_text_list,
                normalize_intentions_stack=deps.normalize_intentions_stack,
                normalize_memory_cache=deps.normalize_memory_cache,
            )

        if existing_state is None:
            if scoped_soul is None:
                raise HTTPException(status_code=400, detail='soul_id is required when creating new conversation state')
            seed = conversation_state_empty(
                cid,
                scoped_soul,
                scoped_user,
                normalize_intentions_stack=deps.normalize_intentions_stack,
            )
            seed['updated_at'] = datetime.now(UTC).isoformat()
            con.execute(
                """
INSERT INTO memu_conversation_state (
    conversation_id,
    soul_id,
    user_id,
    digest_cursor,
    prior_context,
    intentions_active,
    memory_cache,
    pending_diary_memory_ids,
    self_model_id,
    last_retrieval_ids,
    last_memorize_at,
    updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
                (
                    seed['conversation_id'],
                    seed.get('soul_id'),
                    seed.get('user_id'),
                    int(seed.get('digest_cursor') or 0),
                    seed.get('prior_context'),
                    json_to_db(seed.get('intentions_active')),
                    json_to_db(seed.get('memory_cache') or []),
                    json_to_db(seed.get('pending_diary_memory_ids') or []),
                    seed.get('self_model_id'),
                    json_to_db(seed.get('last_retrieval_ids')),
                    seed.get('last_memorize_at'),
                    seed.get('updated_at'),
                ),
            )
            con.commit()
            existing_state = seed

        raw_updates = dict(updates) if updates else {}
        append_pending_diary_memory_ids = raw_updates.pop('append_pending_diary_memory_ids', None)
        field_updates: dict[str, Any] = {}

        for key, value in raw_updates.items():
            if key in {
                'digest_cursor',
                'prior_context',
                'intentions_active',
                'memory_cache',
                'pending_diary_memory_ids',
                'self_model_id',
                'last_retrieval_ids',
                'last_memorize_at',
            }:
                field_updates[key] = value

        if append_pending_diary_memory_ids is not None:
            base_pending = field_updates.get('pending_diary_memory_ids', existing_state.get('pending_diary_memory_ids'))
            field_updates['pending_diary_memory_ids'] = deps.merge_unique_text_lists(
                base_pending,
                append_pending_diary_memory_ids,
            )

        if 'digest_cursor' in field_updates:
            try:
                field_updates['digest_cursor'] = max(0, int(field_updates.get('digest_cursor') or 0))
            except Exception as exc:
                raise HTTPException(status_code=400, detail='digest_cursor must be an integer') from exc
        if 'last_memorize_at' in field_updates:
            raw_last = field_updates.get('last_memorize_at')
            field_updates['last_memorize_at'] = None if raw_last is None else (str(raw_last).strip() or None)
        if 'prior_context' in field_updates:
            raw_prior_context = field_updates.get('prior_context')
            field_updates['prior_context'] = None if raw_prior_context is None else str(raw_prior_context)
        if 'intentions_active' in field_updates:
            field_updates['intentions_active'] = deps.normalize_intentions_stack(field_updates.get('intentions_active'))
        if 'memory_cache' in field_updates:
            field_updates['memory_cache'] = deps.normalize_memory_cache(field_updates.get('memory_cache'))
        if 'pending_diary_memory_ids' in field_updates:
            field_updates['pending_diary_memory_ids'] = deps.normalize_text_list(field_updates.get('pending_diary_memory_ids'))
        if 'self_model_id' in field_updates:
            raw_self_model_id = field_updates.get('self_model_id')
            field_updates['self_model_id'] = None if raw_self_model_id is None else (str(raw_self_model_id).strip() or None)

        if scoped_soul is not None:
            field_updates['soul_id'] = scoped_soul
        if scoped_user is not None:
            field_updates['user_id'] = scoped_user

        if field_updates:
            field_updates['updated_at'] = datetime.now(UTC).isoformat()
            assignments: list[str] = []
            params: list[Any] = []
            for key, value in field_updates.items():
                assignments.append(f"{key} = ?")
                if key in {'intentions_active', 'memory_cache', 'pending_diary_memory_ids', 'last_retrieval_ids'}:
                    params.append(json_to_db(value))
                elif key == 'digest_cursor':
                    params.append(int(value or 0))
                else:
                    params.append(value)
            params.append(cid)
            con.execute(
                f"UPDATE memu_conversation_state SET {', '.join(assignments)} WHERE conversation_id = ?",
                tuple(params),
            )
            con.commit()

        state_out = conversation_state_from_row(
            conversation_state_row(con, cid),
            normalize_text_list=deps.normalize_text_list,
            normalize_intentions_stack=deps.normalize_intentions_stack,
            normalize_memory_cache=deps.normalize_memory_cache,
        )
        return state_out or conversation_state_empty(
            cid, scoped_soul, scoped_user, normalize_intentions_stack=deps.normalize_intentions_stack
        ), db_path
    finally:
        con.close()
