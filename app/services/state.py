from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlite3
from fastapi import HTTPException

from app.db import json_from_db, json_to_db, sqlite_table_columns


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
    normalize_intention_stack: Callable[[Any], dict[str, Any]]
    normalize_memory_cache: Callable[[Any], list[str]]


def conversation_state_from_row(
    row: sqlite3.Row | None,
    *,
    normalize_text_list: Callable[[Any], list[str]],
    normalize_intention_stack: Callable[[Any], dict[str, Any]],
    normalize_memory_cache: Callable[[Any], list[str]],
) -> dict[str, Any] | None:
    if row is None:
        return None
    digest_raw = row['digest_cursor'] if 'digest_cursor' in row.keys() else 0
    try:
        digest_cursor = int(digest_raw) if digest_raw is not None else 0
    except Exception:
        digest_cursor = 0
    prior_context = row['prior_context'] if 'prior_context' in row.keys() else None
    return {
        'conversation_id': row['conversation_id'],
        'soul_id': row['soul_id'] if 'soul_id' in row.keys() else None,
        'user_id': row['user_id'] if 'user_id' in row.keys() else None,
        'digest_cursor': max(0, digest_cursor),
        'prior_context': None if prior_context is None else str(prior_context),
        'active_intentions': normalize_intention_stack(
            json_from_db(row['active_intentions'] if 'active_intentions' in row.keys() else None)
        ),
        'memory_cache': normalize_memory_cache(
            json_from_db(row['memory_cache'] if 'memory_cache' in row.keys() else None)
        ),
        'pending_diary_memory_ids': normalize_text_list(
            row['pending_diary_memory_ids'] if 'pending_diary_memory_ids' in row.keys() else None
        ),
        'self_model_id': row['self_model_id'] if 'self_model_id' in row.keys() else None,
        'last_retrieval_ids': json_from_db(row['last_retrieval_ids'] if 'last_retrieval_ids' in row.keys() else None),
        'last_memorize_at': row['last_memorize_at'] if 'last_memorize_at' in row.keys() else None,
        'updated_at': row['updated_at'] if 'updated_at' in row.keys() else None,
    }


def conversation_state_row(con: sqlite3.Connection, conversation_id: str) -> sqlite3.Row | None:
    cols = set(sqlite_table_columns(con, 'memu_conversation_state'))
    select_cols = [
        'conversation_id',
        *( ['soul_id'] if 'soul_id' in cols else [] ),
        'user_id',
        'digest_cursor',
        *( ['prior_context'] if 'prior_context' in cols else [] ),
        'active_intentions',
        *( ['memory_cache'] if 'memory_cache' in cols else [] ),
        *( ['pending_diary_memory_ids'] if 'pending_diary_memory_ids' in cols else [] ),
        *( ['self_model_id'] if 'self_model_id' in cols else [] ),
        'last_retrieval_ids',
        'last_memorize_at',
        'updated_at',
    ]
    return con.execute(
        f"SELECT {', '.join(select_cols)} FROM memu_conversation_state WHERE conversation_id = ? LIMIT 1",
        (conversation_id,),
    ).fetchone()


def conversation_state_empty(
    conversation_id: str,
    soul_id: str | None = None,
    user_id: str | None = None,
    normalize_intention_stack: Callable[[Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if normalize_intention_stack is None:
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
        default_stack = normalize_intention_stack(None)
    return {
        'conversation_id': conversation_id,
        'soul_id': soul_id,
        'user_id': user_id,
        'digest_cursor': 0,
        'prior_context': None,
        'active_intentions': default_stack,
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
                    normalize_intention_stack=deps.normalize_intention_stack,
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
                normalize_intention_stack=deps.normalize_intention_stack,
                normalize_memory_cache=deps.normalize_memory_cache,
            )
        merged = dict(
            existing_state or conversation_state_empty(
                cid, scoped_soul, scoped_user, normalize_intention_stack=deps.normalize_intention_stack
            )
        )
        merged['conversation_id'] = cid
        if scoped_soul is not None:
            merged['soul_id'] = scoped_soul
        if scoped_user is not None:
            merged['user_id'] = scoped_user

        raw_updates = dict(updates or {})
        append_pending_diary_memory_ids = raw_updates.pop('append_pending_diary_memory_ids', None)

        for key, value in raw_updates.items():
            if key in {
                'digest_cursor',
                'prior_context',
                'active_intentions',
                'memory_cache',
                'pending_diary_memory_ids',
                'self_model_id',
                'last_retrieval_ids',
                'last_memorize_at',
            }:
                merged[key] = value

        if append_pending_diary_memory_ids is not None:
            merged['pending_diary_memory_ids'] = deps.merge_unique_text_lists(
                merged.get('pending_diary_memory_ids'),
                append_pending_diary_memory_ids,
            )

        try:
            merged['digest_cursor'] = max(0, int(merged.get('digest_cursor') or 0))
        except Exception as exc:
            raise HTTPException(status_code=400, detail='digest_cursor must be an integer') from exc

        raw_last = merged.get('last_memorize_at')
        merged['last_memorize_at'] = None if raw_last is None else (str(raw_last).strip() or None)
        raw_prior_context = merged.get('prior_context')
        merged['prior_context'] = None if raw_prior_context is None else str(raw_prior_context)
        merged['active_intentions'] = deps.normalize_intention_stack(merged.get('active_intentions'))
        merged['memory_cache'] = deps.normalize_memory_cache(merged.get('memory_cache'))
        merged['pending_diary_memory_ids'] = deps.normalize_text_list(merged.get('pending_diary_memory_ids'))
        raw_self_model_id = merged.get('self_model_id')
        merged['self_model_id'] = None if raw_self_model_id is None else (str(raw_self_model_id).strip() or None)
        merged['updated_at'] = datetime.now(UTC).isoformat()

        con.execute(
            """
INSERT INTO memu_conversation_state (
    conversation_id,
    soul_id,
    user_id,
    digest_cursor,
    prior_context,
    active_intentions,
    memory_cache,
    pending_diary_memory_ids,
    self_model_id,
    last_retrieval_ids,
    last_memorize_at,
    updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(conversation_id) DO UPDATE SET
    soul_id = excluded.soul_id,
    user_id = excluded.user_id,
    digest_cursor = excluded.digest_cursor,
    prior_context = excluded.prior_context,
    active_intentions = excluded.active_intentions,
    memory_cache = excluded.memory_cache,
    pending_diary_memory_ids = excluded.pending_diary_memory_ids,
    self_model_id = excluded.self_model_id,
    last_retrieval_ids = excluded.last_retrieval_ids,
    last_memorize_at = excluded.last_memorize_at,
    updated_at = excluded.updated_at
""",
            (
                merged['conversation_id'],
                merged.get('soul_id'),
                merged.get('user_id'),
                int(merged.get('digest_cursor') or 0),
                merged.get('prior_context'),
                json_to_db(merged.get('active_intentions')),
                json_to_db(merged.get('memory_cache') or []),
                json_to_db(merged.get('pending_diary_memory_ids') or []),
                merged.get('self_model_id'),
                json_to_db(merged.get('last_retrieval_ids')),
                merged.get('last_memorize_at'),
                merged.get('updated_at'),
            ),
        )
        con.commit()
        state_out = conversation_state_from_row(
            conversation_state_row(con, cid),
            normalize_text_list=deps.normalize_text_list,
            normalize_intention_stack=deps.normalize_intention_stack,
            normalize_memory_cache=deps.normalize_memory_cache,
        )
        return state_out or conversation_state_empty(
            cid, scoped_soul, scoped_user, normalize_intention_stack=deps.normalize_intention_stack
        ), db_path
    finally:
        con.close()
