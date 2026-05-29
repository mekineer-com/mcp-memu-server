from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def sqlite_current_path(
    *,
    user_id: str | None,
    soul_id: str | None,
    storage_status: Mapping[str, Any],
    config: Mapping[str, Any],
    sqlite_dsn_for_scope: Callable[[dict[str, Any], str, dict[str, Any] | None], str],
    sqlite_file_from_dsn: Callable[[str], Path | None],
) -> Path | None:
    base_dsn = str(storage_status.get("dsn") or "")
    sid = str(soul_id or "").strip()
    if not sid:
        return None
    scope = {"soul_id": sid}
    dsn = sqlite_dsn_for_scope(dict(config), base_dsn, scope)
    f = sqlite_file_from_dsn(dsn)
    return f.expanduser().resolve() if f is not None else None


def sqlite_build_scope_where(
    cols: list[str],
    user_id: str | None,
    soul_id: str | None,
    conversation_id: str | None,
) -> tuple[str, list[Any]]:
    where = []
    params: list[Any] = []
    if user_id and "user_id" in cols:
        where.append("user_id = ?")
        params.append(user_id)
    if soul_id and "soul_id" in cols:
        where.append("soul_id = ?")
        params.append(soul_id)
    if conversation_id and "conversation_id" in cols:
        where.append("conversation_id = ?")
        params.append(conversation_id)
    if not where:
        return "", params
    return " WHERE " + " AND ".join(where), params


def sqlite_file_info(p: Path) -> dict[str, Any]:
    try:
        st = p.stat()
        return {
            "exists": p.exists(),
            "path": str(p),
            "size": int(st.st_size),
            "mtime": float(st.st_mtime),
        }
    except OSError as e:
        return {"exists": p.exists(), "path": str(p), "error": f"{type(e).__name__}: {e}"}


def intention_row_to_dict(
    row: Any,
    *,
    normalize_text_list: Callable[[Any], list[str]],
) -> dict[str, Any]:
    item = {k: row[k] for k in row.keys()}
    item["related_memory_ids"] = normalize_text_list(item.get("related_memory_ids"))
    return item


def write_conversation_state(
    conversation_id: str,
    *,
    soul_id: str | None,
    user_id: str | None,
    updates: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    storage_status: Mapping[str, Any],
    sqlite_dir_from_cfg: Callable[[dict[str, Any], str | None], Path],
    write_conversation_state_impl: Callable[..., tuple[dict[str, Any], Path]],
    sqlite_current_path: Callable[[str | None, str | None], Path | None],
) -> tuple[dict[str, Any], Path]:
    sqlite_dir = sqlite_dir_from_cfg(dict(config), str(storage_status.get("dsn") or ""))
    return write_conversation_state_impl(
        conversation_id,
        sqlite_current_path=sqlite_current_path,
        sqlite_dir=sqlite_dir,
        soul_id=soul_id,
        user_id=user_id,
        updates=updates,
    )
