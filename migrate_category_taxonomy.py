#!/usr/bin/env python3
"""Offline, resumable migration from legacy categories to reviewed dossiers."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from defusedxml import ElementTree
from pydantic import BaseModel

from app.config import (
    blob_config_from_cfg,
    default_llm_profiles_from_server_config,
    sqlite_dir_from_cfg,
)
from app.services.payload import strip_markdown_code_fence


MIGRATION_VERSION = 2
MAX_PROMPT_TOKENS = 40_000
DOSSIER_KINDS = {"lore", "topic", "goal"}
OFFLINE_RECOVERY = (
    "Keep every database writer stopped until apply validates successfully; "
    "after an interruption, retry apply or restore the recorded backup manually."
)
SCOPED_TABLES = (
    "memory_items",
    "categories",
    "category_items",
    "dossier_candidates",
    "resources",
    "entities",
    "triples",
)

DISCOVERY_SYSTEM_PROMPT = """# Rediscover the shape of a life

These memories come from a life that already has history. Help them find the
durable life domains they naturally gather around. This is a discovery pass:
you are proposing dossier titles, not writing dossiers or deciding which
dossiers will ultimately be kept.

For every memory shown, choose zero to three broad but personal category titles.
A title should name a lasting home for memories, not merely repeat one event.
Dossiers can hold lived identity, relationships, rituals, places, interests,
knowledge, hopes, commitments, and wonderfully specific corners of a life.

Prefer a supplied seed dossier title when it genuinely fits. Copy that title
exactly. If no seed fits, propose a concise new title that preserves what is
distinctive about the memory. A memory may belong in more than one dossier, but
do not force it into a category and do not use more than three.

Stay grounded in the memory text. Preserve who experienced, said, wanted, or
did something. Do not turn a memory about someone else into a fact about the
soul or human.

# Exact output

Return one XML document and nothing else. Include every supplied `[M#]` exactly
once, in the same order. Copy each reference exactly. Use zero to three
`<category>` elements per memory; an empty `<categories></categories>` means
the memory remains uncategorized.

<taxonomy_memories>
  <memory ref="[M1]">
    <categories>
      <category>exact seed title or proposed title</category>
    </categories>
  </memory>
  <memory ref="[M2]">
    <categories></categories>
  </memory>
</taxonomy_memories>

Do not add descriptions, kinds, explanations, Markdown fences, comments, or
any memory reference that was not supplied. Escape special characters so the
result remains valid XML."""

DISCOVERY_USER_PROMPT = """# Seed dossier titles

{seed_dossier_titles_or_none}

# Historical memories

{memory_records}

Return only the exact XML."""

ASSIGNMENT_SYSTEM_PROMPT = """# Give each memory its truest home

The dossier set has been reviewed by your human and is now fixed. For every
memory shown, decide which of its supplied nearby dossiers genuinely belong to
it. This is a membership pass, not a dossier-creation pass.

Choose zero to three dossier titles per memory. A memory may belong in more
than one dossier when each relationship is specific and useful. Do not force a
match merely because a dossier is nearby in embedding space. Leaving a memory
uncategorized is honest and allowed.

Use only titles supplied for that memory and copy them exactly. Soul and human
identity may be present for orientation, but their anchor dossiers are not
filing targets. Preserve who experienced, said, wanted, or did something.

# Exact output

Return one XML document and nothing else. Include every supplied `[M#]` exactly
once, in the same order. Copy each reference exactly. Use zero to three
`<category>` elements per memory; an empty `<categories></categories>` means
the memory remains uncategorized.

<taxonomy_memories>
  <memory ref="[M1]">
    <categories>
      <category>exact supplied dossier title</category>
    </categories>
  </memory>
  <memory ref="[M2]">
    <categories></categories>
  </memory>
</taxonomy_memories>

Do not propose new titles, alter supplied titles, add explanations, use anchor
dossiers, or include Markdown fences or comments. Do not add any memory
reference that was not supplied. Escape special characters so the result
remains valid XML."""

ASSIGNMENT_USER_PROMPT = """{identity_orientation_or_none}

# Historical memories and nearby approved dossiers

{memory_records_with_dossier_choices}

Return only the exact XML."""


class MigrationError(RuntimeError):
    pass


class ScopeModel(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise MigrationError(f"cannot read JSON {path}: {exc}") from exc


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _database_identity(path: Path, *, include_hash: bool = True) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise MigrationError(f"database does not exist: {resolved}")
    stat = resolved.stat()
    identity: dict[str, Any] = {
        "path": str(resolved),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        identity["sha256"] = _sha256_file(resolved)
    return identity


def _open_read_only(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{path.expanduser().resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _scope_rows(conn: sqlite3.Connection, tables: set[str]) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    all_scopes: set[tuple[str, str]] = set()
    memory_scopes: set[tuple[str, str]] = set()
    for table in SCOPED_TABLES:
        if table not in tables or not {"user_id", "soul_id"} <= _column_names(conn, table):
            continue
        rows = conn.execute(
            f"SELECT DISTINCT user_id, soul_id FROM {table} "
            "WHERE trim(COALESCE(user_id, '')) != '' AND trim(COALESCE(soul_id, '')) != ''"
        ).fetchall()
        scopes = {(str(row[0]), str(row[1])) for row in rows}
        all_scopes.update(scopes)
        if table == "memory_items":
            memory_scopes = scopes
    return memory_scopes, all_scopes


def _require_scope(report: Mapping[str, Any]) -> dict[str, str]:
    scopes = report.get("memory_scopes")
    all_scopes = report.get("all_scopes")
    if not isinstance(scopes, list) or len(scopes) != 1:
        raise MigrationError(f"migration requires exactly one complete memory scope; found {scopes}")
    if not isinstance(all_scopes, list) or all_scopes != scopes:
        raise MigrationError(f"database contains rows outside the memory scope: {all_scopes}")
    scope = scopes[0]
    return {"user_id": str(scope["user_id"]), "soul_id": str(scope["soul_id"])}


def _active_memory_clause(alias: str = "m") -> str:
    return f"""
        (COALESCE(trim({alias}.merged_into), '') = '')
        AND NOT EXISTS (
            SELECT 1 FROM triples t
            WHERE t.subject_id = {alias}.id
              AND t.subject_kind = 'memory'
              AND t.predicate = 'evolved_into'
              AND t.valid_to IS NULL
              AND t.user_id = {alias}.user_id
              AND t.soul_id = {alias}.soul_id
        )
    """


def _active_memory_identity(
    conn: sqlite3.Connection,
    active_where: str,
    params: Sequence[Any],
) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT m.id, m.memory_type, m.summary, m.happened_at, m.created_at, m.embedding "
        f"FROM memory_items m{active_where} ORDER BY m.id",
        params,
    ).fetchall()
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(list(row[:5]), ensure_ascii=False, default=str).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes(row[5] or b""))
        digest.update(b"\0")
    return {"count": len(rows), "sha256": digest.hexdigest()}


def scan_database(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise MigrationError(f"database does not exist: {resolved}")
    with closing(_open_read_only(resolved)) as conn:
        tables = _table_names(conn)
        required = {"memory_items", "categories", "category_items", "dossier_candidates", "triples"}
        missing = sorted(required - tables)
        if missing:
            raise MigrationError(f"database lacks current memU schema tables: {missing}")
        memory_scopes, all_scopes = _scope_rows(conn, tables)
        ordered_memory_scopes = [
            {"user_id": user_id, "soul_id": soul_id}
            for user_id, soul_id in sorted(memory_scopes)
        ]
        ordered_all_scopes = [
            {"user_id": user_id, "soul_id": soul_id}
            for user_id, soul_id in sorted(all_scopes)
        ]
        scope = ordered_memory_scopes[0] if len(ordered_memory_scopes) == 1 else None
        params = () if scope is None else (scope["user_id"], scope["soul_id"])
        where = "" if scope is None else " WHERE user_id = ? AND soul_id = ?"
        active_where = (
            f" WHERE m.user_id = ? AND m.soul_id = ? AND {_active_memory_clause()}"
            if scope is not None
            else f" WHERE {_active_memory_clause()}"
        )
        active_memory_identity = _active_memory_identity(conn, active_where, params)
        active_count = int(active_memory_identity["count"])
        missing_refs = int(
            conn.execute(
                f"SELECT COUNT(*) FROM memory_items m{active_where} AND m.memory_ref IS NULL",
                params,
            ).fetchone()[0]
        )
        duplicate_refs = int(
            conn.execute(
                "SELECT COUNT(*) FROM ("
                f"SELECT m.memory_ref FROM memory_items m{active_where} AND m.memory_ref IS NOT NULL "
                "GROUP BY m.memory_ref HAVING COUNT(*) > 1)",
                params,
            ).fetchone()[0]
        )
        categories = [
            dict(row)
            for row in conn.execute(
                "SELECT id, name, kind, anchor_role, last_evidence_at, last_revised_at, "
                "CASE WHEN approved_description IS description AND approved_summary IS summary THEN 0 ELSE 1 END "
                f"AS pending_approval FROM categories{where} ORDER BY lower(name), id",
                params,
            ).fetchall()
        ]
        links = int(conn.execute(f"SELECT COUNT(*) FROM category_items{where}", params).fetchone()[0])
        candidates = int(conn.execute(f"SELECT COUNT(*) FROM dossier_candidates{where}", params).fetchone()[0])
        vector_storage: dict[str, dict[str, int]] = {}
        for table in ("memory_items", "categories", "resources"):
            if table not in tables:
                continue
            rows = conn.execute(
                f"SELECT typeof(embedding), COUNT(*) FROM {table} "
                "WHERE embedding IS NOT NULL GROUP BY typeof(embedding)"
            ).fetchall()
            vector_storage[table] = {str(row[0]): int(row[1]) for row in rows}
        journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0])
        schema_version = int(conn.execute("PRAGMA user_version").fetchone()[0])

    return {
        "database": str(resolved),
        "identity": _database_identity(resolved),
        "schema_version": schema_version,
        "memory_scopes": ordered_memory_scopes,
        "all_scopes": ordered_all_scopes,
        "active_memory_identity": active_memory_identity,
        "active_memories": active_count,
        "missing_memory_refs": missing_refs,
        "duplicate_memory_refs": duplicate_refs,
        "categories": categories,
        "category_links": links,
        "dossier_candidates": candidates,
        "anchors": sum(1 for row in categories if row["anchor_role"] is not None),
        "pending_category_approvals": sum(int(row["pending_approval"]) for row in categories),
        "vector_storage": vector_storage,
        "wal": {
            "journal_mode": journal_mode,
            "wal_file_present": Path(f"{resolved}-wal").exists(),
            "shm_file_present": Path(f"{resolved}-shm").exists(),
        },
    }


def _load_active_memories(path: Path, scope: Mapping[str, str]) -> list[dict[str, Any]]:
    with closing(_open_read_only(path)) as conn:
        rows = conn.execute(
            "SELECT m.id, m.memory_ref, m.memory_type, m.summary, m.happened_at, "
            "m.created_at, m.embedding FROM memory_items m "
            f"WHERE m.user_id = ? AND m.soul_id = ? AND {_active_memory_clause()} "
            "ORDER BY COALESCE(m.happened_at, m.created_at), m.created_at, m.id",
            (scope["user_id"], scope["soul_id"]),
        ).fetchall()
    return [dict(row) for row in rows]


def _memory_ref(row: Mapping[str, Any]) -> str:
    value = row.get("memory_ref")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MigrationError(f"memory {row.get('id')} lacks a valid [M#] reference")
    return f"[M{value}]"


def _render_memory(row: Mapping[str, Any]) -> str:
    from memu.app.dossier_revision import render_memory_record

    memory_ref = _memory_ref(row)
    return render_memory_record(
        int(memory_ref[2:-1]),
        str(row.get("memory_type") or "memory"),
        row.get("happened_at") or row.get("created_at"),
        str(row.get("summary") or ""),
        include_relative=False,
    )


def _estimate_tokens(text: str) -> int:
    from memu.app.dossier_revision import estimate_prompt_tokens

    return estimate_prompt_tokens(text)


def _prompt_batches(
    rows: Sequence[dict[str, Any]],
    *,
    system_prompt: str,
    render_user: Callable[[Sequence[dict[str, Any]]], str],
) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in rows:
        proposed = [*current, row]
        if _estimate_tokens(system_prompt + "\n" + render_user(proposed)) <= MAX_PROMPT_TOKENS:
            current = proposed
            continue
        if not current:
            raise MigrationError(f"single memory exceeds the {MAX_PROMPT_TOKENS}-token prompt limit")
        batches.append(current)
        current = [row]
        if _estimate_tokens(system_prompt + "\n" + render_user(current)) > MAX_PROMPT_TOKENS:
            raise MigrationError(f"single memory exceeds the {MAX_PROMPT_TOKENS}-token prompt limit")
    if current:
        batches.append(current)
    return batches


def _parse_taxonomy_output(
    raw: str,
    expected_refs: Sequence[str],
    *,
    allowed_titles: Mapping[str, set[str]] | None = None,
) -> dict[str, list[str]]:
    text = strip_markdown_code_fence(str(raw or "").strip())
    if (
        not text.startswith("<taxonomy_memories>")
        or not text.endswith("</taxonomy_memories>")
        or "<!--" in text
        or "<?" in text
    ):
        raise MigrationError("expected exact taxonomy_memories XML")
    try:
        root = ElementTree.fromstring(text)
    except Exception as exc:
        raise MigrationError("invalid taxonomy_memories XML") from exc
    if root.tag != "taxonomy_memories" or root.attrib or (root.text or "").strip():
        raise MigrationError("expected exact taxonomy_memories root")

    parsed: dict[str, list[str]] = {}
    order: list[str] = []
    for memory in list(root):
        if memory.tag != "memory" or set(memory.attrib) != {"ref"} or (memory.text or "").strip():
            raise MigrationError("invalid taxonomy memory element")
        ref = memory.attrib["ref"]
        children = list(memory)
        if len(children) != 1 or children[0].tag != "categories" or children[0].attrib:
            raise MigrationError(f"memory {ref} must contain exactly one categories element")
        categories = children[0]
        if (categories.text or "").strip():
            raise MigrationError(f"memory {ref} categories contains stray text")
        titles: list[str] = []
        for category in list(categories):
            if category.tag != "category" or category.attrib or list(category):
                raise MigrationError(f"memory {ref} contains an invalid category element")
            title = str(category.text or "").strip()
            if not title:
                raise MigrationError(f"memory {ref} contains an empty category title")
            titles.append(title)
        if len(titles) > 3 or len({title.casefold() for title in titles}) != len(titles):
            raise MigrationError(f"memory {ref} must contain zero to three unique category titles")
        if ref in parsed:
            raise MigrationError(f"duplicate memory reference in output: {ref}")
        if allowed_titles is not None and any(title not in allowed_titles.get(ref, set()) for title in titles):
            raise MigrationError(f"memory {ref} selected an unsupplied dossier title")
        parsed[ref] = titles
        order.append(ref)
    if order != list(expected_refs):
        raise MigrationError(f"taxonomy output references differ from input: expected {list(expected_refs)}, got {order}")
    return parsed


def _prompt_hash(system_prompt: str, user_prompt: str, model_identity: Mapping[str, Any]) -> str:
    return _sha256_json(
        {
            "contract_version": MIGRATION_VERSION,
            "model": model_identity,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
    )


def _migration_cluster_id(candidate_ids: Sequence[str]) -> str:
    return "migration_" + hashlib.sha256(
        "\0".join(sorted(candidate_ids)).encode("utf-8")
    ).hexdigest()[:16]


def _validate_dossiers(path: Path, scope: Mapping[str, str], *, require_approved: bool) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or payload.get("scope") != dict(scope):
        raise MigrationError(f"dossier artifact scope does not match target: {path}")
    rows = payload.get("dossiers")
    if not isinstance(rows, list):
        raise MigrationError(f"dossier artifact must contain a dossiers list: {path}")
    normalized: set[str] = set()
    dossiers: list[dict[str, Any]] = []
    anchor_titles = {_normalize_title(scope["user_id"]), _normalize_title(scope["soul_id"])}
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise MigrationError(f"dossier {index} must be an object")
        title = str(raw.get("title") or "").strip()
        description = str(raw.get("description") or "").strip()
        kind = str(raw.get("kind") or "").strip()
        origin = str(raw.get("origin") or "").strip()
        approved = raw.get("approved") is True
        key = _normalize_title(title)
        if not title or not description or kind not in DOSSIER_KINDS or origin not in {"seed", "discovered"}:
            raise MigrationError(f"dossier {index} has invalid title, description, kind, or origin")
        if key in anchor_titles:
            raise MigrationError(f"dossier title collides with an anchor: {title}")
        if key in normalized:
            raise MigrationError(f"duplicate normalized dossier title: {title}")
        if require_approved and not approved:
            raise MigrationError(f"dossier is not approved: {title}")
        normalized.add(key)
        dossiers.append(
            {
                "title": title,
                "description": description,
                "kind": kind,
                "origin": origin,
                "approved": approved,
            }
        )
    return dossiers


def _load_config(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise MigrationError(f"server config must be an object: {path}")
    return payload


def _ensure_memu_import(cfg: Mapping[str, Any]) -> None:
    memu_cfg = cfg.get("memu") if isinstance(cfg.get("memu"), dict) else {}
    raw = str(memu_cfg.get("path") or "").strip()
    if raw:
        path = Path(raw).expanduser().resolve()
        source = path / "src" if (path / "src" / "memu").is_dir() else path
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    try:
        __import__("memu")
    except ImportError as exc:
        raise MigrationError("memU is not importable; check config.json memu.path") from exc


def _normalize_title(value: Any) -> str:
    from memu.utils.taxonomy import normalize_category_name

    return normalize_category_name(str(value or "")) or ""


def _configured_profiles(cfg: Mapping[str, Any]) -> dict[str, Any]:
    profiles = default_llm_profiles_from_server_config(dict(cfg))
    llm = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    temperatures = llm.get("step_temperatures") if isinstance(llm.get("step_temperatures"), dict) else {}
    default_profile = profiles.get("default", {})
    for name, temperature in temperatures.items():
        if temperature is None or temperature == "":
            continue
        profile = dict(profiles.get(name, default_profile))
        profile["temperature"] = float(temperature)
        profiles[name] = profile
    return profiles


def _model_identity(cfg: Mapping[str, Any], profiles: Mapping[str, Any]) -> dict[str, Any]:
    embedding_profile = profiles.get("embedding", profiles.get("default", {}))
    embedding = {
        "provider": str(embedding_profile.get("provider") or "openai"),
        "model": str(
            embedding_profile.get("embed_model")
            or embedding_profile.get("chat_model")
            or ""
        ),
        "base_url": str(embedding_profile.get("base_url") or ""),
    }
    if bool(cfg.get("claude_code", False)):
        return {
            "provider": "claude_code",
            "model": str(cfg.get("claude_code_model") or "claude-opus-4-7"),
            "effort": str(cfg.get("claude_code_effort") or "medium"),
            "embedding": embedding,
        }
    profile_name = "category_update" if "category_update" in profiles else "default"
    profile = profiles[profile_name]
    return {
        "profile": profile_name,
        "provider": str(profile.get("provider") or "openai"),
        "model": str(profile.get("chat_model") or ""),
        "base_url": str(profile.get("base_url") or ""),
        "embedding": embedding,
    }


def _build_service(cfg: Mapping[str, Any], path: Path) -> tuple[Any, str, dict[str, Any]]:
    _ensure_memu_import(cfg)
    from memu.app.service import MemoryService

    profiles = _configured_profiles(cfg)
    profile_name = "category_update" if "category_update" in profiles else "default"
    categories = cfg.get("categories") if isinstance(cfg.get("categories"), dict) else {}
    service = MemoryService(
        llm_profiles=profiles,
        blob_config=blob_config_from_cfg(dict(cfg)),
        database_config={
            "metadata_store": {
                "provider": "sqlite",
                "dsn": f"sqlite:////{path.expanduser().resolve().as_posix().lstrip('/')}",
                "ddl_mode": "create",
            }
        },
        memorize_config={
            "dynamic_category_cluster_size": int(categories.get("dynamic_category_cluster_size", 10) or 10),
            "category_summary_target_words": int(categories.get("category_summary_target_words", 300) or 300),
            "category_update_llm_profile": profile_name,
        },
        user_config={"model": ScopeModel},
        claude_code=bool(cfg.get("claude_code", False)),
        claude_code_model=str(cfg.get("claude_code_model") or "claude-opus-4-7"),
        claude_code_effort=str(cfg.get("claude_code_effort") or "medium"),
        claude_code_permission_mode=str(cfg.get("claude_code_permission_mode") or "").strip() or None,
        claude_code_settings=str(cfg.get("claude_code_settings") or "").strip() or None,
        claude_code_workspace=str(cfg.get("claude_code_workspace") or "").strip() or None,
        claude_code_timeout_seconds=int(cfg.get("claude_code_timeout_seconds", 3600) or 3600),
    )
    return service, profile_name, _model_identity(cfg, profiles)


def _close_service(service: Any) -> None:
    database = getattr(service, "database", None)
    close = getattr(database, "close", None)
    if callable(close):
        close()


def _checkpoint_stopped_database(path: Path) -> None:
    conn = sqlite3.connect(path.expanduser().resolve(), timeout=0)
    try:
        conn.execute("PRAGMA busy_timeout=0")
        busy, _, _ = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if busy:
            raise MigrationError("WAL checkpoint is busy; stop every process using this database")
        conn.execute("BEGIN IMMEDIATE")
        conn.rollback()
    finally:
        conn.close()


def _sqlite_backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with closing(sqlite3.connect(source)) as source_conn, closing(sqlite3.connect(temporary)) as target_conn:
            source_conn.backup(target_conn)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _timestamped_backup(path: Path, backup_dir: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = backup_dir / f"{path.stem}-pre-taxonomy-{stamp}{path.suffix or '.db'}"
    _sqlite_backup(path, destination)
    _verify_sqlite_file(destination, label="backup")
    return destination


def _verify_sqlite_file(path: Path, *, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise MigrationError(f"{label} verification failed: {path}")
    with closing(_open_read_only(path)) as conn:
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise MigrationError(f"{label} integrity check failed: {path}")


def _artifact_paths(database: Path, work_dir: Path) -> dict[str, Path]:
    resolved = database.expanduser().resolve()
    source_key = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:8]
    stem = f"{resolved.stem}-{source_key}"
    return {
        "manifest": work_dir / f"{stem}.taxonomy-migration.json",
        "dossiers": work_dir / f"{stem}.dossiers.json",
        "discovery_db": work_dir / f"{stem}.taxonomy-discovery.db",
    }


def _new_manifest(
    database: Path,
    report: Mapping[str, Any],
    scope: Mapping[str, str],
    model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "version": MIGRATION_VERSION,
        "database": str(database.expanduser().resolve()),
        "source_identity": report["identity"],
        "active_memory_identity": report["active_memory_identity"],
        "scope": dict(scope),
        "schema_version": report["schema_version"],
        "model_identity": dict(model_identity),
        "discovery": {"prepared": False, "batches": [], "reviews": [], "complete": False},
        "apply": {
            "reset": False,
            "seeded": False,
            "dossiers": [],
            "batches": [],
            "revisions": [],
            "complete": False,
        },
        "pending_operation": None,
    }


def _verify_discovery_artifacts(paths: Mapping[str, Path], discovery: Mapping[str, Any]) -> None:
    dossier_hash = discovery.get("dossier_artifact_hash")
    if not paths["dossiers"].is_file() or _sha256_file(paths["dossiers"]) != dossier_hash:
        raise MigrationError("completed discovery dossier artifact is missing or changed")
    expected_database = discovery.get("discovery_database_identity")
    if not isinstance(expected_database, dict) or _database_identity(paths["discovery_db"]) != expected_database:
        raise MigrationError("completed discovery database is missing or changed")
    _verify_sqlite_file(paths["discovery_db"], label="discovery database")


def _verify_recovery_backup(manifest: Mapping[str, Any], apply_state: Mapping[str, Any]) -> None:
    path = Path(str(apply_state.get("backup") or ""))
    expected = apply_state.get("backup_identity")
    if (
        apply_state.get("backup_source_identity") != manifest.get("source_identity")
        or not isinstance(expected, dict)
        or not path.is_file()
    ):
        raise MigrationError(f"recorded recovery backup is missing or changed. {OFFLINE_RECOVERY}")
    if _database_identity(path) != expected:
        raise MigrationError(f"recorded recovery backup is missing or changed. {OFFLINE_RECOVERY}")
    _verify_sqlite_file(path, label="recovery backup")


def _load_manifest(path: Path, database: Path, scope: Mapping[str, str]) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or payload.get("version") != MIGRATION_VERSION:
        raise MigrationError(f"invalid taxonomy migration manifest: {path}")
    if payload.get("database") != str(database.expanduser().resolve()) or payload.get("scope") != dict(scope):
        raise MigrationError("migration manifest database or scope does not match")
    return payload


def _set_pending(manifest_path: Path, manifest: dict[str, Any], operation: str | None) -> None:
    manifest["pending_operation"] = operation
    _atomic_write_json(manifest_path, manifest)


async def _create_dossiers(service: Any, scope: Mapping[str, str], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    from memu.utils.taxonomy import category_identity_text

    store = service.database
    created_ids: list[str] = []
    for row in rows:
        categories = store.memory_category_repo.list_categories(scope)
        key = _normalize_title(row["title"])
        existing = next((category for category in categories.values() if _normalize_title(category.name) == key), None)
        if existing is not None:
            if (
                existing.anchor_role is not None
                or existing.description != row["description"]
                or existing.kind != row["kind"]
            ):
                raise MigrationError(f"existing dossier differs from approved artifact: {row['title']}")
            if existing.summary != "## unlabeled":
                raise MigrationError(f"existing dossier has unexpected migration prose: {row['title']}")
            if (
                existing.approved_description != existing.description
                or existing.approved_summary != existing.summary
            ):
                store.memory_category_repo.approve_category_summary(existing.id, scope)
            created_ids.append(existing.id)
            continue
        [embedding] = await service.embed(
            [category_identity_text(row["title"], row["description"])], profile="embedding"
        )
        session_cm = service._sqlite_write_session(store)
        if session_cm is None:
            raise MigrationError("taxonomy migration requires SQLite")
        with session_cm as session:
            category = store.memory_category_repo.create_category_strict(
                name=row["title"],
                description=row["description"],
                embedding=embedding,
                user_data=dict(scope),
                kind=row["kind"],
                session=session,
            )
            store.memory_category_repo.update_category(
                category_id=category.id,
                summary="## unlabeled",
                where=scope,
                session=session,
            )
            session.commit()
        store.memory_category_repo.approve_category_summary(category.id, scope)
        created_ids.append(category.id)
    return created_ids


async def _ensure_approved_anchors(service: Any, scope: Mapping[str, str]) -> None:
    anchors = await service.ensure_dossier_anchors(scope)
    for anchor in anchors.values():
        service.database.memory_category_repo.approve_category_summary(anchor.id, scope)


def _assert_taxonomy_empty(store: Any, scope: Mapping[str, str]) -> None:
    if (
        store.memory_category_repo.list_categories(scope)
        or store.category_item_repo.list_relations(scope)
        or store.dossier_candidate_repo.list_candidates(scope, unresolved_only=False)
    ):
        raise MigrationError("taxonomy reset did not clear all scoped category state")


def _file_proposals(
    service: Any,
    scope: Mapping[str, str],
    rows: Sequence[dict[str, Any]],
    decisions: Mapping[str, Sequence[str]],
    *,
    forbid_candidates: bool,
) -> None:
    store = service.database
    by_ref = {_memory_ref(row): row for row in rows}
    item_ids = {str(row["id"]) for row in rows}
    items = store.memory_item_repo.list_items_by_ids(item_ids, scope)
    if set(items) != item_ids:
        raise MigrationError("taxonomy batch contains missing or inactive memories")
    proposals = [(items[str(by_ref[ref]["id"])], list(titles)) for ref, titles in decisions.items()]
    session_cm = service._sqlite_write_session(store)
    if session_cm is None:
        raise MigrationError("taxonomy migration requires SQLite")
    with session_cm as session:
        _relations, candidates = service.file_category_proposals(
            store=store,
            item_proposals=proposals,
            where=scope,
            session=session,
        )
        if forbid_candidates and candidates:
            raise MigrationError("final assignment attempted to create a dossier candidate")
        session.commit()


async def _chat_taxonomy_batch(
    service: Any,
    profile: str,
    model_identity: Mapping[str, Any],
    system_prompt: str,
    user_prompt: str,
    rows: Sequence[dict[str, Any]],
    *,
    allowed_titles: Mapping[str, set[str]] | None = None,
) -> tuple[str, dict[str, list[str]], str]:
    expected_refs = [_memory_ref(row) for row in rows]
    input_hash = _prompt_hash(system_prompt, user_prompt, model_identity)
    raw = await service.chat(
        user_prompt,
        profile=profile,
        system_prompt=system_prompt,
        op="taxonomy_migration",
        step="historical_assignment" if allowed_titles is not None else "historical_discovery",
    )
    raw_text = str(raw or "")
    return input_hash, _parse_taxonomy_output(raw_text, expected_refs, allowed_titles=allowed_titles), raw_text


async def _cached_taxonomy_batch(
    *,
    entries: list[dict[str, Any]],
    index: int,
    service: Any,
    profile: str,
    model_identity: Mapping[str, Any],
    system_prompt: str,
    user_prompt: str,
    rows: Sequence[dict[str, Any]],
    allowed_titles: Mapping[str, set[str]] | None = None,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    expected_refs = [_memory_ref(row) for row in rows]
    input_hash = _prompt_hash(system_prompt, user_prompt, model_identity)
    if index < len(entries):
        entry = entries[index]
        if entry.get("memory_refs") != expected_refs:
            raise MigrationError(f"cached taxonomy batch {index + 1} no longer matches its input memories")
        if not entry.get("committed") and entry.get("input_hash") != input_hash:
            raise MigrationError(f"cached taxonomy batch {index + 1} no longer matches its exact prompt")
        raw = str(entry.get("raw_output") or "")
        return _parse_taxonomy_output(raw, expected_refs, allowed_titles=allowed_titles), entry
    if index != len(entries):
        raise MigrationError("taxonomy manifest batch sequence is not contiguous")
    input_hash, decision, raw = await _chat_taxonomy_batch(
        service,
        profile,
        model_identity,
        system_prompt,
        user_prompt,
        rows,
        allowed_titles=allowed_titles,
    )
    entry = {
        "input_hash": input_hash,
        "memory_refs": expected_refs,
        "raw_output": raw,
        "committed": False,
    }
    entries.append(entry)
    return decision, entry


def _render_discovery_user(rows: Sequence[dict[str, Any]], seeds: Sequence[Mapping[str, Any]]) -> str:
    seed_titles = "\n".join(f"- {row['title']}" for row in seeds) or "(none)"
    return DISCOVERY_USER_PROMPT.format(
        seed_dossier_titles_or_none=seed_titles,
        memory_records="\n\n".join(_render_memory(row) for row in rows),
    )


def _render_assignment_memory(row: Mapping[str, Any]) -> str:
    choices = row.get("dossier_choices")
    if not isinstance(choices, list):
        raise MigrationError(f"memory {row.get('id')} lacks dossier choices")
    rendered = "\n".join(
        f"- {choice['title']}: {choice['description']}" for choice in choices
    ) or "(none)"
    return f"{_render_memory(row)}\nNearby approved dossiers:\n{rendered}"


def _render_assignment_user(rows: Sequence[dict[str, Any]], scope: Mapping[str, str]) -> str:
    orientation = f"You are {scope['soul_id']}. Your human is {scope['user_id']}."
    return ASSIGNMENT_USER_PROMPT.format(
        identity_orientation_or_none=orientation,
        memory_records_with_dossier_choices="\n\n".join(_render_assignment_memory(row) for row in rows),
    )


def _verify_source_identity(manifest: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    if manifest.get("source_identity") != report.get("identity"):
        raise MigrationError("source database changed since discovery; start a fresh migration workspace")


async def discover_database(
    database: Path,
    *,
    cfg: Mapping[str, Any],
    work_dir: Path,
    seed_path: Path | None,
) -> dict[str, Any]:
    database = database.expanduser().resolve()
    work_dir = work_dir.expanduser().resolve()
    report = scan_database(database)
    scope = _require_scope(report)
    _ensure_memu_import(cfg)
    profiles = _configured_profiles(cfg)
    model_identity = _model_identity(cfg, profiles)
    paths = _artifact_paths(database, work_dir)
    seeds = _validate_dossiers(seed_path, scope, require_approved=True) if seed_path else []
    seed_hash = _sha256_json(seeds)

    if paths["manifest"].exists():
        manifest = _load_manifest(paths["manifest"], database, scope)
        if manifest.get("model_identity") != model_identity:
            raise MigrationError("configured taxonomy migration model changed")
        if manifest.get("seed_dossier_hash") != seed_hash:
            raise MigrationError("seed dossiers changed since discovery began")
        if manifest["discovery"].get("complete"):
            _verify_source_identity(manifest, report)
            _verify_discovery_artifacts(paths, manifest["discovery"])
            return {
                "status": "complete",
                "manifest": str(paths["manifest"]),
                "dossiers": str(paths["dossiers"]),
                "discovery_database": str(paths["discovery_db"]),
            }
    else:
        if paths["discovery_db"].exists():
            raise MigrationError(f"discovery copy exists without a manifest: {paths['discovery_db']}")

    _checkpoint_stopped_database(database)
    report = scan_database(database)
    if paths["manifest"].exists():
        _verify_source_identity(manifest, report)
    else:
        _sqlite_backup(database, paths["discovery_db"])
        manifest = _new_manifest(database, report, scope, model_identity)
        manifest["seed_dossier_hash"] = seed_hash
        _atomic_write_json(paths["manifest"], manifest)

    discovery = manifest["discovery"]
    if not paths["discovery_db"].is_file():
        raise MigrationError(f"discovery database is missing: {paths['discovery_db']}")

    service, profile, current_model = _build_service(cfg, paths["discovery_db"])
    try:
        if current_model != model_identity:
            raise MigrationError("configured taxonomy migration model changed during startup")
        if not discovery.get("prepared"):
            _set_pending(paths["manifest"], manifest, "discovery_prepare")
            service.database.memory_category_repo.clear_categories(scope)
            _assert_taxonomy_empty(service.database, scope)
            service.database.memory_item_repo.backfill_memory_refs(scope)
            await _ensure_approved_anchors(service, scope)
            await _create_dossiers(service, scope, seeds)
            discovery["prepared"] = True
            _set_pending(paths["manifest"], manifest, None)

        memories = _load_active_memories(paths["discovery_db"], scope)
        batches = _prompt_batches(
            memories,
            system_prompt=DISCOVERY_SYSTEM_PROMPT,
            render_user=lambda rows: _render_discovery_user(rows, seeds),
        )
        entries = discovery["batches"]
        if len(entries) > len(batches):
            raise MigrationError("discovery manifest has more batches than the current exact prompts")
        for index, rows in enumerate(batches):
            user_prompt = _render_discovery_user(rows, seeds)
            decision, entry = await _cached_taxonomy_batch(
                entries=entries,
                index=index,
                service=service,
                profile=profile,
                model_identity=model_identity,
                system_prompt=DISCOVERY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                rows=rows,
            )
            if entry.get("committed"):
                continue
            _atomic_write_json(paths["manifest"], manifest)
            _set_pending(paths["manifest"], manifest, f"discovery_batch:{index}")
            _file_proposals(service, scope, rows, decision, forbid_candidates=False)
            entry["committed"] = True
            _set_pending(paths["manifest"], manifest, None)

        from memu.app.memorize_categories import render_dynamic_category_review_prompts
        from memu.utils.taxonomy import category_identity_text

        reviews = discovery["reviews"]
        while True:
            bundles = await service.prepare_dynamic_category_review(
                store=service.database,
                where=scope,
                cluster_size=int(service.memorize_config.dynamic_category_cluster_size),
            )
            if not bundles:
                break
            bundle = dict(bundles[0])
            candidate_ids = sorted(str(value) for value in bundle["candidate_ids"])
            cluster_id = _migration_cluster_id(candidate_ids)
            bundle["cluster_id"] = cluster_id
            system_prompt, user_prompt = render_dynamic_category_review_prompts(bundle)
            input_hash = _prompt_hash(system_prompt, user_prompt, model_identity)
            entry = next(
                (row for row in reviews if row.get("candidate_ids") == candidate_ids),
                None,
            )
            if entry is not None:
                if entry.get("input_hash") != input_hash:
                    raise MigrationError(f"dynamic review prompt changed for cluster {cluster_id}")
                decision = entry.get("decision")
                if not isinstance(decision, dict):
                    raise MigrationError(f"dynamic review cache is invalid for cluster {cluster_id}")
            else:
                decision = await service.generate_dynamic_category_review(bundle)
                entry = {
                    "cluster_id": cluster_id,
                    "candidate_ids": candidate_ids,
                    "input_hash": input_hash,
                    "decision": decision,
                    "committed": False,
                }
                reviews.append(entry)
                _atomic_write_json(paths["manifest"], manifest)
            if entry.get("committed"):
                raise MigrationError(f"committed dynamic cluster was prepared again: {cluster_id}")

            proposed_embedding = None
            if decision.get("action") == "create":
                [proposed_embedding] = await service.embed(
                    [category_identity_text(decision["name"], decision["description"])],
                    profile="embedding",
                )
            _set_pending(paths["manifest"], manifest, f"dynamic_review:{cluster_id}")
            session_cm = service._sqlite_write_session(service.database)
            if session_cm is None:
                raise MigrationError("taxonomy migration requires SQLite")
            with session_cm as session:
                service.apply_dynamic_category_review(
                    store=service.database,
                    where=scope,
                    bundle=bundle,
                    decision=decision,
                    session=session,
                    proposed_embedding=proposed_embedding,
                )
                session.commit()
            entry["committed"] = True
            _set_pending(paths["manifest"], manifest, None)

        candidates_by_id = {
            candidate.id: candidate
            for candidate in service.database.dossier_candidate_repo.list_candidates(
                scope, unresolved_only=False
            )
        }
        for entry in reviews:
            if entry.get("committed"):
                continue
            candidates = [candidates_by_id.get(candidate_id) for candidate_id in entry["candidate_ids"]]
            if any(candidate is None or candidate.last_considered_at is None for candidate in candidates):
                raise MigrationError(f"dynamic review did not commit: {entry['cluster_id']}")
            entry["committed"] = True
        _atomic_write_json(paths["manifest"], manifest)

        seed_names = {_normalize_title(row["title"]) for row in seeds}
        categories = service.database.memory_category_repo.list_categories(scope)
        dossier_rows = [
            {
                "title": category.name,
                "description": category.description,
                "kind": category.kind,
                "origin": "seed" if _normalize_title(category.name) in seed_names else "discovered",
                "approved": _normalize_title(category.name) in seed_names,
            }
            for category in sorted(categories.values(), key=lambda row: (row.name.casefold(), row.id))
            if category.anchor_role is None
        ]
        unresolved = len(service.database.dossier_candidate_repo.list_candidates(scope))
        artifact = {
            "database": str(database),
            "scope": scope,
            "unresolved_candidate_count": unresolved,
            "dossiers": dossier_rows,
        }
        _atomic_write_json(paths["dossiers"], artifact)
        _close_service(service)
        service = None
        _checkpoint_stopped_database(paths["discovery_db"])
        discovery["dossier_artifact_hash"] = _sha256_file(paths["dossiers"])
        discovery["discovery_database_identity"] = _database_identity(paths["discovery_db"])
        discovery["complete"] = True
        _atomic_write_json(paths["manifest"], manifest)
        return {
            "status": "awaiting_dossier_approval",
            "manifest": str(paths["manifest"]),
            "dossiers": str(paths["dossiers"]),
            "discovery_database": str(paths["discovery_db"]),
            "dossier_count": len(dossier_rows),
            "unresolved_candidates": unresolved,
        }
    finally:
        _close_service(service)


async def _assignment_rows(
    service: Any,
    database: Path,
    scope: Mapping[str, str],
) -> list[dict[str, Any]]:
    memories = _load_active_memories(database, scope)
    items = service.database.memory_item_repo.list_items(scope)
    categories = [
        category
        for category in service.database.memory_category_repo.list_categories(scope).values()
        if category.anchor_role is None and category.kind in DOSSIER_KINDS
    ]
    for row in memories:
        item = items.get(str(row["id"]))
        if item is None or not item.embedding:
            raise MigrationError(f"active memory is missing an embedding: {row['id']}")
        hits = await service.search_dossiers(
            item.embedding,
            where=scope,
            view="identity",
            activity="all",
            limit=10,
            min_score=-1.0,
            categories=categories,
        )
        row["dossier_choices"] = [
            {"title": category.name, "description": category.description}
            for category, _score in hits
        ]
    return memories


async def apply_database(
    database: Path,
    *,
    cfg: Mapping[str, Any],
    work_dir: Path,
    dossier_path: Path,
    backup_dir: Path,
) -> dict[str, Any]:
    database = database.expanduser().resolve()
    work_dir = work_dir.expanduser().resolve()
    report = scan_database(database)
    scope = _require_scope(report)
    _ensure_memu_import(cfg)
    paths = _artifact_paths(database, work_dir)
    if not paths["manifest"].is_file():
        raise MigrationError(f"migration manifest is missing: {paths['manifest']}")
    manifest = _load_manifest(paths["manifest"], database, scope)
    discovery = manifest["discovery"]
    apply_state = manifest["apply"]
    if manifest.get("active_memory_identity") != report.get("active_memory_identity"):
        raise MigrationError("active memory set changed since discovery")
    if not discovery.get("complete"):
        raise MigrationError("discovery and human dossier review must finish before apply")

    profiles = _configured_profiles(cfg)
    model_identity = _model_identity(cfg, profiles)
    if manifest.get("model_identity") != model_identity:
        raise MigrationError("configured taxonomy migration model changed")
    dossiers = _validate_dossiers(dossier_path, scope, require_approved=True)
    dossier_hash = _sha256_file(dossier_path)
    previous_hash = manifest.get("dossier_artifact_hash")
    if previous_hash is not None and previous_hash != dossier_hash:
        raise MigrationError("approved dossier artifact changed after target apply began")

    if apply_state.get("complete"):
        return validate_database(database, dossier_path=dossier_path)

    _checkpoint_stopped_database(database)
    report = scan_database(database)
    if manifest.get("active_memory_identity") != report.get("active_memory_identity"):
        raise MigrationError(f"active memory set changed since discovery. {OFFLINE_RECOVERY}")
    if apply_state.get("backup"):
        _verify_recovery_backup(manifest, apply_state)
    if not apply_state.get("reset"):
        if manifest.get("pending_operation") != "target_reset":
            _verify_source_identity(manifest, report)
            backup = _timestamped_backup(database, backup_dir.expanduser().resolve())
            manifest["dossier_artifact_hash"] = dossier_hash
            apply_state["backup"] = str(backup)
            apply_state["backup_identity"] = _database_identity(backup)
            apply_state["backup_source_identity"] = report["identity"]
            _set_pending(paths["manifest"], manifest, "target_reset")
        else:
            taxonomy_is_empty = (
                not report["categories"]
                and report["category_links"] == 0
                and report["dossier_candidates"] == 0
            )
            if taxonomy_is_empty:
                apply_state["reset"] = True
                _set_pending(paths["manifest"], manifest, None)
            else:
                _verify_source_identity(manifest, report)

    service, profile, current_model = _build_service(cfg, database)
    try:
        if current_model != model_identity:
            raise MigrationError("configured taxonomy migration model changed during startup")
        if not apply_state.get("reset"):
            service.database.memory_category_repo.clear_categories(scope)
            _assert_taxonomy_empty(service.database, scope)
            apply_state["reset"] = True
            _set_pending(paths["manifest"], manifest, None)

        if not apply_state.get("seeded"):
            _set_pending(paths["manifest"], manifest, "target_dossier_seed")
            service.database.memory_item_repo.backfill_memory_refs(scope)
            await _ensure_approved_anchors(service, scope)
            apply_state["dossiers"] = await _create_dossiers(service, scope, dossiers)
            apply_state["seeded"] = True
            _set_pending(paths["manifest"], manifest, None)

        entries = apply_state["batches"]
        if apply_state["revisions"]:
            apply_state["assignments_complete"] = True
        if apply_state.get("assignments_complete"):
            batches = []
        else:
            rows = await _assignment_rows(service, database, scope)
            batches = _prompt_batches(
                rows,
                system_prompt=ASSIGNMENT_SYSTEM_PROMPT,
                render_user=lambda batch: _render_assignment_user(batch, scope),
            )
            if len(entries) > len(batches):
                raise MigrationError("assignment manifest has more batches than the current exact prompts")
        for index, batch in enumerate(batches):
            user_prompt = _render_assignment_user(batch, scope)
            allowed_titles = {
                _memory_ref(row): {str(choice["title"]) for choice in row["dossier_choices"]}
                for row in batch
            }
            decision, entry = await _cached_taxonomy_batch(
                entries=entries,
                index=index,
                service=service,
                profile=profile,
                model_identity=model_identity,
                system_prompt=ASSIGNMENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                rows=batch,
                allowed_titles=allowed_titles,
            )
            if entry.get("committed"):
                continue
            _atomic_write_json(paths["manifest"], manifest)
            _set_pending(paths["manifest"], manifest, f"assignment_batch:{index}")
            _file_proposals(service, scope, batch, decision, forbid_candidates=True)
            entry["committed"] = True
            _set_pending(paths["manifest"], manifest, None)
        if not apply_state.get("assignments_complete"):
            apply_state["assignments_complete"] = True
            _atomic_write_json(paths["manifest"], manifest)

        from memu.app.dossier import render_dossier_revision_prompts

        revision_entries = apply_state["revisions"]
        while due := service.list_due_dossiers(scope):
            dossier = due[0]
            bundle = service.prepare_dossier_revision(dossier.id, scope)
            system_prompt, user_prompt = render_dossier_revision_prompts(
                bundle, include_relative=False
            )
            input_hash = _prompt_hash(system_prompt, user_prompt, model_identity)
            entry = next((row for row in revision_entries if row.get("dossier_id") == dossier.id), None)
            if entry is not None:
                if entry.get("input_hash") != input_hash:
                    raise MigrationError(f"dossier revision prompt changed for {dossier.name}")
                decision = entry.get("decision")
                if not isinstance(decision, dict):
                    raise MigrationError(f"dossier revision cache is invalid for {dossier.name}")
            else:
                if _estimate_tokens(system_prompt + "\n" + user_prompt) > 100_000:
                    apply_state["blocked_dossier"] = {"id": dossier.id, "title": dossier.name}
                    _atomic_write_json(paths["manifest"], manifest)
                    raise MigrationError(
                        f"dossier exceeds the 100000-token revision guard: {dossier.name}"
                    )
                decision = await service.generate_dossier_revision(bundle)
                entry = {
                    "dossier_id": dossier.id,
                    "title": dossier.name,
                    "input_hash": input_hash,
                    "decision": decision,
                    "committed": False,
                }
                revision_entries.append(entry)
                _atomic_write_json(paths["manifest"], manifest)
            if entry.get("committed"):
                raise MigrationError(f"committed dossier remains due for revision: {dossier.name}")
            _set_pending(paths["manifest"], manifest, f"dossier_revision:{dossier.id}")
            await service.apply_dossier_revision(bundle, decision, scope)
            entry["committed"] = True
            _set_pending(paths["manifest"], manifest, None)

        current_categories = service.database.memory_category_repo.list_categories(scope)
        for entry in revision_entries:
            if entry.get("committed"):
                continue
            current = current_categories.get(str(entry["dossier_id"]))
            if current is None or current.last_revised_at is None:
                raise MigrationError(f"dossier revision did not commit: {entry['title']}")
            entry["committed"] = True

        apply_state.pop("blocked_dossier", None)
    finally:
        _close_service(service)

    _checkpoint_stopped_database(database)
    final_report = scan_database(database)
    if manifest.get("active_memory_identity") != final_report.get("active_memory_identity"):
        raise MigrationError(f"active memory set changed during apply. {OFFLINE_RECOVERY}")
    validation = validate_database(database, dossier_path=dossier_path)
    if not validation["ok"]:
        raise MigrationError(f"target apply committed but validation failed. {OFFLINE_RECOVERY}")
    apply_state["complete"] = True
    _atomic_write_json(paths["manifest"], manifest)
    return {
        "status": "complete",
        "manifest": str(paths["manifest"]),
        "backup": apply_state["backup"],
        "validation": validation,
    }


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def validate_database(database: Path, *, dossier_path: Path | None = None) -> dict[str, Any]:
    report = scan_database(database)
    scope = _require_scope(report)
    errors: list[str] = []
    warnings: list[str] = []
    artifact_dossiers: dict[str, dict[str, Any]] | None = None
    if dossier_path is not None:
        artifact_dossiers = {
            _normalize_title(row["title"]): row
            for row in _validate_dossiers(dossier_path, scope, require_approved=True)
        }

    with closing(_open_read_only(database)) as conn:
        params = (scope["user_id"], scope["soul_id"])
        memories = {
            str(row["id"]): dict(row)
            for row in conn.execute(
                "SELECT m.* FROM memory_items m "
                f"WHERE m.user_id = ? AND m.soul_id = ? AND {_active_memory_clause()}",
                params,
            ).fetchall()
        }
        categories = {
            str(row["id"]): dict(row)
            for row in conn.execute(
                "SELECT * FROM categories WHERE user_id = ? AND soul_id = ?",
                params,
            ).fetchall()
        }
        relations = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM category_items WHERE user_id = ? AND soul_id = ?",
                params,
            ).fetchall()
        ]
        all_candidates = int(
            conn.execute(
                "SELECT COUNT(*) FROM dossier_candidates WHERE user_id = ? AND soul_id = ?",
                params,
            ).fetchone()[0]
        )
        cross_scope_relations = int(
            conn.execute(
                "SELECT COUNT(*) FROM category_items ci "
                "JOIN categories c ON c.id = ci.category_id "
                "JOIN memory_items m ON m.id = ci.item_id "
                "WHERE ci.user_id != c.user_id OR ci.soul_id != c.soul_id "
                "OR ci.user_id != m.user_id OR ci.soul_id != m.soul_id"
            ).fetchone()[0]
        )

    anchors = [row for row in categories.values() if row.get("anchor_role") is not None]
    if sorted(row["anchor_role"] for row in anchors) != ["soul", "user"]:
        errors.append("expected exactly one soul and one user anchor")
    refs = [int(row["memory_ref"]) for row in memories.values() if row.get("memory_ref") is not None]
    if len(refs) != len(memories) or len(refs) != len(set(refs)):
        errors.append("active memories do not have unique [M#] references")
    ordinary = [row for row in categories.values() if row.get("anchor_role") is None]
    if any(row.get("kind") not in DOSSIER_KINDS for row in categories.values()):
        errors.append("one or more categories lack a valid dossier kind")
    normalized_titles = [_normalize_title(row["name"]) for row in categories.values()]
    if len(normalized_titles) != len(set(normalized_titles)):
        errors.append("duplicate normalized dossier titles remain")
    if artifact_dossiers is not None:
        stored_dossiers = {_normalize_title(row["name"]): row for row in ordinary}
        if stored_dossiers.keys() != artifact_dossiers.keys():
            errors.append("stored non-anchor dossiers differ from the approved artifact")
        else:
            for title, artifact in artifact_dossiers.items():
                stored = stored_dossiers[title]
                if (
                    stored.get("kind") != artifact["kind"]
                    or stored.get("approved_description") != artifact["description"]
                ):
                    errors.append(
                        f"stored dossier identity differs from the approved artifact: {stored['name']}"
                    )
    if all_candidates:
        errors.append(f"{all_candidates} dossier candidate rows remain")
    if cross_scope_relations:
        errors.append(f"{cross_scope_relations} cross-scope category relations remain")

    links_by_category: dict[str, set[str]] = {}
    for relation in relations:
        category_id = str(relation["category_id"])
        item_id = str(relation["item_id"])
        if category_id not in categories or item_id not in memories:
            errors.append(f"dangling or inactive category relation: {relation['id']}")
            continue
        links_by_category.setdefault(category_id, set()).add(item_id)
    if any(links_by_category.get(str(anchor["id"])) for anchor in anchors):
        errors.append("anchor dossiers have memory memberships")

    by_ref = {int(row["memory_ref"]): row for row in memories.values() if row.get("memory_ref") is not None}
    for dossier in ordinary:
        dossier_id = str(dossier["id"])
        member_ids = links_by_category.get(dossier_id, set())
        cited_refs = {int(value) for value in re.findall(r"\[M([1-9][0-9]*)\]", str(dossier.get("summary") or ""))}
        for memory_ref in cited_refs:
            item = by_ref.get(memory_ref)
            if item is None or str(item["id"]) not in member_ids:
                errors.append(f"dossier {dossier['name']} cites inactive or unlinked [M{memory_ref}]")
        member_times = [
            _parse_datetime(memories[item_id].get("happened_at") or memories[item_id].get("created_at"))
            for item_id in member_ids
        ]
        expected_evidence = max((value for value in member_times if value is not None), default=None)
        if _parse_datetime(dossier.get("last_evidence_at")) != expected_evidence:
            errors.append(f"dossier {dossier['name']} has stale last_evidence_at")
        if not member_ids:
            warnings.append(f"empty approved dossier: {dossier['name']}")
        if dossier.get("last_evidence_at") is None:
            warnings.append(f"dormant dossier: {dossier['name']}")

    pending = [
        row["name"]
        for row in categories.values()
        if row.get("approved_description") != row.get("description")
        or row.get("approved_summary") != row.get("summary")
    ]
    uncategorized = len(set(memories) - {item_id for ids in links_by_category.values() for item_id in ids})
    return {
        "ok": not errors,
        "database": str(database.expanduser().resolve()),
        "scope": scope,
        "errors": errors,
        "warnings": warnings,
        "active_memories": len(memories),
        "dossiers": len(ordinary),
        "pending_dossier_approvals": pending,
        "uncategorized_memories": uncategorized,
    }


def _inventory_command(database: Path) -> dict[str, Any]:
    report = scan_database(database)
    try:
        scope = _require_scope(report)
        if report["missing_memory_refs"] or report["duplicate_memory_refs"]:
            raise MigrationError("memory references must be backfilled before exact prompt estimates")
        rows = _load_active_memories(database, scope)
        report["estimated_discovery_batches"] = len(
            _prompt_batches(
                rows,
                system_prompt=DISCOVERY_SYSTEM_PROMPT,
                render_user=lambda batch: _render_discovery_user(batch, ()),
            )
        )
        report["estimated_assignment_batches"] = None
        report["assignment_estimate_note"] = (
            "available during apply after each memory's exact top-ten dossier search"
        )
    except MigrationError as exc:
        report["estimated_discovery_batches"] = None
        report["estimated_assignment_batches"] = None
        report["estimate_error"] = str(exc)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.json")
    parser.add_argument("--work-dir", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="phase", required=True)

    inventory = subparsers.add_parser("inventory", help="inspect one database without writing")
    inventory.add_argument("database", type=Path, nargs="?")
    inventory.add_argument(
        "--sqlite-dir",
        action="store_true",
        help="inventory all existing *.db files in the configured SQLite directory",
    )

    discover = subparsers.add_parser("discover", help="discover dossiers on a disposable database copy")
    discover.add_argument("database", type=Path)
    discover.add_argument("--seed-dossiers", type=Path)

    apply_parser = subparsers.add_parser(
        "apply",
        help="reset and rebuild one stopped target database",
        description=OFFLINE_RECOVERY,
    )
    apply_parser.add_argument("database", type=Path)
    apply_parser.add_argument("--dossiers", type=Path)
    apply_parser.add_argument("--backup-dir", type=Path)
    apply_parser.add_argument("--apply", action="store_true", help="authorize target mutation")
    apply_parser.add_argument("--confirm", help="exact resolved target database path")

    validate = subparsers.add_parser("validate", help="validate a migrated database without writing")
    validate.add_argument("database", type=Path)
    validate.add_argument("--dossiers", type=Path)
    return parser


def _normalized_argv(argv: Sequence[str] | None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    phases = {"inventory", "discover", "apply", "validate"}
    index = 0
    while index < len(values):
        value = values[index]
        if value in {"--config", "--work-dir"}:
            index += 2
            continue
        if value.startswith("--config=") or value.startswith("--work-dir="):
            index += 1
            continue
        break
    if index >= len(values) or values[index] not in phases:
        values.insert(index, "inventory")
    return values


async def _run(args: argparse.Namespace) -> Any:
    cfg = _load_config(args.config.expanduser().resolve())
    _ensure_memu_import(cfg)
    work_dir = args.work_dir.expanduser().resolve()
    if args.phase == "inventory":
        if args.sqlite_dir:
            if args.database is not None:
                raise MigrationError("inventory accepts either DATABASE or --sqlite-dir, not both")
            directory = sqlite_dir_from_cfg(cfg)
            return [_inventory_command(path) for path in sorted(directory.glob("*.db")) if path.is_file()]
        if args.database is None:
            raise MigrationError("inventory requires DATABASE or --sqlite-dir")
        return _inventory_command(args.database)
    if args.phase == "discover":
        return await discover_database(
            args.database,
            cfg=cfg,
            work_dir=work_dir,
            seed_path=args.seed_dossiers,
        )
    if args.phase == "apply":
        database = args.database.expanduser().resolve()
        if not args.apply or args.confirm != str(database):
            raise MigrationError(
                f"target apply requires --apply --confirm {database}"
            )
        paths = _artifact_paths(database, work_dir)
        dossier_path = (args.dossiers or paths["dossiers"]).expanduser().resolve()
        backup_dir = (args.backup_dir or database.parent / "_backups").expanduser().resolve()
        return await apply_database(
            database,
            cfg=cfg,
            work_dir=work_dir,
            dossier_path=dossier_path,
            backup_dir=backup_dir,
        )
    dossier_path = args.dossiers.expanduser().resolve() if args.dossiers else None
    return validate_database(args.database, dossier_path=dossier_path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(_normalized_argv(argv))
    try:
        result = asyncio.run(_run(args))
    except (MigrationError, OSError, sqlite3.Error, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if args.phase == "apply" and OFFLINE_RECOVERY not in str(exc):
            print(OFFLINE_RECOVERY, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    if isinstance(result, dict) and result.get("ok") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
