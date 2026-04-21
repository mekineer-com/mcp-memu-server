"""Procedural-memory sidecar — curated, server-global, per-domain.

Corpus lives on disk in `memu/procedural/<domain>.yaml`. This module ingests
it into a standalone SQLite file (`procedural.db`) with pre-computed
embeddings, then serves top-k cosine lookups at retrieve time. The soul
never writes here.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS procedural_memory_items (
    id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    framework TEXT,
    summary TEXT NOT NULL,
    applicable_when TEXT,
    source TEXT,
    tags TEXT,
    embedding BLOB,
    embedding_model TEXT,
    source_hash TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_procedural_domain ON procedural_memory_items(domain);
"""


def ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript(_SCHEMA_SQL)


def _entry_hash(entry: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "domain": str(entry.get("domain") or "").strip(),
            "summary": str(entry.get("summary") or "").strip(),
            "applicable_when": str(entry.get("applicable_when") or "").strip(),
            "framework": str(entry.get("framework") or "").strip(),
            "tags": sorted(entry.get("tags") or []),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _pack_embedding(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_embedding(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _load_yaml_entries(yaml_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for yf in sorted(yaml_dir.glob("*.yaml")):
        with yf.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
        for row in data:
            if isinstance(row, dict) and row.get("id") and row.get("summary"):
                entries.append(row)
    return entries


async def ingest(svc: Any, db_path: Path, yaml_dir: Path) -> dict[str, int]:
    """Embed + upsert any new or changed entries from YAML. Idempotent."""
    entries = _load_yaml_entries(yaml_dir)
    if not entries:
        return {"scanned": 0, "added": 0, "updated": 0, "unchanged": 0}

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        ensure_schema(con)
        prior = {
            row["id"]: row["source_hash"]
            for row in con.execute(
                "SELECT id, source_hash FROM procedural_memory_items"
            ).fetchall()
        }

        fresh: list[tuple[dict[str, Any], str]] = []
        for entry in entries:
            h = _entry_hash(entry)
            if prior.get(str(entry["id"])) != h:
                fresh.append((entry, h))

        added = 0
        updated = 0
        if fresh:
            summaries = [str(e.get("summary") or "").strip() for e, _ in fresh]
            embeddings = await svc.embed(summaries, profile="embedding")
            now = datetime.now(UTC).isoformat()
            for (entry, h), vec in zip(fresh, embeddings, strict=True):
                eid = str(entry["id"])
                con.execute(
                    """
INSERT INTO procedural_memory_items
  (id, domain, framework, summary, applicable_when, source, tags, embedding, embedding_model, source_hash, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
  domain=excluded.domain,
  framework=excluded.framework,
  summary=excluded.summary,
  applicable_when=excluded.applicable_when,
  source=excluded.source,
  tags=excluded.tags,
  embedding=excluded.embedding,
  embedding_model=excluded.embedding_model,
  source_hash=excluded.source_hash,
  updated_at=excluded.updated_at
""",
                    (
                        eid,
                        str(entry.get("domain") or "").strip(),
                        str(entry.get("framework") or "").strip() or None,
                        str(entry.get("summary") or "").strip(),
                        str(entry.get("applicable_when") or "").strip() or None,
                        str(entry.get("source") or "").strip() or None,
                        json.dumps(entry.get("tags") or [], ensure_ascii=False),
                        _pack_embedding(list(vec)),
                        "embedding",
                        h,
                        now,
                    ),
                )
                if eid in prior:
                    updated += 1
                else:
                    added += 1
        con.commit()
        return {
            "scanned": len(entries),
            "added": added,
            "updated": updated,
            "unchanged": len(entries) - added - updated,
        }
    finally:
        con.close()


def _cosine_prenorm(a: list[float], b: list[float]) -> float:
    # `a` is caller-normalized; `b` is raw. One sqrt per row.
    mag = sum(x * x for x in b) ** 0.5
    if mag == 0.0:
        return 0.0
    return sum(ax * bx for ax, bx in zip(a, b, strict=False)) / mag


def _normalize(v: list[float]) -> list[float]:
    mag = sum(x * x for x in v) ** 0.5
    if mag == 0.0:
        return v
    return [x / mag for x in v]


def lookup(
    db_path: Path,
    *,
    domain: str,
    query_vec: list[float],
    limit: int = 1,
) -> list[tuple[dict[str, Any], float]]:
    if not db_path.exists() or not query_vec:
        return []
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        ensure_schema(con)
        rows = con.execute(
            "SELECT id, domain, framework, summary, applicable_when, source, tags, embedding "
            "FROM procedural_memory_items WHERE domain = ?",
            (domain,),
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return []
    qn = _normalize(list(query_vec))
    scored: list[tuple[dict[str, Any], float]] = []
    for r in rows:
        emb = _unpack_embedding(r["embedding"])
        if not emb:
            continue
        sim = _cosine_prenorm(qn, emb)
        scored.append(
            (
                {
                    "id": r["id"],
                    "domain": r["domain"],
                    "framework": r["framework"],
                    "summary": r["summary"],
                    "applicable_when": r["applicable_when"],
                    "source": r["source"],
                    "tags": json.loads(r["tags"] or "[]"),
                },
                sim,
            )
        )
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:limit]
