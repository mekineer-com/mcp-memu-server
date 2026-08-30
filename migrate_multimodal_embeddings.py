#!/usr/bin/env python3
"""Build a fully re-embedded memU SQLite replacement without touching the source."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import mimetypes
import os
import sqlite3
import struct
import sys
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Any

DIMENSIONS = 3072
SUPPORTED_MODEL = "gemini-embedding-2"


class MigrationError(RuntimeError):
    pass


def _load_client(config_path: Path) -> Any:
    config = json.loads(config_path.read_text())
    memu_path = Path(str(config.get("memu", {}).get("path") or "../memu/src")).expanduser()
    if not memu_path.is_absolute():
        memu_path = (config_path.parent / memu_path).resolve()
    sys.path.insert(0, str(memu_path))
    from memu.embedding import HTTPEmbeddingClient

    embedding = config.get("llm", {}).get("embedding", {})
    if not isinstance(embedding, dict):
        embedding = {}
    key = str(
        embedding.get("api_key")
        or config.get("mentra", {}).get("gemini_api_key")
        or ""
    ).strip()
    if not key:
        raise MigrationError("config mentra.gemini_api_key is required")
    return HTTPEmbeddingClient(
        base_url=str(embedding.get("base_url") or "https://generativelanguage.googleapis.com/"),
        api_key=key,
        embed_model=str(embedding.get("embed_model") or "gemini-embedding-2"),
        provider="gemini",
        timeout=180,
    )


def _copy_database(source: Path, destination: Path) -> None:
    with closing(sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)) as source_db, closing(sqlite3.connect(destination)) as target_db:
        source_db.backup(target_db)


def _rows(conn: sqlite3.Connection) -> dict[str, list[tuple[Any, ...]]]:
    return {
        "memory_items": conn.execute(
            "SELECT id, memory_type, summary FROM memory_items ORDER BY id"
        ).fetchall(),
        "categories": conn.execute(
            "SELECT id, name, description, summary FROM categories ORDER BY id"
        ).fetchall(),
        "resources": conn.execute(
            "SELECT id, modality, local_path, caption FROM resources ORDER BY id"
        ).fetchall(),
    }


async def _embeddings(client: Any, rows: dict[str, list[tuple[Any, ...]]]) -> dict[str, list[list[float]]]:
    async def embed_texts(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), 64):
            vectors.extend(await client.embed(texts[offset : offset + 64]))
        return vectors

    def required_text(value: Any, label: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise MigrationError(f"{label} has no canonical text to embed")
        return text

    memory_texts = [required_text(summary, f"MemoryItem {_id}") for _id, _kind, summary in rows["memory_items"]]
    dossier_texts = [
        f"{required_text(name, f'Dossier {_id} name')}: {str(description or '').strip()}".rstrip(": ")
        for _id, name, description, _summary in rows["categories"]
    ]
    result = {
        "memory_items": await embed_texts(memory_texts),
        "categories": await embed_texts(dossier_texts),
        "resources": [],
    }
    for _id, modality, local_path, caption in rows["resources"]:
        if modality in {"image", "audio", "video"}:
            path = Path(local_path).expanduser()
            mime_type = mimetypes.guess_type(path.name)[0]
            if not path.is_file() or not mime_type:
                raise MigrationError(f"raw {modality} Resource is unavailable: {path}")
            vector = await client.embed_media(path.read_bytes(), mime_type)
        else:
            text = str(caption or "").strip()
            if not text:
                raise MigrationError(f"non-media Resource {_id} has no caption to embed")
            vector = (await client.embed([text]))[0]
        result["resources"].append(vector)
    return result


def _blob(vector: list[float]) -> bytes:
    if len(vector) != DIMENSIONS:
        raise MigrationError(f"expected {DIMENSIONS} dimensions, got {len(vector)}")
    if not all(math.isfinite(value) for value in vector):
        raise MigrationError("embedding contains a non-finite value")
    return struct.pack(f"{DIMENSIONS}f", *vector)


def _write_and_validate(
    conn: sqlite3.Connection,
    rows: dict[str, list[tuple[Any, ...]]],
    vectors: dict[str, list[list[float]]],
    profile: str,
) -> None:
    for table, table_rows in rows.items():
        table_vectors = vectors[table]
        if len(table_vectors) != len(table_rows):
            raise MigrationError(f"{table} embedding count mismatch")
        conn.executemany(
            f"UPDATE {table} SET embedding = ? WHERE id = ?",
            [(_blob(vector), row[0]) for row, vector in zip(table_rows, table_vectors, strict=True)],
        )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embedding_profile "
        "(id INTEGER PRIMARY KEY CHECK (id = 1), profile TEXT NOT NULL)"
    )
    conn.execute("DELETE FROM embedding_profile")
    conn.execute("INSERT INTO embedding_profile (id, profile) VALUES (1, ?)", (profile,))
    for table, table_rows in rows.items():
        count, wrong = conn.execute(
            f"SELECT COUNT(*), SUM(typeof(embedding) != 'blob' OR length(embedding) != ?) FROM {table}",
            (DIMENSIONS * 4,),
        ).fetchone()
        if count != len(table_rows) or wrong:
            raise MigrationError(f"{table} post-write validation failed")
        expected_ids = [row[0] for row in table_rows]
        actual_ids = [row[0] for row in conn.execute(f"SELECT id FROM {table} ORDER BY id")]
        if actual_ids != expected_ids:
            raise MigrationError(f"{table} IDs changed during migration")
    if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise MigrationError("replacement integrity_check failed")


async def migrate(source: Path, destination: Path, config_path: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_file():
        raise MigrationError(f"source database does not exist: {source}")
    if source == destination:
        raise MigrationError("source and destination must differ")
    if destination.exists():
        raise MigrationError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        _copy_database(source, temporary)
        with closing(sqlite3.connect(temporary)) as conn:
            rows = _rows(conn)
        client = _load_client(config_path)
        if client.provider != "gemini" or client.embed_model != SUPPORTED_MODEL:
            raise MigrationError(
                f"migration requires gemini/{SUPPORTED_MODEL}, got {client.provider}/{client.embed_model}"
            )
        profile = f"{client.embed_model}:{DIMENSIONS}"
        vectors = await _embeddings(client, rows)
        with closing(sqlite3.connect(temporary)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            _write_and_validate(conn, rows, vectors, profile)
            conn.commit()
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"source": str(source), "destination": str(destination), "profile": profile, "counts": {k: len(v) for k, v in rows.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    args = parser.parse_args()
    try:
        result = asyncio.run(migrate(args.source, args.destination, args.config))
    except (MigrationError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
