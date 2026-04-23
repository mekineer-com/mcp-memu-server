from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import yaml

from app import procedural


class _FakeSvc:
    def __init__(self, embed_model: str = "test-embed-2d") -> None:
        cfg = SimpleNamespace(embed_model=embed_model, chat_model="")
        self.llm_profiles = SimpleNamespace(profiles={"embedding": cfg})

    async def embed(self, texts: list[str], profile: str = "embedding") -> list[list[float]]:
        assert profile == "embedding"
        return [[float(idx + 1), float(idx + 2)] for idx, _ in enumerate(texts)]


def test_ensure_schema_migrates_legacy_pk_to_composite(tmp_path: Path) -> None:
    db_path = tmp_path / "procedural.db"
    con = sqlite3.connect(db_path)
    con.execute(
        """
CREATE TABLE procedural_memory_items (
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
)
"""
    )
    con.execute(
        """
INSERT INTO procedural_memory_items (
    id, domain, framework, summary, applicable_when, source, tags, embedding, embedding_model, source_hash, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
        (
            "same-id",
            "mental_health",
            None,
            "legacy row",
            None,
            None,
            "[]",
            procedural._pack_embedding([0.1, 0.2]),
            "legacy-model",
            "h1",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    con.commit()

    procedural.ensure_schema(con)
    con.execute(
        """
INSERT INTO procedural_memory_items (
    id, domain, framework, summary, applicable_when, source, tags, embedding, embedding_model, source_hash, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
        (
            "same-id",
            "tool_use",
            None,
            "domain variant",
            None,
            None,
            "[]",
            procedural._pack_embedding([0.3, 0.4]),
            "legacy-model",
            "h2",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    con.commit()
    count = con.execute("SELECT COUNT(*) FROM procedural_memory_items WHERE id = 'same-id'").fetchone()[0]
    con.close()
    assert count == 2


def test_ingest_retracts_removed_yaml_entries(tmp_path: Path) -> None:
    db_path = tmp_path / "procedural.db"
    yaml_dir = tmp_path / "procedural"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = yaml_dir / "mental_health.yaml"

    yaml.safe_dump(
        [
            {
                "id": "mh_a",
                "domain": "mental_health",
                "summary": "entry a",
                "applicable_when": "when anxious",
                "framework": "cbt",
                "tags": ["grounding"],
            },
            {
                "id": "mh_b",
                "domain": "mental_health",
                "summary": "entry b",
                "applicable_when": "when overwhelmed",
                "framework": "dbt",
                "tags": ["breathing"],
            },
        ],
        yaml_path.open("w", encoding="utf-8"),
        sort_keys=False,
        allow_unicode=True,
    )

    svc = _FakeSvc()
    first = asyncio.run(procedural.ingest(svc, db_path, yaml_dir))
    assert first["scanned"] == 2
    assert first["added"] == 2

    yaml.safe_dump(
        [
            {
                "id": "mh_a",
                "domain": "mental_health",
                "summary": "entry a",
                "applicable_when": "when anxious",
                "framework": "cbt",
                "tags": ["grounding"],
            }
        ],
        yaml_path.open("w", encoding="utf-8"),
        sort_keys=False,
        allow_unicode=True,
    )

    second = asyncio.run(procedural.ingest(svc, db_path, yaml_dir))
    assert second["scanned"] == 1
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT id, embedding_model FROM procedural_memory_items WHERE domain = ? ORDER BY id",
        ("mental_health",),
    ).fetchall()
    con.close()
    assert rows == [("mh_a", "test-embed-2d")]


def test_lookup_fails_loud_on_dimension_or_model_mismatch(tmp_path: Path) -> None:
    db_path = tmp_path / "procedural.db"
    con = sqlite3.connect(db_path)
    procedural.ensure_schema(con)
    con.execute(
        """
INSERT INTO procedural_memory_items (
    id, domain, framework, summary, applicable_when, source, tags, embedding, embedding_model, source_hash, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
        (
            "mh_dim",
            "mental_health",
            None,
            "entry",
            None,
            None,
            "[]",
            procedural._pack_embedding([0.1, 0.2, 0.3]),
            "embed-v1",
            "h",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    con.commit()
    con.close()

    try:
        procedural.lookup(db_path, domain="mental_health", query_vec=[0.1, 0.2], limit=1)
        assert False, "expected dimension mismatch ValueError"
    except ValueError as exc:
        assert "dimension mismatch" in str(exc)

    try:
        procedural.lookup(
            db_path,
            domain="mental_health",
            query_vec=[0.1, 0.2, 0.3],
            expected_embedding_model="embed-v2",
            limit=1,
        )
        assert False, "expected embedding model mismatch ValueError"
    except ValueError as exc:
        assert "model mismatch" in str(exc)
