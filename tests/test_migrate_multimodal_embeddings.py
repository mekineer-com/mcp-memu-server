import sqlite3
from pathlib import Path

import pytest

import migrate_multimodal_embeddings as migration


class FakeClient:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1)] * migration.DIMENSIONS for index, _text in enumerate(texts)]

    async def embed_media(self, data: bytes, mime_type: str) -> list[float]:
        assert data == b"image"
        assert mime_type == "image/png"
        return [3.0] * migration.DIMENSIONS


def _database(path: Path, image: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript("""
CREATE TABLE memory_items(id TEXT PRIMARY KEY, memory_type TEXT, summary TEXT, embedding BLOB);
CREATE TABLE categories(id TEXT PRIMARY KEY, name TEXT, description TEXT, summary TEXT, embedding BLOB);
CREATE TABLE resources(id TEXT PRIMARY KEY, modality TEXT, local_path TEXT, caption TEXT, embedding BLOB);
""")
        conn.execute("INSERT INTO memory_items VALUES ('m1', 'episode', 'A memory', X'00')")
        conn.execute("INSERT INTO categories VALUES ('c1', 'Life', 'Identity', 'Prose', X'00')")
        conn.execute("INSERT INTO resources VALUES ('r1', 'image', ?, 'caption', X'00')", (str(image),))


@pytest.mark.asyncio
async def test_migration_builds_atomic_profiled_replacement(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.db"
    destination = tmp_path / "replacement.db"
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    _database(source, image)
    monkeypatch.setattr(migration, "_load_client", lambda _path: FakeClient())

    result = await migration.migrate(source, destination, tmp_path / "config.json")

    assert result["counts"] == {"memory_items": 1, "categories": 1, "resources": 1}
    with sqlite3.connect(source) as conn:
        assert conn.execute("SELECT embedding FROM memory_items").fetchone()[0] == b"\x00"
    with sqlite3.connect(destination) as conn:
        assert conn.execute("SELECT profile FROM embedding_profile").fetchone()[0] == migration.PROFILE
        assert conn.execute("SELECT length(embedding) FROM memory_items").fetchone()[0] == 12288
        assert conn.execute("SELECT length(embedding) FROM categories").fetchone()[0] == 12288
        assert conn.execute("SELECT length(embedding) FROM resources").fetchone()[0] == 12288


@pytest.mark.asyncio
async def test_migration_failure_publishes_nothing(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.db"
    destination = tmp_path / "replacement.db"
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    _database(source, image)

    class BadClient(FakeClient):
        async def embed_media(self, data: bytes, mime_type: str) -> list[float]:
            return [1.0]

    monkeypatch.setattr(migration, "_load_client", lambda _path: BadClient())
    with pytest.raises(migration.MigrationError, match="expected 3072"):
        await migration.migrate(source, destination, tmp_path / "config.json")
    assert not destination.exists()
