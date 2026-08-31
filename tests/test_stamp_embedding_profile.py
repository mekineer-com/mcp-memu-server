import sqlite3

import pytest

import stamp_embedding_profile as profile


def _database(path, embedding: bytes = b"\0" * profile.VECTOR_BYTES) -> None:
    with sqlite3.connect(path) as conn:
        for table in profile.TABLES:
            conn.execute(f"CREATE TABLE {table}(id TEXT PRIMARY KEY, embedding BLOB)")
            conn.execute(f"INSERT INTO {table} VALUES ('{table}', ?)", (embedding,))


def test_stamp_is_dry_run_then_backed_up_and_idempotent(tmp_path) -> None:
    database = tmp_path / "fictional.db"
    _database(database)

    assert profile.stamp(database, apply=False)["status"] == "ready"
    with sqlite3.connect(database) as conn:
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'embedding_profile'"
        ).fetchone() is None

    applied = profile.stamp(database, apply=True)
    assert applied["status"] == "stamped"
    assert applied["backup"]
    with sqlite3.connect(applied["backup"]) as conn:
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'embedding_profile'"
        ).fetchone() is None
    assert profile.stamp(database, apply=True)["status"] == "already_stamped"


def test_stamp_rejects_wrong_width_before_backup(tmp_path) -> None:
    database = tmp_path / "fictional.db"
    _database(database, b"wrong")

    with pytest.raises(ValueError, match="non-canonical embeddings"):
        profile.stamp(database, apply=True)
    assert list(tmp_path.glob("*-pre-profile-*")) == []
