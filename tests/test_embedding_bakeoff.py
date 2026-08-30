import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest

import embedding_bakeoff as bakeoff


def test_embedding_bakeoff_local_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "fictional.db"
    with sqlite3.connect(database) as con:
        con.executescript("""
CREATE TABLE memory_items(id TEXT, memory_type TEXT, summary TEXT, embedding BLOB, merged_into TEXT);
CREATE TABLE triples(subject_id TEXT, predicate TEXT, valid_to TEXT);
""")
        for index in range(bakeoff.QUERY_COUNT):
            vector = [0.0] * bakeoff.DIMENSIONS
            vector[index] = 1.0
            con.execute(
                "INSERT INTO memory_items VALUES (?, 'episode', ?, ?, NULL)",
                (
                    f"id{index:03}",
                    f"Fictional event number {index}",
                    struct.pack("3072f", *vector),
                ),
            )

    rows = bakeoff.load_memories(bakeoff.require_disposable_db(database))
    assert len(rows) == bakeoff.QUERY_COUNT
    assert bakeoff.four_grams("one two three four five") == {
        ("one", "two", "three", "four"),
        ("two", "three", "four", "five"),
    }
    assert bakeoff.ranks(np.eye(2), np.eye(2), [{0}, {1}]).tolist() == [1, 1]
    assert bakeoff.exact_sign_pvalue(0, 3) == 0.25

    monkeypatch.setattr(bakeoff, "LIVE_DB_DIR", tmp_path)
    with pytest.raises(ValueError, match="Refusing live memU database path"):
        bakeoff.require_disposable_db(database)
