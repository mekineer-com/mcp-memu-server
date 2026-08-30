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
CREATE TABLE memory_items(id TEXT, memory_type TEXT, summary TEXT, embedding BLOB, merged_into TEXT, confidence REAL);
CREATE TABLE triples(subject_id TEXT, predicate TEXT, valid_to TEXT);
""")
        for index in range(bakeoff.QUERY_COUNT + 1):
            vector = [0.0] * bakeoff.DIMENSIONS
            vector[index] = 1.0
            con.execute(
                "INSERT INTO memory_items VALUES (?, 'episode', ?, ?, NULL, ?)",
                (
                    f"id{index:03}",
                    f"Fictional event number {index}",
                    struct.pack("3072f", *vector),
                    0.4 if index == bakeoff.QUERY_COUNT else 0.9,
                ),
            )

    rows, excluded = bakeoff.load_memories(bakeoff.require_disposable_db(database))
    assert len(rows) == bakeoff.QUERY_COUNT
    assert excluded == 1
    assert bakeoff.four_grams("one two three four five") == {
        ("one", "two", "three", "four"),
        ("two", "three", "four", "five"),
    }
    assert bakeoff.ranks(np.eye(2), np.eye(2), [{0}, {1}]).tolist() == [1, 1]
    assert bakeoff.exact_sign_pvalue(0, 3) == 0.25
    with pytest.raises(ValueError, match="returned 1 vectors for 2 inputs"):
        bakeoff.checked_vectors([[0.0] * bakeoff.DIMENSIONS], 2, "Fictional")

    monkeypatch.setattr(bakeoff, "LIVE_DB_DIR", tmp_path)
    with pytest.raises(ValueError, match="Refusing live memU database path"):
        bakeoff.require_disposable_db(database)
