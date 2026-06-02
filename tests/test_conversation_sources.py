import json
import sqlite3
from pathlib import Path

import pytest

from app.services import conversation_sources


def _write_state_db(path: Path, rows: list[tuple[str, str, str, float]]) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute(
            "CREATE TABLE messages ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT, role TEXT, content TEXT, timestamp REAL)"
        )
        con.executemany(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            rows,
        )
        con.commit()
    finally:
        con.close()


def test_load_whatsapp_tail_group_collapses_multiple_sessions(tmp_path: Path) -> None:
    sessions_path = tmp_path / "sessions.json"
    state_db_path = tmp_path / "state.db"
    sessions_path.write_text(
        json.dumps(
            {
                "agent:main:whatsapp:group:18322935409-1579788049@g.us:114628432556258": {
                    "session_id": "s1",
                    "platform": "whatsapp",
                    "origin": {
                        "platform": "whatsapp",
                        "chat_type": "group",
                        "chat_id": "18322935409-1579788049@g.us",
                        "chat_name": "18322935409-1579788049",
                        "user_name": "Marcos",
                    },
                },
                "agent:main:whatsapp:group:18322935409-1579788049@g.us:140063262396533": {
                    "session_id": "s2",
                    "platform": "whatsapp",
                    "origin": {
                        "platform": "whatsapp",
                        "chat_type": "group",
                        "chat_id": "18322935409-1579788049@g.us",
                        "chat_name": "Familia",
                        "user_name": "Raquel",
                    },
                },
                "agent:main:whatsapp:group:18322935409-1579788049@g.us": {
                    "session_id": "s3",
                    "platform": "whatsapp",
                    "origin": {
                        "platform": "whatsapp",
                        "chat_type": "group",
                        "chat_id": "18322935409-1579788049@g.us",
                        "chat_name": "Familia",
                        "user_name": "",
                    },
                },
                "agent:main:whatsapp:group:18322935409-1579788049@g.us:222685531500721": {
                    "session_id": "s4",
                    "platform": "whatsapp",
                    "origin": {
                        "platform": "whatsapp",
                        "chat_type": "group",
                        "chat_id": "18322935409-1579788049@g.us",
                        "chat_name": "18322935409-1579788049",
                        "user_name": "Nico",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    _write_state_db(
        state_db_path,
        [
            ("s1", "user", "[Marcos] one", 100.0),
            ("s2", "user", "[Raquel] two", 101.0),
            ("s3", "assistant", "three", 102.0),
            ("s4", "user", "[Nico] four", 103.0),
        ],
    )

    rows = conversation_sources.load_whatsapp_tail(
        conversation_id="whatsapp:group:18322935409-1579788049@g.us",
        since_cursor=-1,
        recent_fallback_messages=0,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    )
    assert [row["content"] for row in rows] == ["[Marcos] one", "[Raquel] two", "three", "[Nico] four"]
    assert all(row["chat_name"] == "Familia" for row in rows)
    assert all(row["source_label"] == "whatsapp:group" for row in rows)


def test_load_whatsapp_tail_applies_floor_backfill(tmp_path: Path) -> None:
    sessions_path = tmp_path / "sessions.json"
    state_db_path = tmp_path / "state.db"
    sessions_path.write_text(
        json.dumps(
            {
                "agent:main:whatsapp:dm:15133278228": {
                    "session_id": "s1",
                    "platform": "whatsapp",
                    "origin": {
                        "platform": "whatsapp",
                        "chat_type": "dm",
                        "chat_id": "15133278228@s.whatsapp.net",
                        "chat_name": "Marcos",
                        "user_name": "Marcos",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    _write_state_db(
        state_db_path,
        [("s1", "user", f"msg-{i}", 100.0 + i) for i in range(10)],
    )

    rows = conversation_sources.load_whatsapp_tail(
        conversation_id="whatsapp:dm:15133278228",
        since_cursor=8,
        recent_fallback_messages=8,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    )
    assert [row["content"] for row in rows] == [f"msg-{i}" for i in range(2, 10)]


def test_load_whatsapp_tail_raises_when_mapping_missing(tmp_path: Path) -> None:
    sessions_path = tmp_path / "sessions.json"
    state_db_path = tmp_path / "state.db"
    sessions_path.write_text("{}", encoding="utf-8")
    _write_state_db(state_db_path, [])

    with pytest.raises(RuntimeError, match="no WhatsApp session mapping"):
        conversation_sources.load_whatsapp_tail(
            conversation_id="whatsapp:dm:15133278228",
            since_cursor=-1,
            recent_fallback_messages=0,
            sessions_index_path=sessions_path,
            state_db_path=state_db_path,
        )


def test_load_whatsapp_tail_orders_equal_timestamps_by_row_id(tmp_path: Path) -> None:
    sessions_path = tmp_path / "sessions.json"
    state_db_path = tmp_path / "state.db"
    sessions_path.write_text(
        json.dumps(
            {
                "agent:main:whatsapp:dm:15133278228": {
                    "session_id": "s1",
                    "platform": "whatsapp",
                    "origin": {
                        "platform": "whatsapp",
                        "chat_type": "dm",
                        "chat_id": "15133278228@s.whatsapp.net",
                        "chat_name": "Marcos",
                        "user_name": "Marcos",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    _write_state_db(
        state_db_path,
        [
            ("s1", "user", "first-insert", 100.0),
            ("s1", "assistant", "second-insert", 100.0),
        ],
    )

    rows = conversation_sources.load_whatsapp_tail(
        conversation_id="whatsapp:dm:15133278228",
        since_cursor=-1,
        recent_fallback_messages=0,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    )
    assert [row["content"] for row in rows] == ["first-insert", "second-insert"]


def test_load_whatsapp_tail_after_message_id_filters_by_row_id(tmp_path: Path) -> None:
    sessions_path = tmp_path / "sessions.json"
    state_db_path = tmp_path / "state.db"
    sessions_path.write_text(
        json.dumps(
            {
                "agent:main:whatsapp:dm:15133278228": {
                    "session_id": "s1",
                    "platform": "whatsapp",
                    "origin": {
                        "platform": "whatsapp",
                        "chat_type": "dm",
                        "chat_id": "15133278228@s.whatsapp.net",
                        "chat_name": "Marcos",
                        "user_name": "Marcos",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    _write_state_db(
        state_db_path,
        [
            ("s1", "user", "one", 100.0),
            ("s1", "assistant", "two", 101.0),
            ("s1", "user", "three", 102.0),
        ],
    )
    con = sqlite3.connect(state_db_path)
    try:
        second_id = int(
            con.execute(
                "SELECT id FROM messages WHERE content = ?",
                ("two",),
            ).fetchone()[0]
        )
    finally:
        con.close()

    rows = conversation_sources.load_whatsapp_tail_after_message_id(
        conversation_id="whatsapp:dm:15133278228",
        after_message_id=second_id,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    )
    assert [row["content"] for row in rows] == ["three"]
    assert rows[0]["source_conversation_index"] > second_id


def test_sillytavern_snapshot_round_trip_with_floor(tmp_path: Path) -> None:
    storage_dir = tmp_path / "resources"
    conversation_sources.persist_sillytavern_history_snapshot(
        storage_dir=storage_dir,
        user_id="u1",
        soul_id="Echo",
        conversation_id="integrity:chat-a",
        history=[{"role": "user", "name": "Marcos", "content": f"msg-{i}"} for i in range(10)],
        chat_name="Echo",
    )

    rows = conversation_sources.load_sillytavern_tail(
        storage_dir=storage_dir,
        user_id="u1",
        soul_id="Echo",
        conversation_id="integrity:chat-a",
        since_cursor=8,
        recent_fallback_messages=8,
    )
    assert [row["content"] for row in rows] == [f"msg-{i}" for i in range(2, 10)]
    assert all(row["source_label"] == "sillytavern" for row in rows)
    assert all(row["chat_name"] == "Echo" for row in rows)


def test_load_sillytavern_tail_raises_when_snapshot_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sillytavern snapshot missing"):
        conversation_sources.load_sillytavern_tail(
            storage_dir=tmp_path / "resources",
            user_id="u1",
            soul_id="Echo",
            conversation_id="integrity:missing",
            since_cursor=-1,
            recent_fallback_messages=0,
        )
