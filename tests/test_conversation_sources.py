import json
import sqlite3
from pathlib import Path

import pytest

from app.services import conversation_sources


def _write_state_db(path: Path, rows: list[tuple]) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute(
            "CREATE TABLE messages ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT, role TEXT, content TEXT, timestamp REAL, sender_id TEXT, sender_name TEXT)"
        )
        normalized_rows: list[tuple[str, str, str, float, str | None, str | None]] = []
        for row in rows:
            if len(row) == 4:
                session_id, role, content, timestamp = row
                normalized_rows.append((session_id, role, content, timestamp, None, None))
                continue
            if len(row) == 6:
                session_id, role, content, timestamp, sender_id, sender_name = row
                normalized_rows.append((session_id, role, content, timestamp, sender_id, sender_name))
                continue
            raise ValueError(f"unexpected row shape for state db fixture: {row!r}")
        con.executemany(
            "INSERT INTO messages (session_id, role, content, timestamp, sender_id, sender_name) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            normalized_rows,
        )
        con.commit()
    finally:
        con.close()


def _write_sessions_table(path: Path, rows: list[tuple[str, str | None]]) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            "id TEXT PRIMARY KEY, parent_session_id TEXT)"
        )
        con.executemany(
            "INSERT OR REPLACE INTO sessions (id, parent_session_id) VALUES (?, ?)",
            rows,
        )
        con.commit()
    finally:
        con.close()


def test_load_whatsapp_tail_prefers_per_message_sender_fields(tmp_path: Path) -> None:
    sessions_path = tmp_path / "sessions.json"
    state_db_path = tmp_path / "state.db"
    sessions_path.write_text(
        json.dumps(
            {
                "agent:main:whatsapp:dm:140063262396533@lid": {
                    "session_id": "s1",
                    "platform": "whatsapp",
                    "origin": {
                        "platform": "whatsapp",
                        "chat_type": "dm",
                        "chat_id": "140063262396533@lid",
                        "chat_name": "Raquel",
                        "user_name": "Raquel",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    _write_state_db(
        state_db_path,
        [
            ("s1", "user", "from me", 100.0, "15133278228@s.whatsapp.net", "Marcos"),
            ("s1", "assistant", "reply", 101.0, None, None),
            ("s1", "user", "from her", 102.0, "140063262396533@lid", "Raquel"),
        ],
    )

    rows = conversation_sources.load_whatsapp_tail(
        conversation_id="whatsapp:dm:140063262396533",
        since_cursor=-1,
        recent_fallback_messages=0,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    )
    assert [row["speaker"] for row in rows] == ["Marcos", "", "Raquel"]


def test_load_whatsapp_tail_includes_parent_session_lineage(tmp_path: Path) -> None:
    sessions_path = tmp_path / "sessions.json"
    state_db_path = tmp_path / "state.db"
    sessions_path.write_text(
        json.dumps(
            {
                "agent:main:whatsapp:dm:15133278228": {
                    "session_id": "s2",
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
    _write_sessions_table(
        state_db_path,
        [
            ("s1", None),
            ("s2", "s1"),
        ],
    )
    _write_state_db(
        state_db_path,
        [
            ("s1", "user", "older", 100.0, "15133278228@s.whatsapp.net", "Marcos"),
            ("s2", "user", "newer", 101.0, "15133278228@s.whatsapp.net", "Marcos"),
        ],
    )

    rows = conversation_sources.load_whatsapp_tail(
        conversation_id="whatsapp:dm:15133278228",
        since_cursor=-1,
        recent_fallback_messages=0,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    )
    assert [row["content"] for row in rows] == ["older", "newer"]
    assert [row["source_conversation_index"] for row in rows] == [-1, 0]


def test_load_whatsapp_tail_parent_lineage_does_not_shift_existing_cursor(tmp_path: Path) -> None:
    sessions_path = tmp_path / "sessions.json"
    state_db_path = tmp_path / "state.db"
    sessions_path.write_text(
        json.dumps(
            {
                "agent:main:whatsapp:dm:15133278228": {
                    "session_id": "s2",
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
    _write_sessions_table(
        state_db_path,
        [
            ("s1", None),
            ("s2", "s1"),
        ],
    )
    _write_state_db(
        state_db_path,
        [
            ("s1", "user", "older parent", 100.0, "15133278228@s.whatsapp.net", "Marcos"),
            ("s2", "user", "already memorized child", 101.0, "15133278228@s.whatsapp.net", "Marcos"),
            ("s2", "user", "new child", 102.0, "15133278228@s.whatsapp.net", "Marcos"),
        ],
    )

    rows = conversation_sources.load_whatsapp_tail(
        conversation_id="whatsapp:dm:15133278228",
        since_cursor=0,
        recent_fallback_messages=0,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    )
    assert [(row["content"], row["source_conversation_index"]) for row in rows] == [
        ("new child", 1)
    ]


def test_load_whatsapp_tail_preserves_assistant_sender_name_when_present(tmp_path: Path) -> None:
    sessions_path = tmp_path / "sessions.json"
    state_db_path = tmp_path / "state.db"
    sessions_path.write_text(
        json.dumps(
            {
                "agent:main:whatsapp:dm:140063262396533@lid": {
                    "session_id": "s1",
                    "platform": "whatsapp",
                    "origin": {
                        "platform": "whatsapp",
                        "chat_type": "dm",
                        "chat_id": "140063262396533@lid",
                        "chat_name": "Raquel",
                        "user_name": "Raquel",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    _write_state_db(
        state_db_path,
        [
            ("s1", "assistant", "old soul reply", 101.0, None, "Echo"),
            ("s1", "assistant", "new soul reply", 102.0, None, None),
        ],
    )

    rows = conversation_sources.load_whatsapp_tail(
        conversation_id="whatsapp:dm:140063262396533",
        since_cursor=-1,
        recent_fallback_messages=0,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    )
    assert [row["speaker"] for row in rows] == ["Echo", ""]


def _write_lid_mapping_files(base_dir: Path, *, phone_local: str, lid_local: str) -> None:
    session_dir = base_dir / "whatsapp" / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / f"lid-mapping-{phone_local}.json").write_text(
        json.dumps(lid_local),
        encoding="utf-8",
    )
    (session_dir / f"lid-mapping-{lid_local}_reverse.json").write_text(
        json.dumps(phone_local),
        encoding="utf-8",
    )


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


def test_load_whatsapp_tail_since_cursor_handles_sparse_indices(tmp_path: Path) -> None:
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
            ("s1", "assistant", "", 101.0),
            ("s1", "user", "three", 102.0),
        ],
    )

    rows = conversation_sources.load_whatsapp_tail(
        conversation_id="whatsapp:dm:15133278228",
        since_cursor=1,
        recent_fallback_messages=0,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    )
    assert [row["content"] for row in rows] == ["three"]


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


def test_load_whatsapp_tail_dm_unions_phone_and_lid_sessions(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sessions_path = sessions_dir / "sessions.json"
    state_db_path = tmp_path / "state.db"
    _write_lid_mapping_files(tmp_path, phone_local="15133278228", lid_local="114628432556258")
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
                },
                "agent:main:whatsapp:dm:114628432556258": {
                    "session_id": "s2",
                    "platform": "whatsapp",
                    "origin": {
                        "platform": "whatsapp",
                        "chat_type": "dm",
                        "chat_id": "114628432556258@lid",
                        "chat_name": "Marcos",
                        "user_name": "Marcos",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    _write_state_db(
        state_db_path,
        [
            ("s1", "user", "phone-side", 100.0),
            ("s2", "assistant", "lid-side", 101.0),
        ],
    )

    rows = conversation_sources.load_whatsapp_tail(
        conversation_id="whatsapp:dm:15133278228",
        since_cursor=-1,
        recent_fallback_messages=0,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    )
    assert [row["content"] for row in rows] == ["phone-side", "lid-side"]


def test_load_whatsapp_tail_dm_phone_resolves_lid_only_session(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sessions_path = sessions_dir / "sessions.json"
    state_db_path = tmp_path / "state.db"
    _write_lid_mapping_files(tmp_path, phone_local="447879696252", lid_local="247789598601266")
    sessions_path.write_text(
        json.dumps(
            {
                "agent:main:whatsapp:dm:247789598601266": {
                    "session_id": "s_lid_only",
                    "platform": "whatsapp",
                    "origin": {
                        "platform": "whatsapp",
                        "chat_type": "dm",
                        "chat_id": "247789598601266@lid",
                        "chat_name": "Liz Kalverda",
                        "user_name": "Liz Kalverda",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    _write_state_db(
        state_db_path,
        [
            ("s_lid_only", "user", "lid-only message", 100.0),
            ("s_lid_only", "assistant", "reply from soul", 101.0),
        ],
    )

    rows = conversation_sources.load_whatsapp_tail(
        conversation_id="whatsapp:dm:447879696252",
        since_cursor=-1,
        recent_fallback_messages=0,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    )
    assert [row["content"] for row in rows] == ["lid-only message", "reply from soul"]


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


def test_load_sillytavern_tail_since_cursor_handles_sparse_indices(tmp_path: Path) -> None:
    storage_dir = tmp_path / "resources"
    conversation_sources.persist_sillytavern_history_snapshot(
        storage_dir=storage_dir,
        user_id="u1",
        soul_id="Echo",
        conversation_id="integrity:chat-a",
        history=[
            {"role": "user", "name": "Marcos", "content": "one"},
            {"role": "assistant", "name": "Echo", "content": ""},
            {"role": "user", "name": "Marcos", "content": "three"},
        ],
        chat_name="Echo",
    )

    rows = conversation_sources.load_sillytavern_tail(
        storage_dir=storage_dir,
        user_id="u1",
        soul_id="Echo",
        conversation_id="integrity:chat-a",
        since_cursor=1,
        recent_fallback_messages=0,
    )
    assert [row["content"] for row in rows] == ["three"]


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
