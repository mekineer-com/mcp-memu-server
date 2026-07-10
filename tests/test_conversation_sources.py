import json
import os
import sqlite3
from pathlib import Path

import pytest

from app.services import conversation_sources


def test_atomic_snapshot_blank_speakers_fall_back_to_scope_names(tmp_path: Path) -> None:
    conversation_sources.persist_atomic_history_snapshot(
        storage_dir=tmp_path,
        user_id="Marcos",
        soul_id="Siri",
        conversation_id="chat:atomic-one",
        chat_name="Atomic",
        history=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
    )

    rows = conversation_sources.load_atomic_tail(
        storage_dir=tmp_path,
        user_id="Marcos",
        soul_id="Siri",
        conversation_id="chat:atomic-one",
        since_cursor=-1,
        recent_fallback_messages=0,
    )

    assert [(row["role"], row["speaker"]) for row in rows] == [
        ("user", "Marcos"),
        ("assistant", "Siri"),
    ]


def test_hermes_base_defaults_to_channels_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("CHANNELS_HOME", str(tmp_path))

    assert conversation_sources._hermes_base() == tmp_path.resolve()


def _write_state_db(path: Path, rows: list[tuple]) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT, role TEXT, content TEXT, timestamp REAL, "
            "sender_id TEXT, sender_name TEXT, source_message_id TEXT)"
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


def _write_web_source_db(path: Path, *, messages: list[dict], contacts: list[dict] | None = None) -> None:
    con = sqlite3.connect(path)
    try:
        con.executescript(
            """
            CREATE TABLE whatsapp_messages (
              msg_key TEXT PRIMARY KEY,
              chat_id TEXT NOT NULL,
              chat_local_id TEXT NOT NULL,
              from_me INTEGER NOT NULL,
              timestamp INTEGER NOT NULL,
              type TEXT NOT NULL,
              body TEXT,
              author_id TEXT,
              author_local_id TEXT,
              from_id TEXT,
              from_local_id TEXT,
              to_id TEXT,
              to_local_id TEXT,
              has_media INTEGER NOT NULL DEFAULT 0,
              media_placeholder TEXT,
              ack INTEGER,
              revoked INTEGER NOT NULL DEFAULT 0,
              revoke_source TEXT,
              source TEXT NOT NULL,
              first_seen_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL,
              raw_json TEXT NOT NULL,
              reactions TEXT
            );
            CREATE TABLE whatsapp_contacts (
              contact_id TEXT PRIMARY KEY,
              contact_local_id TEXT NOT NULL,
              name TEXT,
              short_name TEXT,
              push_name TEXT,
              verified_name TEXT,
              is_me INTEGER NOT NULL DEFAULT 0,
              is_user INTEGER NOT NULL DEFAULT 0,
              is_group INTEGER NOT NULL DEFAULT 0,
              raw_json TEXT,
              updated_at INTEGER NOT NULL
            );
            """
        )
        for contact in contacts or []:
            con.execute(
                """
                INSERT INTO whatsapp_contacts (
                  contact_id, contact_local_id, name, short_name, push_name, verified_name, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    contact["contact_id"],
                    contact["contact_local_id"],
                    contact.get("name"),
                    contact.get("short_name"),
                    contact.get("push_name"),
                    contact.get("verified_name"),
                ),
            )
        for msg in messages:
            chat_id = msg.get("chat_id", "15133278228@c.us")
            from_id = msg.get("from_id", "15133278228@c.us")
            author_id = msg.get("author_id")
            timestamp = int(msg["timestamp"])
            con.execute(
                """
                INSERT INTO whatsapp_messages (
                  msg_key, chat_id, chat_local_id, from_me, timestamp, type, body,
                  author_id, author_local_id, from_id, from_local_id, to_id, to_local_id,
                  has_media, revoked, source, first_seen_at, updated_at, raw_json, reactions
                ) VALUES (?, ?, ?, ?, ?, 'chat', ?, ?, '', ?, '', null, '', 0, ?, 'test', ?, ?, '{}', ?)
                """,
                (
                    msg["msg_key"],
                    chat_id,
                    chat_id.split("@", 1)[0],
                    int(bool(msg.get("from_me"))),
                    timestamp,
                    msg.get("body", ""),
                    author_id,
                    from_id,
                    int(bool(msg.get("revoked"))),
                    timestamp,
                    timestamp,
                    msg.get("reactions"),
                ),
            )
        con.commit()
    finally:
        con.close()


def test_web_source_cursor_resolves_rebuild_and_missing_key_floor(tmp_path: Path) -> None:
    conversation_id = "whatsapp:dm:15133278228@c.us"
    original = tmp_path / "original.db"
    rebuilt = tmp_path / "rebuilt.db"
    _write_web_source_db(
        original,
        messages=[{"msg_key": "checkpoint", "timestamp": 100}],
    )
    _write_web_source_db(
        rebuilt,
        messages=[
            {"msg_key": "filler", "timestamp": 99},
            {"msg_key": "checkpoint", "timestamp": 100, "revoked": 1},
        ],
    )

    original_rowid, _ = conversation_sources.resolve_whatsapp_web_source_cursor(
        conversation_id, 1, "checkpoint", 100, original, rolling=False
    )
    rebuilt_rowid, floor = conversation_sources.resolve_whatsapp_web_source_cursor(
        conversation_id, original_rowid, "checkpoint", 100, rebuilt, rolling=False
    )
    assert (original_rowid, rebuilt_rowid, floor) == (1, 2, None)
    assert conversation_sources.resolve_whatsapp_web_source_cursor(
        conversation_id, rebuilt_rowid, "missing", 100, rebuilt, rolling=False
    ) == (-1, 100)
    assert conversation_sources.resolve_whatsapp_web_source_cursor(
        conversation_id, rebuilt_rowid, "missing", 100, rebuilt, rolling=True
    ) == (0, 100)
    assert conversation_sources.resolve_whatsapp_web_source_cursor(
        conversation_id, 7, None, None, rebuilt, rolling=False
    ) == (7, None)


def test_web_source_cursor_missing_db_does_not_create_empty_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError, match="web_source db missing"):
        conversation_sources.resolve_whatsapp_web_source_cursor(
            "whatsapp:dm:15133278228@c.us",
            1,
            "checkpoint",
            100,
            db_path,
            rolling=False,
        )
    assert not db_path.exists()


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
                        "chat_name": "Contact A",
                        "user_name": "Contact A",
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
            ("s1", "user", "from her", 102.0, "140063262396533@lid", "Contact A"),
        ],
    )

    rows = conversation_sources.load_whatsapp_tail(
        conversation_id="whatsapp:dm:140063262396533@lid",
        since_cursor=-1,
        recent_fallback_messages=0,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    )
    assert [row["speaker"] for row in rows] == ["Marcos", "", "Contact A"]


def test_load_whatsapp_tail_preserves_state_db_source_message_id(tmp_path: Path) -> None:
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
                        "chat_name": "Contact A",
                        "user_name": "Contact A",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    con = sqlite3.connect(state_db_path)
    try:
        con.execute(
            "CREATE TABLE messages ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT, role TEXT, content TEXT, timestamp REAL, "
            "sender_id TEXT, sender_name TEXT, source_message_id TEXT)"
        )
        con.execute(
            "INSERT INTO messages "
            "(session_id, role, content, timestamp, sender_id, sender_name, source_message_id) "
            "VALUES ('s1', 'user', 'hello', 100.0, NULL, 'Contact A', 'BAILEYS-1')"
        )
        con.commit()
    finally:
        con.close()

    rows = conversation_sources.load_whatsapp_tail(
        conversation_id="whatsapp:dm:140063262396533@lid",
        since_cursor=-1,
        recent_fallback_messages=0,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    )

    assert rows[0]["source_message_id"] == "BAILEYS-1"


def test_load_whatsapp_tail_max_messages_returns_newest_in_order(tmp_path: Path) -> None:
    sessions_path = tmp_path / "sessions.json"
    state_db_path = tmp_path / "state.db"
    sessions_path.write_text(
        json.dumps({
            "agent:main:whatsapp:dm:140063262396533@lid": {
                "session_id": "s1",
                "platform": "whatsapp",
                "origin": {
                    "platform": "whatsapp",
                    "chat_type": "dm",
                    "chat_id": "140063262396533@lid",
                    "chat_name": "Contact A",
                    "user_name": "Contact A",
                },
            }
        }),
        encoding="utf-8",
    )
    _write_state_db(
        state_db_path,
        [
            ("s1", "user", "one", 100.0),
            ("s1", "user", "two", 101.0),
            ("s1", "user", "three", 102.0),
        ],
    )

    rows = conversation_sources.load_whatsapp_tail(
        conversation_id="whatsapp:dm:140063262396533@lid",
        since_cursor=-1,
        recent_fallback_messages=0,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
        max_messages=2,
    )

    assert [row["content"] for row in rows] == ["two", "three"]


def test_pick_chat_name_dm_prefers_chat_name_over_shorter_user_name() -> None:
    assert (
        conversation_sources._pick_chat_name(
            [{"chat_name": "Trusted Contact", "user_name": "Marcos"}],
            "fallback",
            chat_type="dm",
        )
        == "Trusted Contact"
    )


def test_pick_chat_name_dm_uses_user_name_when_chat_name_is_numeric() -> None:
    assert (
        conversation_sources._pick_chat_name(
            [{"chat_name": "140063262396533", "user_name": "Trusted Contact"}],
            "fallback",
            chat_type="dm",
        )
        == "Trusted Contact"
    )


def test_load_whatsapp_web_source_tail_splits_soul_prefix_and_uses_contacts(tmp_path: Path) -> None:
    web_db = tmp_path / "web_source.db"
    _write_web_source_db(
        web_db,
        contacts=[
            {
                "contact_id": "15133278228@c.us",
                "contact_local_id": "15133278228",
                "name": "Contact A Chat",
                "short_name": "Contact A Chat",
            },
            {
                "contact_id": "140063262396533@lid",
                "contact_local_id": "140063262396533",
                "name": "Trusted Contact",
                "short_name": "Contact A",
            }
        ],
        messages=[
            {
                "msg_key": "old",
                "timestamp": 99,
                "body": "before active_since",
                "author_id": "140063262396533@lid",
                "from_id": "140063262396533@lid",
            },
            {
                "msg_key": "inbound",
                "timestamp": 100,
                "body": "hi Siri",
                "author_id": "140063262396533@lid",
                "from_id": "140063262396533@lid",
            },
            {
                "msg_key": "siri",
                "timestamp": 101,
                "body": "✦ *Siri*: hi back",
                "from_me": True,
                "from_id": "15133278228@c.us",
            },
        ],
    )

    rows = conversation_sources.load_whatsapp_web_source_tail(
        conversation_id="whatsapp:dm:15133278228",
        since_cursor=-1,
        recent_fallback_messages=0,
        soul_id="Siri",
        reply_prefix="✦ *Siri*: ",
        web_source_db_path=web_db,
        min_timestamp=100,
    )

    assert [(row.get("role"), row["speaker"], row["content"]) for row in rows] == [
        (None, "Contact A", "hi Siri"),
        ("assistant", "Siri", "hi back"),
    ]
    assert {row["chat_name"] for row in rows} == {"Contact A Chat"}
    assert [row["source_conversation_index"] for row in rows] == [2, 3]


def test_load_whatsapp_web_source_tail_after_rowid_uses_monotonic_cursor(tmp_path: Path) -> None:
    web_db = tmp_path / "web_source.db"
    _write_web_source_db(
        web_db,
        messages=[
            {"msg_key": "one", "timestamp": 100, "body": "one"},
            {"msg_key": "two", "timestamp": 101, "body": "two"},
        ],
    )

    rows = conversation_sources.load_whatsapp_web_source_tail_after_rowid(
        conversation_id="whatsapp:dm:15133278228",
        after_rowid=1,
        soul_id="Siri",
        reply_prefix="✦ *Siri*: ",
        web_source_db_path=web_db,
    )

    assert [row["content"] for row in rows] == ["two"]
    assert rows[0]["source_conversation_index"] == 2


def test_load_whatsapp_web_source_tail_max_messages_returns_newest_in_order(tmp_path: Path) -> None:
    web_db = tmp_path / "web_source.db"
    _write_web_source_db(
        web_db,
        messages=[
            {"msg_key": "one", "timestamp": 100, "body": "one"},
            {"msg_key": "two", "timestamp": 101, "body": "two"},
            {"msg_key": "three", "timestamp": 102, "body": "three"},
        ],
    )

    rows = conversation_sources.load_whatsapp_web_source_tail(
        conversation_id="whatsapp:dm:15133278228",
        since_cursor=-1,
        recent_fallback_messages=0,
        soul_id="Siri",
        reply_prefix="✦ *Siri*: ",
        web_source_db_path=web_db,
        max_messages=2,
    )

    assert [row["content"] for row in rows] == ["two", "three"]


def test_load_whatsapp_web_source_tail_does_not_require_matching_prefix(tmp_path: Path) -> None:
    web_db = tmp_path / "web_source.db"
    _write_web_source_db(
        web_db,
        messages=[
            {
                "msg_key": "from-me-no-prefix",
                "timestamp": 100,
                "body": "plain from me",
                "from_me": True,
            },
        ],
    )

    rows = conversation_sources.load_whatsapp_web_source_tail(
        conversation_id="whatsapp:dm:15133278228",
        since_cursor=-1,
        recent_fallback_messages=0,
        soul_id="Siri",
        reply_prefix="⚕ *Hermes Agent*",
        web_source_db_path=web_db,
    )

    assert [(row.get("role"), row["content"]) for row in rows] == [(None, "plain from me")]


def test_load_whatsapp_web_source_tail_uses_assistant_source_message_ids(tmp_path: Path) -> None:
    web_db = tmp_path / "web_source.db"
    _write_web_source_db(
        web_db,
        messages=[
            {
                "msg_key": "true_15133278228_c_us_SENT-ID",
                "timestamp": 100,
                "body": "plain sent reply",
                "from_me": True,
            },
        ],
    )

    rows = conversation_sources.load_whatsapp_web_source_tail(
        conversation_id="whatsapp:dm:15133278228",
        since_cursor=-1,
        recent_fallback_messages=0,
        soul_id="Siri",
        reply_prefix="",
        web_source_db_path=web_db,
        assistant_source_message_ids={"SENT-ID"},
    )

    assert [(row["role"], row["speaker"], row["content"]) for row in rows] == [
        ("assistant", "Siri", "plain sent reply")
    ]


def test_load_whatsapp_web_source_tail_does_not_substring_match_assistant_ids(tmp_path: Path) -> None:
    web_db = tmp_path / "web_source.db"
    _write_web_source_db(
        web_db,
        messages=[
            {
                "msg_key": "true_15133278228_c_us_prefix-SENT-ID-suffix",
                "timestamp": 100,
                "body": "human sent from linked device",
                "from_me": True,
            },
        ],
    )

    rows = conversation_sources.load_whatsapp_web_source_tail(
        conversation_id="whatsapp:dm:15133278228",
        since_cursor=-1,
        recent_fallback_messages=0,
        soul_id="Siri",
        reply_prefix="",
        web_source_db_path=web_db,
        assistant_source_message_ids={"SENT-ID"},
    )

    assert [(row.get("role"), row["speaker"], row["content"]) for row in rows] == [
        (None, "15133278228", "human sent from linked device")
    ]


def test_load_whatsapp_web_source_tail_filters_gateway_notices(tmp_path: Path) -> None:
    web_db = tmp_path / "web_source.db"
    _write_web_source_db(
        web_db,
        messages=[
            {
                "msg_key": "notice",
                "timestamp": 100,
                "body": "✦ *Siri*: ⚠️ Gateway shutting down — Your current task will be interrupted.",
                "from_me": True,
            },
            {
                "msg_key": "memu-error",
                "timestamp": 100.5,
                "body": "✦ *Siri*: memU turn failed: memU request failed: <urlopen error [Errno 111] Connection refused>",
                "from_me": True,
            },
            {"msg_key": "real", "timestamp": 101, "body": "real message"},
        ],
    )

    rows = conversation_sources.load_whatsapp_web_source_tail(
        conversation_id="whatsapp:dm:15133278228",
        since_cursor=-1,
        recent_fallback_messages=0,
        soul_id="Siri",
        reply_prefix="",
        web_source_db_path=web_db,
    )

    assert [row["content"] for row in rows] == ["real message"]


def test_load_whatsapp_web_source_tail_omits_revoked_rows(tmp_path: Path) -> None:
    web_db = tmp_path / "web_source.db"
    _write_web_source_db(
        web_db,
        messages=[
            {"msg_key": "deleted", "timestamp": 100, "body": "deleted message"},
            {"msg_key": "kept", "timestamp": 101, "body": "kept message"},
        ],
    )
    con = sqlite3.connect(web_db)
    try:
        con.execute("UPDATE whatsapp_messages SET revoked = 1 WHERE msg_key = 'deleted'")
        con.commit()
    finally:
        con.close()

    rows = conversation_sources.load_whatsapp_web_source_tail(
        conversation_id="whatsapp:dm:15133278228",
        since_cursor=-1,
        recent_fallback_messages=0,
        soul_id="Echo",
        reply_prefix="",
        web_source_db_path=web_db,
    )

    assert [row["content"] for row in rows] == ["kept message"]


def test_load_whatsapp_assistant_source_message_ids_reads_state_db(tmp_path: Path) -> None:
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
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    con = sqlite3.connect(state_db_path)
    try:
        con.execute(
            "CREATE TABLE messages ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, "
            "content TEXT, source_message_id TEXT)"
        )
        con.executemany(
            "INSERT INTO messages (session_id, role, content, source_message_id) VALUES (?, ?, ?, ?)",
            [
                ("s1", "assistant", "sent", "SENT-ID"),
                ("s1", "user", "inbound", "USER-ID"),
            ],
        )
        con.commit()
    finally:
        con.close()

    assert conversation_sources.load_whatsapp_assistant_source_message_ids(
        conversation_id="whatsapp:dm:15133278228",
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    ) == {"SENT-ID"}


def test_load_whatsapp_tail_filters_gateway_notices(tmp_path: Path) -> None:
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
            ("s1", "assistant", "⚠️ Gateway restarting — Your current task will be interrupted.", 100.0),
            ("s1", "assistant", "✦ *Siri*: ⚠️ Gateway shutting down — Your current task will be interrupted.", 100.5),
            ("s1", "assistant", "✦ *Siri*: memU turn failed: memU request failed: <urlopen error [Errno 111] Connection refused>", 100.75),
            ("s1", "user", "real message", 101.0),
        ],
    )

    rows = conversation_sources.load_whatsapp_tail(
        conversation_id="whatsapp:dm:15133278228",
        since_cursor=-1,
        recent_fallback_messages=0,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    )

    assert [row["content"] for row in rows] == ["real message"]


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
                        "chat_name": "Contact A",
                        "user_name": "Contact A",
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
        conversation_id="whatsapp:dm:140063262396533@lid",
        since_cursor=-1,
        recent_fallback_messages=0,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    )
    assert [row["speaker"] for row in rows] == ["Echo", ""]


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
                        "chat_name": "Household Group",
                        "user_name": "Contact A",
                    },
                },
                "agent:main:whatsapp:group:18322935409-1579788049@g.us": {
                    "session_id": "s3",
                    "platform": "whatsapp",
                    "origin": {
                        "platform": "whatsapp",
                        "chat_type": "group",
                        "chat_id": "18322935409-1579788049@g.us",
                        "chat_name": "Household Group",
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
            ("s2", "user", "[Contact A] two", 101.0),
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
    assert [row["content"] for row in rows] == ["[Marcos] one", "[Contact A] two", "three", "[Nico] four"]
    assert all(row["chat_name"] == "Household Group" for row in rows)
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


def test_load_whatsapp_tail_does_not_floor_without_new_rows(tmp_path: Path) -> None:
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
        since_cursor=9,
        recent_fallback_messages=8,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    )
    assert rows == []


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
        since_cursor=0,
        recent_fallback_messages=0,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    )
    assert [row["content"] for row in rows] == ["three"]
    assert rows[0]["source_conversation_index"] == 1


def test_load_whatsapp_tail_filters_by_min_timestamp_before_indexing(tmp_path: Path) -> None:
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
            ("s1", "user", "before", 99.0),
            ("s1", "assistant", "", 100.0),
            ("s1", "user", "at-cutoff", 100.0),
            ("s1", "user", "after", 101.0),
        ],
    )

    rows = conversation_sources.load_whatsapp_tail(
        conversation_id="whatsapp:dm:15133278228",
        since_cursor=-1,
        recent_fallback_messages=0,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
        min_timestamp=100.0,
    )
    assert [row["content"] for row in rows] == ["at-cutoff", "after"]
    assert [row["source_conversation_index"] for row in rows] == [0, 1]


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
    assert rows[0]["id"] == rows[0]["source_conversation_index"]
    assert rows[0]["source_conversation_index"] > second_id


def test_load_whatsapp_tail_after_message_id_filters_by_min_timestamp(tmp_path: Path) -> None:
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
            ("s1", "user", "before", 99.0),
            ("s1", "assistant", "", 100.0),
            ("s1", "user", "at-cutoff", 100.0),
            ("s1", "user", "after", 101.0),
        ],
    )

    rows = conversation_sources.load_whatsapp_tail_after_message_id(
        conversation_id="whatsapp:dm:15133278228",
        after_message_id=None,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
        min_timestamp=100.0,
    )
    assert [row["content"] for row in rows] == ["at-cutoff", "after"]


def test_load_soul_active_since_reads_hermes_state_db(tmp_path: Path) -> None:
    state_db_path = tmp_path / "state.db"
    con = sqlite3.connect(state_db_path)
    try:
        con.execute("CREATE TABLE souls (soul_id TEXT PRIMARY KEY, active_since REAL NOT NULL)")
        con.execute(
            "INSERT INTO souls (soul_id, active_since) VALUES (?, ?)",
            ("Siri", 100.0),
        )
        con.commit()
    finally:
        con.close()

    assert (
        conversation_sources.load_soul_active_since(
            soul_id="Siri",
            state_db_path=state_db_path,
        )
        == 100.0
    )
    assert conversation_sources.load_soul_active_since(soul_id="Echo", state_db_path=state_db_path) is None


def test_load_soul_active_since_rejects_invalid_existing_value(tmp_path: Path) -> None:
    state_db_path = tmp_path / "state.db"
    con = sqlite3.connect(state_db_path)
    try:
        con.execute("CREATE TABLE souls (soul_id TEXT PRIMARY KEY, active_since REAL NOT NULL)")
        con.execute(
            "INSERT INTO souls (soul_id, active_since) VALUES (?, ?)",
            ("Siri", "2026-06-03T00:00:00Z"),
        )
        con.commit()
    finally:
        con.close()

    with pytest.raises(RuntimeError, match="invalid active_since"):
        conversation_sources.load_soul_active_since(
            soul_id="Siri",
            state_db_path=state_db_path,
        )


def test_load_whatsapp_tail_dm_keeps_canonical_legacy_sessions_reachable(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sessions_path = sessions_dir / "sessions.json"
    state_db_path = tmp_path / "state.db"
    sessions_path.write_text(
        json.dumps(
            {
                "agent:main:whatsapp:dm:114628432556258@lid": {
                    "session_id": "s1",
                    "platform": "whatsapp",
                    "origin": {
                        "platform": "whatsapp",
                        "chat_type": "dm",
                        "chat_id": "114628432556258@lid",
                        "chat_name": "Marcos",
                        "user_name": "Marcos",
                    },
                },
                "agent:main:whatsapp:dm:114628432556258@lid:legacy:s2": {
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
        conversation_id="whatsapp:dm:114628432556258@lid",
        since_cursor=-1,
        recent_fallback_messages=0,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    )
    assert [row["content"] for row in rows] == ["phone-side", "lid-side"]


def test_load_whatsapp_tail_dm_requires_canonical_lid_id(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sessions_path = sessions_dir / "sessions.json"
    state_db_path = tmp_path / "state.db"
    sessions_path.write_text(
        json.dumps(
            {
                "agent:main:whatsapp:dm:247789598601266@lid": {
                    "session_id": "s_lid_only",
                    "platform": "whatsapp",
                    "origin": {
                        "platform": "whatsapp",
                        "chat_type": "dm",
                        "chat_id": "247789598601266@lid",
                        "chat_name": "Contact B",
                        "user_name": "Contact B",
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

    with pytest.raises(RuntimeError, match="no WhatsApp session mapping"):
        conversation_sources.load_whatsapp_tail(
            conversation_id="whatsapp:dm:447879696252@s.whatsapp.net",
            since_cursor=-1,
            recent_fallback_messages=0,
            sessions_index_path=sessions_path,
            state_db_path=state_db_path,
        )

    rows = conversation_sources.load_whatsapp_tail(
        conversation_id="whatsapp:dm:247789598601266@lid",
        since_cursor=-1,
        recent_fallback_messages=0,
        sessions_index_path=sessions_path,
        state_db_path=state_db_path,
    )
    assert [row["content"] for row in rows] == ["lid-only message", "reply from soul"]


def test_load_whatsapp_tail_dm_does_not_match_same_local_different_domain(tmp_path: Path) -> None:
    sessions_path = tmp_path / "sessions.json"
    state_db_path = tmp_path / "state.db"
    sessions_path.write_text(
        json.dumps(
            {
                "agent:main:whatsapp:dm:12345@lid": {
                    "session_id": "s_lid",
                    "platform": "whatsapp",
                    "origin": {
                        "platform": "whatsapp",
                        "chat_type": "dm",
                        "chat_id": "12345@lid",
                        "chat_name": "LID Contact",
                        "user_name": "LID Contact",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    _write_state_db(state_db_path, [("s_lid", "user", "lid message", 100.0)])

    with pytest.raises(RuntimeError, match="no WhatsApp session mapping"):
        conversation_sources.load_whatsapp_tail(
            conversation_id="whatsapp:dm:12345@s.whatsapp.net",
            since_cursor=-1,
            recent_fallback_messages=0,
            sessions_index_path=sessions_path,
            state_db_path=state_db_path,
        )


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


def test_sillytavern_snapshot_write_is_atomic_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage_dir = tmp_path / "resources"
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def _record_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        replacements.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(conversation_sources.os, "replace", _record_replace)

    conversation_sources.persist_sillytavern_history_snapshot(
        storage_dir=storage_dir,
        user_id="u1",
        soul_id="Echo",
        conversation_id="integrity:chat-a",
        history=[{"role": "user", "name": "Marcos", "content": "hello"}],
        chat_name="Echo",
    )

    assert len(replacements) == 1
    src, dst = replacements[0]
    assert src.parent == dst.parent
    assert src.name.startswith(".latest_history.json.")
    assert dst.name == "latest_history.json"
    assert not src.exists()


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


def test_messages_source_message_id_select_raises_when_column_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    con = sqlite3.connect(db_path)
    try:
        con.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT)")
        con.commit()
        with pytest.raises(RuntimeError, match="missing required column messages.source_message_id"):
            conversation_sources._messages_source_message_id_select(con, db_path)
    finally:
        con.close()


def test_expand_session_ids_propagates_when_parent_session_id_column_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    con = sqlite3.connect(db_path)
    try:
        # sessions table exists but lacks parent_session_id column
        con.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        con.execute("INSERT INTO sessions (id) VALUES ('s1')")
        con.commit()
    finally:
        con.close()
    with pytest.raises(sqlite3.OperationalError, match="no such column"):
        conversation_sources._expand_session_ids_with_lineage(db_path, ["s1"])


def test_load_whatsapp_web_source_tail_raises_on_initial_read_with_no_matching_rows(tmp_path: Path) -> None:
    web_db = tmp_path / "web_source.db"
    _write_web_source_db(web_db, messages=[])

    with pytest.raises(RuntimeError, match="canonical WhatsApp ID mismatch"):
        conversation_sources.load_whatsapp_web_source_tail(
            conversation_id="whatsapp:dm:99999999999",
            since_cursor=-1,
            recent_fallback_messages=0,
            soul_id="Siri",
            reply_prefix="",
            web_source_db_path=web_db,
        )


def test_load_whatsapp_web_source_tail_no_error_on_initial_read_with_messages_filtered_by_timestamp(
    tmp_path: Path,
) -> None:
    web_db = tmp_path / "web_source.db"
    _write_web_source_db(
        web_db,
        messages=[
            {"msg_key": "old", "timestamp": 50, "body": "before cutoff"},
        ],
    )

    rows = conversation_sources.load_whatsapp_web_source_tail(
        conversation_id="whatsapp:dm:15133278228",
        since_cursor=-1,
        recent_fallback_messages=0,
        soul_id="Siri",
        reply_prefix="",
        web_source_db_path=web_db,
        min_timestamp=100.0,
    )
    assert rows == []


def test_load_whatsapp_web_source_tail_after_rowid_no_error_when_no_new_rows(tmp_path: Path) -> None:
    web_db = tmp_path / "web_source.db"
    _write_web_source_db(
        web_db,
        messages=[
            {"msg_key": "one", "timestamp": 100, "body": "already seen"},
        ],
    )

    rows = conversation_sources.load_whatsapp_web_source_tail_after_rowid(
        conversation_id="whatsapp:dm:15133278228",
        after_rowid=999,
        soul_id="Siri",
        reply_prefix="",
        web_source_db_path=web_db,
    )
    assert rows == []


def test_web_source_reaction_rendered_into_content(tmp_path: Path) -> None:
    web_db = tmp_path / "web_source.db"
    _write_web_source_db(
        web_db,
        messages=[
            {
                "msg_key": "m1",
                "timestamp": 100,
                "body": "hello",
                "reactions": '{"15133278228": "❤️"}',
            }
        ],
        contacts=[
            {
                "contact_id": "15133278228@c.us",
                "contact_local_id": "15133278228",
                "name": "Marcos",
            }
        ],
    )

    rows = conversation_sources.load_whatsapp_web_source_tail(
        conversation_id="whatsapp:dm:15133278228",
        since_cursor=-1,
        recent_fallback_messages=0,
        soul_id="Siri",
        reply_prefix="",
        web_source_db_path=web_db,
    )
    assert len(rows) == 1
    assert rows[0]["content"] == "hello [reacted ❤️ — Marcos]"
    assert "role" not in rows[0]


def test_web_source_multiple_reactors(tmp_path: Path) -> None:
    web_db = tmp_path / "web_source.db"
    _write_web_source_db(
        web_db,
        messages=[
            {
                "msg_key": "m1",
                "timestamp": 100,
                "body": "hey",
                "reactions": '{"aaa": "❤️", "bbb": "😂"}',
            }
        ],
        contacts=[
            {"contact_id": "aaa@c.us", "contact_local_id": "aaa", "name": "Alice"},
            {"contact_id": "bbb@c.us", "contact_local_id": "bbb", "name": "Bob"},
        ],
    )

    rows = conversation_sources.load_whatsapp_web_source_tail(
        conversation_id="whatsapp:dm:15133278228",
        since_cursor=-1,
        recent_fallback_messages=0,
        soul_id="Siri",
        reply_prefix="",
        web_source_db_path=web_db,
    )
    assert len(rows) == 1
    content = rows[0]["content"]
    assert content.startswith("hey [reacted ")
    assert "❤️ — Alice" in content
    assert "😂 — Bob" in content


def test_web_source_outdated_schema_raises(tmp_path: Path) -> None:
    web_db = tmp_path / "web_source.db"
    # Create DB without the reactions column (old schema)
    con = sqlite3.connect(str(web_db))
    try:
        con.executescript(
            """
            CREATE TABLE whatsapp_messages (
              msg_key TEXT PRIMARY KEY,
              chat_id TEXT NOT NULL,
              chat_local_id TEXT NOT NULL,
              from_me INTEGER NOT NULL,
              timestamp INTEGER NOT NULL,
              type TEXT NOT NULL,
              body TEXT,
              author_id TEXT,
              author_local_id TEXT,
              from_id TEXT,
              from_local_id TEXT,
              to_id TEXT,
              to_local_id TEXT,
              has_media INTEGER NOT NULL DEFAULT 0,
              media_placeholder TEXT,
              ack INTEGER,
              revoked INTEGER NOT NULL DEFAULT 0,
              revoke_source TEXT,
              source TEXT NOT NULL,
              first_seen_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL,
              raw_json TEXT NOT NULL
            );
            CREATE TABLE whatsapp_contacts (
              contact_id TEXT PRIMARY KEY,
              contact_local_id TEXT NOT NULL,
              name TEXT,
              short_name TEXT,
              push_name TEXT,
              verified_name TEXT,
              is_me INTEGER NOT NULL DEFAULT 0,
              is_user INTEGER NOT NULL DEFAULT 0,
              is_group INTEGER NOT NULL DEFAULT 0,
              raw_json TEXT,
              updated_at INTEGER NOT NULL
            );
            INSERT INTO whatsapp_messages (
              msg_key, chat_id, chat_local_id, from_me, timestamp, type, source,
              first_seen_at, updated_at, raw_json
            ) VALUES ('k1', '15133278228@c.us', '15133278228', 0, 100, 'chat', 'test', 100, 100, '{}');
            """
        )
        con.commit()
    finally:
        con.close()

    with pytest.raises(RuntimeError, match="reactions column missing"):
        conversation_sources.load_whatsapp_web_source_tail(
            conversation_id="whatsapp:dm:15133278228",
            since_cursor=-1,
            recent_fallback_messages=0,
            soul_id="Siri",
            reply_prefix="",
            web_source_db_path=web_db,
        )
