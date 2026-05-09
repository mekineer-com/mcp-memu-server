import json
import sqlite3

from app.db import sqlite_ensure_conversation_state_schema
from app.services import message_log


def _con() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    sqlite_ensure_conversation_state_schema(con)
    return con


def test_append_messages_cumulative_history_fast_path() -> None:
    con = _con()
    try:
        added = message_log.append_messages(
            con,
            "c1",
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
            ],
        )
        assert added == 2

        added = message_log.append_messages(
            con,
            "c1",
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ],
        )
        assert added == 1

        rows = message_log.read_tail(con, "c1", after_cursor=0)
        assert [r["content"] for r in rows] == ["one", "two", "three"]
    finally:
        con.close()


def test_append_messages_incremental_payloads_append_new_rows() -> None:
    con = _con()
    try:
        assert message_log.append_messages(con, "c2", [{"role": "user", "content": "one"}]) == 1
        # New request contains only latest message (incremental mode).
        assert message_log.append_messages(con, "c2", [{"role": "assistant", "content": "two"}]) == 1
        rows = message_log.read_tail(con, "c2", after_cursor=0)
        assert [r["content"] for r in rows] == ["one", "two"]
    finally:
        con.close()


def test_append_messages_incremental_overlap_only_appends_suffix() -> None:
    con = _con()
    try:
        assert message_log.append_messages(
            con,
            "c3",
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
            ],
        ) == 2

        # Incoming payload overlaps the existing tail on "two"; only "three" should append.
        assert message_log.append_messages(
            con,
            "c3",
            [
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ],
        ) == 1

        rows = message_log.read_tail(con, "c3", after_cursor=0)
        assert [r["content"] for r in rows] == ["one", "two", "three"]
    finally:
        con.close()


def test_read_all_tails_falls_back_to_recent_when_unmemorized_tail_empty() -> None:
    con = _con()
    try:
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor) VALUES (?, ?)",
            ("current", 0),
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor, last_memorize_at) VALUES (?, ?, ?)",
            ("c4", 99, "2026-05-08T00:00:00+00:00"),
        )
        assert message_log.append_messages(
            con,
            "c4",
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ],
        ) == 3

        merged = message_log.read_all_tails(con, exclude_conversation_id="current")
        assert [m["content"] for m in merged] == ["one", "two", "three"]
    finally:
        con.close()


def test_read_all_tails_prefers_unmemorized_tail_over_recent_fallback() -> None:
    con = _con()
    try:
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor) VALUES (?, ?)",
            ("current", 0),
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor, last_memorize_at) VALUES (?, ?, ?)",
            ("c5", 1, "2026-05-08T00:00:00+00:00"),
        )
        assert message_log.append_messages(
            con,
            "c5",
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
                {"role": "assistant", "content": "four"},
            ],
        ) == 4

        merged = message_log.read_all_tails(con, exclude_conversation_id="current")
        assert [m["content"] for m in merged] == ["three", "four"]
    finally:
        con.close()


def test_read_all_tails_never_memorized_includes_first_message_without_fallback() -> None:
    con = _con()
    try:
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor) VALUES (?, ?)",
            ("current", 0),
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor) VALUES (?, ?)",
            ("c6", 0),
        )
        assert message_log.append_messages(
            con,
            "c6",
            [{"role": "user", "content": "first"}],
        ) == 1

        merged = message_log.read_all_tails(
            con,
            exclude_conversation_id="current",
            recent_fallback_per_conversation=0,
        )
        assert [m["content"] for m in merged] == ["first"]
    finally:
        con.close()


def test_format_merged_history_reuses_known_speaker_within_same_conversation() -> None:
    rendered = message_log.format_merged_history(
        [
            {
                "conversation_id": "whatsapp:dm:abc",
                "source_label": "whatsapp:dm",
                "role": "user",
                "speaker": "Marcos",
                "content": "first",
                "received_at": "2026-05-08T11:00:00+00:00",
            },
            {
                "conversation_id": "whatsapp:dm:abc",
                "source_label": "whatsapp:dm",
                "role": "user",
                "speaker": None,
                "content": "second",
                "received_at": "2026-05-08T11:00:01+00:00",
            },
        ]
    )

    assert "[whatsapp:dm] [Marcos]: first" in rendered
    assert "[whatsapp:dm] [Marcos]: second" in rendered


def test_conversation_aliases_includes_creds_self_lid_phone_pair(tmp_path, monkeypatch) -> None:
    session_dir = tmp_path / ".hermes" / "whatsapp" / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "creds.json").write_text(
        json.dumps(
            {
                "me": {
                    "id": "15133278228:13@s.whatsapp.net",
                    "lid": "114628432556258:13@lid",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    aliases = message_log.conversation_aliases("whatsapp:dm:114628432556258")
    assert "whatsapp:dm:114628432556258" in aliases
    assert "whatsapp:dm:15133278228" in aliases


def test_read_recent_for_conversation_ids_merges_alias_rows() -> None:
    con = _con()
    try:
        assert message_log.append_messages(
            con,
            "whatsapp:dm:114628432556258",
            [{"role": "user", "name": "Marcos", "content": "older-lid"}],
            source_label="whatsapp:dm",
        ) == 1
        assert message_log.append_messages(
            con,
            "whatsapp:dm:15133278228",
            [{"role": "user", "name": "Marcos", "content": "newer-phone"}],
            source_label="whatsapp:dm",
        ) == 1
        merged = message_log.read_recent_for_conversation_ids(
            con,
            ["whatsapp:dm:114628432556258", "whatsapp:dm:15133278228"],
            limit=8,
        )
        assert [row["content"] for row in merged] == ["older-lid", "newer-phone"]
    finally:
        con.close()
