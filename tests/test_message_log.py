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


def test_append_messages_shared_group_parses_sender_prefix_into_speaker() -> None:
    con = _con()
    try:
        assert message_log.append_messages(
            con,
            "whatsapp:group:18322935409-1579788049@g.us",
            [{"role": "user", "name": "Marcos", "content": "[Raquel] Going to the gym now."}],
            source_label="whatsapp:group",
        ) == 1

        rows = message_log.read_tail(con, "whatsapp:group:18322935409-1579788049@g.us", after_cursor=0)
        assert len(rows) == 1
        assert rows[0]["speaker"] == "Raquel"
        assert rows[0]["content"] == "Going to the gym now."
    finally:
        con.close()


def test_append_messages_dm_keeps_bracket_prefix_in_plain_content() -> None:
    con = _con()
    try:
        assert message_log.append_messages(
            con,
            "whatsapp:dm:15133278228",
            [{"role": "user", "name": "Marcos", "content": "[Raquel] Going to the gym now."}],
            source_label="whatsapp:dm",
        ) == 1

        rows = message_log.read_tail(con, "whatsapp:dm:15133278228", after_cursor=0)
        assert len(rows) == 1
        assert rows[0]["speaker"] == "Marcos"
        assert rows[0]["content"] == "[Raquel] Going to the gym now."
    finally:
        con.close()


def test_append_messages_group_overlap_ignores_speaker_drift_for_prefixed_rows() -> None:
    con = _con()
    try:
        cid = "whatsapp:group:18322935409-1579788049@g.us"
        assert message_log.append_messages(
            con,
            cid,
            [{"role": "user", "content": "[Raquel] Going to the gym now."}],
            source_label="whatsapp:group",
        ) == 1

        # Same semantic row but with a fallback speaker from an older payload.
        assert message_log.append_messages(
            con,
            cid,
            [{"role": "user", "name": "Marcos", "content": "[Raquel] Going to the gym now."}],
            source_label="whatsapp:group",
        ) == 0
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


def test_read_all_tails_caps_per_conversation_to_preserve_cross_chat_mix() -> None:
    con = _con()
    try:
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor) VALUES (?, ?)",
            ("current", 0),
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor) VALUES (?, ?)",
            ("whatsapp:dm:dominant", 0),
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor) VALUES (?, ?)",
            ("whatsapp:group:small@g.us", 0),
        )

        many_rows = [{"role": "user", "content": f"dominant-{i}"} for i in range(20)]
        assert message_log.append_messages(
            con,
            "whatsapp:dm:dominant",
            many_rows,
            source_label="whatsapp:dm",
        ) == 20
        assert message_log.append_messages(
            con,
            "whatsapp:group:small@g.us",
            [{"role": "user", "content": "small-1"}, {"role": "user", "content": "small-2"}],
            source_label="whatsapp:group",
        ) == 2

        merged = message_log.read_all_tails(
            con,
            exclude_conversation_id="current",
            max_messages=50,
        )
        by_cid: dict[str, int] = {}
        for row in merged:
            cid = str(row.get("conversation_id") or "")
            by_cid[cid] = by_cid.get(cid, 0) + 1

        assert by_cid["whatsapp:dm:dominant"] == 8
        assert by_cid["whatsapp:group:small@g.us"] == 2
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

    assert "## My WhatsApp Conversations:" in rendered
    assert "[dm][abc]" in rendered
    assert "[Marcos]: first" in rendered
    assert "[Marcos]: second" in rendered


def test_format_merged_history_groups_sections_and_conversations(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    rendered = message_log.format_merged_history(
        [
            {
                "conversation_id": "sillytavern:chat-a",
                "source_label": "sillytavern",
                "role": "user",
                "speaker": "Marcos",
                "content": "st message",
                "received_at": "2026-05-08T11:00:00+00:00",
            },
            {
                "conversation_id": "whatsapp:group:18322935409-1579788049@g.us",
                "source_label": "whatsapp:group",
                "role": "assistant",
                "speaker": "Echo",
                "content": "wa group message",
                "received_at": "2026-05-08T11:00:01+00:00",
            },
            {
                "conversation_id": "whatsapp:dm:15133278228",
                "source_label": "whatsapp:dm",
                "role": "user",
                "speaker": "Marcos",
                "content": "wa dm message",
                "received_at": "2026-05-08T11:00:02+00:00",
            },
        ]
    )

    assert "## My SillyTavern Conversations:" in rendered
    assert "[dm][chat-a]" in rendered
    assert "[Marcos]: st message" in rendered
    assert "## My WhatsApp Conversations:" in rendered
    assert "[group][18322935409-1579788049@g.us]" in rendered
    assert "[Echo]: wa group message" in rendered
    assert "[dm][15133278228]" in rendered
    assert "[Marcos]: wa dm message" in rendered


def test_format_merged_history_parses_legacy_group_prefix_at_render_time() -> None:
    rendered = message_log.format_merged_history(
        [
            {
                "conversation_id": "whatsapp:group:18322935409-1579788049@g.us",
                "source_label": "whatsapp:group",
                "role": "user",
                "speaker": "Marcos",
                "content": "[Raquel] Going to the gym now.",
                "received_at": "2026-05-09T12:08:26+00:00",
            }
        ]
    )

    assert "[Marcos]: [Raquel] Going to the gym now." not in rendered
    assert "[Raquel]: Going to the gym now." in rendered


def test_format_merged_history_dedupes_same_day_duplicate_lines() -> None:
    rendered = message_log.format_merged_history(
        [
            {
                "conversation_id": "whatsapp:group:18322935409-1579788049@g.us",
                "source_label": "whatsapp:group",
                "role": "user",
                "speaker": "Marcos",
                "content": "[Raquel] Going to the gym now.",
                "received_at": "2026-05-09T12:08:26+00:00",
            },
            {
                "conversation_id": "whatsapp:group:18322935409-1579788049@g.us",
                "source_label": "whatsapp:group",
                "role": "user",
                "speaker": "",
                "content": "[Raquel] Going to the gym now.",
                "received_at": "2026-05-09T12:09:26+00:00",
            },
        ]
    )

    assert rendered.count("[Raquel]: Going to the gym now.") == 1


def test_format_merged_history_uses_channel_directory_names(tmp_path, monkeypatch) -> None:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "channel_directory.json").write_text(
        json.dumps(
            {
                "platforms": {
                    "whatsapp": [
                        {"id": "18322935409-1579788049@g.us", "name": "Work Group", "type": "group"},
                        {"id": "15133278228@s.whatsapp.net", "name": "Marcos", "type": "dm"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    rendered = message_log.format_merged_history(
        [
            {
                "conversation_id": "whatsapp:group:18322935409-1579788049@g.us",
                "source_label": "whatsapp:group",
                "role": "assistant",
                "speaker": "Echo",
                "content": "group hello",
                "received_at": "2026-05-08T11:00:00+00:00",
            },
            {
                "conversation_id": "whatsapp:dm:15133278228",
                "source_label": "whatsapp:dm",
                "role": "user",
                "speaker": "Marcos",
                "content": "dm hello",
                "received_at": "2026-05-08T11:00:01+00:00",
            },
        ]
    )

    assert "[group][Work Group]" in rendered
    assert "[dm][Marcos]" in rendered


def test_format_merged_history_whatsapp_dm_heading_prefers_named_alias_over_numeric(tmp_path, monkeypatch) -> None:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    session_dir = hermes_home / "whatsapp" / "session"
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
    (hermes_home / "channel_directory.json").write_text(
        json.dumps(
            {
                "platforms": {
                    "whatsapp": [
                        {"id": "15133278228@s.whatsapp.net", "name": "15133278228", "type": "dm"},
                        {"id": "114628432556258@lid", "name": "Marcos", "type": "dm"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    rendered = message_log.format_merged_history(
        [
            {
                "conversation_id": "whatsapp:dm:15133278228",
                "source_label": "whatsapp:dm",
                "role": "user",
                "speaker": "Marcos",
                "content": "hello from dm",
                "received_at": "2026-05-10T08:00:00+00:00",
            }
        ]
    )

    assert "[dm][Marcos]" in rendered
    assert "[dm][15133278228]" not in rendered
    assert "[Marcos]: hello from dm" in rendered


def test_format_merged_history_dm_backfills_blank_user_speaker_from_matching_content() -> None:
    rendered = message_log.format_merged_history(
        [
            {
                "conversation_id": "whatsapp:dm:247789598601266",
                "source_label": "whatsapp:dm",
                "role": "user",
                "speaker": "",
                "content": "same message",
                "received_at": "2026-05-10T08:00:00+00:00",
            },
            {
                "conversation_id": "whatsapp:dm:247789598601266",
                "source_label": "whatsapp:dm",
                "role": "user",
                "speaker": "Liz Kalverda",
                "content": "same message",
                "received_at": "2026-05-10T08:00:01+00:00",
            },
        ]
    )

    assert rendered.count("[Liz Kalverda]: same message") == 1
    assert "[user]: same message" not in rendered


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


def test_normalize_whatsapp_identifier_rejects_path_like_values() -> None:
    assert message_log._normalize_whatsapp_identifier("../etc/passwd") == ""
    assert message_log._normalize_whatsapp_identifier("..\\evil") == ""
    assert message_log._normalize_whatsapp_identifier("15133278228:13@s.whatsapp.net") == "15133278228"


def test_read_lid_mapping_value_rejects_non_string_json(tmp_path) -> None:
    mapping_file = tmp_path / "lid-mapping-test.json"
    mapping_file.write_text(json.dumps({"phone": "15133278228"}), encoding="utf-8")
    assert message_log._read_lid_mapping_value(mapping_file) == ""


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


def test_read_all_tails_excludes_current_whatsapp_aliases(tmp_path, monkeypatch) -> None:
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

    con = _con()
    try:
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor) VALUES (?, ?)",
            ("whatsapp:dm:114628432556258", 0),
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor) VALUES (?, ?)",
            ("whatsapp:dm:15133278228", 0),
        )
        assert message_log.append_messages(
            con,
            "whatsapp:dm:114628432556258",
            [{"role": "user", "name": "Marcos", "content": "lid row"}],
            source_label="whatsapp:dm",
        ) == 1
        assert message_log.append_messages(
            con,
            "whatsapp:dm:15133278228",
            [{"role": "assistant", "name": "Echo", "content": "phone row"}],
            source_label="whatsapp:dm",
        ) == 1

        merged = message_log.read_all_tails(
            con,
            exclude_conversation_id="whatsapp:dm:114628432556258",
            max_messages=50,
        )
        assert merged == []
    finally:
        con.close()
