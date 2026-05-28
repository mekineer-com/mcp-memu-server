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


def test_append_messages_existing_single_row_does_not_drop_new_user_row() -> None:
    con = _con()
    try:
        assert message_log.append_messages(
            con,
            "c3-edge",
            [{"role": "user", "content": "existing one"}],
        ) == 1

        assert message_log.append_messages(
            con,
            "c3-edge",
            [
                {"role": "user", "content": "new current user"},
                {"role": "assistant", "content": "assistant reply"},
            ],
        ) == 2

        rows = message_log.read_tail(con, "c3-edge", after_cursor=0)
        assert [r["content"] for r in rows] == [
            "existing one",
            "new current user",
            "assistant reply",
        ]
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


def test_read_all_tails_for_memorize_background_uses_rowid_cursor() -> None:
    con = _con()
    try:
        con.execute(
            "INSERT INTO conversations (conversation_id, memorize_chat, rolling_summary_cursor_id) VALUES (?, ?, ?)",
            ("bg-1", 0, None),
        )
        assert message_log.append_messages(
            con,
            "bg-1",
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ],
        ) == 3
        last_id = message_log.last_message_row_id(con, "bg-1")
        assert last_id is not None
        con.execute(
            "UPDATE conversations SET rolling_summary_cursor_id = ? WHERE conversation_id = ?",
            (int(last_id) - 1, "bg-1"),
        )
        con.commit()

        tails = message_log.read_all_tails_for_memorize(con)
        rows = tails.get("bg-1") or []
        assert [str(r.get("content")) for r in rows] == ["three"]
        assert rows[0]["memorize_chat"] is False
    finally:
        con.close()


def test_read_background_rolling_summaries_only_non_memorized_with_text() -> None:
    con = _con()
    try:
        con.execute(
            "INSERT INTO conversations (conversation_id, memorize_chat, rolling_summary) VALUES (?, ?, ?)",
            ("bg-2", 0, "summary text"),
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, memorize_chat, rolling_summary) VALUES (?, ?, ?)",
            ("primary-1", 1, "should be ignored"),
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, memorize_chat, rolling_summary) VALUES (?, ?, ?)",
            ("bg-empty", 0, ""),
        )
        con.commit()

        rows = message_log.read_background_rolling_summaries(con)
        assert "bg-2" in rows
        assert rows["bg-2"]["summary"] == "summary text"
        assert "primary-1" not in rows
        assert "bg-empty" not in rows
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


def test_read_all_tails_backfills_short_unmemorized_tail_when_history_is_short() -> None:
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
        assert [m["content"] for m in merged] == ["one", "two", "three", "four"]
    finally:
        con.close()


def test_read_all_tails_backfills_short_unmemorized_tail_to_floor() -> None:
    con = _con()
    try:
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor) VALUES (?, ?)",
            ("current", 0),
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor, last_memorize_at) VALUES (?, ?, ?)",
            ("c5-floor", 6, "2026-05-08T00:00:00+00:00"),
        )
        assert message_log.append_messages(
            con,
            "c5-floor",
            [{"role": "user", "content": f"msg-{i}"} for i in range(10)],
        ) == 10

        merged = message_log.read_all_tails(con, exclude_conversation_id="current")
        assert [m["content"] for m in merged] == [f"msg-{i}" for i in range(2, 10)]
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


def test_read_all_tails_keeps_full_unmemorized_tail_per_conversation() -> None:
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

        assert by_cid["whatsapp:dm:dominant"] == 20
        assert by_cid["whatsapp:group:small@g.us"] == 2
    finally:
        con.close()


def test_read_all_tails_fallback_remains_limited_per_conversation() -> None:
    con = _con()
    try:
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor) VALUES (?, ?)",
            ("current", 0),
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor, last_memorize_at) VALUES (?, ?, ?)",
            ("c7", 999, "2026-05-08T00:00:00+00:00"),
        )
        rows = [{"role": "user", "content": f"msg-{i}"} for i in range(20)]
        assert message_log.append_messages(con, "c7", rows) == 20

        merged = message_log.read_all_tails(con, exclude_conversation_id="current")
        assert [m["content"] for m in merged] == [f"msg-{i}" for i in range(12, 20)]
    finally:
        con.close()


def test_format_merged_history_does_not_relabel_blank_speaker_rows() -> None:
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
    assert "[user]: second" in rendered
    assert "[Marcos]: second" not in rendered


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


def test_format_merged_history_integrity_id_uses_persisted_chat_name() -> None:
    rendered = message_log.format_merged_history(
        [
            {
                "conversation_id": "integrity:dc7b08fa-b7a9-4ccc-890c-8dc7eea5082e",
                "source_label": "sillytavern",
                "role": "assistant",
                "speaker": "Echo",
                "chat_name": "Echo",
                "content": "integrity id message",
                "received_at": "2026-05-08T11:00:00+00:00",
            }
        ]
    )

    assert "## My SillyTavern Conversations:" in rendered
    assert "[dm][Echo]" in rendered
    assert "integrity:dc7b08fa-b7a9-4ccc-890c-8dc7eea5082e" not in rendered
    assert "[Echo]: integrity id message" in rendered


def test_format_merged_history_multiple_integrity_conversations_use_distinct_chat_names() -> None:
    rendered = message_log.format_merged_history(
        [
            {
                "conversation_id": "integrity:dc7b08fa-b7a9-4ccc-890c-8dc7eea5082e",
                "source_label": "sillytavern",
                "role": "assistant",
                "speaker": "Echo",
                "chat_name": "Echo",
                "content": "first",
                "received_at": "2026-05-08T11:00:00+00:00",
            },
            {
                "conversation_id": "integrity:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "source_label": "sillytavern",
                "role": "assistant",
                "speaker": "Iris",
                "chat_name": "Iris",
                "content": "second",
                "received_at": "2026-05-08T11:00:01+00:00",
            },
        ]
    )

    assert "[dm][Echo]" in rendered
    assert "[dm][Iris]" in rendered


def test_derive_source_label_maps_integrity_and_chat_to_sillytavern() -> None:
    assert message_log.derive_source_label("integrity:abc-123") == "sillytavern"
    assert message_log.derive_source_label("chat:Echo.chat") == "sillytavern"


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


def test_format_merged_history_preserves_same_day_duplicate_lines() -> None:
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

    assert rendered.count("[Raquel]: Going to the gym now.") == 2


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


def test_format_merged_history_group_heading_uses_group_name_cache(tmp_path, monkeypatch) -> None:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "channel_directory.json").write_text(
        json.dumps(
            {
                "platforms": {
                    "whatsapp": [
                        {
                            "id": "18322935409-1579788049@g.us",
                            "name": "18322935409-1579788049",
                            "type": "group",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (hermes_home / "whatsapp_group_names.json").write_text(
        json.dumps({"18322935409-1579788049@g.us": "Familia"}),
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
            }
        ]
    )

    assert "[group][Familia]" in rendered
    assert "[group][18322935409-1579788049]" not in rendered


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


def test_format_merged_history_dm_preserves_explicit_speaker_per_row(tmp_path, monkeypatch) -> None:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "channel_directory.json").write_text(
        json.dumps(
            {
                "platforms": {
                    "whatsapp": [
                        {"id": "19999999999@lid", "name": "Alice", "type": "dm"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # In a DM with Liz, both Marcos's outbound and Liz's inbound may appear
    # as role=user (in self-chat mode bridging). The speaker stamped at
    # ingestion is the truth — the renderer must not re-attribute either row.
    rendered = message_log.format_merged_history(
        [
            {
                "conversation_id": "whatsapp:dm:19999999999",
                "source_label": "whatsapp:dm",
                "role": "user",
                "speaker": "Marcos",
                "content": "did news travel?",
                "received_at": "2026-05-10T08:00:00+00:00",
            },
            {
                "conversation_id": "whatsapp:dm:19999999999",
                "source_label": "whatsapp:dm",
                "role": "user",
                "speaker": "Alice",
                "content": "yes I saw it",
                "received_at": "2026-05-10T08:00:01+00:00",
            },
        ]
    )

    assert "[dm][Alice]" in rendered
    assert "[Marcos]: did news travel?" in rendered
    assert "[Alice]: yes I saw it" in rendered


def test_format_merged_history_dm_does_not_infer_blank_speaker_from_other_rows(tmp_path, monkeypatch) -> None:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "channel_directory.json").write_text(
        json.dumps(
            {
                "platforms": {
                    "whatsapp": [
                        {"id": "19999999999@lid", "name": "Alice", "type": "dm"},
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
                "conversation_id": "whatsapp:dm:19999999999",
                "source_label": "whatsapp:dm",
                "role": "user",
                "speaker": "Marcos",
                "content": "from me",
                "received_at": "2026-05-10T08:00:00+00:00",
            },
            {
                "conversation_id": "whatsapp:dm:19999999999",
                "source_label": "whatsapp:dm",
                "role": "user",
                "speaker": "",
                "content": "missing speaker row",
                "received_at": "2026-05-10T08:00:01+00:00",
            },
        ]
    )

    assert "[Marcos]: from me" in rendered
    assert "[user]: missing speaker row" in rendered
    assert "[Marcos]: missing speaker row" not in rendered


def test_format_merged_history_whatsapp_dm_keeps_self_speaker_for_self_chat(tmp_path, monkeypatch) -> None:
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
                "conversation_id": "whatsapp:dm:114628432556258",
                "source_label": "whatsapp:dm",
                "role": "user",
                "speaker": "Marcos",
                "content": "self message",
                "received_at": "2026-05-10T08:00:00+00:00",
            }
        ]
    )

    assert "[dm][Marcos]" in rendered
    assert "[Marcos]: self message" in rendered


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
