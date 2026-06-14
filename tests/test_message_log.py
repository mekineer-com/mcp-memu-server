import json

from app.services import message_log


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

    assert "My WhatsApp Conversations:" in rendered
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

    assert "My SillyTavern Conversations:" in rendered
    assert "[dm][chat-a]" in rendered
    assert "[Marcos]: st message" in rendered
    assert "My WhatsApp Conversations:" in rendered
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

    assert "My SillyTavern Conversations:" in rendered
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
                        {"id": "15133278228@lid", "name": "Marcos", "type": "dm"},
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


def test_normalize_whatsapp_identifier_rejects_path_like_values() -> None:
    assert message_log._normalize_whatsapp_identifier("../etc/passwd") == ""
    assert message_log._normalize_whatsapp_identifier("..\\evil") == ""
    assert message_log._normalize_whatsapp_identifier("15133278228:13@s.whatsapp.net") == "15133278228"
