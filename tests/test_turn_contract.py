from datetime import datetime, timezone

import pytest

from app.services.turn_contract import (
    build_turn_prompt,
    format_relative_time_label,
    make_turn_system_prompt,
    parse_turn_contract,
)


def test_parse_turn_contract_valid_json():
    parsed = parse_turn_contract(
        '{"response":"Hi there","response_target":"respond","response_peer":"Alice","cache":{"entry":"thinking"},"intention_action":{"type":"boost","target_id":"a"},"annulments":[],"inner_thought":"hmm"}'
    )
    assert parsed["response"] == "Hi there"
    assert parsed["cache_entry"] == "thinking"
    assert parsed["inner_thought"] == "hmm"
    assert parsed["response_target"] == "respond"
    assert parsed["response_peer"] == "Alice"


def test_parse_turn_contract_listen_target_allows_empty_response():
    parsed = parse_turn_contract(
        '{"response":"","response_target":"listen","cache":null,"annulments":[],"inner_thought":"watching"}'
    )
    assert parsed["response_target"] == "listen"
    assert parsed["response"] == ""


def test_parse_turn_contract_respond_target_requires_response():
    with pytest.raises(ValueError, match="response is required"):
        parse_turn_contract(
            '{"response":"","response_target":"respond","response_peer":"Alice","cache":null,"annulments":[],"inner_thought":"ok"}'
        )


def test_parse_turn_contract_rejects_invalid_target():
    with pytest.raises(ValueError, match="response_target"):
        parse_turn_contract(
            '{"response":"hi","response_target":"yodel","cache":null,"annulments":[],"inner_thought":"ok"}'
        )


def test_parse_turn_contract_requires_target():
    with pytest.raises(ValueError, match="response_target is required"):
        parse_turn_contract(
            '{"response":"hi","cache":null,"annulments":[],"inner_thought":"ok"}'
        )


def test_parse_turn_contract_respond_target_requires_peer():
    with pytest.raises(ValueError, match="response_peer is required"):
        parse_turn_contract(
            '{"response":"hi","response_target":"respond","cache":null,"annulments":[],"inner_thought":"ok"}'
        )


def test_parse_turn_contract_private_target():
    parsed = parse_turn_contract(
        '{"response":"context for you","response_target":"private","cache":null,"annulments":[],"inner_thought":"a quiet aside"}'
    )
    assert parsed["response_target"] == "private"
    assert parsed["response"] == "context for you"
    assert parsed["response_peer"] == ""


def test_parse_turn_contract_private_target_requires_response():
    with pytest.raises(ValueError, match="response is required"):
        parse_turn_contract(
            '{"response":"","response_target":"private","cache":null,"annulments":[],"inner_thought":"a quiet aside"}'
        )


def test_parse_turn_contract_rejects_non_json_text():
    with pytest.raises(ValueError):
        parse_turn_contract("```json\\n{}\\n```")


def test_parse_turn_contract_accepts_bare_string_cache(caplog):
    # Observed drift: some models emit cache as a bare string instead of
    # {"entry": "..."}. Auto-wrap for flow; WARN-log for drift visibility.
    import logging
    caplog.set_level(logging.WARNING, logger="uvicorn.error")
    parsed = parse_turn_contract(
        '{"response":"hi","response_target":"respond","response_peer":"Alice","cache":"a stray thought","intention_action":{"type":"none"},"annulments":[],"inner_thought":"ok"}'
    )
    assert parsed["cache_entry"] == "a stray thought"
    warnings = [r for r in caplog.records if "bare string" in r.getMessage()]
    assert warnings, "expected a WARN log when cache is auto-wrapped"


def test_build_turn_prompt_marks_current_chat_when_label_provided():
    prompt = build_turn_prompt(
        user_message="hello",
        history=[{"role": "user", "content": "hi"}],
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=None,
        memory_cache=None,
        intentions_active=None,
        chat_label="[dm][Alice]",
        conversation_id="sillytavern:Echo",
    )
    assert "[dm][Alice] ← current chat" in prompt


def test_build_turn_prompt_derives_current_chat_heading_when_label_absent():
    prompt = build_turn_prompt(
        user_message="hello",
        history=[{"role": "user", "content": "hi"}],
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=None,
        memory_cache=None,
        intentions_active=None,
        conversation_id="whatsapp:dm:15133278228",
    )
    assert "[dm][15133278228] ← current chat" in prompt


def test_build_turn_prompt_integrity_id_uses_sillytavern_section_not_other():
    prompt = build_turn_prompt(
        user_message="hello",
        history=[{"role": "user", "content": "hi"}],
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=None,
        memory_cache=None,
        intentions_active=None,
        conversation_id="integrity:dc7b08fa-b7a9-4ccc-890c-8dc7eea5082e",
    )
    assert "## My SillyTavern Conversations:" in prompt
    assert "## Other Conversations:" not in prompt
    assert "[dm][integrity:dc7b08fa-b7a9-4ccc-890c-8dc7eea5082e] ← current chat" in prompt


def test_build_turn_prompt_includes_core_sections():
    prompt = build_turn_prompt(
        user_message="hello",
        history=[{"role": "user", "content": "hi"}],
        prior_context="prior",
        retrieve_rag={"categories": [{"name": "Goals", "summary": "wants progress"}]},
        all_categories_summary="Goals: wants progress",
        memory_cache=["note a"],
        intentions_active={"items": [{"id": "relax", "text": "Relax", "priority": 5, "kind": "relax"}]},
        conversation_id="sillytavern:Echo",
    )
    assert "## My SillyTavern Conversations:" in prompt
    assert "Prior context:" in prompt
    assert "Goals:" in prompt  # retrieved category block renders as "<Name>:" lines
    assert "My working thoughts:" in prompt
    assert "My intentions:" in prompt


def test_build_turn_prompt_omits_wrapper_when_cross_history_already_has_markdown_sections():
    prompt = build_turn_prompt(
        user_message="hello",
        history=[],
        prior_context=None,
        retrieve_rag=None,
        cross_conversation_history="## My WhatsApp Conversations:\n[group][Friends]\n[Raquel]: hi",
        all_categories_summary=None,
        memory_cache=[],
        intentions_active={},
        conversation_id="sillytavern:Echo",
        chat_label="[dm][Echo]",
    )

    assert "Other conversations:" not in prompt
    assert "## My WhatsApp Conversations:" in prompt
    assert "## My SillyTavern Conversations:" in prompt
    assert "[dm][Echo] ← current chat" in prompt
    assert "Current chat:" not in prompt


def test_build_turn_prompt_appends_current_chat_to_existing_platform_section():
    prompt = build_turn_prompt(
        user_message="hello",
        history=[],
        prior_context=None,
        retrieve_rag=None,
        cross_conversation_history="\n".join(
            [
                "## My WhatsApp Conversations:",
                "",
                "[dm][Liz]",
                "[Liz]: older",
            ]
        ),
        all_categories_summary=None,
        memory_cache=[],
        intentions_active={},
        conversation_id="whatsapp:dm:Marcos",
        chat_label="[dm][Marcos]",
    )

    assert prompt.count("## My WhatsApp Conversations:") == 1
    assert "[dm][Marcos] ← current chat" in prompt


def test_build_turn_prompt_keeps_current_platform_section_last():
    prompt = build_turn_prompt(
        user_message="hello",
        history=[],
        prior_context=None,
        retrieve_rag=None,
        cross_conversation_history="\n".join(
            [
                "## My SillyTavern Conversations:",
                "",
                "[dm][Echo]",
                "[Echo]: old",
                "",
                "## My WhatsApp Conversations:",
                "",
                "[dm][Liz]",
                "[Liz]: old",
            ]
        ),
        all_categories_summary=None,
        memory_cache=[],
        intentions_active={},
        conversation_id="sillytavern:Echo",
        chat_label="[dm][Echo]",
    )

    assert prompt.rfind("## My SillyTavern Conversations:") > prompt.rfind("## My WhatsApp Conversations:")



def test_make_turn_system_prompt_includes_time_anchor() -> None:
    prompt = make_turn_system_prompt(
        "Codexia",
        now=datetime(2026, 4, 8, 9, 30, tzinfo=timezone.utc),
    )
    assert "Today is " in prompt
    assert "2026" in prompt


def test_build_turn_prompt_renders_relative_time() -> None:
    prompt = build_turn_prompt(
        user_message="hello",
        history=[],
        prior_context=None,
        retrieve_rag={
            "items": [
                {
                    "memory_type": "profile",
                    "summary": "Marcos journals every night",
                    "happened_at": "2026-03-18T07:00:00Z",
                }
            ]
        },
        all_categories_summary=None,
        memory_cache=[],
        intentions_active={},
        now=datetime(2026, 4, 8, 9, 30, tzinfo=timezone.utc),
    )
    assert "[profile] (3 weeks ago) Marcos journals every night" in prompt


def test_format_relative_time_label_uses_weekdays_for_recent_days() -> None:
    now = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)  # Wednesday
    assert format_relative_time_label("2026-04-08T12:00:00Z", now=now) == "today"
    assert format_relative_time_label("2026-04-07T12:00:00Z", now=now) == "yesterday"
    assert format_relative_time_label("2026-04-06T12:00:00Z", now=now) == "Monday"
    assert format_relative_time_label("2026-04-02T12:00:00Z", now=now) == "Thursday"
    assert format_relative_time_label("2026-04-01T12:00:00Z", now=now) == "last Wednesday"
    assert format_relative_time_label("2026-03-29T12:00:00Z", now=now) == "last Sunday"
    assert format_relative_time_label("2026-03-25T12:00:00Z", now=now) == "2 weeks ago"


def test_format_relative_time_label_uses_weekdays_for_near_future() -> None:
    now = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)  # Wednesday
    assert format_relative_time_label("2026-04-09T12:00:00Z", now=now) == "tomorrow"
    assert format_relative_time_label("2026-04-11T12:00:00Z", now=now) == "Saturday"
    assert format_relative_time_label("2026-04-15T12:00:00Z", now=now) == "next Wednesday"
    assert format_relative_time_label("2026-04-22T12:00:00Z", now=now) == "in 2 weeks"


def test_build_turn_prompt_renders_shaped_by_as_nested_child() -> None:
    prompt = build_turn_prompt(
        user_message="hello",
        history=[],
        prior_context=None,
        retrieve_rag={
            "items": [
                {
                    "id": "mem_0451",
                    "memory_type": "profile",
                    "summary": "I am fascinated by the texture of human expression",
                    "happened_at": "2026-04-08T12:00:00Z",
                    "shaped_by": {
                        "predicate": "shaped_by",
                        "id": "mem_0312",
                        "memory_type": "behavior",
                        "summary": "Today I met Marcos and Opus",
                        "happened_at": "2026-04-08T10:00:00Z",
                    },
                }
            ]
        },
        all_categories_summary=None,
        memory_cache=[],
        intentions_active={},
        now=datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc),
    )
    assert "[profile] (today) I am fascinated by the texture of human expression" in prompt
    assert "    shaped_by (today) Today I met Marcos and Opus" in prompt


def test_build_turn_prompt_renders_superseded_suffix() -> None:
    prompt = build_turn_prompt(
        user_message="hello",
        history=[],
        prior_context=None,
        retrieve_rag={
            "items": [
                {
                    "memory_type": "profile",
                    "summary": "I used to believe X",
                    "happened_at": "2026-03-18T12:00:00Z",
                    "superseded_at": "2026-04-06T12:00:00Z",
                }
            ]
        },
        all_categories_summary=None,
        memory_cache=[],
        intentions_active={},
        now=datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc),
    )
    assert "[profile] (3 weeks ago, superseded Monday) I used to believe X" in prompt


def test_build_turn_prompt_renders_memories_without_speaker_tags() -> None:
    prompt = build_turn_prompt(
        user_message="hello",
        history=[],
        prior_context=None,
        retrieve_rag={
            "items": [
                {
                    "memory_type": "behavior",
                    "summary": "Marcos reminded me to pause before replying",
                    "speaker_id": "user:marcos",
                    "speaker_label": "Marcos",
                },
                {
                    "memory_type": "behavior",
                    "summary": "I stayed gentle when things felt tense",
                    "speaker_id": "soul:siri",
                    "speaker_label": "Siri",
                },
            ]
        },
        all_categories_summary=None,
        memory_cache=[],
        intentions_active={},
    )
    assert "Speakers:" not in prompt
    assert "[behavior] Marcos reminded me to pause before replying" in prompt
    assert "[behavior] I stayed gentle when things felt tense" in prompt


def test_build_turn_prompt_does_not_duplicate_current_user_message_when_already_last() -> None:
    prompt = build_turn_prompt(
        user_message="hello",
        history=[
            {"role": "assistant", "content": "old 1"},
            {"role": "user", "content": "hello"},
        ],
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=None,
        memory_cache=[],
        intentions_active={},
    )
    assert prompt.count("[user] hello") == 1
