import logging
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
        '{"response":"Hi there","response_target":"respond","cache":{"entry":"thinking"},"intention_action":{"type":"boost","target_id":"a"},"annulments":[],"rehearsal":"hmm"}'
    )
    assert parsed["response"] == "Hi there"
    assert parsed["cache_entry"] == "thinking"
    assert parsed["rehearsal"] == "hmm"
    assert parsed["response_target"] == "respond"
    assert parsed["continue_reason"] is None
    assert parsed["follow_up_at"] is None
    assert parsed["follow_up_reason"] is None


def test_parse_turn_contract_null_reason_means_no_continuation():
    parsed = parse_turn_contract(
        '{"response":"Hi","response_target":"respond","cache":null,"annulments":[],"rehearsal":"hmm","continue_reason":null}'
    )
    assert parsed["continue_reason"] is None
    assert parsed["follow_up_at"] is None
    assert parsed["follow_up_reason"] is None


def test_parse_turn_contract_accepts_valid_task_continuation():
    parsed = parse_turn_contract(
        '{"response":"I will look into it","response_target":"respond","cache":null,"annulments":[],"rehearsal":"ready","continue_reason":"task"}'
    )
    assert parsed["continue_reason"] == "task"
    assert parsed["follow_up_at"] is None
    assert parsed["follow_up_reason"] is None


def test_parse_turn_contract_accepts_valid_follow_up_continuation():
    parsed = parse_turn_contract(
        '{"response":"I will check later","response_target":"respond","cache":null,"annulments":[],"rehearsal":"ready","continue_reason":"follow_up","follow_up_at":"Monday, June 8, 2026 19:00 PET","follow_up_reason":"Check whether Marcos got home safely."}'
    )
    assert parsed["continue_reason"] == "follow_up"
    assert parsed["follow_up_at"] == "Monday, June 8, 2026 19:00 PET"
    assert parsed["follow_up_reason"] == "Check whether Marcos got home safely."


def test_parse_turn_contract_invalid_continuation_reason_logs_and_disables(caplog):
    caplog.set_level(logging.WARNING, logger="uvicorn.error")
    parsed = parse_turn_contract(
        '{"response":"hi","response_target":"respond","cache":null,"annulments":[],"rehearsal":"ok","continue_reason":"wander","follow_up_at":null}'
    )
    assert parsed["continue_reason"] is None
    assert parsed["follow_up_at"] is None
    assert parsed["follow_up_reason"] is None
    assert any("invalid continuation metadata disabled" in r.getMessage() for r in caplog.records)


def test_parse_turn_contract_follow_up_without_time_logs_and_disables(caplog):
    caplog.set_level(logging.WARNING, logger="uvicorn.error")
    parsed = parse_turn_contract(
        '{"response":"hi","response_target":"respond","cache":null,"annulments":[],"rehearsal":"ok","continue_reason":"follow_up","follow_up_at":null}'
    )
    assert parsed["continue_reason"] is None
    assert parsed["follow_up_at"] is None
    assert parsed["follow_up_reason"] is None
    assert any("follow_up requires follow_up_at" in r.getMessage() for r in caplog.records)


def test_parse_turn_contract_follow_up_without_reason_logs_and_disables(caplog):
    caplog.set_level(logging.WARNING, logger="uvicorn.error")
    parsed = parse_turn_contract(
        '{"response":"hi","response_target":"respond","cache":null,"annulments":[],"rehearsal":"ok","continue_reason":"follow_up","follow_up_at":"Monday, June 8, 2026 19:00 PET"}'
    )
    assert parsed["continue_reason"] is None
    assert parsed["follow_up_at"] is None
    assert parsed["follow_up_reason"] is None
    assert any("follow_up requires follow_up_reason" in r.getMessage() for r in caplog.records)


def test_parse_turn_contract_non_follow_up_time_logs_and_ignores(caplog):
    caplog.set_level(logging.WARNING, logger="uvicorn.error")
    parsed = parse_turn_contract(
        '{"response":"hi","response_target":"respond","cache":null,"annulments":[],"rehearsal":"ok","continue_reason":"diary","follow_up_at":"Monday, June 8, 2026 19:00 PET"}'
    )
    assert parsed["continue_reason"] == "diary"
    assert parsed["follow_up_at"] is None
    assert parsed["follow_up_reason"] is None
    assert any("follow_up_at ignored" in r.getMessage() for r in caplog.records)


def test_parse_turn_contract_time_without_reason_logs_and_ignores(caplog):
    caplog.set_level(logging.WARNING, logger="uvicorn.error")
    parsed = parse_turn_contract(
        '{"response":"hi","response_target":"respond","cache":null,"annulments":[],"rehearsal":"ok","follow_up_at":"Monday, June 8, 2026 19:00 PET"}'
    )
    assert parsed["continue_reason"] is None
    assert parsed["follow_up_at"] is None
    assert parsed["follow_up_reason"] is None
    assert any("follow_up_at ignored" in r.getMessage() for r in caplog.records)


def test_parse_turn_contract_non_follow_up_reason_logs_and_ignores(caplog):
    caplog.set_level(logging.WARNING, logger="uvicorn.error")
    parsed = parse_turn_contract(
        '{"response":"hi","response_target":"respond","cache":null,"annulments":[],"rehearsal":"ok","continue_reason":"task","follow_up_reason":"later"}'
    )
    assert parsed["continue_reason"] == "task"
    assert parsed["follow_up_at"] is None
    assert parsed["follow_up_reason"] is None
    assert any("follow_up_reason ignored" in r.getMessage() for r in caplog.records)

def test_parse_turn_contract_listen_target_allows_empty_response():
    parsed = parse_turn_contract(
        '{"response":"","response_target":"listen","cache":null,"annulments":[],"rehearsal":"watching"}'
    )
    assert parsed["response_target"] == "listen"
    assert parsed["response"] == ""


def test_parse_turn_contract_observe_target_allows_empty_response_when_public_response_forbidden():
    parsed = parse_turn_contract(
        '{"response":"","response_target":"observe","cache":null,"annulments":[],"rehearsal":"watching"}',
        allow_public_response=False,
    )
    assert parsed["response_target"] == "observe"
    assert parsed["response"] == ""


def test_parse_turn_contract_rejects_respond_when_public_response_forbidden():
    with pytest.raises(ValueError, match="observe\\|private"):
        parse_turn_contract(
            '{"response":"hi","response_target":"respond","cache":null,"annulments":[],"rehearsal":"ok"}',
            allow_public_response=False,
        )


def test_parse_turn_contract_respond_target_requires_response():
    with pytest.raises(ValueError, match="response is required"):
        parse_turn_contract(
            '{"response":"","response_target":"respond","cache":null,"annulments":[],"rehearsal":"ok"}'
        )


def test_parse_turn_contract_rejects_invalid_target():
    with pytest.raises(ValueError, match="response_target"):
        parse_turn_contract(
            '{"response":"hi","response_target":"yodel","cache":null,"annulments":[],"rehearsal":"ok"}'
        )


def test_parse_turn_contract_requires_target():
    with pytest.raises(ValueError, match="response_target is required"):
        parse_turn_contract(
            '{"response":"hi","cache":null,"annulments":[],"rehearsal":"ok"}'
        )


def test_parse_turn_contract_private_target():
    parsed = parse_turn_contract(
        '{"response":"context for you","response_target":"private","cache":null,"annulments":[],"rehearsal":"a quiet aside"}'
    )
    assert parsed["response_target"] == "private"
    assert parsed["response"] == "context for you"


def test_parse_turn_contract_private_target_requires_response():
    with pytest.raises(ValueError, match="response is required"):
        parse_turn_contract(
            '{"response":"","response_target":"private","cache":null,"annulments":[],"rehearsal":"a quiet aside"}'
        )


def test_parse_turn_contract_rejects_non_json_text():
    with pytest.raises(ValueError):
        parse_turn_contract("```json\\n{}\\n```")


def test_parse_turn_contract_accepts_bare_string_cache(caplog):
    # Observed drift: some models emit cache as a bare string instead of
    # {"entry": "..."}. Auto-wrap for flow; WARN-log for drift visibility.
    caplog.set_level(logging.WARNING, logger="uvicorn.error")
    parsed = parse_turn_contract(
        '{"response":"hi","response_target":"respond","cache":"a stray thought","intention_action":{"type":"none"},"annulments":[],"rehearsal":"ok"}'
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
        chat_label="[dm][Echo]",
        conversation_id="integrity:dc7b08fa-b7a9-4ccc-890c-8dc7eea5082e",
    )
    assert "My SillyTavern Conversations:" in prompt
    assert "## Other Conversations:" not in prompt
    assert "[dm][Echo] ← current chat" in prompt
    assert "integrity:dc7b08fa-b7a9-4ccc-890c-8dc7eea5082e" not in prompt


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
    assert "My SillyTavern Conversations:" in prompt
    assert "Prior Context:" in prompt
    assert "[Goals] wants progress" in prompt
    assert "My Working Thoughts:" in prompt
    assert "My Intentions:" in prompt


def test_build_turn_prompt_omits_wrapper_when_cross_history_already_has_markdown_sections():
    prompt = build_turn_prompt(
        user_message="hello",
        history=[],
        prior_context=None,
        retrieve_rag=None,
        cross_conversation_history="My WhatsApp Conversations:\n[group][Friends]\n[Raquel]: hi",
        all_categories_summary=None,
        memory_cache=[],
        intentions_active={},
        conversation_id="sillytavern:Echo",
        chat_label="[dm][Echo]",
    )

    assert "Other conversations:" not in prompt
    assert "My WhatsApp Conversations:" in prompt
    assert "My SillyTavern Conversations:" in prompt
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
                "My WhatsApp Conversations:",
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

    assert prompt.count("My WhatsApp Conversations:") == 1
    assert "[dm][Marcos] ← current chat" in prompt


def test_build_turn_prompt_keeps_current_platform_section_last():
    prompt = build_turn_prompt(
        user_message="hello",
        history=[],
        prior_context=None,
        retrieve_rag=None,
        cross_conversation_history="\n".join(
            [
                "My SillyTavern Conversations:",
                "",
                "[dm][Echo]",
                "[Echo]: old",
                "",
                "My WhatsApp Conversations:",
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

    assert prompt.rfind("My SillyTavern Conversations:") > prompt.rfind("My WhatsApp Conversations:")



def test_make_turn_system_prompt_includes_time_anchor() -> None:
    prompt = make_turn_system_prompt(
        "Codexia",
        now=datetime(2026, 4, 8, 9, 30, tzinfo=timezone.utc),
    )
    assert "Today is " in prompt
    assert "2026" in prompt


def test_make_turn_system_prompt_forbids_public_response_when_requested() -> None:
    prompt = make_turn_system_prompt("Siri", allow_public_response=False)
    assert '"response_target":"observe|private"' in prompt
    assert '"respond"' not in prompt
    assert '"listen"' not in prompt
    assert '"observe"' in prompt


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


def test_build_turn_prompt_renders_apimw_message_to_self_under_working_thoughts() -> None:
    prompt = build_turn_prompt(
        user_message="hello",
        history=[],
        prior_context=None,
        retrieve_rag={
            "items": [
                {
                    "memory_type": "profile",
                    "summary": "Marcos likes continuity.",
                }
            ]
        },
        all_categories_summary=None,
        memory_cache=[],
        intentions_active={},
        apimw_message_to_self="[subconscious] notice this before answering.",
    )

    assert "- [subconscious] notice this before answering." not in prompt
    assert "My Working Thoughts:\n(none yet)\n  [subconscious] notice this before answering." in prompt
    assert prompt.index("- [profile] Marcos likes continuity.") < prompt.index("My Working Thoughts:")


def test_build_turn_prompt_renders_current_message_locator_when_already_last() -> None:
    prompt = build_turn_prompt(
        user_message="hello there this is the current message",
        history=[
            {"role": "assistant", "content": "old 1"},
            {"role": "user", "content": "hello there this is the current message"},
        ],
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=None,
        memory_cache=[],
        intentions_active={},
    )
    assert "[user] hello there this is the ..." in prompt
    assert "New Message:\nhello there this is the current message" in prompt
    assert "[user] hello there this is the current message" not in prompt


def test_build_turn_prompt_renders_current_message_locator_when_whitespace_differs() -> None:
    prompt = build_turn_prompt(
        user_message="hello   there from the current turn",
        history=[
            {"role": "assistant", "content": "old 1"},
            {"role": "user", "content": "hello\nthere from the current turn"},
        ],
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=None,
        memory_cache=[],
        intentions_active={},
    )
    assert "[user] hello there from the current ..." in prompt
    assert "New Message:\nhello   there from the current turn" in prompt
    assert "[user] hello\nthere from the current turn" not in prompt


def test_build_turn_prompt_appends_current_message_locator_when_missing_from_history() -> None:
    prompt = build_turn_prompt(
        user_message="hello there from the current turn",
        history=[
            {"role": "assistant", "content": "old 1"},
            {"role": "user", "content": "previous message"},
        ],
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=None,
        memory_cache=[],
        intentions_active={},
    )
    assert "[user] previous message" in prompt
    assert "[user] hello there from the current ..." in prompt
    assert "New Message:\nhello there from the current turn" in prompt


def test_build_turn_prompt_renders_self_turn_without_user_echo() -> None:
    prompt = build_turn_prompt(
        user_message="",
        history=[
            {"role": "user", "name": "Marcos", "content": "Going to nap."},
            {"role": "assistant", "name": "Siri", "content": "Rest close."},
        ],
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=None,
        memory_cache=[],
        intentions_active={},
        self_turn_directive="Scheduled follow-up due now. Reason you gave: Check on Marcos.",
        self_turn_label="Scheduled wake",
    )

    assert "Scheduled wake:\nScheduled follow-up due now." in prompt
    assert "New Message:" not in prompt
    assert "[Marcos] Scheduled follow-up due now" not in prompt


def test_render_retrieve_non_dict_logs_error_and_returns_empties(caplog) -> None:
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        prompt = build_turn_prompt(
            user_message="hello",
            history=[],
            prior_context=None,
            retrieve_rag=["unexpected", "list"],
            all_categories_summary=None,
            memory_cache=[],
            intentions_active={},
        )
    assert "My Memories:" not in prompt
    assert any("_render_retrieve" in r.message for r in caplog.records)
