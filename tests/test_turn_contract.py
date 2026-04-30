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
        '{"response":"Hi there","cache":{"entry":"thinking"},"intention_action":{"type":"boost","target_id":"a"},"annulments":[],"inner_thought":"hmm"}'
    )
    assert parsed["response"] == "Hi there"
    assert parsed["cache_entry"] == "thinking"
    assert parsed["inner_thought"] == "hmm"


def test_parse_turn_contract_rejects_non_json_text():
    with pytest.raises(ValueError):
        parse_turn_contract("```json\\n{}\\n```")


def test_parse_turn_contract_accepts_bare_string_cache(caplog):
    # Observed drift: some models emit cache as a bare string instead of
    # {"entry": "..."}. Auto-wrap for flow; WARN-log for drift visibility.
    import logging
    caplog.set_level(logging.WARNING, logger="uvicorn.error")
    parsed = parse_turn_contract(
        '{"response":"hi","cache":"a stray thought","intention_action":{"type":"none"},"annulments":[],"inner_thought":"ok"}'
    )
    assert parsed["cache_entry"] == "a stray thought"
    warnings = [r for r in caplog.records if "bare string" in r.getMessage()]
    assert warnings, "expected a WARN log when cache is auto-wrapped"


def test_build_turn_prompt_includes_core_sections():
    prompt = build_turn_prompt(
        user_message="hello",
        history=[{"role": "user", "content": "hi"}],
        prior_context="prior",
        retrieve_rag={"categories": [{"name": "Goals", "summary": "wants progress"}]},
        all_categories_summary="Goals: wants progress",
        memory_cache=["note a"],
        intentions_active={"items": [{"id": "relax", "text": "Relax", "priority": 5, "kind": "relax"}]},
    )
    assert "Conversation history:" in prompt
    assert "Prior context:" in prompt
    assert "Goals:" in prompt  # retrieved category block renders as "<Name>:" lines
    assert "Your working thoughts:" in prompt
    assert "Your intentions:" in prompt


def test_build_turn_prompt_limits_history_by_token_budget():
    prompt = build_turn_prompt(
        user_message="hello",
        history=[
            {"message_id": "1", "role": "user", "content": "one two three four five six seven eight nine ten"},
            {"message_id": "2", "role": "assistant", "content": "ok"},
        ],
        history_tail_after_memorize=2,
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=None,
        memory_cache=[],
        intentions_active={},
    )
    assert "[2] [assistant] ok" in prompt
    assert "[1] [user]" not in prompt


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
    assert "- (3 weeks ago) Marcos journals every night" in prompt


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
    assert "- (today) I am fascinated by the texture of human expression" in prompt
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
    assert "- (3 weeks ago, superseded Monday) I used to believe X" in prompt


def test_build_turn_prompt_renders_speakers_block_and_labels() -> None:
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
    assert "Speakers:" in prompt
    assert "- Marcos = user:marcos" in prompt
    assert "- Siri = soul:siri" in prompt
    assert "-[Marcos] Marcos reminded me to pause before replying" in prompt
    assert "-[Siri] I stayed gentle when things felt tense" in prompt


def test_build_turn_prompt_omits_speakers_block_without_labels() -> None:
    prompt = build_turn_prompt(
        user_message="hello",
        history=[],
        prior_context=None,
        retrieve_rag={
            "items": [
                {
                    "memory_type": "behavior",
                    "summary": "A memory without a speaker label",
                    "speaker_id": "user:marcos",
                }
            ]
        },
        all_categories_summary=None,
        memory_cache=[],
        intentions_active={},
    )
    assert "Speakers:" not in prompt
    assert "- A memory without a speaker label" in prompt
