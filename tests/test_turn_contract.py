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
    assert "Retrieved memory context:" in prompt
    assert "Your working thoughts:" in prompt
    assert "Intentions:" in prompt


def test_build_turn_prompt_limits_history_by_token_budget():
    prompt = build_turn_prompt(
        user_message="hello",
        history=[
            {"message_id": "1", "role": "user", "content": "one two three four five six seven eight nine ten"},
            {"message_id": "2", "role": "assistant", "content": "ok"},
        ],
        history_token_budget=2,
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


def test_build_turn_prompt_renders_relative_time_and_reinforcement() -> None:
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
                    "extra": {"reinforcement_count": 5},
                }
            ]
        },
        all_categories_summary=None,
        memory_cache=[],
        intentions_active={},
        now=datetime(2026, 4, 8, 9, 30, tzinfo=timezone.utc),
    )
    assert "- [profile] (3 weeks ago, reinforced 5x) Marcos journals every night" in prompt


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
