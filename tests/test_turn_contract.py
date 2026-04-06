import pytest

from app.services.turn_contract import build_turn_prompt, parse_turn_contract


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
    assert "Your recent thoughts:" in prompt
    assert "Intentions:" in prompt
