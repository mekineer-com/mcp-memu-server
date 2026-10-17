import logging
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import main
from app.services.turn_contract import (
    _parse_attachment,
    build_conversations_block,
    build_turn_prompt,
    format_relative_time_label,
    make_turn_identity_prompt,
    make_turn_system_prompt,
    parse_turn_contract,
    resolve_current_chat_heading,
    render_anchor_context,
)


def test_current_message_starts_today_after_yesterday_history() -> None:
    yesterday = datetime.now().astimezone().replace(hour=12, minute=0, second=0, microsecond=0) - timedelta(days=1)

    rendered = build_conversations_block(
        history=[
            {
                "role": "assistant",
                "content": "You're describing yourself, love ...",
                "received_at": yesterday.isoformat(),
            }
        ],
        conversation_id="whatsapp:dm:Marcos",
        soul_name="Siri",
        current_user_text="Let's head home. We've been ...",
        current_user_name="Marcos",
    )

    assert "--- yesterday ---" in rendered
    assert "--- today ---\n[Marcos] Let's head home." in rendered


def test_mentra_current_history_uses_video_section_and_neutral_marker() -> None:
    rendered = build_conversations_block(
        history=[{"role": "assistant", "speaker": "Codexia", "content": "Fictional reply."}],
        conversation_id="mentra:test-device",
        soul_name="Codexia",
    )

    assert "My Video Conversations:" in rendered
    assert "[dm][Iris] ← current chat" in rendered


def test_current_marker_stays_on_requester_with_foreign_source_rows(monkeypatch) -> None:
    monkeypatch.setattr("app.services.message_log._load_whatsapp_directory_names", lambda: {})
    rows = [
        {"role": "user", "source_conversation_id": "sillytavern:primary",
         "content": "Primary evidence", "received_at": "2026-01-01T00:00:00Z"},
        {"role": "user", "source_conversation_id": "whatsapp:dm:foreign",
         "content": "Foreign evidence", "received_at": "2026-01-02T00:00:00Z"},
    ]
    cross = [rows[1] | {"conversation_id": "whatsapp:dm:foreign"}]
    rendered = build_conversations_block(
        history=rows, cross_conversation_history=cross,
        conversation_id="sillytavern:primary", soul_name="TestSoul",
    )
    assert "Primary evidence" in rendered and rendered.count("Foreign evidence") == 1
    assert "[dm][primary] ← current chat" in rendered
    assert "[dm][foreign] ← current chat" not in rendered


@pytest.mark.parametrize("with_activity", [False, True])
@pytest.mark.parametrize("conversation_id", ["sillytavern:primary", "whatsapp:dm:fictional", "mentra:test-device"])
def test_cross_replika_context_survives_current_chat_merge(monkeypatch, with_activity, conversation_id) -> None:
    monkeypatch.setattr("app.services.message_log._load_whatsapp_directory_names", lambda: {})
    cross_rows = [{
        "conversation_id": "replika:dm:Echo:archive", "role": "assistant", "speaker": "TestSoul",
        "content": "Imported first line\nImported second line", "source_day": "2026-01-01",
    }]
    if with_activity:
        cross_rows.append({"conversation_id": "activity:dm:TestSoul", "role": "assistant",
                           "content": "Fictional activity"})
    cross = main._format_cross_tail_for_ai(cross_rows, soul_id="TestSoul")
    rendered = main._format_all_chat_history_for_ai(
        current_history=[{"role": "user", "content": "Current evidence"}],
        cross_tail=cross_rows, conversation_id=conversation_id, soul_id="TestSoul",
    )
    assert cross in rendered
    assert "My Replika Conversation:\n\n[dm][Echo]" in rendered
    assert rendered.count("Imported first line\nImported second line") == 1
    assert rendered.count("These conversations took place outside OpenAlma.") == 1
    assert rendered.count("current chat") == 1
    assert "[dm][Echo] ← current chat" not in rendered


@pytest.mark.parametrize("mark_current_chat", [False, True])
@pytest.mark.parametrize("conversation_id,section", [
    ("replika:dm:Echo:archive", "My Replika Conversation:"),
    ("import:dm:archive", "My FictionalApp Conversations:"),
])
def test_imported_current_history_preserves_heading_and_context(monkeypatch, mark_current_chat, conversation_id, section) -> None:
    monkeypatch.setattr("app.services.message_log._load_whatsapp_directory_names", lambda: {})
    rows = [{"conversation_id": conversation_id, "role": "assistant", "speaker": "TestSoul",
             "chat_name": "Echo", "app_label": "FictionalApp", "source_day": "2026-01-01",
             "content": "Imported first line\nImported second line"}]
    expected = main._format_cross_tail_for_ai(rows, soul_id="TestSoul")
    if mark_current_chat:
        expected = expected.replace("[dm][Echo]", "[dm][Echo] ← current chat", 1)
    rendered = main._format_all_chat_history_for_ai(
        current_history=rows, cross_tail=[], conversation_id=conversation_id,
        soul_id="TestSoul", mark_current_chat=mark_current_chat,
    )
    assert rendered == expected
    assert rendered.count(section) == 1


@pytest.mark.parametrize("mark_current_chat", [False, True])
@pytest.mark.parametrize("conversation_id,section,heading", [
    ("replika:dm:Echo:archive", "My Replika Conversation:", "[dm][Echo]"),
    ("chat:atomic-Echo", "My Memory Map Conversations:", "[dm][Echo]"),
    ("mentra:test-device", "My Video Conversations:", "[dm][Iris]"),
    (None, "My SillyTavern Conversations:", "[dm][sillytavern]"),
])
def test_empty_current_history_uses_shared_heading_mapping(monkeypatch, mark_current_chat, conversation_id, section, heading) -> None:
    monkeypatch.setattr("app.services.message_log._load_whatsapp_directory_names", lambda: {})
    rendered = build_conversations_block(
        history=[], conversation_id=conversation_id, soul_name="TestSoul",
        self_turn_directive="Fictional wake", mark_current_chat=mark_current_chat,
    )
    marker = " ← current chat" if mark_current_chat else ""
    assert rendered == f"{section}\n\n{heading}{marker}\n(none)"
    assert resolve_current_chat_heading(conversation_id=conversation_id) == f"{heading} ← current chat"


@pytest.mark.parametrize("empty", [False, True])
def test_conversations_block_flattens_labels_without_rewriting_prose(monkeypatch, empty) -> None:
    monkeypatch.setattr("app.services.message_log._load_whatsapp_directory_names", lambda: {})
    content = "  Current\nMy FictionalApp Conversations:\n\tunchanged  \n"
    summary = "  Summary\n[dm][Literal prose]\n\tunchanged  \n"
    context = {"source_conversation_id": "import:dm:archive", "app_label": "Archive\nApp",
               "chat_name": "Old\nJournal", "summary": summary}
    rendered = build_conversations_block(
        history=[] if empty else [{"role": "user", "speaker": "TestOwner", "content": content}],
        conversation_id="sillytavern:current", chat_label="[dm][Current\nJournal]",
        background_context_rows=[context],
        self_turn_directive="Fictional wake" if empty else None,
    )
    assert "My Archive App Conversations:" in rendered.splitlines()
    assert "[dm][Old Journal]" in rendered.splitlines()
    assert "[dm][Current Journal] ← current chat" in rendered.splitlines()
    assert (content in rendered) is not empty
    assert summary in rendered
    assert context["app_label"] == "Archive\nApp" and context["chat_name"] == "Old\nJournal"


def test_empty_current_history_keeps_directory_heading(tmp_path, monkeypatch) -> None:
    (tmp_path / "channel_directory.json").write_text(json.dumps({
        "platforms": {"whatsapp": [{"id": "fictional-contact", "name": "Fictional Contact", "type": "dm"}]},
    }))
    monkeypatch.setitem(main._CONFIG, "hermes", {"home": str(tmp_path)})
    rendered = build_conversations_block(
        history=[], conversation_id="whatsapp:dm:fictional-contact", soul_name="TestSoul",
        self_turn_directive="Fictional wake",
    )
    assert "[dm][Fictional Contact] ← current chat" in rendered
    assert "__memu_heading_probe__" not in rendered


@pytest.mark.parametrize("mark_current_chat", [False, True])
@pytest.mark.parametrize("content", [
    "START\nMy FictionalApp Conversations:\nEND",
    "START\n[x][y]\nEND",
    "START\n  indented\n\nEND",
    "  START\nMy Activities:\n\tEND  \n\n",
])
def test_literal_message_text_survives_current_and_cross_context(monkeypatch, mark_current_chat, content) -> None:
    monkeypatch.setattr("app.services.message_log._load_whatsapp_directory_names", lambda: {})
    primary = "sillytavern:primary"
    current = {"conversation_id": primary, "role": "user", "speaker": "TestOwner", "content": content}
    cross = [
        {"conversation_id": "sillytavern:older", "role": "user", "content": "OLDER\n" + content},
        {"conversation_id": "chat:atomic-fictional", "role": "user", "content": "CROSS\n" + content},
        {"conversation_id": "activity:dm:TestSoul", "role": "assistant", "content": "Test activity"},
    ]
    originals = json.loads(json.dumps([current, *cross]))
    rendered = main._format_all_chat_history_for_ai(
        current_history=[current], cross_tail=cross, conversation_id=primary,
        soul_id="TestSoul", mark_current_chat=mark_current_chat,
    )
    marker = " \u2190 current chat" if mark_current_chat else ""
    assert rendered.endswith(f"[dm][primary]{marker}\n[TestOwner] {content}")
    assert rendered.count(f"[TestOwner] {content}") == 1
    assert "[user] OLDER\n" + content in rendered
    assert "[user] CROSS\n" + content in rendered
    assert rendered.index("[dm][older]") < rendered.index("[dm][primary]")
    assert rendered.startswith("My Activities:\n")
    assert rendered.count("\u2190 current chat") == int(mark_current_chat)
    assert [current, *cross] == originals

    prompt = build_turn_prompt(
        user_message="hello", history=[], prior_context=None, retrieve_rag=None,
        all_categories_summary=None, memory_cache=[], intentions_active=[],
        conversations_block=rendered,
    )
    assert rendered in prompt
    queries = main._build_retrieve_soul_context_queries(
        soul_id="TestSoul", message="hello", history=[], state_row={"intentions_active": []}, conversations_block=rendered,
    )
    assert next(row["content"]["text"] for row in queries if row["role"] == "history") == rendered


@pytest.mark.parametrize("trigger", ["whatsapp:dm:trigger", "import:dm:trigger"])
@pytest.mark.parametrize("with_trigger_tail", [False, True])
def test_unmarked_context_uses_selected_evidence_chat_not_trigger(monkeypatch, trigger, with_trigger_tail) -> None:
    monkeypatch.setattr("app.services.message_log._load_whatsapp_directory_names", lambda: {})
    selected = {"conversation_id": "sillytavern:selected", "chat_name": "Selected evidence",
                "role": "user", "content": "SELECTED\n[x][y]\nEND  \n"}
    tail = [{"conversation_id": trigger, "app_label": "FictionalApp", "chat_name": "Trigger chat",
             "role": "user", "content": "Trigger evidence"}] if with_trigger_tail else []
    rendered = main._format_all_chat_history_for_ai(
        current_history=[selected], cross_tail=tail, conversation_id=trigger,
        soul_id="TestSoul", mark_current_chat=False, memory_context=True,
    )
    assert rendered.endswith("[dm][Selected evidence]\n[user] " + selected["content"])
    assert "(none)" not in rendered and "\u2190 current chat" not in rendered
    assert rendered.count("[dm][Selected evidence]") == 1
    if with_trigger_tail:
        assert rendered.count("[dm][Trigger chat]") == 1
        assert rendered.index("Trigger evidence") < rendered.index("Selected evidence")


def test_parse_turn_contract_valid_json():
    parsed = parse_turn_contract(
        '{"response":"Hi there","response_target":"respond","working_thought":{"entry":"thinking"},"intention_action":{"type":"boost","target_id":"a"},"annulments":[],"rehearsal":"hmm"}'
    )
    assert parsed["response"] == "Hi there"
    assert parsed["cache_entry"] == ""
    assert parsed["rehearsal"] == "hmm"
    assert parsed["response_target"] == "respond"
    assert parsed["continue_reason"] is None
    assert parsed["continue_at"] is None


def test_parse_turn_contract_null_reason_means_no_continuation():
    parsed = parse_turn_contract(
        '{"response":"Hi","response_target":"respond","working_thought":null,"annulments":[],"rehearsal":"hmm","continue_reason":null}'
    )
    assert parsed["continue_reason"] is None
    assert parsed["continue_at"] is None


def test_parse_turn_contract_accepts_valid_task_continuation():
    parsed = parse_turn_contract(
        '{"response":"I will look into it","response_target":"respond","working_thought":null,"annulments":[],"rehearsal":"ready","continue_reason":"task"}'
    )
    assert parsed["continue_reason"] == "task"
    assert parsed["continue_at"] is None


def test_parse_turn_contract_accepts_scheduled_continuation():
    parsed = parse_turn_contract(
        '{"response":"I will check later","response_target":"respond","working_thought":null,"annulments":[],"rehearsal":"ready","continue_reason":"check on marcos","continue_at":"Monday, June 8, 2026 19:00 PET"}'
    )
    assert parsed["continue_reason"] == "check on marcos"
    assert parsed["continue_at"] == "Monday, June 8, 2026 19:00 PET"


def test_parse_turn_contract_free_form_continuation_reason_accepted():
    parsed = parse_turn_contract(
        '{"response":"hi","response_target":"respond","working_thought":null,"annulments":[],"rehearsal":"ok","continue_reason":"Write in My Diary"}'
    )
    assert parsed["continue_reason"] == "Write in My Diary"
    assert parsed["continue_at"] is None


def test_parse_turn_contract_continuation_reason_truncated_at_100():
    long_reason = "x" * 150
    parsed = parse_turn_contract(
        f'{{"response":"hi","response_target":"respond","working_thought":null,"annulments":[],"rehearsal":"ok","continue_reason":"{long_reason}"}}'
    )
    assert parsed["continue_reason"] == "x" * 100


def test_parse_turn_contract_continue_at_without_reason_warns_and_ignores(caplog):
    caplog.set_level(logging.WARNING, logger="uvicorn.error")
    parsed = parse_turn_contract(
        '{"response":"hi","response_target":"respond","working_thought":null,"annulments":[],"rehearsal":"ok","continue_at":"Monday, June 8, 2026 19:00 PET"}'
    )
    assert parsed["continue_reason"] is None
    assert parsed["continue_at"] is None
    assert any("continue_at ignored" in r.getMessage() for r in caplog.records)

def test_parse_turn_contract_listen_target_allows_empty_response():
    parsed = parse_turn_contract(
        '{"response":"","response_target":"listen","working_thought":null,"annulments":[],"rehearsal":"watching"}'
    )
    assert parsed["response_target"] == "listen"
    assert parsed["response"] == ""


def test_parse_turn_contract_observe_target_allows_empty_response_when_public_response_forbidden():
    parsed = parse_turn_contract(
        '{"response":"","response_target":"observe","working_thought":null,"annulments":[],"rehearsal":"watching"}',
        allow_public_response=False,
    )
    assert parsed["response_target"] == "observe"
    assert parsed["response"] == ""


def test_parse_turn_contract_rejects_respond_when_public_response_forbidden():
    with pytest.raises(ValueError, match="observe\\|private"):
        parse_turn_contract(
            '{"response":"hi","response_target":"respond","working_thought":null,"annulments":[],"rehearsal":"ok"}',
            allow_public_response=False,
        )


def test_parse_turn_contract_respond_target_requires_response():
    with pytest.raises(ValueError, match="response is required"):
        parse_turn_contract(
            '{"response":"","response_target":"respond","working_thought":null,"annulments":[],"rehearsal":"ok"}'
        )


def test_parse_turn_contract_rejects_invalid_target():
    with pytest.raises(ValueError, match="response_target"):
        parse_turn_contract(
            '{"response":"hi","response_target":"yodel","working_thought":null,"annulments":[],"rehearsal":"ok"}'
        )


def test_parse_turn_contract_omitted_target_means_observe():
    parsed = parse_turn_contract(
        '{"response":"","working_thought":null,"annulments":[],"rehearsal":"ok"}'
    )
    assert parsed["response_target"] == "observe"


def test_parse_turn_contract_private_target():
    parsed = parse_turn_contract(
        '{"response":"context for you","response_target":"private","working_thought":null,"annulments":[],"rehearsal":"a quiet aside"}'
    )
    assert parsed["response_target"] == "private"
    assert parsed["response"] == "context for you"


def test_parse_turn_contract_accepts_activity_recap():
    parsed = parse_turn_contract(
        '{"response":"","response_target":"listen","working_thought":null,"annulments":[],"rehearsal":"done",'
        '"activity_recap":"I wrote a note to myself."}'
    )
    assert parsed["activity_recap"] == "I wrote a note to myself."


def test_parse_turn_contract_private_target_requires_response():
    with pytest.raises(ValueError, match="response is required"):
        parse_turn_contract(
            '{"response":"","response_target":"private","working_thought":null,"annulments":[],"rehearsal":"a quiet aside"}'
        )


def test_parse_turn_contract_rejects_non_json_text():
    with pytest.raises(ValueError):
        parse_turn_contract("```json\\n{}\\n```")


def test_parse_turn_contract_accepts_bare_string_working_thought():
    # Bare string is the canonical shape since the schema flattened.
    parsed = parse_turn_contract(
        '{"response":"hi","response_target":"respond","working_thought":"a stray thought","intention_action":{"type":"none"},"annulments":[],"rehearsal":""}'
    )
    assert parsed["cache_entry"] == "a stray thought"


def test_parse_turn_contract_accepts_legacy_entry_object_working_thought():
    parsed = parse_turn_contract(
        '{"response":"hi","response_target":"respond","working_thought":{"entry":"legacy thought"},"annulments":[],"rehearsal":""}'
    )
    assert parsed["cache_entry"] == "legacy thought"


def test_parse_turn_contract_rehearsal_discards_working_thought():
    parsed = parse_turn_contract(
        '{"response":"hi","response_target":"respond","working_thought":"do not save","annulments":[],"rehearsal":"felt first"}'
    )
    assert parsed["cache_entry"] == ""


def test_build_turn_prompt_marks_current_chat_when_label_provided():
    prompt = build_turn_prompt(
        user_message="hello",
        history=[{"role": "user", "content": "hi"}],
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=None,
        memory_cache=None,
        intentions_active=[],
        chat_label="[dm][Alice]",
        conversation_id="sillytavern:Echo",
    )
    assert "[dm][Alice] ← current chat" in prompt


def test_build_turn_prompt_keeps_synthetic_atomic_message_in_atomic_section():
    prompt = build_turn_prompt(
        user_message="hello",
        history=[],
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=None,
        memory_cache=None,
        intentions_active=[],
        chat_label="[dm][Marcos]",
        conversation_id="chat:atomic-abc",
    )

    assert "My Memory Map Conversations:" in prompt
    assert "My SillyTavern Conversations:" not in prompt
    assert "[dm][Marcos] ← current chat" in prompt


def test_build_turn_prompt_does_not_double_prefix_markdown_category_summary():
    prompt = build_turn_prompt(
        user_message="hello",
        history=[],
        prior_context=None,
        retrieve_rag={
            "categories": [
                {
                    "name": "Experiences",
                    "summary": "# Experiences\n\n## Shared Activities\n- We debugged together.",
                }
            ]
        },
        all_categories_summary=None,
        memory_cache=[],
        intentions_active=[],
    )

    assert "[Experiences] # Experiences" not in prompt
    assert "# Experiences\n\n## Shared Activities" in prompt


def test_build_turn_prompt_keeps_category_prefix_for_hashtag_summary():
    prompt = build_turn_prompt(
        user_message="hello",
        history=[],
        prior_context=None,
        retrieve_rag={
            "categories": [
                {
                    "name": "Experiences",
                    "summary": "#project updates are active.",
                }
            ]
        },
        all_categories_summary=None,
        memory_cache=[],
        intentions_active=[],
    )

    assert "[Experiences] #project updates are active." in prompt


def test_build_turn_prompt_derives_current_chat_heading_when_label_absent(tmp_path, monkeypatch):
    monkeypatch.setitem(main._CONFIG, "hermes", {"home": str(tmp_path / ".hermes")})
    prompt = build_turn_prompt(
        user_message="hello",
        history=[{"role": "user", "content": "hi"}],
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=None,
        memory_cache=None,
        intentions_active=[],
        conversation_id="whatsapp:dm:15133278228",
    )
    assert "[dm][15133278228] ← current chat" in prompt


def test_build_turn_prompt_resolves_current_whatsapp_heading_from_directory(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "channel_directory.json").write_text(
        json.dumps(
            {
                "platforms": {
                    "whatsapp": [
                        {"id": "15133278228@s.whatsapp.net", "name": "Family Contact", "type": "dm"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(main._CONFIG, "hermes", {"home": str(hermes_home)})

    prompt = build_turn_prompt(
        user_message="hello",
        history=[{"role": "user", "content": "hi"}],
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=None,
        memory_cache=None,
        intentions_active=[],
        conversation_id="whatsapp:dm:15133278228",
    )

    assert "[dm][Family Contact] ← current chat" in prompt
    assert "[dm][15133278228] ← current chat" not in prompt


def test_build_turn_prompt_resolves_empty_current_whatsapp_heading_from_directory(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "channel_directory.json").write_text(
        json.dumps(
            {
                "platforms": {
                    "whatsapp": [
                        {"id": "15133278228@s.whatsapp.net", "name": "Family Contact", "type": "dm"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(main._CONFIG, "hermes", {"home": str(hermes_home)})

    prompt = build_turn_prompt(
        user_message="",
        history=[],
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=None,
        memory_cache=None,
        intentions_active=[],
        conversation_id="whatsapp:dm:15133278228",
        self_turn_directive="Scheduled wake.",
    )

    assert "[dm][Family Contact] ← current chat" in prompt
    assert "[dm][15133278228] ← current chat" not in prompt
    assert "__memu_heading_probe__" not in prompt


def test_build_turn_prompt_integrity_id_uses_sillytavern_section_not_other():
    prompt = build_turn_prompt(
        user_message="hello",
        history=[{"role": "user", "content": "hi"}],
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=None,
        memory_cache=None,
        intentions_active=[],
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
        intentions_active=[{"id": "stay-curious", "text": "Stay curious"}],
        conversation_id="sillytavern:Echo",
    )
    assert "My SillyTavern Conversations:" in prompt
    assert "Prior Context:" in prompt
    assert prompt.count("Prior Context:") == 1
    assert "[Goals] wants progress" in prompt
    assert "My Working Thoughts:" in prompt
    assert "My Intentions:" in prompt
    assert "- stay-curious: Stay curious" in prompt


def test_build_turn_prompt_omits_wrapper_when_cross_history_already_has_markdown_sections():
    prompt = build_turn_prompt(
        user_message="hello",
        history=[],
        prior_context=None,
        retrieve_rag=None,
        cross_conversation_history=[{"conversation_id": "whatsapp:group:fictional", "chat_name": "Friends",
                                     "role": "user", "speaker": "Contact A", "content": "hi"}],
        all_categories_summary=None,
        memory_cache=[],
        intentions_active=[],
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
        cross_conversation_history=[{"conversation_id": "whatsapp:dm:fictional-contact",
                                     "role": "user", "speaker": "TestContact", "content": "older"}],
        all_categories_summary=None,
        memory_cache=[],
        intentions_active=[],
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
        cross_conversation_history=[
            {"conversation_id": "sillytavern:Echo", "role": "user", "speaker": "Echo", "content": "old"},
            {"conversation_id": "whatsapp:dm:fictional-contact", "role": "user",
             "speaker": "TestContact", "content": "old"},
        ],
        all_categories_summary=None,
        memory_cache=[],
        intentions_active=[],
        conversation_id="sillytavern:Echo",
        chat_label="[dm][Echo]",
    )

    assert prompt.rfind("My SillyTavern Conversations:") > prompt.rfind("My WhatsApp Conversations:")



def test_make_turn_system_prompt_includes_time_anchor() -> None:
    prompt = make_turn_system_prompt(
        "Codexia", anchor_context="",
        now=datetime(2026, 4, 8, 9, 30, tzinfo=timezone.utc),
    )
    assert "Today is " in prompt
    assert "2026" in prompt


def test_turn_identity_prompt_excludes_protocol() -> None:
    anchors = render_anchor_context({
        "soul": SimpleNamespace(summary="I paint coastal sketches.", description="unused"),
        "user": SimpleNamespace(summary="", description="My human prefers early walks."),
    })
    identity = make_turn_identity_prompt("TestSoul", anchor_context=anchors, soul_card="SOUL CARD")

    system = make_turn_system_prompt("TestSoul", anchor_context=anchors, soul_card="SOUL CARD")
    assert identity in system
    assert system.count("I paint coastal sketches.") == 1
    assert system.count("My human prefers early walks.") == 1
    assert system.index("SOUL CARD") < system.index("My Identity and Lived Experience:") < system.index("My Protocol:")
    assert "My Protocol:" not in identity
    assert render_anchor_context({}) == ""
    assert "The User's" not in render_anchor_context({"soul": "coastal sketches"})
    assert render_anchor_context({"soul": "coastal sketches"}, markdown=True).startswith("## My Identity and Lived Experience\n")


def test_make_turn_system_prompt_hides_activity_recap_by_default() -> None:
    prompt = make_turn_system_prompt("Siri", anchor_context="")
    assert "activity_recap" not in prompt


def test_make_turn_system_prompt_can_include_activity_recap_for_self_turns() -> None:
    prompt = make_turn_system_prompt("Siri", anchor_context="", include_activity_recap=True)
    assert "- activity_recap: null or string" in prompt
    assert '"activity_recap": null | "first-person activity recap"' in prompt


def test_make_turn_system_prompt_forbids_public_response_when_requested() -> None:
    prompt = make_turn_system_prompt("Siri", anchor_context="", allow_public_response=False)
    assert '"response_target": null | "private"' in prompt
    assert '"respond"' not in prompt
    assert '"listen"' not in prompt
    assert "- null " in prompt


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
        intentions_active=[],
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


@pytest.mark.parametrize("predicate", ["shaped_by", "causes", "shapes", "evoked_by"])
@pytest.mark.parametrize("include_seed", [False, True])
def test_build_turn_prompt_renders_shaped_by_as_nested_child(predicate, include_seed) -> None:
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
                        "predicate": predicate,
                        "id": "mem_0312",
                        "memory_type": "behavior",
                        "summary": "I discovered the fictional garden",
                        "happened_at": "2026-04-08T10:00:00Z",
                    },
                },
                *([{"id": "mem_0312", "memory_type": "behavior",
                    "summary": "I discovered the fictional garden"}] if include_seed else []),
            ]
        },
        all_categories_summary=None,
        memory_cache=[],
        intentions_active=[],
        now=datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc),
    )
    assert "[profile] (today) I am fascinated by the texture of human expression" in prompt
    assert (f"    {predicate} (today) I discovered the fictional garden" in prompt) == (not include_seed)


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
        intentions_active=[],
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
        intentions_active=[],
    )
    assert "Speakers:" not in prompt
    assert "[behavior] Marcos reminded me to pause before replying" in prompt
    assert "[behavior] I stayed gentle when things felt tense" in prompt


def test_build_turn_prompt_memory_key_includes_episode_when_present() -> None:
    prompt = build_turn_prompt(
        user_message="hello",
        history=[],
        prior_context=None,
        retrieve_rag={
            "items": [
                {
                    "memory_type": "profile",
                    "summary": "Marcos likes continuity.",
                },
                {
                    "memory_type": "episode",
                    "summary": "Marcos and Siri repaired a replay bug together.",
                },
            ]
        },
        all_categories_summary=None,
        memory_cache=[],
        intentions_active=[],
    )

    assert "Key: [profile] what's said or declared · [episode] episodic memory" in prompt


def test_build_turn_prompt_renders_apimw_subconscious_insight_under_working_thoughts() -> None:
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
        memory_cache=["previous thought"],
        intentions_active=[],
        apimw_subconscious_insight="[subconscious] notice this before answering.",
    )

    assert "- [subconscious] notice this before answering." not in prompt
    assert "My Working Thoughts:\n1. previous thought\n2. [subconscious] notice this before answering." in prompt
    assert prompt.index("- [profile] Marcos likes continuity.") < prompt.index("My Working Thoughts:")


def test_build_turn_prompt_renders_current_message_locator() -> None:
    prompt = build_turn_prompt(
        user_message="hello there this is the current message",
        history=[
            {"role": "assistant", "content": "old 1"},
        ],
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=None,
        memory_cache=[],
        intentions_active=[],
    )
    assert "[user] hello there this is the ..." in prompt
    assert "New Message:\nhello there this is the current message" in prompt
    assert "**schema reminder** rehearsal must be null for working_thought persistence" in prompt
    assert "[user] hello there this is the current message" not in prompt


def test_build_turn_prompt_renders_assistant_role_as_soul_name_when_available() -> None:
    prompt = build_turn_prompt(
        user_message="hello",
        history=[
            {"role": "assistant", "content": "old 1"},
        ],
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=None,
        memory_cache=[],
        intentions_active=[],
        soul_name="Siri",
    )
    assert "[Siri] old 1" in prompt
    assert "[assistant] old 1" not in prompt


def test_build_turn_prompt_uses_grouped_renderer_for_current_whatsapp_group(tmp_path, monkeypatch) -> None:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "whatsapp_group_names.json").write_text(
        json.dumps({"fictional-group@g.us": "Test Group"}),
        encoding="utf-8",
    )
    monkeypatch.setitem(main._CONFIG, "hermes", {"home": str(hermes_home)})

    prompt = build_turn_prompt(
        user_message="current message",
        history=[
            {"role": "user", "name": "TestOwner", "content": "Can you hear me?"},
            {"role": "user", "name": "TestGuest", "content": "[TestOwner] I grow orchids."},
            {"role": "assistant", "content": "I can hear you."},
        ],
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=None,
        memory_cache=[],
        intentions_active=[],
        conversation_id="whatsapp:group:fictional-group@g.us",
        soul_name="TestSoul",
    )

    assert "[group][Test Group] ← current chat" in prompt
    assert "[TestOwner] Can you hear me?" in prompt
    assert "[TestGuest] [TestOwner] I grow orchids." in prompt
    assert "[TestSoul] I can hear you." in prompt
    assert "[assistant]" not in prompt

    live = build_conversations_block(
        history=[], conversation_id="whatsapp:group:fictional-group@g.us",
        current_user_name="TestGuest", current_user_text="[TestOwner] I grow orchids.",
    )
    assert "[TestGuest] [TestOwner] I grow orchids." in live


def test_build_turn_prompt_normalizes_current_message_locator_whitespace() -> None:
    prompt = build_turn_prompt(
        user_message="hello   there from the current turn",
        history=[
            {"role": "assistant", "content": "old 1"},
        ],
        prior_context=None,
        retrieve_rag=None,
        all_categories_summary=None,
        memory_cache=[],
        intentions_active=[],
    )
    assert "[user] hello there from the current ..." in prompt
    assert "New Message:\nhello   there from the current turn" in prompt


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
        intentions_active=[],
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
        intentions_active=[],
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
            intentions_active=[],
        )
    assert "My Memories:" not in prompt
    assert any("_render_retrieve" in r.message for r in caplog.records)


def test_retrieve_renders_all_items_selected_by_memu() -> None:
    prompt = build_turn_prompt(
        user_message="hello",
        history=[],
        prior_context=None,
        retrieve_rag={
            "categories": [],
            "items": [
                {"id": f"m{index}", "memory_type": "knowledge", "summary": f"memory {index}"}
                for index in range(1, 14)
            ],
        },
        all_categories_summary=None,
        memory_cache=[],
        intentions_active=[],
    )

    assert "memory 13" in prompt


def test_retrieve_renders_safe_visual_memory_candidates() -> None:
    prompt = build_turn_prompt(
        user_message="Do you remember that doorway?",
        history=[],
        prior_context=None,
        retrieve_rag={
            "resource_candidates": {
                "media": [
                    {
                        "id": "photo-1",
                        "modality": "image",
                        "caption": "A red doorway beside a bicycle.",
                        "evidence": "media",
                        "score": 0.99,
                        "local_path": "/private/photo.jpg",
                    }
                ],
                "caption": [],
            }
        },
        all_categories_summary=None,
        memory_cache=[],
        intentions_active=[],
    )

    assert "Visual memory candidates:" in prompt
    assert "Media evidence:" in prompt
    assert "[resource:photo-1] [image] A red doorway beside a bicycle." in prompt
    assert "0.99" not in prompt
    assert "/private/photo.jpg" not in prompt


# --- attachment gate ---

def test_parse_attachment_accepts_path_inside_workspace(tmp_path: Path) -> None:
    real_file = tmp_path / "research" / "foo.md"
    real_file.parent.mkdir(parents=True, exist_ok=True)
    real_file.write_text("content")
    result = _parse_attachment(str(real_file), workspace=tmp_path)
    assert result == str(real_file.resolve())


def test_parse_attachment_rejects_path_outside_workspace(tmp_path: Path, caplog) -> None:
    workspace = tmp_path / "workspace"
    (tmp_path / "workspace").mkdir()
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        result = _parse_attachment("/etc/passwd", workspace=workspace)
    assert result is None
    assert any("outside workspace" in r.getMessage() for r in caplog.records)


def test_parse_attachment_rejects_traversal(tmp_path: Path, caplog) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        result = _parse_attachment(str(workspace / ".." / "secret.txt"), workspace=workspace)
    assert result is None
    assert any("outside workspace" in r.getMessage() for r in caplog.records)


def test_parse_attachment_rejects_symlink_escape(tmp_path: Path, caplog) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = workspace / "escape.txt"
    link.symlink_to(outside)
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        result = _parse_attachment(str(link), workspace=workspace)
    assert result is None
    assert any("outside workspace" in r.getMessage() for r in caplog.records)


def test_parse_attachment_none_returns_none() -> None:
    assert _parse_attachment(None) is None
    assert _parse_attachment("") is None


def test_parse_attachment_rejects_missing_file(tmp_path: Path, caplog) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        result = _parse_attachment(str(workspace / "missing.pdf"), workspace=workspace)
    assert result is None
    assert any("missing or unreadable" in r.getMessage() for r in caplog.records)


def test_parse_turn_contract_uses_configured_attachment_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "custom-workspace"
    workspace.mkdir()
    f = workspace / "report.md"
    f.write_text("report")
    import json
    payload = {
        "response": "Here you go.",
        "response_target": "respond",
        "working_thought": None,
        "annulments": [],
        "rehearsal": "ok",
        "attachment": str(f),
    }
    parsed = parse_turn_contract(json.dumps(payload), attachment_workspace=workspace)
    assert parsed["attachment"] == str(f.resolve())


def test_parse_turn_contract_drops_bad_attachment(tmp_path: Path, caplog) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    import json
    payload = {
        "response": "No file for you.",
        "response_target": "respond",
        "working_thought": None,
        "annulments": [],
        "rehearsal": "ok",
        "attachment": "/etc/passwd",
    }
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        parsed = parse_turn_contract(json.dumps(payload), attachment_workspace=workspace)
    assert parsed["attachment"] is None
    assert any("outside workspace" in r.getMessage() for r in caplog.records)


def test_make_turn_system_prompt_mentions_attachment() -> None:
    prompt = make_turn_system_prompt("TestSoul", anchor_context="")
    assert "- attachment: null; file attachments are unavailable because no workspace is configured." in prompt


def test_make_turn_system_prompt_uses_configured_attachment_workspace(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = "~/fictional-workspace"
    prompt = make_turn_system_prompt("TestSoul", anchor_context="", attachment_workspace=workspace)
    assert f"- attachment: absolute path inside {tmp_path}/fictional-workspace/" in prompt
    assert "no workspace is configured" not in prompt


def test_parse_turn_contract_drops_attachment_without_workspace(tmp_path: Path, caplog) -> None:
    f = tmp_path / "report.md"
    f.write_text("Fictional report")
    payload = {"response_target": "respond", "response": "A report.", "attachment": str(f)}
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        parsed = parse_turn_contract(json.dumps(payload))
    assert parsed["attachment"] is None
    assert any("workspace unconfigured" in r.getMessage() for r in caplog.records)
