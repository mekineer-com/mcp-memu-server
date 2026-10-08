from app.services.payload import _normalize_conversation, _normalize_turn_history


def test_normalize_conversation_preserves_speaker_as_name_for_memorize() -> None:
    source_ref = "whatsapp:fictional-bound-sender@lid"
    messages = [
        {
            "role": "user",
            "speaker": name,
            "content": "hi TestSoul",
            "source_label": "whatsapp:dm",
            "source_ref": source_ref,
        }
        for name in ("Fictional Sender", "Renamed Fictional Sender")
    ]
    out = _normalize_conversation(messages)

    assert out == [
        {
            "role": "user",
            "name": message["speaker"],
            "content": "hi TestSoul",
            "speaker": message["speaker"],
            "source_label": "whatsapp:dm",
            "source_ref": source_ref,
        }
        for message in messages
    ]

    for normalize in (_normalize_conversation, _normalize_turn_history):
        normalized = normalize(messages)
        assert [row["name"] for row in normalized] == [row["speaker"] for row in messages]
        assert [row["source_ref"] for row in normalized] == [source_ref, source_ref]
        assert [row["role"] for row in normalized] == ["user", "user"]

    assert _normalize_conversation(_normalize_turn_history(messages)) == _normalize_turn_history(messages)
    imported = [{"role": "user", "content": "Imported owner's words", "source_label": "import",
                 "source_owner_id": "TestOwner"}]
    for normalize in (_normalize_conversation, _normalize_turn_history):
        assert normalize(imported)[0]["source_owner_id"] == "TestOwner"


def test_normalize_conversation_preserves_mentra_event_metadata() -> None:
    out = _normalize_conversation([
        {
            "role": "assistant",
            "content": "Fictional reflection.",
            "event_id": "fictional-sitting:2",
            "sequence": 2,
            "event_kind": "sitting_summary",
            "transcript_status": "complete",
            "media_ref": "mentra_media/fictional/image.png",
        }
    ])

    assert out[0]["event_id"] == "fictional-sitting:2"
    assert out[0]["sequence"] == 2
    assert out[0]["event_kind"] == "sitting_summary"
    assert out[0]["transcript_status"] == "complete"
    assert out[0]["media_ref"] == "mentra_media/fictional/image.png"
