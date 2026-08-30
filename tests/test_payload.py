from app.services.payload import _normalize_conversation


def test_normalize_conversation_preserves_speaker_as_name_for_memorize() -> None:
    out = _normalize_conversation([
        {
            "role": "user",
            "speaker": "Contact A",
            "content": "hi Siri",
            "source_label": "whatsapp:dm",
        }
    ])

    assert out == [
        {
            "role": "user",
            "name": "Contact A",
            "content": "hi Siri",
            "speaker": "Contact A",
            "source_label": "whatsapp:dm",
        }
    ]


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
