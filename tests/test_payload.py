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
