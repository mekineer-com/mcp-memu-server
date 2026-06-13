from app.services.payload import _normalize_conversation


def test_normalize_conversation_preserves_speaker_as_name_for_memorize() -> None:
    out = _normalize_conversation([
        {
            "role": "user",
            "speaker": "Raquel",
            "content": "hi Siri",
            "source_label": "whatsapp:dm",
        }
    ])

    assert out == [
        {
            "role": "user",
            "name": "Raquel",
            "content": "hi Siri",
            "speaker": "Raquel",
            "source_label": "whatsapp:dm",
        }
    ]
