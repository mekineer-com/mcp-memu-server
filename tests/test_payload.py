from copy import deepcopy

import pytest

from app.services.payload import _label_owner_messages, _normalize_conversation, _normalize_turn_history


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


@pytest.mark.parametrize("normalize", [_normalize_conversation, _normalize_turn_history])
def test_normalize_conversation_preserves_mentra_event_metadata(normalize) -> None:
    out = normalize([
        {
            "role": "assistant",
            "content": "Fictional reflection.",
            "event_id": "fictional-sitting:2",
            "sequence": 2,
            "event_kind": "sitting_summary",
            "transcript_status": "complete",
            "media_ref": "mentra_media/fictional/image.png",
        },
        {"role": "user", "content": "Fictional view.", "event_kind": "transcript",
         "event_id": "fictional-sitting:3", "sequence": 3, "mic_source": "glasses"},
    ])

    assert out[0]["event_id"] == "fictional-sitting:2"
    assert out[0]["sequence"] == 2
    assert out[0]["event_kind"] == "sitting_summary"
    assert out[0]["transcript_status"] == "complete"
    assert out[0]["media_ref"] == "mentra_media/fictional/image.png"
    assert "mic_source" not in out[0]
    assert out[1]["mic_source"] == "glasses"
    assert _normalize_conversation(normalize(out))[1]["mic_source"] == "glasses"


@pytest.mark.parametrize("user_name", ["FirstHuman", "SecondHuman"])
def test_owner_labels_use_evidence_without_changing_sources(user_name):
    rows = [
        {"role": "user", "name": "Persona", "source_label": "sillytavern"},
        {"role": "unknown", "name": "OtherPersona", "source_label": "sillytavern"},
        {"role": "user", "source_label": "atomic"},
        {"role": "user", "source_label": "mentra"},
        {"role": "user", "speaker": "PhoneName", "source_label": "whatsapp:group"},
        {"role": "unknown", "name": "Guest", "source_label": "whatsapp:group", "source_ref": "whatsapp:guest"},
        {"role": "assistant", "name": "TestSoul", "source_label": "whatsapp:group"},
        {"role": "user", "name": "ImportedName", "source_label": "import", "source_owner_id": "TestOwner"},
        {"role": "user", "name": "Unproven", "source_label": "import"},
        {"role": "user", "name": "WrongOwner", "source_owner_id": "OtherOwner"},
    ]
    for index, row in enumerate(rows):
        row.update(content="unchanged text", source_message_id=str(index), source_conversation_index=index, ts_ms=123)
    original = deepcopy(rows)
    projected = _label_owner_messages(rows, user_id="TestOwner", user_name=user_name)
    owners = {0, 2, 3, 4, 7}
    for index, row in enumerate(projected):
        if index in owners:
            assert row["name"] == row["speaker"] == user_name
            assert row["source_owner_id"] == "TestOwner"
            assert {k: v for k, v in row.items() if k not in {"name", "speaker", "source_owner_id"}} == {
                k: v for k, v in rows[index].items() if k not in {"name", "speaker", "source_owner_id"}
            }
        else:
            assert row == rows[index]
    assert rows == original
    assert all(a is not b for a, b in zip(rows, projected))
    for normalize in (_normalize_conversation, _normalize_turn_history):
        assert normalize(projected)[7]["source_owner_id"] == "TestOwner"
    from memu.app.memorize_speakers import _build_speaker_map
    speakers = _build_speaker_map(projected, {"user_id": "TestOwner", "soul_id": "TestSoul"})
    assert all(speakers[index] == ("user:testowner", user_name) for index in owners)
    assert speakers[1] != ("user:testowner", user_name)


@pytest.mark.parametrize("marker,expected", [(None, "PhoneName"), ("OtherOwner", "PhoneName"), ("TestOwner", "SoulHuman")])
def test_live_whatsapp_requires_explicit_owner_marker(marker, expected):
    projected = _label_owner_messages([
        {"role": "user", "name": "PhoneName", "source_owner_id": marker},
    ], user_id="TestOwner", user_name="SoulHuman", conversation_id="whatsapp:group:fictional")
    assert projected[0]["name"] == expected


@pytest.mark.parametrize("role", ["user", "human", "participant", " USER "])
def test_direct_memorize_owner_roles_preserve_ids_and_role(role):
    row = {"role": role, "content": "fictional words"}
    projected = _label_owner_messages([row], user_id="TestOwner", user_name="SoulHuman")[0]
    assert projected["role"] == role
    assert projected["name"] == projected["speaker"] == "SoulHuman"
