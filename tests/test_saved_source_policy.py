"""Offline regressions for captured policy and unpaid frozen preparation."""

from copy import deepcopy
import json

import pytest
from fastapi import BackgroundTasks

from app import main
from app.services import conversation_sources


UID, SID, CID = "PolicyOwner", "PolicySoul", "chat:policy"
STAMP = "2026-10-01T00:00:00+00:00"


@pytest.fixture
def saved_chat(monkeypatch):
    monkeypatch.setattr(main, "_MEMORIZE_PROGRESS", {})
    monkeypatch.setattr(main, "_FORCED_MEMORIZE_INFLIGHT", {})
    monkeypatch.setattr(main, "_FORCED_MEMORIZE_RECHECK", {})
    monkeypatch.setattr(main, "_CONSOLIDATION_RUNNING", {})
    storage = main._get_storage_dir(main._CONFIG)
    monkeypatch.setattr(main, "_resolve_cross_source_paths", lambda: (storage, None, None, None))
    rows = [
        {"role": "user", "name": UID, "content": f"captured row {index}",
         "source_conversation_index": index, "ts_ms": 1790812800000 + index * 86400000}
        for index in range(6)
    ]
    main._write_conversation_state(CID, user_id=UID, soul_id=SID, updates={
        "digest_cursor": 4, "last_memorize_at": STAMP,
    })
    conversation_sources.persist_sillytavern_history_snapshot(
        storage_dir=storage, user_id=UID, soul_id=SID, conversation_id=CID,
        history=rows, source_message_count=6,
    )
    return rows


@pytest.mark.parametrize(("producing", "rolling"), [(True, 4), (False, 4), (False, None)])
def test_retry_capture_keeps_failure_policy_after_setting_change(saved_chat, producing, rolling):
    other = "chat:other-policy"
    other_producing = not producing
    main._write_conversation_state(CID, user_id=UID, soul_id=SID, updates={
        "memorize_chat": not producing, "digest_cursor": 2, "rolling_summary_cursor_id": rolling,
    })
    main._write_conversation_state(other, user_id=UID, soul_id=SID, updates={
        "memorize_chat": not other_producing, "digest_cursor": 1, "rolling_summary_cursor_id": 3,
        "last_memorize_at": STAMP,
    })
    conversation_sources.persist_sillytavern_history_snapshot(
        storage_dir=main._get_storage_dir(main._CONFIG), user_id=UID, soul_id=SID, conversation_id=other,
        history=[{**row, "content": f"other captured row {row['source_conversation_index']}"} for row in saved_chat],
        source_message_count=6,
    )
    failure = {"conversation_id": CID, "error": "restore captured source", "paused": True,
               "targets": {CID: {"cursor": 5, "memory_producing": producing},
                           other: {"cursor": 5, "memory_producing": other_producing}}}
    original = deepcopy(failure)

    payload = main._saved_memorize_payload(CID, UID, SID, failure)

    cursor = 2 if producing else max(2, rolling if rolling is not None else -1)
    trigger_rows = [row for row in payload["conversation"] if row["source_conversation_id"] == CID]
    other_rows = [row for row in payload["conversation"] if row["source_conversation_id"] == other]
    assert [row["source_conversation_index"] for row in trigger_rows] == list(range(cursor + 1, 6))
    assert all(row["memorize_chat"] is producing for row in trigger_rows)
    other_cursor = 1 if other_producing else 3
    assert [row["source_conversation_index"] for row in other_rows] == list(range(other_cursor + 1, 6))
    assert [row["content"] for row in other_rows] == [f"other captured row {i}" for i in range(other_cursor + 1, 6)]
    assert all(row["memorize_chat"] is other_producing for row in other_rows)
    assert payload["_final_cursors"][CID] == {"cursor": 5, "memory_producing": producing}
    assert payload["_final_cursors"][other] == {"cursor": 5, "memory_producing": other_producing}
    assert failure == original


@pytest.mark.asyncio
@pytest.mark.parametrize("rolling", [1, None, 5])
async def test_manual_context_capture_slices_after_both_checkpoints(saved_chat, monkeypatch, rolling):
    main._write_conversation_state(CID, user_id=UID, soul_id=SID, updates={
        "memorize_chat": True, "rolling_summary_cursor_id": rolling,
    })
    monkeypatch.setattr(main, "_get_service_from_payload", lambda _payload: pytest.fail("background-only capture must not construct a service"))
    captures = []
    prepare = main._prepare_frozen_memorize
    def capture(safe, uid, sid):
        eligible = prepare(safe, uid, sid)
        captures.append(deepcopy(safe))
        return eligible
    monkeypatch.setattr(main, "_prepare_frozen_memorize", capture)
    tasks = BackgroundTasks()
    payload = {"intent": "manual", "user": {"user_id": UID, "soul_id": SID},
               "conversation_id": CID, "memorize_chat": False, "conversation": saved_chat}

    response = await main._memorize_owned(payload, tasks, True, admitted=True, batch_owned=True)

    assert response.status_code == 200
    assert json.loads(response.body)["status"] == "nothing_to_memorize"
    assert tasks.tasks == []
    expected = list(range(max(4, rolling if rolling is not None else -1) + 1, 6))
    assert len(captures) == bool(expected)
    if expected:
        captured = captures[0]
        assert [row["source_conversation_index"] for row in captured["conversation"]] == expected
        assert all(row["memorize_chat"] is False for row in captured["conversation"])
        assert captured["_final_cursors"][CID] == {"cursor": 5, "memory_producing": False}
    state, _, _ = main._load_turn_state_and_soul_card(CID, user_id=UID, soul_id=SID)
    assert state["digest_cursor"] == 4
    assert state["rolling_summary_cursor_id"] == rolling


@pytest.mark.parametrize("cursor", [-1, 0])
def test_frozen_lineage_context_is_retained_once(saved_chat, monkeypatch, cursor):
    cid = "whatsapp:dm:policy-lineage"
    main._write_conversation_state(cid, user_id=UID, soul_id=SID, updates={
        "digest_cursor": cursor, "last_memorize_at": STAMP,
    })
    monkeypatch.setattr(main, "_resolve_source_cursor", lambda _cid, value, *_args, **_kwargs: (value, None, True))

    def resolve(_cid, source_id):
        assert source_id == "current-1", "negative lineage must not resolve or advance a source checkpoint"
        return 1

    monkeypatch.setattr(main, "_resolve_web_source_checkpoint", resolve)
    context = {"role": "user", "name": UID, "content": "ancestor context", "ts_ms": 1790812800000,
               "source_conversation_id": cid, "source_conversation_index": -2, "source_message_id": "ancestor",
               "memorize_chat": True}
    current = {**context, "content": "current captured row", "ts_ms": 1790899200000,
               "source_conversation_index": 1, "source_message_id": "current-1"}
    safe = {"conversation_id": cid, "conversation": [context, current],
            "_final_cursors": {cid: {"cursor": 1, "memory_producing": True}}}

    assert main._prepare_frozen_memorize(safe, UID, SID)

    assert safe["conversation"] == [context, current]
    assert safe["_final_cursors"][cid] == {
        "cursor": 1, "source_message_id": "current-1", "ts": 1790899200, "memory_producing": True,
    }


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("importing", [False, True])
def test_preturn_checks_import_due_without_capturing(saved_chat, monkeypatch, dry_run, importing):
    cid = "import:dm:policy-history"
    if importing:
        main._write_conversation_state(cid, user_id=UID, soul_id=SID, updates={"import_state": {
            "history_end_index": 2, "memorize_cursor": -1, "stage": "memorize",
            "pending_segment_ids": [], "error": None, "ordinary_waiting": None,
        }})
    main._write_conversation_state(CID, user_id=UID, soul_id=SID, updates={"digest_cursor": -1})
    monkeypatch.setattr(main, "_MIN_CHUNK_TOKENS", 1)

    def forbidden(*_args, **_kwargs):
        pytest.fail("pre-turn due checks must not assemble a batch or acquire a model service")

    monkeypatch.setattr(main, "_build_cross_conversation_payload", forbidden)
    monkeypatch.setattr(main, "_get_service_from_payload", forbidden)

    result = main._turn_state_read(CID, UID, SID, {"memorize_chat": True}, [], dry_run, saved_chat)

    assert result[5] > 1
    assert result[6] is None
    if importing:
        record = main._soul_import_state(UID, SID)[1]
        assert record["ordinary_waiting"] == (None if dry_run else CID)
        assert record["memorize_cursor"] == -1
        assert record["history_end_index"] == 2
