import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main
from app.db import sqlite_ensure_conversation_state_schema
from app.services import consolidation, conversation_sources
from app.services.import_routes import ImportPreview, ImportScope

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "openalma" / "launcher"))
import chat_import


def _endpoint(path):
    return next(route.endpoint for route in main.app.routes if getattr(route, "path", None) == path)


def test_registration_extends_history_without_resetting_progress(tmp_path, monkeypatch):
    source, db = tmp_path / "imports.db", tmp_path / "TestSoul.db"
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul", "label": "Replika"}
    monkeypatch.setattr(conversation_sources, "import_source_path", lambda: source)
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_kw: db)
    with sqlite3.connect(db) as con:
        sqlite_ensure_conversation_state_schema(con)
    rows, _, _ = chat_import.normalize_messages([
        {"id": str(i), "role": "user", "content": f"message {i}", "timestamp": "2025-01-01"} for i in range(5)])
    upload = chat_import.store_upload(source, **scope, messages=rows, history_count=2)
    register = _endpoint("/imports/register")
    request = ImportScope(**scope)
    result = register(request)
    assert result["import_state"] == {"history_end_index": 2, "memorize_cursor": -1,
        "pending_segment_ids": [], "stage": "memorize", "error": None}
    saved = {**result["import_state"], "memorize_cursor": 1, "pending_segment_ids": ["test-segment"],
             "stage": "consolidation", "error": "Historical failure"}
    main._write_conversation_state(upload["conversation_id"], user_id="TestOwner", soul_id="TestSoul",
                                  updates={"import_state": saved, "digest_cursor": 3,
                                           "last_memorize_at": "2025-01-01T12:00:00Z"})
    chat_import.store_upload(source, **scope, messages=rows, history_count=5)
    assert register(request)["import_state"] == saved
    older, _, _ = chat_import.normalize_messages([
        {"id": "older", "role": "assistant", "content": "older", "timestamp": "2024-01-01"}])
    chat_import.store_upload(source, **scope, messages=older, history_count=1)
    assert register(request)["import_state"] == {**saved, "history_end_index": 6}
    # An old extraction result must patch the newly registered bound, not replace it.
    main._write_conversation_state(upload["conversation_id"], user_id="TestOwner", soul_id="TestSoul",
        updates={"import_error": "Interrupted"})
    main._write_conversation_state(upload["conversation_id"], user_id="TestOwner", soul_id="TestSoul",
        updates={"import_memorize_cursor": 1, "append_import_pending_segment_ids": ["later-segment"], "import_error": None})
    patched = register(request)["import_state"]
    assert patched["history_end_index"] == 6 and patched["memorize_cursor"] == 1
    assert patched["pending_segment_ids"] == ["test-segment", "later-segment"] and patched["error"] is None
    state, _, _ = main._load_turn_state_and_soul_card(upload["conversation_id"], user_id="TestOwner", soul_id="TestSoul")
    assert state["digest_cursor"] == 3 and state["last_memorize_at"] == "2025-01-01T12:00:00Z"
    assert register(request)["import_state"] == patched
    main._write_conversation_state(upload["conversation_id"], user_id="TestOwner", soul_id="TestSoul",
        updates={"import_state": {**patched, "pending_segment_ids": [], "stage": "memorize"}})
    advanced, _ = main._write_conversation_state(upload["conversation_id"], user_id="TestOwner", soul_id="TestSoul",
        updates={"import_memorize_cursor": 5, "import_error": None})
    assert advanced["import_state"]["stage"] == "complete"
    late, _ = main._write_conversation_state(upload["conversation_id"], user_id="TestOwner", soul_id="TestSoul",
        updates={"import_memorize_cursor": 1, "import_error": "Late failure"})
    assert late["import_state"]["memorize_cursor"] == 5 and late["import_state"]["history_end_index"] == 6
    with pytest.raises(HTTPException) as refused:
        register(ImportScope(**{**scope, "soul_id": "OtherSoul"}))
    assert refused.value.status_code == 404


def test_preview_counts_stored_display_and_cross_context_without_calls_or_inserts(tmp_path, monkeypatch):
    source, db = tmp_path / "imports.db", tmp_path / "TestSoul.db"
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul", "label": "Nomi"}
    monkeypatch.setattr(conversation_sources, "import_source_path", lambda: source)
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_kw: db)
    monkeypatch.setattr(main, "_resolve_cross_source_paths", lambda: (tmp_path, None, None, None))
    with sqlite3.connect(db) as con:
        sqlite_ensure_conversation_state_schema(con)
    rows, _, _ = chat_import.normalize_messages([
        {"id": str(i), "role": "user", "name": "TestSpeaker", "content": f"stored {i}", "timestamp": "2025-01-01"}
        for i in range(3)])
    upload = chat_import.store_upload(source, **scope, messages=rows, history_count=0)
    cid = upload["conversation_id"]
    _endpoint("/imports/register")(ImportScope(**scope))
    main._write_conversation_state(cid, user_id="TestOwner", soul_id="TestSoul",
        updates={"digest_cursor": 2, "last_memorize_at": "2025-01-01T12:00:00Z",
                 "last_display_segment_start_index": 0, "last_display_segment_end_index": 2,
                 "last_display_segment_at": "2025-01-01T12:00:00Z"})
    other = "chat:other-fictional"
    main._write_conversation_state(other, user_id="TestOwner", soul_id="TestSoul", updates={})
    conversation_sources.persist_sillytavern_history_snapshot(
        storage_dir=tmp_path, user_id="TestOwner", soul_id="TestSoul", conversation_id=other,
        history=[{"role": "user", "name": "OtherSpeaker", "content": "another chat " * 200}])
    monkeypatch.setattr(main, "_load_activity_tail_for_ai", lambda *_a, **_kw: [{
        "conversation_id": "activity:dm:TestSoul", "role": "assistant", "content": "sitting recap " * 200}])
    profile = SimpleNamespace(context_window_tokens=100000, max_tokens=1000, chat_model="fictional-model")
    svc = SimpleNamespace(_claude_code=False, llm_profiles=SimpleNamespace(profiles={"default": profile}),
                          build_dossier_index=lambda _: "Known dossier index")
    monkeypatch.setattr(main, "_get_service_from_payload", lambda _: svc)
    prompts = []
    build = main._build_turn_prompt
    def capture(**kwargs):
        text = build(**kwargs)
        prompts.append(text)
        return text
    monkeypatch.setattr(main, "_build_turn_prompt", capture)
    validate = _endpoint("/imports/validate")
    message = {"role": "user", "content": "proposed", "timestamp": "2025-01-02T12:00:00Z",
               "source_day": "2025-01-02", "position": 3}
    preview = ImportPreview(**scope, conversation_id=cid, current_messages=[message])
    first = validate(preview)
    assert first["ok"] and prompts[-1].count("[TestSpeaker] stored 2") == 1
    assert "another chat" in prompts[-1] and "sitting recap" in prompts[-1]
    assert "My Nomi Conversations:" in prompts[-1] and "Known dossier index" in prompts[-1]
    client = TestClient(main.app)
    response = client.post("/imports/validate", json=preview.model_dump(mode="json"))
    assert response.status_code == 200 and response.json()["ok"]
    invalid = {**preview.model_dump(mode="json"), "current_messages": [{**message, "role": "system"}]}
    assert client.post("/imports/validate", json=invalid).status_code == 422
    profile.context_window_tokens = 1000 + (first["estimated_tokens"] + 10) * 5 // 4
    larger = ImportPreview(**scope, conversation_id=cid,
                          current_messages=[{**message, "content": "additional words " * 100}])
    with pytest.raises(HTTPException, match="context"):
        validate(larger)
    with sqlite3.connect(source) as con:
        assert con.execute("SELECT COUNT(*) FROM imported_messages").fetchone()[0] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["calendar", "calendar_retry", "capacity", "failure", "error_write", "consolidation_failure", "no_memories", "no_memories_calendar"])
async def test_import_batch_reuses_extraction_and_one_consolidation(tmp_path, monkeypatch, stop):
    import asyncio
    from app.services import memorize_endpoint
    source, db = tmp_path / "imports.db", tmp_path / "TestSoul.db"
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul", "label": "Replika"}
    scoped = {key: scope[key] for key in ("user_id", "soul_id")}
    monkeypatch.setattr(conversation_sources, "import_source_path", lambda: source)
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_kw: db)
    monkeypatch.setattr(memorize_endpoint, "_FORCE_MEMORIZE_MAX_CHUNK_TOKENS", 80)
    with sqlite3.connect(db) as con:
        sqlite_ensure_conversation_state_schema(con)
    rows, _, _ = chat_import.normalize_messages([
        {"id": str(i), "role": "user", "content": "fictional story " * 20,
         "timestamp": "2025-01-09" if stop in {"calendar", "calendar_retry", "no_memories_calendar"} and i else "2025-01-01"}
        for i in range(3)])
    stored = chat_import.store_upload(source, **scope, messages=rows, history_count=3)
    request = ImportScope(**scope)
    _endpoint("/imports/register")(request)
    cid = stored["conversation_id"]
    marker = main._memorize_lock_key(**scoped)
    extracts, consolidations, releases = [], [], []
    fail = stop in {"failure", "calendar_retry", "error_write", "consolidation_failure"}
    if stop == "error_write":
        write = main._write_conversation_state
        def failing_write(*args, **kwargs):
            if str(kwargs.get("updates", {}).get("import_error", "")).startswith("ValueError:"):
                raise sqlite3.OperationalError("Fictional error-write lock")
            return write(*args, **kwargs)
        monkeypatch.setattr(main, "_write_conversation_state", failing_write)
    profile = SimpleNamespace(context_window_tokens=100, max_tokens=0, chat_model="fictional")
    class Service:
        memorize_config = SimpleNamespace(category_update_llm_profile="default")
        llm_profiles = SimpleNamespace(profiles={"default": profile})
        async def memorize_segments_batch(self, **kwargs):
            assert kwargs["enforce_input_budget"] is True
            assert main._FORCED_MEMORIZE_INFLIGHT[marker] is True
            extracts.append([row["segment"]["segment_id"] for row in kwargs["segments"]])
            with pytest.raises(HTTPException) as busy:
                await _endpoint("/imports/process")(request)
            assert busy.value.status_code == 409
            if stop in {"failure", "calendar_retry", "error_write"} and fail and len(extracts) == 2:
                raise ValueError("Fictional extraction failure")
            if stop in {"no_memories", "no_memories_calendar"}:
                return [{} for _ in kwargs["segments"]]
            return [{"pending_segment_ids": [row["segment"]["segment_id"]]} for row in kwargs["segments"]]
    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_: Service())
    def gather(_deps, **kwargs):
        pairs = kwargs["selected_segments"]
        assert all(owner == cid for owner, _ in pairs)
        return {"segment_inputs": [{} for _ in pairs], "current_chat_messages": [
            {"source_day": rows[int(segment_id.rsplit(":", 1)[1].split("-")[0])]["source_day"]}
            for _owner, segment_id in pairs]}
    monkeypatch.setattr(consolidation, "gather_consolidation_inputs", gather)
    monkeypatch.setattr(consolidation, "_prepare_dossier_consolidation_prompts", lambda *_a, **kw:
        ([], "", "", {"dossiers": 60 if stop in {"capacity", "consolidation_failure"} and len(kw["inputs"]["segment_inputs"]) >= 2 else 10}))
    def record():
        return main._load_turn_state_and_soul_card(cid, **scoped)[0]["import_state"]
    entered, finish = asyncio.Event(), asyncio.Event()
    async def pipeline(**kwargs):
        assert kwargs["historical"] and main._FORCED_MEMORIZE_INFLIGHT[marker] is True
        consolidations.append(kwargs["selected_segments"])
        assert kwargs["selected_segments"] == {(cid, segment_id) for segment_id in record()["pending_segment_ids"]}
        assert (await main.memorize_cancel(scoped))["status"] == "cancel_requested"
        entered.set()
        await finish.wait()
        if stop == "consolidation_failure" and fail:
            raise ValueError("Fictional consolidation failure")
        current = record()
        # Simulate the existing fitting-prefix selector leaving one pending span.
        remaining = current["pending_segment_ids"][1:]
        main._write_conversation_state(cid, **scoped, updates={"import_state": {
            **current, "pending_segment_ids": remaining, "error": None,
            "stage": "consolidation" if remaining else "memorize",
        }})
        return {"status": "ok", "result": {"revised": 1}}
    monkeypatch.setattr(main, "_run_consolidation_pipeline_once", pipeline)
    release = main._finish_memorize_claim
    async def finished(key, success):
        assert key not in main._MEMORIZE_CANCEL
        releases.append((key, success))
        await release(key, success)
    monkeypatch.setattr(main, "_finish_memorize_claim", finished)
    response = await _endpoint("/imports/process")(request)
    assert response.status_code == 202
    tasks = list(main._BACKGROUND_TASKS)
    if stop not in {"failure", "calendar_retry", "error_write", "no_memories", "no_memories_calendar"}:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(HTTPException) as busy:
            await _endpoint("/imports/process")(request)
        assert busy.value.status_code == 409
        finish.set()
    await asyncio.gather(*tasks)
    assert len(extracts) == (3 if stop == "no_memories" else 2) and marker not in main._FORCED_MEMORIZE_INFLIGHT
    assert releases == [(marker, stop not in {"failure", "calendar_retry", "error_write", "consolidation_failure"})]
    if stop in {"failure", "calendar_retry", "error_write"}:
        assert record()["stage"] == "memorize" and record()["error"]
        assert record()["memorize_cursor"] == 0 and not consolidations
        with pytest.raises(HTTPException, match="Retry"):
            await _endpoint("/imports/process")(request)
        fail = False
        finish.set()
        await _endpoint("/imports/retry")(request)
        await asyncio.gather(*list(main._BACKGROUND_TASKS))
        assert len(extracts) == (3 if stop == "calendar_retry" else 4) and len(consolidations) == 1
    elif stop == "consolidation_failure":
        assert record()["stage"] == "consolidation" and record()["error"]
        assert len(consolidations) == 1
        with pytest.raises(HTTPException, match="Retry"):
            await _endpoint("/imports/process")(request)
        fail = False
        await _endpoint("/imports/retry")(request)
        await asyncio.gather(*list(main._BACKGROUND_TASKS))
        assert len(extracts) == 2 and len(consolidations) == 2
    elif stop == "no_memories":
        assert record()["stage"] == "complete" and record()["memorize_cursor"] == 2
        assert not consolidations
    elif stop == "no_memories_calendar":
        assert record()["stage"] == "memorize" and record()["memorize_cursor"] == 1
        assert not consolidations
    else:
        assert len(consolidations) == 1 and record()["pending_segment_ids"]
        await _endpoint("/imports/process")(request)
        await asyncio.gather(*list(main._BACKGROUND_TASKS))
        assert len(extracts) == 2 and len(consolidations) == 2
    assert marker not in main._FORCED_MEMORIZE_INFLIGHT
    status = _endpoint("/imports/status")(**scope)
    assert not status["running"] and status["progress"]["active"] is False
