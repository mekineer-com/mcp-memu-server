import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main
from app.db import sqlite_ensure_conversation_state_schema
from app.services import conversation_sources
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
    state, _, _ = main._load_turn_state_and_soul_card(upload["conversation_id"], user_id="TestOwner", soul_id="TestSoul")
    assert state["digest_cursor"] == 3 and state["last_memorize_at"] == "2025-01-01T12:00:00Z"
    assert register(request)["import_state"] == {**saved, "history_end_index": 6}
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
