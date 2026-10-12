import asyncio
from copy import deepcopy
from datetime import UTC, datetime
import json
import sqlite3
from types import SimpleNamespace

import pytest

from app import main
from tests import AnchorService, SavedBatchService
from app.services import apimw, conversation_sources
from app import procedural
from fastapi import HTTPException
from memu.database.models import Triple


@pytest.mark.asyncio
async def test_prior_context_renders_current_scoped_rows_without_consuming_selection(tmp_path, monkeypatch):
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    cid = "chat:prior-context"
    svc = SavedBatchService.make_engine(scope)
    repo = svc.database.memory_item_repo
    day = datetime(2026, 1, 2, 12, tzinfo=UTC)
    def item(summary, user=scope):
        return repo.create_item(memory_type="knowledge", summary=summary, embedding=[1.0, 0.0],
                                user_data=user, happened_at=day)
    first, seed, last, deleted, merged, superseded = [item(name) for name in (
        "Old selected prose", "Old supporting prose", "Last selected memory", "Deleted prose",
        "Merged prose", "Superseded prose")]
    foreign = item("Other Soul private prose", {"user_id": "TestOwner", "soul_id": "OtherSoul"})
    selection = [{"id": last.id}, {"id": first.id, "via_graph": "evokes",
                  "shaped_by": {"id": seed.id, "predicate": "evokes"}},
                 *({"id": row.id} for row in (deleted, merged, superseded, foreign)),
                 {"id": "guidance", "memory_type": "procedural", "domain": "mental_health"}]
    main._write_conversation_state(cid, **scope, updates={"prior_context": selection,
        "append_prior_context_ids_since_consolidation": [first.id, last.id]})
    repo.update_item(item_id=first.id, summary="Corrected selected prose")
    repo.update_item(item_id=seed.id, summary="Corrected supporting prose")
    repo.hard_delete_item(deleted.id, where=scope)
    repo.update_item(item_id=merged.id, merged_into=last.id)
    svc.database.triple_repo.add(Triple(subject_id=superseded.id, subject_kind="memory",
        predicate="evolved_into", object_id=last.id, object_kind="memory"), user_data=scope)
    sidecar = tmp_path / "procedural.db"
    with sqlite3.connect(sidecar) as con:
        procedural.ensure_schema(con)
        con.executemany("INSERT INTO procedural_memory_items (id, domain, summary) VALUES (?, ?, ?)",
            [("guidance", "mental_health", "Old curated guidance"),
             ("guidance", "another_domain", "Wrong domain guidance")])
        con.execute("UPDATE procedural_memory_items SET summary = 'Current curated guidance' WHERE domain = 'mental_health'")
    current = apimw.render_prior_context(selection, svc=svc, scope=scope, procedural_db_path=sidecar,
                                        now=datetime(2026, 1, 3, 12, tzinfo=UTC))
    assert "yesterday" in current and "today" not in current
    assert "Corrected selected prose" in current and "Corrected supporting prose" in current
    assert current.index(last.summary) < current.index("Corrected selected prose")
    assert "Current curated guidance" in current
    assert all(text not in current for text in ("Old selected", "Old supporting", "Deleted prose",
        "Merged prose", "Superseded prose", "Other Soul", "Old curated", "Wrong domain"))
    repo.hard_delete_item(seed.id, where=scope)
    with sqlite3.connect(sidecar) as con:
        con.execute("DELETE FROM procedural_memory_items WHERE domain = 'mental_health'")
    current = apimw.render_prior_context(selection, svc=svc, scope=scope, procedural_db_path=sidecar)
    assert "Corrected selected prose" in current and "Corrected supporting" not in current
    assert "guidance" not in current
    absent = tmp_path / "absent.db"
    assert procedural.get_item(absent, domain="mental_health", item_id="guidance") is None
    assert not absent.exists()
    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_kw: svc)
    async def retrieve(*_a, **_kw):
        return {"needs_retrieval": False, "items": []}
    monkeypatch.setattr(svc, "retrieve", retrieve)
    out = await main._run_retrieve({"user": scope, "where": {**scope, "id__in": [last.id]},
                                    "conversation_id": cid, "query": "Fictional garden"})
    assert out["prior_context"] == current
    state = main._load_turn_state_and_soul_card(cid, **scope)[0]
    assert state["prior_context"] == selection
    with main._sqlite_connect(main._sqlite_current_path(**scope)) as con:
        stored = con.execute("SELECT prior_context FROM conversations WHERE conversation_id = ?", (cid,)).fetchone()[0]
        assert json.loads(stored) == selection and "prose" not in stored
    updated, _, _ = main._turn_state_write(cid, scope["user_id"], scope["soul_id"], "", [], [],
        svc=svc, scope=scope, prepared_annulments=[])
    assert updated["prior_context"] is None
    assert updated["prior_context_ids_since_consolidation"] == [first.id, last.id]
    await main.conversation_turn_undo(cid, {"user": scope, "turn_id": updated["undo_snapshot"]["turn_id"]})
    assert main._load_turn_state_and_soul_card(cid, **scope)[0]["prior_context"] is None


@pytest.mark.parametrize("invalid", ["Old prose", ["memory"], [{}], [{"id": ""}],
    [{"id": "memory", "summary": "Duplicated prose"}],
    [{"id": "guidance", "memory_type": "procedural"}],
    [{"id": "memory", "shaped_by": {"id": "seed"}}],
    [{"id": "memory", "memory_type": "knowledge"}], [{"id": "memory", "domain": "mental_health"}]])
def test_prior_context_refuses_nonreference_state_before_writes(invalid):
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    cid = "chat:invalid-prior-context"
    before = main._write_conversation_state(cid, **scope, updates={})[0]
    with pytest.raises(HTTPException) as failure:
        main._write_conversation_state(cid, **scope, updates={"prior_context": invalid, "memory_cache": ["Must not save"]})
    assert failure.value.status_code == 400
    assert main._load_turn_state_and_soul_card(cid, **scope)[0] == before


def test_prior_context_normalizes_identifiers_without_mutating_input():
    from app.services.state import normalize_prior_context
    references = [{"id": " memory ", "shaped_by": {"id": " seed ", "predicate": " evokes "}},
                  {"id": " guidance ", "memory_type": " procedural ", "domain": " mental_health "}]
    original = deepcopy(references)
    assert normalize_prior_context(references) == [
        {"id": "memory", "shaped_by": {"id": "seed", "predicate": "evokes"}},
        {"id": "guidance", "memory_type": "procedural", "domain": "mental_health"}]
    assert references == original


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["unchanged", "consume", "failure", "cancel", "replace", "retire", "paused", "before_start", "stale_prior"])
async def test_apimw_embedding_wait_preserves_replies_and_publication(monkeypatch, change):
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    cid = "chat:apimw-concurrent"
    monkeypatch.setattr(main, "_MEMORIZE_LOCKS", {})
    main._write_conversation_state(cid, **scope, updates={"prior_context": [{"id": "old"}],
        "last_background_error": "apimw_failed: old", "last_background_error_at": "2026-01-01T00:00:00Z"})
    path = main._sqlite_current_path(**scope)
    identity = apimw._apimw_database_identity(path)
    entered, release = asyncio.Event(), asyncio.Event()
    items = []
    async def embed(*_a, **_kw):
        entered.set()
        await release.wait()
        if change == "failure":
            raise RuntimeError("Fictional embedding failure")
        return [[0.1, 0.2]]
    svc = SimpleNamespace(embed=embed, database=SimpleNamespace(
        memory_item_repo=SimpleNamespace(create_item=lambda **kwargs: items.append(kwargs))))
    def state():
        return main._load_turn_state_and_soul_card(cid, **scope)[0]
    def replace():
        path.rename(path.with_suffix(".old"))
        main._write_conversation_state(cid, **scope, updates={"apimw_subconscious_insight": "New database note"})
    kwargs = dict(svc=svc, result_json={"subconscious_insight": "Fictional quiet signal"},
                  items_by_id={}, id_map={}, scope=scope, conversation_id=cid,
                  **scope, expected_prior_context=[{"id": "old"}], database_identity=identity)
    if change in {"before_start", "stale_prior"}:
        if change == "before_start":
            replace()
        else:
            main._write_conversation_state(cid, **scope, updates={"prior_context": [{"id": "new"}]})
        before = state()
        await main._apimw_persist(**kwargs)
        assert not entered.is_set() and items == [] and state() == before
        return
    task = asyncio.create_task(main._apimw_persist(**kwargs))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        lock = main._get_memorize_lock(main._memorize_lock_key(**scope))
        await asyncio.wait_for(lock.acquire(), 1)
        try:
            assert state()["apimw_subconscious_insight"] == "[subconscious] Fictional quiet signal"
            assert state()["last_background_error"] is None and items == []
            if change == "replace":
                replace()
            else:
                updates = {"memory_cache": ["Concurrent reply"], "last_background_error": "apimw_failed: newer",
                           "last_background_error_at": "2026-01-02T00:00:00Z"}
                if change in {"consume", "cancel"}:
                    updates.update(prior_context=None, apimw_subconscious_insight=None)
                if change == "paused":
                    updates["memorize_failure"] = {"conversation_id": cid, "paused": True,
                        "targets": {}, "error": "Fictional Memorize failure"}
                main._write_conversation_state(cid, **scope, updates=updates)
                if change == "retire":
                    conversation_sources.retire_sillytavern_source(
                        storage_dir=main._get_storage_dir(main._CONFIG), conversation_id=cid, **scope)
        finally:
            lock.release()
        before = state()
        if change == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            await task
        assert state() == before
        assert len(items) == (change in {"unchanged", "consume"})
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_apimw_launcher_rejects_rebuild_before_worker_start(monkeypatch):
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    cid = "chat:apimw-before-worker"
    monkeypatch.setattr(main, "_BACKGROUND_TASKS", set())
    monkeypatch.setattr(main, "_APIMW_INFLIGHT", set())
    main._write_conversation_state(cid, **scope, updates={})
    path = main._sqlite_current_path(**scope)
    monkeypatch.setattr(main, "_apimw_cadence_due", lambda *_a: True)
    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_kw: SimpleNamespace(
        list_active_dossiers=lambda *_a: pytest.fail("stale worker must not start provider preparation")))
    assert main._turn_launch_apimw(cid, "TestOwner", "TestSoul", {}, []) == "started"
    tasks = tuple(main._BACKGROUND_TASKS)
    path.rename(path.with_suffix(".old"))
    main._write_conversation_state(cid, **scope, updates={"apimw_subconscious_insight": "New database note"})
    await asyncio.gather(*tasks)
    assert main._load_turn_state_and_soul_card(cid, **scope)[0]["apimw_subconscious_insight"] == "New database note"
    assert cid not in main._APIMW_INFLIGHT


@pytest.mark.asyncio
@pytest.mark.parametrize("cli", [False, True])
async def test_st_apimw_captures_srmm_and_reflection_with_admission_top_k(monkeypatch, cli):
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    cid = "integrity:fictional-apimw"
    entered, release = asyncio.Event(), asyncio.Event()
    services, calls, persisted = [], [], []
    path = main._sqlite_current_path(**scope)
    monkeypatch.setitem(main._CONFIG, "claude_code", cli)
    monkeypatch.setitem(main._CONFIG, "apimw_test_model", "captured")
    monkeypatch.setattr(main, "_apimw_memory_count_from_cfg", lambda cfg: cfg["apimw_test_k"])
    monkeypatch.setitem(main._CONFIG, "apimw_test_k", 17)
    monkeypatch.setattr(main, "_apimw_random_count_from_cfg", lambda _cfg: 0)

    class Service(AnchorService):
        def __init__(self, operation, payload):
            self.operation = operation
            self.model = main._CONFIG["apimw_test_model"]
            self._claude_code = cli if operation == "retrieve" else False
            self.retrieve_config = SimpleNamespace(item=SimpleNamespace(top_k=payload.get("retrieve_config", {}).get("item", {}).get("top_k")))

        def list_active_dossiers(self, _scope):
            return []

        def build_dossier_index(self, *_args, **_kwargs):
            return ""

        async def chat(self, prompt, **kwargs):
            calls.append((self, kwargs))
            return '{"prior_context":[],"subconscious_insight":null}'

    def construct(payload, *, operation=None):
        assert operation in {"retrieve", "st_reply"}
        assert payload["user"] == scope
        assert "llm_profiles" not in payload and "claude_code" not in payload
        if operation == "retrieve":
            assert payload["retrieve_config"]["category"]["top_k"] == 5
        service = Service(operation, payload)
        services.append(service)
        return service

    async def retrieve(payload, *, service, **kwargs):
        assert service is services[0]
        assert service.operation == "retrieve" and service._claude_code is cli
        assert service.retrieve_config.item.top_k == 17
        assert payload["retrieve_config"]["item"]["top_k"] == 17
        assert "llm_profiles" not in payload and "claude_code" not in payload
        assert payload["force_retrieve"] is True
        entered.set()
        await release.wait()
        return {"result": {"items": []}}

    async def persist(service, **kwargs):
        assert kwargs["database_identity"] == apimw._apimw_database_identity(path)
        persisted.append(service)

    monkeypatch.setattr(main, "_get_service_from_payload", construct)
    monkeypatch.setattr(main, "_run_retrieve", retrieve)
    monkeypatch.setattr(main, "_apimw_persist", persist)
    monkeypatch.setattr(main, "_apimw_cadence_due", lambda *_args: True)
    monkeypatch.setattr(main, "_load_turn_state_and_soul_card", lambda *_a, **_kw: ({"intentions_active": [], "memory_cache": []}, None, path))
    monkeypatch.setattr(main, "_current_history_for_ai", lambda _cid, _uid, _sid, history, _state, **_kw: history)
    monkeypatch.setattr(main, "_load_cross_tail_for_ai", lambda **_kw: [])
    monkeypatch.setattr(main, "_soul_activity_pause", lambda *_args: None)
    monkeypatch.setattr(main, "_require_soul_active", lambda *_args: None)
    monkeypatch.setattr(main, "_clear_background_error_if_apimw_owned", lambda *_a, **_kw: None)
    monkeypatch.setattr(main, "_set_background_error", lambda *_a, **_kw: pytest.fail("APImw unexpectedly failed"))
    payload = {"user": scope, "message": "Inspect a fictional garden.",
               "llm_profiles": {"default": {"chat_model": "old-reply"}}, "claude_code": not cli,
               "retrieve_config": {"item": {"top_k": 1}, "category": {"top_k": 5}}}
    original = deepcopy(payload)
    before = set(main._BACKGROUND_TASKS)
    assert main._turn_launch_apimw(cid, "TestOwner", "TestSoul", payload, []) == "started"
    tasks = main._BACKGROUND_TASKS - before
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        main._CONFIG["apimw_test_model"] = "later"
        main._CONFIG["apimw_test_k"] = 3
        main._CONFIG["claude_code"] = not cli
    finally:
        release.set()
        await asyncio.gather(*tasks)
    assert len(services) == 2
    assert services[1].operation == "st_reply" and services[1]._claude_code is False
    assert calls[0][0] is services[1] and calls[0][1]["profile"] is None
    assert [service.model for service in services] == ["captured", "captured"]
    assert persisted == [services[1]]
    assert payload == original
    assert cid not in main._APIMW_INFLIGHT
