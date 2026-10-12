from __future__ import annotations

import json
import sqlite3
import asyncio

import pytest

from app.services import soul_state, soul_summaries
from app import main


def _route_endpoint(name: str):
    return next(route.endpoint for route in main.app.routes if route.name == name)


def _connection() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    soul_state.ensure_schema(con)
    return con


def test_current_schema_reopen_preserves_approved_baseline(tmp_path) -> None:
    path = tmp_path / "soul.db"
    con = sqlite3.connect(path)
    soul_state.ensure_schema(con)
    con.execute("UPDATE soul_state SET narrative_self = 'current self', narrative_self_approved = 'approved self' WHERE id = 1")
    con.commit()
    soul_state.ensure_schema(con)
    con.close()

    check = sqlite3.connect(path)
    check.row_factory = sqlite3.Row
    state = soul_state.read(check)
    assert state["narrative_self_approved"] == "approved self"
    assert state["narrative_self"] == "current self"
    assert "all_categories_summary" not in state
    check.execute("UPDATE soul_state SET narrative_self = 'new self' WHERE id = 1")
    check.commit()
    soul_state.ensure_schema(check)
    assert soul_state.read(check)["narrative_self_approved"] == "approved self"
    check.close()


def test_schema_ensure_commits_missing_singleton_row(tmp_path) -> None:
    path = tmp_path / "empty.db"
    con = sqlite3.connect(path)
    soul_state.ensure_schema(con)
    con.execute("DELETE FROM soul_state")
    con.commit()
    soul_state.ensure_schema(con)
    assert not con.in_transaction
    con.close()

    check = sqlite3.connect(path)
    assert check.execute("SELECT COUNT(*) FROM soul_state").fetchone()[0] == 1
    check.close()


def test_fresh_schema_omits_holistic_summary_columns() -> None:
    con = _connection()
    columns = {row[1] for row in con.execute("PRAGMA table_info(soul_state)")}

    assert not {
        "all_categories_summary",
        "all_categories_summary_previous",
        "all_categories_summary_approved",
    } & columns


@pytest.mark.parametrize("route_name", ["soul_summary_update", "soul_summary_approve"])
def test_dossier_index_routes_are_read_only(route_name) -> None:
    route = _route_endpoint(route_name)
    with pytest.raises(main.HTTPException) as exc_info:
        asyncio.run(route(kind="all_categories_summary", user_id="u", soul_id="s", payload={}))

    assert exc_info.value.status_code == 409


def test_soul_summary_write_and_approve() -> None:
    con = _connection()

    state = soul_summaries.write_live(
        con,
        kind="narrative_self",
        summary="first self",
    )
    assert state["narrative_self"] == "first self"
    assert state["narrative_self_previous"] is None
    assert state["narrative_self_approved"] is None
    assert state["summaries_revision"] == 1
    assert soul_summaries.write_live(con, kind="narrative_self", summary="first self")["summaries_revision"] == 1

    state = soul_summaries.approve(con, kind="narrative_self")
    assert state["narrative_self_approved"] == "first self"
    assert state["summaries_revision"] == 2
    assert soul_summaries.list_for_review(con)[0]["pending"] is False


def test_pending_endpoint_omits_approved_soul_summary(monkeypatch, tmp_path) -> None:
    path = tmp_path / "soul.db"
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    soul_state.ensure_schema(con)
    con.execute(
        "UPDATE soul_state SET narrative_self = 'approved self', "
        "narrative_self_approved = 'approved self' WHERE id = 1"
    )
    con.commit()
    con.close()

    class Service:
        @staticmethod
        def graph_list_pending(*, where):
            assert where == {"user_id": "u", "soul_id": "s"}
            return {"items": [], "categories": []}

    monkeypatch.setattr(main, "_get_service_from_payload", lambda _payload: Service())
    monkeypatch.setattr(main, "_sqlite_current_path", lambda _uid, _sid: path)

    pending = asyncio.run(_route_endpoint("memory_graph_pending")(user_id="u", soul_id="s"))

    assert pending["soul_summaries"] == []


def test_soul_summary_route_rejects_stale_snapshot(monkeypatch, tmp_path) -> None:
    path = tmp_path / "soul.db"
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    soul_state.ensure_schema(con)
    con.execute(
        "UPDATE soul_state SET narrative_self = 'current', narrative_self_approved = 'approved', summaries_revision = 4"
    )
    con.commit()
    con.close()
    monkeypatch.setattr(main, "_sqlite_current_path", lambda _uid, _sid: path)
    monkeypatch.setattr(soul_summaries, "append_summary_journal", lambda **_kwargs: None)

    with pytest.raises(main.HTTPException) as exc_info:
        asyncio.run(
            _route_endpoint("soul_summary_update")(
                kind="narrative_self",
                user_id="u",
                soul_id="s",
                payload={
                    "summary": "edited",
                    "displayed_summary": "current",
                    "summaries_revision": 3,
                },
            )
        )
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "summary_snapshot_stale"


@pytest.mark.parametrize("journal_failure", [False, True])
def test_atomic_narrative_correction_updates_state_without_evolution(monkeypatch, tmp_path, journal_failure, caplog) -> None:
    path = tmp_path / "soul.db"
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    soul_state.ensure_schema(con)
    con.execute(
        "CREATE TABLE narrative_history (id TEXT, narrative_self TEXT, related_memory_ids TEXT, created_at TEXT)"
    )
    con.execute(
        "UPDATE soul_state SET narrative_self = 'current', narrative_self_approved = 'approved', "
        "summaries_revision = 4, updated_at = 'old'"
    )
    con.commit()
    con.close()
    journal_entries = []
    monkeypatch.setattr(main, "_sqlite_current_path", lambda _uid, _sid: path)

    def append(**entry):
        check = sqlite3.connect(path)
        try:
            assert check.execute("SELECT narrative_self, summaries_revision FROM soul_state").fetchone() == ("edited", 5)
        finally:
            check.close()
        journal_entries.append(entry)
        if journal_failure:
            raise OSError("fictional journal failure")

    monkeypatch.setattr(soul_summaries, "append_summary_journal", append)
    monkeypatch.setattr(
        main,
        "_get_service_from_payload",
        lambda _payload: (_ for _ in ()).throw(AssertionError("corrections do not resolve MemoryService")),
    )
    monkeypatch.setattr(
        main,
        "snapshot_previous_narrative_self",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("corrections do not snapshot old self")),
    )

    out = asyncio.run(
        _route_endpoint("soul_summary_update")(
            kind="narrative_self",
            user_id="u",
            soul_id="s",
            payload={
                "summary": "edited",
                "displayed_summary": "current",
                "summaries_revision": 4,
            },
        )
    )
    assert out["summary"] == "edited"
    assert out["approved_summary"] == "edited"
    assert out["previous_summary"] == "current"
    assert out["summaries_revision"] == 5
    assert len(journal_entries) == 1
    assert journal_entries[0]["summary_before"] == "current"
    if journal_failure:
        assert "Failed to journal committed soul summary" in caplog.text

    check = sqlite3.connect(path)
    check.row_factory = sqlite3.Row
    assert check.execute("SELECT COUNT(*) FROM narrative_history").fetchone()[0] == 0
    assert soul_state.read(check)["updated_at"] != "old"
    check.close()


@pytest.mark.parametrize("case", ["accepted", "concurrent_chat", "concurrent_embedding", "embedding_failure", "companion_write_failure", "declined", "unchanged"])
def test_narrative_suggestion_preserves_concurrent_card(monkeypatch, tmp_path, request, case):
    from memu.app import category_summary_journal
    from app.services import crud_endpoints, narrative_self
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy import event

    path = tmp_path / "soul.db"
    con = sqlite3.connect(path)
    soul_state.ensure_schema(con)
    con.execute("UPDATE soul_state SET narrative_self = 'Original card  ', summaries_revision = 0")
    con.commit()
    con.close()
    monkeypatch.setattr(category_summary_journal, "JOURNAL_DIR", tmp_path / "journals")
    service_payloads = []
    prompts = []
    svc = main.MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": f"sqlite:///{path}"}},
        user_config={"model": main.STUserModel, "user_name": "FictionalHuman"},
    )
    request.addfinalizer(svc.database.close)
    returned_row_factories = []
    event.listen(svc.database._sessions.engine, "checkin",
        lambda connection, _record: returned_row_factories.append(connection.row_factory))
    if case == "companion_write_failure":
        with sqlite3.connect(path) as writer:
            writer.execute("CREATE TRIGGER reject_companion BEFORE INSERT ON memory_items WHEN NEW.memory_type = 'reflection' BEGIN SELECT RAISE(ABORT, 'fictional companion failure'); END")

    async def chat(prompt, **kwargs):
        prompts.append((prompt, kwargs["system_prompt"]))
        if case == "concurrent_chat":
            with sqlite3.connect(path) as writer:
                writer.execute("UPDATE soul_state SET narrative_self = 'Newer card'")
        narrative = "" if case == "declined" else "Original card" if case == "unchanged" else "Suggested card"
        return json.dumps({"narrative_self": narrative, "companion_memory": "I considered the suggestion."})

    async def embed(texts, **_kwargs):
        if "I considered the suggestion." in texts:
            if case == "embedding_failure":
                raise RuntimeError("fictional embedding failure")
            if case == "concurrent_embedding":
                with sqlite3.connect(path) as writer:
                    writer.execute("UPDATE soul_state SET narrative_self = 'Newer card'")
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(svc, "chat", chat)
    monkeypatch.setattr(svc, "embed", embed)
    monkeypatch.setattr(svc, "build_dossier_index", lambda _scope: "")

    runtime_payload = {"user_id": "TestOwner", "suggestion": "A fictional suggestion", "claude_code": False,
                       "llm_profiles": {"default": {"chat_model": "fictional-captured-model"}}}
    def service_from_payload(payload):
        service_payloads.append(payload)
        return svc
    suggestion = crud_endpoints.narrative_suggestion_endpoint(
        soul_id="TestSoul", payload=runtime_payload,
        sqlite_current_path=lambda _u, _s: path, sqlite_connect=main._sqlite_connect,
        sqlite_ensure_conversation_state_schema=main._sqlite_ensure_conversation_state_schema,
        sqlite_ensure_nonempty=main._sqlite_ensure_nonempty,
        get_service_from_payload=service_from_payload,
        build_retrieve_identity_context=lambda _name: "",
        snapshot_previous_narrative_self=narrative_self.snapshot_previous_narrative_self,
    )
    concurrent_edit = case in ("concurrent_chat", "concurrent_embedding")
    if concurrent_edit:
        with pytest.raises(main.HTTPException) as stale:
            asyncio.run(suggestion)
        assert stale.value.status_code == 409
        assert "changed" in stale.value.detail
    elif case == "embedding_failure":
        with pytest.raises(RuntimeError, match="fictional embedding failure"):
            asyncio.run(suggestion)
    elif case == "companion_write_failure":
        with pytest.raises(IntegrityError, match="fictional companion failure"):
            asyncio.run(suggestion)
    else:
        assert asyncio.run(suggestion) == {"narrative_self": "Suggested card" if case == "accepted" else "Original card"}
    assert service_payloads == [{**runtime_payload, "user": {"user_id": "TestOwner", "soul_id": "TestSoul"}}]
    assert "FictionalHuman's suggestion:\nA fictional suggestion" in prompts[0][0]
    assert "received a suggestion from FictionalHuman" in prompts[0][1]
    assert "It's up to me to decide if I want that." in prompts[0][1]
    assert "TestOwner" not in prompts[0][0] + prompts[0][1]
    assert len(prompts) == 1
    accepted = case == "accepted"
    companion_saved = case in ("accepted", "declined", "unchanged")
    with sqlite3.connect(path) as check:
        assert check.execute("SELECT narrative_self, summaries_revision FROM soul_state").fetchone() == (
            ("Newer card", 0) if concurrent_edit else ("Suggested card", 1) if accepted else ("Original card  ", 0)
        )
        assert check.execute("SELECT COUNT(*) FROM narrative_history").fetchone()[0] == int(accepted)
        assert check.execute("SELECT COUNT(*) FROM memory_items WHERE memory_type = 'narrative_self'").fetchone()[0] == int(accepted)
        assert check.execute("SELECT COUNT(*) FROM memory_items WHERE memory_type = 'reflection'").fetchone()[0] == int(companion_saved)
        assert check.execute("SELECT COUNT(*) FROM triples").fetchone()[0] == 0
    assert (tmp_path / "journals").exists() == accepted
    assert all(factory is None for factory in returned_row_factories)


@pytest.mark.parametrize("surface", ["atomic", "suggestion"])
def test_failed_narrative_save_does_not_journal(monkeypatch, tmp_path, request, surface) -> None:
    from memu.app import category_summary_journal

    path = tmp_path / "soul.db"
    con = sqlite3.connect(path)
    soul_state.ensure_schema(con)
    con.execute("UPDATE soul_state SET narrative_self = 'Original card', summaries_revision = 0")
    con.execute("CREATE TRIGGER reject_card BEFORE UPDATE OF narrative_self ON soul_state BEGIN SELECT RAISE(ABORT, 'fictional write failure'); END")
    con.commit()
    con.close()
    journal_dir = tmp_path / "journals"
    monkeypatch.setattr(category_summary_journal, "JOURNAL_DIR", journal_dir)
    monkeypatch.setattr(main, "_sqlite_current_path", lambda _uid, _sid: path)

    if surface == "atomic":
        request = _route_endpoint("soul_summary_update")(
            "narrative_self", "TestOwner", "TestSoul",
            {"summary": "Rejected card", "displayed_summary": "Original card", "summaries_revision": 0},
        )
    else:
        from app.services import crud_endpoints

        svc = main.MemoryService(
            database_config={"metadata_store": {"provider": "sqlite", "dsn": f"sqlite:///{path}"}},
            user_config={"model": main.STUserModel, "user_name": "FictionalHuman"},
        )
        request.addfinalizer(svc.database.close)

        async def chat(*_args, **_kwargs):
            return json.dumps({"narrative_self": "Rejected card", "companion_memory": ""})

        async def embed(texts, **_kwargs):
            return [[1.0, 0.0] for _ in texts]

        monkeypatch.setattr(svc, "chat", chat)
        monkeypatch.setattr(svc, "embed", embed)
        monkeypatch.setattr(svc, "build_dossier_index", lambda _scope: "")

        request = crud_endpoints.narrative_suggestion_endpoint(
            soul_id="TestSoul", payload={"user_id": "TestOwner", "suggestion": "A fictional suggestion"},
            sqlite_current_path=lambda _u, _s: path, sqlite_connect=main._sqlite_connect,
            sqlite_ensure_conversation_state_schema=main._sqlite_ensure_conversation_state_schema,
            sqlite_ensure_nonempty=main._sqlite_ensure_nonempty,
            get_service_from_payload=lambda _payload: svc,
            build_retrieve_identity_context=lambda _name: "",
            snapshot_previous_narrative_self=lambda *_args, **_kwargs: None,
        )
    with pytest.raises(sqlite3.IntegrityError, match="fictional write failure"):
        asyncio.run(request)
    assert not journal_dir.exists()
    con = sqlite3.connect(path)
    try:
        assert con.execute("SELECT narrative_self, summaries_revision FROM soul_state").fetchone() == ("Original card", 0)
    finally:
        con.close()
