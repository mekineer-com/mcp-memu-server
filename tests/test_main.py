"""Basic tests for the application."""

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import main


def test_placeholder():
    assert True


def test_retrieve_method_from_cfg_forces_public_rag():
    assert main._retrieve_method_from_cfg({"retrieve": {"method": "rag"}}) == "rag"
    assert main._retrieve_method_from_cfg({"retrieve": {"method": "llm"}}) == "rag"
    assert main._retrieve_method_from_cfg({"retrieve": {"method": "unknown"}}) == "rag"


def test_retrieve_apimw_enabled_from_cfg_defaults_and_override():
    assert main._retrieve_apimw_enabled_from_cfg(None) is True
    assert main._retrieve_apimw_enabled_from_cfg({"retrieve": {}}) is True
    assert main._retrieve_apimw_enabled_from_cfg({"retrieve": {"apimw_enabled": True}}) is True
    assert main._retrieve_apimw_enabled_from_cfg({"retrieve": {"apimw_enabled": False}}) is False


def test_imports():
    assert hasattr(main, "app")


def test_merge_memorize_segment_results_flattens_top_level_lists():
    out = main._merge_memorize_segment_results(
        [
            {
                "resource": {"id": "r1", "url": "file://a"},
                "items": [{"id": "m1", "summary": "one"}],
                "categories": [{"id": "c1", "name": "Profiles"}],
                "relations": [{"item_id": "m1", "category_id": "c1"}],
                "skipped_reasons": ["skip-a"],
            },
            {
                "resource": {"id": "r2", "url": "file://b"},
                "items": [{"id": "m2", "summary": "two"}, {"id": "m1", "summary": "one"}],
                "categories": [{"id": "c1", "name": "Profiles"}, {"id": "c2", "name": "Goals"}],
                "relations": [{"item_id": "m2", "category_id": "c2"}],
                "skipped_reasons": ["skip-b", "skip-a"],
            },
        ],
        ["m2", "m1", "m2"],
    )

    assert out["segment_count"] == 2
    assert [item["id"] for item in out["items"]] == ["m1", "m2"]
    assert [cat["id"] for cat in out["categories"]] == ["c1", "c2"]
    assert out["pending_episode_ids"] == ["m2", "m1"]
    assert out["skipped_reasons"] == ["skip-a", "skip-b"]
    assert "results" in out
    assert [res["id"] for res in out["resources"]] == ["r1", "r2"]


def test_estimate_unmemorized_tokens_respects_digest_cursor():
    messages = [
        {"content": "one two three"},
        {"content": "four five six"},
        {"content": "seven eight nine"},
    ]
    assert main._estimate_unmemorized_tokens(messages, -1) == main._estimate_tokens(messages)
    assert main._estimate_unmemorized_tokens(messages, 1) == main._estimate_tokens(messages[2:])
    assert main._estimate_unmemorized_tokens(messages, 99) == 0


def test_split_indices_by_sleep_keeps_all_qualifying_boundaries():
    def _ts(y: int, m: int, d: int, hh: int, mm: int = 0) -> int:
        return int(datetime(y, m, d, hh, mm, tzinfo=UTC).timestamp() * 1000)

    messages = [
        {"ts_ms": _ts(2026, 1, 1, 21, 0)},
        {"ts_ms": _ts(2026, 1, 2, 9, 0)},
        {"ts_ms": _ts(2026, 1, 2, 21, 0)},
        {"ts_ms": _ts(2026, 1, 3, 9, 0)},
        {"ts_ms": _ts(2026, 1, 3, 21, 0)},
        {"ts_ms": _ts(2026, 1, 4, 9, 0)},
    ]
    splits, stats = main._split_indices_by_sleep(
        messages,
        UTC,
        True,
        3 * 60 * 60,
    )

    assert splits == [1, 3, 5]
    assert stats["nights_qual"] == 3


def test_turn_launch_apimw_uses_periodic_cadence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "_apimw_cadence_from_cfg", lambda *_a, **_k: 3)
    monkeypatch.setattr(main, "_mark_apimw_inflight", lambda *_a, **_k: False)

    history_three = [
        {"role": "assistant", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "assistant", "content": "c"},
    ]
    status_three = main._turn_launch_apimw(
        "cid",
        "u1",
        "Echo",
        {},
        history_three,
    )
    assert status_three == "skipped_inflight"

    history_four = history_three + [{"role": "assistant", "content": "d"}]
    status_four = main._turn_launch_apimw(
        "cid",
        "u1",
        "Echo",
        {},
        history_four,
    )
    assert status_four == "skipped_cadence"



def test_normalize_conversation_uses_created_at_when_timestamp_missing():
    conv = [{"role": "user", "content": "hello", "created_at": "2026-04-16T12:00:00Z"}]
    out = main._normalize_conversation(conv)

    assert isinstance(out, list) and out
    assert out[0]["ts_ms"] == int(datetime(2026, 4, 16, 12, 0, tzinfo=UTC).timestamp() * 1000)


def test_normalize_conversation_preserves_cross_memorize_metadata():
    conv = [
        {
            "role": "user",
            "name": "Marcos",
            "content": "hello",
            "source_label": "whatsapp:group",
            "source_conversation_id": "whatsapp:group:123@g.us",
            "source_conversation_index": 42,
            "received_at": "2026-05-15T12:00:00+00:00",
            "memorize_chat": False,
        }
    ]
    out = main._normalize_conversation(conv)
    assert isinstance(out, list) and out
    assert out[0]["source_label"] == "whatsapp:group"
    assert out[0]["source_conversation_id"] == "whatsapp:group:123@g.us"
    assert out[0]["source_conversation_index"] == 42
    assert out[0]["received_at"] == "2026-05-15T12:00:00+00:00"
    assert out[0]["memorize_chat"] is False


def test_parse_as_of_datetime_accepts_iso_date_and_datetime():
    date_only = main._parse_as_of_datetime("2026-04-18")
    assert date_only is not None
    assert date_only.isoformat().startswith("2026-04-18T00:00:00")

    zulu = main._parse_as_of_datetime("2026-04-18T10:15:00Z")
    assert zulu is not None
    assert zulu.tzinfo is not None
    assert zulu.isoformat().startswith("2026-04-18T10:15:00")


def test_parse_as_of_datetime_rejects_invalid():
    with pytest.raises(main.HTTPException):
        main._parse_as_of_datetime("not-a-date")


def test_build_retrieve_identity_context_uses_shared_time_anchor(monkeypatch: pytest.MonkeyPatch):
    sentinel_anchor = "__ANCHOR_FROM_SHARED_FORMATTER__"
    monkeypatch.setattr(main, "_format_time_anchor", lambda *_a, **_k: sentinel_anchor)
    out = main._build_retrieve_identity_context("Echo")
    assert out.startswith(f"Today is {sentinel_anchor}.")


def test_apply_turn_history_window_keeps_full_unmemorized_tail():
    history_full = [{"message_id": f"m{i}", "role": "user", "content": f"msg {i}"} for i in range(1, 13)]
    history_tail = list(history_full)

    out = main._apply_turn_history_window(
        conversation_id="cid",
        history_tail=history_tail,
        history_full=history_full,
        db_path=None,
    )

    assert len(out) == 12
    assert [item["message_id"] for item in out] == [f"m{i}" for i in range(1, 13)]


def test_apply_turn_history_window_backfills_from_payload_when_tail_short():
    history_full = [{"message_id": f"m{i}", "role": "user", "content": f"msg {i}"} for i in range(1, 11)]
    history_tail = history_full[-2:]

    out = main._apply_turn_history_window(
        conversation_id="cid",
        history_tail=history_tail,
        history_full=history_full,
        db_path=None,
    )

    assert len(out) == 8
    assert [item["message_id"] for item in out] == [f"m{i}" for i in range(3, 11)]


def test_apply_turn_history_window_backfills_from_db_when_available(tmp_path: Path):
    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    try:
        con.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                speaker TEXT,
                content TEXT NOT NULL,
                source_label TEXT,
                received_at TEXT
            )
            """
        )
        for i in range(1, 11):
            con.execute(
                "INSERT INTO messages (conversation_id, role, speaker, content, source_label, received_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("cid", "user", "Marcos", f"db {i}", "sillytavern", f"2026-05-08T00:00:{i:02d}+00:00"),
            )
        con.commit()
    finally:
        con.close()

    history_full = [{"message_id": "m1", "role": "user", "content": "payload only"}]
    history_tail = history_full[-1:]

    out = main._apply_turn_history_window(
        conversation_id="cid",
        history_tail=history_tail,
        history_full=history_full,
        db_path=db_path,
    )

    assert len(out) == 8
    assert [item["content"] for item in out] == [f"db {i}" for i in range(3, 11)]


def test_apply_turn_history_window_backfills_from_whatsapp_alias_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    hermes_home = tmp_path / ".hermes"
    session_dir = hermes_home / "whatsapp" / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "creds.json").write_text(
        '{"me":{"id":"15133278228:13@s.whatsapp.net","lid":"114628432556258:13@lid"}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    try:
        con.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                speaker TEXT,
                content TEXT NOT NULL,
                source_label TEXT,
                received_at TEXT
            )
            """
        )
        con.execute(
            "INSERT INTO messages (conversation_id, role, speaker, content, source_label, received_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("whatsapp:dm:114628432556258", "user", "Marcos", "lid-older", "whatsapp:dm", "2026-05-08T00:00:01+00:00"),
        )
        con.execute(
            "INSERT INTO messages (conversation_id, role, speaker, content, source_label, received_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("whatsapp:dm:15133278228", "assistant", "Echo", "phone-newer", "whatsapp:dm", "2026-05-08T00:00:02+00:00"),
        )
        con.commit()
    finally:
        con.close()

    history_full = [{"message_id": "m1", "role": "user", "content": "payload only"}]
    history_tail = history_full[-1:]

    out = main._apply_turn_history_window(
        conversation_id="whatsapp:dm:15133278228",
        history_tail=history_tail,
        history_full=history_full,
        db_path=db_path,
    )

    assert [item["content"] for item in out] == ["lid-older", "phone-newer"]


def test_build_retrieve_soul_context_queries_uses_last_8_messages_for_rewrite() -> None:
    history = [
        {"message_id": f"m{i}", "role": "user", "content": f"msg {i}"}
        for i in range(1, 16)
    ]
    queries = main._build_retrieve_soul_context_queries(
        soul_id="Echo",
        message="current",
        history=history,
        state_row={"memory_cache": [], "intentions_active": {"items": []}},
    )
    history_rows = [q for q in queries if isinstance(q, dict) and q.get("role") == "history"]
    assert len(history_rows) == 1
    text = str((history_rows[0].get("content") or {}).get("text") or "")
    assert "[user] msg 8" in text
    assert "[user] msg 15" in text
    assert "[user] msg 7" not in text


def test_build_retrieve_soul_context_queries_uses_last_12_messages_for_apimw_rewrite() -> None:
    history = [
        {"message_id": f"m{i}", "role": "user", "content": f"msg {i}"}
        for i in range(1, 16)
    ]
    queries = main._build_retrieve_soul_context_queries(
        soul_id="Echo",
        message="current",
        history=history,
        state_row={"memory_cache": [], "intentions_active": {"items": []}},
        identity_mode="apimw",
    )
    history_rows = [q for q in queries if isinstance(q, dict) and q.get("role") == "history"]
    assert len(history_rows) == 1
    text = str((history_rows[0].get("content") or {}).get("text") or "")
    assert "[user] msg 4" in text
    assert "[user] msg 15" in text
    assert "[user] msg 3" not in text


@pytest.mark.asyncio
async def test_run_memorize_episodes_clears_progress_on_exception(tmp_path):
    episodes_dir = tmp_path / "episodes"
    episodes_dir.mkdir()

    class _FailingService:
        async def split_segment_into_episodes(self, **_kwargs):
            return [{"text": "episode text", "caption": "episode"}]

        async def memorize_episodes_batch(self, **_kwargs):
            raise RuntimeError("boom")

    user_id = "u"
    soul_id = "s"
    key = main._memorize_lock_key(user_id, soul_id)
    main._MEMORIZE_PROGRESS.pop(key, None)
    main._MEMORIZE_CANCEL.discard(key)

    with pytest.raises(RuntimeError):
        await main._run_memorize_episodes(
            memorize_segments=[("/tmp/day.json", [{"role": "user", "content": "x"}], 0)],
            svc=_FailingService(),
            scope={"user_id": user_id, "soul_id": soul_id},
            conversation_id=None,
            soul_id=soul_id,
            uid=user_id,
            processed_cursor=-1,
            safe={},
            resource_url="/tmp/day.json",
            chat_file=None,
            resource_url_in=None,
            chat_key=None,
            chat_key_source=None,
            tz_name=None,
            prev_len=0,
            merged_len=1,
            force=True,
            sleep_stats=None,
            episodes_dir=episodes_dir,
        )

    assert key not in main._MEMORIZE_PROGRESS
    assert key not in main._MEMORIZE_CANCEL


def test_timeline_endpoint_returns_entity_edges(monkeypatch: pytest.MonkeyPatch):
    class _EntityRepo:
        def list_all(self, where=None):
            return [SimpleNamespace(id="ent_1", name="Marcos", entity_type="person", normalized="marcos")]

    class _TripleRepo:
        def __init__(self) -> None:
            self.captured_as_of = None

        def get_edges_from(self, *_args, **kwargs):
            self.captured_as_of = kwargs.get("as_of")
            return [
                SimpleNamespace(
                    id="edge_1",
                    subject_id="ent_1",
                    subject_kind="entity",
                    predicate="parallels",
                    object_id="mem_1",
                    object_kind="memory",
                    valid_from=datetime(2026, 4, 18, 8, 0, tzinfo=UTC),
                    valid_to=None,
                    confidence=0.9,
                    source_memory_id="mem_1",
                )
            ]

        def get_edges_to(self, *_args, **kwargs):
            self.captured_as_of = kwargs.get("as_of")
            return []

    class _MemoryRepo:
        def get_item(self, item_id: str):
            if item_id != "mem_1":
                return None
            return SimpleNamespace(
                id="mem_1",
                memory_type="knowledge",
                summary="Marcos values continuity.",
                happened_at=datetime(2026, 4, 18, 7, 0, tzinfo=UTC),
            )

    triple_repo = _TripleRepo()
    fake_db = SimpleNamespace(entity_repo=_EntityRepo(), triple_repo=triple_repo, memory_item_repo=_MemoryRepo())
    fake_svc = SimpleNamespace(database=fake_db)
    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_k: fake_svc)

    out = asyncio.run(
        main.timeline(
            entity="Marcos",
            user_id="u",
            soul_id="s",
            as_of="2026-04-18T12:00:00Z",
        )
    )

    assert out["ok"] is True
    assert out["entity"]["id"] == "ent_1"
    assert out["count"] == 1
    assert out["timeline"][0]["predicate"] == "parallels"
    assert out["timeline"][0]["memory"]["id"] == "mem_1"
    assert triple_repo.captured_as_of is not None


def test_validate_relationship_speaker_id_rejects_reserved_prefixes():
    assert main._validate_relationship_speaker_id("entity:brother") == "brother"
    with pytest.raises(main.HTTPException):
        main._validate_relationship_speaker_id("user:marcos")
    with pytest.raises(main.HTTPException):
        main._validate_relationship_speaker_id("entity:brother!")


def test_relationship_item_from_values_filters_non_declared_or_inactive():
    item = main._relationship_item_from_values(
        normalized="brother",
        name="Brother",
        entity_type="person",
        properties={"origin": "user_declared", "relationship": "sibling", "active": True},
    )
    assert item is not None
    assert item["speaker_id"] == "entity:brother"
    assert item["relationship"] == "sibling"

    assert main._relationship_item_from_values(
        normalized="brother",
        name="Brother",
        entity_type="person",
        properties={"origin": "extracted", "relationship": "sibling", "active": True},
    ) is None
    assert main._relationship_item_from_values(
        normalized="brother",
        name="Brother",
        entity_type="person",
        properties={"origin": "user_declared", "active": False},
    ) is None


def test_assert_user_declared_relationship_is_strict():
    main._assert_user_declared_relationship({"origin": "user_declared"})
    with pytest.raises(main.HTTPException):
        main._assert_user_declared_relationship({"origin": ""})
    with pytest.raises(main.HTTPException):
        main._assert_user_declared_relationship({"origin": "extracted"})


@pytest.mark.asyncio
async def test_apimw_persist_remaps_numbered_prior_context_ids(monkeypatch: pytest.MonkeyPatch):
    captured_updates: dict[str, object] = {}

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: ({"prior_context": "", "memory_cache": [], "intentions_active": []}, None, None),
    )

    def _fake_write_conversation_state(conversation_id: str, soul_id: str, user_id: str, updates: dict[str, object]):
        captured_updates.update(updates)
        return {"conversation_id": conversation_id, **updates}, Path("/tmp/fake.json")

    monkeypatch.setattr(main, "_write_conversation_state", _fake_write_conversation_state)

    await main._apimw_persist(
        svc=SimpleNamespace(),
        result_json={"prior_context": ["1", "mem_raw", "2", "1"]},
        items_by_id={
            "mem_one": {"id": "mem_one", "memory_type": "profile", "summary": "Marcos likes continuity."},
            "mem_two": {"id": "mem_two", "memory_type": "behavior", "summary": "I paused before replying."},
            "mem_raw": {"id": "mem_raw", "memory_type": "knowledge", "summary": "Raw IDs can still appear."},
        },
        id_map={"1": "mem_one", "2": "mem_two"},
        combined_items=[],
        scope={"user_id": "u", "soul_id": "s"},
        conversation_id="c",
        user_id="u",
        soul_id="s",
    )

    assert captured_updates["append_prior_context_ids_since_consolidation"] == ["mem_one", "mem_raw", "mem_two"]


@pytest.mark.asyncio
async def test_conversation_retrieve_injects_cross_context_even_with_prebuilt_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor) VALUES (?, ?)",
            ("cid-current", 0),
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor) VALUES (?, ?)",
            ("whatsapp:dm:other", 0),
        )
        con.execute(
            "INSERT INTO messages (conversation_id, role, speaker, content, source_label, received_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("whatsapp:dm:other", "user", "Marcos", "wa-1", "whatsapp:dm", "2026-05-08T11:00:00+00:00"),
        )
        con.execute(
            "INSERT INTO messages (conversation_id, role, speaker, content, source_label, received_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("whatsapp:dm:other", "assistant", "Echo", "wa-2", "whatsapp:dm", "2026-05-08T11:00:01+00:00"),
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: ({"prior_context": "", "memory_cache": [], "intentions_active": {"items": []}}, None, db_path),
    )

    captured: dict[str, object] = {}

    async def _fake_run_retrieve(safe: dict[str, object], *, conversation_id: str | None = None) -> dict[str, object]:
        captured["safe"] = safe
        return {"ok": True, "should_respond": True, "result": {}, "conversation_id": conversation_id}

    monkeypatch.setattr(main, "_run_retrieve", _fake_run_retrieve)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Echo"},
        "message": "hello from st",
        "query": "hello from st",
        "history": [{"role": "user", "content": "hello from st"}],
        "queries": [{"role": "message", "content": {"text": "hello from st"}}],
    }

    out = await main.conversation_retrieve("cid-current", payload)

    assert out["ok"] is True
    safe = captured["safe"]
    assert isinstance(safe, dict)
    cross_text = str(safe.get("_cross_conversation_history") or "")
    assert "wa-2" in cross_text
    queries = safe.get("queries")
    assert isinstance(queries, list)
    cross_roles = [
        str(q.get("role") or "").strip()
        for q in queries
        if isinstance(q, dict)
    ]
    assert cross_roles.count("cross_conversation") == 1


@pytest.mark.asyncio
async def test_conversation_retrieve_does_not_persist_current_user_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User message persistence is deferred to conversation_turn (paired with
    the assistant response) so that aborted/inspected turns leave no orphan
    rows in the messages table.
    """
    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor) VALUES (?, ?)",
            ("cid-current", 0),
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: ({"prior_context": "", "memory_cache": [], "intentions_active": {"items": []}}, None, db_path),
    )

    async def _fake_run_retrieve(safe: dict[str, object], *, conversation_id: str | None = None) -> dict[str, object]:
        return {"ok": True, "should_respond": True, "result": {}, "conversation_id": conversation_id}

    monkeypatch.setattr(main, "_run_retrieve", _fake_run_retrieve)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Echo"},
        "message": "current message",
        "query": "current message",
        "user_name": "Alice",
        "history": [{"role": "user", "content": "prior message"}],
        "queries": [{"role": "message", "content": {"text": "current message"}}],
    }

    out = await main.conversation_retrieve("cid-current", payload)
    assert out["ok"] is True

    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT role, speaker, content FROM messages WHERE conversation_id = ?",
            ("cid-current",),
        ).fetchall()
    finally:
        con.close()

    # No rows should be persisted by retrieve alone: that is the orphan-prevention contract.
    assert rows == []


@pytest.mark.asyncio
async def test_conversation_retrieve_does_not_duplicate_preexisting_cross_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor) VALUES (?, ?)",
            ("cid-current", 0),
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor) VALUES (?, ?)",
            ("whatsapp:dm:other", 0),
        )
        con.execute(
            "INSERT INTO messages (conversation_id, role, speaker, content, source_label, received_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("whatsapp:dm:other", "user", "Marcos", "wa-1", "whatsapp:dm", "2026-05-08T11:00:00+00:00"),
        )
        con.execute(
            "INSERT INTO messages (conversation_id, role, speaker, content, source_label, received_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("whatsapp:dm:other", "assistant", "Echo", "wa-2", "whatsapp:dm", "2026-05-08T11:00:01+00:00"),
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: ({"prior_context": "", "memory_cache": [], "intentions_active": {"items": []}}, None, db_path),
    )

    captured: dict[str, object] = {}

    async def _fake_run_retrieve(safe: dict[str, object], *, conversation_id: str | None = None) -> dict[str, object]:
        captured["safe"] = safe
        return {"ok": True, "should_respond": True, "result": {}, "conversation_id": conversation_id}

    monkeypatch.setattr(main, "_run_retrieve", _fake_run_retrieve)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Echo"},
        "message": "hello from st",
        "query": "hello from st",
        "history": [{"role": "user", "content": "hello from st"}],
        "queries": [
            {"role": "cross_conversation", "content": {"text": "existing"}},
            {"role": "message", "content": {"text": "hello from st"}},
        ],
    }

    out = await main.conversation_retrieve("cid-current", payload)

    assert out["ok"] is True
    safe = captured["safe"]
    assert isinstance(safe, dict)
    queries = safe.get("queries")
    assert isinstance(queries, list)
    cross_roles = [
        str(q.get("role") or "").strip()
        for q in queries
        if isinstance(q, dict)
    ]
    assert cross_roles.count("cross_conversation") == 1


@pytest.mark.asyncio
async def test_conversation_turn_persists_assistant_message_for_cross_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.commit()
    finally:
        con.close()

    class _FakeSvc:
        async def chat(self, *_args, **_kwargs) -> str:
            return '{"cache":null,"annulments":[],"inner_thought":"ok","response":"assistant says hi"}'

    async def _fake_persist_annulment_memories(**_kwargs):
        return []

    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_k: _FakeSvc())
    monkeypatch.setattr(main, "_load_soul_gen_config", lambda *_a, **_k: {})
    monkeypatch.setattr(
        main,
        "_turn_state_read",
        lambda *_a, **_k: (
            {"digest_cursor": 0},
            None,
            db_path,
            [],
            {"items": []},
            0,
            None,
        ),
    )
    monkeypatch.setattr(main, "_turn_state_write", lambda *_a, **_k: ({"digest_cursor": 0}, db_path))
    monkeypatch.setattr(main, "_persist_annulment_memories", _fake_persist_annulment_memories)
    monkeypatch.setattr(main, "_record_call", lambda *_a, **_k: None)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Echo", "conversation_id": "cid-turn"},
        "message": "hello",
        "user_name": "Alice",
        "history": [{"role": "user", "content": "hello"}],
        "run_apimw": False,
        "apply_turn_maintenance": False,
        "prompt_override_payload": {
            "user_prompt": "prompt",
            "system_prompt": "system",
            "memory_cache": [],
            "intentions_active": {"items": []},
            "retrieve_rag": {"items": [], "categories": [], "resources": []},
        },
    }

    out = await main.conversation_turn("cid-turn", payload)

    assert out["ok"] is True
    assert out["response"] == "assistant says hi"

    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT role, speaker, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            ("cid-turn",),
        ).fetchall()
    finally:
        con.close()

    # The current user message and the assistant response are persisted as a
    # pair so that an aborted turn (no response) leaves no orphan user row.
    assert len(rows) == 2
    assert str(rows[0]["role"]) == "user"
    assert str(rows[0]["speaker"]) == "Alice"
    assert str(rows[0]["content"]) == "hello"
    assert str(rows[1]["role"]) == "assistant"
    assert str(rows[1]["speaker"]) == "Echo"
    assert str(rows[1]["content"]) == "assistant says hi"


@pytest.mark.asyncio
async def test_conversation_turn_drops_response_when_peer_mismatches_chat_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the soul declares response_target=respond with a peer that does not
    match the originating chat's chat_name, the response is dropped (empty
    response_text) and nothing is persisted. The contract validation lives in
    conversation_turn so an unintended cross-chat reply never reaches a
    downstream transport.
    """
    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.commit()
    finally:
        con.close()

    class _FakeSvc:
        async def chat(self, *_args, **_kwargs) -> str:
            # Soul thinks she is responding to Alice, but the originating
            # chat is with Bob — the validator must drop the reply.
            return (
                '{"cache":null,"annulments":[],"inner_thought":"answering Alice",'
                '"response_target":"respond","response_peer":"Alice",'
                '"response":"hi Alice"}'
            )

    async def _fake_persist_annulment_memories(**_kwargs):
        return []

    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_k: _FakeSvc())
    monkeypatch.setattr(main, "_load_soul_gen_config", lambda *_a, **_k: {})
    monkeypatch.setattr(
        main,
        "_turn_state_read",
        lambda *_a, **_k: (
            {"digest_cursor": 0},
            None,
            db_path,
            [],
            {"items": []},
            0,
            None,
        ),
    )
    monkeypatch.setattr(main, "_turn_state_write", lambda *_a, **_k: ({"digest_cursor": 0}, db_path))
    monkeypatch.setattr(main, "_persist_annulment_memories", _fake_persist_annulment_memories)
    monkeypatch.setattr(main, "_record_call", lambda *_a, **_k: None)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Echo", "conversation_id": "cid-turn"},
        "message": "hey",
        "user_name": "Bob",
        "chat_name": "Bob",
        "chat_type": "dm",
        "history": [],
        "run_apimw": False,
        "apply_turn_maintenance": False,
        "prompt_override_payload": {
            "user_prompt": "prompt",
            "system_prompt": "system",
            "memory_cache": [],
            "intentions_active": {"items": []},
            "retrieve_rag": {"items": [], "categories": [], "resources": []},
        },
    }

    out = await main.conversation_turn("cid-turn", payload)

    assert out["ok"] is True
    assert out["response"] == ""
    assert out["response_target"] == "respond"
    assert out["response_peer"] == "Alice"

    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT role FROM messages WHERE conversation_id = ?",
            ("cid-turn",),
        ).fetchall()
    finally:
        con.close()
    # Mismatch means no response text, which means no user+assistant pair.
    assert rows == []
