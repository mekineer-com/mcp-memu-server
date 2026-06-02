"""Basic tests for the application."""

import asyncio
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app import main
from app.services import conversation_sources, crud_endpoints


def test_placeholder():
    assert True


def test_merge_llm_profiles_rejects_null_fields_from_client():
    defaults = {
        "default": {
            "provider": "openai",
            "api_key": "k-default",
            "base_url": "https://api.example/v1",
            "chat_model": "gpt-default",
        },
        "reflection": {
            "provider": "openai",
            "api_key": "k-default",
            "base_url": "https://api.example/v1",
            "chat_model": "gpt-reflection-default",
        },
    }
    client = {
        "default": {"api_key": None, "base_url": "https://override.example/v1"},
        "reflection": {"chat_model": "gpt-reflection-override", "api_key": None},
    }

    with pytest.raises(main.HTTPException, match="llm_profiles.default.api_key cannot be null"):
        main._merge_llm_profiles(defaults, client)


def test_merge_llm_profiles_rejects_null_profile_object():
    defaults = {
        "default": {
            "provider": "openai",
            "api_key": "k-default",
            "base_url": "https://api.example/v1",
            "chat_model": "gpt-default",
        },
    }
    client = {
        "ranking": None,
    }

    with pytest.raises(main.HTTPException, match="llm_profiles.ranking cannot be null"):
        main._merge_llm_profiles(defaults, client)


def test_retrieve_apimw_enabled_from_cfg_defaults_and_override():
    assert main._retrieve_apimw_enabled_from_cfg(None) is True
    assert main._retrieve_apimw_enabled_from_cfg({"retrieve": {}}) is True
    assert main._retrieve_apimw_enabled_from_cfg({"retrieve": {"apimw_enabled": True}}) is True
    assert main._retrieve_apimw_enabled_from_cfg({"retrieve": {"apimw_enabled": False}}) is False


def test_resolve_profile_raises_when_profile_missing():
    svc = SimpleNamespace(llm_profiles=SimpleNamespace(profiles={"default": {}}))
    with pytest.raises(main.HTTPException, match="llm profile 'memory_extract' is not configured"):
        main._resolve_profile(svc, "memory_extract")


def test_imports():
    assert hasattr(main, "app")


def test_run_retrieve_reports_rag_method(monkeypatch: pytest.MonkeyPatch):
    class _FakeSvc:
        async def retrieve(self, *_args, **_kwargs):
            return {"items": []}

    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_k: _FakeSvc())
    out = asyncio.run(main._run_retrieve({"query": "hello", "user": {"user_id": "u", "soul_id": "s"}}))
    assert out["method"] == "rag"


def test_prompt_log_before_only_sets_timer(caplog: pytest.LogCaptureFixture) -> None:
    ctx = SimpleNamespace()
    request_view = SimpleNamespace(kind="chat")
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        main._prompt_log_before(ctx, request_view)
    assert hasattr(ctx, "_llm_start")
    assert caplog.text == ""


def test_prompt_log_after_emits_single_block(caplog: pytest.LogCaptureFixture) -> None:
    ctx = SimpleNamespace(
        _llm_start=0.0,
        operation="turn",
        step_id="respond",
        request_id="req-1",
        model="minimax/minimax-m2.7",
    )
    request_view = SimpleNamespace(
        kind="chat",
        metadata={"payload": {"messages": [{"role": "system", "content": "hello"}]}},
    )
    response_view = SimpleNamespace(content="ok")
    usage = SimpleNamespace(
        finish_reason="stop",
        input_tokens=1,
        output_tokens=2,
        total_tokens=3,
    )
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        main._prompt_log_after(ctx, request_view, response_view, usage)
    text = caplog.text
    assert "===== TURN · respond" in text
    assert "[PROMPT] trace=- req=req-1 op=turn step=respond model=minimax/minimax-m2.7" in text
    assert "[PAYLOAD] trace=- req=req-1 op=turn step=respond kind=chat model=minimax/minimax-m2.7" in text
    assert "[RESPONSE] trace=- req=req-1 op=turn step=respond" in text
    assert "content_chars=2" in text
    assert "\nok\n" in text


def test_prompt_log_on_error_emits_error_block(caplog: pytest.LogCaptureFixture) -> None:
    ctx = SimpleNamespace(
        _llm_start=0.0,
        operation="retrieve",
        step_id="decide_retrieval",
        request_id="req-2",
        model="minimax/minimax-m2.7",
    )
    request_view = SimpleNamespace(
        kind="chat",
        metadata={"payload": {"messages": [{"role": "system", "content": "hello"}]}},
    )
    usage = SimpleNamespace(status="error")
    error = TimeoutError("timed out")
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        main._prompt_log_on_error(ctx, request_view, error, usage)
    text = caplog.text
    assert "===== RETRIEVE · decide_retrieval" in text
    assert "[PROMPT] trace=- req=req-2 op=retrieve step=decide_retrieval model=minimax/minimax-m2.7" in text
    assert "[PAYLOAD] trace=- req=req-2 op=retrieve step=decide_retrieval kind=chat model=minimax/minimax-m2.7" in text
    assert "[ERROR] trace=- req=req-2 op=retrieve step=decide_retrieval" in text
    assert "type=TimeoutError message=timed out" in text


@pytest.mark.parametrize(
    ("addon_enabled", "expected"),
    [
        (True, True),
        (False, False),
    ],
)
def test_run_retrieve_forwards_mental_health_toggle(
    monkeypatch: pytest.MonkeyPatch,
    addon_enabled: bool,
    expected: bool,
):
    captured: dict[str, Any] = {}

    class _FakeSvc:
        async def retrieve(self, *_args, **kwargs):
            captured.update(kwargs)
            return {"items": []}

    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_k: _FakeSvc())
    out = asyncio.run(
        main._run_retrieve(
            {
                "query": "hello",
                "user": {"user_id": "u", "soul_id": "s"},
                "mental_health_addon": addon_enabled,
            }
        )
    )
    assert out["method"] == "rag"
    assert captured["mental_health_enabled"] is expected


def test_run_retrieve_forwards_force_retrieve(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    class _FakeSvc:
        async def retrieve(self, *_args, **kwargs):
            captured.update(kwargs)
            return {"items": []}

    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_k: _FakeSvc())
    out = asyncio.run(
        main._run_retrieve(
            {
                "query": "hello",
                "user": {"user_id": "u", "soul_id": "s"},
                "force_retrieve": True,
            }
        )
    )
    assert out["method"] == "rag"
    assert captured["force_retrieve"] is True


@pytest.mark.asyncio
async def test_apimw_retrieve_pass_sets_force_retrieve(monkeypatch: pytest.MonkeyPatch):
    captured_payload: dict[str, Any] = {}

    async def _fake_run_retrieve(
        payload: dict[str, Any],
        *,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        captured_payload.update(payload)
        return {"result": {"items": []}}

    monkeypatch.setattr(main, "_run_retrieve", _fake_run_retrieve)

    await main._apimw_retrieve_pass(
        payload={"user": {"user_id": "u1", "soul_id": "Echo"}},
        query_text="test topic",
        soul_id="Echo",
        history=[{"role": "user", "name": "Marcos", "content": "hello"}],
        state_row={},
        conversation_id="cid",
        apimw_k=12,
    )

    assert captured_payload["force_retrieve"] is True


def test_run_retrieve_rejects_non_boolean_force_retrieve(monkeypatch: pytest.MonkeyPatch):
    class _FakeSvc:
        async def retrieve(self, *_args, **_kwargs):
            return {"items": []}

    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_k: _FakeSvc())

    with pytest.raises(main.HTTPException, match="'force_retrieve' must be a boolean"):
        asyncio.run(
            main._run_retrieve(
                {
                    "query": "hello",
                    "user": {"user_id": "u", "soul_id": "s"},
                    "force_retrieve": "true",
                }
            )
        )


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


def test_build_cross_conversation_payload_uses_state_default_memorize_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.commit()
    finally:
        con.close()
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_k: db_path)
    out = main._build_cross_conversation_payload(
        "whatsapp:dm:123",
        "u1",
        "Echo",
        {},
        [{"role": "user", "content": "hello"}],
        -1,
        False,
    )
    assert isinstance(out, dict)
    conversation = out.get("conversation")
    assert isinstance(conversation, list) and conversation
    assert conversation[0].get("memorize_chat") is False


def test_build_cross_conversation_payload_request_flag_overrides_state_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.commit()
    finally:
        con.close()
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_k: db_path)
    out = main._build_cross_conversation_payload(
        "whatsapp:dm:123",
        "u1",
        "Echo",
        {"memorize_chat": True},
        [{"role": "user", "content": "hello"}],
        -1,
        False,
    )
    assert isinstance(out, dict)
    conversation = out.get("conversation")
    assert isinstance(conversation, list) and conversation
    assert conversation[0].get("memorize_chat") is True


def test_build_cross_conversation_payload_includes_background_rolling_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.execute(
            "INSERT INTO conversations (conversation_id, memorize_chat, rolling_summary) VALUES (?, ?, ?)",
            ("bg-chat", 0, "rolled summary"),
        )
        con.commit()
    finally:
        con.close()
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_k: db_path)
    monkeypatch.setattr(
        main._message_log,
        "read_background_rolling_summaries",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("legacy helper should not run")),
    )
    out = main._build_cross_conversation_payload(
        "whatsapp:dm:123",
        "u1",
        "Echo",
        {"memorize_chat": True},
        [{"role": "user", "content": "hello"}],
        -1,
        True,
    )
    assert isinstance(out, dict)
    rs = out.get("_background_rolling_summaries")
    assert isinstance(rs, dict)
    assert rs.get("bg-chat", {}).get("summary") == "rolled summary"


def test_build_cross_conversation_payload_preserves_listen_only_cursor_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        now_iso = datetime.now(UTC).isoformat()
        con.execute(
            "INSERT INTO conversations (conversation_id, memorize_chat, digest_cursor, last_memorize_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("trigger", 1, -1, None, now_iso),
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, memorize_chat, rolling_summary_cursor_id, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ("bg-chat", 0, None, now_iso),
        )
        con.executemany(
            "INSERT INTO messages (conversation_id, role, speaker, chat_name, content, source_label, received_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("bg-chat", "user", "Marcos", "Marcos", "old", "whatsapp:dm", "2026-05-01T00:00:00+00:00"),
                ("bg-chat", "user", "Marcos", "Marcos", "new", "whatsapp:dm", "2026-05-01T00:01:00+00:00"),
            ],
        )
        first_id = int(
            con.execute(
                "SELECT MIN(id) AS id FROM messages WHERE conversation_id = ?",
                ("bg-chat",),
            ).fetchone()["id"]
        )
        con.execute(
            "UPDATE conversations SET rolling_summary_cursor_id = ? WHERE conversation_id = ?",
            (first_id, "bg-chat"),
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_k: db_path)
    out = main._build_cross_conversation_payload(
        "trigger",
        "u1",
        "Echo",
        {"memorize_chat": True},
        [{"role": "user", "content": "hello"}],
        -1,
        True,
    )
    assert isinstance(out, dict)
    rows = [
        row for row in list(out.get("conversation") or [])
        if str(row.get("source_conversation_id") or "") == "bg-chat"
    ]
    assert len(rows) == 1
    assert rows[0]["content"] == "new"
    assert rows[0]["memorize_chat"] is False
    assert out["_final_cursors"]["bg-chat"] == rows[0]["source_conversation_index"]


def test_load_cross_tail_from_sources_reads_whatsapp_conversations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor, last_memorize_at) VALUES (?, ?, ?)",
            ("whatsapp:dm:15133278228", 0, "2026-05-01T00:00:00+00:00"),
        )
        con.commit()

        monkeypatch.setattr(
            main._conversation_sources,
            "load_whatsapp_tail",
            lambda **_kwargs: [
                {
                    "conversation_id": "whatsapp:dm:15133278228",
                    "source_conversation_index": 1,
                    "received_at": "2026-05-01T00:00:00+00:00",
                    "content": "hi",
                }
            ],
        )
        monkeypatch.setattr(
            main._conversation_sources,
            "load_sillytavern_tail",
            lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not call sillytavern loader")),
        )
        rows = main._load_cross_tail_from_sources(
            con,
            user_id="u1",
            soul_id="Echo",
            exclude_conversation_id="",
        )
    finally:
        con.close()
    assert len(rows) == 1
    assert rows[0]["content"] == "hi"


def test_turn_state_read_marks_background_error_when_source_assembly_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: (
            {"memorize_chat": True, "digest_cursor": -1, "last_memorize_at": None},
            None,
            None,
        ),
    )
    monkeypatch.setattr(main, "_estimate_unmemorized_tokens", lambda *_a, **_k: main._MIN_CHUNK_TOKENS + 1)
    monkeypatch.setattr(main, "_unmemorized_sleep_gap_detected", lambda *_a, **_k: True)
    monkeypatch.setattr(
        main,
        "_build_cross_conversation_payload",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        main,
        "_set_background_error",
        lambda _cid, **kwargs: captured.update(
            {"code": str(kwargs.get("code") or ""), "detail": str(kwargs.get("detail") or "")}
        ),
    )
    _state, _card, _db, _cache, _intentions, _tokens, queued = main._turn_state_read(
        "cid",
        "u1",
        "Echo",
        {},
        [],
        {"items": []},
        False,
        [{"role": "user", "content": "hello"}],
    )
    assert queued is None
    assert captured["code"] == "forced_memorize_source_failed"
    assert "RuntimeError: boom" in captured["detail"]


def test_turn_state_read_excludes_background_chat_from_segment_trigger(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: (
            {"memorize_chat": False, "digest_cursor": 0, "last_memorize_at": "2026-05-10T00:00:00+00:00"},
            None,
            None,
        ),
    )
    monkeypatch.setattr(main, "_unmemorized_sleep_gap_detected", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not run")))
    state, _soul_card, _db, _cache, _intentions, tokens, queued = main._turn_state_read(
        "cid",
        "u1",
        "Echo",
        {},
        [],
        {"items": []},
        False,
        [{"role": "user", "content": "w " * 10000}],
    )
    assert state["memorize_chat"] is False
    assert tokens == 0
    assert queued is None


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
    from app.services import retrieve_orchestration
    monkeypatch.setattr(retrieve_orchestration, "_format_time_anchor", lambda *_a, **_k: sentinel_anchor)
    out = main._build_retrieve_identity_context("Echo")
    assert out.startswith(f"Today is {sentinel_anchor}.")


def test_build_retrieve_soul_context_queries_includes_full_history() -> None:
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
    assert "[user] current" in text
    assert "[user] msg 1" in text
    assert "[user] msg 15" in text


def test_build_retrieve_soul_context_queries_uses_full_history_for_apimw_rewrite() -> None:
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
    assert "[user] current" in text
    assert "[user] msg 1" in text
    assert "[user] msg 15" in text


def test_build_retrieve_soul_context_queries_orders_chats_before_working_and_intentions() -> None:
    history = [
        {"message_id": "m1", "role": "user", "content": "msg 1"},
        {"message_id": "m2", "role": "user", "content": "msg 2"},
    ]
    queries = main._build_retrieve_soul_context_queries(
        soul_id="Echo",
        message="current",
        history=history,
        state_row={
            "memory_cache": ["cache entry"],
            "intentions_active": {"items": [{"id": "relax", "text": "Relax"}]},
        },
    )
    roles = [str(q.get("role")) for q in queries if isinstance(q, dict)]
    assert roles.index("history") < roles.index("memory_cache")
    assert roles.index("memory_cache") < roles.index("intentions")
    assert roles[-1] == "user"
    working_rows = [q for q in queries if isinstance(q, dict) and q.get("role") == "memory_cache"]
    assert len(working_rows) == 1
    working_text = str((working_rows[0].get("content") or {}).get("text") or "")
    assert working_text.startswith("1. cache entry")


def test_build_retrieve_soul_context_queries_includes_current_chat_heading_for_whatsapp_dm() -> None:
    history = [
        {"message_id": "m1", "role": "user", "name": "Marcos", "content": "hello"},
    ]
    queries = main._build_retrieve_soul_context_queries(
        soul_id="Echo",
        message="current",
        history=history,
        state_row={"memory_cache": [], "intentions_active": {"items": []}},
        conversation_id="whatsapp:dm:Marcos",
    )
    history_rows = [q for q in queries if isinstance(q, dict) and q.get("role") == "history"]
    assert len(history_rows) == 1
    text = str((history_rows[0].get("content") or {}).get("text") or "")
    assert "My WhatsApp Conversations:" in text
    assert "[dm][Marcos] \u2190 current chat" in text


@pytest.mark.asyncio
async def test_run_memorize_episodes_records_failure_progress_on_exception(tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

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
            memorize_segments=[("/tmp/day.json", [{"role": "user", "content": "x"}], 0, 0)],
            svc=_FailingService(),
            scope={"user_id": user_id, "soul_id": soul_id},
            conversation_id=None,
            soul_id=soul_id,
            uid=user_id,
            processed_cursor=-1,
            safe={},
            resource_url="/tmp/day.json",
            chat_key=None,
            tz_name=None,
            prev_len=0,
            merged_len=1,
            force=True,
            sleep_stats=None,
            segments_dir=segments_dir,
        )

    row = main._MEMORIZE_PROGRESS.get(key) or {}
    assert row.get("active") is False
    assert row.get("last_result") == "failure"
    assert "RuntimeError: boom" in str(row.get("error") or "")
    assert key not in main._MEMORIZE_CANCEL


@pytest.mark.asyncio
async def test_run_memorize_episodes_clears_pending_ids_on_extraction_failure(monkeypatch: pytest.MonkeyPatch, tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    class _FailingService:
        async def memorize_episodes_batch(self, **_kwargs):
            raise RuntimeError("boom")

    user_id = "u"
    soul_id = "s"
    conversation_id = "cid-1"
    key = main._memorize_lock_key(user_id, soul_id)
    main._MEMORIZE_PROGRESS.pop(key, None)
    main._MEMORIZE_CANCEL.discard(key)

    state_row: dict[str, Any] = {
        "pending_episode_ids": ["cid-1:0-1"],
        "digest_cursor": 0,
    }

    def fake_load_turn_state_and_soul_card(*_args, **_kwargs):
        return dict(state_row), None, None

    def fake_write_conversation_state(_cid: str, *, updates: dict[str, Any], **_kwargs):
        if "pending_episode_ids" in updates:
            state_row["pending_episode_ids"] = list(updates.get("pending_episode_ids") or [])
        if "digest_cursor" in updates:
            state_row["digest_cursor"] = int(updates["digest_cursor"])
        return dict(state_row), tmp_path / "Echo.db"

    monkeypatch.setattr(main, "_load_turn_state_and_soul_card", fake_load_turn_state_and_soul_card)
    monkeypatch.setattr(main, "_write_conversation_state", fake_write_conversation_state)

    with pytest.raises(RuntimeError):
        await main._run_memorize_episodes(
            memorize_segments=[("/tmp/day.json", [{"role": "user", "content": "x"}], 0, 0)],
            svc=_FailingService(),
            scope={"user_id": user_id, "soul_id": soul_id},
            conversation_id=conversation_id,
            soul_id=soul_id,
            uid=user_id,
            processed_cursor=-1,
            safe={},
            resource_url="/tmp/day.json",
            chat_key=None,
            tz_name=None,
            prev_len=0,
            merged_len=1,
            force=False,
            sleep_stats=None,
            segments_dir=segments_dir,
        )

    assert state_row["pending_episode_ids"] == []
    row = main._MEMORIZE_PROGRESS.get(key) or {}
    assert row.get("active") is False
    assert row.get("last_result") == "failure"


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
    assert crud_endpoints._validate_relationship_speaker_id("entity:brother") == "brother"
    with pytest.raises(main.HTTPException):
        crud_endpoints._validate_relationship_speaker_id("user:marcos")
    with pytest.raises(main.HTTPException):
        crud_endpoints._validate_relationship_speaker_id("entity:brother!")


def test_relationship_item_from_values_filters_non_declared_or_inactive():
    item = crud_endpoints._relationship_item_from_values(
        normalized="brother",
        name="Brother",
        entity_type="person",
        properties={"origin": "user_declared", "relationship": "sibling", "active": True},
    )
    assert item is not None
    assert item["speaker_id"] == "entity:brother"
    assert item["relationship"] == "sibling"

    assert crud_endpoints._relationship_item_from_values(
        normalized="brother",
        name="Brother",
        entity_type="person",
        properties={"origin": "extracted", "relationship": "sibling", "active": True},
    ) is None
    assert crud_endpoints._relationship_item_from_values(
        normalized="brother",
        name="Brother",
        entity_type="person",
        properties={"origin": "user_declared", "active": False},
    ) is None


def test_assert_user_declared_relationship_is_strict():
    crud_endpoints._assert_user_declared_relationship({"origin": "user_declared"})
    with pytest.raises(main.HTTPException):
        crud_endpoints._assert_user_declared_relationship({"origin": ""})
    with pytest.raises(main.HTTPException):
        crud_endpoints._assert_user_declared_relationship({"origin": "extracted"})


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
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: ({"prior_context": "", "memory_cache": [], "intentions_active": {"items": []}}, None, db_path),
    )
    monkeypatch.setattr(
        main,
        "_load_cross_tail_from_sources",
        lambda *_a, **_k: [
            {
                "conversation_id": "whatsapp:dm:other",
                "role": "assistant",
                "speaker": "Echo",
                "chat_name": "Marcos",
                "content": "wa-2",
                "source_label": "whatsapp:dm",
                "received_at": "2026-05-08T11:00:01+00:00",
                "source_conversation_index": 1,
            }
        ],
    )

    captured: dict[str, object] = {}

    async def _fake_run_retrieve(safe: dict[str, object], *, conversation_id: str | None = None) -> dict[str, object]:
        captured["safe"] = safe
        return {"ok": True, "result": {}, "conversation_id": conversation_id}

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
    assert cross_roles.count("history") == 1
    assert cross_roles.count("cross_conversation") == 0


@pytest.mark.asyncio
async def test_conversation_retrieve_uses_payload_history_for_primary_chat_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor, last_memorize_at) VALUES (?, ?, ?)",
            ("whatsapp:dm:15133278228", 0, None),
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
        return {"ok": True, "result": {}, "conversation_id": conversation_id}

    monkeypatch.setattr(main, "_run_retrieve", _fake_run_retrieve)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Echo"},
        "message": "payload current",
        "query": "payload current",
        "history": [
            {"role": "user", "name": "Marcos", "content": "payload prior"},
            {"role": "user", "name": "Marcos", "content": "payload current"},
        ],
    }

    out = await main.conversation_retrieve("whatsapp:dm:15133278228", payload)
    assert out["ok"] is True

    safe = captured["safe"]
    assert isinstance(safe, dict)
    queries = safe.get("queries")
    assert isinstance(queries, list)
    history_entries = [
        q for q in queries
        if isinstance(q, dict) and str(q.get("role") or "").strip() == "history"
    ]
    assert len(history_entries) == 1
    history_text = str(history_entries[0].get("content", {}).get("text", "")).strip()
    assert "payload prior" in history_text
    assert "payload current" in history_text
    assert "[Marcos]" in history_text


@pytest.mark.asyncio
async def test_conversation_retrieve_uses_same_payload_history_for_turn_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor, last_memorize_at) VALUES (?, ?, ?)",
            ("whatsapp:dm:Marcos", 0, None),
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
        return {"ok": True, "result": {}, "conversation_id": conversation_id}

    monkeypatch.setattr(main, "_run_retrieve", _fake_run_retrieve)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Echo"},
        "message": "payload current",
        "query": "payload current",
        "history": [
            {"role": "user", "content": "payload prior"},
            {"role": "user", "content": "payload current"},
        ],
        "build_turn_prompt": True,
    }

    out = await main.conversation_retrieve("whatsapp:dm:Marcos", payload)
    assert out["ok"] is True
    turn_prompt = str(out.get("turn_user_prompt") or "")
    assert "payload prior" in turn_prompt
    assert "payload current" in turn_prompt


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
        return {"ok": True, "result": {}, "conversation_id": conversation_id}

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
async def test_conversation_retrieve_writes_sillytavern_snapshot_not_messages_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Echo.db"
    storage_dir = tmp_path / "resources"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: ({"prior_context": "", "memory_cache": [], "intentions_active": {"items": []}}, None, db_path),
    )
    monkeypatch.setattr(main, "_get_storage_dir", lambda *_a, **_k: storage_dir)

    async def _fake_run_retrieve(safe: dict[str, object], *, conversation_id: str | None = None) -> dict[str, object]:
        return {"ok": True, "result": {}, "conversation_id": conversation_id}

    monkeypatch.setattr(main, "_run_retrieve", _fake_run_retrieve)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Echo"},
        "message": "latest",
        "query": "latest",
        "chat_name": "Echo",
        "history": [
            {"role": "user", "name": "Marcos", "content": "m1"},
            {"role": "assistant", "name": "Echo", "content": "a1"},
        ],
        "queries": [{"role": "message", "content": {"text": "latest"}}],
    }

    out = await main.conversation_retrieve("integrity:chat-1", payload)
    assert out["ok"] is True

    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT role, speaker, chat_name, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            ("integrity:chat-1",),
        ).fetchall()
    finally:
        con.close()
    assert rows == []

    snapshot_rows = conversation_sources.load_sillytavern_tail(
        storage_dir=storage_dir,
        user_id="u1",
        soul_id="Echo",
        conversation_id="integrity:chat-1",
        since_cursor=-1,
        recent_fallback_messages=0,
    )
    assert [row["content"] for row in snapshot_rows] == ["m1", "a1"]


@pytest.mark.asyncio
async def test_conversation_retrieve_includes_sillytavern_cross_tail_from_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Echo.db"
    storage_dir = tmp_path / "resources"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor) VALUES (?, ?)",
            ("whatsapp:dm:15133278228", 0),
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor) VALUES (?, ?)",
            ("integrity:other-chat", 0),
        )
        con.commit()
    finally:
        con.close()

    conversation_sources.persist_sillytavern_history_snapshot(
        storage_dir=storage_dir,
        user_id="u1",
        soul_id="Echo",
        conversation_id="integrity:other-chat",
        history=[{"role": "assistant", "name": "Echo", "content": "st-other-msg"}],
        chat_name="Echo",
    )

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: ({"prior_context": "", "memory_cache": [], "intentions_active": {"items": []}}, None, db_path),
    )
    monkeypatch.setattr(main, "_get_storage_dir", lambda *_a, **_k: storage_dir)

    captured: dict[str, object] = {}

    async def _fake_run_retrieve(safe: dict[str, object], *, conversation_id: str | None = None) -> dict[str, object]:
        captured["safe"] = safe
        return {"ok": True, "result": {}, "conversation_id": conversation_id}

    monkeypatch.setattr(main, "_run_retrieve", _fake_run_retrieve)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Echo"},
        "message": "hello",
        "query": "hello",
        "history": [{"role": "user", "content": "hello"}],
    }

    out = await main.conversation_retrieve("whatsapp:dm:15133278228", payload)
    assert out["ok"] is True
    safe = captured["safe"]
    assert isinstance(safe, dict)
    cross_text = str(safe.get("_cross_conversation_history") or "")
    assert "My SillyTavern Conversations:" in cross_text
    assert "st-other-msg" in cross_text


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
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: ({"prior_context": "", "memory_cache": [], "intentions_active": {"items": []}}, None, db_path),
    )
    monkeypatch.setattr(
        main,
        "_load_cross_tail_from_sources",
        lambda *_a, **_k: [
            {
                "conversation_id": "whatsapp:dm:other",
                "role": "assistant",
                "speaker": "Echo",
                "chat_name": "Marcos",
                "content": "wa-2",
                "source_label": "whatsapp:dm",
                "received_at": "2026-05-08T11:00:01+00:00",
                "source_conversation_index": 1,
            }
        ],
    )

    captured: dict[str, object] = {}

    async def _fake_run_retrieve(safe: dict[str, object], *, conversation_id: str | None = None) -> dict[str, object]:
        captured["safe"] = safe
        return {"ok": True, "result": {}, "conversation_id": conversation_id}

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
            return (
                '{"cache":null,"annulments":[],"rehearsal":"ok",'
                '"response_target":"respond","response":"assistant says hi"}'
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
        "message": "hello",
        "user_name": "Alice",
        "chat_name": "Alice",
        "chat_type": "dm",
        "history": [{"role": "user", "content": "hello"}],
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
async def test_conversation_turn_keeps_response_when_chat_name_differs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """respond no longer depends on chat_name matching."""
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
            return (
                '{"cache":null,"annulments":[],"rehearsal":"answering",'
                '"response_target":"respond","response":"hi Alice"}'
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
    assert out["response"] == "hi Alice"
    assert out["response_target"] == "respond"

    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT role FROM messages WHERE conversation_id = ?",
            ("cid-turn",),
        ).fetchall()
    finally:
        con.close()
    assert [r["role"] for r in rows] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_conversation_turn_private_response_not_persisted_in_origin_chat(
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
            return (
                '{"cache":null,"annulments":[],"rehearsal":"ok",'
                '"response_target":"private","response":"private note to Marcos"}'
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
        "user": {"user_id": "u1", "soul_id": "Siri", "conversation_id": "whatsapp:dm:Raquel"},
        "message": "Hello Siri.",
        "user_name": "Raquel",
        "chat_name": "Raquel",
        "chat_type": "dm",
        "history": [],
        "prompt_override_payload": {
            "user_prompt": "prompt",
            "system_prompt": "system",
            "memory_cache": [],
            "intentions_active": {"items": []},
            "retrieve_rag": {"items": [], "categories": [], "resources": []},
        },
    }

    out = await main.conversation_turn("whatsapp:dm:Raquel", payload)
    assert out["ok"] is True
    assert out["response_target"] == "private"
    assert out["response"] == "private note to Marcos"

    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT role, speaker, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            ("whatsapp:dm:Raquel",),
        ).fetchall()
    finally:
        con.close()

    assert len(rows) == 1
    assert str(rows[0]["role"]) == "user"
    assert str(rows[0]["speaker"]) == "Raquel"
    assert str(rows[0]["content"]) == "Hello Siri."


@pytest.mark.asyncio
async def test_conversation_turn_retries_once_on_parse_failure(
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
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, *_args, **_kwargs) -> str:
            self.calls += 1
            if self.calls == 1:
                return (
                    '{"cache":null,"annulments":[],"rehearsal":"first malformed"},"response_target":"respond",'
                    '"response":"assistant says hi"}'
                )
            return (
                '{"cache":null,"annulments":[],"rehearsal":"retry good",'
                '"response_target":"respond","response":"assistant says hi"}'
            )

    svc = _FakeSvc()

    async def _fake_persist_annulment_memories(**_kwargs):
        return []

    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_k: svc)
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
        "chat_name": "Alice",
        "chat_type": "dm",
        "history": [{"role": "user", "content": "hello"}],
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
    assert svc.calls == 2


@pytest.mark.asyncio
async def test_conversation_turn_allows_respond_when_chat_name_missing_and_logs_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
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
            return (
                '{"cache":null,"annulments":[],"rehearsal":"replying",'
                '"response_target":"respond","response":"hi Bob"}'
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
        "history": [],
        "prompt_override_payload": {
            "user_prompt": "prompt",
            "system_prompt": "system",
            "memory_cache": [],
            "intentions_active": {"items": []},
            "retrieve_rag": {"items": [], "categories": [], "resources": []},
        },
    }

    caplog.set_level(logging.WARNING)
    out = await main.conversation_turn("cid-turn", payload)

    assert out["ok"] is True
    assert out["response"] == "hi Bob"
    assert "missing chat_name for respond" in caplog.text


def test_clear_background_error_if_apimw_owned_preserves_non_apimw_error(monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[dict[str, object]] = []
    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: (
            {"last_background_error": "forced_memorize: RuntimeError: LLM refused"},
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        main,
        "_write_conversation_state",
        lambda conversation_id, soul_id, user_id, updates: writes.append(dict(updates)) or ({"ok": True}, Path("/tmp/fake.db")),
    )
    main._clear_background_error_if_apimw_owned("cid", soul_id="Echo", user_id="u1")
    assert writes == []


def test_clear_background_error_if_apimw_owned_clears_apimw_error(monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[dict[str, object]] = []
    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: (
            {"last_background_error": "apimw_failed: RuntimeError: boom"},
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        main,
        "_write_conversation_state",
        lambda conversation_id, soul_id, user_id, updates: writes.append(dict(updates)) or ({"ok": True}, Path("/tmp/fake.db")),
    )
    main._clear_background_error_if_apimw_owned("cid", soul_id="Echo", user_id="u1")
    assert len(writes) == 1
    assert writes[0]["last_background_error"] is None
    assert writes[0]["last_background_error_at"] is None


@pytest.mark.asyncio
async def test_run_background_rollup_deletes_only_summarized_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.execute(
            "INSERT INTO conversations (conversation_id, soul_id, user_id, memorize_chat, digest_cursor, rolling_summary_cursor_id, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("cid-rollup", "Echo", "u1", 0, 0, None, datetime.now(UTC).isoformat()),
        )
        con.executemany(
            "INSERT INTO messages (conversation_id, role, speaker, content, source_label, received_at) VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("cid-rollup", "user", "U", "old one", "sillytavern", "2026-05-19T00:00:00+00:00"),
                ("cid-rollup", "assistant", "Echo", "old two", "sillytavern", "2026-05-19T00:00:01+00:00"),
            ],
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: (
            {"memorize_chat": False, "rolling_summary_cursor_id": None, "rolling_summary": None},
            None,
            db_path,
        ),
    )
    monkeypatch.setattr(main, "_BACKGROUND_SUMMARY_TOKENS", 1)
    monkeypatch.setattr(main, "_background_sleep_gap_detected", lambda **_kwargs: True)

    class _SlowSvc:
        async def summarize_background_chat_rollup(self, *, prior_summary, messages):  # noqa: ARG002
            c = main._sqlite_connect(db_path)
            try:
                c.execute(
                    "INSERT INTO messages (conversation_id, role, speaker, content, source_label, received_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ("cid-rollup", "user", "U", "NEW_UNSUMMARIZED", "sillytavern", "2026-05-19T00:00:02+00:00"),
                )
                c.commit()
            finally:
                c.close()
            return "rolled summary"

    status = await main._run_background_rollup_for_conversation(
        conversation_id="cid-rollup",
        user_id="u1",
        soul_id="Echo",
        safe_payload={},
        service=_SlowSvc(),
    )
    assert status == "rolled_up"

    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            ("cid-rollup",),
        ).fetchall()
    finally:
        con.close()
    assert [str(row["content"]) for row in rows] == ["NEW_UNSUMMARIZED"]


@pytest.mark.asyncio
async def test_run_background_rollup_persists_background_error_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.execute(
            "INSERT INTO conversations (conversation_id, soul_id, user_id, memorize_chat, digest_cursor, rolling_summary_cursor_id, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("cid-fail", "Echo", "u1", 0, 0, None, datetime.now(UTC).isoformat()),
        )
        con.executemany(
            "INSERT INTO messages (conversation_id, role, speaker, content, source_label, received_at) VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("cid-fail", "user", "U", "one", "sillytavern", "2026-05-19T00:00:00+00:00"),
                ("cid-fail", "assistant", "Echo", "two", "sillytavern", "2026-05-19T00:00:01+00:00"),
            ],
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: (
            {"memorize_chat": False, "rolling_summary_cursor_id": None, "rolling_summary": None},
            None,
            db_path,
        ),
    )
    monkeypatch.setattr(main, "_BACKGROUND_SUMMARY_TOKENS", 1)
    monkeypatch.setattr(main, "_background_sleep_gap_detected", lambda **_kwargs: True)
    writes: list[dict[str, object]] = []
    monkeypatch.setattr(
        main,
        "_write_conversation_state",
        lambda conversation_id, soul_id, user_id, updates: writes.append(dict(updates)) or ({"ok": True}, db_path),
    )

    class _FailingSvc:
        async def summarize_background_chat_rollup(self, *, prior_summary, messages):  # noqa: ARG002
            raise RuntimeError("summary failed")

    with pytest.raises(RuntimeError, match="summary failed"):
        await main._run_background_rollup_for_conversation(
            conversation_id="cid-fail",
            user_id="u1",
            soul_id="Echo",
            safe_payload={},
            service=_FailingSvc(),
        )
    assert writes
    assert str(writes[-1].get("last_background_error") or "").startswith(
        "background_rollup_failed: RuntimeError: summary failed"
    )
    assert str(writes[-1].get("last_background_error_at") or "").strip()
