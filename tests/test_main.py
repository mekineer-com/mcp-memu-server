"""Basic tests for the application."""

import asyncio
import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app import main
from app.services import conversation_sources, crud_endpoints


def _use_hermes_state_whatsapp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        main._CONFIG,
        "hermes",
        {
            "home": "~/.hermes",
            "whatsapp_history_source": "hermes_state",
            "whatsapp_web_source_db": "~/.hermes/whatsapp/web_source.db",
            "whatsapp_reply_prefix": "",
        },
    )


def _messages_table_exists(con: sqlite3.Connection) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'messages'"
    ).fetchone()
    return row is not None


def test_format_all_chat_history_for_ai_merges_current_and_cross_chats() -> None:
    rendered = main._format_all_chat_history_for_ai(
        current_history=[
            {
                "role": "user",
                "name": "Marcos",
                "content": "current hello",
                "ts_ms": 1_770_000_000_000,
            }
        ],
        cross_tail=[
            {
                "conversation_id": "sillytavern:Siri",
                "role": "assistant",
                "speaker": "Siri",
                "chat_name": "Siri",
                "content": "cross hello",
                "received_at": "2026-05-08T11:00:00+00:00",
            }
        ],
        conversation_id="whatsapp:group:family@g.us",
        soul_id="Siri",
        chat_label="[group][Household Group]",
    )

    assert "My SillyTavern Conversations:" in rendered
    assert "[dm][Siri]" in rendered
    assert "[Siri] cross hello" in rendered
    assert "My WhatsApp Conversations:" in rendered
    assert "[group][Household Group] \u2190 current chat" in rendered
    assert "[Marcos] current hello" in rendered


def test_format_all_chat_history_for_ai_can_render_without_current_chat_marker() -> None:
    rendered = main._format_all_chat_history_for_ai(
        current_history=[
            {
                "role": "user",
                "name": "Marcos",
                "content": "window hello",
                "ts_ms": 1_770_000_000_000,
            }
        ],
        cross_tail=[
            {
                "conversation_id": "sillytavern:Siri",
                "role": "assistant",
                "speaker": "Siri",
                "chat_name": "Siri",
                "content": "cross hello",
                "received_at": "2026-05-08T11:00:00+00:00",
            }
        ],
        conversation_id="whatsapp:group:family@g.us",
        soul_id="Siri",
        chat_label="[group][Household Group]",
        mark_current_chat=False,
    )

    assert "current chat" not in rendered
    assert "[group][Household Group]" in rendered
    assert "[Marcos] window hello" in rendered
    assert "[dm][Siri]" in rendered
    assert "[Siri] cross hello" in rendered


def test_format_all_chat_history_for_ai_places_activities_before_chats() -> None:
    rendered = main._format_all_chat_history_for_ai(
        current_history=[
            {
                "role": "user",
                "name": "User A",
                "content": "current hello",
                "ts_ms": 1_770_000_000_000,
            }
        ],
        cross_tail=[
            {
                "conversation_id": "activity:dm:SoulA",
                "role": "assistant",
                "speaker": "SoulA",
                "chat_name": "SoulA",
                "content": "I wrote a note to myself.",
                "received_at": "2026-05-08T10:00:00+00:00",
            },
            {
                "conversation_id": "sillytavern:SoulA",
                "role": "assistant",
                "speaker": "SoulA",
                "chat_name": "SoulA",
                "content": "cross hello",
                "received_at": "2026-05-08T11:00:00+00:00",
            },
        ],
        conversation_id="whatsapp:dm:contact-a",
        soul_id="SoulA",
        chat_label="[dm][Contact A]",
    )

    assert rendered.index("My Activities:") < rendered.index("My SillyTavern Conversations:")
    assert rendered.index("My Activities:") < rendered.index("My WhatsApp Conversations:")
    assert "[dm][SoulA]" in rendered
    assert "[SoulA] I wrote a note to myself." in rendered


def test_format_all_chat_history_for_ai_uses_current_chat_name_without_label() -> None:
    rendered = main._format_all_chat_history_for_ai(
        current_history=[
            {
                "role": "assistant",
                "speaker": "Siri",
                "chat_name": "Siri",
                "content": "current hello",
                "ts_ms": 1_770_000_000_000,
            }
        ],
        cross_tail=[],
        conversation_id="integrity:32bfed88-ee89-4053-81f8-3dba8b973857",
        soul_id="Siri",
    )

    assert "[dm][Siri] \u2190 current chat" in rendered
    assert "integrity:32bfed88-ee89-4053-81f8-3dba8b973857" not in rendered


def test_conversation_state_schema_migrates_pending_segment_ids_from_old_name(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        con.execute(
            """
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY,
                soul_id TEXT,
                user_id TEXT,
                pending_episode_ids JSON DEFAULT '[]',
                memorize_chat INTEGER DEFAULT 1,
                digest_cursor INTEGER DEFAULT 0,
                rolling_summary TEXT,
                rolling_summary_cursor_id INTEGER,
                rolling_summary_updated_at DATETIME,
                prior_context TEXT,
                last_memorize_at DATETIME,
                updated_at DATETIME,
                undo_snapshot JSON,
                last_background_error TEXT,
                last_background_error_at DATETIME,
                last_consolidation_error TEXT,
                last_consolidation_error_at DATETIME
            )
            """
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, pending_episode_ids) VALUES (?, ?)",
            ("cid-old", json.dumps(["cid-old:0-1"])),
        )

        main._sqlite_ensure_conversation_state_schema(con)
        row = main._conversation_state_row(con, "cid-old")
        state = main._conversation_state_from_row(row)

    assert state is not None
    assert state["pending_segment_ids"] == ["cid-old:0-1"]


@pytest.mark.asyncio
async def test_run_consolidation_task_repeats_until_pending_span_is_short(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    state_writes: list[dict[str, Any]] = []

    async def fake_pipeline_once(**_kwargs):
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            return {"status": "ok", "result": {"remaining_segment_ids": ["seg-2", "seg-3"]}}
        if len(calls) == 2:
            return {"status": "ok", "result": {"remaining_segment_ids": ["seg-3"]}}
        return {"status": "skipped", "reason": "pending_span_too_short"}

    def fake_write_state(_cid, *, soul_id, user_id, updates):
        state_writes.append({"soul_id": soul_id, "user_id": user_id, "updates": updates})
        return ({}, Path("/tmp/unused.db"))

    monkeypatch.setattr(main, "_run_consolidation_pipeline_once", fake_pipeline_once)
    monkeypatch.setattr(main, "_write_conversation_state", fake_write_state)

    out = await main._run_consolidation_task(
        object(),
        conversation_id="cid-loop",
        soul_id="SoulLoop",
        uid="UserLoop",
    )

    assert out == {"ok": True, "status": "ok"}
    assert len(calls) == 3
    assert state_writes[-1]["updates"] == {
        "last_consolidation_error": None,
        "last_consolidation_error_at": None,
    }


@pytest.mark.asyncio
async def test_turn_launch_apimw_tracks_background_task(monkeypatch: pytest.MonkeyPatch) -> None:
    release = asyncio.Event()

    async def fake_run_apimw(*_args: object, **_kwargs: object) -> None:
        await release.wait()

    monkeypatch.setattr(main, "_apimw_cadence_from_cfg", lambda *_a, **_k: 1)
    monkeypatch.setattr(main, "_apimw_cadence_due", lambda *_a, **_k: True)
    monkeypatch.setattr(main, "_mark_apimw_inflight", lambda *_a, **_k: True)
    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: ({"digest_cursor": -1}, None, None),
    )
    monkeypatch.setattr(main, "_run_apimw", fake_run_apimw)

    before = set(main._BACKGROUND_TASKS)
    status = main._turn_launch_apimw("cid-apimw-track", "u1", "Echo", {}, [])

    created = set(main._BACKGROUND_TASKS) - before
    assert status == "started"
    assert len(created) == 1
    assert main._active_background_task_count() >= 1

    release.set()
    await asyncio.wait_for(asyncio.gather(*created), timeout=1)
    await asyncio.sleep(0)
    assert all(task not in main._BACKGROUND_TASKS for task in created)


@pytest.mark.asyncio
async def test_shutdown_waits_for_tracked_background_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    release = asyncio.Event()
    kill_calls: list[tuple[int, int]] = []

    async def slow_background() -> None:
        await release.wait()

    task = asyncio.create_task(slow_background())
    main._BACKGROUND_TASKS.add(task)
    monkeypatch.setattr(main.os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))

    shutdown_task = asyncio.create_task(main._shutdown_when_idle(max_wait_sec=1))
    await asyncio.sleep(0.05)
    assert kill_calls == []

    release.set()
    await asyncio.wait_for(shutdown_task, timeout=2)
    await asyncio.sleep(0)
    assert kill_calls
    main._BACKGROUND_TASKS.discard(task)


def test_current_whatsapp_history_uses_configured_web_source_and_filters_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main,
        "_resolve_cross_source_paths",
        lambda: (tmp_path, tmp_path / ".hermes", None, tmp_path / "state.db"),
    )
    monkeypatch.setattr(
        main,
        "_resolve_whatsapp_source_config",
        lambda: ("web_source", tmp_path / "web_source.db", "✦ *Siri*: "),
    )

    captured: dict[str, Any] = {}

    def _fake_web_tail(**kwargs):
        captured.update(kwargs)
        return [
            {"source_message_id": "before", "role": "user", "content": "before"},
            {"source_message_id": "true_chat_CURRENT_me", "role": "user", "content": "current"},
        ]

    monkeypatch.setattr(main._conversation_sources, "load_whatsapp_web_source_tail", _fake_web_tail)

    rows = main._load_current_whatsapp_history_from_source(
        "whatsapp:dm:15133278228",
        "Siri",
        active_since=1780160400.0,
        external_message_id="CURRENT",
    )

    assert [row["content"] for row in rows or []] == ["before"]
    assert captured["since_cursor"] == -1
    assert captured["recent_fallback_messages"] == 0
    assert captured["min_timestamp"] == 1780160400.0
    assert captured["max_messages"] == 250


def test_current_whatsapp_history_empty_web_source_does_not_fallback_to_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main,
        "_resolve_cross_source_paths",
        lambda: (tmp_path, tmp_path / ".hermes", None, tmp_path / "state.db"),
    )
    monkeypatch.setattr(
        main,
        "_resolve_whatsapp_source_config",
        lambda: ("web_source", tmp_path / "web_source.db", "✦ *Echo*: "),
    )
    monkeypatch.setattr(main._conversation_sources, "load_whatsapp_web_source_tail", lambda **_kwargs: [])
    monkeypatch.setattr(
        main._conversation_sources,
        "load_whatsapp_tail",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("state.db fallback should not run")),
    )

    rows = main._load_current_whatsapp_history_from_source(
        "whatsapp:dm:15133278228",
        "Echo",
        active_since=None,
    )

    assert rows == []


def test_normalize_turn_history_preserves_source_speaker_and_received_at() -> None:
    rows = main._normalize_turn_history([
        {
            "role": "user",
            "speaker": "Contact A",
            "content": "hi",
            "received_at": "2026-05-30T17:00:00+00:00",
        }
    ])

    assert rows == [
        {
            "role": "user",
            "content": "hi",
            "source_message_id": "0",
            "name": "Contact A",
            "ts_ms": 1780160400000,
        }
    ]


def test_stamp_assistant_display_name_applies_only_when_missing():
    rows = [
        {"role": "assistant", "content": "a"},
        {"role": "assistant", "name": "Existing", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    main._stamp_assistant_display_name(rows, "Siri")
    assert rows[0]["name"] == "Siri"
    assert rows[0]["speaker"] == "Siri"
    assert rows[1]["name"] == "Existing"
    assert "speaker" not in rows[1]
    assert "name" not in rows[2]


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
        "custom": None,
    }

    with pytest.raises(main.HTTPException, match="llm_profiles.custom cannot be null"):
        main._merge_llm_profiles(defaults, client)


def test_turn_generation_metadata_uses_default_profile(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(main._CONFIG, "claude_code", False)
    payload = {
        "llm_profiles": {
            "default": {
                "provider": "nanogpt",
                "chat_model": "mistralai/mistral-small-4-119b-2603",
            }
        }
    }

    assert main._turn_generation_metadata(payload) == {
        "api": "nanogpt",
        "model": "mistralai/mistral-small-4-119b-2603",
    }


def test_turn_generation_metadata_prefers_claude_code(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(main._CONFIG, "claude_code", True)
    monkeypatch.setitem(main._CONFIG, "claude_code_model", "claude-sonnet-4-6")

    assert main._turn_generation_metadata({"llm_profiles": {"default": {"provider": "nanogpt", "chat_model": "mistral"}}}) == {
        "api": "claude_code",
        "model": "claude-sonnet-4-6",
    }


def test_retrieve_apimw_enabled_from_cfg_defaults_and_override():
    assert main._retrieve_apimw_enabled_from_cfg(None) is True
    assert main._retrieve_apimw_enabled_from_cfg({"retrieve": {}}) is True
    assert main._retrieve_apimw_enabled_from_cfg({"retrieve": {"apimw_enabled": True}}) is True
    assert main._retrieve_apimw_enabled_from_cfg({"retrieve": {"apimw_enabled": False}}) is False


def test_resolve_profile_if_configured_defaults_when_step_profile_missing():
    svc = SimpleNamespace(llm_profiles=SimpleNamespace(profiles={"default": {}}))
    assert main._resolve_profile_if_configured(svc, "memory_extract") is None


def test_resolve_profile_if_configured_uses_named_step_profile():
    svc = SimpleNamespace(llm_profiles=SimpleNamespace(profiles={"default": {}, "memory_extract": {}}))
    assert main._resolve_profile_if_configured(svc, "memory_extract") == "memory_extract"


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


@pytest.mark.parametrize(
    ("config_enabled", "expected"),
    [
        (True, True),
        (False, False),
    ],
)
def test_run_retrieve_uses_config_mental_health_default(
    monkeypatch: pytest.MonkeyPatch,
    config_enabled: bool,
    expected: bool,
):
    captured: dict[str, Any] = {}

    class _FakeSvc:
        async def retrieve(self, *_args, **kwargs):
            captured.update(kwargs)
            return {"items": []}

    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_k: _FakeSvc())
    monkeypatch.setitem(main._CONFIG, "retrieve", {"mental_health_query": config_enabled})
    out = asyncio.run(
        main._run_retrieve(
            {
                "query": "hello",
                "user": {"user_id": "u", "soul_id": "s"},
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
async def test_apimw_retrieve_items_sets_force_retrieve_and_item_count(monkeypatch: pytest.MonkeyPatch):
    captured_payload: dict[str, Any] = {}

    async def _fake_run_retrieve(
        payload: dict[str, Any],
        *,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        captured_payload.update(payload)
        return {"result": {"items": []}}

    monkeypatch.setattr(main, "_run_retrieve", _fake_run_retrieve)

    await main._apimw_retrieve_items(
        payload={"user": {"user_id": "u1", "soul_id": "Echo"}, "trace_id": "turn-trace"},
        focus_text="recent conversation",
        conversations_block="My WhatsApp Conversations:\n\n[dm][Marcos]\n[Marcos] hello",
        soul_id="Echo",
        history=[{"role": "user", "name": "Marcos", "content": "hello"}],
        state_row={},
        conversation_id="cid",
        apimw_k=12,
        trace_id="apimw-trace",
    )

    assert captured_payload["force_retrieve"] is True
    assert captured_payload["query"] == "recent conversation"
    history_text = "\n".join(
        str((query.get("content") or {}).get("text") or "")
        for query in captured_payload["queries"]
        if isinstance(query, dict) and query.get("role") == "history"
    )
    assert "My WhatsApp Conversations:" in history_text
    assert "[dm][Marcos]" in history_text
    assert captured_payload["retrieve_config"]["item"]["top_k"] == 12
    assert captured_payload["trace_id"] == "apimw-trace"


@pytest.mark.asyncio
async def test_apimw_random_items_request_active_only(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    async def _fake_retrieve_items(*_args, **_kwargs):
        return {}, []

    class _Repo:
        def list_items(self, scope, *, include_superseded=False):
            captured["scope"] = scope
            captured["include_superseded"] = include_superseded
            return {}

    monkeypatch.setattr(main, "_apimw_retrieve_items", _fake_retrieve_items)
    svc = SimpleNamespace(database=SimpleNamespace(memory_item_repo=_Repo()))

    await main._apimw_collect_memory_items(
        svc,
        payload={"user": {"user_id": "u1", "soul_id": "Echo"}},
        focus_text="recent conversation",
        conversations_block="My WhatsApp Conversations:\n\n[dm][Marcos]\n[Marcos] hello",
        history=[],
        state_row={},
        conversation_id="cid",
        soul_id="Echo",
        apimw_k=20,
        apimw_random_count=5,
        scope={"user_id": "u1", "soul_id": "Echo"},
        trace_id="apimw-trace",
    )

    assert captured == {
        "scope": {"user_id": "u1", "soul_id": "Echo"},
        "include_superseded": False,
    }


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
    assert out["pending_segment_ids"] == ["m2", "m1"]
    assert out["skipped_reasons"] == ["skip-a", "skip-b"]
    assert "results" in out
    assert [res["id"] for res in out["resources"]] == ["r1", "r2"]


def test_estimate_unmemorized_tokens_respects_digest_cursor():
    messages = [
        {"content": "one two three"},
        {"content": "four five six"},
        {"content": "seven eight nine"},
    ]
    full_tokens = main._estimate_unmemorized_tokens(messages, -1)
    tail_tokens = main._estimate_unmemorized_tokens(messages, 1)
    assert full_tokens > tail_tokens > 0
    assert tail_tokens == main._estimate_unmemorized_tokens(messages[2:], -1)
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
        3 * 60 * 60,
    )

    assert splits == [1, 3, 5]
    assert stats["nights_qual"] == 3


def test_turn_launch_apimw_uses_global_turn_cadence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(main, "_apimw_cadence_from_cfg", lambda *_a, **_k: 3)
    monkeypatch.setattr(main, "_mark_apimw_inflight", lambda *_a, **_k: False)
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_k: db_path)

    for cid in ("cid-1", "cid-2"):
        assert main._turn_launch_apimw(cid, "u1", "Echo", {}, []) == "skipped_cadence"

    status_three = main._turn_launch_apimw(
        "cid-3",
        "u1",
        "Echo",
        {},
        [],
    )
    assert status_three == "skipped_inflight"


@pytest.mark.asyncio
async def test_turn_launch_apimw_passes_floored_history_after_memorize(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "_apimw_cadence_from_cfg", lambda *_a, **_k: 1)
    monkeypatch.setattr(main, "_mark_apimw_inflight", lambda *_a, **_k: True)
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_k: tmp_path / "state.db")
    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: ({"digest_cursor": 8, "last_memorize_at": "2026-05-01T00:00:00+00:00"}, None, None),
    )

    captured: dict[str, object] = {}

    async def _fake_run_apimw(*_args: object, **kwargs: object) -> None:
        captured["current_history"] = kwargs.get("current_history")

    monkeypatch.setattr(main, "_run_apimw", _fake_run_apimw)

    history = [
        {"role": "assistant", "content": f"msg_{idx}", "source_conversation_index": idx}
        for idx in range(10)
    ]
    status = main._turn_launch_apimw("cid", "u1", "Echo", {}, history)
    await asyncio.sleep(0)

    assert status == "started"
    assert [row["content"] for row in captured.get("current_history") or []] == [f"msg_{idx}" for idx in range(2, 10)]



def test_normalize_conversation_uses_created_at_when_timestamp_missing():
    conv = [{"role": "user", "content": "hello", "created_at": "2026-04-16T12:00:00Z"}]
    out = main._normalize_conversation(conv)

    assert isinstance(out, list) and out
    assert out[0]["ts_ms"] == int(datetime(2026, 4, 16, 12, 0, tzinfo=UTC).timestamp() * 1000)


def test_normalize_conversation_uses_received_at_when_timestamp_missing():
    conv = [{"role": "user", "content": "hello", "received_at": "2026-04-16T12:00:00Z"}]
    out = main._normalize_conversation(conv)

    assert isinstance(out, list) and out
    assert out[0]["ts_ms"] == int(datetime(2026, 4, 16, 12, 0, tzinfo=UTC).timestamp() * 1000)
    assert out[0]["received_at"] == "2026-04-16T12:00:00Z"


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


def test_normalize_conversation_preserves_chat_name_for_rendering():
    conv = [
        {
            "role": "user",
            "name": "Marcos",
            "chat_name": "Siri",
            "content": "hello",
        }
    ]
    out = main._normalize_conversation(conv)
    assert isinstance(out, list) and out
    assert out[0]["chat_name"] == "Siri"


def test_stamp_current_conversation_metadata_adds_render_labels():
    rows = [
        {"role": "user", "name": "Marcos", "content": "hello"},
        {
            "role": "user",
            "name": "Liz",
            "content": "cross",
            "source_conversation_id": "whatsapp:dm:447879696252",
            "chat_name": "Contact B",
        },
    ]
    main._memorize_endpoint.stamp_current_conversation_metadata(
        rows,
        conversation_id="integrity:32bfed88-ee89-4053-81f8-3dba8b973857",
        chat_name="Siri",
    )
    assert rows[0]["source_conversation_id"] == "integrity:32bfed88-ee89-4053-81f8-3dba8b973857"
    assert rows[0]["chat_name"] == "Siri"
    assert rows[1]["source_conversation_id"] == "whatsapp:dm:447879696252"
    assert rows[1]["chat_name"] == "Contact B"


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
        {"memorize_chat": True, "chat_name": "Contact B"},
        [{"role": "user", "content": "hello"}],
        -1,
        False,
    )
    assert isinstance(out, dict)
    conversation = out.get("conversation")
    assert isinstance(conversation, list) and conversation
    assert conversation[0].get("memorize_chat") is True
    assert conversation[0].get("chat_name") == "Contact B"


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


def test_build_cross_conversation_payload_preserves_background_cursor_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_hermes_state_whatsapp(monkeypatch)
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
            ("whatsapp:dm:bg-chat", 0, 10, now_iso),
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_k: db_path)
    monkeypatch.setattr(
        main._conversation_sources,
        "load_whatsapp_tail_after_message_id",
        lambda **_kwargs: [
            {
                "id": 11,
                "role": "user",
                "speaker": "Marcos",
                "chat_name": "Marcos",
                "content": "new",
                "source_label": "whatsapp:dm",
                "received_at": "2026-05-01T00:01:00+00:00",
                "conversation_id": "whatsapp:dm:bg-chat",
                "source_conversation_id": "whatsapp:dm:bg-chat",
                "source_conversation_index": 11,
            }
        ],
    )
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
        if str(row.get("source_conversation_id") or "") == "whatsapp:dm:bg-chat"
    ]
    assert len(rows) == 1
    assert rows[0]["content"] == "new"
    assert rows[0]["memorize_chat"] is False
    assert out["_final_cursors"]["whatsapp:dm:bg-chat"] == rows[0]["source_conversation_index"]


def test_build_cross_conversation_payload_isolates_cross_source_failures(
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
            "INSERT INTO conversations (conversation_id, memorize_chat, digest_cursor, last_memorize_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("bg-good", 1, -1, None, now_iso),
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, memorize_chat, digest_cursor, last_memorize_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("bg-bad", 1, -1, None, now_iso),
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_k: db_path)
    monkeypatch.setattr(main, "_resolve_cross_source_paths", lambda: (tmp_path, None, None, None))

    def _fake_tail_loader(**kwargs):
        cid = str(kwargs.get("conversation_id") or "")
        if cid == "bg-bad":
            raise RuntimeError("source missing")
        if cid == "bg-good":
            return [
                {
                    "role": "user",
                    "speaker": "Marcos",
                    "chat_name": "Marcos",
                    "content": "hello from good",
                    "source_label": "whatsapp:dm",
                    "received_at": "2026-05-01T00:01:00+00:00",
                    "conversation_id": "bg-good",
                    "source_conversation_id": "bg-good",
                    "source_conversation_index": 0,
                }
            ]
        return []

    monkeypatch.setattr(main, "_load_tail_for_source_conversation", _fake_tail_loader)

    out = main._build_cross_conversation_payload(
        "trigger",
        "u1",
        "Echo",
        {"memorize_chat": True},
        [{"role": "user", "content": "hello from trigger"}],
        -1,
        True,
    )
    assert isinstance(out, dict)
    rows = [
        row for row in list(out.get("conversation") or [])
        if str(row.get("source_conversation_id") or "") == "bg-good"
    ]
    assert len(rows) == 1
    assert rows[0]["content"] == "hello from good"


def test_build_cross_conversation_payload_uses_max_nonnegative_cursor_for_lineage_tail(
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
            "INSERT INTO conversations (conversation_id, memorize_chat, digest_cursor, last_memorize_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("whatsapp:dm:bg-chat", 1, 0, now_iso, now_iso),
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_k: db_path)
    monkeypatch.setattr(main, "_resolve_cross_source_paths", lambda: (tmp_path, None, None, None))
    monkeypatch.setattr(
        main,
        "_load_tail_for_source_conversation",
        lambda **kwargs: [
            {
                "role": "user",
                "speaker": "Marcos",
                "chat_name": "Marcos",
                "content": "new child",
                "source_label": "whatsapp:dm",
                "received_at": "2026-05-01T00:01:00+00:00",
                "conversation_id": kwargs["conversation_id"],
                "source_conversation_id": kwargs["conversation_id"],
                "source_conversation_index": 1,
            },
            {
                "role": "user",
                "speaker": "Marcos",
                "chat_name": "Marcos",
                "content": "newer parent context",
                "source_label": "whatsapp:dm",
                "received_at": "2026-05-01T00:02:00+00:00",
                "conversation_id": kwargs["conversation_id"],
                "source_conversation_id": kwargs["conversation_id"],
                "source_conversation_index": -1,
            },
        ]
        if kwargs["conversation_id"] == "whatsapp:dm:bg-chat"
        else [],
    )

    out = main._build_cross_conversation_payload(
        "trigger",
        "u1",
        "Echo",
        {"memorize_chat": True},
        [{"role": "user", "content": "hello from trigger"}],
        -1,
        True,
    )
    assert isinstance(out, dict)
    assert out["_final_cursors"]["whatsapp:dm:bg-chat"] == 1


def test_load_tail_for_source_conversation_uses_web_source_when_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setitem(
        main._CONFIG,
        "hermes",
        {
            "whatsapp_history_source": "web_source",
            "whatsapp_web_source_db": str(tmp_path / "web_source.db"),
            "whatsapp_reply_prefix": "✦ *Siri*: ",
        },
    )
    monkeypatch.setattr(main, "_load_soul_active_since", lambda *_a, **_k: 100.0)

    def _fake_web_tail(**kwargs):
        captured.update(kwargs)
        return [
            {
                "role": "assistant",
                "speaker": "Siri",
                "content": "from web source",
                "source_conversation_index": 12,
            }
        ]

    monkeypatch.setattr(main._conversation_sources, "load_whatsapp_web_source_tail", _fake_web_tail)

    rows = main._load_tail_for_source_conversation(
        conversation_id="whatsapp:dm:15133278228",
        user_id="u1",
        soul_id="Siri",
        since_cursor=10,
        recent_fallback_messages=0,
        storage_dir=tmp_path,
        hermes_home_path=tmp_path / ".hermes",
        sessions_index_path=None,
        state_db_path=None,
    )

    assert rows[0]["content"] == "from web source"
    assert captured["since_cursor"] == 10
    assert captured["soul_id"] == "Siri"
    assert captured["reply_prefix"] == "✦ *Siri*: "
    assert captured["min_timestamp"] == 100.0


def test_load_background_rollup_tail_uses_assistant_source_ids_for_web_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main,
        "_resolve_cross_source_paths",
        lambda: (tmp_path, tmp_path / ".hermes", tmp_path / "sessions.json", tmp_path / "state.db"),
    )
    monkeypatch.setattr(
        main,
        "_resolve_whatsapp_source_config",
        lambda: ("web_source", tmp_path / "web_source.db", "✦ *Echo*: "),
    )
    monkeypatch.setattr(main, "_load_soul_active_since", lambda *_a, **_k: 100.0)
    monkeypatch.setattr(
        main._conversation_sources,
        "load_whatsapp_assistant_source_message_ids",
        lambda **_kwargs: {"ASSISTANT-ID"},
    )
    captured: dict[str, Any] = {}

    def _fake_tail(**kwargs):
        captured.update(kwargs)
        return [
            {
                "role": "assistant",
                "content": "reply",
                "source_conversation_index": 11,
            }
        ]

    monkeypatch.setattr(main._conversation_sources, "load_whatsapp_web_source_tail_after_rowid", _fake_tail)

    rows = main._load_background_rollup_tail(
        conversation_id="whatsapp:dm:bg-chat",
        user_id="u1",
        soul_id="Echo",
        rolling_summary_cursor_id=10,
    )

    assert rows[0]["speaker"] == "Echo"
    assert captured["assistant_source_message_ids"] == {"ASSISTANT-ID"}
    assert captured["after_rowid"] == 10
    assert captured["min_timestamp"] == 100.0


def test_build_cross_conversation_payload_does_not_advance_cursor_for_parent_only_tail(
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
            "INSERT INTO conversations (conversation_id, memorize_chat, digest_cursor, last_memorize_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("whatsapp:dm:bg-chat", 1, -1, None, now_iso),
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_k: db_path)
    monkeypatch.setattr(main, "_resolve_cross_source_paths", lambda: (tmp_path, None, None, None))
    monkeypatch.setattr(
        main,
        "_load_tail_for_source_conversation",
        lambda **kwargs: [
            {
                "role": "user",
                "speaker": "Marcos",
                "chat_name": "Marcos",
                "content": "parent only",
                "source_label": "whatsapp:dm",
                "received_at": "2026-05-01T00:01:00+00:00",
                "conversation_id": kwargs["conversation_id"],
                "source_conversation_id": kwargs["conversation_id"],
                "source_conversation_index": -1,
            }
        ]
        if kwargs["conversation_id"] == "whatsapp:dm:bg-chat"
        else [],
    )

    out = main._build_cross_conversation_payload(
        "trigger",
        "u1",
        "Echo",
        {"memorize_chat": True},
        [{"role": "user", "content": "hello from trigger"}],
        -1,
        True,
    )
    assert isinstance(out, dict)
    assert "whatsapp:dm:bg-chat" not in out["_final_cursors"]


@pytest.mark.asyncio
async def test_build_cross_conversation_payload_keeps_background_tail_without_rollup_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_hermes_state_whatsapp(monkeypatch)
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
            ("whatsapp:dm:bg-chat", 0, 10, now_iso),
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_k: db_path)
    monkeypatch.setattr(
        main._conversation_sources,
        "load_whatsapp_tail_after_message_id",
        lambda **_kwargs: [
            {
                "id": 11,
                "role": "user",
                "speaker": "Marcos",
                "chat_name": "Marcos",
                "content": "new one",
                "source_label": "whatsapp:dm",
                "received_at": "2026-05-01T00:01:00+00:00",
                "conversation_id": "whatsapp:dm:bg-chat",
                "source_conversation_id": "whatsapp:dm:bg-chat",
                "source_conversation_index": 11,
            },
            {
                "id": 12,
                "role": "user",
                "speaker": "Marcos",
                "chat_name": "Marcos",
                "content": "new two",
                "source_label": "whatsapp:dm",
                "received_at": "2026-05-01T04:01:00+00:00",
                "conversation_id": "whatsapp:dm:bg-chat",
                "source_conversation_id": "whatsapp:dm:bg-chat",
                "source_conversation_index": 12,
            },
        ],
    )
    queued: list[dict[str, Any]] = []
    monkeypatch.setattr(
        main,
        "_queue_background_rollup_task",
        lambda **kwargs: queued.append(kwargs),
    )

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
    assert queued == []
    conversation = out.get("conversation")
    assert isinstance(conversation, list)
    background_rows = [
        msg for msg in conversation
        if isinstance(msg, dict) and msg.get("source_conversation_id") == "whatsapp:dm:bg-chat"
    ]
    assert len(background_rows) == 2
    assert all(row.get("memorize_chat") is False for row in background_rows)


@pytest.mark.asyncio
async def test_run_background_rollup_for_conversation_updates_summary_and_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Echo.db"
    db_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: (
            {
                "memorize_chat": False,
                "rolling_summary": "old summary",
                "rolling_summary_cursor_id": 10,
            },
            None,
            db_path,
        ),
    )
    monkeypatch.setattr(main, "_estimate_tokens", lambda *_a, **_k: 1200)
    monkeypatch.setattr(main, "_background_sleep_gap_detected", lambda **_kwargs: True)
    monkeypatch.setattr(
        main,
        "_load_background_rollup_tail",
        lambda **_kwargs: [
            {
                "role": "user",
                "speaker": "Marcos",
                "content": "first",
                "source_label": "whatsapp:dm",
                "source_conversation_index": 11,
                "received_at": "2026-05-01T00:01:00+00:00",
            },
            {
                "role": "user",
                "speaker": "Marcos",
                "content": "second",
                "source_label": "whatsapp:dm",
                "source_conversation_index": 12,
                "received_at": "2026-05-01T04:01:00+00:00",
            },
        ],
    )

    class _FakeSvc:
        async def summarize_background_chat_rollup(
            self,
            *,
            prior_summary: str | None,
            messages: list[dict[str, Any]],
            soul_name: str | None = None,
        ) -> str:
            assert prior_summary == "old summary"
            assert len(messages) == 2
            assert soul_name == "Echo"
            return "rolled summary"

    captured_updates: dict[str, Any] = {}

    def _capture_write_state(
        _conversation_id: str,
        *,
        soul_id: str,
        user_id: str,
        updates: dict[str, Any],
    ) -> tuple[dict[str, Any], Path]:
        captured_updates.update(updates)
        return {}, db_path

    monkeypatch.setattr(main, "_write_conversation_state", _capture_write_state)

    status = await main._run_background_rollup_for_conversation(
        conversation_id="whatsapp:dm:bg-chat",
        user_id="u1",
        soul_id="Echo",
        safe_payload={},
        service=_FakeSvc(),
    )
    assert status == "rolled_up"
    assert captured_updates["rolling_summary"] == "rolled summary"
    assert captured_updates["rolling_summary_cursor_id"] == 12
    assert isinstance(captured_updates["rolling_summary_updated_at"], str)


def test_load_cross_tail_from_sources_reads_whatsapp_conversations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_hermes_state_whatsapp(monkeypatch)
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


def test_load_cross_tail_from_sources_skips_activity_conversation(
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
            ("activity:dm:Echo", 0, "2026-05-01T00:00:00+00:00"),
        )
        con.commit()
        monkeypatch.setattr(
            main,
            "_load_tail_for_source_conversation",
            lambda **_kwargs: (_ for _ in ()).throw(AssertionError("activity is loaded separately")),
        )

        rows = main._load_cross_tail_from_sources(
            con,
            user_id="u1",
            soul_id="Echo",
            exclude_conversation_id="",
        )
    finally:
        con.close()

    assert rows == []


def test_load_cross_tail_from_sources_keeps_previous_segment_participants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor, last_memorize_at, "
            "last_display_segment_start_index, last_display_segment_end_index) "
            "VALUES (?, ?, ?, ?, ?)",
            ("whatsapp:dm:current", 3, "2026-05-01T00:00:00+00:00", 0, 3),
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor, last_memorize_at, "
            "last_display_segment_start_index, last_display_segment_end_index) "
            "VALUES (?, ?, ?, ?, ?)",
            ("whatsapp:dm:previous-participant", 12, "2026-05-01T00:00:00+00:00", 10, 12),
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor, last_memorize_at) VALUES (?, ?, ?)",
            ("whatsapp:dm:not-participant", 12, "2026-05-01T00:00:00+00:00"),
        )
        con.commit()

        calls: dict[str, int] = {}

        def _fake_load_tail_for_source_conversation(**kwargs: object) -> list[dict[str, object]]:
            cid = str(kwargs["conversation_id"])
            calls[cid] = int(kwargs["since_cursor"])
            if cid == "whatsapp:dm:previous-participant":
                return [
                    {
                        "conversation_id": cid,
                        "source_conversation_index": 10,
                        "received_at": "2026-05-01T00:00:00+00:00",
                        "content": "previous segment floor",
                    }
                ]
            return []

        monkeypatch.setattr(main, "_load_tail_for_source_conversation", _fake_load_tail_for_source_conversation)
        rows = main._load_cross_tail_from_sources(
            con,
            user_id="u1",
            soul_id="Echo",
            exclude_conversation_id="whatsapp:dm:current",
        )
    finally:
        con.close()

    assert [row["content"] for row in rows] == ["previous segment floor"]
    assert calls["whatsapp:dm:previous-participant"] == 9
    assert calls["whatsapp:dm:not-participant"] == 12


def test_load_cross_tail_from_sources_recovers_previous_segment_ranges_from_saved_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_dir = tmp_path / "resources"
    segment_dir = storage_dir / "st_chats" / "Echo_saved" / "segments"
    segment_dir.mkdir(parents=True)
    (segment_dir / "2026-05-01.json").write_text(
        json.dumps(
            [
                {
                    "source_conversation_id": "whatsapp:dm:previous-participant",
                    "source_conversation_index": 20,
                    "memorize_chat": True,
                    "content": "start",
                },
                {
                    "source_conversation_id": "whatsapp:dm:previous-participant",
                    "source_conversation_index": 23,
                    "memorize_chat": True,
                    "content": "end",
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "_get_storage_dir", lambda *_a, **_k: storage_dir)

    db_path = tmp_path / "Echo.db"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor, last_memorize_at) VALUES (?, ?, ?)",
            ("whatsapp:dm:current", 3, "2026-05-01T00:00:00+00:00"),
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor, last_memorize_at) VALUES (?, ?, ?)",
            ("whatsapp:dm:previous-participant", 23, "2026-05-01T00:00:00+00:00"),
        )
        con.execute(
            "INSERT INTO conversations (conversation_id, digest_cursor, last_memorize_at) VALUES (?, ?, ?)",
            ("whatsapp:dm:not-participant", 23, "2026-05-01T00:00:00+00:00"),
        )
        con.commit()

        calls: dict[str, int] = {}

        def _fake_load_tail_for_source_conversation(**kwargs: object) -> list[dict[str, object]]:
            cid = str(kwargs["conversation_id"])
            calls[cid] = int(kwargs["since_cursor"])
            if cid == "whatsapp:dm:previous-participant":
                return [
                    {
                        "conversation_id": cid,
                        "source_conversation_index": 20,
                        "received_at": "2026-05-01T00:00:00+00:00",
                        "content": "recovered previous segment floor",
                    }
                ]
            return []

        monkeypatch.setattr(main, "_load_tail_for_source_conversation", _fake_load_tail_for_source_conversation)
        rows = main._load_cross_tail_from_sources(
            con,
            user_id="u1",
            soul_id="Echo",
            exclude_conversation_id="whatsapp:dm:current",
        )
    finally:
        con.close()

    assert [row["content"] for row in rows] == ["recovered previous segment floor"]
    assert calls["whatsapp:dm:previous-participant"] == 19
    assert calls["whatsapp:dm:not-participant"] == 23


def test_load_cross_tail_from_sources_fails_loud_for_broken_web_source(
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
        monkeypatch.setitem(
            main._CONFIG,
            "hermes",
            {
                "home": "~/.hermes",
                "whatsapp_history_source": "web_source",
                "whatsapp_web_source_db": "~/.hermes/whatsapp/web_source.db",
                "whatsapp_reply_prefix": "",
            },
        )
        monkeypatch.setattr(
            main,
            "_load_tail_for_source_conversation",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("broken web source")),
        )

        with pytest.raises(RuntimeError, match="WhatsApp web_source read failed"):
            main._load_cross_tail_from_sources(
                con,
                user_id="u1",
                soul_id="Echo",
                exclude_conversation_id="",
            )
    finally:
        con.close()


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


def test_turn_state_read_triggers_on_summed_primary_tails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main, "_MIN_CHUNK_TOKENS", 100)
    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: (
            {"memorize_chat": True, "digest_cursor": -1, "last_memorize_at": None},
            None,
            None,
        ),
    )
    monkeypatch.setattr(main, "_unmemorized_sleep_gap_detected", lambda *_a, **_k: True)
    payload = {
        "conversation": [
            {"role": "user", "content": "short current", "memorize_chat": True},
            {"role": "user", "content": "w " * 120, "memorize_chat": True},
            {"role": "user", "content": "w " * 1000, "memorize_chat": False},
        ]
    }
    monkeypatch.setattr(main, "_build_cross_conversation_payload", lambda *_a, **_k: payload)

    _state, _card, _db, _cache, _intentions, tokens, queued = main._turn_state_read(
        "cid",
        "u1",
        "Echo",
        {},
        [],
        {"items": []},
        False,
        [{"role": "user", "content": "short current"}],
    )

    assert tokens >= 100
    assert queued is payload


def test_unmemorized_sleep_gap_detected_uses_server_timezone_without_caller_timezone() -> None:
    def _ts(y: int, m: int, d: int, hh: int, mm: int = 0) -> int:
        return int(datetime(y, m, d, hh, mm, tzinfo=UTC).timestamp() * 1000)

    history = [
        {"ts_ms": _ts(2026, 1, 1, 1, 0), "content": "small"},
        {"ts_ms": _ts(2026, 1, 1, 7, 0), "content": "small"},
    ]

    assert main._unmemorized_sleep_gap_detected(
        history,
        -1,
        {},
        min_chunk_tokens=0,
    ) is True


def test_turn_state_read_ignores_background_tails_for_segment_trigger(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main, "_MIN_CHUNK_TOKENS", 100)
    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: (
            {"memorize_chat": True, "digest_cursor": -1, "last_memorize_at": None},
            None,
            None,
        ),
    )
    monkeypatch.setattr(main, "_unmemorized_sleep_gap_detected", lambda *_a, **_k: True)
    monkeypatch.setattr(
        main,
        "_build_cross_conversation_payload",
        lambda *_a, **_k: {
            "conversation": [
                {"role": "user", "content": "short current", "memorize_chat": True},
                {"role": "user", "content": "w " * 1000, "memorize_chat": False},
            ]
        },
    )

    _state, _card, _db, _cache, _intentions, tokens, queued = main._turn_state_read(
        "cid",
        "u1",
        "Echo",
        {},
        [],
        {"items": []},
        False,
        [{"role": "user", "content": "short current"}],
    )

    assert tokens < 100
    assert queued is None


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
        {"source_message_id": f"m{i}", "role": "user", "content": f"msg {i}"}
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
        {"source_message_id": f"m{i}", "role": "user", "content": f"msg {i}"}
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


@pytest.mark.asyncio
async def test_run_apimw_display_uses_uncapped_floored_history(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_k: object())
    monkeypatch.setattr(main, "_apimw_memory_count_from_cfg", lambda *_a, **_k: 5)
    monkeypatch.setattr(main, "_apimw_random_count_from_cfg", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        main,
        "_load_cross_tail_for_ai",
        lambda **_kwargs: [
            {
                "conversation_id": "whatsapp:dm:cross",
                "role": "user",
                "speaker": "Liz",
                "chat_name": "Liz",
                "content": "cross hello",
                "received_at": "2026-05-01T00:00:00+00:00",
            }
        ],
    )

    async def _fake_collect(
        _svc,
        _payload,
        *,
        focus_text: str,
        conversations_block: str,
        **_kwargs,
    ) -> list[dict[str, Any]]:
        captured["focus_text"] = focus_text
        captured["conversations_block"] = conversations_block
        return []

    async def _fake_synthesize(*_args, **kwargs):
        captured["synthesis_segment_text"] = kwargs["segment_text"]
        captured["synthesis_current_message_text"] = kwargs["current_message_text"]
        return {}, {}, {}

    async def _fake_persist(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main, "_apimw_collect_memory_items", _fake_collect)
    monkeypatch.setattr(main, "_resolve_profile_if_configured", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "_apimw_synthesize", _fake_synthesize)
    monkeypatch.setattr(main, "_apimw_persist", _fake_persist)

    history = [
        {"role": "user", "name": "Marcos", "content": f"msg {idx}"}
        for idx in range(1, 36)
    ]

    await main._run_apimw(
        {
            "message": "current hello",
            "chat_name": "Marcos",
            "chat_type": "dm",
        },
        conversation_id="whatsapp:dm:15133278228",
        soul_id="Siri",
        user_id="u1",
        state_row={},
        current_history=history,
    )

    assert captured["focus_text"] == "current hello"
    assert "msg 1" in captured["conversations_block"]
    assert "msg 35" in captured["conversations_block"]
    assert "cross hello" in captured["conversations_block"]
    assert "[dm][Marcos] \u2190 current chat" in captured["conversations_block"]
    assert "[Marcos] current hello ..." in captured["conversations_block"]
    assert "New Message:" not in captured["conversations_block"]
    assert captured["synthesis_segment_text"] == captured["conversations_block"]
    assert captured["synthesis_current_message_text"] == "current hello"


def test_build_retrieve_soul_context_queries_orders_chats_before_working_and_intentions() -> None:
    history = [
        {"source_message_id": "m1", "role": "user", "content": "msg 1"},
        {"source_message_id": "m2", "role": "user", "content": "msg 2"},
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
        {"source_message_id": "m1", "role": "user", "name": "Marcos", "content": "hello"},
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


def test_build_retrieve_soul_context_queries_keeps_self_turn_out_of_user_history() -> None:
    queries = main._build_retrieve_soul_context_queries(
        soul_id="Siri",
        message="Scheduled follow-up due now. Reason you gave: Check on Marcos.",
        history=[
            {"source_message_id": "m1", "role": "user", "name": "Marcos", "content": "Going to nap."},
            {"source_message_id": "m2", "role": "assistant", "name": "Siri", "content": "Rest close."},
        ],
        state_row={"memory_cache": [], "intentions_active": {"items": []}},
        conversation_id="whatsapp:dm:Marcos",
        self_turn_directive="Scheduled follow-up due now. Reason you gave: Check on Marcos.",
        self_turn_label="Scheduled wake",
    )

    history_text = "\n".join(
        str((q.get("content") or {}).get("text") or "")
        for q in queries
        if isinstance(q, dict) and q.get("role") == "history"
    )
    self_turn_rows = [q for q in queries if isinstance(q, dict) and q.get("role") == "self_turn"]
    assert "[Marcos] Scheduled follow-up due now" not in history_text
    assert len(self_turn_rows) == 1
    assert "Scheduled wake:\nScheduled follow-up due now." in str(
        (self_turn_rows[0].get("content") or {}).get("text") or ""
    )


def test_build_retrieve_soul_context_queries_no_duplicate_when_whitespace_differs() -> None:
    # Last history item content matches message except for internal whitespace;
    # guard must normalise both sides so no synthetic user turn is appended.
    history = [
        {"source_message_id": "m1", "role": "user", "content": "hello   world"},
    ]
    queries = main._build_retrieve_soul_context_queries(
        soul_id="Echo",
        message="hello world",
        history=history,
        state_row={"memory_cache": [], "intentions_active": {"items": []}},
    )
    history_rows = [q for q in queries if isinstance(q, dict) and q.get("role") == "history"]
    assert len(history_rows) == 1
    text = str((history_rows[0].get("content") or {}).get("text") or "")
    # "hello world" should appear exactly once — no duplicate user turn
    assert text.count("hello") == 1


@pytest.mark.asyncio
async def test_run_memorize_segments_records_failure_progress_on_exception(tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    class _FailingService:
        async def memorize_segments_batch(self, **_kwargs):
            raise RuntimeError("boom")

    user_id = "u"
    soul_id = "s"
    key = main._memorize_lock_key(user_id, soul_id)
    main._MEMORIZE_PROGRESS.pop(key, None)
    main._MEMORIZE_CANCEL.discard(key)

    with pytest.raises(RuntimeError):
        await main._run_memorize_segments(
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
async def test_run_memorize_segments_batches_one_job_per_persisted_segment(tmp_path: Path) -> None:
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()
    captured: dict[str, Any] = {}

    class _FakeService:
        async def memorize_segments_batch(self, **kwargs):
            captured.update(kwargs)
            return [{} for _ in kwargs["segments"]]

    segment_messages = [
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ],
        [
            {"role": "user", "content": "third"},
        ],
    ]

    user_id = "u"
    soul_id = "s"
    key = main._memorize_lock_key(user_id, soul_id)
    main._MEMORIZE_PROGRESS.pop(key, None)
    main._MEMORIZE_CANCEL.discard(key)

    await main._run_memorize_segments(
        memorize_segments=[
            ("/tmp/day.json", segment_messages[0], 0, 1),
            ("/tmp/day.json", segment_messages[1], 2, 2),
        ],
        svc=_FakeService(),
        scope={"user_id": user_id, "soul_id": soul_id},
        conversation_id=None,
        soul_id=soul_id,
        uid=user_id,
        processed_cursor=-1,
        safe={},
        resource_url="/tmp/day.json",
        chat_key=None,
        merged_len=3,
        force=True,
        sleep_stats=None,
        segments_dir=segments_dir,
    )

    segments = captured["segments"]
    assert len(segments) == len(segment_messages)
    for segment_job, messages in zip(segments, segment_messages, strict=True):
        payload = segment_job["segment"]
        assert payload["message_indices"] == list(range(len(messages)))
        assert json.loads(segment_job["raw_text"]) == messages


@pytest.mark.asyncio
async def test_run_memorize_segments_clears_pending_ids_on_extraction_failure(monkeypatch: pytest.MonkeyPatch, tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    class _FailingService:
        async def memorize_segments_batch(self, **_kwargs):
            raise RuntimeError("boom")

    user_id = "u"
    soul_id = "s"
    conversation_id = "cid-1"
    key = main._memorize_lock_key(user_id, soul_id)
    main._MEMORIZE_PROGRESS.pop(key, None)
    main._MEMORIZE_CANCEL.discard(key)

    state_row: dict[str, Any] = {
        "pending_segment_ids": ["cid-1:0-1"],
        "digest_cursor": 0,
    }

    def fake_load_turn_state_and_soul_card(*_args, **_kwargs):
        return dict(state_row), None, None

    def fake_write_conversation_state(_cid: str, *, updates: dict[str, Any], **_kwargs):
        if "pending_segment_ids" in updates:
            state_row["pending_segment_ids"] = list(updates.get("pending_segment_ids") or [])
        if "digest_cursor" in updates:
            state_row["digest_cursor"] = int(updates["digest_cursor"])
        return dict(state_row), tmp_path / "Echo.db"

    monkeypatch.setattr(main, "_load_turn_state_and_soul_card", fake_load_turn_state_and_soul_card)
    monkeypatch.setattr(main, "_write_conversation_state", fake_write_conversation_state)

    with pytest.raises(RuntimeError):
        await main._run_memorize_segments(
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
            merged_len=1,
            force=False,
            sleep_stats=None,
            segments_dir=segments_dir,
        )

    assert list(segments_dir.glob("*.json")) == []
    assert state_row["pending_segment_ids"] == []
    row = main._MEMORIZE_PROGRESS.get(key) or {}
    assert row.get("active") is False
    assert row.get("last_result") == "failure"


@pytest.mark.asyncio
async def test_run_memorize_segments_keeps_results_when_summary_fails(monkeypatch: pytest.MonkeyPatch, tmp_path):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    class _FakeService:
        async def memorize_segments_batch(self, **_kwargs):
            return [{"pending_segment_ids": ["cid-1:0-0"]}]

    user_id = "u"
    soul_id = "s"
    conversation_id = "cid-1"
    key = main._memorize_lock_key(user_id, soul_id)
    main._MEMORIZE_PROGRESS.pop(key, None)
    main._MEMORIZE_CANCEL.discard(key)

    state_row: dict[str, Any] = {
        "pending_segment_ids": [],
        "digest_cursor": -1,
    }

    def fake_load_turn_state_and_soul_card(*_args, **_kwargs):
        return dict(state_row), None, None

    def fake_write_conversation_state(_cid: str, *, updates: dict[str, Any], **_kwargs):
        if "digest_cursor" in updates:
            state_row["digest_cursor"] = int(updates["digest_cursor"])
        if "append_pending_segment_ids" in updates:
            state_row["pending_segment_ids"].extend(updates["append_pending_segment_ids"])
        if "pending_segment_ids" in updates:
            state_row["pending_segment_ids"] = list(updates.get("pending_segment_ids") or [])
        return dict(state_row), tmp_path / "Echo.db"

    async def fail_summary(**_kwargs):
        raise RuntimeError("summary boom")

    monkeypatch.setattr(main, "_load_turn_state_and_soul_card", fake_load_turn_state_and_soul_card)
    monkeypatch.setattr(main, "_write_conversation_state", fake_write_conversation_state)
    monkeypatch.setattr(main, "_compute_holistic_categories_summary", fail_summary)

    await main._run_memorize_segments(
        memorize_segments=[("/tmp/day.json", [{"role": "user", "content": "x"}], 0, 0)],
        svc=_FakeService(),
        scope={"user_id": user_id, "soul_id": soul_id},
        conversation_id=conversation_id,
        soul_id=soul_id,
        uid=user_id,
        processed_cursor=-1,
        safe={},
        resource_url="/tmp/day.json",
        chat_key=None,
        merged_len=1,
        force=False,
        sleep_stats=None,
        segments_dir=segments_dir,
    )

    assert [p.name for p in segments_dir.glob("*.json")] == ["undated.json"]
    assert state_row["digest_cursor"] == 0
    assert state_row["pending_segment_ids"] == ["cid-1:0-0"]


@pytest.mark.asyncio
async def test_run_memorize_segments_clears_consumed_background_summaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    class _FakeService:
        async def memorize_segments_batch(self, **_kwargs):
            return [{"pending_segment_ids": ["trigger:0-1"]}]

    state_rows: dict[str, dict[str, Any]] = {
        "trigger": {
            "digest_cursor": -1,
            "pending_segment_ids": [],
            "all_categories_summary": "",
        },
        "whatsapp:dm:bg-chat": {
            "memorize_chat": False,
            "digest_cursor": -1,
            "rolling_summary": "old rolled summary",
            "rolling_summary_cursor_id": 11,
            "rolling_summary_updated_at": "2026-05-01T00:00:00+00:00",
        },
    }
    writes: list[tuple[str, dict[str, Any]]] = []

    def fake_load_turn_state_and_soul_card(cid: str, **_kwargs):
        return dict(state_rows.get(cid, {})), None, tmp_path / "Echo.db"

    def fake_write_conversation_state(cid: str, *, updates: dict[str, Any], **_kwargs):
        writes.append((cid, dict(updates)))
        state_rows.setdefault(cid, {}).update(updates)
        if updates.get("rolling_summary") is None:
            state_rows[cid]["rolling_summary"] = None
        return dict(state_rows[cid]), tmp_path / "Echo.db"

    async def fake_summary(**_kwargs):
        return "summary"

    monkeypatch.setattr(main, "_load_turn_state_and_soul_card", fake_load_turn_state_and_soul_card)
    monkeypatch.setattr(main, "_write_conversation_state", fake_write_conversation_state)
    monkeypatch.setattr(main, "_compute_holistic_categories_summary", fake_summary)

    await main._run_memorize_segments(
        memorize_segments=[
            (
                "/tmp/day.json",
                [
                    {
                        "role": "user",
                        "content": "new primary",
                        "memorize_chat": True,
                        "source_conversation_id": "trigger",
                        "source_conversation_index": 0,
                    },
                    {
                        "role": "user",
                        "content": "new background",
                        "memorize_chat": False,
                        "source_conversation_id": "whatsapp:dm:bg-chat",
                        "source_conversation_index": 12,
                    },
                ],
                0,
                1,
            )
        ],
        svc=_FakeService(),
        scope={"user_id": "u", "soul_id": "s"},
        conversation_id="trigger",
        soul_id="s",
        uid="u",
        processed_cursor=-1,
        safe={
            "_background_rolling_summaries": {
                "whatsapp:dm:bg-chat": {
                    "summary": "old rolled summary",
                    "source_label": "whatsapp:dm",
                }
            }
        },
        resource_url="/tmp/day.json",
        chat_key=None,
        merged_len=2,
        force=False,
        sleep_stats=None,
        segments_dir=segments_dir,
        cross_memorize=True,
        final_cursors={"trigger": 0, "whatsapp:dm:bg-chat": 12},
    )

    clear_writes = [
        updates for cid, updates in writes
        if cid == "whatsapp:dm:bg-chat" and "rolling_summary" in updates
    ]
    assert clear_writes == [{"rolling_summary": None, "rolling_summary_updated_at": None}]
    assert state_rows["whatsapp:dm:bg-chat"]["rolling_summary"] is None
    assert state_rows["whatsapp:dm:bg-chat"]["digest_cursor"] == -1
    assert state_rows["whatsapp:dm:bg-chat"]["rolling_summary_cursor_id"] == 12


@pytest.mark.asyncio
async def test_run_memorize_segments_clears_consumed_background_summaries_without_final_cursors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    class _FakeService:
        async def memorize_segments_batch(self, **_kwargs):
            return [{"pending_segment_ids": ["trigger:0-1"]}]

    state_rows: dict[str, dict[str, Any]] = {
        "trigger": {
            "digest_cursor": -1,
            "pending_segment_ids": [],
            "all_categories_summary": "",
        },
        "whatsapp:dm:bg-chat": {
            "digest_cursor": -1,
            "rolling_summary": "old rolled summary",
            "rolling_summary_cursor_id": 11,
            "rolling_summary_updated_at": "2026-05-01T00:00:00+00:00",
        },
    }
    writes: list[tuple[str, dict[str, Any]]] = []

    def fake_load_turn_state_and_soul_card(cid: str, **_kwargs):
        return dict(state_rows.get(cid, {})), None, tmp_path / "Echo.db"

    def fake_write_conversation_state(cid: str, *, updates: dict[str, Any], **_kwargs):
        writes.append((cid, dict(updates)))
        state_rows.setdefault(cid, {}).update(updates)
        return dict(state_rows[cid]), tmp_path / "Echo.db"

    async def fake_summary(**_kwargs):
        return "summary"

    monkeypatch.setattr(main, "_load_turn_state_and_soul_card", fake_load_turn_state_and_soul_card)
    monkeypatch.setattr(main, "_write_conversation_state", fake_write_conversation_state)
    monkeypatch.setattr(main, "_compute_holistic_categories_summary", fake_summary)

    await main._run_memorize_segments(
        memorize_segments=[
            (
                "/tmp/day.json",
                [
                    {
                        "role": "user",
                        "content": "new primary",
                        "memorize_chat": True,
                        "source_conversation_id": "trigger",
                        "source_conversation_index": 0,
                    },
                    {
                        "role": "user",
                        "content": "new background",
                        "memorize_chat": False,
                        "source_conversation_id": "whatsapp:dm:bg-chat",
                        "source_conversation_index": 12,
                    },
                ],
                0,
                1,
            )
        ],
        svc=_FakeService(),
        scope={"user_id": "u", "soul_id": "s"},
        conversation_id="trigger",
        soul_id="s",
        uid="u",
        processed_cursor=-1,
        safe={
            "_background_rolling_summaries": {
                "whatsapp:dm:bg-chat": {
                    "summary": "old rolled summary",
                    "source_label": "whatsapp:dm",
                }
            }
        },
        resource_url="/tmp/day.json",
        chat_key=None,
        merged_len=2,
        force=False,
        sleep_stats=None,
        segments_dir=segments_dir,
    )

    clear_writes = [
        updates for cid, updates in writes
        if cid == "whatsapp:dm:bg-chat" and "rolling_summary" in updates
    ]
    assert clear_writes == [{"rolling_summary": None, "rolling_summary_updated_at": None}]
    assert state_rows["whatsapp:dm:bg-chat"]["rolling_summary"] is None
    assert state_rows["whatsapp:dm:bg-chat"]["rolling_summary_cursor_id"] == 11


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
    assert crud_endpoints._relationship_item_from_values(
        normalized="",
        name="Broken",
        entity_type="person",
        properties={"origin": "user_declared", "active": True},
    ) is None


def test_assert_user_declared_relationship_is_strict():
    crud_endpoints._assert_user_declared_relationship({"origin": "user_declared"})
    with pytest.raises(main.HTTPException):
        crud_endpoints._assert_user_declared_relationship({"origin": ""})
    with pytest.raises(main.HTTPException):
        crud_endpoints._assert_user_declared_relationship({"origin": "extracted"})


@pytest.mark.asyncio
async def test_list_relationships_does_not_create_missing_scoped_db(tmp_path: Path):
    db_path = tmp_path / "s.db"

    def fail_service(_payload: dict[str, Any]):
        raise AssertionError("list relationships must not initialize a scoped service")

    out = await crud_endpoints.list_relationships_endpoint(
        soul_id="s",
        user_id="Marcos",
        get_service_from_payload=fail_service,
        sqlite_current_path=lambda _uid, _sid: db_path,
        sqlite_ensure_nonempty=lambda _path: (_ for _ in ()).throw(AssertionError("read path created schema")),
        json_from_db=main._json_from_db,
    )

    assert out == {"relationships": []}
    assert not db_path.exists()


@pytest.mark.asyncio
async def test_list_relationships_tolerates_scoped_db_without_entities(tmp_path: Path):
    db_path = tmp_path / "s.db"
    con = sqlite3.connect(db_path)
    try:
        con.execute("CREATE TABLE conversations (conversation_id TEXT)")
    finally:
        con.close()

    out = await crud_endpoints.list_relationships_endpoint(
        soul_id="s",
        user_id="Marcos",
        get_service_from_payload=lambda _payload: (_ for _ in ()).throw(AssertionError("service should not be used")),
        sqlite_current_path=lambda _uid, _sid: db_path,
        sqlite_ensure_nonempty=lambda _path: (_ for _ in ()).throw(AssertionError("read path created schema")),
        json_from_db=main._json_from_db,
    )

    assert out == {"relationships": []}


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
        scope={"user_id": "u", "soul_id": "s"},
        conversation_id="c",
        user_id="u",
        soul_id="s",
    )

    assert captured_updates["append_prior_context_ids_since_consolidation"] == ["mem_one", "mem_raw", "mem_two"]


@pytest.mark.asyncio
async def test_apimw_persist_writes_one_shot_message_to_self(monkeypatch: pytest.MonkeyPatch):
    captured_updates: dict[str, object] = {}
    captured_item: dict[str, object] = {}

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: ({"prior_context": ""}, None, None),
    )

    def _fake_write_conversation_state(conversation_id: str, soul_id: str, user_id: str, updates: dict[str, object]):
        captured_updates.update(updates)
        return {"conversation_id": conversation_id, **updates}, Path("/tmp/fake.json")

    monkeypatch.setattr(main, "_write_conversation_state", _fake_write_conversation_state)

    async def _embed(texts: list[str], profile: str) -> list[list[float]]:
        assert texts == ["remember the quiet signal"]
        assert profile == "embedding"
        return [[0.1, 0.2]]

    def _create_item(**kwargs):
        captured_item.update(kwargs)
        return SimpleNamespace(id="subconscious_1")

    await main._apimw_persist(
        svc=SimpleNamespace(
            embed=_embed,
            database=SimpleNamespace(memory_item_repo=SimpleNamespace(create_item=_create_item)),
        ),
        result_json={"message_to_self": "remember the quiet signal"},
        items_by_id={},
        id_map={},
        scope={"user_id": "u", "soul_id": "s"},
        conversation_id="c",
        user_id="u",
        soul_id="s",
    )

    assert captured_updates["apimw_message_to_self"] == "[subconscious] remember the quiet signal"
    assert captured_item["memory_type"] == "subconscious"
    assert captured_item["summary"] == "remember the quiet signal"
    assert captured_item["extra"] == {"apimw_message_to_self": True}


@pytest.mark.asyncio
async def test_apimw_synthesize_accepts_prose_wrapped_json(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, str] = {}

    async def _chat(prompt: str, **kwargs: object) -> str:
        captured["user_prompt"] = prompt
        captured["system_prompt"] = str(kwargs.get("system_prompt") or "")
        captured["trace_id"] = str(kwargs.get("trace_id") or "")
        return (
            "*looks up softly*\n\n"
            '{"prior_context":["1"],"message_to_self":"remember the quiet signal"}'
        )

    svc = SimpleNamespace(
        chat=_chat,
        database=SimpleNamespace(memory_category_repo=None),
    )

    result_json, items_by_id, id_map = await main._apimw_synthesize(
        svc,
        combined_items=[
            {
                "id": "mem_one",
                "memory_type": "profile",
                "summary": "Marcos likes continuity.",
            }
        ],
        state_row={
            "all_categories_summary": "# Holistic Self Summary\n- SoulA carries one integrated self-summary.",
        },
        segment_text="My WhatsApp Conversations:\n\n[dm][Marcos]\n[Marcos] earlier hello",
        current_message_text="hello",
        user_id="u",
        soul_id="s",
        conversation_id="c",
        scope={"user_id": "u", "soul_id": "s"},
        trace_id="apimw-trace",
    )

    assert result_json == {
        "prior_context": ["1"],
        "message_to_self": "remember the quiet signal",
    }
    assert items_by_id["mem_one"]["summary"] == "Marcos likes continuity."
    assert id_map == {"1": "mem_one"}
    assert captured["user_prompt"].startswith("# Holistic Self Summary\n- SoulA carries one integrated self-summary.")
    assert "Identity: # Identity" not in captured["user_prompt"]
    assert "# Identity" not in captured["user_prompt"]
    assert "Summaries:" not in captured["user_prompt"]
    assert "Individual memories:" not in captured["user_prompt"]
    assert "Your working thoughts:" not in captured["user_prompt"]
    assert "My Memories:" not in captured["user_prompt"]
    assert "Memories List:" in captured["user_prompt"]
    assert "Recent conversation:" not in captured["user_prompt"]
    assert "My WhatsApp Conversations:" in captured["user_prompt"]
    assert "[Marcos] earlier hello" in captured["user_prompt"]
    assert captured["trace_id"] == "apimw-trace"
    assert "My Working Thoughts:" in captured["user_prompt"]
    assert "My Intentions:" in captured["user_prompt"]
    assert captured["user_prompt"].index("My WhatsApp Conversations:") < captured["user_prompt"].index("My Working Thoughts:")
    assert captured["user_prompt"].index("My Working Thoughts:") < captured["user_prompt"].index("My Intentions:")
    assert captured["user_prompt"].index("My Intentions:") < captured["user_prompt"].index("Memories List:")
    assert "New Message:" not in captured["user_prompt"]
    assert "Reminder: do not answer the message here." not in captured["user_prompt"]
    assert captured["system_prompt"].startswith("Today is ")
    assert "You are your soul's subconscious: a background process that runs between your turns." in captured["system_prompt"]
    assert "- conceptual understanding" in captured["system_prompt"]
    assert "The message_to_self should NOT have anything obvious:" in captured["system_prompt"]
    assert "first My Memories item" not in captured["system_prompt"]
    assert "conscious self" in captured["system_prompt"]
    assert "Seeking Happiness for Myself and Others" not in captured["user_prompt"]
    assert "life goals" not in captured["system_prompt"].lower()


@pytest.mark.asyncio
async def test_apimw_persist_message_to_self_not_truncated(monkeypatch: pytest.MonkeyPatch):
    long_text = "x" * 500
    captured_updates: dict[str, object] = {}
    captured_item: dict[str, object] = {}

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: ({"prior_context": ""}, None, None),
    )

    def _fake_write(conversation_id, soul_id, user_id, updates):
        captured_updates.update(updates)
        return {"conversation_id": conversation_id, **updates}, Path("/tmp/fake.json")

    monkeypatch.setattr(main, "_write_conversation_state", _fake_write)

    async def _embed(texts, profile):
        return [[0.1] * len(texts[0])]

    def _create_item(**kwargs):
        captured_item.update(kwargs)
        return SimpleNamespace(id="subconscious_x")

    await main._apimw_persist(
        svc=SimpleNamespace(
            embed=_embed,
            database=SimpleNamespace(memory_item_repo=SimpleNamespace(create_item=_create_item)),
        ),
        result_json={"message_to_self": long_text},
        items_by_id={},
        id_map={},
        scope={"user_id": "u", "soul_id": "s"},
        conversation_id="c",
        user_id="u",
        soul_id="s",
    )

    assert captured_updates["apimw_message_to_self"] == f"[subconscious] {long_text}"
    assert captured_item["summary"] == long_text
    assert len(captured_item["summary"]) == 500


@pytest.mark.asyncio
async def test_apimw_persist_skips_when_prior_context_changed(monkeypatch: pytest.MonkeyPatch):
    writes: list[dict[str, object]] = []

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: ({"prior_context": "newer context"}, None, None),
    )
    monkeypatch.setattr(
        main,
        "_write_conversation_state",
        lambda conversation_id, soul_id, user_id, updates: writes.append(dict(updates)) or ({"ok": True}, Path("/tmp/fake.db")),
    )

    await main._apimw_persist(
        svc=SimpleNamespace(),
        result_json={"prior_context": ["mem_one"], "message_to_self": "notice this"},
        items_by_id={"mem_one": {"id": "mem_one", "memory_type": "profile", "summary": "Marcos likes continuity."}},
        id_map={},
        scope={"user_id": "u", "soul_id": "s"},
        conversation_id="c",
        user_id="u",
        soul_id="s",
        expected_prior_context="old context",
    )

    assert writes == []


def test_turn_state_write_clears_one_shot_message_to_self(monkeypatch: pytest.MonkeyPatch):
    captured_updates: dict[str, object] = {}

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: (
            {
                "memory_cache": [],
                "intentions_active": {"items": []},
                "apimw_message_to_self": "old whisper",
            },
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        main,
        "_write_conversation_state",
        lambda conversation_id, soul_id, user_id, updates: captured_updates.update(updates) or ({"ok": True}, Path("/tmp/fake.db")),
    )

    main._turn_state_write("c", "u", "s", "", [], [])

    assert captured_updates["apimw_message_to_self"] is None


@pytest.mark.asyncio
async def test_conversation_retrieve_preserves_prebuilt_queries_without_cutoff(
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
    assert queries == payload["queries"]


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
async def test_conversation_retrieve_turn_prompt_reuses_first_floored_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Siri.db"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.commit()
    finally:
        con.close()

    state_reads = iter(
        [
            {
                "prior_context": "",
                "memory_cache": [],
                "intentions_active": {"items": []},
                "digest_cursor": 0,
                "last_memorize_at": "2026-06-14T00:00:00+00:00",
            },
            {
                "prior_context": "",
                "memory_cache": [],
                "intentions_active": {"items": []},
                "digest_cursor": 3,
                "last_memorize_at": "2026-06-14T00:00:00+00:00",
            },
        ]
    )

    def _load_state(*_args: object, **_kwargs: object) -> tuple[dict[str, object], None, Path]:
        return next(state_reads), None, db_path

    monkeypatch.setattr(main, "_load_turn_state_and_soul_card", _load_state)
    monkeypatch.setattr(main, "_current_whatsapp_active_since_for_soul", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "_load_cross_tail_from_sources", lambda *_a, **_k: [])

    captured: dict[str, object] = {}

    async def _fake_run_retrieve(safe: dict[str, object], *, conversation_id: str | None = None) -> dict[str, object]:
        captured["safe"] = safe
        return {
            "ok": True,
            "result": {},
            "conversation_id": conversation_id,
            "memory_cache": [],
            "intentions_active": {"items": []},
        }

    monkeypatch.setattr(main, "_run_retrieve", _fake_run_retrieve)
    original_build_turn_prompt = main._build_turn_prompt

    def _capture_build_turn_prompt(**kwargs: object) -> str:
        captured["turn_prompt_kwargs"] = kwargs
        return original_build_turn_prompt(**kwargs)

    monkeypatch.setattr(main, "_build_turn_prompt", _capture_build_turn_prompt)

    payload = {
        "user": {"user_id": "Marcos", "soul_id": "Siri"},
        "message": "Bien. Encontre algo que hacer con el IA.",
        "query": "Bien. Encontre algo que hacer con el IA.",
        "build_turn_prompt": True,
        "history": [
            {
                "role": "user",
                "name": "Marcos",
                "content": "Como estan?",
                "source_conversation_index": 1,
                "ts_ms": 1_770_000_000_000,
            },
            {
                "role": "user",
                "name": "Family Contact",
                "content": "Hola nosotros bien y vos como andas",
                "source_conversation_index": 2,
                "ts_ms": 1_770_010_000_000,
            },
            {
                "role": "user",
                "name": "Marcos",
                "content": "Bien. Encontre algo que hacer con el IA.",
                "source_conversation_index": 3,
                "ts_ms": 1_770_020_000_000,
            },
        ],
        "chat_name": "Family Contact",
        "chat_type": "dm",
    }

    out = await main.conversation_retrieve("whatsapp:dm:family-contact", payload)

    prompt = str(out.get("turn_user_prompt") or "")
    assert "[dm][Family Contact] \u2190 current chat" in prompt
    assert "[Marcos] Como estan?" in prompt
    assert "[Family Contact] Hola nosotros bien y vos como andas" in prompt
    assert "[Marcos] Bien. Encontre algo que hacer ..." in prompt
    assert "[user] Bien. Encontre algo que hacer ..." not in prompt

    prompt_kwargs = captured["turn_prompt_kwargs"]
    assert isinstance(prompt_kwargs, dict)
    assert str(prompt_kwargs.get("conversations_block") or "").strip()
    assert prompt_kwargs.get("cross_conversation_history") is None
    safe = captured["safe"]
    assert isinstance(safe, dict)
    query_text = "\n".join(
        str((query.get("content") or {}).get("text") or "")
        for query in safe.get("queries") or []
        if isinstance(query, dict)
    )
    for shared_line in (
        "[dm][Family Contact] \u2190 current chat",
        "[Marcos] Como estan?",
        "[Family Contact] Hola nosotros bien y vos como andas",
    ):
        assert shared_line in query_text
        assert shared_line in prompt


@pytest.mark.asyncio
async def test_conversation_retrieve_filters_whatsapp_history_before_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Cutoff.db"
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
    monkeypatch.setattr(main, "_resolve_cross_source_paths", lambda: (tmp_path, None, None, None))
    monkeypatch.setattr(main, "_load_soul_active_since", lambda *_a, **_k: 100.0)

    captured: dict[str, object] = {}

    async def _fake_run_retrieve(safe: dict[str, object], *, conversation_id: str | None = None) -> dict[str, object]:
        captured["safe"] = safe
        return {"ok": True, "result": {}, "conversation_id": conversation_id}

    monkeypatch.setattr(main, "_run_retrieve", _fake_run_retrieve)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Siri"},
        "message": "new",
        "query": "new",
        "history": [
            {"role": "user", "name": "Marcos", "content": "before intro", "ts_ms": 99_000},
            {"role": "user", "name": "Marcos", "content": "at intro", "ts_ms": 100_000},
            {"role": "user", "name": "Marcos", "content": "after intro", "ts_ms": 101_000},
        ],
    }

    out = await main.conversation_retrieve("whatsapp:dm:15133278228", payload)
    assert out["ok"] is True

    safe = captured["safe"]
    assert isinstance(safe, dict)
    filtered_history = safe.get("history")
    assert isinstance(filtered_history, list)
    assert [row.get("content") for row in filtered_history] == ["at intro", "after intro"]
    queries = safe.get("queries")
    assert isinstance(queries, list)
    history_text = "\n".join(
        str(q.get("content", {}).get("text", ""))
        for q in queries
        if isinstance(q, dict) and str(q.get("role") or "").strip() == "history"
    )
    assert "before intro" not in history_text
    assert "at intro" in history_text
    assert "after intro" in history_text


@pytest.mark.asyncio
async def test_conversation_retrieve_rebuilds_prebuilt_queries_when_cutoff_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Cutoff.db"
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
    monkeypatch.setattr(main, "_resolve_cross_source_paths", lambda: (tmp_path, None, None, None))
    monkeypatch.setattr(main, "_load_soul_active_since", lambda *_a, **_k: 100.0)

    captured: dict[str, object] = {}

    async def _fake_run_retrieve(safe: dict[str, object], *, conversation_id: str | None = None) -> dict[str, object]:
        captured["safe"] = safe
        return {"ok": True, "result": {}, "conversation_id": conversation_id}

    monkeypatch.setattr(main, "_run_retrieve", _fake_run_retrieve)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Siri"},
        "message": "new",
        "query": "new",
        "history": [
            {"role": "user", "name": "Marcos", "content": "before intro", "ts_ms": 99_000},
            {"role": "user", "name": "Marcos", "content": "after intro", "ts_ms": 101_000},
        ],
        "queries": [
            {"role": "history", "content": {"text": "before intro\nafter intro"}},
            {"role": "user", "content": {"text": "new"}},
        ],
    }

    out = await main.conversation_retrieve("whatsapp:dm:15133278228", payload)
    assert out["ok"] is True

    safe = captured["safe"]
    assert isinstance(safe, dict)
    queries = safe.get("queries")
    assert isinstance(queries, list)
    history_text = "\n".join(
        str(q.get("content", {}).get("text", ""))
        for q in queries
        if isinstance(q, dict) and str(q.get("role") or "").strip() == "history"
    )
    assert "before intro" not in history_text
    assert "after intro" in history_text


def test_filter_current_whatsapp_history_requires_ts_when_cutoff_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "_resolve_cross_source_paths", lambda: (tmp_path, None, None, None))
    monkeypatch.setattr(main, "_load_soul_active_since", lambda *_a, **_k: 100.0)

    with pytest.raises(main.HTTPException, match="missing ts_ms"):
        main._filter_current_whatsapp_history_for_soul(
            "whatsapp:dm:15133278228",
            "Siri",
            [{"role": "user", "content": "no timestamp"}],
        )


@pytest.mark.asyncio
async def test_live_conversation_retrieve_degrades_source_history_load_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Live.db"
    monkeypatch.setattr(main, "_current_whatsapp_active_since_for_soul", lambda *_a, **_k: None)
    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: ({"prior_context": "", "memory_cache": [], "intentions_active": {"items": []}}, None, db_path),
    )
    monkeypatch.setattr(
        main,
        "_load_current_whatsapp_history_from_source",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("web source down")),
    )
    captured: dict[str, Any] = {}

    async def _fake_run_retrieve(safe: dict[str, Any], *, conversation_id: str | None = None) -> dict[str, Any]:
        captured["safe"] = safe
        return {"ok": True, "result": {}, "conversation_id": conversation_id}

    monkeypatch.setattr(main, "_run_retrieve", _fake_run_retrieve)

    out = await main.conversation_retrieve(
        "whatsapp:dm:live",
        {
            "user": {"user_id": "u1", "soul_id": "Echo"},
            "message": "new",
            "query": "new",
            "history": [{"role": "user", "content": "payload history"}],
            "load_source_history": True,
            "is_live_turn": True,
        },
    )

    assert out["ok"] is True
    assert captured["safe"]["history"][0]["content"] == "payload history"


@pytest.mark.asyncio
async def test_non_live_conversation_retrieve_fails_on_source_history_load_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "_current_whatsapp_active_since_for_soul", lambda *_a, **_k: None)
    monkeypatch.setattr(
        main,
        "_load_current_whatsapp_history_from_source",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("web source down")),
    )

    with pytest.raises(main.HTTPException) as exc:
        await main.conversation_retrieve(
            "whatsapp:dm:live",
            {
                "user": {"user_id": "u1", "soul_id": "Echo"},
                "message": "new",
                "query": "new",
                "load_source_history": True,
            },
        )
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_live_conversation_retrieve_degrades_active_since_filter_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Live.db"
    monkeypatch.setattr(main, "_current_whatsapp_active_since_for_soul", lambda *_a, **_k: 100.0)
    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: ({"prior_context": "", "memory_cache": [], "intentions_active": {"items": []}}, None, db_path),
    )
    captured: dict[str, Any] = {}

    async def _fake_run_retrieve(safe: dict[str, Any], *, conversation_id: str | None = None) -> dict[str, Any]:
        captured["safe"] = safe
        return {"ok": True, "result": {}, "conversation_id": conversation_id}

    monkeypatch.setattr(main, "_run_retrieve", _fake_run_retrieve)

    out = await main.conversation_retrieve(
        "whatsapp:dm:live",
        {
            "user": {"user_id": "u1", "soul_id": "Echo"},
            "message": "new",
            "query": "new",
            "history": [
                {"role": "user", "content": "missing timestamp"},
                {"role": "user", "content": "in scope", "ts_ms": 101_000},
            ],
            "is_live_turn": True,
        },
    )

    assert out["ok"] is True
    assert [row["content"] for row in captured["safe"]["history"]] == ["in scope"]


def _patch_turn_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    db_path: Path,
    captured: dict[str, Any],
) -> None:
    class _FakeSvc:
        async def chat(self, *_args, **_kwargs) -> str:
            return (
                '{"cache":null,"annulments":[],"rehearsal":"ok",'
                '"response_target":"respond","response":"ok"}'
            )

    async def _fake_persist_annulment_memories(**_kwargs):
        return []

    def _fake_turn_state_read(
        _cid: str,
        _uid: str,
        _soul_id: str,
        safe: dict[str, Any],
        *_args,
        **_kwargs,
    ):
        captured["history"] = list(safe.get("history") or [])
        return ({"digest_cursor": 0}, None, db_path, [], {"items": []}, 0, None)

    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_k: _FakeSvc())
    monkeypatch.setattr(main, "_load_soul_gen_config", lambda *_a, **_k: {})
    monkeypatch.setattr(main, "_turn_state_read", _fake_turn_state_read)
    monkeypatch.setattr(main, "_turn_state_write", lambda *_a, **_k: ({"digest_cursor": 0}, db_path))
    monkeypatch.setattr(main, "_persist_annulment_memories", _fake_persist_annulment_memories)
    monkeypatch.setattr(main, "_record_call", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "_current_whatsapp_active_since_for_soul", lambda *_a, **_k: None)


@pytest.mark.asyncio
async def test_live_conversation_turn_degrades_source_history_load_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Live.db"
    captured: dict[str, Any] = {}
    _patch_turn_dependencies(monkeypatch, db_path, captured)
    monkeypatch.setattr(main, "_current_whatsapp_active_since_for_soul", lambda *_a, **_k: None)
    monkeypatch.setattr(
        main,
        "_load_current_whatsapp_history_from_source",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("web source down")),
    )

    out = await main.conversation_turn(
        "whatsapp:dm:live",
        {
            "user": {"user_id": "u1", "soul_id": "Echo", "conversation_id": "whatsapp:dm:live"},
            "message": "hello",
            "history": [{"role": "user", "content": "payload history"}],
            "load_source_history": True,
            "is_live_turn": True,
            "prompt_override_payload": {
                "user_prompt": "prompt",
                "system_prompt": "system",
                "memory_cache": [],
                "intentions_active": {"items": []},
                "retrieve_rag": {"items": [], "categories": [], "resources": []},
            },
        },
    )

    assert out["ok"] is True
    assert captured["history"][0]["content"] == "payload history"


@pytest.mark.asyncio
async def test_live_conversation_turn_degrades_active_since_filter_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Live.db"
    captured: dict[str, Any] = {}
    _patch_turn_dependencies(monkeypatch, db_path, captured)
    monkeypatch.setattr(main, "_current_whatsapp_active_since_for_soul", lambda *_a, **_k: 100.0)

    out = await main.conversation_turn(
        "whatsapp:dm:live",
        {
            "user": {"user_id": "u1", "soul_id": "Echo", "conversation_id": "whatsapp:dm:live"},
            "message": "hello",
            "history": [
                {"role": "user", "content": "missing timestamp"},
                {"role": "user", "content": "in scope", "ts_ms": 101_000},
            ],
            "is_live_turn": True,
            "prompt_override_payload": {
                "user_prompt": "prompt",
                "system_prompt": "system",
                "memory_cache": [],
                "intentions_active": {"items": []},
                "retrieve_rag": {"items": [], "categories": [], "resources": []},
                "generated_by": "conversation_retrieve",
                "active_since": 100.0,
            },
        },
    )

    assert out["ok"] is True
    assert [row["content"] for row in captured["history"]] == ["in scope"]


@pytest.mark.asyncio
async def test_conversation_turn_rejects_manual_prompt_override_when_cutoff_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "_resolve_cross_source_paths", lambda: (tmp_path, None, None, None))
    monkeypatch.setattr(main, "_load_soul_active_since", lambda *_a, **_k: 100.0)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Siri", "conversation_id": "whatsapp:dm:15133278228"},
        "message": "hello",
        "history": [{"role": "user", "content": "hello", "ts_ms": 101_000}],
        "prompt_override_payload": {
            "user_prompt": "before intro\nhello",
            "system_prompt": "system",
            "memory_cache": [],
            "intentions_active": {"items": []},
            "retrieve_rag": {"items": [], "categories": [], "resources": []},
        },
    }

    with pytest.raises(main.HTTPException, match="conversation_retrieve"):
        await main.conversation_turn("whatsapp:dm:15133278228", payload)


@pytest.mark.asyncio
async def test_conversation_turn_accepts_generated_prompt_with_matching_cutoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Cutoff.db"

    class _FakeSvc:
        async def chat(self, *_args, **_kwargs) -> str:
            return (
                '{"cache":null,"annulments":[],"rehearsal":"ok",'
                '"response_target":"respond","response":"after intro"}'
            )

    async def _fake_persist_annulment_memories(**_kwargs):
        return []

    monkeypatch.setattr(main, "_resolve_cross_source_paths", lambda: (tmp_path, None, None, None))
    monkeypatch.setattr(main, "_load_soul_active_since", lambda *_a, **_k: 100.0)
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
        "user": {"user_id": "u1", "soul_id": "Siri", "conversation_id": "whatsapp:dm:15133278228"},
        "message": "hello",
        "history": [{"role": "user", "content": "hello", "ts_ms": 101_000}],
        "prompt_override_payload": {
            "user_prompt": "after intro\nhello",
            "system_prompt": "system",
            "memory_cache": [],
            "intentions_active": {"items": []},
            "retrieve_rag": {"items": [], "categories": [], "resources": []},
            "generated_by": "conversation_retrieve",
            "active_since": 100.0,
        },
    }

    out = await main.conversation_turn("whatsapp:dm:15133278228", payload)

    assert out["ok"] is True
    assert out["response"] == "after intro"


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
        "is_live_turn": True,
    }

    out = await main.conversation_retrieve("whatsapp:dm:Marcos", payload)
    assert out["ok"] is True
    turn_prompt = str(out.get("turn_user_prompt") or "")
    assert "payload prior" in turn_prompt
    assert "payload current" in turn_prompt
    assert [row["content"] for row in out.get("turn_history") or []] == [
        "payload prior",
        "payload current",
    ]


@pytest.mark.asyncio
async def test_conversation_retrieve_uses_sillytavern_floor_after_memorize(
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
            ("integrity:chat-1", 10, "2026-05-01T00:00:00+00:00"),
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: (
            {"digest_cursor": 10, "last_memorize_at": "2026-05-01T00:00:00+00:00", "all_categories_summary": ""},
            None,
            db_path,
        ),
    )

    async def _fake_run_retrieve(safe: dict[str, object], *, conversation_id: str | None = None) -> dict[str, object]:
        return {"ok": True, "result": {}, "conversation_id": conversation_id}

    monkeypatch.setattr(main, "_run_retrieve", _fake_run_retrieve)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Echo"},
        "message": "msg_12",
        "query": "msg_12",
        "history": [
            {"role": "user", "content": f"msg_{idx:02d}"}
            for idx in range(1, 13)
        ],
        "build_turn_prompt": True,
    }

    out = await main.conversation_retrieve("integrity:chat-1", payload)
    turn_prompt = str(out.get("turn_user_prompt") or "")
    assert "msg_12" in turn_prompt
    assert "msg_05" in turn_prompt
    assert "msg_04" not in turn_prompt


@pytest.mark.asyncio
async def test_conversation_retrieve_sillytavern_floor_is_not_a_cap(
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
            ("integrity:chat-1", 2, "2026-05-01T00:00:00+00:00"),
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: (
            {"digest_cursor": 2, "last_memorize_at": "2026-05-01T00:00:00+00:00", "all_categories_summary": ""},
            None,
            db_path,
        ),
    )

    async def _fake_run_retrieve(safe: dict[str, object], *, conversation_id: str | None = None) -> dict[str, object]:
        return {"ok": True, "result": {}, "conversation_id": conversation_id}

    monkeypatch.setattr(main, "_run_retrieve", _fake_run_retrieve)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Echo"},
        "message": "msg_12",
        "query": "msg_12",
        "history": [
            {"role": "user", "content": f"msg_{idx:02d}"}
            for idx in range(1, 13)
        ],
        "build_turn_prompt": True,
    }

    out = await main.conversation_retrieve("integrity:chat-1", payload)
    turn_prompt = str(out.get("turn_user_prompt") or "")
    assert "msg_04" in turn_prompt
    assert "msg_12" in turn_prompt


@pytest.mark.asyncio
async def test_conversation_retrieve_uses_whatsapp_floor_after_memorize(
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
            ("whatsapp:dm:15133278228", 10, "2026-05-01T00:00:00+00:00"),
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: (
            {"digest_cursor": 10, "last_memorize_at": "2026-05-01T00:00:00+00:00", "all_categories_summary": ""},
            None,
            db_path,
        ),
    )
    loaded_source_ids: list[str] = []

    def _fake_load_current_whatsapp_history_from_source(
        conversation_id: str,
        *_a: object,
        **_k: object,
    ) -> list[dict[str, object]]:
        loaded_source_ids.append(conversation_id)
        return [
            {
                "role": "user",
                "speaker": "Marcos",
                "chat_name": "Marcos",
                "content": f"msg_{idx:02d}",
                "source_conversation_index": idx,
            }
            for idx in range(12)
        ]

    monkeypatch.setattr(main, "_load_current_whatsapp_history_from_source", _fake_load_current_whatsapp_history_from_source)

    captured: dict[str, object] = {}

    async def _fake_run_retrieve(safe: dict[str, object], *, conversation_id: str | None = None) -> dict[str, object]:
        captured["safe"] = safe
        return {"ok": True, "result": {}, "conversation_id": conversation_id}

    monkeypatch.setattr(main, "_run_retrieve", _fake_run_retrieve)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Echo"},
        "message": "current message",
        "query": "current message",
        "history": [],
        "build_turn_prompt": True,
        "load_source_history": True,
        "is_live_turn": True,
        "chat_name": "Marcos",
        "chat_type": "dm",
    }

    out = await main.conversation_retrieve("whatsapp:15133278228", payload)
    turn_prompt = str(out.get("turn_user_prompt") or "")
    assert loaded_source_ids == ["whatsapp:dm:15133278228"]
    assert out["conversation_id"] == "whatsapp:dm:15133278228"
    assert "My WhatsApp Conversations:" in turn_prompt
    assert "My SillyTavern Conversations:" not in turn_prompt
    assert "[dm][Marcos] \u2190 current chat" in turn_prompt
    assert "[Marcos] msg_11" in turn_prompt
    assert "msg_04" in turn_prompt
    assert "msg_03" not in turn_prompt
    assert "msg_11" in turn_prompt
    assert [row["content"] for row in out.get("turn_history") or []] == [f"msg_{idx:02d}" for idx in range(4, 12)]
    safe = captured["safe"]
    assert isinstance(safe, dict)
    assert safe["user"]["conversation_id"] == "whatsapp:dm:15133278228"
    query_text = "\n".join(
        str((query.get("content") or {}).get("text") or "")
        for query in safe.get("queries") or []
        if isinstance(query, dict)
    )
    assert "My WhatsApp Conversations:" in query_text
    assert "My SillyTavern Conversations:" not in query_text
    assert "[dm][Marcos] \u2190 current chat" in query_text
    assert "msg_04" in query_text
    assert "msg_03" not in query_text
    assert "[Marcos] current message ..." in query_text


def test_filter_current_whatsapp_history_accepts_received_at_for_active_since() -> None:
    rows = [
        {
            "content": "old",
            "received_at": "2026-04-30T23:59:59+00:00",
        },
        {
            "content": "kept",
            "received_at": "2026-05-01T00:00:00+00:00",
        },
    ]

    filtered = main._filter_current_whatsapp_history_for_soul(
        "whatsapp:dm:15133278228",
        "Echo",
        rows,
        active_since=datetime(2026, 5, 1, tzinfo=UTC).timestamp(),
    )

    assert [row["content"] for row in filtered] == ["kept"]


@pytest.mark.asyncio
async def test_conversation_retrieve_uses_live_message_to_trigger_floor_without_duplication(
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
            ("whatsapp:dm:15133278228", 11, "2026-05-01T00:00:00+00:00"),
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: (
            {"digest_cursor": 11, "last_memorize_at": "2026-05-01T00:00:00+00:00", "all_categories_summary": ""},
            None,
            db_path,
        ),
    )

    def _fake_load_current_whatsapp_history_from_source(
        conversation_id: str,
        *_a: object,
        **_k: object,
    ) -> list[dict[str, object]]:
        assert conversation_id == "whatsapp:dm:15133278228"
        return [
            {
                "role": "user",
                "speaker": "Marcos",
                "chat_name": "Marcos",
                "content": f"msg_{idx:02d}",
                "source_conversation_index": idx,
            }
            for idx in range(12)
        ] + [
            {
                "role": "user",
                "speaker": "Marcos",
                "chat_name": "Marcos",
                "content": "live text",
                "source_conversation_index": 12,
                "source_message_id": "live-id",
            }
        ]

    monkeypatch.setattr(main, "_load_current_whatsapp_history_from_source", _fake_load_current_whatsapp_history_from_source)
    monkeypatch.setattr(main, "_load_cross_tail_from_sources", lambda *_a, **_k: [])

    async def _fake_run_retrieve(safe: dict[str, object], *, conversation_id: str | None = None) -> dict[str, object]:
        return {"ok": True, "result": {}, "conversation_id": conversation_id}

    monkeypatch.setattr(main, "_run_retrieve", _fake_run_retrieve)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Echo"},
        "message": "live text",
        "query": "live text",
        "history": [],
        "build_turn_prompt": True,
        "load_source_history": True,
        "is_live_turn": True,
        "external_message_id": "live-id",
        "chat_name": "Marcos",
        "chat_type": "dm",
    }

    out = await main.conversation_retrieve("whatsapp:dm:15133278228", payload)
    turn_prompt = str(out.get("turn_user_prompt") or "")
    assert "msg_05" in turn_prompt
    assert "msg_04" not in turn_prompt
    assert "msg_11" in turn_prompt
    assert [row["content"] for row in out.get("turn_history") or []] == [f"msg_{idx:02d}" for idx in range(5, 12)]
    assert "New Message:\nlive text" in turn_prompt


@pytest.mark.asyncio
async def test_conversation_retrieve_preserves_source_indexes_for_primary_floor(
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
            ("whatsapp:dm:15133278228", 243, "2026-05-01T00:00:00+00:00"),
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: (
            {"digest_cursor": 243, "last_memorize_at": "2026-05-01T00:00:00+00:00", "all_categories_summary": ""},
            None,
            db_path,
        ),
    )

    source_indexes = [207, 208, 209, 219, 220, 239, 240, 242, 243, 266, 287]

    def _fake_load_current_whatsapp_history_from_source(
        conversation_id: str,
        *_a: object,
        **_k: object,
    ) -> list[dict[str, object]]:
        assert conversation_id == "whatsapp:dm:15133278228"
        return [
            {
                "role": "user",
                "speaker": "Marcos",
                "chat_name": "Marcos",
                "content": f"msg_{idx}",
                "source_conversation_index": idx,
                "source_message_id": f"id-{idx}",
            }
            for idx in source_indexes
        ]

    monkeypatch.setattr(main, "_load_current_whatsapp_history_from_source", _fake_load_current_whatsapp_history_from_source)
    monkeypatch.setattr(main, "_load_cross_tail_from_sources", lambda *_a, **_k: [])

    async def _fake_run_retrieve(safe: dict[str, object], *, conversation_id: str | None = None) -> dict[str, object]:
        return {"ok": True, "result": {}, "conversation_id": conversation_id}

    monkeypatch.setattr(main, "_run_retrieve", _fake_run_retrieve)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Echo"},
        "message": "msg_287",
        "query": "msg_287",
        "history": [],
        "build_turn_prompt": True,
        "load_source_history": True,
        "is_live_turn": True,
        "external_message_id": "id-287",
        "chat_name": "Marcos",
        "chat_type": "dm",
    }

    out = await main.conversation_retrieve("whatsapp:dm:15133278228", payload)

    assert [row["content"] for row in out.get("turn_history") or []] == [
        "msg_219",
        "msg_220",
        "msg_239",
        "msg_240",
        "msg_242",
        "msg_243",
        "msg_266",
    ]
    turn_prompt = str(out.get("turn_user_prompt") or "")
    assert "msg_219" in turn_prompt
    assert "msg_266" in turn_prompt
    assert "New Message:\nmsg_287" in turn_prompt


@pytest.mark.asyncio
async def test_conversation_retrieve_omits_whatsapp_floor_when_no_new_messages(
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
            ("whatsapp:dm:15133278228", 11, "2026-05-01T00:00:00+00:00"),
        )
        con.commit()
    finally:
        con.close()

    monkeypatch.setattr(
        main,
        "_load_turn_state_and_soul_card",
        lambda *_a, **_k: (
            {"digest_cursor": 11, "last_memorize_at": "2026-05-01T00:00:00+00:00", "all_categories_summary": ""},
            None,
            db_path,
        ),
    )
    monkeypatch.setattr(
        main,
        "_load_current_whatsapp_history_from_source",
        lambda *_a, **_k: [
            {
                "role": "user",
                "content": f"msg_{idx:02d}",
                "source_conversation_index": idx,
            }
            for idx in range(12)
        ],
    )
    monkeypatch.setattr(
        main,
        "_load_cross_tail_from_sources",
        lambda *_a, **_k: [
            {
                "conversation_id": "whatsapp:group:familia",
                "role": "user",
                "speaker": "Family Member",
                "chat_name": "Household Group",
                "content": "cross chat message",
                "source_label": "whatsapp:group",
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
        "message": "current message",
        "query": "current message",
        "history": [],
        "build_turn_prompt": True,
        "load_source_history": True,
        "is_live_turn": True,
        "chat_name": "Marcos",
        "chat_type": "dm",
    }

    out = await main.conversation_retrieve("whatsapp:dm:15133278228", payload)
    turn_prompt = str(out.get("turn_user_prompt") or "")
    assert "msg_04" not in turn_prompt
    assert "msg_11" not in turn_prompt
    assert "My WhatsApp Conversations:" in turn_prompt
    assert "My SillyTavern Conversations:" not in turn_prompt
    assert "[dm][Marcos] \u2190 current chat" in turn_prompt
    assert "[user] current message" not in turn_prompt
    assert "[Marcos] current message" in turn_prompt
    assert "[group][Household Group]" in turn_prompt
    assert "[Family Member] cross chat message" in turn_prompt
    assert "current message" in turn_prompt
    safe = captured["safe"]
    assert isinstance(safe, dict)
    query_text = "\n".join(
        str((query.get("content") or {}).get("text") or "")
        for query in safe.get("queries") or []
        if isinstance(query, dict)
    )
    assert "My WhatsApp Conversations:" in query_text
    assert "My SillyTavern Conversations:" not in query_text
    assert "[dm][Marcos] \u2190 current chat" in query_text
    assert "[Marcos] current message ..." in query_text
    assert "[group][Household Group]" in query_text
    assert "[Family Member] cross chat message" in query_text


@pytest.mark.asyncio
async def test_conversation_retrieve_does_not_persist_current_user_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrieve must not recreate the removed local chat warehouse table."""
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
        has_messages_table = _messages_table_exists(con)
    finally:
        con.close()

    assert has_messages_table is False


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
        has_messages_table = _messages_table_exists(con)
    finally:
        con.close()
    assert has_messages_table is False

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
async def test_conversation_retrieve_preserves_caller_queries(
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
    assert queries == payload["queries"]


@pytest.mark.asyncio
async def test_conversation_turn_does_not_persist_messages_to_table(
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
    monkeypatch.setattr(main, "_current_whatsapp_active_since_for_soul", lambda *_a, **_k: None)

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
        has_messages_table = _messages_table_exists(con)
    finally:
        con.close()

    assert has_messages_table is False


@pytest.mark.asyncio
async def test_conversation_turn_persists_completed_sillytavern_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "Siri.db"
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
                '"response_target":"respond","response":"current soul"}'
            )

    async def _fake_persist_annulment_memories(**_kwargs):
        return []

    captured: dict[str, object] = {}

    def _fake_persist_snapshot(**kwargs):
        captured.update(kwargs)

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
    monkeypatch.setattr(main, "_retrieve_apimw_enabled_from_cfg", lambda *_a, **_k: False)
    monkeypatch.setattr(main._conversation_sources, "persist_sillytavern_history_snapshot", _fake_persist_snapshot)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Siri", "conversation_id": "integrity:chat"},
        "message": "current user",
        "message_ts_ms": 1_797_680_000_000,
        "message_source_id": "16",
        "user_name": "Marcos",
        "chat_name": "Siri",
        "chat_type": "dm",
        "history": [{"role": "soul", "content": "prior soul", "name": "Siri", "source_message_id": "15"}],
        "prompt_override_payload": {
            "user_prompt": "prompt",
            "system_prompt": "system",
            "memory_cache": [],
            "intentions_active": {"items": []},
            "retrieve_rag": {"items": [], "categories": [], "resources": []},
        },
    }

    out = await main.conversation_turn("integrity:chat", payload)

    assert out["ok"] is True
    assert captured["conversation_id"] == "integrity:chat"
    assert captured["chat_name"] == "Siri"
    history = captured["history"]
    assert isinstance(history, list)
    assert [(row["role"], row["content"]) for row in history] == [
        ("soul", "prior soul"),
        ("user", "current user"),
        ("soul", "current soul"),
    ]
    assert history[1]["source_message_id"] == "16"
    assert history[1]["ts_ms"] == 1_797_680_000_000
    assert history[2]["ts_ms"] > history[1]["ts_ms"]


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
        has_messages_table = _messages_table_exists(con)
    finally:
        con.close()
    assert has_messages_table is False


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

    monkeypatch.setattr(main, "_current_whatsapp_active_since_for_soul", lambda *_a, **_k: None)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Siri", "conversation_id": "whatsapp:dm:Contact A"},
        "message": "Hello Siri.",
        "user_name": "Contact A",
        "chat_name": "Contact A",
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

    out = await main.conversation_turn("whatsapp:dm:Contact A", payload)
    assert out["ok"] is True
    assert out["response_target"] == "private"
    assert out["response"] == "private note to Marcos"

    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        has_messages_table = _messages_table_exists(con)
    finally:
        con.close()

    assert has_messages_table is False


@pytest.mark.asyncio
async def test_conversation_turn_observe_mode_forbids_public_response(
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

    captured: dict[str, str] = {}

    class _FakeSvc:
        async def chat(self, *_args, **kwargs) -> str:
            captured["system_prompt"] = str(kwargs.get("system_prompt") or "")
            return (
                '{"cache":null,"annulments":[],"rehearsal":"watching",'
                '"response_target":"observe","response":""}'
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
    monkeypatch.setattr(main, "_current_whatsapp_active_since_for_soul", lambda *_a, **_k: None)

    payload = {
        "user": {"user_id": "u1", "soul_id": "Siri", "conversation_id": "whatsapp:group:familia"},
        "message": "Siri is listening.",
        "chat_name": "Household Group",
        "chat_type": "group",
        "history": [],
        "allow_public_response": False,
        "prompt_override_payload": {
            "user_prompt": "prompt",
            "memory_cache": [],
            "intentions_active": {"items": []},
            "retrieve_rag": {"items": [], "categories": [], "resources": []},
        },
    }

    out = await main.conversation_turn("whatsapp:group:familia", payload)

    assert out["ok"] is True
    assert out["response_target"] == "observe"
    assert out["response"] == ""
    assert '"response_target":"observe|private"' in captured["system_prompt"]
    assert '"respond"' not in captured["system_prompt"]


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
async def test_conversation_turn_uses_fresh_session_id_for_retry(
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
            self.session_ids: list[str | None] = []

        async def chat(self, *_args, **kwargs) -> str:
            self.calls += 1
            self.session_ids.append(kwargs.get("session_id"))
            if self.calls == 1:
                return '{"cache":null,"annulments":[],"rehearsal":"bad"}'
            return (
                '{"cache":null,"annulments":[],"rehearsal":"retry good",'
                '"response_target":"respond","response":"assistant says hi"}'
            )

    svc = _FakeSvc()

    async def _fake_persist_annulment_memories(**_kwargs):
        return []

    monkeypatch.setattr(main, "_CONFIG", {**main._CONFIG, "claude_code": True})
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
    assert svc.calls == 2
    assert svc.session_ids[0]
    assert svc.session_ids[1]
    assert svc.session_ids[0] != svc.session_ids[1]


@pytest.mark.asyncio
async def test_free_turn_chain_caps_at_three_without_direct_memorize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "SoulTest.db"
    monkeypatch.setattr(main, "_sqlite_current_path", lambda _user_id, _soul_id: db_path)

    class _FakeSvc:
        def __init__(self) -> None:
            self.chat_calls: list[dict[str, object]] = []

        async def chat(self, *_args, **kwargs) -> str:
            self.chat_calls.append(dict(kwargs))
            return (
                '{"cache":null,"annulments":[],"rehearsal":"continued",'
                '"response_target":"listen","response":"",'
                '"activity_recap":"I continued the task.",'
                '"continue_reason":"task"}'
            )

        async def memorize(self, **_kwargs) -> dict[str, object]:
            raise AssertionError("free-turn continuation must not bypass the normal memorize trigger")

    svc = _FakeSvc()
    try:
        await main._run_free_turn_chain(
            marker="u1::Siri",
            service=svc,
            user_id="u1",
            soul_id="Siri",
            conversation_id="whatsapp:dm:Marcos",
            session_id="session-123",
            initial_reason="task",
            initial_contract={
                "response_target": "listen",
                "response": "",
                "rehearsal": "starting",
            },
            system_prompt="system",
            allow_public_response=True,
            safe_payload={},
            soul_card=None,
        )
    finally:
        main._FREE_TURN_INFLIGHT.clear()

    assert len(svc.chat_calls) == 3
    assert all(call["resume_session_id"] == "session-123" for call in svc.chat_calls)

    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM activity_messages ORDER BY source_conversation_index").fetchall()
    finally:
        con.close()

    assert [row["content"] for row in rows] == [
        "I continued the task.",
        "I continued the task.",
        "I continued the task.",
    ]


def test_free_turn_prompt_uses_observe_for_listen_only_policy() -> None:
    prompt = main._build_free_turn_prompt(
        reason="research",
        continuation_index=1,
        origin_conversation_id="whatsapp:group:familia",
        previous_contract={"response_target": "observe", "response": "", "rehearsal": "thinking"},
        allow_public_response=False,
    )

    assert "observe/private" in prompt
    assert "Do not use listen/respond" in prompt


@pytest.mark.asyncio
async def test_whatsapp_outbound_claim_and_mark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "SiriTest.db"
    monkeypatch.setattr(main, "_sqlite_current_path", lambda _user_id, _soul_id: db_path)

    out_id = main._insert_whatsapp_outbound(
        user_id="u1",
        soul_id="Siri",
        origin_conversation_id="whatsapp:dm:Marcos",
        target="respond",
        response_text="hello",
        metadata={"source": "test"},
    )

    claimed = await main.whatsapp_outbounds_claim(
        {"user_id": "u1", "soul_id": "Siri", "claimed_by": "hermes-test", "limit": 10}
    )
    assert [row["id"] for row in claimed["outbounds"]] == [out_id]
    assert claimed["outbounds"][0]["target_conversation_id"] == "whatsapp:dm:Marcos"
    assert claimed["outbounds"][0]["metadata"] == {"source": "test"}

    claimed_again = await main.whatsapp_outbounds_claim(
        {"user_id": "u1", "soul_id": "Siri", "claimed_by": "hermes-test", "limit": 10}
    )
    assert claimed_again["outbounds"] == []

    marked = await main.whatsapp_outbounds_mark(
        {
            "user_id": "u1",
            "soul_id": "Siri",
            "outbound_id": out_id,
            "status": "sent",
            "provider_message_id": "wa-msg-1",
        }
    )
    assert marked["outbound"]["status"] == "sent"
    assert marked["outbound"]["provider_message_id"] == "wa-msg-1"


@pytest.mark.asyncio
async def test_whatsapp_outbound_empty_claim_does_not_create_wal_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "SiriTest.db"
    monkeypatch.setattr(main, "_sqlite_current_path", lambda _user_id, _soul_id: db_path)

    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA journal_mode=WAL")
        main._ensure_whatsapp_outbounds_schema(con)
    finally:
        con.close()
    for suffix in ("-wal", "-shm"):
        (tmp_path / f"SiriTest.db{suffix}").unlink(missing_ok=True)

    claimed = await main.whatsapp_outbounds_claim(
        {"user_id": "u1", "soul_id": "Siri", "claimed_by": "hermes-test", "limit": 10}
    )

    assert claimed["outbounds"] == []
    assert not (tmp_path / "SiriTest.db-wal").exists()
    assert not (tmp_path / "SiriTest.db-shm").exists()


@pytest.mark.asyncio
async def test_free_turn_chain_queues_whatsapp_outbound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "SiriTest.db"
    monkeypatch.setattr(main, "_sqlite_current_path", lambda _user_id, _soul_id: db_path)

    class _FakeSvc:
        async def chat(self, *_args, **_kwargs) -> str:
            return (
                '{"cache":null,"annulments":[],"rehearsal":"continued",'
                '"response_target":"private","response":"I found something."}'
            )

        async def memorize(self, **_kwargs) -> dict[str, object]:
            return {"ok": True}

    try:
        await main._run_free_turn_chain(
            marker="u1::Siri",
            service=_FakeSvc(),
            user_id="u1",
            soul_id="Siri",
            conversation_id="whatsapp:dm:Marcos",
            session_id="session-123",
            initial_reason="research",
            initial_contract={
                "response_target": "listen",
                "response": "",
                "rehearsal": "starting",
            },
            system_prompt="system",
            allow_public_response=True,
            safe_payload={"timezone": "America/Lima"},
            soul_card=None,
        )
    finally:
        main._FREE_TURN_INFLIGHT.clear()

    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM whatsapp_pending_outbounds").fetchall()
    finally:
        con.close()

    assert len(rows) == 1
    assert rows[0]["target"] == "private"
    assert rows[0]["target_conversation_id"] is None
    assert rows[0]["response_text"] == "I found something."


@pytest.mark.asyncio
async def test_free_turn_chain_extracts_json_contract_after_prose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    db_path = tmp_path / "SiriTest.db"
    monkeypatch.setattr(main, "_sqlite_current_path", lambda _user_id, _soul_id: db_path)

    class _FakeSvc:
        async def chat(self, *_args, **_kwargs) -> str:
            return (
                "Good. Research is saved. I'll message him privately.\n\n"
                '{"cache":null,"annulments":[],"rehearsal":"continued",'
                '"response_target":"private","response":"Research is ready."}'
            )

        async def memorize(self, **_kwargs) -> dict[str, object]:
            return {"ok": True}

    try:
        with caplog.at_level(logging.WARNING):
            await main._run_free_turn_chain(
                marker="u1::Siri",
                service=_FakeSvc(),
                user_id="u1",
                soul_id="Siri",
                conversation_id="whatsapp:dm:Marcos",
                session_id="session-123",
                initial_reason="research",
                initial_contract={
                    "response_target": "listen",
                    "response": "",
                    "rehearsal": "starting",
                },
                system_prompt="system",
                allow_public_response=True,
                safe_payload={"timezone": "America/Lima"},
                soul_card=None,
            )
    finally:
        main._FREE_TURN_INFLIGHT.clear()

    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM whatsapp_pending_outbounds").fetchall()
    finally:
        con.close()

    assert len(rows) == 1
    assert rows[0]["target"] == "private"
    assert rows[0]["response_text"] == "Research is ready."
    assert any("extracted turn contract" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_free_turn_chain_ignores_non_whatsapp_outbound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "SiriTest.db"
    monkeypatch.setattr(main, "_sqlite_current_path", lambda _user_id, _soul_id: db_path)

    class _FakeSvc:
        async def chat(self, *_args, **_kwargs) -> str:
            return (
                '{"cache":null,"annulments":[],"rehearsal":"continued",'
                '"response_target":"private","response":"I found something."}'
            )

        async def memorize(self, **_kwargs) -> dict[str, object]:
            return {"ok": True}

    try:
        await main._run_free_turn_chain(
            marker="u1::Siri",
            service=_FakeSvc(),
            user_id="u1",
            soul_id="Siri",
            conversation_id="sillytavern:chat-1",
            session_id="session-123",
            initial_reason="research",
            initial_contract={
                "response_target": "listen",
                "response": "",
                "rehearsal": "starting",
            },
            system_prompt="system",
            allow_public_response=True,
            safe_payload={"timezone": "America/Lima"},
            soul_card=None,
        )
    finally:
        main._FREE_TURN_INFLIGHT.clear()

    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        outbound_table = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'whatsapp_pending_outbounds'"
        ).fetchone()
        activity_rows = con.execute("SELECT content FROM activity_messages").fetchall()
    finally:
        con.close()

    assert outbound_table is None
    assert [row["content"] for row in activity_rows] == ["continued"]


def test_free_turn_follow_up_payload_excludes_caller_timezone() -> None:
    payload = main._free_turn_followup_payload(
        {
            "user": {"user_id": "u1", "soul_id": "Siri"},
            "chat_name": "Marcos",
            "time_zone": "America/Lima",
            "time_zone_offset_min": -300,
        }
    )

    assert payload["chat_name"] == "Marcos"
    assert "time_zone" not in payload
    assert "time_zone_offset_min" not in payload


def test_parse_free_turn_follow_up_at_naive_iso_uses_server_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    import zoneinfo
    fake_tz = zoneinfo.ZoneInfo("America/Lima")  # UTC-5
    monkeypatch.setattr(main._memorize_endpoint, "server_timezone", lambda: fake_tz)

    result = main._parse_free_turn_follow_up_at("2026-06-15T14:00:00")

    assert result is not None
    assert result.tzinfo is not None
    # Lima is UTC-5 → 14:00 local = 19:00 UTC
    assert result.hour == 19
    assert result.minute == 0


def test_parse_free_turn_follow_up_at_tz_aware_iso_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    import zoneinfo
    fake_tz = zoneinfo.ZoneInfo("America/Lima")
    monkeypatch.setattr(main._memorize_endpoint, "server_timezone", lambda: fake_tz)

    # Explicit +02:00 → 14:00+02:00 = 12:00 UTC, not affected by server tz
    result = main._parse_free_turn_follow_up_at("2026-06-15T14:00:00+02:00")

    assert result is not None
    assert result.hour == 12
    assert result.minute == 0


def test_parse_free_turn_follow_up_at_z_suffix_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    import zoneinfo
    fake_tz = zoneinfo.ZoneInfo("America/Lima")
    monkeypatch.setattr(main._memorize_endpoint, "server_timezone", lambda: fake_tz)

    result = main._parse_free_turn_follow_up_at("2026-06-15T14:00:00Z")

    assert result is not None
    assert result.hour == 14  # already UTC
    assert result.minute == 0


def test_parse_free_turn_follow_up_at_human_format_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    import zoneinfo
    fake_tz = zoneinfo.ZoneInfo("America/Lima")
    monkeypatch.setattr(main._memorize_endpoint, "server_timezone", lambda: fake_tz)

    result = main._parse_free_turn_follow_up_at("Monday, June 15, 2026 14:00 PET")

    assert result is not None
    assert result.hour == 19  # Lima UTC-5
    assert result.minute == 0


def test_parse_free_turn_follow_up_at_garbage_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    import zoneinfo
    fake_tz = zoneinfo.ZoneInfo("America/Lima")
    monkeypatch.setattr(main._memorize_endpoint, "server_timezone", lambda: fake_tz)

    assert main._parse_free_turn_follow_up_at("not a date at all") is None
    assert main._parse_free_turn_follow_up_at("") is None
    assert main._parse_free_turn_follow_up_at(None) is None  # type: ignore[arg-type]


def test_free_turn_follow_up_schedule_persists_pending_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "SiriTest.db"
    monkeypatch.setattr(main, "_sqlite_current_path", lambda _user_id, _soul_id: db_path)

    followup_id = main._schedule_free_turn_follow_up(
        user_id="u1",
        soul_id="Siri",
        conversation_id="whatsapp:dm:Marcos",
        follow_up_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        follow_up_reason="Check whether Marcos got home safely.",
        safe_payload={"user": {"user_id": "u1", "soul_id": "Siri"}},
    )

    assert followup_id
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM free_turn_followups WHERE id = ?", (followup_id,)).fetchone()
    finally:
        con.close()
    assert row is not None
    assert row["status"] == "pending"
    assert row["conversation_id"] == "whatsapp:dm:Marcos"
    payload = json.loads(row["payload_json"])
    assert payload["follow_up_reason"] == "Check whether Marcos got home safely."


def test_free_turn_empty_claim_does_not_create_wal_sidecars(tmp_path: Path) -> None:
    db_path = tmp_path / "SiriTest.db"
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA journal_mode=WAL")
        main._ensure_free_turn_followups_schema(con)
    finally:
        con.close()
    for suffix in ("-wal", "-shm"):
        (tmp_path / f"SiriTest.db{suffix}").unlink(missing_ok=True)

    claimed = main._claim_due_free_turn_followups(db_path, now=datetime.now(UTC))

    assert claimed == []
    assert not (tmp_path / "SiriTest.db-wal").exists()
    assert not (tmp_path / "SiriTest.db-shm").exists()


@pytest.mark.asyncio
async def test_due_free_turn_follow_up_runs_fresh_turn_and_queues_outbound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "SiriTest.db"
    monkeypatch.setattr(main, "_sqlite_current_path", lambda _user_id, _soul_id: db_path)
    monkeypatch.setattr(main, "_free_turn_followup_db_paths", lambda: [db_path])

    followup_id = main._schedule_free_turn_follow_up(
        user_id="u1",
        soul_id="Siri",
        conversation_id="whatsapp:dm:Marcos",
        follow_up_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
        follow_up_reason="Check whether Marcos got home safely.",
        safe_payload={
            "user": {"user_id": "u1", "soul_id": "Siri"},
            "chat_name": "Marcos",
            "allow_public_response": True,
        },
    )
    assert followup_id
    calls: dict[str, Any] = {}

    async def _fake_retrieve(conversation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls["retrieve"] = {"conversation_id": conversation_id, "payload": payload}
        return {
            "ok": True,
            "turn_user_prompt": "fresh prompt",
            "turn_system_prompt": "system",
            "memory_cache": [],
            "intentions_active": {"items": []},
            "result": {"categories": [], "items": [], "resources": []},
            "turn_prompt_source": "conversation_retrieve",
            "turn_history": [{"role": "user", "content": "fresh history"}],
        }

    async def _fake_turn(conversation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls["turn"] = {"conversation_id": conversation_id, "payload": payload}
        return {"ok": True, "response_target": "private", "response": "follow-up note"}

    monkeypatch.setattr(main, "conversation_retrieve", _fake_retrieve)
    monkeypatch.setattr(main, "conversation_turn", _fake_turn)

    assert await main._run_due_free_turn_followups_once() == 1

    assert calls["retrieve"]["payload"]["load_source_history"] is True
    assert calls["retrieve"]["payload"]["is_live_turn"] is False
    assert "message" not in calls["retrieve"]["payload"]
    assert "query" not in calls["retrieve"]["payload"]
    assert calls["retrieve"]["payload"]["self_turn_label"] == "Scheduled wake"
    assert "Check whether Marcos got home safely." in calls["retrieve"]["payload"]["self_turn_directive"]
    assert "message" not in calls["turn"]["payload"]
    assert calls["turn"]["payload"]["load_source_history"] is True
    assert calls["turn"]["payload"]["history"] == []
    assert calls["turn"]["payload"]["self_turn_label"] == "Scheduled wake"
    assert "Check whether Marcos got home safely." in calls["turn"]["payload"]["self_turn_directive"]
    assert calls["turn"]["payload"]["prompt_override_payload"]["user_prompt"] == "fresh prompt"
    trace_id = calls["retrieve"]["payload"]["trace_id"]
    assert isinstance(trace_id, str)
    assert len(trace_id) == 32
    assert calls["turn"]["payload"]["trace_id"] == trace_id

    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        followup = con.execute("SELECT * FROM free_turn_followups WHERE id = ?", (followup_id,)).fetchone()
        outbounds = con.execute("SELECT * FROM whatsapp_pending_outbounds").fetchall()
    finally:
        con.close()

    assert followup["status"] == "completed"
    assert len(outbounds) == 1
    assert outbounds[0]["target"] == "private"
    assert outbounds[0]["response_text"] == "follow-up note"
    metadata = json.loads(outbounds[0]["metadata_json"])
    assert metadata["followup_id"] == followup_id
    assert metadata["requested_target"] == "private"


@pytest.mark.asyncio
async def test_due_free_turn_follow_up_from_sillytavern_queues_private_whatsapp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "SiriTest.db"
    monkeypatch.setattr(main, "_sqlite_current_path", lambda _user_id, _soul_id: db_path)
    monkeypatch.setattr(main, "_free_turn_followup_db_paths", lambda: [db_path])

    followup_id = main._schedule_free_turn_follow_up(
        user_id="u1",
        soul_id="Siri",
        conversation_id="sillytavern:chat-1",
        follow_up_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
        follow_up_reason="Tell Marcos what I found.",
        safe_payload={"user": {"user_id": "u1", "soul_id": "Siri"}},
    )
    assert followup_id

    async def _fake_retrieve(_conversation_id: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": True,
            "turn_user_prompt": "fresh prompt",
            "turn_system_prompt": "system",
            "memory_cache": [],
            "intentions_active": {"items": []},
            "result": {"categories": [], "items": [], "resources": []},
            "turn_prompt_source": "conversation_retrieve",
        }

    async def _fake_turn(_conversation_id: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "response_target": "respond", "response": "I found it."}

    monkeypatch.setattr(main, "conversation_retrieve", _fake_retrieve)
    monkeypatch.setattr(main, "conversation_turn", _fake_turn)

    assert await main._run_due_free_turn_followups_once() == 1

    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        followup = con.execute("SELECT * FROM free_turn_followups WHERE id = ?", (followup_id,)).fetchone()
        outbounds = con.execute("SELECT * FROM whatsapp_pending_outbounds").fetchall()
    finally:
        con.close()

    assert followup["status"] == "completed"
    assert len(outbounds) == 1
    assert outbounds[0]["origin_conversation_id"] == "sillytavern:chat-1"
    assert outbounds[0]["target"] == "private"
    metadata = json.loads(outbounds[0]["metadata_json"])
    assert metadata["requested_target"] == "respond"


@pytest.mark.asyncio
async def test_due_free_turn_follow_up_enqueue_failure_marks_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "SiriTest.db"
    monkeypatch.setattr(main, "_sqlite_current_path", lambda _user_id, _soul_id: db_path)
    monkeypatch.setattr(main, "_free_turn_followup_db_paths", lambda: [db_path])

    followup_id = main._schedule_free_turn_follow_up(
        user_id="u1",
        soul_id="Siri",
        conversation_id="whatsapp:dm:Marcos",
        follow_up_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
        follow_up_reason="Check in.",
        safe_payload={"user": {"user_id": "u1", "soul_id": "Siri"}},
    )
    assert followup_id

    async def _fake_retrieve(_conversation_id: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": True,
            "turn_user_prompt": "fresh prompt",
            "turn_system_prompt": "system",
            "memory_cache": [],
            "intentions_active": {"items": []},
            "result": {"categories": [], "items": [], "resources": []},
            "turn_prompt_source": "conversation_retrieve",
        }

    async def _fake_turn(_conversation_id: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "response_target": "private", "response": "Checking in."}

    def _fail_insert(**_kwargs: Any) -> str:
        raise RuntimeError("queue unavailable")

    monkeypatch.setattr(main, "conversation_retrieve", _fake_retrieve)
    monkeypatch.setattr(main, "conversation_turn", _fake_turn)
    monkeypatch.setattr(main, "_insert_whatsapp_outbound", _fail_insert)

    assert await main._run_due_free_turn_followups_once() == 1

    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        followup = con.execute("SELECT * FROM free_turn_followups WHERE id = ?", (followup_id,)).fetchone()
    finally:
        con.close()

    assert followup["status"] == "failed"
    assert "RuntimeError: queue unavailable" in followup["last_error"]


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


# --- whatsapp outbound media_path ---

@pytest.mark.asyncio
async def test_whatsapp_outbound_insert_claim_row_with_media_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "SiriTest.db"
    monkeypatch.setattr(main, "_sqlite_current_path", lambda _u, _s: db_path)

    out_id = main._insert_whatsapp_outbound(
        user_id="u1",
        soul_id="Siri",
        origin_conversation_id="whatsapp:dm:Marcos",
        target="private",
        response_text="",
        media_path="/home/marcos/Desktop/siri/report.pdf",
        metadata={"source": "test"},
    )
    assert out_id.startswith("waout_")

    claimed = await main.whatsapp_outbounds_claim(
        {"user_id": "u1", "soul_id": "Siri", "claimed_by": "hermes-test", "limit": 10}
    )
    rows = claimed["outbounds"]
    assert len(rows) == 1
    assert rows[0]["id"] == out_id
    assert rows[0]["media_path"] == "/home/marcos/Desktop/siri/report.pdf"
    assert rows[0]["response_text"] == ""


@pytest.mark.asyncio
async def test_whatsapp_outbound_text_or_media_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "SiriTest.db"
    monkeypatch.setattr(main, "_sqlite_current_path", lambda _u, _s: db_path)

    with pytest.raises(ValueError, match="response_text or media_path is required"):
        main._insert_whatsapp_outbound(
            user_id="u1",
            soul_id="Siri",
            origin_conversation_id="whatsapp:dm:Marcos",
            target="respond",
            response_text="",
            media_path=None,
        )


def test_whatsapp_outbounds_schema_migration_idempotent(tmp_path: Path) -> None:
    """ALTER TABLE on a DB that already has the column must not raise."""
    import sqlite3 as _sqlite3
    db_path = tmp_path / "existing.db"
    con = _sqlite3.connect(str(db_path))
    con.row_factory = _sqlite3.Row
    # Create table without media_path first, simulating a pre-migration DB.
    con.execute("""
CREATE TABLE whatsapp_pending_outbounds (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    soul_id TEXT NOT NULL,
    origin_conversation_id TEXT NOT NULL,
    target TEXT NOT NULL,
    target_conversation_id TEXT,
    response_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    claimed_at TEXT,
    claimed_by TEXT,
    sent_at TEXT,
    failed_at TEXT,
    provider_message_id TEXT,
    last_error TEXT,
    metadata_json TEXT
)
""")
    con.commit()
    # Running the schema function twice must not raise.
    main._ensure_whatsapp_outbounds_schema(con)
    main._ensure_whatsapp_outbounds_schema(con)
    # Confirm the column now exists.
    cols = {row[1] for row in con.execute("PRAGMA table_info(whatsapp_pending_outbounds)")}
    assert "media_path" in cols
    con.close()


# --- normal-turn attachment delivery ---

def _make_turn_monkeypatches(
    monkeypatch: pytest.MonkeyPatch,
    db_path: "Path",
    chat_response: str,
) -> None:
    """Shared setup for conversation_turn attachment tests."""

    class _FakeSvc:
        async def chat(self, *_args, **_kwargs) -> str:
            return chat_response

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
    monkeypatch.setattr(main, "_current_whatsapp_active_since_for_soul", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_a, **_k: db_path)


@pytest.mark.asyncio
async def test_conversation_turn_attachment_enqueues_captioned_outbound_without_inline_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal WhatsApp turn with attachment queues one captioned media outbound row."""
    db_path = tmp_path / "SiriTest.db"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.commit()
    finally:
        con.close()

    workspace = tmp_path / "siri-workspace"
    workspace.mkdir()
    media_path = workspace / "report.pdf"
    media_path.write_text("report")
    media = str(media_path)
    monkeypatch.setitem(main._CONFIG, "claude_code_workspace", str(workspace))
    _make_turn_monkeypatches(
        monkeypatch,
        db_path,
        '{"cache":null,"annulments":[],"rehearsal":"done",'
        f'"response_target":"respond","response":"Here is your report.",'
        f'"attachment":"{media}"}}',
    )

    payload = {
        "user": {"user_id": "u1", "soul_id": "Siri", "conversation_id": "whatsapp:dm:Marcos"},
        "message": "send me the report",
        "user_name": "Marcos",
        "chat_name": "Marcos",
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

    out = await main.conversation_turn("whatsapp:dm:Marcos", payload)

    assert out["ok"] is True
    assert out["response"] == ""
    assert "attachment" not in out

    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM whatsapp_pending_outbounds").fetchall()
    finally:
        con.close()

    assert len(rows) == 1
    assert rows[0]["response_text"] == "Here is your report."
    assert rows[0]["media_path"] == media
    assert rows[0]["target"] == "respond"
    import json as _json
    meta = _json.loads(rows[0]["metadata_json"] or "{}")
    assert meta.get("source") == "turn_attachment"


@pytest.mark.asyncio
async def test_conversation_turn_listen_target_attachment_does_not_enqueue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Attachment on a listen-target turn is logged as an error but not enqueued."""
    db_path = tmp_path / "SiriTest.db"
    con = main._sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        main._sqlite_ensure_conversation_state_schema(con)
        con.commit()
    finally:
        con.close()

    workspace = tmp_path / "siri-workspace"
    workspace.mkdir()
    media_path = workspace / "note.txt"
    media_path.write_text("note")
    media = str(media_path)
    monkeypatch.setitem(main._CONFIG, "claude_code_workspace", str(workspace))
    _make_turn_monkeypatches(
        monkeypatch,
        db_path,
        '{"cache":null,"annulments":[],"rehearsal":"watching",'
        f'"response_target":"listen","response":"",'
        f'"attachment":"{media}"}}',
    )

    payload = {
        "user": {"user_id": "u1", "soul_id": "Siri", "conversation_id": "whatsapp:dm:Marcos"},
        "message": "quiet",
        "user_name": "Marcos",
        "chat_name": "Marcos",
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

    with caplog.at_level(logging.ERROR):
        out = await main.conversation_turn("whatsapp:dm:Marcos", payload)

    assert out["ok"] is True
    assert out["response"] == ""
    assert any("attachment dropped" in r.getMessage() for r in caplog.records)
