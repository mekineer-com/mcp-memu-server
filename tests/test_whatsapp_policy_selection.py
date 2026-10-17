import json
import logging
import sqlite3
from types import SimpleNamespace

import pytest

from app.services import conversation_sources, cross_history, message_log, payload, turn_contract
from memu.utils.conversation import format_grouped_chat_history


def write_policy(home, channels, default="full"):
    (home / "memu.json").write_text(json.dumps({"whatsapp": {
        "default_policy": default, "channels": channels,
    }}))


@pytest.mark.parametrize("routing", ["full", "listen_only"])
@pytest.mark.parametrize("memorize", [False, True])
def test_whatsapp_policy_routing_and_mem_are_independent(tmp_path, routing, memorize):
    write_policy(tmp_path, {"19990000001@s.whatsapp.net": {"policy": routing, "memorize": memorize}})
    assert conversation_sources.load_whatsapp_channel_settings(
        "whatsapp:dm:19990000001:4@s.whatsapp.net", hermes_home=tmp_path,
    ) == (routing, memorize)


@pytest.mark.parametrize("self_alias", [False, True])
def test_whatsapp_policy_aliases_keep_channels_precedence_without_contact_store(tmp_path, self_alias):
    session = tmp_path / "whatsapp" / "session"
    session.mkdir(parents=True)
    if self_alias:
        (session / "creds.json").write_text(json.dumps({"me": {
            "id": "19990000001@s.whatsapp.net", "lid": "999000000001@lid",
        }}))
    else:
        (session / "lid-mapping-19990000001.json").write_text('"999000000001"')
        (session / "lid-mapping-999000000001_reverse.json").write_text('"19990000001"')
    entries = {"19990000001@s.whatsapp.net": {"policy": "listen_only", "memorize": False},
               "999000000001@lid": {"policy": "full", "memorize": True}}
    write_policy(tmp_path, entries)
    for cid in ("whatsapp:dm:19990000001", "whatsapp:dm:999000000001@lid"):
        assert conversation_sources.load_whatsapp_channel_settings(cid, hermes_home=tmp_path) == ("listen_only", True)
    entries["19990000001@s.whatsapp.net"]["policy"] = "excluded"
    write_policy(tmp_path, entries)
    assert conversation_sources.load_whatsapp_channel_settings(
        "whatsapp:dm:999000000001@lid", hermes_home=tmp_path,
    ) == ("excluded", False)


@pytest.mark.parametrize("raw", [None, "{", "[]", '{"whatsapp":{"channels":[]}}'])
def test_whatsapp_policy_preserves_channels_missing_malformed_defaults(tmp_path, raw):
    if raw is not None:
        (tmp_path / "memu.json").write_text(raw)
    assert conversation_sources.load_whatsapp_channel_settings(
        "whatsapp:group:19990000001-19990000002@g.us", hermes_home=tmp_path,
    ) == ("full", True)
    write_policy(tmp_path, {}, default="excluded")
    assert conversation_sources.load_whatsapp_channel_settings(
        "whatsapp:group:19990000001-19990000002@g.us", hermes_home=tmp_path,
    ) == ("excluded", False)


def selection_fixture(tmp_path, monkeypatch):
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE conversations (
        conversation_id TEXT, user_id TEXT DEFAULT 'TestOwner', soul_id TEXT DEFAULT 'TestSoul',
        memorize_chat INTEGER, digest_cursor INTEGER,
        last_memorize_at TEXT, digest_cursor_source_message_id TEXT, digest_cursor_ts INTEGER,
        rolling_summary_cursor_id INTEGER, rolling_summary_cursor_source_message_id TEXT,
        rolling_summary_cursor_ts INTEGER, rolling_summary TEXT, rolling_summary_updated_at TEXT,
        last_display_segment_start_index INTEGER, last_display_segment_end_index INTEGER,
        last_display_segment_at TEXT, import_state TEXT)""")
    cid = "whatsapp:group:19990000001-19990000002@g.us"
    con.execute("INSERT INTO conversations (conversation_id, memorize_chat, digest_cursor, last_memorize_at, "
                "rolling_summary_cursor_id, rolling_summary_cursor_source_message_id, rolling_summary_cursor_ts, "
                "rolling_summary, rolling_summary_updated_at) VALUES (?, 1, 9, 'saved', 20, 'old-key', 100, 'Verbatim summary', 'stamp')", (cid,))
    calls = []

    def background(**kwargs):
        calls.append(("background", kwargs))
        return [{"content": "Fictional new tail", "source_conversation_index": 78}]

    def primary(**kwargs):
        calls.append(("primary", kwargs))
        return [{"content": "Fictional primary tail", "source_conversation_index": 10}]

    runtime = SimpleNamespace(
        _CONFIG={"hermes": {"home": str(tmp_path)}},
        _resolve_cross_source_paths=lambda: (tmp_path, tmp_path, None, None),
        _load_background_rollup_tail=background,
        _load_tail_for_source_conversation=primary,
        _stamp_assistant_display_name=lambda *args: None,
        logger=logging.getLogger(__name__),
    )
    monkeypatch.setattr(cross_history, "_main", runtime)
    monkeypatch.setattr(message_log, "_main", runtime)
    monkeypatch.setattr(cross_history, "_resolve_source_cursor", lambda cid, cursor, *args, rolling, **kwargs:
                        (77, 123, True) if rolling else (cursor, None, True))
    (tmp_path / "channel_directory.json").write_text(json.dumps({"platforms": {"whatsapp": [
        {"id": cid.removeprefix("whatsapp:group:"), "name": "Fictional Work Group"},
    ]}}))
    return con, cid, calls


def test_cross_selection_current_policy_and_captured_retry_agree(tmp_path, monkeypatch):
    con, cid, calls = selection_fixture(tmp_path, monkeypatch)
    try:
        write_policy(tmp_path, {cid.removeprefix("whatsapp:group:"): {"policy": "full", "memorize": False}})
        live = cross_history._load_cross_tail_from_sources(con, user_id="TestOwner", soul_id="TestSoul")
        tails = cross_history._load_cross_memorize_tails_from_sources(con, user_id="TestOwner", soul_id="TestSoul", strict=True)
        assert live == tails[cid] and live[0]["memorize_chat"] is False
        assert all(kind == "background" and kw["rolling_summary_cursor_id"] == 77 and kw["min_timestamp"] == 123
                   for kind, kw in calls)
        normalized = dict(con.execute("SELECT * FROM conversations").fetchone())
        normalized["import_state"] = {"history_end_index": 2}
        cross_history._load_background_tail_from_source(cid, normalized, user_id="TestOwner", soul_id="TestSoul")
        assert calls[-1][1]["import_state"] == normalized["import_state"]
        summaries = cross_history._read_background_rolling_summaries_from_conversations(con)
        assert summaries[cid]["summary"] == "Verbatim summary"
        assert summaries[cid]["chat_name"] == "Fictional Work Group"
        write_policy(tmp_path, {cid.removeprefix("whatsapp:group:"): {"policy": "excluded", "memorize": True}})
        calls.clear()
        assert cross_history._load_cross_tail_from_sources(con, user_id="TestOwner", soul_id="TestSoul") == []
        assert cross_history._load_cross_memorize_tails_from_sources(con, user_id="TestOwner", soul_id="TestSoul", strict=True) == {}
        assert cross_history._read_background_rolling_summaries_from_conversations(con) == {}
        assert calls == []
        captured = {cid: False}
        assert cross_history._load_cross_memorize_tails_from_sources(con, user_id="TestOwner", soul_id="TestSoul",
                                                                    filing_policy=captured, strict=True)[cid][0]["memorize_chat"] is False
        assert cid in cross_history._read_background_rolling_summaries_from_conversations(con, filing_policy=captured)
        captured[cid] = True
        assert cross_history._load_cross_memorize_tails_from_sources(con, user_id="TestOwner", soul_id="TestSoul",
                                                                    filing_policy=captured, strict=True)[cid][0]["memorize_chat"] is True
        assert cross_history._read_background_rolling_summaries_from_conversations(con, filing_policy=captured) == {}
    finally:
        con.close()


def test_current_excluded_source_is_not_read_and_primary_floors_stay(tmp_path, monkeypatch):
    con, cid, calls = selection_fixture(tmp_path, monkeypatch)
    try:
        write_policy(tmp_path, {cid.removeprefix("whatsapp:group:"): {"policy": "listen_only", "memorize": True}})
        rows = cross_history._load_cross_tail_from_sources(con, user_id="TestOwner", soul_id="TestSoul")
        assert rows[0]["memorize_chat"] is True
        assert calls[-1][0] == "primary" and calls[-1][1]["since_cursor"] == 9
        assert calls[-1][1]["recent_fallback_messages"] == 8
        con.execute("UPDATE conversations SET last_display_segment_start_index=1, "
                    "last_display_segment_end_index=9, last_display_segment_at='latest'")
        cross_history._load_cross_tail_from_sources(con, user_id="TestOwner", soul_id="TestSoul")
        assert calls[-1][1]["include_floor_without_new"] is True
        write_policy(tmp_path, {cid.removeprefix("whatsapp:group:"): {"policy": "excluded"}})
        assert cross_history._load_current_whatsapp_history_from_source(cid, "TestSoul", active_since=None) == []
    finally:
        con.close()


def test_quiet_import_summary_uses_scoped_metadata_without_tail(tmp_path, monkeypatch):
    con, _, calls = selection_fixture(tmp_path, monkeypatch)
    source_path = tmp_path / "imports.db"
    with sqlite3.connect(source_path) as source:
        source.execute("CREATE TABLE imported_chats (chat_id TEXT PRIMARY KEY, user_id TEXT, "
                       "soul_id TEXT, label TEXT, title TEXT)")
        source.execute("INSERT INTO imported_chats VALUES ('fictional-import', 'TestOwner', 'TestSoul', "
                       "'TestApp', 'Fictional Archive')")
    monkeypatch.setattr(conversation_sources, "import_source_path", lambda: source_path)
    cid = "import:dm:fictional-import"
    prose = "Saved fictional context.\n\nSecond paragraph stays verbatim."
    con.execute("INSERT INTO conversations (conversation_id, memorize_chat, rolling_summary) VALUES (?, 0, ?)",
                (cid, prose))
    try:
        summary = cross_history._read_background_rolling_summaries_from_conversations(con)[cid]
        assert summary["source_conversation_id"] == cid
        assert summary["app_label"] == "TestApp" and summary["chat_name"] == "Fictional Archive"
        assert summary["summary"] == prose and calls == []
        rendered = format_grouped_chat_history([], background_context_rows=[summary], soul_name="TestSoul")
        assert "My TestApp Conversations:" in rendered and "Fictional Archive" in rendered and prose in rendered
        con.execute("UPDATE conversations SET user_id='OtherOwner' WHERE conversation_id=?", (cid,))
        with pytest.raises(ValueError, match="Imported chat metadata missing"):
            cross_history._read_background_rolling_summaries_from_conversations(con)
    finally:
        con.close()


def test_whatsapp_keys_preserve_primary_fallback_and_reflection_rebase(tmp_path, monkeypatch):
    con, _, _ = selection_fixture(tmp_path, monkeypatch)
    try:
        runtime = cross_history._main
        runtime._build_conversations_block = turn_contract.build_conversations_block
        runtime._chat_label_from_history = cross_history._chat_label_from_history
        cid = "whatsapp:dm:19990000001"
        captured = {"role": "user", "content": "Captured fictional fact", "source_conversation_id": cid,
                    "source_message_id": "physical-a", "source_conversation_index": 5}
        rebased = {**captured, "source_conversation_index": 500, "content": "Duplicate rebased fact"}
        distinct = {**captured, "source_message_id": "physical-b", "content": "Distinct fictional fact"}
        assert payload.message_source_keys(captured, prefer_id=True) == payload.message_source_keys(rebased, prefer_id=True)
        assert payload.message_source_keys(captured, prefer_id=True).isdisjoint(payload.message_source_keys(distinct, prefer_id=True))
        assert not payload.message_source_keys(captured).isdisjoint(payload.message_source_keys(distinct))
        rendered = cross_history._format_all_chat_history_for_ai(
            current_history=[captured], cross_tail=[rebased, distinct], conversation_id=cid,
            soul_id="TestSoul", mark_current_chat=False, memory_context=True, background_context_rows=[],
        )
        assert rendered.count("Captured fictional fact") == 1
        assert "Duplicate rebased fact" not in rendered and "Distinct fictional fact" not in rendered
        distinct["source_conversation_index"] = 501
        rendered = cross_history._format_all_chat_history_for_ai(
            current_history=[captured], cross_tail=[rebased, distinct], conversation_id=cid,
            soul_id="TestSoul", mark_current_chat=False, memory_context=True, background_context_rows=[],
        )
        assert rendered.count("Captured fictional fact") == 1
        assert "Duplicate rebased fact" not in rendered and "Distinct fictional fact" in rendered
    finally:
        con.close()


def test_scoped_background_context_keeps_two_souls_separate(tmp_path, monkeypatch):
    from app import main
    from tests.test_conversation_sources import _write_web_source_db

    home = tmp_path / "channels"
    home.mkdir()
    source_path = home / "web_source.db"
    first, second, excluded = [f"whatsapp:dm:199900000{suffix}@s.whatsapp.net" for suffix in (11, 12, 13)]
    write_policy(home, {
        first.removeprefix("whatsapp:dm:"): {"policy": "full", "memorize": False},
        second.removeprefix("whatsapp:dm:"): {"policy": "listen_only", "memorize": False},
        excluded.removeprefix("whatsapp:dm:"): {"policy": "excluded", "memorize": True},
    })
    _write_web_source_db(source_path, messages=[
        {"msg_key": f"physical-{index}", "timestamp": 100 + index, "body": body,
         "chat_id": cid.removeprefix("whatsapp:dm:"), "from_id": "19990000999@s.whatsapp.net"}
        for index, (cid, body) in enumerate([
            (first, "Older first source"), (first, "First Soul fresh tail"),
            (second, "Older second source"), (second, "Second Soul fresh tail"),
            (excluded, "Excluded fresh tail"),
        ], 1)
    ])
    monkeypatch.setattr(main, "_CONFIG", {**main._CONFIG, "hermes": {
        "home": str(home), "whatsapp_web_source_db": str(source_path),
        "sessions_index_path": str(home / "sessions.json"), "state_db_path": str(home / "state.db"),
    }})
    monkeypatch.setattr(main, "_get_service_from_payload", lambda *_a, **_k: pytest.fail("context reads need no service"))
    cases = [("FirstTestSoul", first, 1, "First Soul saved summary", "First Soul fresh tail", 2),
             ("SecondTestSoul", second, 3, "Second Soul saved summary", "Second Soul fresh tail", 4)]
    for sid, cid, cursor, summary, _, _ in cases:
        main._write_conversation_state(cid, user_id="TestOwner", soul_id=sid, updates={
            "memorize_chat": True, "rolling_summary_cursor_id": cursor,
            "rolling_summary": summary,
        })
        main._write_conversation_state(excluded, user_id="TestOwner", soul_id=sid, updates={
            "memorize_chat": True, "rolling_summary": "Excluded saved summary",
        })
    paths = [main._sqlite_current_path("TestOwner", sid) for sid, *_ in cases]
    assert paths[0] != paths[1] and all(path.is_relative_to(tmp_path) for path in paths)
    for sid, cid, _, summary, tail, rowid in cases:
        rows = main._load_cross_tail_for_ai(user_id="TestOwner", soul_id=sid, conversation_id="")
        assert [row["content"] for row in rows] == [tail]
        assert rows[0]["source_conversation_id"] == cid and rows[0]["memorize_chat"] is False
        assert rows[0]["source_conversation_index"] == rowid
        assert rows[0]["source_message_id"] == f"physical-{rowid}"
        with main._sqlite_connect(main._sqlite_current_path("TestOwner", sid)) as con:
            con.row_factory = sqlite3.Row
            summaries = main._read_background_rolling_summaries_from_conversations(con)
        assert set(summaries) == {cid} and summaries[cid]["summary"] == summary
        rendered = main._format_all_chat_history_for_ai(
            current_history=[], cross_tail=rows, conversation_id="chat:context-reader",
            user_id="TestOwner", soul_id=sid, user_name="TestOwner",
        )
        assert summary in rendered and tail in rendered and "Excluded" not in rendered
        for other_sid, _, _, other_summary, other_tail, _ in cases:
            if other_sid != sid:
                assert other_summary not in rendered and other_tail not in rendered
