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
            soul_id="TestSoul", mark_current_chat=False, background_context_rows=[],
        )
        assert rendered.count("Captured fictional fact") == 1
        assert "Duplicate rebased fact" not in rendered and "Distinct fictional fact" not in rendered
        distinct["source_conversation_index"] = 501
        rendered = cross_history._format_all_chat_history_for_ai(
            current_history=[captured], cross_tail=[rebased, distinct], conversation_id=cid,
            soul_id="TestSoul", mark_current_chat=False, background_context_rows=[],
        )
        assert rendered.count("Captured fictional fact") == 1
        assert "Duplicate rebased fact" not in rendered and "Distinct fictional fact" in rendered
    finally:
        con.close()
