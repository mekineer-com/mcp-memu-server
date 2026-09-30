import json
import sqlite3
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from memu.app.dossier import DossierRevisionStaleError

from app.db import json_to_db, normalize_text_list, sqlite_connect, sqlite_ensure_conversation_state_schema, sqlite_ensure_nonempty
from app.services import consolidation, segment
from app.services import soul_state as _soul_state
from app.services import soul_summaries as _soul_summaries
from app.services.consolidation import ConsolidationDeps, write_consolidation_outputs
from app.services.consolidation import _format_segment_memory_items_for_prompt
from app.services.consolidation import _format_episode_memories
from app.services.consolidation import _parse_weekly_reflection_xml
from app.services.consolidation import _remap_edges_with_memory_ids
from app.services.consolidation import consolidation_due
from app.services.consolidation import gather_consolidation_inputs
from app.services.consolidation import prepare_dossier_consolidation_context
from app.services.consolidation import preflight_consolidation_profiles
from app.services.consolidation import run_consolidation_llm
from app.services.graph_edges import invalidate_memory_edges, write_memory_edges
from app.services.state import conversation_state_from_row, conversation_state_row, write_conversation_state


class _DossierContextService:
    def __init__(self, *, due_ids=(), profiles=("default", "revision"), stale_id=None) -> None:
        self.memorize_config = SimpleNamespace(category_update_llm_profile="revision")
        self.llm_profiles = SimpleNamespace(
            profiles={name: SimpleNamespace(max_tokens=8000) for name in profiles}
        )
        self.due = [SimpleNamespace(id=dossier_id) for dossier_id in due_ids]
        self.stale_id = stale_id
        self.calls: list[tuple] = []
        self.prompts: list[str] = []

    def list_due_dossiers(self, scope, *, segment_ids):
        self.calls.append(("due", scope, segment_ids))
        return self.due

    def prepare_dossier_revision(self, dossier_id, scope, **context):
        self.calls.append(("prepare", dossier_id, scope, context))
        return {
            "dossier": SimpleNamespace(
                id=dossier_id, kind="topic", name=dossier_id.title(),
                description=f"{dossier_id} description", summary="## Current\nStable.",
            ),
            "target_words": 300,
            "cleanup_items": [], "cited_items": [], "pending_items": [],
            "candidate_items": [], "linked_item_ids": [],
            "linked_inactive_item_ids": [], "cited_unlinked_item_ids": [],
        }

    def prepare_anchor_revision(self, role, scope, actionable_ids):
        self.calls.append(("anchor", role, scope, actionable_ids))
        return {
            "dossier": SimpleNamespace(
                id=f"anchor-{role}", anchor_role=role, name=role,
                description=f"{role} description", summary="## Current\nStable.",
            ),
            "cited_items": [], "cleanup_items": [], "pending_items": [],
            "candidate_items": [], "linked_item_ids": [],
            "linked_inactive_item_ids": [], "actionable_item_ids": list(actionable_ids),
        }

    def prepare_anchor_continuity_context(self, _scope):
        return {
            "dossiers": [SimpleNamespace(
                name="Health", description="Current health", summary="Body [M4].",
                anchor_role=None,
            )],
            "episodes": [],
        }

    async def chat(self, prompt, **kwargs):
        step = kwargs["step"]
        self.calls.append(("chat", step))
        self.prompts.append(prompt)
        if step == "anchors":
            return """<identity_maintenance>
  <narrative_self action="keep"></narrative_self>
  <anchor_revisions>
    <anchor role="soul"><description>soul description</description><prose_action>keep</prose_action><prose_patches></prose_patches></anchor>
    <anchor role="user"><description>user description</description><prose_action>keep</prose_action><prose_patches></prose_patches></anchor>
  </anchor_revisions>
  <life_goals action="keep"><add></add><remove></remove></life_goals>
</identity_maintenance>"""
        if step == "weekly":
            return """<weekly_reflection><intentions/><edges></edges>
<companion_memory>I felt steady while looking back.</companion_memory></weekly_reflection>"""
        revisions = "".join(
            f'<dossier_revision dossier_id="{row.id}"><description>{row.id} description</description>'
            '<prose_action>keep</prose_action><prose_patches></prose_patches>'
            '<decisions></decisions></dossier_revision>'
            for row in self.due
        )
        return f"<dossier_revisions>{revisions}</dossier_revisions>"

    async def apply_dossier_revision(self, bundle, decision, scope):
        dossier_id = bundle["dossier"].id
        self.calls.append(("apply", dossier_id, decision, scope))
        if dossier_id == self.stale_id:
            raise DossierRevisionStaleError("changed")

    async def apply_anchor_revision(self, bundle, decision, scope):
        self.calls.append(("apply_anchor", decision["anchor_role"], scope))

    async def embed(self, texts, **_kwargs):
        return [[1.0] for _ in texts]

    def build_dossier_index(self, scope):
        self.calls.append(("index", scope))
        return "- Health: Current health"


@pytest.mark.parametrize(
    ("profiles", "consolidation_profile", "missing"),
    [(('default',), None, 'revision'), (('default', 'revision'), 'reflection', 'reflection')],
)
def test_consolidation_preflight_checks_both_profiles(profiles, consolidation_profile, missing) -> None:
    svc = _DossierContextService(due_ids=("first",), profiles=profiles)
    with pytest.raises(KeyError, match=missing):
        preflight_consolidation_profiles(svc, consolidation_profile)


def _inputs() -> dict:
    return {
        "narrative_self": "I am steady.",
        "active_life_goals": ["Stay curious"],
        "removed_life_goals": [],
        "selected_segment_ids": ["segment-2"],
        "state": {"intentions_active": []},
        "segment_inputs": [],
        "prior_context_memory_items": [],
        "existing_memory_edges": [],
        "all_chat_history": "A lived span.",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("due_ids", [("first",), ("first", "second"), ("first", "second", "third")])
async def test_dossier_context_uses_one_holistic_call_and_preserves_due_order(due_ids) -> None:
    svc = _DossierContextService(due_ids=due_ids)
    inputs = _inputs()
    result = await prepare_dossier_consolidation_context(
        svc, inputs=inputs, soul_id="TestSoul", user_id="TestUser"
    )
    assert result is inputs
    assert [call for call in svc.calls if call[0] == "chat"] == [("chat", "dossiers")]
    assert [call[1] for call in svc.calls if call[0] == "apply"] == list(due_ids)
    assert "Complete active life-domain dossiers" in svc.prompts[0]
    assert "Life-domain dossiers needing my care now" in svc.prompts[0]


@pytest.mark.asyncio
async def test_dossier_context_keeps_first_apply_when_second_is_stale() -> None:
    svc = _DossierContextService(due_ids=("first", "second"), stale_id="second")
    with pytest.raises(DossierRevisionStaleError):
        await prepare_dossier_consolidation_context(
            svc, inputs=_inputs(), soul_id="TestSoul", user_id="TestUser"
        )
    assert [call[1] for call in svc.calls if call[0] == "apply"] == ["first", "second"]


@pytest.mark.asyncio
async def test_consolidation_preflight_fails_before_paid_call(monkeypatch) -> None:
    monkeypatch.setattr(consolidation, "CONSOLIDATION_PROMPT_TOKEN_LIMIT", 10)
    svc = _DossierContextService(due_ids=("first",))
    with pytest.raises(ValueError, match="provider-safe"):
        await prepare_dossier_consolidation_context(
            svc, inputs=_inputs(), soul_id="TestSoul", user_id="TestUser"
        )
    assert not [call for call in svc.calls if call[0] in {"chat", "apply"}]


@pytest.mark.asyncio
async def test_consolidation_preflight_ignores_output_ceiling() -> None:
    svc = _DossierContextService(due_ids=("first",))
    for profile in svc.llm_profiles.profiles.values():
        profile.max_tokens = 1_000_000
    await prepare_dossier_consolidation_context(
        svc, inputs=_inputs(), soul_id="TestSoul", user_id="TestUser"
    )
    assert [call for call in svc.calls if call[0] == "chat"] == [
        ("chat", "dossiers")
    ]


@pytest.mark.asyncio
async def test_anchor_then_weekly_stages_apply_only_after_both_validate() -> None:
    svc = _DossierContextService()
    inputs = _inputs()
    await prepare_dossier_consolidation_context(
        svc, inputs=inputs, soul_id="TestSoul", user_id="TestUser"
    )
    out = await run_consolidation_llm(
        svc, inputs=inputs, soul_id="TestSoul", user_id="TestUser"
    )
    assert [call for call in svc.calls if call[0] == "chat"] == [
        ("chat", "anchors"), ("chat", "weekly")
    ]
    assert [call[1] for call in svc.calls if call[0] == "apply_anchor"] == ["soul", "user"]
    assert out["intentions_snapshot"] == []
    assert out["intentions_replacement"] == []
    assert "Description: soul description" in svc.prompts[0]


@pytest.mark.asyncio
async def test_short_embedding_response_fails_before_anchor_apply() -> None:
    svc = _DossierContextService()
    inputs = _inputs()
    await prepare_dossier_consolidation_context(
        svc, inputs=inputs, soul_id="TestSoul", user_id="TestUser"
    )

    async def short_embed(_texts, **_kwargs):
        return []

    svc.embed = short_embed
    with pytest.raises(ValueError, match="embedding count"):
        await run_consolidation_llm(
            svc, inputs=inputs, soul_id="TestSoul", user_id="TestUser"
        )
    assert not [call for call in svc.calls if call[0] == "apply_anchor"]


@pytest.mark.asyncio
async def test_weekly_edges_are_limited_to_full_supplied_memories() -> None:
    svc = _DossierContextService()
    original_chat = svc.chat

    async def chat(prompt, **kwargs):
        if kwargs["step"] == "weekly":
            svc.calls.append(("chat", "weekly"))
            svc.prompts.append(prompt)
            return """<weekly_reflection><intentions/><edges><edge>
<subject_id>[M1]</subject_id><predicate>parallels</predicate><object_id>[M2]</object_id><confidence>0.8</confidence>
</edge></edges><companion_memory>I noticed the echo.</companion_memory></weekly_reflection>"""
        return await original_chat(prompt, **kwargs)

    svc.chat = chat
    inputs = _inputs()
    inputs["prior_context_memory_items"] = [
        {"id": "older", "memory_ref": 1, "memory_type": "episode", "summary": "Older"}
    ]
    inputs["segment_inputs"] = [{
        "memory_summaries": [
            {"id": "newer", "memory_ref": 2, "memory_type": "episode", "summary": "Newer"}
        ]
    }]
    inputs["existing_memory_edges"] = [
        {"subject_id": "older", "predicate": "parallels", "object_id": "newer"}
    ]
    await prepare_dossier_consolidation_context(
        svc, inputs=inputs, soul_id="TestSoul", user_id="TestUser"
    )

    out = await run_consolidation_llm(
        svc, inputs=inputs, soul_id="TestSoul", user_id="TestUser"
    )

    assert out["edges"] == [{
        "subject_id": "older", "predicate": "parallels",
        "object_id": "newer", "confidence": 0.8,
    }]
    assert "[M1] parallels [M2]" in svc.prompts[-1]


@pytest.mark.asyncio
async def test_weekly_edge_outside_supplied_evidence_fails_before_anchor_apply() -> None:
    svc = _DossierContextService()
    original_chat = svc.chat

    async def chat(prompt, **kwargs):
        if kwargs["step"] == "weekly":
            return """<weekly_reflection><intentions/><edges><edge>
<subject_id>[M999]</subject_id><predicate>parallels</predicate><object_id>[M998]</object_id>
</edge></edges><companion_memory>I noticed an echo.</companion_memory></weekly_reflection>"""
        return await original_chat(prompt, **kwargs)

    svc.chat = chat
    inputs = _inputs()
    await prepare_dossier_consolidation_context(
        svc, inputs=inputs, soul_id="TestSoul", user_id="TestUser"
    )

    with pytest.raises(ValueError, match="outside supplied evidence"):
        await run_consolidation_llm(
            svc, inputs=inputs, soul_id="TestSoul", user_id="TestUser"
        )
    assert not [call for call in svc.calls if call[0] == "apply_anchor"]


def test_weekly_reflection_requires_complete_intention_list() -> None:
    with pytest.raises(ValueError, match="exact weekly_reflection"):
        _parse_weekly_reflection_xml(
            '<weekly_reflection version="1"><intentions/><edges></edges>'
            '<companion_memory>Present.</companion_memory></weekly_reflection>'
        )
    with pytest.raises(ValueError, match="requires intentions"):
        _parse_weekly_reflection_xml(
            "<weekly_reflection><edges></edges><companion_memory>Present.</companion_memory></weekly_reflection>"
        )
    assert _parse_weekly_reflection_xml(
        "<weekly_reflection><intentions/><edges></edges><companion_memory>Present.</companion_memory></weekly_reflection>"
    )["intentions"] == []


def test_episode_rendering_prefers_full_summary_and_falls_back_to_item() -> None:
    at = datetime(2026, 1, 1, tzinfo=UTC)
    rendered = _format_episode_memories([
        SimpleNamespace(
            memory_ref=1, summary="Compact", extra={"episode_summary": "Full episode"},
            happened_at=at, created_at=at,
        ),
        SimpleNamespace(
            memory_ref=2, summary="Fallback episode", extra={},
            happened_at=at, created_at=at,
        ),
    ])
    assert "[M1]" in rendered and "Full episode" in rendered and "Compact" not in rendered
    assert "[M2]" in rendered and "Fallback episode" in rendered

def test_format_segment_memory_items_for_prompt_shows_memory_ids() -> None:
    id_map: dict[str, str] = {}
    out = _format_segment_memory_items_for_prompt(
        [
            {
                "segment_id": "ep:1-2",
                "memory_summaries": [
                    {"id": "mem_1", "memory_ref": 7, "summary": "one", "memory_type": "behavior"},
                    {"id": "mem_2", "memory_ref": 9, "summary": "two", "memory_type": "knowledge"},
                ],
            }
        ],
        id_map,
    )
    assert "Key:" in out
    assert "Segment 1" not in out
    assert "Conversation " not in out
    assert "Related memories:" not in out
    assert "- [M7] [behavior] one" in out
    assert "- [M9] [knowledge] two" in out
    assert id_map == {"M7": "mem_1", "M9": "mem_2"}


def test_build_segment_inputs_dates_received_at_only_rows() -> None:
    messages = [{"role": "user", "content": "hi", "received_at": "2026-04-16T12:00:00Z"}]
    rows = segment.build_segment_inputs(messages, ["cid:0-0"])

    assert rows
    assert rows[0]["happened_at"] == datetime(2026, 4, 16, 12, 0, tzinfo=UTC)


def test_build_segment_inputs_rejects_range_past_stored_history() -> None:
    with pytest.raises(ValueError, match="segment range exceeds stored history"):
        segment.build_segment_inputs([{"content": "only row"}], ["cid:0-1"])


def test_consolidation_due_uses_last_success_clock() -> None:
    now = datetime(2026, 1, 8, tzinfo=UTC)

    assert consolidation_due(None, interval_days=7, now=now)
    assert not consolidation_due("2026-01-01T00:00:01+00:00", interval_days=7, now=now)
    assert consolidation_due("2026-01-01T00:00:00+00:00", interval_days=7, now=now)


def test_remap_edges_with_memory_ids_accepts_exact_prompt_refs() -> None:
    payload = [
        {"subject_id": "[M1]", "predicate": "parallels", "object_id": "[M2]", "confidence": 0.9},
        {"subject_id": "[M2]", "predicate": "evokes", "object_id": "[M1]"},
    ]
    mapped = _remap_edges_with_memory_ids(
        payload,
        id_map={"M1": "deadbeef", "M2": "cafebabe"},
        include_confidence=True,
    )
    assert mapped == [
        {"subject_id": "deadbeef", "predicate": "parallels", "object_id": "cafebabe", "confidence": 0.9},
        {"subject_id": "cafebabe", "predicate": "evokes", "object_id": "deadbeef"},
    ]


def test_remap_edges_with_memory_ids_rejects_unreviewed_raw_ids() -> None:
    payload = [{"subject_id": "deadbeef", "predicate": "shaped_by", "object_id": "cafebabe"}]
    mapped = _remap_edges_with_memory_ids(payload, id_map={}, include_confidence=False)
    assert mapped == []


def test_write_memory_edges_ignores_invalid_confidence() -> None:
    svc = _make_svc_stub()
    assert write_memory_edges(
        svc.database.triple_repo,
        [{"subject_id": "m1", "predicate": "evokes", "object_id": "m2", "confidence": 2}],
        scope={},
    ) == 0


def test_write_consolidation_outputs_consumes_each_conversation_snapshot() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        db_path = tmp_dir / "soul.db"
        con = sqlite3.connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            sqlite_ensure_conversation_state_schema(con)
        finally:
            con.close()

        cid = "conv-clear"
        other_cid = "conv-other"
        soul_id = "SoulX"
        user_id = "UserX"

        write_conversation_state(
            cid,
            sqlite_current_path=lambda _user, _soul: db_path,
            soul_id=soul_id,
            user_id=user_id,
            updates={
                "pending_segment_ids": ["ep:1-2", "ep:3-4"],
                "intentions_active": [{"id": "keep", "text": "Keep"}],
            },
        )
        write_conversation_state(
            other_cid,
            sqlite_current_path=lambda _user, _soul: db_path,
            soul_id=soul_id,
            user_id=user_id,
            updates={"pending_segment_ids": ["other:1-2", "other:new"]},
        )

        deps = ConsolidationDeps(
            sqlite_current_path=lambda _user, _soul: db_path,
            sqlite_ensure_nonempty=sqlite_ensure_nonempty,
            sqlite_connect=sqlite_connect,
            sqlite_ensure_conversation_state_schema=sqlite_ensure_conversation_state_schema,
            conversation_state_row=conversation_state_row,
            conversation_state_from_row=lambda row, **kw: conversation_state_from_row(row),
            write_conversation_state=lambda conversation_id, *, soul_id, user_id, updates, connection=None: write_conversation_state(
                conversation_id,
                sqlite_current_path=lambda _user, _soul: db_path,
                soul_id=soul_id,
                user_id=user_id,
                updates=updates,
                connection=connection,
            ),
            get_storage_dir=lambda _cfg: tmp_dir,
            config={},
            find_chat_dir_for_conversation=lambda _a, _b, _c, _d: None,
            read_list=lambda _p: [],
            normalize_text_list=normalize_text_list,
            json_to_db=json_to_db,
        )

        class _TripleRepoStub:
            def add(self, _triple, user_data=None):  # pragma: no cover - not used in this test
                return None

            def invalidate(self, _subject_id, _predicate, _object_id, scope=None):  # pragma: no cover - not used here
                return None

        class _SvcStub:
            def __init__(self) -> None:
                self.database = type("DB", (), {"triple_repo": _TripleRepoStub()})()

        result = write_consolidation_outputs(
            deps,
            _SvcStub(),
            inputs={
                "db_path": db_path,
                "selected_segment_ids": ["ep:1-2", "other:1-2"],
                "selected_segment_ids_by_conversation": {
                    cid: ["ep:1-2"],
                    other_cid: ["other:1-2"],
                },
            },
            llm_results={
                "narrative_self": None,
                "old_narrative_text": None,
                "old_narrative_embedding": None,
                "companion_memory": None,
                "companion_embedding": None,
                "life_goal_add": [],
                "life_goal_remove": [],
                "edges": [],
                "edge_invalidations": [],
                "intentions_snapshot": [
                    {"id": "keep", "text": "Keep"},
                    {"id": "done", "text": "Done"},
                ],
                "intentions_replacement": [
                    {"id": "done", "text": "Done"},
                    {"id": "new", "text": "New"},
                ],
            },
            conversation_id=cid,
            soul_id=soul_id,
            user_id=user_id,
        )

        assert result["consumed_segment_ids"] == ["ep:1-2", "other:1-2"]
        assert result["state"]["intentions_active"] == [{"id": "new", "text": "New"}]
        assert result["state"]["pending_segment_ids"] == ["ep:3-4"]
        other_state, _ = write_conversation_state(
            other_cid,
            sqlite_current_path=lambda _user, _soul: db_path,
            soul_id=soul_id,
            user_id=user_id,
            updates={},
        )
        assert other_state["pending_segment_ids"] == ["other:new"]


def test_gather_consolidation_inputs_skips_when_no_pending_segments() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        db_path = tmp_dir / "soul.db"
        con = sqlite3.connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            sqlite_ensure_conversation_state_schema(con)
        finally:
            con.close()

        cid = "conv-skip-empty-pending"
        soul_id = "SoulX"
        user_id = "UserX"

        write_conversation_state(
            cid,
            sqlite_current_path=lambda _user, _soul: db_path,
            soul_id=soul_id,
            user_id=user_id,
            updates={"pending_segment_ids": []},
        )

        deps = ConsolidationDeps(
            sqlite_current_path=lambda _user, _soul: db_path,
            sqlite_ensure_nonempty=sqlite_ensure_nonempty,
            sqlite_connect=sqlite_connect,
            sqlite_ensure_conversation_state_schema=sqlite_ensure_conversation_state_schema,
            conversation_state_row=conversation_state_row,
            conversation_state_from_row=lambda row, **kw: conversation_state_from_row(row),
            write_conversation_state=lambda conversation_id, *, soul_id, user_id, updates, connection=None: write_conversation_state(
                conversation_id,
                sqlite_current_path=lambda _user, _soul: db_path,
                soul_id=soul_id,
                user_id=user_id,
                updates=updates,
                connection=connection,
            ),
            get_storage_dir=lambda _cfg: tmp_dir,
            config={},
            find_chat_dir_for_conversation=lambda _a, _b, _c, _d: None,
            read_list=lambda _p: [],
            normalize_text_list=normalize_text_list,
            json_to_db=json_to_db,
        )

        out = gather_consolidation_inputs(
            deps,
            conversation_id=cid,
            soul_id=soul_id,
            user_id=user_id,
        )
        assert out == {"status": "skip", "reason": "no_pending_segments"}


def test_gather_consolidation_inputs_collects_all_pending_conversations(tmp_path: Path) -> None:
    db_path = tmp_path / "soul.db"
    con = sqlite3.connect(db_path)
    try:
        sqlite_ensure_conversation_state_schema(con)
        con.executescript(
            """
CREATE TABLE memory_items (
    id TEXT, memory_ref INTEGER, summary TEXT, memory_type TEXT,
    happened_at DATETIME, created_at DATETIME, soul_id TEXT, user_id TEXT,
    conversation_id TEXT, segment_id TEXT, merged_into TEXT
);
CREATE TABLE triples (
    subject_id TEXT, predicate TEXT, object_id TEXT, valid_to DATETIME
);
CREATE TABLE resources (
    soul_id TEXT, user_id TEXT, created_at DATETIME, memory_prior_context TEXT
);
"""
        )
        con.commit()
    finally:
        con.close()

    soul_id = "SoulX"
    user_id = "UserX"
    chat_dirs: dict[str, Path] = {}
    expected: dict[str, list[str]] = {}
    for index, cid in enumerate(("conv-a", "conv-b")):
        segment_id = f"{cid}:0-0"
        expected[cid] = [segment_id]
        write_conversation_state(
            cid,
            sqlite_current_path=lambda _user, _soul: db_path,
            soul_id=soul_id,
            user_id=user_id,
            updates={"pending_segment_ids": [segment_id]},
        )
        chat_dir = tmp_path / cid
        segments_dir = chat_dir / "segments"
        segments_dir.mkdir(parents=True)
        (chat_dir / "manifest.json").write_text(
            json.dumps({"segments": [{"start": 0, "end": 0}]}),
            encoding="utf-8",
        )
        (segments_dir / "segment_0.json").write_text(
            json.dumps([{
                "role": "user",
                "content": f"message {index}",
                "ts_ms": 1_767_225_600_000 + index,
                "source_conversation_id": cid,
            }]),
            encoding="utf-8",
        )
        chat_dirs[cid] = chat_dir

    con = sqlite3.connect(db_path)
    try:
        con.executemany(
            """
INSERT INTO memory_items (
    id, memory_ref, summary, memory_type, happened_at, created_at,
    soul_id, user_id, conversation_id, segment_id, merged_into
) VALUES (?, ?, ?, 'knowledge', ?, ?, ?, ?, 'conv-a', 'conv-a:0-0', NULL)
""",
            [
                (
                    f"mem-{index}",
                    index + 1,
                    f"memory {index}",
                    "2026-01-01T00:00:00+00:00",
                    f"2026-01-01T00:00:{index:02d}+00:00",
                    soul_id,
                    user_id,
                )
                for index in range(30)
            ],
        )
        con.executemany(
            """
INSERT INTO memory_items (
    id, memory_ref, summary, memory_type, happened_at, created_at,
    soul_id, user_id, conversation_id, segment_id, merged_into
) VALUES (?, ?, ?, 'knowledge', ?, ?, ?, ?, 'conv-a', 'conv-a:0-0', ?)
""",
            [
                (
                    "prior-merged",
                    101,
                    "merged prior",
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:01:01+00:00",
                    soul_id,
                    user_id,
                    "mem-0",
                ),
                (
                    "prior-evolved",
                    102,
                    "evolved prior",
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:01:02+00:00",
                    soul_id,
                    user_id,
                    None,
                ),
            ],
        )
        con.execute(
            "INSERT INTO triples (subject_id, predicate, object_id, valid_to) "
            "VALUES ('prior-evolved', 'evolved_into', 'mem-0', NULL)"
        )
        con.execute(
            "INSERT INTO resources (soul_id, user_id, created_at, memory_prior_context) "
            "VALUES (?, ?, ?, ?)",
            (
                soul_id,
                user_id,
                "2026-01-01 00:02:00.000000",
                json.dumps(["mem-0", "prior-merged", "prior-evolved"]),
            ),
        )
        con.commit()
    finally:
        con.close()

    con = sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        _soul_state.write(
            con,
            {"last_consolidation_at": "2026-01-01T00:01:00+00:00"},
        )
        con.commit()
    finally:
        con.close()

    deps = _make_consolidation_deps(db_path, tmp_path)
    deps = replace(
        deps,
        find_chat_dir_for_conversation=lambda _a, _b, _c, cid: chat_dirs.get(cid),
    )
    out = gather_consolidation_inputs(
        deps,
        conversation_id="conv-a",
        soul_id=soul_id,
        user_id=user_id,
    )

    assert out["selected_segment_ids_by_conversation"] == expected
    assert [row["conversation_id"] for row in out["segment_inputs"]] == ["conv-a", "conv-b"]
    assert [row["content"] for row in out["current_chat_messages"]] == ["message 0", "message 1"]
    assert len(out["segment_inputs"][0]["memory_summaries"]) == 30
    assert [item["id"] for item in out["prior_context_memory_items"]] == ["mem-0"]

    con = sqlite_connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        _soul_state.write(
            con,
            {
                "last_consolidation_error": "RuntimeError: failed",
                "last_consolidation_error_at": datetime.now(UTC).isoformat(),
            },
        )
        con.commit()
    finally:
        con.close()

    blocked = gather_consolidation_inputs(
        deps,
        conversation_id="conv-a",
        soul_id=soul_id,
        user_id=user_id,
    )
    assert blocked == {"status": "skip", "reason": "failure_requires_retry"}

    forced = gather_consolidation_inputs(
        deps,
        conversation_id="conv-a",
        soul_id=soul_id,
        user_id=user_id,
        force=True,
    )
    assert forced["status"] == "ready"

    write_conversation_state(
        "conv-a",
        sqlite_current_path=lambda _user, _soul: db_path,
        soul_id=soul_id,
        user_id=user_id,
        updates={"append_pending_segment_ids": ["conv-a:1-1"]},
    )
    changed_pending = gather_consolidation_inputs(
        deps,
        conversation_id="conv-a",
        soul_id=soul_id,
        user_id=user_id,
    )
    assert changed_pending == {"status": "skip", "reason": "failure_requires_retry"}


def test_gather_rejects_noncanonical_pending_owner(tmp_path: Path) -> None:
    db_path = tmp_path / "soul.db"
    con = sqlite_connect(db_path)
    try:
        sqlite_ensure_conversation_state_schema(con)
        con.execute(
            "INSERT INTO conversations "
            "(conversation_id, soul_id, user_id, pending_segment_ids) "
            "VALUES (?, ?, ?, ?)",
            (
                "whatsapp:group:123@g.us:sender",
                "SoulX",
                "UserX",
                json.dumps(["whatsapp:group:123@g.us:sender:0-1"]),
            ),
        )
        con.commit()
    finally:
        con.close()
    write_conversation_state(
        "trigger",
        sqlite_current_path=lambda _user, _soul: db_path,
        soul_id="SoulX",
        user_id="UserX",
        updates={},
    )

    with pytest.raises(HTTPException, match="pending consolidation owner is not canonical"):
        gather_consolidation_inputs(
            _make_consolidation_deps(db_path, tmp_path),
            conversation_id="trigger",
            soul_id="SoulX",
            user_id="UserX",
        )


@pytest.mark.parametrize(
    ("bad_content", "segment_end", "error_text"),
    [
        ("{not json", 0, "segment history unreadable"),
        (json.dumps([{"role": "user", "content": "valid"}, 7]), 0, "non-message row"),
        (json.dumps([{"role": "user", "content": "only row"}]), 1, "range exceeds stored history"),
    ],
)
def test_gather_consolidation_inputs_rejects_invalid_segment_file(
    tmp_path: Path,
    bad_content: str,
    segment_end: int,
    error_text: str,
) -> None:
    db_path = tmp_path / "soul.db"
    con = sqlite3.connect(db_path)
    try:
        sqlite_ensure_conversation_state_schema(con)
        con.executescript(
            """
CREATE TABLE memory_items (
    id TEXT, memory_ref INTEGER, summary TEXT, memory_type TEXT,
    happened_at DATETIME, created_at DATETIME, soul_id TEXT, user_id TEXT,
    conversation_id TEXT, segment_id TEXT, merged_into TEXT
);
CREATE TABLE triples (
    subject_id TEXT, predicate TEXT, object_id TEXT, valid_to DATETIME
);
CREATE TABLE resources (
    soul_id TEXT, user_id TEXT, created_at DATETIME, memory_prior_context TEXT
);
"""
        )
        con.commit()
    finally:
        con.close()

    cid = "conv-bad-file"
    soul_id = "SoulX"
    user_id = "UserX"
    segment_id = f"{cid}:0-{segment_end}"
    write_conversation_state(
        cid,
        sqlite_current_path=lambda _user, _soul: db_path,
        soul_id=soul_id,
        user_id=user_id,
        updates={"pending_segment_ids": [segment_id]},
    )
    chat_dir = tmp_path / cid
    segments_dir = chat_dir / "segments"
    segments_dir.mkdir(parents=True)
    (chat_dir / "manifest.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": segment_end}]}),
        encoding="utf-8",
    )
    bad_file = segments_dir / "segment_0.json"
    bad_file.write_text(bad_content, encoding="utf-8")

    deps = replace(
        _make_consolidation_deps(db_path, tmp_path),
        find_chat_dir_for_conversation=lambda _a, _b, _c, _d: chat_dir,
    )
    with pytest.raises(HTTPException, match=error_text) as exc_info:
            gather_consolidation_inputs(
                deps,
                conversation_id=cid,
                soul_id=soul_id,
                user_id=user_id,
            )

    expected_identifier = segment_id if "range exceeds" in error_text else str(bad_file)
    assert expected_identifier in str(exc_info.value.detail)


def _make_consolidation_deps(db_path: Path, tmp_dir: Path) -> ConsolidationDeps:
    """Helper: wires up a ConsolidationDeps pointing at a single db_path."""
    return ConsolidationDeps(
        sqlite_current_path=lambda _user, _soul: db_path,
        sqlite_ensure_nonempty=sqlite_ensure_nonempty,
        sqlite_connect=sqlite_connect,
        sqlite_ensure_conversation_state_schema=sqlite_ensure_conversation_state_schema,
        conversation_state_row=conversation_state_row,
        conversation_state_from_row=lambda row, **kw: conversation_state_from_row(row),
        write_conversation_state=lambda cid, *, soul_id, user_id, updates, connection=None: write_conversation_state(
            cid,
            sqlite_current_path=lambda _u, _s: db_path,
            soul_id=soul_id,
            user_id=user_id,
            updates=updates,
            connection=connection,
        ),
        get_storage_dir=lambda _cfg: tmp_dir,
        config={},
        find_chat_dir_for_conversation=lambda _a, _b, _c, _d: None,
        read_list=lambda _p: [],
        normalize_text_list=normalize_text_list,
        json_to_db=json_to_db,
    )


def _make_svc_stub() -> object:
    class _TripleRepo:
        def add(self, _t, user_data=None): return None
        def invalidate(self, _s, _p, _o, scope=None): return None

    class _SvcStub:
        def __init__(self) -> None:
            self.database = type("DB", (), {"triple_repo": _TripleRepo()})()

    return _SvcStub()


def _base_llm_results(**overrides) -> dict:
    base = {
        "narrative_self": None,
        "old_narrative_text": None,
        "old_narrative_embedding": None,
        "companion_memory": None,
        "companion_embedding": None,
        "life_goal_add": [],
        "life_goal_remove": [],
        "edges": [],
        "edge_invalidations": [],
        "intentions_snapshot": [],
        "intentions_replacement": [],
    }
    base.update(overrides)
    return base


def test_write_consolidation_outputs_subtracts_gathered_accumulators() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        db_path = tmp_dir / "soul.db"
        con = sqlite3.connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            sqlite_ensure_conversation_state_schema(con)
            _soul_state.ensure_schema(con)
            con.commit()
        finally:
            con.close()

        cid = "conv-accum"
        soul_id = "SoulA"
        user_id = "UserA"

        # Seed some accumulator ids
        write_conversation_state(
            cid,
            sqlite_current_path=lambda _u, _s: db_path,
            soul_id=soul_id,
            user_id=user_id,
            updates={
                "pending_segment_ids": ["ep:1"],
                "intentions_active": [],
                "append_retrieval_ids_since_consolidation": ["mem-r1", "mem-r2"],
                "append_prior_context_ids_since_consolidation": ["mem-p1"],
            },
        )

        # Verify they were stored
        check_con = sqlite_connect(db_path)
        check_con.row_factory = sqlite3.Row
        _soul_state.write(
            check_con,
            {
                "last_consolidation_error": "RuntimeError: failed",
                "last_consolidation_error_at": "2026-01-01T00:00:00+00:00",
            },
        )
        check_con.commit()
        ss_before = _soul_state.read(check_con)
        check_con.close()
        assert ss_before["retrieval_ids_since_consolidation"] == ["mem-r1", "mem-r2"]
        assert ss_before["prior_context_ids_since_consolidation"] == ["mem-p1"]

        write_conversation_state(
            cid,
            sqlite_current_path=lambda _u, _s: db_path,
            soul_id=soul_id,
            user_id=user_id,
            updates={
                "append_retrieval_ids_since_consolidation": ["mem-r3"],
                "append_prior_context_ids_since_consolidation": ["mem-p2"],
            },
        )
        started_at = "2026-01-02T00:00:00+00:00"

        write_consolidation_outputs(
            _make_consolidation_deps(db_path, tmp_dir),
            _make_svc_stub(),
            inputs={
                "db_path": db_path,
                "started_at": started_at,
                "state": {
                    "retrieval_ids_since_consolidation": ["mem-r1", "mem-r2"],
                    "prior_context_ids_since_consolidation": ["mem-p1"],
                },
            },
            llm_results=_base_llm_results(),
            conversation_id=cid,
            soul_id=soul_id,
            user_id=user_id,
        )

        check_con2 = sqlite_connect(db_path)
        check_con2.row_factory = sqlite3.Row
        ss_after = _soul_state.read(check_con2)
        check_con2.close()
        assert ss_after["retrieval_ids_since_consolidation"] == ["mem-r3"]
        assert ss_after["prior_context_ids_since_consolidation"] == ["mem-p2"]
        assert ss_after["last_consolidation_at"] == started_at
        assert ss_after["last_consolidation_error"] is None
        assert ss_after["last_consolidation_error_at"] is None


def test_write_consolidation_outputs_uses_life_goals_table() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        db_path = tmp_dir / "soul.db"
        con = sqlite3.connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            sqlite_ensure_conversation_state_schema(con)
            _soul_state.ensure_schema(con)
            con.execute(
                "INSERT INTO life_goals (id, soul_id, user_id, description, status) VALUES (?, ?, ?, ?, 'active')",
                ("goal-old", "SoulLG", "UserLG", "old goal",),
            )
            con.execute(
                "INSERT INTO life_goals (id, soul_id, user_id, description, status) VALUES (?, ?, ?, ?, 'removed')",
                ("goal-return", "SoulLG", "UserLG", "return goal",),
            )
            con.commit()
        finally:
            con.close()

        write_conversation_state(
            "conv-life-goals",
            sqlite_current_path=lambda _u, _s: db_path,
            soul_id="SoulLG",
            user_id="UserLG",
            updates={"pending_segment_ids": ["ep:1"], "intentions_active": []},
        )

        write_consolidation_outputs(
            _make_consolidation_deps(db_path, tmp_dir),
            _make_svc_stub(),
            inputs={"db_path": db_path},
            llm_results=_base_llm_results(
                life_goal_remove=["old goal"],
            life_goal_add=["new goal", "return goal"],
            ),
            conversation_id="conv-life-goals",
            soul_id="SoulLG",
            user_id="UserLG",
        )

        check_con = sqlite_connect(db_path)
        try:
            rows = check_con.execute(
                "SELECT description, status FROM life_goals WHERE soul_id = ? AND user_id = ? ORDER BY description",
                ("SoulLG", "UserLG"),
            ).fetchall()
            old_rows = check_con.execute(
                "SELECT description FROM intentions WHERE source = 'life_goal'"
            ).fetchall()
        finally:
            check_con.close()

        assert [(row[0], row[1]) for row in rows] == [
            ("new goal", "active"),
            ("old goal", "removed"),
            ("return goal", "active"),
        ]
        assert old_rows == []


def test_consolidation_rejects_a_concurrent_narrative_edit() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        db_path = tmp_dir / "soul.db"
        con = sqlite_connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            sqlite_ensure_conversation_state_schema(con)
            _soul_summaries.write_live(
                con,
                kind="narrative_self",
                summary="Gathered story.",
                scope={"user_id": "UserN", "soul_id": "SoulN"},
                edited_by="setup",
                journal=False,
            )
            con.commit()
            gathered = _soul_state.read(con)
            _soul_summaries.write_live(
                con,
                kind="narrative_self",
                summary="Atomic edit.",
                scope={"user_id": "UserN", "soul_id": "SoulN"},
                edited_by="user",
                expected_revision=gathered["summaries_revision"],
                displayed_summary="Gathered story.",
                journal=False,
            )
            con.commit()
        finally:
            con.close()
        write_conversation_state(
            "conv-narrative",
            sqlite_current_path=lambda _u, _s: db_path,
            soul_id="SoulN",
            user_id="UserN",
            updates={"pending_segment_ids": ["ep:1"], "intentions_active": []},
        )

        with pytest.raises(ValueError, match="summary_snapshot_stale"):
            write_consolidation_outputs(
                _make_consolidation_deps(db_path, tmp_dir),
                _make_svc_stub(),
                inputs={
                    "db_path": db_path,
                    "narrative_self": "Gathered story.",
                    "state": gathered,
                },
                llm_results=_base_llm_results(
                    narrative_self="Consolidated story.",
                    old_narrative_text="Gathered story.",
                    old_narrative_embedding=[1.0],
                ),
                conversation_id="conv-narrative",
                soul_id="SoulN",
                user_id="UserN",
            )

        check = sqlite_connect(db_path)
        try:
            check.row_factory = sqlite3.Row
            assert _soul_state.read(check)["narrative_self"] == "Atomic edit."
        finally:
            check.close()


def test_write_consolidation_outputs_state_preflight_failure_produces_no_companion_memory() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        db_path = tmp_dir / "soul.db"
        con = sqlite3.connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            sqlite_ensure_conversation_state_schema(con)
            _soul_state.ensure_schema(con)
            con.commit()
        finally:
            con.close()

        cid = "conv-db-fail"
        soul_id = "SoulB"
        user_id = "UserB"

        write_conversation_state(
            cid,
            sqlite_current_path=lambda _u, _s: db_path,
            soul_id=soul_id,
            user_id=user_id,
            updates={"pending_segment_ids": ["ep:2"], "intentions_active": []},
        )

        companion_calls: list[str] = []

        def _fake_write_state(cid, *, soul_id, user_id, updates, connection=None):
            raise RuntimeError("simulated DB failure")

        failing_deps = ConsolidationDeps(
            sqlite_current_path=lambda _u, _s: db_path,
            sqlite_ensure_nonempty=sqlite_ensure_nonempty,
            sqlite_connect=sqlite_connect,
            sqlite_ensure_conversation_state_schema=sqlite_ensure_conversation_state_schema,
            conversation_state_row=conversation_state_row,
            conversation_state_from_row=lambda row, **kw: conversation_state_from_row(row),
            write_conversation_state=_fake_write_state,
            get_storage_dir=lambda _cfg: tmp_dir,
            config={},
            find_chat_dir_for_conversation=lambda _a, _b, _c, _d: None,
            read_list=lambda _p: [],
            normalize_text_list=normalize_text_list,
            json_to_db=json_to_db,
        )

        import app.services.consolidation as _consol_mod

        original_create = _consol_mod.create_companion_memory

        def _tracking_create(*args, **kwargs):
            companion_calls.append("called")
            return original_create(*args, **kwargs)

        _consol_mod.create_companion_memory = _tracking_create
        try:
            with pytest.raises(RuntimeError, match="simulated DB failure"):
                write_consolidation_outputs(
                    failing_deps,
                    _make_svc_stub(),
                    inputs={"db_path": db_path},
                    llm_results=_base_llm_results(
                        companion_memory="Something to remember.",
                        companion_embedding=[0.1, 0.2, 0.3],
                    ),
                    conversation_id=cid,
                    soul_id=soul_id,
                    user_id=user_id,
                )
        finally:
            _consol_mod.create_companion_memory = original_create

        assert companion_calls == [], "companion memory must not be created when DB phase fails"


def test_write_consolidation_outputs_late_failure_keeps_pending_segment_ids() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        db_path = tmp_dir / "soul.db"
        con = sqlite3.connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            sqlite_ensure_conversation_state_schema(con)
            _soul_state.ensure_schema(con)
            con.commit()
        finally:
            con.close()

        cid = "conv-late-fail"
        soul_id = "SoulC"
        user_id = "UserC"
        pending = ["ep:late"]

        write_conversation_state(
            cid,
            sqlite_current_path=lambda _u, _s: db_path,
            soul_id=soul_id,
            user_id=user_id,
            updates={"pending_segment_ids": pending, "intentions_active": []},
        )

        import app.services.consolidation as _consol_mod

        original_create = _consol_mod.create_companion_memory

        def _failing_create(*args, **kwargs):
            raise RuntimeError("simulated companion failure")

        _consol_mod.create_companion_memory = _failing_create
        try:
            with pytest.raises(RuntimeError, match="simulated companion failure"):
                write_consolidation_outputs(
                    _make_consolidation_deps(db_path, tmp_dir),
                    _make_svc_stub(),
                    inputs={"db_path": db_path},
                    llm_results=_base_llm_results(
                        companion_memory="Something to remember.",
                        companion_embedding=[0.1, 0.2, 0.3],
                    ),
                    conversation_id=cid,
                    soul_id=soul_id,
                    user_id=user_id,
                )
        finally:
            _consol_mod.create_companion_memory = original_create

        check_con = sqlite_connect(db_path)
        try:
            check_con.row_factory = sqlite3.Row
            state = conversation_state_from_row(conversation_state_row(check_con, cid))
        finally:
            check_con.close()

        assert state is not None
        assert state["pending_segment_ids"] == pending


def test_write_consolidation_outputs_rolls_back_final_transaction() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        db_path = tmp_dir / "soul.db"
        con = sqlite_connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            sqlite_ensure_conversation_state_schema(con)
            _soul_state.ensure_schema(con)
            con.commit()
        finally:
            con.close()

        cid = "conv-final"
        other_cid = "conv-other"
        soul_id = "SoulFinal"
        user_id = "UserFinal"
        for owner, pending in ((cid, ["final:0-1"]), (other_cid, ["other:0-1"])):
            write_conversation_state(
                owner,
                sqlite_current_path=lambda _u, _s: db_path,
                soul_id=soul_id,
                user_id=user_id,
                updates={"pending_segment_ids": pending, "intentions_active": []},
            )

        deps = _make_consolidation_deps(db_path, tmp_dir)
        real_write = deps.write_conversation_state

        def fail_after_owner_write(
            owner,
            *,
            soul_id,
            user_id,
            updates,
            connection=None,
        ):
            result = real_write(
                owner,
                soul_id=soul_id,
                user_id=user_id,
                updates=updates,
                connection=connection,
            )
            if owner == cid and connection is not None:
                raise RuntimeError("simulated finalization failure")
            return result

        failing_deps = replace(deps, write_conversation_state=fail_after_owner_write)
        with pytest.raises(RuntimeError, match="simulated finalization failure"):
            write_consolidation_outputs(
                failing_deps,
                _make_svc_stub(),
                inputs={
                    "db_path": db_path,
                    "narrative_self": None,
                    "state": {"narrative_self": None, "summaries_revision": 0},
                    "selected_segment_ids": ["final:0-1", "other:0-1"],
                    "selected_segment_ids_by_conversation": {
                        cid: ["final:0-1"],
                        other_cid: ["other:0-1"],
                    },
                },
                llm_results=_base_llm_results(narrative_self="A new self."),
                conversation_id=cid,
                soul_id=soul_id,
                user_id=user_id,
            )

        con = sqlite_connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            states = {
                owner: conversation_state_from_row(conversation_state_row(con, owner))
                for owner in (cid, other_cid)
            }
            soul = _soul_state.read(con)
            narrative_count = con.execute("SELECT COUNT(*) FROM narrative_history").fetchone()[0]
        finally:
            con.close()

        assert states[cid]["pending_segment_ids"] == ["final:0-1"]
        assert states[other_cid]["pending_segment_ids"] == ["other:0-1"]
        assert soul["narrative_self"] is None
        assert soul["last_consolidation_at"] is None
        assert narrative_count == 0
