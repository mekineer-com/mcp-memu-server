import asyncio
import json
import sqlite3
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from memu.app import MemoryService
from memu.app.dossier import DossierRevisionStaleError
from pydantic import BaseModel
from sqlalchemy import event
from sqlalchemy.pool import NullPool
from sqlmodel import Session, create_engine

from app.db import json_to_db, normalize_text_list, sqlite_connect, sqlite_ensure_conversation_state_schema, sqlite_ensure_nonempty
from app.services import consolidation, cross_history, message_log, payload, segment, turn_contract
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
            profiles={name: SimpleNamespace(max_tokens=8000, context_window_tokens=1_000_000, chat_model=name) for name in profiles}
        )
        self.due = [SimpleNamespace(id=dossier_id) for dossier_id in due_ids]
        self.stale_id = stale_id
        self.calls: list[tuple] = []
        self.prompts: list[str] = []

    def list_due_dossiers(self, scope, *, segment_ids=None, excluded_segment_ids=()):
        self.calls.append(("due", scope, segment_ids))
        self.excluded_segment_ids = list(excluded_segment_ids)
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

    async def prepare_anchor_revision_apply(self, bundle, decision, scope):
        self.calls.append(("prepare_anchor", decision["anchor_role"], scope))
        return {"scope": scope}

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
@pytest.mark.parametrize("over_budget", [False, True])
async def test_dry_prompt_preparation_is_reused_without_model_calls_or_rebuilding(monkeypatch, over_budget):
    svc = _DossierContextService(due_ids=("first", "second"))
    if over_budget:
        svc.llm_profiles.profiles["revision"].context_window_tokens = 8010
    inputs = _inputs()
    prepared = consolidation._prepare_dossier_consolidation_prompts(
        svc, inputs=inputs, soul_id="TestSoul", user_id="TestUser",
    )
    assert not [call for call in svc.calls if call[0] in {"chat", "apply", "prepare_anchor"}]
    assert set(prepared[3]) == {"dossiers", "anchors", "weekly"}
    assert prepared[3]["dossiers"] == consolidation.estimate_prompt_tokens(prepared[1] + "\n" + prepared[2])
    if over_budget:
        with pytest.raises(ValueError, match="provider-safe"):
            await prepare_dossier_consolidation_context(
                svc, inputs=inputs, soul_id="TestSoul", user_id="TestUser", prepared=prepared,
            )
        assert not [call for call in svc.calls if call[0] in {"chat", "apply", "prepare_anchor"}]
        return
    await prepare_dossier_consolidation_context(
        svc, inputs=inputs, soul_id="TestSoul", user_id="TestUser", prepared=prepared,
    )
    assert svc.prompts == [prepared[2]]
    assert len([call for call in svc.calls if call[0] == "due"]) == 1
    assert len([call for call in svc.calls if call[0] == "prepare"]) == 2


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
    svc = _DossierContextService(due_ids=("first",))
    svc.llm_profiles.profiles["revision"].context_window_tokens = 8010
    with pytest.raises(ValueError, match="provider-safe"):
        await prepare_dossier_consolidation_context(
            svc, inputs=_inputs(), soul_id="TestSoul", user_id="TestUser"
        )
    assert not [call for call in svc.calls if call[0] in {"chat", "apply"}]


@pytest.mark.asyncio
async def test_consolidation_preflight_reserves_output_ceiling() -> None:
    svc = _DossierContextService(due_ids=("first",))
    for profile in svc.llm_profiles.profiles.values():
        profile.max_tokens = 1_000_000
    with pytest.raises(ValueError, match="must exceed max_tokens"):
        await prepare_dossier_consolidation_context(
            svc, inputs=_inputs(), soul_id="TestSoul", user_id="TestUser"
        )
    assert not [call for call in svc.calls if call[0] == "chat"]


@pytest.mark.asyncio
async def test_anchor_then_weekly_stages_prepare_writes_only_after_both_validate() -> None:
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
    assert [call[1] for call in svc.calls if call[0] == "prepare_anchor"] == ["soul", "user"]
    assert out["intentions_snapshot"] == []
    assert out["intentions_replacement"] == []
    assert "Description: soul description" in svc.prompts[0]


@pytest.mark.asyncio
async def test_historical_stages_skip_weekly_but_embed_previous_narrative(monkeypatch) -> None:
    svc = _DossierContextService(due_ids=("first",))
    inputs = {**_inputs(), "historical": True}
    monkeypatch.setattr(consolidation, "_render_weekly_prompt",
                        lambda *_a, **_kw: pytest.fail("historical work must not render weekly"))
    original_chat = svc.chat
    async def chat(prompt, **kwargs):
        raw = await original_chat(prompt, **kwargs)
        return raw.replace('<narrative_self action="keep"></narrative_self>',
                           '<narrative_self action="replace">A broader self.</narrative_self>')
    embedded = []
    async def embed(texts, **_kwargs):
        embedded.extend(texts)
        return [[1.0] for _ in texts]
    svc.chat, svc.embed = chat, embed
    prepared = consolidation._prepare_dossier_consolidation_prompts(
        svc, inputs=inputs, soul_id="TestSoul", user_id="TestUser",
    )
    assert prepared[3]["weekly"] == 0
    assert next(call for call in svc.calls if call[0] == "prepare")[3]["segment_ids"] == ["segment-2"]
    await prepare_dossier_consolidation_context(
        svc, inputs=inputs, soul_id="TestSoul", user_id="TestUser", prepared=prepared,
    )
    result = await run_consolidation_llm(svc, inputs=inputs, soul_id="TestSoul", user_id="TestUser")
    assert [call for call in svc.calls if call[0] == "chat"] == [("chat", "dossiers"), ("chat", "anchors")]
    assert embedded == ["I am steady."]
    assert result["old_narrative_embedding"] == [1.0]
    assert not {"edges", "intentions_replacement", "companion_memory"} & result.keys()


@pytest.mark.asyncio
@pytest.mark.parametrize("growth_stage", ["anchors", "weekly"])
async def test_later_stage_rechecks_after_valid_prose_growth(growth_stage):
    svc = _DossierContextService(due_ids=("first",))
    stored = {"summary": "## Current\nStable.", "description": "first description"}
    svc.prepare_anchor_continuity_context = lambda _scope: {
        "dossiers": [SimpleNamespace(name="First", anchor_role=None, **stored)], "episodes": [],
    }
    svc.build_dossier_index = lambda _scope: "- First: first description"
    inputs = _inputs()
    prepared = consolidation._prepare_dossier_consolidation_prompts(
        svc, inputs=inputs, soul_id="TestSoul", user_id="TestUser",
    )
    budget = max(prepared[3]["anchors"], prepared[3]["weekly"]) + 100
    svc.llm_profiles.profiles["default"].context_window_tokens = 8000 + budget * 5 // 4 + 1
    grown = "A broader self. " * 1000
    original_chat, original_apply = svc.chat, svc.apply_dossier_revision
    async def chat(prompt, **kwargs):
        raw = await original_chat(prompt, **kwargs)
        if growth_stage == "anchors" and kwargs["step"] == "dossiers":
            raw = raw.replace("<prose_action>keep</prose_action><prose_patches></prose_patches>",
                '<prose_action>patch</prose_action><prose_patches><section ref="S1" action="replace">'
                f"## Current\n{grown}</section></prose_patches>")
        if growth_stage == "weekly" and kwargs["step"] == "anchors":
            raw = raw.replace('<narrative_self action="keep"></narrative_self>',
                              f'<narrative_self action="replace">{grown}</narrative_self>')
        assert consolidation.estimate_prompt_tokens(raw) < 8000
        return raw
    async def apply(bundle, decision, scope):
        await original_apply(bundle, decision, scope)
        stored["summary"] = decision["resulting_prose"]
    svc.chat, svc.apply_dossier_revision = chat, apply
    await prepare_dossier_consolidation_context(
        svc, inputs=inputs, soul_id="TestSoul", user_id="TestUser", prepared=prepared,
    )
    with pytest.raises(ValueError, match=f"{growth_stage} consolidation prompt"):
        await run_consolidation_llm(svc, inputs=inputs, soul_id="TestSoul", user_id="TestUser")
    expected = [("chat", "dossiers")] + ([("chat", "anchors")] if growth_stage == "weekly" else [])
    assert [call for call in svc.calls if call[0] == "chat"] == expected
    assert len([call for call in svc.calls if call[0] == "apply"]) == 1
    assert not [call for call in svc.calls if call[0] == "prepare_anchor"]


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
    assert not [call for call in svc.calls if call[0] == "prepare_anchor"]


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
    assert not [call for call in svc.calls if call[0] == "prepare_anchor"]


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

        result = write_consolidation_outputs(
            deps,
            _make_svc_stub(db_path),
            inputs={
                "db_path": db_path,
                "selected_segment_ids": ["ep:1-2", "other:1-2"],
                "selected_segment_ids_by_conversation": {
                    cid: ["ep:1-2"],
                    other_cid: ["other:1-2"],
                },
            },
            llm_results={
                "anchor_writes": [],
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


@pytest.mark.parametrize("ambiguous_path", [False, True])
def test_gather_consolidation_inputs_collects_all_pending_conversations(tmp_path: Path, ambiguous_path, monkeypatch) -> None:
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
    soul_id TEXT, user_id TEXT, created_at DATETIME, memory_prior_context TEXT,
    conversation_id TEXT, segment_id TEXT, local_path TEXT, modality TEXT
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
                "conversation_id": cid,
            }]),
            encoding="utf-8",
        )
        chat_dirs[cid] = chat_dir
        with sqlite3.connect(db_path) as resource_con:
            resource_con.execute(
                "INSERT INTO resources (soul_id, user_id, conversation_id, segment_id, local_path, modality) "
                "VALUES (?, ?, ?, ?, ?, 'conversation')",
                (soul_id, user_id, cid, segment_id, str(segments_dir / "segment_0.json")),
            )

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
            "INSERT INTO resources (soul_id, user_id, created_at, memory_prior_context, "
            "conversation_id, segment_id, local_path, modality) VALUES (?, ?, ?, ?, ?, ?, ?, 'conversation')",
            (
                soul_id,
                user_id,
                "2026-01-01 00:02:00.000000",
                json.dumps(["mem-0", "prior-merged", "prior-evolved"]),
                "conv-a", "conv-a:0-0", str(chat_dirs["conv-a"] / "segments" / "segment_0.json"),
            ),
        )
        con.commit()
    finally:
        con.close()

    historical_id = "conv-a:1-1"
    historical_file = chat_dirs["conv-a"] / "segments" / "2020-01-01.json"
    historical_file.write_text(json.dumps([{
        "role": "user", "content": "older imported history", "ts_ms": 1_577_836_800_000,
        "source_conversation_id": "conv-a",
        "conversation_id": "conv-a",
    }]), encoding="utf-8")
    expected["conv-a"].append(historical_id)
    write_conversation_state(
        "conv-a", sqlite_current_path=lambda *_: db_path, soul_id=soul_id, user_id=user_id,
        updates={"append_pending_segment_ids": [historical_id]},
    )
    with sqlite3.connect(db_path) as con:
        con.executemany(
            "INSERT INTO resources (soul_id, user_id, conversation_id, segment_id, local_path, modality) "
            "VALUES (?, ?, 'conv-a', ?, ?, 'conversation')",
            [
                (soul_id, user_id, historical_id, str(historical_file)),
                ("OtherSoul", user_id, "conv-a:0-0", str(historical_file)),
                (soul_id, "OtherOwner", "conv-a:0-0", str(historical_file)),
                (soul_id, user_id, "conv-a:2-2", str(historical_file)),
            ],
        )
        if ambiguous_path:
            con.execute(
                "INSERT INTO resources (soul_id, user_id, conversation_id, segment_id, local_path, modality) "
                "VALUES (?, ?, 'conv-a', 'conv-a:0-0', ?, 'conversation')",
                (soul_id, user_id, str(historical_file)),
            )
        con.execute(
            "UPDATE resources SET local_path = ? WHERE conversation_id = 'conv-b'",
            (r"C:\old-app\segments\segment_0.json",),
        )

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
    if ambiguous_path:
        with pytest.raises(HTTPException, match="ownership missing or ambiguous"):
            gather_consolidation_inputs(deps, conversation_id="conv-a", soul_id=soul_id, user_id=user_id)
        return
    out = gather_consolidation_inputs(
        deps,
        conversation_id="conv-a",
        soul_id=soul_id,
        user_id=user_id,
    )

    assert out["selected_segment_ids_by_conversation"] == expected
    assert [row["segment_id"] for row in out["segment_inputs"]] == [historical_id, "conv-a:0-0", "conv-b:0-0"]
    assert out["segment_inputs"][0]["start_idx"] == out["segment_inputs"][0]["end_idx"] == 1
    assert [row["content"] for row in out["current_chat_messages"]] == ["older imported history", "message 0", "message 1"]
    monkeypatch.setattr(message_log, "_load_whatsapp_directory_names", lambda: {})
    render_rows = [*out["current_chat_messages"], {
            "conversation_id": "whatsapp:dm:fictional", "role": "user",
            "source_conversation_id": "whatsapp:dm:fictional",
            "content": "a separate platform", "ts_ms": 1_767_225_601_000,
        }]
    for rows in (render_rows, payload._normalize_conversation(render_rows)):
        rendered = cross_history._format_all_chat_history_for_ai(
            current_history=rows, cross_tail=[], conversation_id="conv-a",
            soul_id=soul_id, mark_current_chat=False,
        )
        assert "[dm][conv-a]" in rendered and "[dm][conv-b]" in rendered
        assert "a separate platform" in rendered
        assert rendered.index("older imported history") < rendered.index("message 0") < rendered.index("message 1")
    assert len(out["segment_inputs"][1]["memory_summaries"]) == 30
    assert [item["id"] for item in out["prior_context_memory_items"]] == ["mem-0"]

    dateless_file = chat_dirs["conv-b"] / "segments" / "segment_0.json"
    original = dateless_file.read_text()
    dateless_file.write_text(json.dumps([{"role": "user", "content": "message 1", "source_conversation_id": "conv-b"}]))
    dateless = gather_consolidation_inputs(deps, conversation_id="conv-a", soul_id=soul_id, user_id=user_id)
    assert [row["content"] for row in dateless["current_chat_messages"]] == ["older imported history", "message 0", "message 1"]
    dateless_file.write_text(original)

    deps.write_conversation_state("conv-a", soul_id=soul_id, user_id=user_id, updates={"import_state": {
        "history_end_index": 2, "memorize_cursor": 1, "pending_segment_ids": [historical_id],
        "stage": "consolidation", "error": None,
    }})
    ordinary = gather_consolidation_inputs(deps, conversation_id="conv-a", soul_id=soul_id, user_id=user_id)
    assert ordinary["excluded_segment_ids"] == [historical_id]
    svc = _DossierContextService(due_ids=("first",))
    ordinary["state"] = {**_inputs()["state"], **ordinary["state"]}
    ordinary["all_chat_history"] = rendered
    prepared = consolidation._prepare_dossier_consolidation_prompts(svc, inputs=ordinary, soul_id=soul_id, user_id=user_id)
    assert all(text in prepared[2] for text in ("older imported history", "message 0", "message 1", "a separate platform"))
    assert svc.excluded_segment_ids == [historical_id]
    assert [call[3]["excluded_segment_ids"] for call in svc.calls if call[0] == "prepare"] == [[historical_id]]
    deps.write_conversation_state("conv-a", soul_id=soul_id, user_id=user_id, updates={"import_state": {
        "history_end_index": 3, "memorize_cursor": 2,
        "pending_segment_ids": [historical_id, "conv-a:2-2"], "stage": "consolidation", "error": None,
    }})
    queries = []
    original_connect = deps.sqlite_connect
    def trace_connect(path):
        con = original_connect(path)
        con.set_trace_callback(queries.append)
        return con
    deps = replace(deps, sqlite_connect=trace_connect)
    for prefix in ([historical_id, "conv-a:2-2"], [historical_id]):
        historical = gather_consolidation_inputs(
            deps, conversation_id="conv-a", soul_id=soul_id, user_id=user_id, historical=True,
            selected_segments={("conv-a", sid) for sid in prefix},
        )
        excluded = {"conv-a:0-0", historical_id, "conv-a:2-2", "conv-b:0-0"} - set(prefix)
        assert set(historical["excluded_segment_ids"]) == excluded
        svc = _DossierContextService(due_ids=("first",))
        historical["state"] = {**_inputs()["state"], **historical["state"]}
        prepared = consolidation._prepare_dossier_consolidation_prompts(
            svc, inputs=historical, soul_id=soul_id, user_id=user_id,
        )
        assert set(svc.excluded_segment_ids) == excluded
        context = next(call[3] for call in svc.calls if call[0] == "prepare")
        assert context["segment_ids"] == prefix
        assert set(context["excluded_segment_ids"]) == excluded
        assert prepared[3]["weekly"] == 0
    assert historical["selected_segment_ids_by_conversation"] == {"conv-a": [historical_id]}
    assert [row["content"] for row in historical["current_chat_messages"]] == ["older imported history"]
    assert historical["prior_context_memory_items"] == []
    assert any("segment_id IN" in query and "FROM resources" in query for query in queries)
    assert not any("SELECT subject_id, predicate, object_id FROM triples" in query for query in queries)

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
    assert forced["segment_inputs"] == out["segment_inputs"]
    assert forced["current_chat_messages"] == out["current_chat_messages"]

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


def test_resource_retry_after_failed_publication_reuses_one_owned_file(tmp_path, request):
    class PublicationScope(BaseModel):
        user_id: str | None = None
        soul_id: str | None = None

    db_path = tmp_path / "publication.db"
    scope = {"user_id": "TestOwner", "soul_id": "TestSoul"}
    svc = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": f"sqlite:///{db_path}"}},
        user_config={"model": PublicationScope},
    )
    store = svc.database
    request.addfinalizer(store.close)
    cid, segment_id = "replika:test-publication", "replika:test-publication:0-0"
    chat_dir = tmp_path / "publication"
    (chat_dir / "segments").mkdir(parents=True)
    file = chat_dir / "segments" / "2020-01-01.json"
    content = json.dumps([{"role": "user", "content": "history", "ts_ms": 1_577_836_800_000}])
    file.write_text(content, encoding="utf-8")
    state_args = {"sqlite_current_path": lambda *_: db_path, **scope}
    write_conversation_state(cid, **state_args)
    resource_args = {"url": str(file), "local_path": str(file), "modality": "conversation",
                     "caption": None, "embedding": None, "user_data": scope,
                     "conversation_id": cid, "segment_id": segment_id}
    first = store.resource_repo.create_resource(**resource_args)
    with sqlite_connect(db_path) as con:
        con.execute("CREATE TRIGGER fail_publish BEFORE UPDATE OF pending_segment_ids ON conversations "
                    "BEGIN SELECT RAISE(ABORT, 'injected publication failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match="publication failure"):
        write_conversation_state(cid, **state_args, updates={"append_pending_segment_ids": [segment_id]})
    deps = replace(_make_consolidation_deps(db_path, tmp_path),
                   find_chat_dir_for_conversation=lambda *_: chat_dir)
    assert gather_consolidation_inputs(deps, conversation_id=cid, **scope)["reason"] == "no_pending_segments"
    # Match the existing runner's disposable-file cleanup after failed publication.
    file.unlink()
    file.write_text(content, encoding="utf-8")
    retried = store.resource_repo.create_resource(**resource_args)
    assert retried.id == first.id
    store.resource_repo.create_resource(**{**resource_args, "user_data": {**scope, "soul_id": "OtherSoul"}})
    with sqlite_connect(db_path) as con:
        con.execute("DROP TRIGGER fail_publish")
    write_conversation_state(cid, **state_args, updates={"append_pending_segment_ids": [segment_id]})
    result = gather_consolidation_inputs(deps, conversation_id=cid, **scope)
    assert [entry["segment_id"] for entry in result["segment_inputs"]] == [segment_id]
    assert [row["content"] for row in result["current_chat_messages"]] == ["history"]


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bad_content", "segment_end", "error_text"),
    [
        ("{not json", 0, "segment history unreadable"),
        (json.dumps([{"role": "user", "content": "valid"}, 7]), 0, "non-message row"),
        (json.dumps([{"role": "user", "content": "only row"}]), 1, "range does not match stored history"),
    ],
)
async def test_gather_consolidation_inputs_rejects_invalid_segment_file(
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
    soul_id TEXT, user_id TEXT, created_at DATETIME, memory_prior_context TEXT,
    conversation_id TEXT, segment_id TEXT, local_path TEXT, modality TEXT
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
    with sqlite3.connect(db_path) as con:
        con.execute(
            "INSERT INTO resources (soul_id, user_id, conversation_id, segment_id, local_path, modality) "
            "VALUES (?, ?, ?, ?, ?, 'conversation')",
            (soul_id, user_id, cid, segment_id, str(bad_file)),
        )

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

    expected_identifier = segment_id if "range" in error_text else str(bad_file)
    assert expected_identifier in str(exc_info.value.detail)

    broken = {"history_end_index": segment_end + 1, "memorize_cursor": segment_end,
              "pending_segment_ids": [segment_id], "stage": "consolidation", "error": None}
    deps.write_conversation_state(cid, soul_id=soul_id, user_id=user_id,
                                  updates={"import_state": broken, "pending_segment_ids": ["ordinary-next"]})
    with sqlite_connect(db_path) as con:
        con.row_factory = sqlite3.Row
        before = _soul_state.read(con)
    running = {}
    with pytest.raises(HTTPException, match=error_text):
        await consolidation._run_consolidation_pipeline_once(
            svc=object(), deps=deps, state_lock=asyncio.Lock(), running=running,
            load_cross_tail_for_ai=lambda **_kw: pytest.fail("gather failure must precede prompt/model work"),
            format_all_chat_history_for_ai=lambda **_kw: pytest.fail("gather failure must precede prompt/model work"),
            conversation_id=cid, soul_id=soul_id, user_id=user_id, historical=True,
        )
    assert running == {}
    with sqlite_connect(db_path) as con:
        con.row_factory = sqlite3.Row
        assert _soul_state.read(con) == before
        current = conversation_state_from_row(conversation_state_row(con, cid, soul_id=soul_id, user_id=user_id))
        assert current["pending_segment_ids"] == ["ordinary-next"]
        record = current["import_state"]
        assert record == broken  # The import caller owns failure reporting.


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


def _make_svc_stub(db_path: Path | None = None) -> object:
    class _TripleRepo:
        def add(self, _t, user_data=None, session=None): return None
        def invalidate(self, _s, _p, _o, scope=None, session=None): return None

    class _SvcStub:
        def __init__(self) -> None:
            self.database = type("DB", (), {"triple_repo": _TripleRepo(), "dsn": f"sqlite:///{db_path}"})()

    svc = _SvcStub()
    svc._sqlite_write_session = lambda _store: Session(create_engine(f"sqlite:///{db_path}", poolclass=NullPool))
    return svc


def _base_llm_results(**overrides) -> dict:
    base = {
        "anchor_writes": [],
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


@pytest.mark.asyncio
@pytest.mark.parametrize("ordinary_error", [None, "Ordinary failed", _soul_state.CONSOLIDATION_UNFINISHED])
async def test_historical_runner_failure_is_import_only(tmp_path, monkeypatch, ordinary_error):
    from app import main
    path = tmp_path / "test.db"
    with sqlite_connect(path) as con:
        sqlite_ensure_conversation_state_schema(con)
    deps = _make_consolidation_deps(path, tmp_path)
    scope = {"soul_id": "TestSoul", "user_id": "TestUser"}
    deps.write_conversation_state("chat", **scope, updates={
        "pending_segment_ids": ["ordinary"], "last_consolidation_error": ordinary_error,
        "last_consolidation_error_at": "2026-01-02T00:00:00+00:00" if ordinary_error else None,
        "import_state": {"history_end_index": 2, "memorize_cursor": 1,
                         "pending_segment_ids": ["historical"], "stage": "consolidation", "error": None},
    })
    with sqlite_connect(path) as con:
        con.row_factory = sqlite3.Row
        before = _soul_state.read(con)
    svc = _DossierContextService()
    svc.database = _make_svc_stub(path).database
    monkeypatch.setattr(main, "_sqlite_current_path", lambda *_: path)
    async def fail_chat(_prompt, **kwargs):
        assert kwargs["step"] == "anchors"
        if ordinary_error:
            with pytest.raises(HTTPException):
                main._require_soul_active(scope["user_id"], scope["soul_id"])
        else:
            main._require_soul_active(scope["user_id"], scope["soul_id"])
        raise RuntimeError("Import model call failed")
    svc.chat = fail_chat
    def gather(*_args, **kwargs):
        assert kwargs["historical"] is True
        return {**_inputs(), "historical": True, "status": "ready", "db_path": path,
                "selected_segment_ids_by_conversation": {"chat": ["historical"]}}
    monkeypatch.setattr(consolidation, "gather_consolidation_inputs", gather)
    running = main._CONSOLIDATION_RUNNING
    with pytest.raises(RuntimeError, match="Import model call failed"):
        await consolidation._run_consolidation_pipeline_once(
            svc=svc, deps=deps, state_lock=asyncio.Lock(), running=running,
            load_cross_tail_for_ai=lambda **_kw: pytest.fail("no live tail in historical work"),
            format_all_chat_history_for_ai=lambda **_kw: "Historical evidence",
            conversation_id="chat", historical=True, **scope,
        )
    assert running == {}
    with sqlite_connect(path) as con:
        con.row_factory = sqlite3.Row
        assert _soul_state.read(con) == before
        assert bool(_soul_state.activity_pause(before, memorize_running=False, consolidation_running=False)) is bool(ordinary_error)
        state = conversation_state_from_row(conversation_state_row(con, "chat", **scope))
        assert state["pending_segment_ids"] == ["ordinary"]
        assert state["import_state"]["pending_segment_ids"] == ["historical"]
        assert state["import_state"]["history_end_index"] == 2
        assert state["import_state"]["error"] == "Import interrupted. Retry required."


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_historical_consolidation_only_changes_its_import(tmp_path, monkeypatch, fail):
    path = tmp_path / "test.db"
    with sqlite_connect(path) as con:
        sqlite_ensure_conversation_state_schema(con)
    deps = _make_consolidation_deps(path, tmp_path)
    scope = {"soul_id": "TestSoul", "user_id": "TestOwner"}
    contributor = {"history_end_index": 2, "memorize_cursor": 1, "pending_segment_ids": ["other:0-1"],
                   "stage": "consolidation", "error": None}
    deps.write_conversation_state("other", **scope, updates={"import_state": contributor})
    deps.write_conversation_state("ordinary", **scope, updates={"pending_segment_ids": ["ordinary:0-1"]})
    svc = _DossierContextService()
    stub = _make_svc_stub(path)
    svc.database, svc._sqlite_write_session = stub.database, stub._sqlite_write_session
    def record(cid):
        with sqlite_connect(path) as con:
            con.row_factory = sqlite3.Row
            return conversation_state_from_row(conversation_state_row(con, cid, **scope))["import_state"]
    monkeypatch.setattr(consolidation, "gather_consolidation_inputs", lambda *_a, **_kw: {
        **_inputs(), "status": "ready", "db_path": path, "historical": True,
        "selected_segment_ids": ["other:0-1"],
        "selected_segment_ids_by_conversation": {"other": ["other:0-1"]},
    })
    async def llm(*_a, **_kw):
        assert record("other")["error"] == "Import interrupted. Retry required."
        if fail:
            raise RuntimeError("Selected import failed")
        return _base_llm_results()
    monkeypatch.setattr(consolidation, "run_consolidation_llm", llm)
    kwargs = dict(svc=svc, deps=deps, state_lock=asyncio.Lock(), running={},
                  load_cross_tail_for_ai=lambda **_kw: [], format_all_chat_history_for_ai=lambda **_kw: "",
                  conversation_id="other", historical=True, **scope)
    if fail:
        with pytest.raises(RuntimeError, match="Selected import failed"):
            await consolidation._run_consolidation_pipeline_once(**kwargs)
        assert record("other")["pending_segment_ids"] == contributor["pending_segment_ids"]
        assert record("other")["error"] == "Import interrupted. Retry required."
    else:
        result = await consolidation._run_consolidation_pipeline_once(**kwargs)
        assert result["status"] == "ok" and record("other")["stage"] == "complete"
        assert record("other")["pending_segment_ids"] == [] and record("other")["error"] is None
    with sqlite_connect(path) as con:
        con.row_factory = sqlite3.Row
        assert conversation_state_from_row(conversation_state_row(con, "ordinary", **scope))["pending_segment_ids"] == ["ordinary:0-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("fits", "anchor_capacity"), [(False, 8001), (False, None), (True, 1_000_000)])
async def test_historical_selection_shrinks_whole_prefix_before_any_model_call(tmp_path, monkeypatch, fits, anchor_capacity):
    path = tmp_path / "test.db"
    with sqlite_connect(path) as con:
        sqlite_ensure_conversation_state_schema(con)
    svc = _DossierContextService(due_ids=("first",))
    svc.database = _make_svc_stub(path).database
    segments = [{
        "conversation_id": "import:dm:example", "segment_id": f"import:dm:example:{i}-{i}",
        "memory_summaries": [{"id": f"memory-{i}", "memory_ref": i + 1, "memory_type": "knowledge",
                              "summary": f"Evidence {i}: " + "Detail. " * 500}],
    } for i in range(3)]
    def inputs(rows):
        return {**_inputs(), "status": "ready", "historical": True, "db_path": path,
                "segment_inputs": rows, "current_chat_messages": [],
                "selected_segment_ids": [row["segment_id"] for row in rows],
                "selected_segment_ids_by_conversation": {"import:dm:example": [row["segment_id"] for row in rows]}}
    deps = _make_consolidation_deps(path, tmp_path)
    scope = {"soul_id": "TestSoul", "user_id": "TestUser"}
    original = {"history_end_index": 3, "memorize_cursor": 2, "stage": "consolidation", "error": None,
                "pending_segment_ids": [row["segment_id"] for row in segments]}
    deps.write_conversation_state("import:dm:example", **scope, updates={"import_state": original})
    deps.write_conversation_state("ordinary", **scope, updates={
        "last_consolidation_error": "Ordinary failed", "last_consolidation_error_at": "2026-01-01T00:00:00+00:00"})
    with sqlite_connect(path) as con:
        con.row_factory = sqlite3.Row
        ordinary_before = _soul_state.read(con)
    first = consolidation._prepare_dossier_consolidation_prompts(
        svc, inputs=inputs(segments[:1]), soul_id="TestSoul", user_id="TestUser",
    )[3]["dossiers"]
    second = consolidation._prepare_dossier_consolidation_prompts(
        svc, inputs=inputs(segments[:2]), soul_id="TestSoul", user_id="TestUser",
    )[3]["dossiers"]
    assert first < second
    svc.llm_profiles.profiles["revision"].context_window_tokens = 8000 + ((first + second) // 2) * 5 // 4 + 1
    svc.llm_profiles.profiles["default"].context_window_tokens = anchor_capacity
    svc.calls.clear()
    captures = []
    def gather(*_args, **kwargs):
        selection = kwargs["selected_segments"]
        rows = [row for row in segments if selection is None or (row["conversation_id"], row["segment_id"]) in selection]
        captures.append([row["segment_id"] for row in rows])
        return inputs(rows)
    monkeypatch.setattr(consolidation, "gather_consolidation_inputs", gather)
    monkeypatch.setattr(consolidation, "write_consolidation_outputs", lambda *_a, **kw: {
        "consumed_segment_ids": kw["inputs"]["selected_segment_ids"],
    })
    kwargs = dict(svc=svc, deps=deps, state_lock=asyncio.Lock(), running={},
                  load_cross_tail_for_ai=lambda **_kw: [], format_all_chat_history_for_ai=lambda **_kw: "A lived span.",
                  conversation_id="import:dm:example", historical=True, **scope)
    if fits:
        result = await consolidation._run_consolidation_pipeline_once(**kwargs)
        assert result["result"]["consumed_segment_ids"] == [segments[0]["segment_id"]]
        assert [call for call in svc.calls if call[0] == "chat"] == [("chat", "dossiers"), ("chat", "anchors")]
    else:
        error_text = "No whole historical segment" if anchor_capacity is not None else "context_window_tokens is required"
        with pytest.raises(ValueError, match=error_text):
            await consolidation._run_consolidation_pipeline_once(**kwargs)
        assert not [call for call in svc.calls if call[0] in {"chat", "apply"}]
    assert captures == [[row["segment_id"] for row in segments[:count]] for count in (3, 2, 1)]
    assert kwargs["running"] == {}
    with sqlite_connect(path) as con:
        con.row_factory = sqlite3.Row
        assert _soul_state.read(con) == ordinary_before
        record = conversation_state_from_row(conversation_state_row(con, "import:dm:example", **scope))["import_state"]
        assert {**record, "error": None} == original
        assert record["error"] == ("Import interrupted. Retry required." if fits else None)


@pytest.mark.asyncio
@pytest.mark.parametrize("historical", [False, True])
async def test_consolidation_shared_transaction_rolls_back_and_retries(tmp_path, monkeypatch, request, historical):
    class Scope(BaseModel):
        user_id: str | None = None
        soul_id: str | None = None

    class Embed:
        async def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    path = tmp_path / "transaction.db"
    scope = {"user_id": "Fictional User", "soul_id": "Fictional Soul"}
    svc = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": f"sqlite:///{path}"}},
        user_config={"model": Scope},
    )
    svc.llm_profiles.profiles["default"].context_window_tokens = 1_000_000
    store = svc.database
    request.addfinalizer(store.close)
    journals = []
    import memu.app.dossier as dossier_module
    monkeypatch.setattr(dossier_module, "append_category_summary_journal", lambda **kw: journals.append(kw))
    monkeypatch.setattr(_soul_summaries, "append_summary_journal", lambda **kw: journals.append(kw))
    anchors = {
        role: store.memory_category_repo.get_or_create_category(
            name=name, description="Original description", embedding=[1.0, 0.0],
            user_data=scope, kind="lore", lore_subtype="person", anchor_role=role,
        )
        for role, name in (("soul", scope["soul_id"]), ("user", scope["user_id"]))
    }
    first = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="First evidence", embedding=[1.0, 0.0],
        user_data=scope, conversation_id="chat", segment_id="chat:0-1",
    )
    previous = store.memory_item_repo.create_item(
        memory_type="narrative_self", summary="Earlier self", embedding=[1.0, 0.0], user_data=scope,
    )
    stale = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="Deleted while thinking", embedding=[1.0, 0.0], user_data=scope,
    )
    ordinary = store.memory_category_repo.get_or_create_category(
        name="Life domain", description="Evidence", embedding=[1.0, 0.0], user_data=scope,
        kind="topic", last_evidence_at=datetime.now(UTC),
    )
    store.category_item_repo.link_item_category(first.id, ordinary.id, scope)
    bundle = svc.prepare_dossier_revision(ordinary.id, scope)
    await svc.apply_dossier_revision(bundle, {
        "dossier_id": ordinary.id, "description": ordinary.description,
        "resulting_prose": f"Evidence [M{first.memory_ref}].", "cited_item_ids": [first.id],
        "add_item_ids": [], "remove_item_ids": [], "cleanup_item_ids": [],
    }, scope)
    assert svc.list_due_dossiers(scope, segment_ids=["chat:0-1"]) == []
    anchor_writes = []
    for role, anchor in anchors.items():
        bundle = svc.prepare_anchor_revision(role, scope, [])
        decision = {
            "anchor_role": role, "dossier_id": anchor.id, "description": "Revised description",
            "resulting_prose": "## Identity\nRevised prose.", "cited_item_ids": [],
            "add_item_ids": [], "remove_item_ids": [], "cleanup_item_ids": [],
        }
        anchor_writes.append((bundle, await svc.prepare_anchor_revision_apply(
            bundle, decision, scope, embedding_client=Embed(),
        )))
    later = store.memory_item_repo.create_item(
        memory_type="knowledge", summary="Later evidence", embedding=[1.0, 0.0],
        user_data=scope, conversation_id="chat", segment_id="chat:2-3",
    )
    store.category_item_repo.link_item_category(later.id, ordinary.id, scope)
    svc.graph_delete_memory(stale.id, where=scope)
    deps = _make_consolidation_deps(path, tmp_path)
    deps.write_conversation_state("chat", **scope, updates={"pending_segment_ids": ["chat:0-1", "chat:2-3"]})
    if historical:
        deps.write_conversation_state("chat", **scope, updates={"import_state": {
            "history_end_index": 4, "memorize_cursor": 3,
            "pending_segment_ids": ["chat:0-1", "chat:2-3"], "stage": "consolidation", "error": "Earlier import failure",
        }})
    with sqlite_connect(path) as con:
        con.row_factory = sqlite3.Row
        _soul_state.write(con, {
            "intentions_active": [], "last_consolidation_at": "2026-01-01T00:00:00+00:00",
            "last_consolidation_error": "Independent ordinary failure",
            "last_consolidation_error_at": "2026-01-02T00:00:00+00:00",
            "retrieval_ids_since_consolidation": ["tracked"],
            "prior_context_ids_since_consolidation": ["prior"],
        })
        _soul_summaries.write_live(con, kind="narrative_self", summary="Current self")
        con.commit()
        state = _soul_state.read(con)
        con.execute("UPDATE soul_state SET summaries_revision = summaries_revision + 1 WHERE id = 1")
        con.commit()
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    inputs = {"db_path": path, "historical": historical, "state": state,
              "narrative_self": "Current self", "selected_segment_ids": ["chat:0-1"]}
    results = _base_llm_results(
        anchor_writes=anchor_writes, narrative_self="New self", old_narrative_text="Current self",
        old_narrative_embedding=[1.0, 0.0], companion_memory="Reflection", companion_embedding=[1.0, 0.0],
        life_goal_add=["Be curious"], intentions_replacement=[{"id": "new", "text": "Explore"}],
        edges=[{"subject_id": first.id, "predicate": "evokes", "object_id": later.id},
               {"subject_id": stale.id, "predicate": "evokes", "object_id": later.id}],
    )

    def snapshot():
        with sqlite_connect(path) as con:
            return {
                table: con.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()  # nosec B608: fixed table tuple below
                for table in ("memory_items", "categories", "category_items", "triples",
                              "memory_ref_counters", "conversations", "soul_state", "life_goals", "narrative_history")
            }

    before = snapshot()
    wrong_path = tmp_path / "wrong.db"
    with pytest.raises(ValueError, match="different databases"):
        write_consolidation_outputs(deps, svc, inputs={**inputs, "db_path": wrong_path},
            llm_results=results, conversation_id="chat", **scope)
    assert not wrong_path.exists()
    with monkeypatch.context() as patch:
        patch.setattr(consolidation, "gather_consolidation_inputs",
                      lambda *_args, **_kwargs: {**inputs, "status": "ready", "db_path": wrong_path})
        patch.setattr(consolidation, "_record_consolidation_failure", lambda **_kwargs: None)
        with pytest.raises(ValueError, match="different databases"):
            await consolidation._run_consolidation_pipeline_once(
                svc=svc, deps=deps, state_lock=asyncio.Lock(), running={},
                load_cross_tail_for_ai=lambda **_kw: pytest.fail("must fail before context/model work"),
                format_all_chat_history_for_ai=lambda **_kw: pytest.fail("must fail before context/model work"),
                conversation_id="chat", **scope,
            )
    cached_before = {key: value.model_dump() for key, value in store.categories.items()}
    journals.clear()
    window = {"active": False, "expect_begin": False, "checkout_begin": False}
    bad_commits = []
    first_statements = []
    session_factory = svc._sqlite_write_session

    def write_session(db):
        window["checkout_begin"] = True
        return session_factory(db)

    monkeypatch.setattr(svc, "_sqlite_write_session", write_session)

    def trace(sql):
        if window["expect_begin"]:
            first_statements.append(sql)
            window["expect_begin"] = False
        if sql == "COMMIT" and window["active"]:
            bad_commits.append(sql)

    def checkout(con, *_args):
        assert not window["active"], "a helper opened a second repository connection"
        assert con.row_factory is None
        if window["checkout_begin"]:
            window["expect_begin"] = True
            window["checkout_begin"] = False
        con.set_trace_callback(trace)

    def before_sql(_con, _cursor, statement, *_args):
        if statement == "BEGIN IMMEDIATE":
            window["active"] = True

    def end_window(con):
        assert con.connection.driver_connection.row_factory is None
        window["active"] = False

    engine = store._sessions.engine
    event.listen(engine, "checkout", checkout)
    event.listen(engine, "before_cursor_execute", before_sql)
    event.listen(engine, "commit", end_window)
    event.listen(engine, "rollback", end_window)
    connect = deps.sqlite_connect

    def guarded_connect(db_path):
        assert not window["active"], "a helper opened a second MCP connection"
        return connect(db_path)

    from app import db as db_module
    from app.services import state as state_module
    monkeypatch.setattr(db_module, "sqlite_connect", guarded_connect)
    monkeypatch.setattr(state_module, "sqlite_connect", guarded_connect)

    write = deps.write_conversation_state

    def fail_at_end(cid, **kwargs):
        result = write(cid, **kwargs)
        if kwargs["updates"].get("last_consolidation_at") or "import_state" in kwargs["updates"]:
            raise RuntimeError("late apply failure")
        return result

    with pytest.raises(RuntimeError, match="late apply failure"):
        write_consolidation_outputs(replace(deps, sqlite_connect=guarded_connect, write_conversation_state=fail_at_end),
            svc, inputs=inputs, llm_results=results, conversation_id="chat", **scope)
    assert snapshot() == before
    assert {key: value.model_dump() for key, value in store.categories.items()} == cached_before
    assert journals == [] and bad_commits == []
    assert first_statements == ["BEGIN IMMEDIATE"]
    assert svc.list_due_dossiers(scope, segment_ids=["chat:0-1"]) == []

    async def no_dossier_call(*_args, **_kwargs):
        pytest.fail("Retry must not regenerate the checkpointed dossier")

    monkeypatch.setattr(svc, "chat", no_dossier_call)
    await prepare_dossier_consolidation_context(svc, inputs={**_inputs(), **inputs},
        soul_id=scope["soul_id"], user_id=scope["user_id"])

    write_consolidation_outputs(replace(deps, sqlite_connect=guarded_connect), svc,
        inputs=inputs, llm_results=results, conversation_id="chat", **scope)
    assert bad_commits == [] and len(journals) == 3
    assert first_statements == ["BEGIN IMMEDIATE", "BEGIN IMMEDIATE"]
    assert [row["edited_by"] for row in journals] == ["anchor_revision", "anchor_revision", "consolidation"]
    for role, anchor in anchors.items():
        assert store.categories[anchor.id].summary == "## Identity\nRevised prose."
    assert ordinary.id in {row.id for row in svc.list_due_dossiers(scope, segment_ids=["chat:2-3"])}
    with sqlite_connect(path) as con:
        con.row_factory = sqlite3.Row
        final_state = _soul_state.read(con)
        assert final_state["narrative_self"] == "New self"
        assert final_state["intentions_active"] == ([] if historical else [{"id": "new", "text": "Explore"}])
        assert con.execute("SELECT COUNT(*) FROM narrative_history").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM triples WHERE subject_id=? AND predicate='evokes'", (first.id,)).fetchone()[0] == (0 if historical else 1)
        assert con.execute("SELECT COUNT(*) FROM triples WHERE subject_id=?", (stale.id,)).fetchone()[0] == 0
        row = con.execute("SELECT * FROM conversations WHERE conversation_id='chat'").fetchone()
        assert json.loads(row["pending_segment_ids"]) == (["chat:0-1", "chat:2-3"] if historical else ["chat:2-3"])
        if historical:
            record = json.loads(row["import_state"])
            assert record["pending_segment_ids"] == ["chat:2-3"] and record["error"] is None
            for key in ("last_consolidation_at", "last_consolidation_error", "last_consolidation_error_at",
                        "retrieval_ids_since_consolidation", "prior_context_ids_since_consolidation"):
                assert final_state[key] == state[key]
        assert con.execute("SELECT COUNT(*) FROM memory_items WHERE summary='Reflection'").fetchone()[0] == (0 if historical else 1)
        assert con.execute("SELECT COUNT(*) FROM memory_items WHERE summary='Current self'").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM triples WHERE subject_id=? AND predicate='evolved_into'", (previous.id,)).fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM life_goals WHERE description='Be curious'").fetchone()[0] == 1


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
            _make_svc_stub(db_path),
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
            _make_svc_stub(db_path),
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
            )
            con.commit()
            gathered = _soul_state.read(con)
            _soul_summaries.write_live(
                con,
                kind="narrative_self",
                summary="Atomic edit.",
                expected_revision=gathered["summaries_revision"],
                displayed_summary="Gathered story.",
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
                _make_svc_stub(db_path),
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
                    _make_svc_stub(db_path),
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
                _make_svc_stub(db_path),
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


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["wait", "force", "time", "revision-size", "reflection-size", "cli-size", "too-large", "missing"])
async def test_ordinary_automatic_consolidation_uses_time_or_actual_stage_capacity(tmp_path, monkeypatch, mode):
    from datetime import timedelta
    path = tmp_path / "cadence.db"
    deps = _make_consolidation_deps(path, tmp_path)
    scope = {"user_id": "TestUser", "soul_id": "TestSoul"}
    last = (datetime.now(UTC) - timedelta(days=8 if mode == "time" else 0)).isoformat()
    selected = {"chat-a": ["chat-a:0-0", "chat-a:1-1"], "chat-b": ["chat-b:0-0"]}
    for cid, ids in selected.items():
        deps.write_conversation_state(cid, **scope, updates={"pending_segment_ids": ids, "last_consolidation_at": last})
    with sqlite_connect(path) as con:
        con.row_factory = sqlite3.Row
        before = _soul_state.read(con)
    svc = _DossierContextService(due_ids=("first",), profiles=("default", "revision", "consolidation"))
    svc.database = _make_svc_stub(path).database
    rows = [{"conversation_id": cid, "segment_id": sid, "memory_summaries": []}
            for cid, ids in selected.items() for sid in ids]
    inputs = {**_inputs(), "status": "ready", "db_path": path, "last_consolidation_at": last,
              "state": before, "segment_inputs": rows, "current_chat_messages": [],
              "selected_segment_ids": [row["segment_id"] for row in rows],
              "selected_segment_ids_by_conversation": selected}
    prepare = consolidation._prepare_dossier_consolidation_prompts
    estimates = prepare(svc, inputs=inputs, **scope)[3]
    if mode in {"revision-size", "reflection-size", "cli-size", "too-large"}:
        tokens = (estimates["dossiers"] if mode in {"revision-size", "too-large"}
                  else max(estimates.values()) if mode == "cli-size"
                  else max(estimates["anchors"], estimates["weekly"]))
        budget = tokens - 1 if mode == "too-large" else tokens + (tokens + 2) // 3 - 1
        context = 8000 + (budget * 5 + 3) // 4
        if mode == "cli-size":
            svc._claude_code = True
            svc._claude_code_model = "TestCLIModel"
            svc._claude_code_context_window_tokens = context - 8000
        else:
            svc.llm_profiles.profiles["revision" if mode in {"revision-size", "too-large"} else "consolidation"].context_window_tokens = context
    elif mode == "missing":
        svc.llm_profiles.profiles["consolidation"].context_window_tokens = None
    gathers, preparations, applied = [], [], []
    def gather(*_args, **kwargs):
        assert kwargs["selected_segments"] is None and kwargs["force"] is (mode == "force")
        gathers.append(True)
        return inputs
    def prepared(*args, **kwargs):
        preparations.append(True)
        return prepare(*args, **kwargs)
    def apply(*_args, **kwargs):
        applied.append(kwargs["inputs"]["selected_segment_ids"])
        return {"consumed_segment_ids": applied[-1]}
    monkeypatch.setattr(consolidation, "gather_consolidation_inputs", gather)
    monkeypatch.setattr(consolidation, "_prepare_dossier_consolidation_prompts", prepared)
    monkeypatch.setattr(consolidation, "write_consolidation_outputs", apply)
    svc.calls.clear()
    running = {}
    kwargs = dict(svc=svc, deps=deps, state_lock=asyncio.Lock(), running=running,
                  load_cross_tail_for_ai=lambda **_kw: [], format_all_chat_history_for_ai=lambda **_kw: "A lived span.",
                  conversation_id="chat-a", force=mode == "force", **scope)
    if mode in {"too-large", "missing"}:
        with pytest.raises(ValueError, match="token limit" if mode == "too-large" else "context_window_tokens"):
            await consolidation._run_consolidation_pipeline_once(**kwargs)
    else:
        result = await consolidation._run_consolidation_pipeline_once(**kwargs)
        assert result["status"] == ("skipped" if mode == "wait" else "ok")
    assert running == {} and len(gathers) == len(preparations) == 1
    model_calls = [call for call in svc.calls if call[0] == "chat"]
    if mode in {"wait", "too-large", "missing"}:
        assert model_calls == [] and applied == []
        with sqlite_connect(path) as con:
            con.row_factory = sqlite3.Row
            after = _soul_state.read(con)
            assert after["last_consolidation_at"] == before["last_consolidation_at"]
            assert after["memorize_failure"] is None
            assert bool(after["last_consolidation_error"]) == (mode != "wait")
            for cid, ids in selected.items():
                assert conversation_state_from_row(conversation_state_row(con, cid, **scope))["pending_segment_ids"] == ids
    else:
        assert model_calls == [("chat", "dossiers"), ("chat", "anchors"), ("chat", "weekly")]
        assert applied == [inputs["selected_segment_ids"]]
