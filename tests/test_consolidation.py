import sqlite3
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from memu.app.dossier import DossierRevisionStaleError

from app.db import json_to_db, normalize_text_list, sqlite_connect, sqlite_ensure_conversation_state_schema, sqlite_ensure_nonempty
from app.services import segment
from app.services import soul_state as _soul_state
from app.services.consolidation import ConsolidationDeps, write_consolidation_outputs
from app.services.consolidation import _format_dossier_context_for_prompt
from app.services.consolidation import _format_segment_memory_items_for_prompt
from app.services.consolidation import _parse_reflection_xml
from app.services.consolidation import _remap_edges_with_memory_ids
from app.services.consolidation import _select_prompt_objective
from app.services.consolidation import _select_interval_segment_window
from app.services.consolidation import gather_consolidation_inputs
from app.services.consolidation import prepare_dossier_consolidation_context
from app.services.consolidation import preflight_consolidation_profiles
from app.services.consolidation import run_consolidation_llm
from app.services.graph_edges import invalidate_memory_edges, write_memory_edges
from app.services.state import conversation_state_from_row, conversation_state_row, write_conversation_state


class _DossierContextService:
    def __init__(self, *, due_ids=(), profiles=("default", "revision"), stale_id=None) -> None:
        self.memorize_config = SimpleNamespace(category_update_llm_profile="revision")
        self.llm_profiles = SimpleNamespace(profiles={name: object() for name in profiles})
        self.due = [SimpleNamespace(id=dossier_id) for dossier_id in due_ids]
        self.stale_id = stale_id
        self.calls: list[tuple] = []
        self.prompts: list[str] = []

    def list_due_dossiers(self, scope):
        self.calls.append(("due", scope))
        return self.due

    def prepare_dossier_revision(self, dossier_id, scope, **context):
        self.calls.append(("prepare", dossier_id, scope, context))
        return {
            "dossier": SimpleNamespace(
                id=dossier_id,
                kind="topic",
                name=dossier_id.title(),
                description=f"{dossier_id} description",
                summary="## Current\nStable.",
            ),
            "target_words": 300,
            "cleanup_items": [],
            "cited_items": [],
            "pending_items": [],
            "candidate_items": [],
            "linked_item_ids": [],
            "cited_unlinked_item_ids": [],
        }

    def prepare_anchor_revision(self, role, scope, actionable_ids):
        self.calls.append(("anchor", role, scope, actionable_ids))
        return {
            "dossier": SimpleNamespace(
                id=f"anchor-{role}",
                anchor_role=role,
                summary="## Current\nStable.",
            ),
            "cited_items": [],
            "candidate_items": [],
            "linked_item_ids": [],
            "linked_inactive_item_ids": [],
            "actionable_item_ids": [],
        }

    async def chat(self, prompt, **kwargs):
        self.calls.append(("chat", kwargs["step"]))
        self.prompts.append(prompt)
        if kwargs["step"] == "reflection":
            return """<reflection>
  <narrative_self>I am steady.</narrative_self>
  <anchor_revisions>
    <anchor role="soul"><description>My living history.</description><prose_action>keep</prose_action><prose_patches></prose_patches></anchor>
    <anchor role="user"><description>My human's living history.</description><prose_action>keep</prose_action><prose_patches></prose_patches></anchor>
  </anchor_revisions>
  <life_goals><add></add><remove></remove></life_goals>
  <intentions><boost target_id="relax" /></intentions>
  <edges></edges><companion_memory></companion_memory>
</reflection>"""
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

    async def embed(self, _texts, **_kwargs):
        return []

    def build_dossier_index(self, scope):
        self.calls.append(("index", scope))
        return "- Health: Current health"

    def list_dossiers_for_segments(self, scope, *, segment_ids):
        self.calls.append(("relevant", scope, list(segment_ids)))
        return [SimpleNamespace(name="Health", description="Current health", summary="Body [M4].")]


@pytest.mark.parametrize(
    ("profiles", "consolidation_profile", "missing"),
    [
        (("default",), None, "revision"),
        (("default", "revision"), "reflection", "reflection"),
    ],
)
def test_consolidation_preflight_checks_both_profiles(
    profiles, consolidation_profile, missing
) -> None:
    svc = _DossierContextService(due_ids=("first",), profiles=profiles)
    with pytest.raises(KeyError, match=missing):
        preflight_consolidation_profiles(svc, consolidation_profile)
    assert svc.calls == []


@pytest.mark.asyncio
async def test_dossier_context_uses_one_holistic_call_and_preserves_due_order() -> None:
    svc = _DossierContextService(due_ids=("first", "second"))
    inputs = {
        "narrative_self": "I am steady.",
        "active_life_goals": ["Stay curious"],
        "removed_life_goals": ["Old goal"],
        "selected_segment_ids": ["segment-2", "segment-3"],
        "intention_activity": [],
        "state": {},
        "segment_inputs": [],
        "prior_context_memory_items": [],
        "all_chat_history": "A lived span.",
    }

    result = await prepare_dossier_consolidation_context(
        svc,
        inputs=inputs,
        soul_id="TestSoul",
        user_id="TestUser",
        consolidation_llm_profile=None,
    )

    assert result is inputs
    assert [call for call in svc.calls if call[0] == "chat"] == [("chat", "dossiers")]
    assert [call[1] for call in svc.calls if call[0] == "apply"] == ["first", "second"]
    assert result["dossier_index"] == "- Health: Current health"
    assert result["relevant_dossiers"][0].summary == "Body [M4]."
    assert (
        "relevant",
        {"soul_id": "TestSoul", "user_id": "TestUser"},
        ["segment-2", "segment-3"],
    ) in svc.calls
    first_context = next(call[3] for call in svc.calls if call[0] == "prepare")
    assert first_context == {
        "narrative_self": "I am steady.",
        "active_life_goals": ["Stay curious"],
        "removed_life_goals": ["Old goal"],
    }


@pytest.mark.asyncio
async def test_dossier_context_keeps_first_apply_when_second_is_stale() -> None:
    svc = _DossierContextService(due_ids=("first", "second"), stale_id="second")
    with pytest.raises(DossierRevisionStaleError):
        await prepare_dossier_consolidation_context(
            svc,
            inputs={
                "active_life_goals": [],
                "removed_life_goals": [],
                "selected_segment_ids": ["segment-1"],
                "intention_activity": [],
                "state": {},
                "segment_inputs": [],
            },
            soul_id="TestSoul",
            user_id="TestUser",
            consolidation_llm_profile=None,
        )
    assert [call[1] for call in svc.calls if call[0] == "apply"] == ["first", "second"]
    assert not any(call[0] in {"index", "relevant"} for call in svc.calls)


@pytest.mark.asyncio
async def test_dossier_context_without_due_dossiers_still_builds_context() -> None:
    svc = _DossierContextService()
    inputs = {
        "active_life_goals": [],
        "removed_life_goals": [],
        "selected_segment_ids": ["segment-4"],
        "intention_activity": [],
        "state": {},
        "segment_inputs": [],
        "prior_context_memory_items": [],
    }
    await prepare_dossier_consolidation_context(
        svc,
        inputs=inputs,
        soul_id="TestSoul",
        user_id="TestUser",
        consolidation_llm_profile=None,
    )
    assert inputs["dossier_index"] == "- Health: Current health"
    assert (
        "relevant",
        {"soul_id": "TestSoul", "user_id": "TestUser"},
        ["segment-4"],
    ) in svc.calls
    assert not any(call[0] == "chat" for call in svc.calls)


@pytest.mark.asyncio
async def test_reflection_uses_new_root_and_applies_both_validated_anchors() -> None:
    svc = _DossierContextService()
    inputs = {
        "narrative_self": "I am steady.",
        "active_life_goals": [],
        "removed_life_goals": [],
        "selected_segment_ids": ["segment-4"],
        "intention_activity": [],
        "state": {"intentions_active": None},
        "segment_inputs": [],
        "prior_context_memory_items": [],
        "all_chat_history": "A lived span.",
    }
    await prepare_dossier_consolidation_context(
        svc,
        inputs=inputs,
        soul_id="TestSoul",
        user_id="TestUser",
        consolidation_llm_profile=None,
    )

    out = await run_consolidation_llm(
        svc,
        inputs=inputs,
        soul_id="TestSoul",
        user_id="TestUser",
        llm_profile=None,
    )

    assert out["narrative_self"] == "I am steady."
    assert out["intention_actions"] == []
    assert [call[1] for call in svc.calls if call[0] == "apply_anchor"] == ["soul", "user"]
    assert "A lived span." in svc.prompts[-1]
    assert "Body [4]." in svc.prompts[-1]


def test_format_dossier_context_preserves_prose_and_rejects_oversize() -> None:
    prose = "## Timeline\n- A remembered day [M12]."
    rendered = _format_dossier_context_for_prompt(
        "- Health: A living account",
        [SimpleNamespace(name="Health", description="A living account", summary=prose)],
    )
    assert rendered == (
        "# Dossier index\n"
        "- Health: A living account\n\n"
        "# Relevant dossiers\n"
        "## Health\n"
        "Description: A living account\n"
        "## Timeline\n- A remembered day [12]."
    )
    assert "[M12]" not in rendered
    assert "[12]" in rendered

    with pytest.raises(ValueError, match="exceeds 100000 tokens"):
        _format_dossier_context_for_prompt(
            "",
            [SimpleNamespace(name="Large", description="Large", summary="word " * 75_001)],
        )



def test_select_prompt_objective_unwraps_only_requested_block() -> None:
    prompt = "before\n<first_time>first</first_time>\n<ongoing>later</ongoing>\nafter"
    assert _select_prompt_objective(prompt, first_time=True) == "before\nfirst\nafter"
    assert _select_prompt_objective(prompt, first_time=False) == "before\nlater\nafter"


def test_first_reflection_requires_narrative_self() -> None:
    with pytest.raises(ValueError, match="requires narrative_self"):
        _parse_reflection_xml("<reflection></reflection>", {}, first_time=True)


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


def test_select_interval_segment_window_uses_oldest_pending_baseline() -> None:
    rows = [
        {"segment_id": "cid:0-3", "start_idx": 0, "end_idx": 3, "happened_at": datetime(2026, 1, 1, tzinfo=UTC)},
        {"segment_id": "cid:4-7", "start_idx": 4, "end_idx": 7, "happened_at": datetime(2026, 1, 4, tzinfo=UTC)},
        {"segment_id": "cid:8-9", "start_idx": 8, "end_idx": 9, "happened_at": datetime(2026, 1, 8, tzinfo=UTC)},
        {"segment_id": "cid:10-11", "start_idx": 10, "end_idx": 11, "happened_at": datetime(2026, 1, 12, tzinfo=UTC)},
    ]

    out = _select_interval_segment_window(rows, interval_days=7, force=False)

    assert out["selected_segment_ids"] == ["cid:0-3", "cid:4-7", "cid:8-9"]
    assert out["remaining_segment_ids"] == ["cid:10-11"]
    assert out["reason"] is None


def test_select_interval_segment_window_leaves_short_tail_pending() -> None:
    rows = [
        {"segment_id": "cid:0-3", "start_idx": 0, "end_idx": 3, "happened_at": datetime(2026, 1, 1, tzinfo=UTC)},
        {"segment_id": "cid:4-7", "start_idx": 4, "end_idx": 7, "happened_at": datetime(2026, 1, 4, tzinfo=UTC)},
    ]

    out = _select_interval_segment_window(rows, interval_days=7, force=False)

    assert out["selected_segment_ids"] == []
    assert out["remaining_segment_ids"] == ["cid:0-3", "cid:4-7"]
    assert out["reason"] == "pending_span_too_short"


def test_select_interval_segment_window_force_consumes_all_pending() -> None:
    rows = [
        {"segment_id": "cid:0-3", "start_idx": 0, "end_idx": 3, "happened_at": datetime(2026, 1, 1, tzinfo=UTC)},
        {"segment_id": "cid:4-7", "start_idx": 4, "end_idx": 7, "happened_at": datetime(2026, 1, 4, tzinfo=UTC)},
    ]

    out = _select_interval_segment_window(rows, interval_days=7, force=True)

    assert out["selected_segment_ids"] == ["cid:0-3", "cid:4-7"]
    assert out["remaining_segment_ids"] == []
    assert out["reason"] is None


def test_remap_edges_with_memory_ids_accepts_numbered_and_bracketed_refs() -> None:
    payload = [
        {"subject_id": "1", "predicate": "parallels", "object_id": "2", "confidence": 0.9},
        {"subject_id": "[2]", "predicate": "evokes", "object_id": "#1"},
    ]
    mapped = _remap_edges_with_memory_ids(
        payload,
        id_map={"1": "deadbeef", "2": "cafebabe"},
        include_confidence=True,
    )
    assert mapped == [
        {"subject_id": "deadbeef", "predicate": "parallels", "object_id": "cafebabe", "confidence": 0.9},
        {"subject_id": "cafebabe", "predicate": "evokes", "object_id": "deadbeef"},
    ]


def test_remap_edges_with_memory_ids_drops_unresolved_ids() -> None:
    payload = [{"subject_id": "partner-marker", "predicate": "shaped_by", "object_id": "52"}]
    mapped = _remap_edges_with_memory_ids(payload, id_map={}, include_confidence=False)
    assert mapped == []


def test_write_consolidation_outputs_preserves_remaining_pending_segment_ids() -> None:
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
        soul_id = "SoulX"
        user_id = "UserX"

        write_conversation_state(
            cid,
            sqlite_current_path=lambda _user, _soul: db_path,
            soul_id=soul_id,
            user_id=user_id,
            updates={"pending_segment_ids": ["ep:1-2", "ep:3-4"], "intentions_active": []},
        )

        deps = ConsolidationDeps(
            sqlite_current_path=lambda _user, _soul: db_path,
            sqlite_ensure_nonempty=sqlite_ensure_nonempty,
            sqlite_connect=sqlite_connect,
            sqlite_ensure_conversation_state_schema=sqlite_ensure_conversation_state_schema,
            conversation_state_row=conversation_state_row,
            conversation_state_from_row=lambda row, **kw: conversation_state_from_row(row),
            write_conversation_state=lambda conversation_id, *, soul_id, user_id, updates: write_conversation_state(
                conversation_id,
                sqlite_current_path=lambda _user, _soul: db_path,
                soul_id=soul_id,
                user_id=user_id,
                updates=updates,
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
                "selected_segment_ids": ["ep:1-2"],
                "remaining_segment_ids": ["ep:3-4"],
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
                "intention_actions": [],
            },
            conversation_id=cid,
            soul_id=soul_id,
            user_id=user_id,
        )

        assert result["consumed_segment_ids"] == ["ep:1-2"]
        assert result["remaining_segment_ids"] == ["ep:3-4"]
        assert result["state"]["pending_segment_ids"] == ["ep:3-4"]


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
            write_conversation_state=lambda conversation_id, *, soul_id, user_id, updates: write_conversation_state(
                conversation_id,
                sqlite_current_path=lambda _user, _soul: db_path,
                soul_id=soul_id,
                user_id=user_id,
                updates=updates,
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
            force=False,
            interval_days=7,
            stale_after=timedelta(seconds=3600),
        )
        assert out == {"status": "skip", "reason": "no_pending_segments"}

        check_con = sqlite_connect(db_path)
        try:
            check_con.row_factory = sqlite3.Row
            state = conversation_state_from_row(conversation_state_row(check_con, cid))
        finally:
            check_con.close()
        assert state is not None
        assert bool(state.get("consolidation_in_progress")) is False
        assert state.get("consolidation_started_at") is None


def _make_consolidation_deps(db_path: Path, tmp_dir: Path) -> ConsolidationDeps:
    """Helper: wires up a ConsolidationDeps pointing at a single db_path."""
    return ConsolidationDeps(
        sqlite_current_path=lambda _user, _soul: db_path,
        sqlite_ensure_nonempty=sqlite_ensure_nonempty,
        sqlite_connect=sqlite_connect,
        sqlite_ensure_conversation_state_schema=sqlite_ensure_conversation_state_schema,
        conversation_state_row=conversation_state_row,
        conversation_state_from_row=lambda row, **kw: conversation_state_from_row(row),
        write_conversation_state=lambda cid, *, soul_id, user_id, updates: write_conversation_state(
            cid,
            sqlite_current_path=lambda _u, _s: db_path,
            soul_id=soul_id,
            user_id=user_id,
            updates=updates,
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
        "intention_actions": [],
    }
    base.update(overrides)
    return base


def test_write_consolidation_outputs_clears_accumulators() -> None:
    """After a successful run the retrieval and prior-context accumulators are reset to []."""
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
        ss_before = _soul_state.read(check_con)
        check_con.close()
        assert ss_before["retrieval_ids_since_consolidation"] == ["mem-r1", "mem-r2"]
        assert ss_before["prior_context_ids_since_consolidation"] == ["mem-p1"]

        write_consolidation_outputs(
            _make_consolidation_deps(db_path, tmp_dir),
            _make_svc_stub(),
            inputs={"db_path": db_path},
            llm_results=_base_llm_results(),
            conversation_id=cid,
            soul_id=soul_id,
            user_id=user_id,
        )

        check_con2 = sqlite_connect(db_path)
        check_con2.row_factory = sqlite3.Row
        ss_after = _soul_state.read(check_con2)
        check_con2.close()
        assert ss_after["retrieval_ids_since_consolidation"] == []
        assert ss_after["prior_context_ids_since_consolidation"] == []


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
                life_goal_add=["new goal"],
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

        assert [(row[0], row[1]) for row in rows] == [("new goal", "active"), ("old goal", "removed")]
        assert old_rows == []


def test_write_consolidation_outputs_created_ephemeral_survives_turns_until_next_consolidation() -> None:
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

        write_conversation_state(
            "conv-intentions",
            sqlite_current_path=lambda _u, _s: db_path,
            soul_id="SoulI",
            user_id="UserI",
            updates={"pending_segment_ids": ["ep:1"], "intentions_active": []},
        )

        write_consolidation_outputs(
            _make_consolidation_deps(db_path, tmp_dir),
            _make_svc_stub(),
            inputs={"db_path": db_path},
            llm_results=_base_llm_results(
                intention_actions=[{"type": "create", "id": "new-thread", "text": "Follow the thread."}],
            ),
            conversation_id="conv-intentions",
            soul_id="SoulI",
            user_id="UserI",
        )

        check_con = sqlite_connect(db_path)
        try:
            check_con.row_factory = sqlite3.Row
            before_turn = _soul_state.read(check_con)["intentions_active"]
        finally:
            check_con.close()
        assert "new-thread" in {item["id"] for item in before_turn["items"]}

        from app.services.intention_state import apply_intention_turn_maintenance

        after_turn = apply_intention_turn_maintenance(before_turn)
        after_turn_items = {item["id"]: item for item in after_turn["items"]}
        assert after_turn_items["new-thread"]["ephemeral"] is True


def test_write_consolidation_outputs_drops_unpromoted_old_ephemeral() -> None:
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

        write_conversation_state(
            "conv-drop-eph",
            sqlite_current_path=lambda _u, _s: db_path,
            soul_id="SoulE",
            user_id="UserE",
            updates={
                "pending_segment_ids": ["ep:1"],
                "intentions_active": {
                    "items": [
                        {"id": "old-eph", "text": "Old ephemeral", "ephemeral": True},
                        {"id": "stable", "text": "Stable", "priority": 8.0, "ephemeral": False},
                    ]
                },
            },
        )

        write_consolidation_outputs(
            _make_consolidation_deps(db_path, tmp_dir),
            _make_svc_stub(),
            inputs={"db_path": db_path, "last_consolidation_at": "2026-06-01T00:00:00+00:00"},
            llm_results=_base_llm_results(),
            conversation_id="conv-drop-eph",
            soul_id="SoulE",
            user_id="UserE",
        )

        check_con = sqlite_connect(db_path)
        try:
            check_con.row_factory = sqlite3.Row
            items = {
                item["id"]: item
                for item in _soul_state.read(check_con)["intentions_active"]["items"]
            }
        finally:
            check_con.close()

        assert "old-eph" not in items
        assert "stable" in items


def test_write_consolidation_outputs_db_failure_produces_no_companion_memory() -> None:
    """If the DB transaction fails, companion memory must NOT be created (it runs after the state write)."""
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

        def _fake_write_state(cid, *, soul_id, user_id, updates):
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
