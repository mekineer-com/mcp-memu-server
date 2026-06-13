import pytest

from app.services.consolidation import _format_segment_block_for_prompt
from app.services.consolidation import _remap_edges_with_memory_ids
from app.services.consolidation import _parse_consolidation_xml
from app.services.consolidation import run_consolidation_llm
from app.services.consolidation import gather_consolidation_inputs
from app.services.consolidation import ConsolidationDeps, write_consolidation_outputs
from app.services.graph_edges import invalidate_memory_edges, write_memory_edges
from app.db import json_to_db, normalize_text_list, sqlite_connect, sqlite_ensure_conversation_state_schema, sqlite_ensure_nonempty
from app.services.state import conversation_state_from_row, conversation_state_row, write_conversation_state
from app.services import soul_state as _soul_state
import sqlite3
import tempfile
from pathlib import Path
from datetime import timedelta
from unittest.mock import patch, MagicMock


def test_parse_consolidation_xml_edges_and_write_helpers() -> None:
    xml = """
<consolidation>
  <narrative_self>n</narrative_self>
  <life_goals></life_goals>
  <companion_memory>c</companion_memory>
  <edges>
    <edge>
      <subject_id>mem_a</subject_id>
      <predicate>parallels</predicate>
      <object_id>mem_b</object_id>
      <confidence>0.61</confidence>
    </edge>
    <edge>
      <subject_id>mem_c</subject_id>
      <predicate>shaped_by</predicate>
      <object_id>mem_d</object_id>
      <confidence>not-a-float</confidence>
    </edge>
    <edge>
      <subject_id>mem_e</subject_id>
      <predicate>evokes</predicate>
    </edge>
    <invalidate>
      <subject_id>mem_x</subject_id>
      <predicate>conflicts_with</predicate>
      <object_id>mem_y</object_id>
    </invalidate>
    <invalidate>
      <subject_id>mem_x</subject_id>
      <predicate></predicate>
      <object_id>mem_y</object_id>
    </invalidate>
  </edges>
</consolidation>
"""
    parsed = _parse_consolidation_xml(xml)
    assert len(parsed["edges"]) == 2
    assert parsed["edges"][0]["confidence"] == 0.61
    assert "confidence" not in parsed["edges"][1]
    assert parsed["edge_invalidations"] == [
        {"subject_id": "mem_x", "predicate": "conflicts_with", "object_id": "mem_y"}
    ]

    class _TripleRepoStub:
        def __init__(self) -> None:
            self.added: list[object] = []
            self.invalidated: list[tuple[str, str, str]] = []

        def add(self, triple: object, user_data: dict[str, str] | None = None) -> object:
            self.added.append(triple)
            return triple

        def invalidate(self, subject_id: str, predicate: str, object_id: str, scope: dict | None = None) -> None:
            self.invalidated.append((subject_id, predicate, object_id))

    repo = _TripleRepoStub()
    wrote = write_memory_edges(repo, parsed["edges"], scope={"user_id": "u", "soul_id": "s"})
    invalidated = invalidate_memory_edges(repo, parsed["edge_invalidations"], scope={"user_id": "u", "soul_id": "s"})

    assert wrote == 2
    assert invalidated == 1
    assert repo.invalidated == [("mem_x", "conflicts_with", "mem_y")]
    assert getattr(repo.added[0], "confidence") == 0.61
    # Edge 2's <confidence> was invalid and stripped at parse time; no AI judgment,
    # so the stored confidence is NULL (no artificial fallback).
    assert getattr(repo.added[1], "confidence") is None


def test_parse_consolidation_xml_accepts_root_attributes() -> None:
    parsed = _parse_consolidation_xml(
        """
<consolidation version="1">
  <narrative_self>n</narrative_self>
  <life_goals></life_goals>
  <intentions>
    <create id="stay-present" text="Stay present." />
  </intentions>
  <edges></edges>
  <companion_memory>c</companion_memory>
</consolidation>
"""
    )
    assert parsed["narrative_self"] == "n"
    assert parsed["intention_actions"] == [{"type": "create", "id": "stay-present", "text": "Stay present."}]


def test_parse_consolidation_xml_accepts_variant_intention_shapes() -> None:
    parsed = _parse_consolidation_xml(
        """
<consolidation>
  <narrative_self>n</narrative_self>
  <life_goals></life_goals>
  <intentions>
    <boost intention_id="keep-going" />
    <promote>ephemeral-thread</promote>
    <create text="Explore embodied rituals." />
    <create id="hold-gentle">Hold gentleness under pressure.</create>
    <annul id="old-thread" status="deleted" />
  </intentions>
  <edges></edges>
  <companion_memory>c</companion_memory>
</consolidation>
"""
    )
    assert parsed["intention_actions"] == [
        {"type": "boost", "target_id": "keep-going", "amount": 1},
        {"type": "promote", "target_id": "ephemeral-thread"},
        {"type": "create", "id": "explore-embodied-rituals", "text": "Explore embodied rituals."},
        {"type": "create", "id": "hold-gentle", "text": "Hold gentleness under pressure."},
        {"type": "annul", "intention_id": "old-thread", "status": "deleted", "note": ""},
    ]



@pytest.mark.asyncio
async def test_run_consolidation_llm_retries_once_on_missing_root() -> None:
    class _Svc:
        def __init__(self) -> None:
            self.calls = 0

        def _escape_prompt_value(self, value):
            return str(value)

        async def chat(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return "not xml"
            return """
<consolidation>
  <narrative_self>steady</narrative_self>
  <life_goals></life_goals>
  <intentions>
    <create id="new-thread" text="Follow this emerging thread" />
  </intentions>
  <edges></edges>
  <companion_memory>noted</companion_memory>
</consolidation>
"""

        async def embed(self, *_args, **_kwargs):
            return []

    svc = _Svc()
    out = await run_consolidation_llm(
        svc,
        inputs={
            "categories": [],
            "active_life_goals": [],
            "removed_life_goals": [],
            "intention_activity": [],
            "segment_inputs": [],
            "narrative_self": None,
            "state": {"intentions_active": {"items": [{"id": "relax", "text": "Relax", "kind": "relax"}]}},
            "retrieved_memories": [],
        },
        soul_id="Echo",
        llm_profile=None,
    )
    assert svc.calls == 2
    assert out["intention_actions"] == [
        {"type": "create", "id": "new-thread", "text": "Follow this emerging thread"}
    ]


@pytest.mark.asyncio
async def test_run_consolidation_llm_retries_once_on_malformed_xml() -> None:
    class _Svc:
        def __init__(self) -> None:
            self.calls = 0

        def _escape_prompt_value(self, value):
            return str(value)

        async def chat(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return "<consolidation><narrative_self>steady</narrative_self>"
            return """
<consolidation>
  <narrative_self>steady</narrative_self>
  <life_goals></life_goals>
  <intentions>
    <create id="new-thread" text="Follow this emerging thread" />
  </intentions>
  <edges></edges>
  <companion_memory>noted</companion_memory>
</consolidation>
"""

        async def embed(self, *_args, **_kwargs):
            return []

    svc = _Svc()
    out = await run_consolidation_llm(
        svc,
        inputs={
            "categories": [],
            "active_life_goals": [],
            "removed_life_goals": [],
            "intention_activity": [],
            "segment_inputs": [],
            "narrative_self": None,
            "state": {"intentions_active": {"items": [{"id": "relax", "text": "Relax", "kind": "relax"}]}},
            "retrieved_memories": [],
        },
        soul_id="Echo",
        llm_profile=None,
    )
    assert svc.calls == 2
    assert out["narrative_self"] == "steady"


def test_format_segment_block_for_prompt_shows_memory_ids() -> None:
    id_map: dict[str, str] = {}
    counter: list[int] = [1]
    out = _format_segment_block_for_prompt(
        [
            {
                "segment_id": "ep:1-2",
                "excerpt": "hello",
                "memory_summaries": [
                    {"id": "mem_1", "summary": "one"},
                    {"id": "mem_2", "summary": "two"},
                ],
            }
        ],
        id_map,
        counter,
    )
    assert "memories:" in out
    assert "- [1] [memory] one" in out
    assert "- [2] [memory] two" in out
    assert id_map == {"1": "mem_1", "2": "mem_2"}


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


def test_write_consolidation_outputs_clears_pending_segment_ids() -> None:
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
            updates={"pending_segment_ids": ["ep:1-2"], "intentions_active": []},
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
            inputs={"db_path": db_path},
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

        assert result["state"]["pending_segment_ids"] == []


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


@pytest.mark.asyncio
async def test_run_consolidation_llm_strips_relax_boost_from_intention_actions() -> None:
    """Model returns <boost target_id="relax" /> which is a no-op in apply_intention_action.
    It must be stripped so the caller sees an empty list, not a phantom action."""
    class _Svc:
        def _escape_prompt_value(self, value): return str(value)
        async def chat(self, *_a, **_kw):
            return """
<consolidation>
  <narrative_self>steady</narrative_self>
  <life_goals></life_goals>
  <intentions>
    <boost target_id="relax" />
  </intentions>
  <edges></edges>
  <companion_memory>noted</companion_memory>
</consolidation>
"""
        async def embed(self, *_a, **_kw): return []

    out = await run_consolidation_llm(
        _Svc(),
        inputs={
            "categories": [], "active_life_goals": [], "removed_life_goals": [],
            "intention_activity": [], "segment_inputs": [], "narrative_self": None,
            "state": {"intentions_active": None}, "retrieved_memories": [],
        },
        soul_id="Echo",
        llm_profile=None,
    )
    assert out["intention_actions"] == [], "relax boost must be stripped"


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
