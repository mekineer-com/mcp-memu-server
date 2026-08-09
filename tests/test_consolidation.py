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
from app.services.consolidation import _remap_edges_with_memory_ids
from app.services.consolidation import _parse_consolidation_xml
from app.services.consolidation import _select_interval_segment_window
from app.services.consolidation import gather_consolidation_inputs
from app.services.consolidation import prepare_dossier_consolidation_context
from app.services.consolidation import run_consolidation_llm
from app.services.graph_edges import invalidate_memory_edges, write_memory_edges
from app.services.state import conversation_state_from_row, conversation_state_row, write_conversation_state


class _DossierContextService:
    def __init__(self, *, due_ids=(), profiles=("default", "revision"), stale_count=0) -> None:
        self.memorize_config = SimpleNamespace(category_update_llm_profile="revision")
        self.llm_profiles = SimpleNamespace(profiles={name: object() for name in profiles})
        self.due = [SimpleNamespace(id=dossier_id) for dossier_id in due_ids]
        self.stale_count = stale_count
        self.calls: list[tuple] = []

    def list_due_dossiers(self, scope):
        self.calls.append(("due", scope))
        return self.due

    def prepare_dossier_revision(self, dossier_id, scope, **context):
        self.calls.append(("prepare", dossier_id, scope, context))
        return {"dossier": SimpleNamespace(id=dossier_id)}

    async def generate_dossier_revision(self, bundle):
        dossier_id = bundle["dossier"].id
        self.calls.append(("generate", dossier_id))
        return {"dossier_id": dossier_id}

    async def apply_dossier_revision(self, bundle, decision, scope):
        dossier_id = bundle["dossier"].id
        self.calls.append(("apply", dossier_id, decision, scope))
        if self.stale_count:
            self.stale_count -= 1
            raise DossierRevisionStaleError("changed")

    def build_dossier_index(self, scope):
        self.calls.append(("index", scope))
        return "- Health: Current health"

    def list_dossiers_for_segments(self, scope, *, segment_ids):
        self.calls.append(("relevant", scope, list(segment_ids)))
        return [SimpleNamespace(name="Health", description="Current health", summary="Body [M4].")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profiles", "consolidation_profile", "missing"),
    [
        (("default",), None, "revision"),
        (("default", "revision"), "reflection", "reflection"),
    ],
)
async def test_dossier_context_preflights_both_profiles(
    profiles, consolidation_profile, missing
) -> None:
    svc = _DossierContextService(due_ids=("first",), profiles=profiles)
    with pytest.raises(KeyError, match=missing):
        await prepare_dossier_consolidation_context(
            svc,
            inputs={
                "active_life_goals": [],
                "removed_life_goals": [],
                "selected_segment_ids": ["segment-1"],
            },
            soul_id="TestSoul",
            user_id="TestUser",
            consolidation_llm_profile=consolidation_profile,
        )
    assert svc.calls == []


@pytest.mark.asyncio
async def test_dossier_context_retries_one_stale_and_preserves_due_order() -> None:
    svc = _DossierContextService(due_ids=("first", "second"), stale_count=1)
    inputs = {
        "narrative_self": "I am steady.",
        "active_life_goals": ["Stay curious"],
        "removed_life_goals": ["Old goal"],
        "selected_segment_ids": ["segment-2", "segment-3"],
    }

    result = await prepare_dossier_consolidation_context(
        svc,
        inputs=inputs,
        soul_id="TestSoul",
        user_id="TestUser",
        consolidation_llm_profile=None,
    )

    assert result is inputs
    assert [call[1] for call in svc.calls if call[0] == "generate"] == [
        "first",
        "first",
        "second",
    ]
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
async def test_dossier_context_aborts_after_second_stale() -> None:
    svc = _DossierContextService(due_ids=("first", "second"), stale_count=2)
    with pytest.raises(DossierRevisionStaleError):
        await prepare_dossier_consolidation_context(
            svc,
            inputs={
                "active_life_goals": [],
                "removed_life_goals": [],
                "selected_segment_ids": ["segment-1"],
            },
            soul_id="TestSoul",
            user_id="TestUser",
            consolidation_llm_profile=None,
        )
    assert [call[1] for call in svc.calls if call[0] == "generate"] == ["first", "first"]
    assert not any(call[0] in {"index", "relevant"} for call in svc.calls)


@pytest.mark.asyncio
async def test_dossier_context_without_due_dossiers_still_builds_context() -> None:
    svc = _DossierContextService()
    inputs = {
        "active_life_goals": [],
        "removed_life_goals": [],
        "selected_segment_ids": ["segment-4"],
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
        f"{prose}"
    )
    assert "[M12]" in rendered
    assert "[1]" not in rendered

    with pytest.raises(ValueError, match="exceeds 100000 tokens"):
        _format_dossier_context_for_prompt(
            "",
            [SimpleNamespace(name="Large", description="Large", summary="word " * 75_001)],
        )


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
            "prior_context_memory_items": [],
        },
        soul_id="Echo",
        llm_profile=None,
    )
    assert svc.calls == 2
    assert out["intention_actions"] == [
        {"type": "create", "id": "new-thread", "text": "Follow this emerging thread"}
    ]


@pytest.mark.asyncio
async def test_run_consolidation_llm_includes_all_chat_history() -> None:
    class _Svc:
        def __init__(self) -> None:
            self.prompt = ""

        def _escape_prompt_value(self, value):
            return str(value)

        async def chat(self, prompt, *_args, **_kwargs):
            self.prompt = str(prompt)
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
    await run_consolidation_llm(
        svc,
        inputs={
            "categories": [],
            "active_life_goals": [],
            "removed_life_goals": [],
            "intention_activity": [],
            "segment_inputs": [],
            "all_chat_history": "My WhatsApp Conversations:\n\n[dm][Contact A]\n[Contact A] cross hello",
            "narrative_self": None,
            "state": {"intentions_active": {"items": [{"id": "relax", "text": "Relax", "kind": "relax"}]}},
            "prior_context_memory_items": [],
        },
        soul_id="Echo",
        llm_profile=None,
    )

    assert "# My conversations" not in svc.prompt
    assert "My WhatsApp Conversations:" in svc.prompt
    assert "[dm][Contact A]" in svc.prompt
    assert "[Contact A] cross hello" in svc.prompt


@pytest.mark.asyncio
async def test_run_consolidation_llm_dedupes_segment_memory_items_against_prior_context() -> None:
    class _Svc:
        def __init__(self) -> None:
            self.prompt = ""

        def _escape_prompt_value(self, value):
            return str(value)

        async def chat(self, prompt, *_args, **_kwargs):
            self.prompt = str(prompt)
            return """
<consolidation>
  <narrative_self>steady</narrative_self>
  <life_goals></life_goals>
  <intentions></intentions>
  <edges>
    <edge>
      <subject_id>1</subject_id>
      <predicate>parallels</predicate>
      <object_id>3</object_id>
    </edge>
  </edges>
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
            "segment_inputs": [
                {
                    "segment_id": "cid:0-2",
                    "memory_summaries": [
                        {"id": "mem_dup", "summary": "duplicate memory", "memory_type": "behavior"},
                        {"id": "mem_segment", "summary": "segment only memory", "memory_type": "knowledge"},
                    ],
                }
            ],
            "all_chat_history": "(none)",
            "narrative_self": "steady",
            "state": {"intentions_active": None},
            "prior_context_memory_items": [
                {"id": "mem_dup", "summary": "duplicate memory", "memory_type": "behavior"},
                {"id": "mem_prior", "summary": "prior only memory", "memory_type": "preference"},
            ],
        },
        soul_id="Echo",
        llm_profile=None,
    )

    assert svc.prompt.count("duplicate memory") == 1
    assert "[1] [behavior] duplicate memory" in svc.prompt
    assert "- [3] [knowledge] segment only memory" in svc.prompt
    assert out["edges"] == [
        {"subject_id": "mem_dup", "predicate": "parallels", "object_id": "mem_segment"}
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
            "prior_context_memory_items": [],
        },
        soul_id="Echo",
        llm_profile=None,
    )
    assert svc.calls == 2
    assert out["narrative_self"] == "steady"


def test_format_segment_memory_items_for_prompt_shows_memory_ids() -> None:
    id_map: dict[str, str] = {}
    counter: list[int] = [1]
    out = _format_segment_memory_items_for_prompt(
        [
            {
                "segment_id": "ep:1-2",
                "memory_summaries": [
                    {"id": "mem_1", "summary": "one", "memory_type": "behavior"},
                    {"id": "mem_2", "summary": "two", "memory_type": "knowledge"},
                ],
            }
        ],
        id_map,
        counter,
    )
    assert "Key:" in out
    assert "Segment 1" not in out
    assert "Conversation " not in out
    assert "Related memories:" not in out
    assert "- [1] [behavior] one" in out
    assert "- [2] [knowledge] two" in out
    assert id_map == {"1": "mem_1", "2": "mem_2"}


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
            "state": {"intentions_active": None}, "prior_context_memory_items": [],
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
