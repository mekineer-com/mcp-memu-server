from app.services.consolidation import _format_episode_block_for_prompt
from app.services.consolidation import _parse_consolidation_xml
from app.services.consolidation import ConsolidationDeps, write_consolidation_outputs
from app.services.graph_edges import invalidate_memory_edges, write_memory_edges
from app.db import json_to_db, normalize_text_list, sqlite_connect, sqlite_ensure_conversation_state_schema, sqlite_ensure_nonempty
from app.services.state import conversation_state_from_row, conversation_state_row, write_conversation_state
import sqlite3
import tempfile
from pathlib import Path


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
    assert getattr(repo.added[1], "confidence") == 0.8


def test_format_episode_block_for_prompt_shows_memory_ids() -> None:
    out = _format_episode_block_for_prompt(
        [
            {
                "episode_id": "ep:1-2",
                "excerpt": "hello",
                "memory_summaries": ["[mem_1] one", "[mem_2] two"],
            }
        ]
    )
    assert "Extracted memory summaries:" in out
    assert "- [mem_1] one" in out
    assert "- [mem_2] two" in out


def test_write_consolidation_outputs_clears_pending_episode_ids() -> None:
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
            sqlite_dir=tmp_dir,
            soul_id=soul_id,
            user_id=user_id,
            updates={"pending_episode_ids": ["ep:1-2"], "intentions_active": []},
        )

        deps = ConsolidationDeps(
            sqlite_current_path=lambda _user, _soul: db_path,
            sqlite_ensure_nonempty=sqlite_ensure_nonempty,
            sqlite_connect=sqlite_connect,
            sqlite_ensure_conversation_state_schema=sqlite_ensure_conversation_state_schema,
            conversation_state_row=conversation_state_row,
            conversation_state_from_row=conversation_state_from_row,
            write_conversation_state=lambda conversation_id, *, soul_id, user_id, updates: write_conversation_state(
                conversation_id,
                sqlite_current_path=lambda _user, _soul: db_path,
                sqlite_dir=tmp_dir,
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
            inputs={"db_path": db_path, "self_model_id": None},
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

        assert result["state"]["pending_episode_ids"] == []
