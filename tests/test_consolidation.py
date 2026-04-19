import pytest

from app.services.consolidation import _format_episode_block_for_prompt
from app.services.consolidation import _parse_consolidation_xml
from app.services.graph_edges import invalidate_memory_edges, write_memory_edges

def test_parse_consolidation_xml_requires_all_expected_episode_diaries() -> None:
    xml = """
	<consolidation>
  <narrative_self>n</narrative_self>
  <life_goals></life_goals>
  <companion_memory>c</companion_memory>
    </consolidation>
	"""
    with pytest.raises(ValueError, match="missing episode_ids"):
        _parse_consolidation_xml(xml, expected_episode_ids={"ep:1-2"})


def test_parse_consolidation_xml_rejects_unexpected_episode_diary() -> None:
    xml = """
<consolidation>
  <narrative_self>n</narrative_self>
  <life_goals></life_goals>
  <companion_memory>c</companion_memory>
  <diaries>
    <diary>
      <episode_id>ep:1-2</episode_id>
      <prose>p</prose>
      <unresolved></unresolved>
    </diary>
  </diaries>
    </consolidation>
	"""
    with pytest.raises(ValueError, match="unknown consolidation diary episode_id"):
        _parse_consolidation_xml(xml, expected_episode_ids=set())


def test_parse_consolidation_xml_accepts_exact_episode_set() -> None:
    xml = """
<consolidation>
  <narrative_self>n</narrative_self>
  <life_goals>
    <add>g</add>
  </life_goals>
  <companion_memory>c</companion_memory>
  <diaries>
    <diary>
      <episode_id>ep:1-2</episode_id>
      <prose>p</prose>
      <unresolved></unresolved>
    </diary>
  </diaries>
	</consolidation>
"""
    parsed = _parse_consolidation_xml(xml, expected_episode_ids={"ep:1-2"})

    assert parsed["life_goal_add"] == ["g"]
    assert parsed["diaries"][0]["episode_id"] == "ep:1-2"


def test_parse_consolidation_xml_edges_and_write_helpers() -> None:
    xml = """
<consolidation>
  <narrative_self>n</narrative_self>
  <life_goals></life_goals>
  <companion_memory>c</companion_memory>
  <diaries>
    <diary>
      <episode_id>ep:1-2</episode_id>
      <prose>p</prose>
      <unresolved></unresolved>
    </diary>
  </diaries>
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
    parsed = _parse_consolidation_xml(xml, expected_episode_ids={"ep:1-2"})
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


def test_parse_consolidation_xml_shaped_by_hints_in_diary() -> None:
    xml = """
<consolidation>
  <narrative_self>n</narrative_self>
  <life_goals></life_goals>
  <companion_memory>c</companion_memory>
  <diaries>
    <diary>
      <episode_id>ep:1-2</episode_id>
      <prose>p</prose>
      <unresolved></unresolved>
      <shaped_by_hints>
        <memory_id>mem_aaa</memory_id>
        <memory_id>mem_bbb</memory_id>
        <memory_id>mem_aaa</memory_id>
      </shaped_by_hints>
    </diary>
  </diaries>
</consolidation>
"""
    parsed = _parse_consolidation_xml(xml, expected_episode_ids={"ep:1-2"})
    diary = parsed["diaries"][0]
    # deduped
    assert diary["shaped_by_hints"] == ["mem_aaa", "mem_bbb"]
    # companion hints absent → empty list
    assert parsed["companion_shaped_by_hints"] == []


def test_parse_consolidation_xml_companion_shaped_by_hints() -> None:
    xml = """
<consolidation>
  <narrative_self>n</narrative_self>
  <life_goals></life_goals>
  <companion_memory>c</companion_memory>
  <diaries>
    <diary>
      <episode_id>ep:3-4</episode_id>
      <prose>p</prose>
      <unresolved></unresolved>
    </diary>
  </diaries>
  <companion_shaped_by_hints>
    <memory_id>mem_x</memory_id>
    <memory_id>  </memory_id>
    <memory_id>mem_x</memory_id>
  </companion_shaped_by_hints>
</consolidation>
"""
    parsed = _parse_consolidation_xml(xml, expected_episode_ids={"ep:3-4"})
    # empty / whitespace IDs dropped, dupes removed
    assert parsed["companion_shaped_by_hints"] == ["mem_x"]
    # diary entry with no hints block → empty list
    assert parsed["diaries"][0]["shaped_by_hints"] == []


def test_parse_consolidation_xml_missing_hints_blocks_give_empty_lists() -> None:
    xml = """
<consolidation>
  <narrative_self>n</narrative_self>
  <life_goals></life_goals>
  <companion_memory>c</companion_memory>
  <diaries>
    <diary>
      <episode_id>ep:5-6</episode_id>
      <prose>p</prose>
      <unresolved></unresolved>
    </diary>
  </diaries>
</consolidation>
"""
    parsed = _parse_consolidation_xml(xml, expected_episode_ids={"ep:5-6"})
    assert parsed["diaries"][0]["shaped_by_hints"] == []
    assert parsed["companion_shaped_by_hints"] == []


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
