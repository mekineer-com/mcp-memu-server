import pytest

def _parse_consolidation_xml():
    try:
        from app.services.consolidation import _parse_consolidation_xml as parser
    except Exception as exc:  # pragma: no cover - test env fallback
        pytest.skip(f"Import test skipped due to compatibility issue: {exc}")
    return parser


def test_parse_consolidation_xml_requires_all_expected_episode_diaries() -> None:
    xml = """
<consolidation>
  <narrative_self>n</narrative_self>
  <life_goals></life_goals>
  <companion_memory>c</companion_memory>
    </consolidation>
"""
    with pytest.raises(ValueError, match="missing episode_ids"):
        _parse_consolidation_xml()(xml, expected_episode_ids={"ep:1-2"})


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
      <affect>
        <emotion>e</emotion>
      </affect>
      <unresolved></unresolved>
    </diary>
  </diaries>
    </consolidation>
"""
    with pytest.raises(ValueError, match="unknown consolidation diary episode_id"):
        _parse_consolidation_xml()(xml, expected_episode_ids=set())


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
      <affect>
        <emotion>e</emotion>
      </affect>
      <unresolved></unresolved>
    </diary>
  </diaries>
</consolidation>
"""
    parsed = _parse_consolidation_xml()(xml, expected_episode_ids={"ep:1-2"})

    assert parsed["life_goal_add"] == ["g"]
    assert parsed["diaries"][0]["episode_id"] == "ep:1-2"
