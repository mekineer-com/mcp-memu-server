# Historical schema-upgrade test excerpt; reference only.
# Imports and helpers are in the original Git version of tests/test_soul_summaries.py.

def test_legacy_migration_persists_approved_baseline_once(tmp_path) -> None:
    path = tmp_path / "soul.db"
    con = sqlite3.connect(path)
    con.execute(
        """
CREATE TABLE soul_state (
    id INTEGER PRIMARY KEY,
    narrative_self TEXT,
    all_categories_summary TEXT,
    all_categories_summary_previous TEXT,
    all_categories_summary_approved TEXT,
    memory_cache JSON DEFAULT '[]',
    intentions_active JSON,
    retrieve_rewrite_angle INTEGER DEFAULT 0,
    retrieval_ids_since_consolidation JSON DEFAULT '[]',
    prior_context_ids_since_consolidation JSON DEFAULT '[]',
    last_consolidation_at DATETIME,
    updated_at DATETIME
)
"""
    )
    con.execute(
        "INSERT INTO soul_state (id, narrative_self, all_categories_summary, "
        "all_categories_summary_previous, all_categories_summary_approved) "
        "VALUES (1, 'old self', 'old cats', 'older cats', 'approved cats')"
    )
    con.execute("UPDATE soul_state SET intentions_active = '[]' WHERE id = 1")
    con.commit()
    soul_state.ensure_schema(con)
    con.close()

    check = sqlite3.connect(path)
    check.row_factory = sqlite3.Row
    state = soul_state.read(check)
    assert state["narrative_self_approved"] == "old self"
    assert "all_categories_summary" not in state
    legacy_values = check.execute(
        "SELECT all_categories_summary, all_categories_summary_previous, "
        "all_categories_summary_approved FROM soul_state WHERE id = 1"
    ).fetchone()
    assert tuple(legacy_values) == ("old cats", "older cats", "approved cats")
    check.execute("UPDATE soul_state SET narrative_self = 'new self' WHERE id = 1")
    check.commit()
    soul_state.ensure_schema(check)
    assert soul_state.read(check)["narrative_self_approved"] == "old self"
    check.close()
