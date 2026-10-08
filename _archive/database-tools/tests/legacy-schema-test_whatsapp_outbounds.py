# Historical schema-upgrade test excerpt; reference only.
# Imports and helpers are in the original Git version of tests/test_main.py.

def test_whatsapp_outbounds_schema_migration_idempotent(tmp_path: Path) -> None:
    """ALTER TABLE on a DB that already has the column must not raise."""
    import sqlite3 as _sqlite3
    db_path = tmp_path / "existing.db"
    con = _sqlite3.connect(str(db_path))
    con.row_factory = _sqlite3.Row
    # Create table without media_path first, simulating a pre-migration DB.
    con.execute("""
CREATE TABLE whatsapp_pending_outbounds (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    soul_id TEXT NOT NULL,
    origin_conversation_id TEXT NOT NULL,
    target TEXT NOT NULL,
    target_conversation_id TEXT,
    response_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    claimed_at TEXT,
    claimed_by TEXT,
    sent_at TEXT,
    failed_at TEXT,
    provider_message_id TEXT,
    last_error TEXT,
    metadata_json TEXT
)
""")
    con.commit()
    # Running the schema function twice must not raise.
    main._ensure_whatsapp_outbounds_schema(con)
    main._ensure_whatsapp_outbounds_schema(con)
    # Confirm the column now exists.
    cols = {row[1] for row in con.execute("PRAGMA table_info(whatsapp_pending_outbounds)")}
    assert "media_path" in cols
    con.close()


# --- normal-turn attachment delivery ---
