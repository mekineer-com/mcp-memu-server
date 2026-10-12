import json
import sqlite3

import pytest

import migrate_release


@pytest.mark.parametrize("legacy", [False, True])
def test_subconscious_insight_migration_preserves_state_and_memory_metadata(tmp_path, monkeypatch, legacy):
    database = tmp_path / "FictionalSoul.db"
    field = "apimw_message_to_self" if legacy else "apimw_subconscious_insight"
    with sqlite3.connect(database) as connection:
        connection.execute(f"CREATE TABLE soul_state (id INTEGER PRIMARY KEY, {field} TEXT)")
        connection.execute("INSERT INTO soul_state VALUES (1, '[subconscious] A quiet pattern')")
        connection.execute("CREATE TABLE memory_items (id INTEGER PRIMARY KEY, extra JSON)")
        connection.execute("INSERT INTO memory_items VALUES (1, ?)", (json.dumps({field: True, "retained": "evidence"}),))
        connection.execute("INSERT INTO memory_items VALUES (2, ?)", (json.dumps({"retained": "other"}),))
    monkeypatch.setattr(migrate_release, "_initialize_schema", lambda *_args: None)

    migrate_release.migrate_database(database, "fictional-profile")
    migrate_release.migrate_database(database, "fictional-profile")

    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(soul_state)")}
        assert "apimw_message_to_self" not in columns
        assert connection.execute("SELECT apimw_subconscious_insight FROM soul_state").fetchone()[0] == "[subconscious] A quiet pattern"
        assert json.loads(connection.execute("SELECT extra FROM memory_items WHERE id = 1").fetchone()[0]) == {
            "apimw_subconscious_insight": True, "retained": "evidence",
        }
        assert json.loads(connection.execute("SELECT extra FROM memory_items WHERE id = 1").fetchone()[0])["apimw_subconscious_insight"] is True
        assert json.loads(connection.execute("SELECT extra FROM memory_items WHERE id = 2").fetchone()[0]) == {"retained": "other"}


def test_release_upgrade_runs_before_current_schema_initialization(tmp_path, monkeypatch):
    database = tmp_path / "FictionalSoul.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE fictional_value (value TEXT)")
        connection.execute("INSERT INTO fictional_value VALUES ('preserved')")

    def initialize(path, _profile):
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT value, new_value FROM fictional_value").fetchone() == ("preserved", "upgraded")

    def upgrade(connection):
        connection.execute("ALTER TABLE fictional_value ADD COLUMN new_value TEXT DEFAULT 'upgraded'")

    monkeypatch.setattr(migrate_release, "_initialize_schema", initialize)
    migrate_release.migrate_database(database, "fictional-profile", (("fictional-format-v1", upgrade),))
    migrate_release.migrate_database(database, "fictional-profile", (("fictional-format-v1", upgrade),))


def test_release_data_migration_is_marked_and_rerunnable(tmp_path, monkeypatch):
    database = tmp_path / "FictionalSoul.db"
    monkeypatch.setattr(migrate_release, "_initialize_schema", lambda path, _profile: path.touch())
    calls = []

    def migration(connection):
        calls.append(True)
        connection.execute("CREATE TABLE fictional_value (value TEXT)")

    migrations = (("fictional-v1", migration),)
    migrate_release.migrate_database(database, "gemini-embedding-2-preview:3072", migrations)
    migrate_release.migrate_database(database, "gemini-embedding-2-preview:3072", migrations)

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT migration_id FROM openalma_release_migrations"
        ).fetchall() == [("fictional-v1",)]
    assert calls == [True]


def test_failed_release_data_migration_rolls_back_schema_and_ledger(tmp_path, monkeypatch):
    database = tmp_path / "FictionalSoul.db"
    monkeypatch.setattr(migrate_release, "_initialize_schema", lambda path, _profile: path.touch())

    def migration(connection):
        connection.execute("CREATE TABLE unfinished_value (value TEXT)")
        raise RuntimeError("failed")

    with pytest.raises(RuntimeError, match="failed"):
        migrate_release.migrate_database(database, "gemini-embedding-2:3072", (("bad-v1", migration),))

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'unfinished_value'"
        ).fetchone() is None
        assert connection.execute(
            "SELECT migration_id FROM openalma_release_migrations"
        ).fetchall() == []


def test_soul_databases_excludes_base_hidden_and_symlink(tmp_path):
    base = tmp_path / "memu.db"
    soul = tmp_path / "Fictional Soul.db"
    hidden = tmp_path / ".replacement.db"
    procedural = tmp_path / "procedural.db"
    for path in (base, soul, hidden, procedural):
        path.touch()
    (tmp_path / "Linked.db").symlink_to(soul)
    config = {
        "llm": {"embedding": {}},
        "storage": {
            "sqlite_dir": str(tmp_path),
            "metadata_store": {"dsn": f"sqlite:///{base}"},
        },
        "procedural": {"db_path": str(procedural)},
    }

    assert migrate_release.soul_databases(config) == [soul]


def test_release_migration_accepts_no_souls_and_derives_profile(tmp_path, monkeypatch, capsys):
    config = {
        "llm": {"embedding": {}},
        "storage": {
            "sqlite_dir": str(tmp_path),
            "metadata_store": {"dsn": f"sqlite:///{tmp_path / 'memu.db'}"},
        },
    }
    monkeypatch.setattr(migrate_release, "load_config", lambda: config)

    migrate_release.main()

    assert capsys.readouterr().out == "Migrated 0 soul database(s)\n"


def test_release_migration_hides_failed_soul_filename(tmp_path, monkeypatch):
    database = tmp_path / "Private Soul.db"
    monkeypatch.setattr(migrate_release, "load_config", lambda: {
        "llm": {"embedding": {}},
        "storage": {"metadata_store": {"dsn": f"sqlite:///{tmp_path / 'memu.db'}"}},
    })
    monkeypatch.setattr(migrate_release, "soul_databases", lambda _config: [database])
    monkeypatch.setattr(
        migrate_release, "migrate_database",
        lambda *_args: (_ for _ in ()).throw(RuntimeError(str(database))),
    )

    with pytest.raises(RuntimeError) as error:
        migrate_release.main()

    assert str(error.value) == "Soul database migration failed"
