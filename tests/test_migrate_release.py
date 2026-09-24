import sqlite3

import migrate_release


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
