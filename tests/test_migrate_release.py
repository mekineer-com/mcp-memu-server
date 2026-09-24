import sqlite3

import pytest

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
    for path in (base, soul, hidden):
        path.touch()
    (tmp_path / "Linked.db").symlink_to(soul)
    config = {
        "llm": {"embedding": {"embed_model": "gemini-embedding-2"}},
        "storage": {
            "sqlite_dir": str(tmp_path),
            "metadata_store": {"dsn": f"sqlite:///{base}"},
        },
    }

    assert migrate_release.soul_databases(config) == [soul]


def test_release_migration_accepts_no_souls_and_derives_profile(tmp_path, monkeypatch, capsys):
    config = {
        "llm": {"embedding": {"embed_model": "gemini-embedding-2"}},
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
        "llm": {"embedding": {"embed_model": "gemini-embedding-2"}},
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
