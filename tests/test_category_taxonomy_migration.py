from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from datetime import date

import pytest
from pydantic import BaseModel

import migrate_category_taxonomy as migration


MODEL_IDENTITY = {"profile": "category_update", "provider": "test", "model": "fixed"}


class _Scope(BaseModel):
    user_id: str | None = None
    soul_id: str | None = None


def _database(path) -> None:
    scope = "user_id TEXT, soul_id TEXT"
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(
            f"""
            CREATE TABLE memory_items (
                id TEXT PRIMARY KEY, memory_ref INTEGER, memory_type TEXT, summary TEXT,
                happened_at DATETIME, created_at DATETIME, merged_into TEXT, embedding BLOB,
                {scope}
            );
            CREATE TABLE categories (
                id TEXT PRIMARY KEY, name TEXT, description TEXT, summary TEXT,
                embedding BLOB,
                approved_description TEXT, approved_summary TEXT, kind TEXT, anchor_role TEXT,
                last_evidence_at DATETIME, last_revised_at DATETIME, {scope}
            );
            CREATE TABLE category_items (id TEXT PRIMARY KEY, item_id TEXT, category_id TEXT, {scope});
            CREATE TABLE dossier_candidates (id TEXT PRIMARY KEY, {scope});
            CREATE TABLE triples (
                id TEXT PRIMARY KEY, subject_id TEXT, subject_kind TEXT, predicate TEXT,
                valid_to DATETIME, {scope}
            );
            CREATE TABLE resources (id TEXT PRIMARY KEY, embedding BLOB, {scope});
            CREATE TABLE entities (id TEXT PRIMARY KEY, {scope});
            """
        )
        conn.execute(
            "INSERT INTO memory_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "memory-1",
                1,
                "episode",
                "A quiet test memory.",
                "2026-01-01",
                "2026-01-01T01:00:00",
                None,
                b"\0\0\0\0",
                "test-user",
                "test-soul",
            ),
        )
        conn.commit()


def test_inventory_is_read_only_and_reports_scope(tmp_path) -> None:
    path = tmp_path / "soul.db"
    _database(path)
    before = path.read_bytes()

    report = migration.scan_database(path)

    assert path.read_bytes() == before
    assert report["active_memories"] == 1
    assert report["memory_scopes"] == [{"user_id": "test-user", "soul_id": "test-soul"}]
    assert report["missing_memory_refs"] == 0
    assert report["vector_storage"]["memory_items"] == {"blob": 1}


def test_taxonomy_parser_requires_exact_complete_refs_and_supplied_assignment_titles() -> None:
    raw = """<taxonomy_memories>
  <memory ref="[M1]"><categories><category>Shared Adventures</category></categories></memory>
  <memory ref="[M2]"><categories></categories></memory>
</taxonomy_memories>"""

    assert migration._parse_taxonomy_output(
        raw,
        ["[M1]", "[M2]"],
        allowed_titles={"[M1]": {"Shared Adventures"}, "[M2]": set()},
    ) == {"[M1]": ["Shared Adventures"], "[M2]": []}

    with pytest.raises(migration.MigrationError, match="unsupplied"):
        migration._parse_taxonomy_output(
            raw.replace("Shared Adventures", "Invented Dossier"),
            ["[M1]", "[M2]"],
            allowed_titles={"[M1]": {"Shared Adventures"}, "[M2]": set()},
        )
    with pytest.raises(migration.MigrationError, match="differ from input"):
        migration._parse_taxonomy_output(raw, ["[M2]", "[M1]"])


def test_database_argument_defaults_to_inventory_phase() -> None:
    assert migration._normalized_argv(["soul.db"]) == ["inventory", "soul.db"]
    assert migration._normalized_argv(["--config", "cfg.json", "soul.db"]) == [
        "--config",
        "cfg.json",
        "inventory",
        "soul.db",
    ]


def test_dynamic_cluster_cache_identity_is_order_independent() -> None:
    assert migration._migration_cluster_id(["candidate-b", "candidate-a"]) == (
        migration._migration_cluster_id(["candidate-a", "candidate-b"])
    )


def test_resume_identity_includes_embedding_model() -> None:
    first = migration._model_identity(
        {},
        {
            "default": {"chat_model": "chat"},
            "embedding": {"provider": "test", "embed_model": "embed-a"},
        },
    )
    second = migration._model_identity(
        {},
        {
            "default": {"chat_model": "chat"},
            "embedding": {"provider": "test", "embed_model": "embed-b"},
        },
    )
    assert first != second


@pytest.mark.asyncio
async def test_interrupted_reset_rejects_source_drift_before_clearing(tmp_path, monkeypatch) -> None:
    database = tmp_path / "test-soul.db"
    work_dir = tmp_path / "work"
    backup = tmp_path / "backup.db"
    _database(database)
    migration._checkpoint_stopped_database(database)
    report = migration.scan_database(database)
    scope = migration._require_scope(report)
    paths = migration._artifact_paths(database, work_dir)
    dossier_payload = {"database": str(database), "scope": scope, "dossiers": []}
    migration._atomic_write_json(paths["dossiers"], dossier_payload)
    backup.write_bytes(database.read_bytes())
    manifest = migration._new_manifest(database, report, scope, MODEL_IDENTITY)
    manifest["discovery"]["complete"] = True
    manifest["dossier_artifact_hash"] = migration._sha256_file(paths["dossiers"])
    manifest["apply"]["backup"] = str(backup)
    manifest["pending_operation"] = "target_reset"
    migration._atomic_write_json(paths["manifest"], manifest)

    with closing(sqlite3.connect(database)) as conn:
        conn.execute(
            "INSERT INTO categories "
            "(id, name, description, kind, user_id, soul_id) VALUES (?, ?, ?, ?, ?, ?)",
            ("changed", "Changed", "Changed after backup", "topic", "test-user", "test-soul"),
        )
        conn.commit()

    monkeypatch.setattr(migration, "_configured_profiles", lambda _cfg: {"default": {}})
    monkeypatch.setattr(migration, "_model_identity", lambda _cfg, _profiles: MODEL_IDENTITY)
    with pytest.raises(migration.MigrationError, match="source database changed"):
        await migration.apply_database(
            database,
            cfg={},
            work_dir=work_dir,
            dossier_path=paths["dossiers"],
            backup_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_complete_migration_allows_uncategorized_memory_and_completed_rerun_is_read_only(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from memu.app.service import MemoryService

    database = tmp_path / "test-soul.db"
    work_dir = tmp_path / "work"
    backup_dir = tmp_path / "backups"
    scope = {"user_id": "test-user", "soul_id": "test-soul"}
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": f"sqlite:///{database}"}},
        user_config={"model": _Scope},
    )
    service.database.memory_item_repo.create_item(
        memory_type="episode",
        summary="A neutral migration rehearsal memory.",
        embedding=[1.0, 0.0],
        happened_at=date(2026, 1, 1),
        user_data=scope,
    )
    service.database.close()
    migration._checkpoint_stopped_database(database)
    source_before = database.read_bytes()

    builds = 0

    def build_service(_cfg, path):
        nonlocal builds
        builds += 1
        built = MemoryService(
            database_config={
                "metadata_store": {"provider": "sqlite", "dsn": f"sqlite:///{path}"}
            },
            memorize_config={"dynamic_category_cluster_size": 10},
            user_config={"model": _Scope},
        )

        async def embed(texts, profile="embedding"):
            assert profile == "embedding"
            return [[1.0, 0.0] for _text in texts]

        async def chat(prompt, **_kwargs):
            refs = re.findall(r"\[M\d+\]", prompt)
            unique_refs = list(dict.fromkeys(refs))
            memories = "".join(
                f'<memory ref="{ref}"><categories></categories></memory>'
                for ref in unique_refs
            )
            return f"<taxonomy_memories>{memories}</taxonomy_memories>"

        built.embed = embed
        built.chat = chat
        built._llm_clients["embedding"] = built
        return built, "category_update", MODEL_IDENTITY

    monkeypatch.setattr(migration, "_configured_profiles", lambda _cfg: {"default": {}})
    monkeypatch.setattr(migration, "_model_identity", lambda _cfg, _profiles: MODEL_IDENTITY)
    monkeypatch.setattr(migration, "_build_service", build_service)

    discovered = await migration.discover_database(
        database, cfg={}, work_dir=work_dir, seed_path=None
    )
    assert discovered["status"] == "awaiting_dossier_approval"
    assert database.read_bytes() == source_before

    dossier_path = work_dir / "test-soul.dossiers.json"
    applied = await migration.apply_database(
        database,
        cfg={},
        work_dir=work_dir,
        dossier_path=dossier_path,
        backup_dir=backup_dir,
    )
    assert applied["validation"]["ok"] is True
    assert applied["validation"]["uncategorized_memories"] == 1
    assert migration.scan_database(database)["anchors"] == 2

    build_count = builds
    rerun = await migration.apply_database(
        database,
        cfg={},
        work_dir=work_dir,
        dossier_path=dossier_path,
        backup_dir=backup_dir,
    )
    assert rerun["ok"] is True
    assert builds == build_count
