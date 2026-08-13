from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

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
    assert report["active_memory_identity"]["count"] == 1
    assert report["memory_scopes"] == [{"user_id": "test-user", "soul_id": "test-soul"}]
    assert report["missing_memory_refs"] == 0
    assert report["vector_storage"]["memory_items"] == {"blob": 1}


def test_migration_memory_render_is_date_stable() -> None:
    row = {
        "id": "memory-1",
        "memory_ref": 1,
        "memory_type": "knowledge",
        "summary": "A stable memory.",
        "happened_at": "2026-08-01T10:00:00",
    }

    assert migration._render_memory(row) == "[M1] [knowledge] (2026-08-01) A stable memory."


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
    assert migration._parse_taxonomy_output(f"```xml\n{raw}\n```", ["[M1]", "[M2]"])

    with pytest.raises(migration.MigrationError, match="unsupplied"):
        migration._parse_taxonomy_output(
            raw.replace("Shared Adventures", "Invented Dossier"),
            ["[M1]", "[M2]"],
            allowed_titles={"[M1]": {"Shared Adventures"}, "[M2]": set()},
        )
    with pytest.raises(migration.MigrationError, match="differ from input"):
        migration._parse_taxonomy_output(raw, ["[M2]", "[M1]"])
    with pytest.raises(migration.MigrationError, match="zero to three"):
        migration._parse_taxonomy_output(
            raw.replace(
                "<category>Shared Adventures</category>",
                "".join(f"<category>Choice {index}</category>" for index in range(4)),
            ),
            ["[M1]", "[M2]"],
        )


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


def test_artifacts_are_source_keyed_and_prompt_limit_handles_unbroken_text(tmp_path) -> None:
    first = migration._artifact_paths(tmp_path / "a" / "soul.db", tmp_path)
    second = migration._artifact_paths(tmp_path / "b" / "soul.db", tmp_path)
    assert first != second

    oversized = {"text": "x" * (migration.MAX_PROMPT_TOKENS * 4 + 1)}
    with pytest.raises(migration.MigrationError, match="single memory exceeds"):
        migration._prompt_batches(
            [{"text": "short"}, oversized],
            system_prompt="system",
            render_user=lambda rows: "".join(row["text"] for row in rows),
        )


def test_completed_discovery_artifacts_are_reverified(tmp_path) -> None:
    database = tmp_path / "copy.db"
    dossiers = tmp_path / "dossiers.json"
    _database(database)
    migration._checkpoint_stopped_database(database)
    dossiers.write_text("{}", encoding="utf-8")
    paths = {"discovery_db": database, "dossiers": dossiers}
    discovery = {
        "discovery_database_identity": migration._database_identity(database),
        "dossier_artifact_hash": migration._sha256_file(dossiers),
    }
    migration._verify_discovery_artifacts(paths, discovery)
    dossiers.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(migration.MigrationError, match="artifact is missing or changed"):
        migration._verify_discovery_artifacts(paths, discovery)


def test_dossier_artifact_rejects_scope_mismatch_duplicates_and_unapproved_rows(tmp_path) -> None:
    scope = {"user_id": "test-user", "soul_id": "test-soul"}
    path = tmp_path / "dossiers.json"
    dossier = {
        "title": "Daily Life",
        "description": "A personal daily-life brief.",
        "kind": "topic",
        "origin": "seed",
        "approved": True,
    }
    migration._atomic_write_json(
        path,
        {"scope": {"user_id": "other-user", "soul_id": "test-soul"}, "dossiers": [dossier]},
    )
    with pytest.raises(migration.MigrationError, match="scope does not match"):
        migration._validate_dossiers(path, scope, require_approved=True)

    migration._atomic_write_json(path, {"scope": scope, "dossiers": [dossier, dossier]})
    with pytest.raises(migration.MigrationError, match="duplicate normalized"):
        migration._validate_dossiers(path, scope, require_approved=True)

    migration._atomic_write_json(
        path,
        {"scope": scope, "dossiers": [{**dossier, "approved": False}]},
    )
    with pytest.raises(migration.MigrationError, match="not approved"):
        migration._validate_dossiers(path, scope, require_approved=True)


@pytest.mark.asyncio
async def test_taxonomy_batch_cache_rejects_drift_and_malformed_output_is_not_cached() -> None:
    row = {"id": "memory-1", "memory_ref": 1}
    raw = (
        '<taxonomy_memories><memory ref="[M1]"><categories></categories>'
        "</memory></taxonomy_memories>"
    )

    class Service:
        def __init__(self, response):
            self.response = response
            self.calls = 0

        async def chat(self, *_args, **_kwargs):
            self.calls += 1
            return self.response

    service = Service(raw)
    entries = []
    decision, _entry = await migration._cached_taxonomy_batch(
        entries=entries,
        index=0,
        service=service,
        profile="category_update",
        model_identity=MODEL_IDENTITY,
        system_prompt="system",
        user_prompt="user",
        rows=[row],
    )
    assert decision == {"[M1]": []}
    assert service.calls == 1
    await migration._cached_taxonomy_batch(
        entries=entries,
        index=0,
        service=service,
        profile="category_update",
        model_identity=MODEL_IDENTITY,
        system_prompt="system",
        user_prompt="user",
        rows=[row],
    )
    assert service.calls == 1
    with pytest.raises(migration.MigrationError, match="no longer matches"):
        await migration._cached_taxonomy_batch(
            entries=entries,
            index=0,
            service=service,
            profile="category_update",
            model_identity=MODEL_IDENTITY,
            system_prompt="system changed",
            user_prompt="user",
            rows=[row],
        )

    malformed_entries = []
    with pytest.raises(migration.MigrationError, match="expected exact"):
        await migration._cached_taxonomy_batch(
            entries=malformed_entries,
            index=0,
            service=Service("not xml"),
            profile="category_update",
            model_identity=MODEL_IDENTITY,
            system_prompt="system",
            user_prompt="user",
            rows=[row],
        )
    assert malformed_entries == []


@pytest.mark.asyncio
async def test_interrupted_reset_rejects_changed_memories_even_when_taxonomy_is_empty(
    tmp_path, monkeypatch
) -> None:
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
    manifest = migration._new_manifest(database, report, scope, MODEL_IDENTITY)
    manifest["discovery"]["complete"] = True
    manifest["dossier_artifact_hash"] = migration._sha256_file(paths["dossiers"])
    manifest["apply"]["backup"] = str(backup)
    manifest["pending_operation"] = "target_reset"
    migration._atomic_write_json(paths["manifest"], manifest)

    monkeypatch.setattr(migration, "_configured_profiles", lambda _cfg: {"default": {}})
    monkeypatch.setattr(migration, "_model_identity", lambda _cfg, _profiles: MODEL_IDENTITY)
    with pytest.raises(migration.MigrationError, match="recovery backup is missing or changed"):
        await migration.apply_database(
            database,
            cfg={},
            work_dir=work_dir,
            dossier_path=paths["dossiers"],
            backup_dir=tmp_path,
        )

    backup.write_bytes(database.read_bytes())
    manifest["apply"]["backup_identity"] = migration._database_identity(backup)
    manifest["apply"]["backup_source_identity"] = manifest["source_identity"]
    migration._atomic_write_json(paths["manifest"], manifest)
    backup.write_bytes(b"corrupt")
    with pytest.raises(migration.MigrationError, match="recovery backup is missing or changed"):
        await migration.apply_database(
            database,
            cfg={},
            work_dir=work_dir,
            dossier_path=paths["dossiers"],
            backup_dir=tmp_path,
        )

    backup.write_bytes(database.read_bytes())
    manifest["apply"]["backup_identity"] = migration._database_identity(backup)
    migration._atomic_write_json(paths["manifest"], manifest)
    with closing(sqlite3.connect(database)) as conn:
        conn.execute("UPDATE memory_items SET summary = ? WHERE id = ?", ("Changed", "memory-1"))
        conn.commit()

    with pytest.raises(migration.MigrationError, match="active memory set changed"):
        await migration.apply_database(
            database,
            cfg={},
            work_dir=work_dir,
            dossier_path=paths["dossiers"],
            backup_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_dossier_seed_resume_approves_committed_row_and_validation_checks_identity(
    tmp_path, monkeypatch
) -> None:
    from memu.app.service import MemoryService

    database = tmp_path / "seed.db"
    scope = {"user_id": "test-user", "soul_id": "test-soul"}
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": f"sqlite:///{database}"}},
        user_config={"model": _Scope},
    )

    async def embed(texts, profile="embedding"):
        assert profile == "embedding"
        return [[1.0, 0.0] for _text in texts]

    service.embed = embed
    service._llm_clients["embedding"] = service
    service.database.memory_item_repo.create_item(
        memory_type="episode",
        summary="A neutral seed rehearsal memory.",
        embedding=[1.0, 0.0],
        happened_at=date(2026, 1, 1),
        user_data=scope,
    )
    service.database.memory_item_repo.backfill_memory_refs(scope)
    await migration._ensure_approved_anchors(service, scope)
    rows = [{"title": "Daily Life", "description": "A personal daily-life brief.", "kind": "topic"}]
    repo = service.database.memory_category_repo
    approve = repo.approve_category_summary

    def fail_approval(*_args, **_kwargs):
        raise OSError("stop")

    monkeypatch.setattr(repo, "approve_category_summary", fail_approval)
    with pytest.raises(OSError, match="stop"):
        await migration._create_dossiers(service, scope, rows)

    unapproved = next(row for row in repo.list_categories(scope).values() if row.anchor_role is None)
    assert unapproved.summary == "## unlabeled"
    assert unapproved.approved_summary is None
    monkeypatch.setattr(repo, "approve_category_summary", approve)
    await migration._create_dossiers(service, scope, rows)
    approved = repo.list_categories(scope)[unapproved.id]
    assert approved.approved_description == approved.description
    assert approved.approved_summary == "## unlabeled"

    artifact = tmp_path / "dossiers.json"
    migration._atomic_write_json(
        artifact,
        {
            "scope": scope,
            "dossiers": [{**rows[0], "origin": "seed", "approved": True}],
        },
    )
    assert migration.validate_database(database, dossier_path=artifact)["ok"] is True
    repo.update_category(category_id=approved.id, kind="goal", where=scope)
    result = migration.validate_database(database, dossier_path=artifact)
    assert result["ok"] is False
    assert any("identity differs" in error for error in result["errors"])
    service.database.close()


@pytest.mark.asyncio
async def test_complete_migration_replaces_legacy_taxonomy_and_completed_rerun_is_read_only(
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
    item = service.database.memory_item_repo.create_item(
        memory_type="episode",
        summary="A neutral migration rehearsal memory.",
        embedding=[1.0, 0.0],
        happened_at=date(2026, 1, 1),
        user_data=scope,
    )
    legacy = service.database.memory_category_repo.get_or_create_category(
        name="Legacy",
        description="Legacy category to replace.",
        embedding=[1.0, 0.0],
        user_data=scope,
        kind="topic",
    )
    service.database.category_item_repo.link_item_category(item.id, legacy.id, scope)
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

        async def chat(prompt, **kwargs):
            system_prompt = str(kwargs.get("system_prompt") or "")
            if "<dossier_revision dossier_id=" in system_prompt:
                dossier_id = re.search(r'<dossier_revision dossier_id="([^"]+)">', system_prompt)
                memory_ref = re.search(r"\[M\d+\]", prompt)
                assert dossier_id is not None and memory_ref is not None
                return f"""<dossier_revision dossier_id="{dossier_id.group(1)}">
  <description>A personal account of daily life.</description>
  <prose_action>patch</prose_action><prose></prose>
  <prose_patches><section ref="S1" action="replace"><body>## Daily Life
A neutral rehearsal memory {memory_ref.group(0)}.</body></section></prose_patches>
  <decisions><decision ref="{memory_ref.group(0)}" action="add" /></decisions>
</dossier_revision>"""
            refs = re.findall(r"\[M\d+\]", prompt)
            unique_refs = list(dict.fromkeys(refs))
            memories = "".join(
                f'<memory ref="{ref}"><categories><category>Daily Life</category></categories></memory>'
                for ref in unique_refs
            )
            return f"<taxonomy_memories>{memories}</taxonomy_memories>"

        built.embed = embed
        built.chat = chat
        built._llm_clients["embedding"] = built
        built._select_chat_client = lambda *_args, **_kwargs: built
        return built, "category_update", MODEL_IDENTITY

    monkeypatch.setattr(migration, "_configured_profiles", lambda _cfg: {"default": {}})
    monkeypatch.setattr(migration, "_model_identity", lambda _cfg, _profiles: MODEL_IDENTITY)
    monkeypatch.setattr(migration, "_build_service", build_service)

    seed_path = tmp_path / "seed-dossiers.json"
    migration._atomic_write_json(
        seed_path,
        {
            "scope": scope,
            "dossiers": [{
                "title": "Daily Life",
                "description": "A brief about daily life.",
                "kind": "topic",
                "origin": "seed",
                "approved": True,
            }],
        },
    )
    discovered = await migration.discover_database(
        database, cfg={}, work_dir=work_dir, seed_path=seed_path
    )
    assert discovered["status"] == "awaiting_dossier_approval"
    assert database.read_bytes() == source_before

    completed_discovery = await migration.discover_database(
        database, cfg={}, work_dir=work_dir, seed_path=seed_path
    )
    assert completed_discovery["status"] == "complete"
    seed_payload = migration._read_json(seed_path)
    seed_payload["dossiers"][0]["description"] = "Changed seed identity."
    migration._atomic_write_json(seed_path, seed_payload)
    with pytest.raises(migration.MigrationError, match="seed dossiers changed"):
        await migration.discover_database(
            database, cfg={}, work_dir=work_dir, seed_path=seed_path
        )
    seed_payload["dossiers"][0]["description"] = "A brief about daily life."
    migration._atomic_write_json(seed_path, seed_payload)

    dossier_path = Path(discovered["dossiers"])
    manifest_path = Path(discovered["manifest"])
    real_scan = migration.scan_database
    scan_count = 0

    def drift_on_final_scan(path):
        nonlocal scan_count
        report = real_scan(path)
        if Path(path).resolve() == database.resolve():
            scan_count += 1
            if scan_count == 3:
                report["active_memory_identity"] = {"count": 2, "sha256": "changed"}
        return report

    monkeypatch.setattr(migration, "scan_database", drift_on_final_scan)
    with pytest.raises(migration.MigrationError, match="active memory set changed during apply"):
        await migration.apply_database(
            database,
            cfg={},
            work_dir=work_dir,
            dossier_path=dossier_path,
            backup_dir=backup_dir,
        )
    monkeypatch.setattr(migration, "scan_database", real_scan)
    applied = await migration.apply_database(
        database, cfg={}, work_dir=work_dir, dossier_path=dossier_path, backup_dir=backup_dir
    )
    assert applied["validation"]["ok"] is True
    assert applied["validation"]["uncategorized_memories"] == 0
    assert applied["validation"]["dossiers"] == 1
    assert applied["validation"]["pending_dossier_approvals"] == ["Daily Life"]
    assert migration.scan_database(database)["anchors"] == 2
    reopened = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": f"sqlite:///{database}"}},
        user_config={"model": _Scope},
    )
    reopened.require_dossier_cutover_ready(scope)
    reopened.database.close()
    with closing(sqlite3.connect(database)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM categories WHERE id = ?", (legacy.id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM category_items WHERE category_id = ?", (legacy.id,)).fetchone()[0] == 0
    manifest = migration._read_json(manifest_path)
    assert manifest["apply"]["seeded"] is True
    assert manifest["apply"]["assignments_complete"] is True

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
