import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import PureWindowsPath
from time import sleep

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app import config, main
from app.services import free_turn, memorize_endpoint, owner, payload, souls
from app.services.mentra_routes import register_mentra_routes


def client_for(tmp_path, *, with_owner=True):
    cfg = {
        "storage": {"metadata_store": {"dsn": f"sqlite:///{tmp_path / 'memu.db'}"}},
        "mentra": {"enabled": False},
    }
    app = FastAPI()
    souls.register_soul_routes(app, get_config=lambda: cfg)
    owner.register_owner_routes(app, get_config=lambda: cfg)
    register_mentra_routes(app, get_config=lambda: cfg, get_activity_pause=lambda *_args: None)
    if with_owner:
        owner.create_owner(cfg, "TestOwner")
    return TestClient(app), cfg


def post(client, name, consent=False, path="/souls", headers=None):
    return client.post(path, json={"soul_id": name, "use_existing": consent}, headers=headers)


def test_exact_names_discovery_and_confirmation(tmp_path, monkeypatch):
    client, cfg = client_for(tmp_path)
    names = ["TestSoul", "Henrietta Jones", "Henrietta_Jones", "Écho!"]
    for name in names:
        assert post(client, f" {name} ").json() == {"soul_id": name, "created": True}
        assert (tmp_path / f"{name}.db").exists()
        assert post(client, name).json()["detail"]["reason"] == "existing_exact"
        assert post(client, name, True).json() == {"soul_id": name, "created": False}
    soul_bytes = (tmp_path / "TestSoul.db").read_bytes()
    assert post(client, "TestSoul", True).json()["created"] is False
    assert (tmp_path / "TestSoul.db").read_bytes() == soul_bytes

    (tmp_path / "memu.db").write_bytes(b"base")
    assert post(client, "memu", True).status_code == 409
    (tmp_path / "Linked.db").symlink_to(tmp_path / "TestSoul.db")
    assert post(client, "Linked", True).status_code == 409
    (tmp_path / "Unreadable.db").write_bytes(b"not sqlite")
    cfg["procedural"] = {"db_path": str(tmp_path / "Reference.db")}
    (tmp_path / "Reference.db").write_bytes(b"procedural")
    assert post(client, "Reference", True).status_code == 409
    with pytest.raises(config.SoulIdError, match="procedural database"):
        config.sqlite_dsn_for_scope(cfg, cfg["storage"]["metadata_store"]["dsn"],
                                   {"user_id": "TestOwner", "soul_id": "Reference"})
    monkeypatch.setattr(sqlite3, "connect", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("discovery opened a DB")))
    assert client.get("/souls").json() == {"souls": sorted(names + ["Unreadable"])}
    chat_dir, _, _ = memorize_endpoint.resolve_chat_storage_dir(tmp_path, "TestOwner", "Henrietta Jones", "chat")
    assert chat_dir.name.startswith("Henrietta Jones_")
    assert config.soul_gen_config_path({}, "TestOwner", "Henrietta Jones").name == "TestOwner__Henrietta Jones.gen.json"


def test_soul_names_are_unique_ignoring_case(tmp_path):
    client, cfg = client_for(tmp_path)
    assert post(client, "TestSoul").status_code == 200

    conflict = post(client, "testSoul")
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "Soul name is already taken as 'TestSoul'"
    with pytest.raises(config.SoulNameConflictError, match="taken as 'TestSoul'"):
        config.sqlite_dsn_for_scope(
            cfg,
            cfg["storage"]["metadata_store"]["dsn"],
            {"user_id": "TestOwner", "soul_id": "testSoul"},
        )
    assert not (tmp_path / "testSoul.db").exists()


@pytest.mark.parametrize("name", ["TestOwner", "testowner"])
def test_soul_name_must_differ_from_owner(tmp_path, name):
    client, _ = client_for(tmp_path)
    response = post(client, name)
    assert response.status_code == 422
    assert response.json()["detail"] == "Soul name must differ from the owner name"
    assert not list(tmp_path.glob("*.db"))


def test_windows_sqlite_dsn_keeps_drive_path_shape():
    assert config.sqlite_dsn_from_path(PureWindowsPath("C:/Users/Test/TestSoul.db")) == (
        "sqlite:///C:/Users/Test/TestSoul.db"
    )


@pytest.mark.parametrize("name", [
    "", " ", ".", "..", "bad/name", "bad\\name", "bad*name", "bad?name", "bad[name",
    'bad:name', 'bad"name', "bad<name", "bad>name", "bad|name", "bad.", "CON", "com1.txt",
    "bad\x00name", "é" * 41,
])
def test_invalid_names_fail_without_creating_a_database(tmp_path, name):
    client, _ = client_for(tmp_path)
    assert post(client, name).status_code == 422
    assert not list(tmp_path.glob("*.db"))


def test_mentra_alias_requires_enabled_and_uses_same_contract(tmp_path):
    client, cfg = client_for(tmp_path, with_owner=False)
    alias = "/integration/mentra/souls"
    owner_alias = "/integration/mentra/owner"
    assert client.get(alias).status_code == 404
    assert post(client, "Echo", path=alias).status_code == 404
    assert client.get(owner_alias).status_code == 404
    cfg["mentra"]["enabled"] = True
    assert post(client, "Echo", path=alias).status_code == 409

    assert client.get("/owner").json() == {"user_id": None}
    assert client.post("/owner", json={"user_id": " Fictional User "}).json() == {
        "user_id": "Fictional User",
        "created": True,
    }
    assert post(client, "Echo", path=alias).json()["created"] is True
    assert client.get(alias).json() == {"souls": ["Echo"]}
    assert client.get(owner_alias).json() == {"user_id": "Fictional User"}
    assert client.post(owner_alias, json={"user_id": "Other"}).status_code == 405
    cfg["mentra"]["enabled"] = False
    assert client.get(alias).status_code == 404
    assert post(client, "Another Soul", path=alias).status_code == 404
    assert client.get(owner_alias).status_code == 404
    assert not (tmp_path / "Another Soul.db").exists()


def test_concurrent_creation_never_overwrites(tmp_path):
    client, _ = client_for(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: post(client, "Parallel Soul", True), range(2)))
    assert sorted(response.json()["created"] for response in responses) == [False, True]
    with sqlite3.connect(tmp_path / "Parallel Soul.db") as con:
        assert con.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert con.execute("PRAGMA user_version").fetchone() == (1,)


def test_publication_uses_reopenable_temporary_file_and_cleans_it(tmp_path, monkeypatch):
    named_temporary_file = souls.tempfile.NamedTemporaryFile
    options = {}

    def tracked_temporary_file(*args, **kwargs):
        options.update(kwargs)
        return named_temporary_file(*args, **kwargs)

    monkeypatch.setattr(souls.tempfile, "NamedTemporaryFile", tracked_temporary_file)
    path = tmp_path / "Portable Soul.db"

    assert souls.publish_soul_db(path, "TestOwner") is True
    assert options["delete_on_close"] is False
    assert path.is_file()
    assert list(tmp_path.glob("*.tmp")) == []
    assert souls.read_soul_name(path) == "TestOwner"


def test_display_names_are_per_soul_and_immutable(tmp_path):
    client, _cfg = client_for(tmp_path)
    for soul_id, user_name in (("FirstSoul", "FirstHuman"), ("SecondSoul", "SecondHuman")):
        assert client.post("/souls", json={
            "soul_id": soul_id, "user_name": user_name, "use_existing": False,
        }).status_code == 200
        assert client.get(f"/souls/{soul_id}").json() == {"soul_id": soul_id, "user_name": user_name}
        assert client.post("/souls", json={
            "soul_id": soul_id, "user_name": "ReplacementHuman", "use_existing": True,
        }).status_code == 200
        assert client.get(f"/souls/{soul_id}").json()["user_name"] == user_name
    for name in ("SameName", "samename"):
        response = client.post("/souls", json={"soul_id": "SameName", "user_name": name, "use_existing": False})
        assert response.status_code == 422
    assert not (tmp_path / "SameName.db").exists()
    assert post(client, "CanonicalSoul").status_code == 200
    assert client.get("/souls/CanonicalSoul").json()["user_name"] == "TestOwner"


@pytest.mark.parametrize("state", ["missing", "no_table", "no_column", "no_row", "blank"])
def test_metadata_refuses_missing_binding_without_writing_or_migrating(tmp_path, state):
    client, _cfg = client_for(tmp_path)
    path = tmp_path / "Unbound.db"
    if state != "missing":
        with sqlite3.connect(path) as con:
            if state != "no_table":
                con.execute("CREATE TABLE soul_state (id INTEGER PRIMARY KEY" +
                            (", user_name TEXT" if state != "no_column" else "") + ")")
                if state == "blank":
                    con.execute("INSERT INTO soul_state VALUES (1, '')")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    assert client.get("/souls/Unbound").status_code == 409
    if state != "missing":
        assert client.post("/souls", json={
            "soul_id": "Unbound", "user_name": "DoNotSeed", "use_existing": True,
        }).status_code == 409
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before


@pytest.mark.parametrize("bound", [False, True])
def test_metadata_stopped_wal_read_has_no_side_effects(tmp_path, bound):
    client, _cfg = client_for(tmp_path)
    path = tmp_path / "WalSoul.db"
    if bound:
        souls.publish_soul_db(path, "TestHuman")
    con = sqlite3.connect(path)
    if not bound:
        con.execute("CREATE TABLE soul_state (id INTEGER PRIMARY KEY)")
        con.commit()
    con.execute("PRAGMA journal_mode=WAL")
    con.close()
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    assert not any(name.endswith(("-wal", "-shm")) for name in before)
    response = client.get("/souls/WalSoul")
    assert response.status_code == (200 if bound else 409)
    if bound:
        assert response.json()["user_name"] == "TestHuman"
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before


def test_non_file_soul_occupant_is_rejected(tmp_path):
    client, cfg = client_for(tmp_path)
    occupant = tmp_path / "Occupied Soul.db"
    occupant.mkdir()

    assert post(client, "Occupied Soul").status_code == 409
    assert post(client, "Occupied Soul", True).status_code == 409
    with pytest.raises(config.SoulIdError, match="must be a file"):
        config.sqlite_dsn_for_scope(
            cfg,
            cfg["storage"]["metadata_store"]["dsn"],
            {"user_id": "TestOwner", "soul_id": "Occupied Soul"},
        )
    assert occupant.is_dir()


def test_concurrent_case_variants_publish_only_one_soul(tmp_path, monkeypatch):
    client, cfg = client_for(tmp_path)
    publish = souls.publish_soul_db

    def delayed_publish(path, user_name):
        sleep(0.05)
        return publish(path, user_name)

    monkeypatch.setattr(souls, "publish_soul_db", delayed_publish)
    names = ["AuditSoul", "auditsoul"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda name: post(client, name, True), names))

    assert sorted(response.status_code for response in responses) == [200, 409]
    winner = names[next(i for i, response in enumerate(responses) if response.status_code == 200)]
    assert [path.name for path in tmp_path.glob("*.db")] == [f"{winner}.db"]
    dsn = config.sqlite_dsn_for_scope(
        cfg,
        cfg["storage"]["metadata_store"]["dsn"],
        {"user_id": "TestOwner", "soul_id": winner},
    )
    assert config.sqlite_file_from_dsn(dsn) == tmp_path / f"{winner}.db"


def test_failed_creation_publishes_nothing_and_can_retry(tmp_path, monkeypatch):
    client, cfg = client_for(tmp_path)
    before = set(tmp_path.iterdir())
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", lambda *_a, **_k: (_ for _ in ()).throw(sqlite3.OperationalError("failed")))
        with pytest.raises(sqlite3.OperationalError):
            souls.create_soul(cfg, souls.SoulCreate(soul_id="Retry Soul", use_existing=False))
    assert set(tmp_path.iterdir()) == before
    assert post(client, "Retry Soul").json()["created"] is True


def test_unknown_scoped_soul_is_rejected_without_publication(tmp_path):
    _, cfg = client_for(tmp_path)
    base = cfg["storage"]["metadata_store"]["dsn"]
    with pytest.raises(config.SoulIdError, match="does not exist"):
        config.sqlite_dsn_for_scope(cfg, base, {"user_id": "TestOwner", "soul_id": "First Soul"})
    assert not (tmp_path / "First Soul.db").exists()
    with pytest.raises(config.SoulIdError, match="reserved"):
        config.sqlite_dsn_for_scope(cfg, base, {"user_id": "TestOwner", "soul_id": "memu"})
    with pytest.raises(HTTPException) as reserved:
        main._get_service_from_payload({"user": {"user_id": "TestOwner", "soul_id": "memu"}})
    assert reserved.value.status_code == 422


def test_discovery_reports_unreadable_directory(tmp_path, monkeypatch):
    client, _ = client_for(tmp_path)
    monkeypatch.setattr(type(tmp_path), "iterdir", lambda *_: (_ for _ in ()).throw(PermissionError("no access")))
    assert client.get("/souls").status_code == 503


def test_invalid_scope_is_a_client_error_and_free_turn_skips_base_db(tmp_path):
    with pytest.raises(HTTPException) as invalid:
        payload._extract_scope({"soul_id": "bad/name"})
    assert invalid.value.status_code == 422

    base = tmp_path / "memu.db"
    soul = tmp_path / "TestSoul.db"
    base.touch()
    soul.touch()
    assert free_turn._free_turn_followup_db_paths(
        storage_status={"dsn": f"sqlite:///{base}"},
        config={"storage": {"sqlite_dir": str(tmp_path)}},
        sqlite_dir_from_cfg=lambda *_a, **_k: tmp_path,
        logger=None,
    ) == [soul]
