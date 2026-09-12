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
        "mentra": {"enabled": False, "integration_bearer_token": "test-secret"},
    }
    app = FastAPI()
    souls.register_soul_routes(app, get_config=lambda: cfg)
    owner.register_owner_routes(app, get_config=lambda: cfg)
    register_mentra_routes(app, get_config=lambda: cfg)
    if with_owner:
        owner.create_owner(cfg, "Marcos")
    return TestClient(app), cfg


def post(client, name, consent=False, path="/souls", headers=None):
    return client.post(path, json={"soul_id": name, "use_existing": consent}, headers=headers)


def test_exact_names_discovery_and_confirmation(tmp_path, monkeypatch):
    client, _ = client_for(tmp_path)
    names = ["Siri", "Henrietta Jones", "Henrietta_Jones", "Écho!"]
    for name in names:
        assert post(client, f" {name} ").json() == {"soul_id": name, "created": True}
        assert (tmp_path / f"{name}.db").exists()
        assert post(client, name).json()["detail"]["reason"] == "existing_exact"
        assert post(client, name, True).json() == {"soul_id": name, "created": False}
    siri_bytes = (tmp_path / "Siri.db").read_bytes()
    assert post(client, "Siri", True).json()["created"] is False
    assert (tmp_path / "Siri.db").read_bytes() == siri_bytes

    (tmp_path / "memu.db").write_bytes(b"base")
    assert post(client, "memu", True).status_code == 409
    (tmp_path / "Linked.db").symlink_to(tmp_path / "Siri.db")
    assert post(client, "Linked", True).status_code == 409
    (tmp_path / "Unreadable.db").write_bytes(b"not sqlite")
    monkeypatch.setattr(sqlite3, "connect", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("discovery opened a DB")))
    assert client.get("/souls").json() == {"souls": sorted(names + ["Unreadable"])}
    chat_dir, _, _ = memorize_endpoint.resolve_chat_storage_dir(tmp_path, "Marcos", "Henrietta Jones", "chat")
    assert chat_dir.name.startswith("Henrietta Jones_")
    assert config.soul_gen_config_path({}, "Marcos", "Henrietta Jones").name == "Marcos__Henrietta Jones.gen.json"


def test_soul_names_are_unique_ignoring_case(tmp_path):
    client, cfg = client_for(tmp_path)
    assert post(client, "Siri").status_code == 200

    conflict = post(client, "siri")
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "Soul name is already taken as 'Siri'"
    with pytest.raises(config.SoulNameConflictError, match="taken as 'Siri'"):
        config.sqlite_dsn_for_scope(
            cfg,
            cfg["storage"]["metadata_store"]["dsn"],
            {"user_id": "Marcos", "soul_id": "siri"},
        )
    assert not (tmp_path / "siri.db").exists()


def test_windows_sqlite_dsn_keeps_drive_path_shape():
    assert config.sqlite_dsn_from_path(PureWindowsPath("C:/Users/Test/Siri.db")) == (
        "sqlite:///C:/Users/Test/Siri.db"
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


def test_mentra_alias_is_authenticated_and_uses_same_contract(tmp_path):
    client, _ = client_for(tmp_path, with_owner=False)
    alias = "/integration/mentra/souls"
    assert client.get(alias).status_code == 401
    assert post(client, "Echo", path=alias).status_code == 401
    auth = {"Authorization": "Bearer test-secret"}
    assert post(client, "Echo", path=alias, headers=auth).status_code == 409

    assert client.get("/owner").json() == {"user_id": None}
    assert client.post("/owner", json={"user_id": " Marcos "}).json() == {
        "user_id": "Marcos",
        "created": True,
    }
    assert post(client, "Echo", path=alias, headers=auth).json()["created"] is True
    assert client.get(alias, headers=auth).json() == {"souls": ["Echo"]}
    owner_alias = "/integration/mentra/owner"
    assert client.get(owner_alias).status_code == 401
    assert client.get(owner_alias, headers=auth).json() == {"user_id": "Marcos"}
    assert client.post(owner_alias, json={"user_id": "Other"}, headers=auth).status_code == 409


def test_concurrent_creation_never_overwrites(tmp_path):
    client, _ = client_for(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: post(client, "Parallel Soul", True), range(2)))
    assert sorted(response.json()["created"] for response in responses) == [False, True]
    with sqlite3.connect(tmp_path / "Parallel Soul.db") as con:
        assert con.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert con.execute("PRAGMA user_version").fetchone() == (1,)


def test_concurrent_case_variants_publish_only_one_soul(tmp_path, monkeypatch):
    client, cfg = client_for(tmp_path)
    publish = souls.publish_soul_db

    def delayed_publish(path):
        sleep(0.05)
        return publish(path)

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
        {"user_id": "Marcos", "soul_id": winner},
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
        config.sqlite_dsn_for_scope(cfg, base, {"user_id": "Marcos", "soul_id": "First Soul"})
    assert not (tmp_path / "First Soul.db").exists()
    with pytest.raises(config.SoulIdError, match="reserved"):
        config.sqlite_dsn_for_scope(cfg, base, {"user_id": "Marcos", "soul_id": "memu"})
    with pytest.raises(HTTPException) as reserved:
        main._get_service_from_payload({"user": {"user_id": "Marcos", "soul_id": "memu"}})
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
    soul = tmp_path / "Siri.db"
    base.touch()
    soul.touch()
    assert free_turn._free_turn_followup_db_paths(
        storage_status={"dsn": f"sqlite:///{base}"},
        config={"storage": {"sqlite_dir": str(tmp_path)}},
        sqlite_dir_from_cfg=lambda *_a, **_k: tmp_path,
        logger=None,
    ) == [soul]
