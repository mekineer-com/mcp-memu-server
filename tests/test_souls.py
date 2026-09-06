import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config
from app.services import memorize_endpoint, souls
from app.services.mentra_routes import register_mentra_routes


def client_for(tmp_path):
    cfg = {
        "storage": {"metadata_store": {"dsn": f"sqlite:///{tmp_path / 'memu.db'}"}},
        "mentra": {"enabled": False, "integration_bearer_token": "test-secret"},
    }
    app = FastAPI()
    souls.register_soul_routes(app, get_config=lambda: cfg)
    register_mentra_routes(app, get_config=lambda: cfg)
    return TestClient(app), cfg


def post(client, name, consent=False, path="/souls", headers=None):
    return client.post(path, json={"soul_id": name, "use_existing": consent}, headers=headers)


def test_exact_names_discovery_and_confirmation(tmp_path, monkeypatch):
    client, _ = client_for(tmp_path)
    names = ["Siri", "siri", "Henrietta Jones", "Henrietta_Jones", "Écho!"]
    for name in names:
        assert post(client, f" {name} ").json() == {"soul_id": name, "created": True}
        assert (tmp_path / f"{name}.db").exists()
        assert post(client, name).json()["detail"]["reason"] == "existing_exact"
        assert post(client, name, True).json() == {"soul_id": name, "created": False}

    (tmp_path / "memu.db").write_bytes(b"base")
    (tmp_path / "Linked.db").symlink_to(tmp_path / "Siri.db")
    (tmp_path / "Unreadable.db").write_bytes(b"not sqlite")
    monkeypatch.setattr(sqlite3, "connect", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("discovery opened a DB")))
    assert client.get("/souls").json() == {"souls": sorted(names + ["Unreadable"])}
    chat_dir, _, _ = memorize_endpoint.resolve_chat_storage_dir(tmp_path, "Marcos", "Henrietta Jones", "chat")
    assert chat_dir.name.startswith("Henrietta Jones_")
    assert config.soul_gen_config_path({}, "Marcos", "Henrietta Jones").name == "Marcos__Henrietta Jones.gen.json"


@pytest.mark.parametrize("name", ["", " ", ".", "..", "bad/name", "bad\\name", "bad*name", "bad?name", "bad[name", "bad\x00name", "é" * 41])
def test_invalid_names_fail_without_creating_a_database(tmp_path, name):
    client, _ = client_for(tmp_path)
    assert post(client, name).status_code == 422
    assert not list(tmp_path.glob("*.db"))


def test_mentra_alias_is_authenticated_and_uses_same_contract(tmp_path):
    client, _ = client_for(tmp_path)
    alias = "/integration/mentra/souls"
    assert client.get(alias).status_code == 401
    assert post(client, "Echo", path=alias).status_code == 401
    auth = {"Authorization": "Bearer test-secret"}
    assert post(client, "Echo", path=alias, headers=auth).json()["created"] is True
    assert client.get(alias, headers=auth).json() == {"souls": ["Echo"]}


def test_concurrent_creation_never_overwrites(tmp_path):
    client, _ = client_for(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: post(client, "Parallel Soul", True), range(2)))
    assert sorted(response.json()["created"] for response in responses) == [False, True]
    with sqlite3.connect(tmp_path / "Parallel Soul.db") as con:
        assert con.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_first_scoped_use_creates_exact_database_without_picker_policy(tmp_path, caplog):
    _, cfg = client_for(tmp_path)
    base = cfg["storage"]["metadata_store"]["dsn"]
    dsn = config.sqlite_dsn_for_scope(cfg, base, {"user_id": "Marcos", "soul_id": "First Soul"})
    assert config.sqlite_file_from_dsn(dsn) == tmp_path / "First Soul.db"
    assert "Created soul 'First Soul'" in caplog.text


def test_discovery_reports_unreadable_directory(tmp_path, monkeypatch):
    client, _ = client_for(tmp_path)
    monkeypatch.setattr(type(tmp_path), "iterdir", lambda *_: (_ for _ in ()).throw(PermissionError("no access")))
    assert client.get("/souls").status_code == 503
