import shutil
import socket
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config, main
from app.services import service_factory, souls
from app.services.mentra_routes import register_mentra_routes


USER = "Fictional User"


def client_for(tmp_path):
    cfg = {"storage": {"metadata_store": {"dsn": f"sqlite:///{tmp_path / 'base.db'}"}},
           "mentra": {"enabled": False, "integration_bearer_token": "test-secret"}}
    app = FastAPI()
    souls.register_soul_routes(app, get_config=lambda: cfg)
    register_mentra_routes(app, get_config=lambda: cfg)
    return TestClient(app), cfg


def post(client, name, consent=False, **kwargs):
    return client.post("/souls", json={"user_id": USER, "soul_id": name, "use_existing": consent}, **kwargs)


def test_contract_auth_readonly_and_canonical_path(tmp_path, monkeypatch):
    client, cfg = client_for(tmp_path)
    assert post(client, "Fictional Soul").json() == {"soul_id": "Fictional Soul", "created": True}
    target = tmp_path / "Fictional_Soul.db"
    assert config.sqlite_file_from_dsn(config.sqlite_dsn_for_scope(cfg, cfg["storage"]["metadata_store"]["dsn"], {"soul_id": "Fictional Soul"})) == target
    before = target.read_bytes()
    assert post(client, "Fictional Soul").json()["detail"]["reason"] == "existing_exact"
    assert post(client, "Fictional Soul", True).json()["created"] is False
    original = souls._readonly_connect
    def read_only(path):
        con = original(path)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            con.execute("CREATE TABLE forbidden (id INTEGER)")
        return con
    monkeypatch.setattr(souls, "_readonly_connect", read_only)
    alias = "/integration/mentra/souls"
    for method in ("get", "post"):
        assert getattr(client, method)(alias).status_code == 401
        assert getattr(client, method)(alias, headers={"Authorization": "Bearer wrong"}).status_code == 401
    auth = {"Authorization": "Bearer test-secret"}
    assert client.get(alias, params={"user_id": USER}, headers=auth).json() == {"souls": ["Fictional Soul"]}
    assert client.post(alias, headers=auth, json={"user_id": USER, "soul_id": "Fictional Soul", "use_existing": True}).json()["created"] is False
    assert client.get("/souls", params={"user_id": "Other User"}).json() == {"souls": []}
    assert target.read_bytes() == before


@pytest.mark.parametrize("name", ["..", " ", "!!!", "fictional USER"])
def test_invalid_scope(tmp_path, name):
    client, _ = client_for(tmp_path)
    assert post(client, name).status_code == 422
    assert not list(tmp_path.glob("*.db"))


def test_collisions_artifacts_and_legacy_metadata(tmp_path):
    client, _ = client_for(tmp_path)
    for table, name in (("categories", "Legacy One"), ("conversations", "Legacy Two")):
        path = tmp_path / (name.replace(" ", "_") + ".db")
        with sqlite3.connect(path) as con:
            con.execute(f"CREATE TABLE {table} (user_id TEXT, soul_id TEXT)")
            query = "INSERT INTO categories VALUES (?, ?)" if table == "categories" else "INSERT INTO conversations VALUES (?, ?)"
            con.execute(query, (USER, name))
        shutil.copyfile(path, tmp_path / (name + "-backup.db"))
        assert post(client, name, True).json()["created"] is False
    with sqlite3.connect(tmp_path / "artifact.db") as con:
        con.execute("CREATE TABLE artifact (id INTEGER)")
    assert client.get("/souls", params={"user_id": USER}).json() == {"souls": ["Legacy One", "Legacy Two"]}
    assert post(client, "Legacy_One", True).json()["detail"]["reason"] == "sanitized_collision"
    assert post(client, "A" * 80 + "x").status_code == 200
    assert post(client, "A" * 80 + "y", True).status_code == 409
    unknown = tmp_path / "UnknownTarget.db"
    unknown.write_bytes(b"corrupt database")
    assert client.get("/souls", params={"user_id": USER}).status_code == 503
    assert post(client, "UnknownTarget", True).json()["detail"]["reason"] == "sanitized_collision"
    assert unknown.read_bytes() == b"corrupt database"


def test_concurrent_collision_cannot_overwrite(tmp_path):
    client, _ = client_for(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda name: post(client, name, True), ["Parallel Soul", "Parallel_Soul"]))
    assert sorted(response.status_code for response in responses) == [200, 409]
    assert len(client.get("/souls", params={"user_id": USER}).json()["souls"]) == 1


def test_failed_creation_publishes_nothing_and_can_retry(tmp_path, monkeypatch):
    client, _ = client_for(tmp_path)
    client = TestClient(client.app, raise_server_exceptions=False)
    before = set(tmp_path.iterdir())
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise sqlite3.OperationalError("fictional initialization failure")
        patch.setattr(sqlite3, "connect", fail)
        assert post(client, "Retry Soul").status_code == 500
    assert set(tmp_path.iterdir()) == before
    assert post(client, "Retry Soul").json()["created"] is True


def test_identity_only_database_initializes_real_scoped_service(tmp_path, monkeypatch):
    client, cfg = client_for(tmp_path)
    cfg["llm"] = {"provider": "openai", "api_key": "fictional", "chat_model": "fictional-chat", "embed_model": "text-embedding-3-large"}
    cfg["storage"]["resources_dir"] = str(tmp_path / "resources")
    def forbidden(*args, **kwargs):
        raise AssertionError("Creation and scoped initialization must not call providers")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(service_factory.MemoryService, "_init_llm_client", forbidden)
    monkeypatch.setattr(main, "_CONFIG", cfg)
    assert post(client, "Fresh Fictional Soul").status_code == 200
    path = tmp_path / "Fresh_Fictional_Soul.db"
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [("soul_identity",)]
    service_factory._clear_cached_services()
    try:
        svc = main._get_service_from_payload({"user": {"user_id": USER, "soul_id": "Fresh Fictional Soul"}})
        with sqlite3.connect(path) as con:
            assert con.execute("SELECT COUNT(*) FROM categories").fetchone() == (0,)
            assert con.execute("SELECT profile FROM embedding_profile WHERE id = 1").fetchone() is None
        assert svc.database_config.metadata_store.embedding_profile == "text-embedding-3-large:3072"
        assert svc._llm_clients == {}
    finally:
        service_factory._clear_cached_services()
    assert client.get("/souls", params={"user_id": USER}).json() == {"souls": ["Fresh Fictional Soul"]}


def test_unknown_empty_and_symlink_targets_are_never_reused(tmp_path):
    client, _ = client_for(tmp_path)
    target = tmp_path / "Empty.db"
    target.touch()
    assert post(client, "Empty", True).status_code == 409
    assert target.read_bytes() == b""
    (tmp_path / "Linked.db").symlink_to(tmp_path / "Missing.db")
    assert post(client, "Linked", True).status_code == 409
    assert not (tmp_path / "Missing.db").exists()


def test_discovery_reports_unreadable_directory(tmp_path, monkeypatch):
    client, _ = client_for(tmp_path)
    def unreadable(*args):
        raise PermissionError("fictional inaccessible directory")
    monkeypatch.setattr(type(tmp_path), "iterdir", unreadable)
    assert client.get("/souls", params={"user_id": USER}).status_code == 503
