from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.mentra_routes import register_mentra_routes


def test_mentra_health_requires_enabled_configured_bearer() -> None:
    config = {"mentra": {"enabled": False, "integration_bearer_token": ""}}
    app = FastAPI()
    register_mentra_routes(app, get_config=lambda: config)
    client = TestClient(app)

    assert client.get("/integration/mentra/health").status_code == 404

    config["mentra"]["enabled"] = True
    assert client.get("/integration/mentra/health").status_code == 503

    config["mentra"]["integration_bearer_token"] = "test-secret"
    missing = client.get("/integration/mentra/health")
    wrong = client.get("/integration/mentra/health", headers={"Authorization": "Bearer wrong"})
    assert missing.status_code == wrong.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"

    accepted = client.get(
        "/integration/mentra/health",
        headers={"Authorization": "Bearer test-secret"},
    )
    assert accepted.status_code == 200
    assert accepted.json() == {"ok": True}
