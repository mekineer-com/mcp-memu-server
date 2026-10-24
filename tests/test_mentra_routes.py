from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import CancelledError as FutureCancelledError, ThreadPoolExecutor
from datetime import UTC, datetime
from types import SimpleNamespace
from threading import Event
from typing import Any, Callable

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.services import conversation_sources, mentra_routes, owner
from app.services.mentra_routes import register_mentra_routes


START = {
    "user_id": "Fictional User",
    "soul_id": "Codexia",
    "device_session_id": "phone-1",
    "mode": "continuous",
    "photo_work_protocol": "iris-photo-work-v2",
}


def _photo_caption_event(sitting: str, sequence: int, image_id: str, caption: str) -> dict[str, Any]:
    media_dir = mentra_routes._media_directory(
        Path(), user_id=START["user_id"], soul_id=START["soul_id"],
    )
    return {
        "event_id": f"{sitting}:{sequence}", "sequence": sequence,
        "event_kind": "photo_caption", "role": "assistant", "content": caption,
        "media_ref": str(media_dir / f"{image_id}.png"),
    }


@pytest.fixture(autouse=True)
def clear_leases() -> None:
    mentra_routes._leases.clear()
    mentra_routes._start_claims.clear()
    mentra_routes._image_finalize_tasks.clear()
    mentra_routes._image_finalize_errors.clear()
    mentra_routes._transcript_conflicts.clear()
    yield
    mentra_routes._leases.clear()
    mentra_routes._start_claims.clear()
    mentra_routes._image_finalize_tasks.clear()
    mentra_routes._image_finalize_errors.clear()
    mentra_routes._transcript_conflicts.clear()


def _configured() -> dict[str, Any]:
    return {
        "llm": {
            "embedding": {
                "api_key": "fictional-embedding-key",
            }
        },
        "storage": {
            "metadata_store": {"embedding_profile": "gemini-embedding-2:3072"}
        },
        "mentra": {
            "gemini_api_key": "permanent-secret",
            "model": "gemini-2.5-flash-native-audio-preview-12-2025",
            "voice": "Kore",
        }
    }


def _session_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    token_results: list[str | BaseException] | None = None,
    state_results: list[Exception | None] | None = None,
    raise_server_exceptions: bool = True,
    prepare_auto_memorize: Callable[..., tuple[int, dict[str, Any] | None]] | None = None,
    schedule_auto_memorize: Callable[..., str] | None = None,
    retrieve_result: dict[str, Any] | BaseException | None = None,
    retrieve_hook: Callable[[str, dict[str, Any]], Any] | None = None,
    memorize_error: Exception | None = None,
    memorize_hook: Callable[[dict[str, Any]], Any] | None = None,
    read_config: Callable[[], dict[str, Any]] | None = None,
) -> tuple[TestClient, dict[str, Any], dict[str, Any]]:
    from app import main

    for name in ("_FORCED_MEMORIZE_INFLIGHT", "_FORCED_MEMORIZE_RECHECK", "_MEMORIZE_PROGRESS", "_CONSOLIDATION_RUNNING"):
        monkeypatch.setattr(main, name, {})
    monkeypatch.setattr(main, "_paid_work_state", lambda *_args: {})
    config = _configured()
    config["storage"]["metadata_store"]["dsn"] = f"sqlite:///{tmp_path / 'memu.db'}"
    owner.create_owner(config, START["user_id"])
    calls: dict[str, Any] = {
        "service": 0,
        "token": [],
        "state": [],
        "cross": [],
        "state_writes": [],
        "memorize_checks": [],
        "recalls": [],
        "route_telemetry": [],
        "prompts": [],
        "image_memorize": [],
        "background_errors": [],
        "working_state": {"memory_cache": [], "intentions_active": []},
        "narrative": "I value continuity.",
        "anchor_rows": {
            "soul": SimpleNamespace(summary="I am Codexia.", description="soul"),
            "user": SimpleNamespace(summary="The user likes careful work.", description="user"),
        },
        "category_index": "- Fictional Places: A remembered coast.",
        "embeddings": [],
        "deleted_memories": [],
    }
    results = iter(token_results or ["ephemeral-1", "ephemeral-2", "ephemeral-3"])
    writes = iter(state_results or [])
    sitting_ids = iter(("sitting-1", "sitting-2", "sitting-3", "sitting-4"))
    soul_lock = asyncio.Lock()
    calls["soul_lock"] = soul_lock

    resources: dict[str, Any] = {}
    memory_items: dict[str, Any] = {}
    calls["image_resources"] = resources
    calls["memory_items"] = memory_items

    class ResourceRepo:
        def list_resources(self, _scope: dict[str, str]) -> dict[str, Any]:
            return dict(resources)

    class MemoryItemRepo:
        def list_items(self, where: dict[str, str], **kwargs: Any) -> dict[str, Any]:
            return {
                key: item
                for key, item in memory_items.items()
                if getattr(item, "resource_id", None) == where["resource_id"]
                and (kwargs.get("include_merged") or not getattr(item, "merged_into", None))
                and (kwargs.get("include_superseded") or not getattr(item, "superseded", False))
            }

        def create_item(self, **kwargs: Any) -> Any:
            item = SimpleNamespace(id=f"reflection-{len(memory_items) + 1}", **kwargs)
            memory_items[item.id] = item
            return item

    class Service:
        database = SimpleNamespace(
            resource_repo=ResourceRepo(), memory_item_repo=MemoryItemRepo()
        )

        async def ensure_dossier_anchors(self, scope: dict[str, str]) -> dict[str, Any]:
            calls["anchors"] = dict(scope)
            return calls["anchor_rows"]

        def build_dossier_index(self, scope: dict[str, str]) -> str:
            calls["index_scope"] = dict(scope)
            return calls["category_index"]

        async def embed(self, summaries: list[str], **kwargs: Any) -> list[list[float]]:
            assert not soul_lock.locked()
            calls["embeddings"].append((summaries, kwargs))
            if calls.get("embedding_hook"):
                await calls["embedding_hook"]()
            return [[0.5] for _ in summaries]

        def graph_delete_memories(self, ids: list[str], **kwargs: Any) -> None:
            calls["deleted_memories"].append((ids, kwargs))
            for memory_id in ids:
                memory_items.pop(memory_id)

        async def memorize(self, **kwargs: Any) -> dict[str, Any]:
            calls["image_memorize"].append(copy.deepcopy(kwargs))
            calls["image_lock_during_memorize"] = soul_lock.locked()
            if memorize_hook is not None:
                await memorize_hook(kwargs)
            await asyncio.sleep(0)
            number = len(calls["image_memorize"])
            resource = next((row for row in resources.values() if row.url == kwargs["resource_url"]), None) or SimpleNamespace(
                id="image-resource" if number == 1 else f"image-resource-{number}",
                url=kwargs["resource_url"],
                modality=kwargs["modality"],
                caption=kwargs["caption"],
            )
            resources[resource.id] = resource
            if memorize_error is not None:
                raise memorize_error
            memory_items["image-item" if number == 1 else f"image-item-{number}"] = SimpleNamespace(resource_id=resource.id)
            return {"resources": [resource]}

    def get_service(scope: dict[str, str]) -> Service:
        calls["service"] += 1
        calls["service_scope"] = dict(scope)
        return Service()

    def load_state(conversation_id: str, **scope: str) -> tuple[dict[str, Any], str, None]:
        calls["state"].append((conversation_id, scope))
        return copy.deepcopy(calls["working_state"]), calls["narrative"], None

    def load_cross(**kwargs: str) -> str:
        calls["cross"].append(kwargs)
        if loader := calls.get("cross_loader"):
            return loader(**kwargs)
        return "[dm][Fictional Friend]\nFictional Friend: hello"

    def load_current(**kwargs: str) -> list[dict[str, Any]]:
        calls["current"] = kwargs
        return [
            {
                "role": "user",
                "content": "A fictional current-chat line.",
                "source_conversation_index": 7,
            }
        ]

    async def retrieve(conversation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls["recalls"].append((conversation_id, copy.deepcopy(payload)))
        calls["lease_locked_during_retrieve"] = mentra_routes._lease_lock.locked()
        calls["soul_locked_during_retrieve"] = soul_lock.locked()
        if retrieve_hook is not None:
            return await retrieve_hook(conversation_id, payload)
        if isinstance(retrieve_result, BaseException):
            raise retrieve_result
        return retrieve_result or {
            "ok": True,
            "retrieve_ms": 123,
            "result": {
                "categories": [{"name": "Fictional Places", "summary": "A remembered coast."}],
                "items": [
                    {
                        "id": "memory-secret-id",
                        "memory_type": "profile",
                        "summary": "The lighthouse keeper preferred cedar tea.",
                    }
                ],
                "resources": [],
            },
        }

    async def mint(**kwargs: str) -> str:
        calls["token"].append(kwargs)
        result = next(results)
        if isinstance(result, BaseException):
            raise result
        return result

    def write_state(*args: Any, **kwargs: Any) -> None:
        calls["state_writes"].append((args, kwargs))
        result = next(writes, None)
        if isinstance(result, Exception):
            raise result
        calls["working_state"].update(copy.deepcopy(kwargs["updates"]))

    async def prepare_annulments(**kwargs: Any) -> list[dict[str, Any]]:
        from app import main

        return await main._prepare_annulment_memories(**kwargs)

    def save_annulments(**kwargs: Any) -> list[str]:
        from app import main

        @contextmanager
        def fake_transaction(*_args):
            before_items = dict(memory_items)
            before_state = copy.deepcopy(calls["working_state"])
            try:
                yield None, None
            except BaseException:
                memory_items.clear()
                memory_items.update(before_items)
                calls["working_state"] = before_state
                raise

        assert soul_lock.locked()
        with monkeypatch.context() as patch:
            patch.setattr(main, "_get_service_from_payload", lambda *_args, **_kwargs: kwargs["service"])
            patch.setattr(main, "_write_conversation_state", write_state)
            patch.setattr(main, "_annulment_write_session", fake_transaction)
            return main._save_actual_annulments(**kwargs)

    def prepare(*args: Any, **kwargs: Any) -> tuple[int, dict[str, Any] | None]:
        calls["memorize_checks"].append((args, kwargs))
        return (prepare_auto_memorize or (lambda *_a, **_k: (0, None)))(*args, **kwargs)

    def schedule(*args: Any, **kwargs: Any) -> str:
        return (schedule_auto_memorize or (lambda *_a, **_k: "launched"))(*args, **kwargs)

    def record_call(*args: Any, **kwargs: Any) -> None:
        calls["route_telemetry"].append((args, copy.deepcopy(kwargs)))

    def test_soul_lock(_user_id: str, _soul_id: str) -> asyncio.Lock:
        if calls.get("reject_soul_lock"):
            raise AssertionError("session end must not wait for soul work")
        return soul_lock

    monkeypatch.setattr(mentra_routes, "_mint_gemini_token", mint)
    monkeypatch.setattr(mentra_routes.secrets, "token_urlsafe", lambda _length: next(sitting_ids))

    def update_config(body):
        config.update(main._merge_model_config(config, body))
        return config

    app = FastAPI()
    calls["background_tasks"] = set()
    register_mentra_routes(
        app,
        get_config=lambda: config,
        update_config=update_config,
        read_config=read_config,
        get_activity_pause=lambda *_args: calls.get("pause_reason"),
        get_service_from_scope=get_service,
        load_turn_state_and_soul_card=load_state,
        build_identity_context=lambda soul_id: f"Today is server time.\nYou are {soul_id}.",
        load_cross_chat_context=load_cross,
        load_current_history=load_current,
        conversation_retrieve=retrieve,
        record_call=record_call,
        log_prompt=lambda instruction, model: calls["prompts"].append((instruction, model)),
        get_storage_dir=lambda: tmp_path,
        get_soul_lock=test_soul_lock,
        write_conversation_state=write_state,
        prepare_annulment_memories=prepare_annulments,
        save_actual_annulments=save_annulments,
        prepare_auto_memorize=prepare,
        schedule_auto_memorize=schedule,
        get_resource_storage_dir=lambda: tmp_path / "resources",
        background_tasks=calls["background_tasks"],
        memorize_lock_key=main._memorize_lock_key,
        memorize_inflight=main._FORCED_MEMORIZE_INFLIGHT,
        consolidation_running=main._CONSOLIDATION_RUNNING,
        finish_memorize_claim=main._finish_memorize_claim,
        set_background_error=lambda *args, **kwargs: calls["background_errors"].append(
            (args, kwargs)
        ),
    )
    return TestClient(app, raise_server_exceptions=raise_server_exceptions), calls, config


def test_mentra_health_and_earcons() -> None:
    config = {"mentra": {}}
    app = FastAPI()
    register_mentra_routes(app, get_config=lambda: config, get_activity_pause=lambda *_args: None)
    client = TestClient(app)

    assert client.get("/integration/mentra/unrelated").status_code == 404

    earcon = client.get("/integration/mentra/earcons/listen-start.wav")
    assert earcon.status_code == 200
    assert earcon.headers["content-type"] == "audio/wav"
    assert earcon.content.startswith(b"RIFF")

    accepted = client.get("/integration/mentra/health")
    assert accepted.status_code == 200
    assert accepted.json() == {"ok": True}


def test_mentra_status_reports_configuration_and_lease_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, _, config = _session_app(monkeypatch, tmp_path)
    path = "/integration/mentra/status?user_id=Fictional%20User&soul_id=Codexia"

    assert client.get(path).status_code == 200
    config["mentra"]["model"] = ""
    assert client.get(path).json()["state"] == "ready"
    key = config["mentra"].pop("gemini_api_key")
    assert client.get(path).json()["state"] == "degraded"
    config["mentra"]["gemini_api_key"] = key
    config["mentra"]["model"] = "gemini-2.5-flash-native-audio-preview-12-2025"
    assert client.get(path).json()["state"] == "ready"

    sitting_id = client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    active = client.get(path).json()
    assert active["state"] == "active"
    assert active["active"] is True
    assert active["mode"] == "continuous"
    assert 0 < active["expires_in"] <= 90
    snapshot_threads = []

    def read_snapshot(**_kwargs):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            snapshot_threads.append("worker")
        else:
            snapshot_threads.append("event-loop")
        return []

    monkeypatch.setattr(mentra_routes.conversation_sources, "load_mentra_history_snapshot", read_snapshot)
    live = client.get(path).json()
    assert live["active"] is True and live["busy"] is True
    assert snapshot_threads == ["worker"]
    assert client.get("/integration/mentra/status").json()["busy"] is True
    assert mentra_routes._leases[START["soul_id"]].sitting_id == sitting_id

    mentra_routes._image_finalize_errors[(START["user_id"], START["soul_id"], sitting_id)] = {
        "mentra_media/phone-1/fictional.png"
    }
    assert client.get(path).json()["state"] == "degraded"
    mentra_routes._leases[START["soul_id"]] = mentra_routes._leases[START["soul_id"]]._replace(
        expires_at=0
    )
    assert client.get(path).json()["state"] == "ready"


def test_mentra_installation_report_persists_and_status_selects_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, _, _ = _session_app(monkeypatch, tmp_path)
    first = {
        "user_id": "Fictional User",
        "soul_id": "Codexia",
        "device_session_id": "fictional-phone",
        "package_name": "com.openalma.mentra",
        "version": "0.1.0",
    }
    assert client.post(
        "/integration/mentra/installation/seen",
        json={**first, "package_name": "example.wrong"},
    ).status_code == 422
    with pytest.raises(owner.OwnerMismatchError):
        client.post(
            "/integration/mentra/installation/seen",
            json={**first, "user_id": "Different User"},
        )
    assert client.post(
        "/integration/mentra/installation/seen", json=first
    ).json() == {"package_name": "com.openalma.mentra", "version": "0.1.0"}
    client.post("/integration/mentra/installation/seen", json=first).raise_for_status()

    records = json.loads((tmp_path / "installations.json").read_text())
    assert list(records["com.openalma.mentra"]) == ["fictional-phone"]

    restarted, _, _ = _session_app(monkeypatch, tmp_path)
    status = restarted.get(
        "/integration/mentra/status",
        params={
            "user_id": "Different User",
            "soul_id": "Different Soul",
            "device_session_id": "fictional-phone",
        },
    ).json()
    assert status["installed_package"] == "com.openalma.mentra"
    assert status["installed_version"] == "0.1.0"
    assert status["installed_soul"] == "Codexia"
    assert isinstance(status["installed_seen_at"], float)
    assert "device_session_id" not in status
    assert "user_id" not in status

    host = {
        "user_id": first["user_id"], "device_session_id": first["device_session_id"],
        "host_package": "com.mentra.mentra.openalma", "host_version": "3.2.0",
        "default_name": "Wrong host",
    }
    before = (tmp_path / "installations.json").read_text()
    assert restarted.post("/integration/mentra/host/seen", json=host).status_code == 409
    assert restarted.post("/integration/mentra/installation/seen", json={
        **first, "host_package": host["host_package"], "host_version": host["host_version"],
    }).status_code == 409
    assert (tmp_path / "installations.json").read_text() == before


def test_mentra_host_capability_merges_with_installation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, _, _ = _session_app(monkeypatch, tmp_path)
    host = {
        "user_id": START["user_id"],
        "device_session_id": START["device_session_id"],
        "host_package": "com.mentra.mentra.openalma",
        "host_version": "3.2.0",
        "default_name": "Test Phone",
    }
    endpoint = "/integration/mentra/host/seen"
    assert client.post(
        endpoint,
        json={**host, "default_name": ""},
    ).status_code == 422
    assert client.post(
        endpoint,
        json={**host, "host_package": "com.mentra.mentra"},
    ).status_code == 422
    with pytest.raises(owner.OwnerMismatchError):
        client.post(endpoint, json={**host, "user_id": "Different User"})

    response = client.post(endpoint, json=host)
    response.raise_for_status()
    assert response.json()["host_package"] == host["host_package"]
    assert isinstance(response.json()["seen_at"], float)

    scoped = client.get(
        "/integration/mentra/status",
        params={"device_session_id": START["device_session_id"]},
    ).json()
    assert scoped["installed_package"] is None
    assert scoped["installations"][0]["display_name"] == "Test Phone"
    assert client.get("/integration/mentra/status").json()["host"] is None
    local = TestClient(client.app, client=("127.0.0.1", 12345))
    local.patch(f"/mentra/installations/{START['device_session_id']}",
                json={"display_name": "My fork"}).raise_for_status()

    installation = {
        **{key: START[key] for key in ("user_id", "soul_id", "device_session_id")},
        "package_name": "com.openalma.mentra",
        "version": "0.1.0",
    }
    client.post(
        "/integration/mentra/installation/seen", json=installation
    ).raise_for_status()
    merged = client.get(
        "/integration/mentra/status",
        params={"device_session_id": START["device_session_id"]},
    ).json()
    assert merged["installed_version"] == "0.1.0"
    assert merged["host"]["host_package"] == host["host_package"]

    client.post(endpoint, json={**host, "host_version": "3.2.1"}).raise_for_status()
    merged = client.get(
        "/integration/mentra/status",
        params={"device_session_id": START["device_session_id"]},
    ).json()
    assert merged["installed_version"] == "0.1.0"
    assert merged["host"]["host_version"] == "3.2.1"
    assert merged["installations"][0]["display_name"] == "My fork"


def test_forget_waits_for_host_reports_to_stop(monkeypatch, tmp_path):
    remote, _, _ = _session_app(monkeypatch, tmp_path)
    client = TestClient(remote.app, client=("127.0.0.1", 12345))
    now = 1000
    monkeypatch.setattr(mentra_routes.time, "time", lambda: now)
    host = {"user_id": START["user_id"], "device_session_id": "fork-phone",
            "host_package": "com.mentra.mentra.openalma", "host_version": "3.2.1",
            "default_name": "Test Phone"}
    endpoint = "/mentra/installations/fork-phone"
    client.post("/integration/mentra/host/seen", json=host).raise_for_status()
    assert client.get("/integration/mentra/status").json()["installations"][0]["host_reporting"]
    assert client.delete(endpoint).status_code == 409
    now += 15
    assert not client.get("/integration/mentra/status").json()["installations"][0]["host_reporting"]
    client.delete(endpoint).raise_for_status()
    assert not client.get("/integration/mentra/status").json()["installations"]
    client.post("/integration/mentra/host/seen", json=host).raise_for_status()
    assert client.get("/integration/mentra/status").json()["installations"][0]["host_reporting"]


def test_installation_metadata_is_local_and_keeps_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    random_id = mentra_routes.secrets.token_urlsafe
    remote, _, _ = _session_app(monkeypatch, tmp_path)
    monkeypatch.setattr(mentra_routes.secrets, "token_urlsafe", random_id)
    endpoint = "/mentra/installations"
    assert remote.post(endpoint).status_code == 403
    client = TestClient(remote.app, client=("127.0.0.1", 12345))
    first = client.post(endpoint).json()
    assert first["display_name"] == "stock_01"
    assert remote.get("/integration/mentra/status").json()["installations"] == []
    another = client.post(endpoint).json()
    assert another["installation_ticket"] != first["installation_ticket"]
    assert first["device_session_id"] == ""
    before = (tmp_path / "installations.json").read_text()
    assert client.post("/integration/mentra/host/seen", json={
        "user_id": START["user_id"], "device_session_id": first["installation_ticket"],
        "host_package": "com.mentra.mentra.openalma", "host_version": "3.2.0",
        "default_name": "Not an app",
    }).status_code == 409
    assert (tmp_path / "installations.json").read_text() == before
    client.delete(f"{endpoint}/{another['installation_ticket']}").raise_for_status()
    key = "stock-app-one"
    first_report = {
        "user_id": START["user_id"], "device_session_id": key,
        "package_name": "com.openalma.mentra", "version": "0.1.0",
        "installation_ticket": first["installation_ticket"],
    }
    seen = "/integration/mentra/installation/seen"
    assert client.post(seen, json={**first_report, "device_session_id": first["installation_ticket"]}).status_code == 409
    async def claim_from_two_apps():
        async with AsyncClient(transport=ASGITransport(app=remote.app), base_url="http://test") as requester:
            return await asyncio.gather(
                requester.post(seen, json=first_report),
                requester.post(seen, json={**first_report, "device_session_id": "copied-profile"}),
            )
    assert [r.status_code for r in asyncio.run(claim_from_two_apps())] == [200, 409]
    client.post(seen, json=first_report).raise_for_status()
    refused = client.post(seen, json={**first_report, "device_session_id": "copied-profile"})
    assert refused.status_code == 409
    assert refused.json()["detail"] == "Setup ticket is unavailable or already used"
    second = client.post(endpoint).json()
    assert second["display_name"] == "stock_02"
    assert second["installation_ticket"] != first["installation_ticket"]
    client.patch(f"{endpoint}/{key}", json={"display_name": "My app"}).raise_for_status()
    client.post("/integration/mentra/installation/seen", json={
        "user_id": START["user_id"], "device_session_id": key,
        "package_name": "com.openalma.mentra", "version": "0.1.1",
    }).raise_for_status()
    status = client.get("/integration/mentra/status").json()
    assert status["installed_device"] is None
    assert status["installations"][0]["display_name"] == "My app"
    receipt = next(r for r in status["installations"] if r["device_session_id"] == key)
    assert receipt["installation_ticket"] == first["installation_ticket"]
    client.post(seen, json=first_report).raise_for_status()
    update = client.post(endpoint, json={"device_session_id": key}).json()
    assert update["device_session_id"] == key
    update_report = {**first_report, "version": "0.1.1", "installation_ticket": update["installation_ticket"]}
    refused = client.post(seen, json={**update_report, "device_session_id": "wrong-update-app"})
    assert refused.status_code == 409
    assert refused.json()["detail"] == "Setup ticket does not match this app"
    client.post(seen, json=update_report).raise_for_status()
    assert client.get(f"/integration/mentra/status?device_session_id={key}").json()["installations"][0]["display_name"] == "My app"
    client.delete(f"{endpoint}/{second['installation_ticket']}").raise_for_status()
    assert len(client.get("/integration/mentra/status").json()["installations"]) == 1
    report = {"user_id": START["user_id"], "device_session_id": "unreserved-stock",
              "package_name": "com.openalma.mentra", "version": "0.1.1"}
    client.post("/integration/mentra/installation/seen", json=report).raise_for_status()
    client.post("/integration/mentra/installation/seen", json=report).raise_for_status()
    names = {r["device_session_id"]: r["display_name"] for r in client.get("/integration/mentra/status").json()["installations"]}
    assert names["unreserved-stock"] == "stock_01"
    waiting = [client.post(endpoint).json() for _ in range(2)]
    assert waiting[0]["display_name"] == waiting[1]["display_name"] == "stock_02"
    for index, reservation in enumerate(reversed(waiting)):
        client.post("/integration/mentra/installation/seen", json={**report,
                    "device_session_id": f"stock-waiting-{index}",
                    "installation_ticket": reservation["installation_ticket"]}).raise_for_status()
    names = {r["device_session_id"]: r["display_name"] for r in client.get("/integration/mentra/status").json()["installations"]}
    assert [names[f"stock-waiting-{index}"] for index in range(2)] == ["stock_02", "stock_03"]
    used = {str(i): {"display_name": f"stock_{i:02d}", "package_name": "com.openalma.mentra"} for i in range(1, 99)}
    assert mentra_routes._stock_name(used) == "stock_99"
    used["99"] = {"display_name": "stock_99", "package_name": "com.openalma.mentra"}
    with pytest.raises(HTTPException, match="stock_99"):
        mentra_routes._stock_name(used)
    entered, release = Event(), Event()
    mint = mentra_routes._mint_gemini_token

    async def waiting_mint(**kwargs: Any) -> str:
        entered.set()
        assert await asyncio.to_thread(release.wait, 2)
        return await mint(**kwargs)

    monkeypatch.setattr(mentra_routes, "_mint_gemini_token", waiting_mint)
    other = client.post(endpoint).json()["installation_ticket"]
    with ThreadPoolExecutor(max_workers=1) as pool:
        starting = pool.submit(client.post, "/integration/mentra/session/start",
                               json={**START, "device_session_id": key})
        try:
            assert entered.wait(2), starting.result(timeout=2).text
            assert client.get(f"/integration/mentra/status?device_session_id={key}").json()["starting"] is True
            assert client.get(f"/integration/mentra/status?device_session_id={other}").json()["starting"] is False
            assert client.delete(f"{endpoint}/{key}").status_code == 409
            client.delete(f"{endpoint}/{other}").raise_for_status()
        finally:
            release.set()
        sitting = starting.result(timeout=2).json()["session_id"]
    assert client.delete(f"{endpoint}/{key}").status_code == 409
    client.post(f"/integration/mentra/session/{sitting}/end",
                json={"user_id": START["user_id"], "soul_id": START["soul_id"]}).raise_for_status()
    client.delete(f"{endpoint}/{key}").raise_for_status()


def test_installation_registry_failure_preserves_the_saved_records(monkeypatch, tmp_path):
    previous = {"com.openalma.mentra": {"saved-app": {"display_name": "Keep me"}}}
    mentra_routes._write_installations(tmp_path, previous)
    path = tmp_path / "installations.json"
    original = path.read_bytes()
    sync_calls = []

    def fail_sync(fd):
        sync_calls.append(fd)
        raise OSError("Fictional sync failure")

    monkeypatch.setattr(mentra_routes.os, "fsync", fail_sync)
    with pytest.raises(OSError, match="Fictional sync failure"):
        mentra_routes._write_installations(tmp_path, {})
    assert len(sync_calls) == 1
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".installations.json.*.tmp"))


def test_invalid_installation_registry_is_reported_without_overwriting(monkeypatch, tmp_path):
    remote, _, _ = _session_app(monkeypatch, tmp_path)
    local = TestClient(remote.app, client=("127.0.0.1", 12345))
    path = tmp_path / "installations.json"
    for contents in ('{"incomplete":', '[]'):
        path.write_text(contents)
        for response in (remote.get("/integration/mentra/status"), local.post("/mentra/installations")):
            assert response.status_code == 503
            assert "installation registry" in response.json()["detail"]
        assert path.read_text() == contents


def test_mentra_status_distinguishes_interruption_conflict_and_missing_transcript(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, _, _ = _session_app(monkeypatch, tmp_path)
    client.post("/integration/mentra/installation/seen", json={
        "user_id": START["user_id"], "soul_id": "Original Installed Soul",
        "device_session_id": START["device_session_id"],
        "package_name": "com.openalma.mentra", "version": "0.1.1",
    }).raise_for_status()
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    append = f"/integration/mentra/session/{sitting_id}/transcripts/append"
    status = "/integration/mentra/status?user_id=Fictional%20User&soul_id=Codexia"

    interrupted = {
        "event_id": f"{sitting_id}:1",
        "sequence": 1,
        "event_kind": "transcript",
        "role": "user",
        "content": "A fictional interrupted sentence.",
        "status": "interrupted",
    }
    assert client.post(append, json={**scope, "events": [interrupted]}).status_code == 200
    assert client.get(status).json()["state"] == "active"

    skipped = {**interrupted, "event_id": f"{sitting_id}:3", "sequence": 3}
    assert client.post(append, json={**scope, "events": [skipped]}).status_code == 409
    assert client.get(status).json()["state"] == "transcript_gap"

    second = {**interrupted, "event_id": f"{sitting_id}:2", "sequence": 2}
    assert client.post(append, json={**scope, "events": [second]}).status_code == 200
    assert client.get(status).json()["state"] == "transcript_gap"

    third = {**interrupted, "event_id": f"{sitting_id}:3", "sequence": 3}
    assert client.post(append, json={**scope, "events": [third]}).status_code == 200
    assert client.get(status).json()["state"] == "active"

    gap = {
        "event_id": f"{sitting_id}:4",
        "sequence": 4,
        "event_kind": "transcript_gap",
        "role": "assistant",
        "content": "Transcript unavailable.",
    }
    assert client.post(append, json={**scope, "events": [gap]}).status_code == 200
    assert client.get(status).json()["state"] == "transcript_gap"
    device_status = client.get("/integration/mentra/status?device_session_id=phone-1").json()
    assert device_status["state"] == "transcript_gap" and device_status["transcript_gap"] is True
    tail = mentra_routes.conversation_sources.load_mentra_tail(
        storage_dir=tmp_path,
        user_id=START["user_id"],
        soul_id=START["soul_id"],
        conversation_id=conversation_sources.MENTRA_CONVERSATION_ID,
        since_cursor=-1,
        recent_fallback_messages=0,
    )
    assert all(row.get("event_kind") != "transcript_gap" for row in tail)

    assert client.post(
        f"/integration/mentra/session/{sitting_id}/end", json=scope
    ).status_code == 200
    for device_id in ("phone-1", "phone-2"):
        assert client.get(status + f"&device_session_id={device_id}").json()["state"] == "transcript_gap"
    assert client.get(
        "/integration/mentra/status",
        params={**scope, "soul_id": "TestSoul", "device_session_id": "phone-2"},
    ).json()["transcript_gap"] is False
    unscoped = client.get("/integration/mentra/status").json()
    assert unscoped["state"] == "ready"
    assert unscoped["installed_soul"] is None
    assert unscoped["installations"][0]["soul_id"] == "Original Installed Soul"
    client.post("/integration/mentra/installation/seen", json={
        "user_id": START["user_id"], "soul_id": START["soul_id"],
        "device_session_id": START["device_session_id"],
        "package_name": "com.openalma.mentra", "version": "0.1.1",
    }).raise_for_status()
    unscoped = client.get("/integration/mentra/status").json()
    assert unscoped["state"] == "ready"
    assert unscoped["installed_soul"] is None
    assert unscoped["installations"][0]["soul_id"] == START["soul_id"]
    next_sitting = client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    assert next_sitting != sitting_id
    assert client.get(status).json()["state"] == "active"
    clean = {**interrupted, "event_id": f"{next_sitting}:5", "sequence": 5}
    client.post(
        f"/integration/mentra/session/{next_sitting}/transcripts/append",
        json={**scope, "events": [clean]},
    ).raise_for_status()
    client.post(f"/integration/mentra/session/{next_sitting}/end", json=scope).raise_for_status()
    assert client.get(status + "&device_session_id=phone-1").json()["state"] == "ready"
    assert client.get("/integration/mentra/status").json()["state"] == "ready"


def test_mentra_status_discovery_and_global_busy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client, _, _ = _session_app(monkeypatch, tmp_path)
    path = "/integration/mentra/status"
    assert client.get(path).status_code == 200
    assert client.get(path + "?user_id=Fictional").status_code == 422
    empty = client.get(path).json()
    assert empty["installed_package"] is None
    assert empty["active"] is False and empty["busy"] is False
    mentra_routes._start_claims["Other Soul"] = ("pending-sitting", "other-phone")
    assert client.get(path).json()["busy"] is True
    mentra_routes._start_claims.clear()
    client.post("/integration/mentra/installation/seen", json={
        **{key: START[key] for key in ("user_id", "soul_id", "device_session_id")},
        "package_name": "com.openalma.mentra", "version": "0.1.0",
    }).raise_for_status()
    client.post("/integration/mentra/session/start", json=START).raise_for_status()
    discovered = client.get(path).json()
    assert discovered["installed_user"] is None
    assert discovered["installations"][0]["user_id"] == START["user_id"]
    assert discovered["installed_device"] is None
    assert discovered["sessions"][0]["device_session_id"] == START["device_session_id"]
    assert discovered["active"] is True and discovered["busy"] is True
    other = client.get(path + "?device_session_id=other-phone").json()
    assert other["active"] is False and other["busy"] is True


def test_start_validation_precedes_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, config = _session_app(monkeypatch, tmp_path)

    assert client.post("/integration/mentra/session/start", json={}).status_code == 422
    invalid_device = {**START, "device_session_id": "bad/id"}
    assert client.post(
        "/integration/mentra/session/start", json=invalid_device
    ).status_code == 422
    misspelled = client.post("/integration/mentra/session/start", json={
        **START, "recovrey": _recovered_transcripts(),
    })
    assert misspelled.status_code == 422
    assert misspelled.json() == {"detail": "Invalid session start"}
    assert "Recovered fictional line" not in misspelled.text
    assert calls["service"] == 0
    assert calls["token"] == []


@pytest.mark.parametrize(("field", "value"), [
    *[("model", model) for model in (
        "", "custom-model", "models/gemini-3.8-live", " gemini-3.8-live",
        "gemini-3.8-live-extended-thinking ", "GEMINI-3.8-LIVE", 38,
    )],
    *[("vision", vision) for vision in (None, 0, 1, "true", "false", "", [], {})],
])
def test_start_rejects_invalid_model_or_vision_before_side_effects(monkeypatch, tmp_path, field, value):
    client, calls, config = _session_app(monkeypatch, tmp_path)
    before = copy.deepcopy(config)
    files = sorted(tmp_path.rglob("*"))

    def unexpected_history(**_kwargs):
        raise AssertionError("invalid start flag must not read or recover history")

    monkeypatch.setattr(mentra_routes.conversation_sources, "load_mentra_history_snapshot", unexpected_history)
    response = client.post("/integration/mentra/session/start", json={
        **START, field: value, "recovery": _recovered_transcripts(),
    })
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid session start"}
    assert calls["service"] == 0
    assert calls["token"] == calls["state_writes"] == calls["route_telemetry"] == []
    assert not mentra_routes._leases and not mentra_routes._start_claims
    assert config == before
    assert sorted(tmp_path.rglob("*")) == files


def _model_app(monkeypatch, tmp_path):
    from app import config as app_config, main

    config = _configured()
    config["storage"].update(sqlite_dir=str(tmp_path), resources_dir=str(tmp_path / "resources"))
    config["storage"]["metadata_store"]["dsn"] = f"sqlite:///{tmp_path / 'memu.db'}"
    owner.create_owner(config, START["user_id"])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(app_config, "config_path", lambda: path)
    monkeypatch.setattr(main, "_config_path", lambda: path)
    monkeypatch.setattr(main, "_CONFIG", config)

    async def forbidden_mint(**_kwargs):
        pytest.fail("Model settings must not call the provider")

    monkeypatch.setattr(mentra_routes, "_mint_gemini_token", forbidden_mint)
    return TestClient(main.app), config


def test_model_setting_reads_disk_and_preserves_unrelated_edits(monkeypatch, tmp_path):
    from app import main

    client, _ = _model_app(monkeypatch, tmp_path)
    path = tmp_path / "config.json"
    edited = json.loads(path.read_text())
    edited["mentra"].update(model="hand-edited-live-model", voice="Aoede")
    edited["retrieve"] = {"top_k": 17}
    path.write_text(json.dumps(edited))
    response = client.get("/integration/mentra/status")
    assert response.status_code == 200
    assert response.json()["configured_model"] == "hand-edited-live-model"

    # Another writer changes unrelated settings after the phone's last read.
    edited["mentra"]["session_warning_seconds"] = 90
    edited["debug"] = {"log_prompts": False}
    path.write_text(json.dumps(edited))
    response = client.post("/integration/mentra/model", json={
        "user_id": START["user_id"], "model": "gemini-3.8-live-extended-thinking",
    })
    assert response.status_code == 200
    assert response.json() == {"ok": True, "configured_model": "gemini-3.8-live-extended-thinking"}
    edited["mentra"]["model"] = "gemini-3.8-live-extended-thinking"
    assert json.loads(path.read_text()) == main._CONFIG == edited


def test_model_setting_save_failure_keeps_disk_and_runtime(monkeypatch, tmp_path):
    from app import main

    client, config = _model_app(monkeypatch, tmp_path)
    path = tmp_path / "config.json"
    original = path.read_bytes()
    before = copy.deepcopy(config)

    def fail_save(_config):
        raise OSError("fictional save failure with permanent-secret")

    monkeypatch.setattr(main, "_save_config", fail_save)

    def forbidden(*_args):
        pytest.fail("Failed save must not update runtime or close services")

    for name in ("_refresh_runtime_limits", "_ensure_storage_paths", "_clear_cached_services"):
        monkeypatch.setattr(main, name, forbidden)
    response = client.post("/integration/mentra/model", json={
        "user_id": START["user_id"], "model": "gemini-3.8-live",
    })
    assert response.status_code == 400
    assert response.json() == {"detail": "Internal Server Error. Check server logs."}
    assert path.read_bytes() == original
    assert main._CONFIG is config and config == before


def test_generic_config_malformed_json_keeps_existing_error_contract(monkeypatch, tmp_path):
    client, _ = _model_app(monkeypatch, tmp_path)
    path = tmp_path / "config.json"
    original = path.read_bytes()
    response = client.post("/config", content="{", headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Internal Server Error. Check server logs."}
    assert path.read_bytes() == original


@pytest.mark.parametrize("body", [
    {"model": "gemini-3.8-live"},
    {"user_id": START["user_id"]},
    {"user_id": START["user_id"], "model": "custom-model"},
    {"user_id": START["user_id"], "model": "gemini-3.8-live", "voice": "Aoede"},
])
def test_model_setting_rejects_invalid_body_without_writes(monkeypatch, tmp_path, body):
    client, _ = _model_app(monkeypatch, tmp_path)
    path = tmp_path / "config.json"
    original = path.read_bytes()
    response = client.post("/integration/mentra/model", json=body)
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid model setting"}
    assert path.read_bytes() == original


def test_model_setting_requires_owner(monkeypatch, tmp_path):
    client, _ = _model_app(monkeypatch, tmp_path)
    path = tmp_path / "config.json"
    original = path.read_bytes()
    assert client.post("/integration/mentra/model", json={
        "user_id": "Other User", "model": "gemini-3.8-live",
    }).status_code == 409
    assert path.read_bytes() == original


def test_start_reloads_config_before_model_validation(monkeypatch, tmp_path):
    from app import main

    client, _ = _model_app(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "_soul_activity_pause", lambda *_args: None)
    path = tmp_path / "config.json"
    edited = json.loads(path.read_text())
    edited["mentra"]["model"] = ""
    path.write_text(json.dumps(edited))
    response = client.post("/integration/mentra/session/start", json=START)
    assert response.status_code == 503, response.json()
    assert response.json() == {"detail": "Mentra Gemini model is not configured"}


def test_active_sitting_does_not_read_in_progress_config_edits(monkeypatch, tmp_path):
    path = tmp_path / "config.json"
    reads = []

    def read_config():
        reads.append(True)
        return json.loads(path.read_text())

    client, _, config = _session_app(monkeypatch, tmp_path, read_config=read_config)
    path.write_text(json.dumps(config))
    assert client.get("/integration/mentra/status").status_code == 200
    response = client.post("/integration/mentra/session/start", json=START)
    assert response.status_code == 200
    assert len(reads) == 2
    base = f"/integration/mentra/session/{response.json()['session_id']}"
    scope = {key: START[key] for key in ("user_id", "soul_id")}
    path.write_text('{"mentra":')
    assert client.post(f"{base}/heartbeat", json=scope).status_code == 200
    assert client.post(f"{base}/token", json=scope).status_code == 200
    assert client.post(f"{base}/transcripts/append", json={**scope, "events": [{
        "event_id": f"{response.json()['session_id']}:1", "sequence": 1,
        "event_kind": "transcript", "role": "user", "status": "complete",
        "content": "A fictional conversation during a settings edit.",
    }]}).status_code == 200
    assert client.post(f"{base}/end", json=scope).status_code == 200
    assert len(reads) == 2
    assert path.read_text() == '{"mentra":'


def test_model_setting_keeps_existing_sitting_pinned(monkeypatch, tmp_path):
    client, calls, config = _session_app(monkeypatch, tmp_path)
    config["mentra"]["model"] = "gemini-3.8-live"
    response = client.post("/integration/mentra/session/start", json=START)
    assert response.status_code == 200
    assert response.json()["model"] == calls["token"][0]["model"] == "gemini-3.8-live"

    response_model = client.post("/integration/mentra/model", json={
        "user_id": START["user_id"], "model": "gemini-3.8-live-extended-thinking",
    })
    assert response_model.status_code == 200
    scope = {key: START[key] for key in ("user_id", "soul_id", "device_session_id")}
    base = f"/integration/mentra/session/{response.json()['session_id']}"
    assert client.post(f"{base}/token", json=scope).status_code == 200
    assert calls["token"][1] == calls["token"][0]
    assert client.post(f"{base}/end", json=scope).status_code == 200
    next_start = client.post("/integration/mentra/session/start", json=START)
    assert next_start.status_code == 200
    assert next_start.json()["model"] == calls["token"][2]["model"] == "gemini-3.8-live-extended-thinking"


@pytest.mark.parametrize(("choice", "configured"), [
    (None, "custom-live-model"),
    ("gemini-3.8-live", "custom-live-model"),
    ("gemini-3.8-live-extended-thinking", "custom-live-model"),
    ("gemini-3.8-live", None),
    ("gemini-3.8-live-extended-thinking", None),
])
def test_start_model_choice_and_renewal_remain_lease_pinned(monkeypatch, tmp_path, choice, configured):
    client, calls, config = _session_app(monkeypatch, tmp_path)
    if configured is None:
        config["mentra"].pop("model")
    else:
        config["mentra"]["model"] = configured
    before = copy.deepcopy(config)
    scope = {key: START[key] for key in ("user_id", "soul_id", "device_session_id")}
    status = client.get("/integration/mentra/status", params=scope).json()
    assert status["configured_model"] == configured
    assert status["state"] == "ready"

    response = client.post("/integration/mentra/session/start", json={
        **START, **({"model": choice} if choice is not None else {}),
    })
    assert response.status_code == 200
    resolved = choice or configured
    assert response.json()["model"] == calls["token"][0]["model"] == resolved
    prompt = calls["token"][0]["system_instruction"]
    assert prompt.count(mentra_routes._THINKING_COMPLETION_GUIDANCE) == (
        1 if resolved == "gemini-3.8-live-extended-thinking" else 0
    )
    assert config == before

    config["mentra"]["model"] = "edited-server-model"
    base = f"/integration/mentra/session/{response.json()['session_id']}"
    assert client.post(f"{base}/token", json=scope).status_code == 200
    assert calls["token"][1] == calls["token"][0]
    status = client.get("/integration/mentra/status", params=scope).json()
    assert status["configured_model"] == "edited-server-model"
    assert status["state"] == "active"
    assert client.post(f"{base}/end", json=scope).status_code == 200
    assert client.get("/integration/mentra/status", params=scope).json()["state"] == "ready"


@pytest.mark.parametrize("marker", [None, "iris-photo-work-v0", "iris-photo-work-v1"])
def test_start_rejects_incompatible_photo_protocol_before_mint_or_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, marker: str | None,
) -> None:
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    body = {**START, "photo_work_protocol": marker}
    if marker is None:
        body.pop("photo_work_protocol")
    response = client.post("/integration/mentra/session/start", json=body)
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid session start"}
    assert calls["service"] == 0
    assert calls["token"] == []
    assert calls["state_writes"] == []
    assert not mentra_routes._leases
    assert not mentra_routes._start_claims


def test_start_allows_same_user_and_soul_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client, _, _ = _session_app(monkeypatch, tmp_path)
    assert client.post(
        "/integration/mentra/session/start",
        json={**START, "user_id": " Codexia "},
    ).status_code == 200


def test_start_requires_model_and_voice_before_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, config = _session_app(monkeypatch, tmp_path)

    config["mentra"]["model"] = ""
    assert client.post("/integration/mentra/session/start", json=START).status_code == 503
    config["mentra"]["model"] = "model"
    config["mentra"]["voice"] = ""
    assert client.post("/integration/mentra/session/start", json=START).status_code == 503
    assert calls["service"] == 0


@pytest.mark.parametrize("mode", ["continuous", "manual"])
def test_start_threads_speech_mode_to_token_mint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    client, calls, _ = _session_app(monkeypatch, tmp_path)

    response = client.post(
        "/integration/mentra/session/start",
        json={**START, "mode": mode},
    )

    assert response.status_code == 200
    assert calls["token"][0]["mode"] == mode


def test_start_builds_bounded_instruction_and_returns_only_client_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(monkeypatch, tmp_path)

    response = client.post("/integration/mentra/session/start", json=START)
    assert response.status_code == 200
    body = response.json()
    assert conversation_sources.MENTRA_CONVERSATION_ID == "mentra:iris"
    assert body == {
        "ok": True,
        "session_id": "sitting-1",
        "conversation_id": conversation_sources.MENTRA_CONVERSATION_ID,
        "next_transcript_sequence": 1,
        "recovery": None,
        "photo_work_protocol": "iris-photo-work-v2",
        "model": "gemini-2.5-flash-native-audio-preview-12-2025",
        "voice": "Kore",
        "ephemeral_token": "ephemeral-1",
        "websocket": {
            "api_version": "v1beta",
            "method": "BidiGenerateContentConstrained",
            "input_audio_rate_hz": 16000,
            "output_audio_rate_hz": 24000,
        },
        "lease_seconds": 90,
        "session_warning_seconds": 0,
    }
    assert calls["anchors"] == {"user_id": "Fictional User", "soul_id": "Codexia"}
    assert calls["state"][0][0] == conversation_sources.MENTRA_CONVERSATION_ID
    assert calls["cross"][0]["conversation_id"] == conversation_sources.MENTRA_CONVERSATION_ID
    prompt = calls["token"][0]["system_instruction"]
    assert prompt.count(mentra_routes._PHOTO_CAPTION_GUIDANCE) == 1
    assert calls["prompts"] == [(prompt, body["model"])]
    headings = [
        "Today is server time.",
        "## My Narrative Self",
        "## My Identity and Lived Experience",
        "## The User's Identity and Lived Experience",
        "## My Categories",
        "## Recent Conversations and Activities",
        "Use recall_memory continuously during conversations so you have relevant memories for context.",
        "Speak naturally and concisely",
    ]
    assert [prompt.index(text) for text in headings] == sorted(prompt.index(text) for text in headings)
    assert calls["index_scope"] == calls["anchors"]
    assert prompt.count(calls["category_index"]) == 1
    assert prompt.count("I am Codexia.") == prompt.count("The user likes careful work.") == 1
    assert "My Working Thoughts" not in prompt and "My Intentions" not in prompt
    serialized = json.dumps(body)
    assert "permanent-secret" not in serialized
    assert "I am Codexia" not in serialized
    assert "Fictional Friend" not in serialized
    start_call = calls["route_telemetry"][-1]
    assert start_call[0] == (
        "mentra.start",
        {"user": {"user_id": "Fictional User", "soul_id": "Codexia"}},
    )
    assert set(start_call[1]["info"]) == {
        "setupMs",
        "anchorsMs",
        "stateMs",
        "contextMs",
        "tokenMs",
        "totalMs",
    }


def test_start_vision_replaces_only_photo_and_invitation_guidance(monkeypatch, tmp_path):
    client, calls, config = _session_app(monkeypatch, tmp_path)
    before = copy.deepcopy(config)
    for flags in ({}, {"vision": False}, {"vision": True}):
        response = client.post("/integration/mentra/session/start", json={**START, **flags})
        assert response.status_code == 200
        assert response.json()["recovery"] is None
        assert response.json()["photo_work_protocol"] == "iris-photo-work-v2"

    default, audio, vision = [call["system_instruction"] for call in calls["token"]]
    approved = (
        'The text "nokoru" is an internal invitation from Iris to reconsider the conversation, not a request '
        'from your human. When you receive it, continue the conversation only if a reserved human would '
        'have something more to add. Otherwise call nothing_to_add instead of speaking. If a memory arrives, '
        'reconsider continuing the conversation if the new context brings something to light. '
        'Do not acknowledge or describe the cue. Photo tasks arrive when you call nothing_to_add. '
        'A task labeled please-caption-<timestamp> asks you to caption the photo attached to that task; '
        'use the complete timestamp after please-caption-, including any collision suffix, to identify the image. '
        'A photo-selection task shows you recent photos and says how many you may keep at most. '
        'Keep only photos you find worth remembering; keeping none is fine. '
        'Use the complete timestamp printed on each kept photo, including any suffix, to identify the image. '
        'For each photo you caption or keep, use save_photo_caption(image_id, caption) to save your perception '
        'of it, so you will remember what this moment was about later on; pass its complete timestamp as image_id. '
        'Do not speak the caption out loud. When you are done with a task, call nothing_to_add to check for another task. '
        'Live frames and catch-up collages provide context; they are not caption or photo-selection tasks. '
        'Catch-up collages are frames you missed while you were working on captioning memories. '
        'Collage reading order is like prose in a book.'
    )
    original = f"{mentra_routes._PHOTO_CAPTION_GUIDANCE}\n\n{mentra_routes._IRIS_INVITATION_GUIDANCE}"
    assert default == audio
    assert audio.count(original) == 1
    frame_recall = "Also use recall_memory if you see something you want to talk about."
    assert frame_recall not in audio
    assert vision.count(frame_recall) == 1
    assert vision == audio.replace(original, approved).replace(
        "The memories will guide you going forward.\n\n",
        f"The memories will guide you going forward.\n\n{frame_recall}\n\n",
        1,
    )
    assert vision.count(approved) == 1
    assert vision.count('The text "nokoru"') == 1
    assert config == before


def test_bootstrap_omits_empty_sections_and_uses_shared_anchor_renderer():
    prompt = mentra_routes._build_bootstrap_instruction(
        identity="A fictional identity.", narrative="", anchors={"user": "A fictional user fact."},
        all_categories_summary="", chats="",
    )
    assert "## The User's Identity and Lived Experience\nA fictional user fact." in prompt
    for absent in ("## My Narrative Self", "## My Identity and Lived Experience", "## My Categories",
                   "## Recent Conversations and Activities"):
        assert absent not in prompt
    assert 'If a write reports "busy, not saved", nothing was saved.' in prompt
    assert "If you never hear back, it may have been saved" in prompt
    photo_guidance = (
        "A task labeled please-caption-<image_id> asks you to caption the photo attached to that task. "
        "Use save_photo_caption(image_id, caption) to save your perception of that photo, so you will "
        "remember what this moment was about later on. Use the exact text after please-caption- as image_id. "
        "After the caption is saved, call nothing_to_add to check for another task."
    )
    assert prompt.count(photo_guidance) == 1
    assert "If a photo successfully arrives" not in prompt
    assert prompt.index(photo_guidance) < prompt.index(mentra_routes._IRIS_INVITATION_GUIDANCE)
    assert prompt.count(mentra_routes._IRIS_INVITATION_GUIDANCE) == 1
    assert 'The text "nokoru" is an internal invitation from Iris' in prompt
    assert "Otherwise call nothing_to_add instead of speaking" in prompt


def test_live_working_state_is_fresh_and_thought_write_is_bounded(monkeypatch, tmp_path):
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    calls["working_state"] = {
        "memory_cache": [f"Fictional thought {i}" for i in range(7)],
        "intentions_active": [{"id": "learn-tides", "text": "Learn fictional tides."}],
        "prior_context": "Keep prior context.", "undo": "Keep undo.",
    }
    sitting = client.post("/integration/mentra/session/start", json=START).json()["session_id"]
    root = f"/integration/mentra/session/{sitting}"
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    prompt = calls["token"][0]["system_instruction"]
    assert "Fictional thought" not in prompt and "Learn fictional tides" not in prompt
    read = client.post(f"{root}/working-state", json=scope)
    assert read.json() == {
        "ok": True, "working_thoughts": calls["working_state"]["memory_cache"],
        "intentions": calls["working_state"]["intentions_active"],
    }
    assert client.post(f"{root}/working-thought", json={**scope, "thought": "x" * 601}).json() == {
        "ok": True, "status": "saved", "truncated": True,
    }
    current = client.post(f"{root}/working-state", json=scope).json()
    assert current["working_thoughts"] == [f"Fictional thought {i}" for i in range(1, 7)] + ["x" * 600]
    assert calls["working_state"]["prior_context"] == "Keep prior context."
    assert calls["working_state"]["undo"] == "Keep undo."
    assert set(calls["state_writes"][-1][1]["updates"]) == {"memory_cache"}
    assert calls["state_writes"][-1][0] == (conversation_sources.MENTRA_CONVERSATION_ID,)
    assert calls["state_writes"][-1][1]["user_id"] == START["user_id"]
    assert calls["state_writes"][-1][1]["soul_id"] == START["soul_id"]
    assert calls["embeddings"] == [] and calls["memorize_checks"] == []


@pytest.mark.parametrize("status", ["completed", "deleted"])
def test_annul_intention_records_only_actual_removal(monkeypatch, tmp_path, status):
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    calls["working_state"]["intentions_active"] = [
        {"id": "learn-tides", "text": "Learn fictional tides."},
        {"id": "paint-coast", "text": "Paint a fictional coast."},
    ]
    client.post("/integration/mentra/session/start", json=START)
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    path = "/integration/mentra/session/sitting-1/annul-intention"
    body = {**scope, "intention_id": "learn-tides", "status": status, "note": "A fictional reason."}
    response = client.post(path, json=body)
    assert response.json() == {"ok": True, "status": "removed", "memory_ids": ["reflection-1"]}
    item = calls["memory_items"]["reflection-1"]
    assert f'as {status}. Note: A fictional reason.' in item.summary
    assert item.user_data == scope and item.conversation_id == conversation_sources.MENTRA_CONVERSATION_ID
    assert calls["embeddings"][0][1] == {"profile": "embedding"}
    assert calls["working_state"]["intentions_active"] == [{"id": "paint-coast", "text": "Paint a fictional coast."}]
    assert set(calls["state_writes"][-1][1]["updates"]) == {"intentions_active"}
    assert client.post(path, json=body).json()["status"] == "not_found"
    assert len(calls["embeddings"]) == len(calls["state_writes"]) == len(calls["memory_items"]) == 1


@pytest.mark.parametrize("tool,arguments", [
    ("working-thought", {"thought": "A fictional hypothesis."}),
    ("annul-intention", {"intention_id": "learn-tides", "status": "completed"}),
])
@pytest.mark.parametrize("failure", [RuntimeError, TimeoutError])
def test_live_write_save_failure_keeps_state_and_releases_lock(monkeypatch, tmp_path, tool, arguments, failure):
    client, calls, _ = _session_app(
        monkeypatch, tmp_path, state_results=[failure("fictional save failure")],
        raise_server_exceptions=False,
    )
    calls["working_state"]["intentions_active"] = [{"id": "learn-tides", "text": "Learn fictional tides."}]
    before = copy.deepcopy(calls["working_state"])
    client.post("/integration/mentra/session/start", json=START)
    response = client.post(f"/integration/mentra/session/sitting-1/{tool}", json={
        "user_id": START["user_id"], "soul_id": START["soul_id"], **arguments,
    })
    assert response.status_code == 500
    assert calls["working_state"] == before and not calls["soul_lock"].locked()
    assert calls["memory_items"] == {}
    if tool == "annul-intention":
        assert calls["deleted_memories"] == []


def test_live_tools_reject_wrong_scope_ended_sitting_pause_and_invalid_arguments(monkeypatch, tmp_path):
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    client.post("/integration/mentra/session/start", json=START)
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    tools = [("working-state", {}), ("working-thought", {"thought": "Fictional thought."}),
             ("annul-intention", {"intention_id": "learn-tides", "status": "deleted"})]
    for tool, arguments in tools:
        assert client.post(f"/integration/mentra/session/wrong/{tool}", json={**scope, **arguments}).status_code == 404
        assert client.post(f"/integration/mentra/session/sitting-1/{tool}", json={
            **scope, **arguments, "soul_id": "Another Fictional Soul",
        }).status_code == 404
        assert client.post(f"/integration/mentra/session/sitting-1/{tool}", json={
            **scope, **arguments, "user_id": "Another Fictional User",
        }).status_code == 404
        calls["pause_reason"] = "Fictional pause."
        paused = client.post(f"/integration/mentra/session/sitting-1/{tool}", json={**scope, **arguments})
        assert paused.status_code == 409 and paused.json()["detail"]["code"] == "soul_paused"
        calls["pause_reason"] = None
    for tool, arguments in [("working-state", {"unexpected": True}), ("working-thought", {"thought": "  "}),
                            ("annul-intention", {"intention_id": "learn-tides", "status": "active"})]:
        assert client.post(f"/integration/mentra/session/sitting-1/{tool}", json={**scope, **arguments}).status_code == 422
    client.post("/integration/mentra/session/sitting-1/end", json=scope)
    for tool, arguments in tools:
        assert client.post(f"/integration/mentra/session/sitting-1/{tool}", json={**scope, **arguments}).status_code == 404
    assert calls["state_writes"] == calls["embeddings"] == []


def test_live_writes_return_busy_without_preparation_and_bound_admission(monkeypatch, tmp_path):
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    client.post("/integration/mentra/session/start", json=START)
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    busy = {"ok": False, "status": "busy", "message": "busy, not saved"}

    async def contend():
        async with AsyncClient(transport=ASGITransport(app=client.app), base_url="http://test") as http:
            async with calls["soul_lock"]:
                for tool, args in [("working-thought", {"thought": "Fictional thought."}),
                                   ("annul-intention", {"intention_id": "learn-tides", "status": "completed"})]:
                    result = await http.post(f"/integration/mentra/session/sitting-1/{tool}", json={**scope, **args})
                    assert result.json() == busy
                assert (await http.post("/integration/mentra/session/sitting-1/working-state", json=scope)).status_code == 200

    asyncio.run(asyncio.wait_for(contend(), timeout=2))
    assert calls["state_writes"] == calls["embeddings"] == []

    async def delayed_acquire():
        await asyncio.sleep(1)
        raise AssertionError("admission must be bounded")

    monkeypatch.setattr(calls["soul_lock"], "acquire", delayed_acquire)
    monkeypatch.setattr(mentra_routes, "_RECALL_TIMEOUT_SECONDS", 0.01)
    assert client.post("/integration/mentra/session/sitting-1/working-thought", json={
        **scope, "thought": "Fictional thought.",
    }).json() == busy
    assert not calls["soul_lock"].locked() and calls["state_writes"] == []


@pytest.mark.parametrize("provider_timeout", [False, True])
def test_annulment_preparation_timeout_reports_not_saved(monkeypatch, tmp_path, provider_timeout):
    client, calls, _ = _session_app(monkeypatch, tmp_path, raise_server_exceptions=False)
    calls["working_state"]["intentions_active"] = [{"id": "learn-tides", "text": "Learn fictional tides."}]

    async def stalled_embedding():
        if provider_timeout:
            raise TimeoutError("fictional provider timeout before persistence")
        await asyncio.sleep(1)

    calls["embedding_hook"] = stalled_embedding
    client.post("/integration/mentra/session/start", json=START)
    monkeypatch.setattr(mentra_routes, "_RECALL_TIMEOUT_SECONDS", 0.01)
    result = client.post("/integration/mentra/session/sitting-1/annul-intention", json={
        "user_id": START["user_id"], "soul_id": START["soul_id"],
        "intention_id": "learn-tides", "status": "completed",
    })
    assert result.status_code == 200
    assert result.json() == {"ok": False, "status": "busy", "message": "busy, not saved"}
    assert calls["state_writes"] == [] and calls["memory_items"] == {}
    assert not calls["soul_lock"].locked()


def test_annulment_shares_one_deadline_with_lock_admission(monkeypatch, tmp_path):
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    calls["working_state"]["intentions_active"] = [{"id": "learn-tides", "text": "Learn fictional tides."}]

    async def prepare_then_delay_admission():
        await asyncio.sleep(0.03)

        async def delayed_acquire():
            await asyncio.sleep(0.03)
            raise AssertionError("must not start another full timeout after preparation")

        monkeypatch.setattr(calls["soul_lock"], "acquire", delayed_acquire)

    calls["embedding_hook"] = prepare_then_delay_admission
    client.post("/integration/mentra/session/start", json=START)
    monkeypatch.setattr(mentra_routes, "_RECALL_TIMEOUT_SECONDS", 0.05)
    result = client.post("/integration/mentra/session/sitting-1/annul-intention", json={
        "user_id": START["user_id"], "soul_id": START["soul_id"],
        "intention_id": "learn-tides", "status": "completed",
    })
    assert result.json() == {"ok": False, "status": "busy", "message": "busy, not saved"}
    assert calls["state_writes"] == [] and calls["memory_items"] == {}


@pytest.mark.parametrize("change,expected", [("stop", 404), ("pause", 409), ("remove", 200), ("busy", 200)])
def test_annulment_rechecks_after_embedding(monkeypatch, tmp_path, change, expected):
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    calls["working_state"]["intentions_active"] = [{"id": "learn-tides", "text": "Learn fictional tides."}]
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}

    async def run():
        async with AsyncClient(transport=ASGITransport(app=client.app), base_url="http://test") as http:
            async def change_during_embedding():
                assert not calls["soul_lock"].locked()
                if change == "stop":
                    assert (await http.post("/integration/mentra/session/sitting-1/end", json=scope)).status_code == 200
                    # A replacement sitting must not authorize the old request.
                    assert (await http.post("/integration/mentra/session/start", json=START)).status_code == 200
                elif change == "pause":
                    calls["pause_reason"] = "Fictional pause during embedding."
                elif change == "remove":
                    calls["working_state"]["intentions_active"] = []
                else:
                    await calls["soul_lock"].acquire()

            calls["embedding_hook"] = change_during_embedding
            assert (await http.post("/integration/mentra/session/start", json=START)).status_code == 200
            return await http.post("/integration/mentra/session/sitting-1/annul-intention", json={
                **scope, "intention_id": "learn-tides", "status": "completed",
            })

    response = asyncio.run(asyncio.wait_for(run(), timeout=3))
    assert response.status_code == expected
    if change in {"remove", "busy"}:
        assert response.json()["status"] == {"remove": "not_found", "busy": "busy"}[change]
    if change == "busy":
        calls["soul_lock"].release()
    assert len(calls["embeddings"]) == 1
    assert calls["state_writes"] == [] and calls["memory_items"] == {}
    assert not calls["soul_lock"].locked()


@pytest.mark.parametrize("change,expected", [("stop", 404), ("pause", 409)])
def test_thought_rechecks_lease_and_pause_after_lock_admission(monkeypatch, tmp_path, change, expected):
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    client.post("/integration/mentra/session/start", json=START)
    original_acquire = calls["soul_lock"].acquire

    async def acquire_and_change():
        result = await original_acquire()
        if change == "stop":
            mentra_routes._leases.clear()
        else:
            calls["pause_reason"] = "Fictional pause during admission."
        return result

    monkeypatch.setattr(calls["soul_lock"], "acquire", acquire_and_change)
    result = client.post("/integration/mentra/session/sitting-1/working-thought", json={
        "user_id": START["user_id"], "soul_id": START["soul_id"], "thought": "Fictional thought.",
    })
    assert result.status_code == expected
    assert not calls["soul_lock"].locked() and calls["state_writes"] == []


def test_lost_write_response_does_not_undo_or_replay_and_can_be_read(monkeypatch, tmp_path):
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    client.post("/integration/mentra/session/start", json=START)
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}

    async def lose_response(asgi_scope, receive, send):
        async def drop(_message):
            raise asyncio.CancelledError()

        await client.app(asgi_scope, receive, drop)

    async def run():
        async with AsyncClient(transport=ASGITransport(app=lose_response), base_url="http://test") as http:
            with pytest.raises(asyncio.CancelledError):
                await http.post("/integration/mentra/session/sitting-1/working-thought", json={
                    **scope, "thought": "A fictional surviving thought.",
                })

    asyncio.run(run())
    assert len(calls["state_writes"]) == 1 and not calls["soul_lock"].locked()
    result = client.post("/integration/mentra/session/sitting-1/working-state", json=scope)
    assert result.json()["working_thoughts"] == ["A fictional surviving thought."]
    assert len(calls["state_writes"]) == 1


def test_recall_preserves_structured_pause_through_real_server_wrapper(monkeypatch, tmp_path):
    detail = {"code": "soul_paused", "message": "Codexia is paused. Retry in OpenAlma launcher."}
    client, _, _ = _session_app(monkeypatch, tmp_path, retrieve_result=HTTPException(status_code=409, detail=detail))
    session = client.post("/integration/mentra/session/start", json=START).json()["session_id"]
    response = client.post(f"/integration/mentra/session/{session}/recall", json={
        "user_id": START["user_id"], "soul_id": START["soul_id"], "query": "Test",
    })
    assert response.status_code == 409 and response.json()["detail"] == detail


def test_pause_blocks_new_sitting_but_keeps_lease_renewal_heartbeat_and_end(monkeypatch, tmp_path):
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    calls["pause_reason"] = "Memorize failed"
    response = client.post("/integration/mentra/session/start", json=START)
    assert response.status_code == 409 and response.json()["detail"]["code"] == "soul_paused"
    assert not calls["token"] and not mentra_routes._leases
    calls["pause_reason"] = None
    response = client.post("/integration/mentra/session/start", json=START)
    response.raise_for_status()
    session_id = response.json()["session_id"]
    calls["pause_reason"] = "Memorize failed"
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    heartbeat = client.post(f"/integration/mentra/session/{session_id}/heartbeat", json=scope)
    assert heartbeat.status_code == 200 and heartbeat.json()["pause_reason"] == "Memorize failed"
    assert client.post(f"/integration/mentra/session/{session_id}/token", json=scope).status_code == 200
    assert client.post(f"/integration/mentra/session/{session_id}/end", json=scope).status_code == 200


def test_lease_resume_heartbeat_and_end_are_scoped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(monkeypatch, tmp_path)

    async def replace_with_waiting_append():
        async with AsyncClient(
            transport=ASGITransport(app=client.app), base_url="http://test"
        ) as async_client:
            first = await async_client.post("/integration/mentra/session/start", json=START)
            batch = {
                "user_id": START["user_id"],
                "soul_id": START["soul_id"],
                "events": [{
                    "event_id": "sitting-1:1",
                    "sequence": 1,
                    "event_kind": "transcript",
                    "role": "user",
                    "content": "A fictional waiting line.",
                    "status": "complete",
                }],
            }
            lock = calls["soul_lock"]
            async with lock:
                replacement = asyncio.create_task(async_client.post(
                    "/integration/mentra/session/start", json=START
                ))
                await asyncio.sleep(0)
                assert len(lock._waiters) == 1
                old_append = asyncio.create_task(async_client.post(
                    "/integration/mentra/session/sitting-1/transcripts/append", json=batch
                ))
                await asyncio.sleep(0)
                assert len(lock._waiters) == 2
            resumed = await replacement
            assert resumed.status_code == 200
            assert resumed.json()["next_transcript_sequence"] == 1
            assert (await old_append).status_code == 404
            assert calls["state_writes"] == []
            assert conversation_sources.load_mentra_history_snapshot(
                storage_dir=tmp_path,
                user_id=START["user_id"],
                soul_id=START["soul_id"],
                conversation_id=conversation_sources.MENTRA_CONVERSATION_ID,
            ) == []
            batch["events"][0]["event_id"] = "sitting-2:1"
            appended = await async_client.post(
                "/integration/mentra/session/sitting-2/transcripts/append", json=batch
            )
            assert appended.status_code == 200
            assert appended.json()["ack_sequence"] == 1
            return first, resumed

    first, resumed = asyncio.run(asyncio.wait_for(replace_with_waiting_append(), timeout=5))
    assert first.status_code == resumed.status_code == 200
    assert first.json()["session_id"] == "sitting-1"
    assert resumed.json()["session_id"] == "sitting-2"
    assert first.json()["ephemeral_token"] != resumed.json()["ephemeral_token"]

    competing = {**START, "device_session_id": "phone-2"}
    assert client.post(
        "/integration/mentra/session/start", json=competing
    ).status_code == 409
    other_user = {**START, "user_id": "Another Fictional User"}
    assert client.post(
        "/integration/mentra/session/start", json=other_user
    ).status_code == 409
    assert calls["service"] == 2

    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    lease_before_refresh = mentra_routes._leases[START["soul_id"]]
    calls["narrative"] = "A changed fictional narrative."
    calls["anchor_rows"] = {"soul": "A changed fictional anchor.", "user": "A changed user anchor."}
    calls["category_index"] = "- Changed Category: New description."
    state_reads_before_refresh = len(calls["state"])
    refreshed = client.post(
        "/integration/mentra/session/sitting-2/token", json=scope
    )
    assert refreshed.status_code == 200
    assert refreshed.json() == {"ephemeral_token": "ephemeral-3"}
    assert mentra_routes._leases[START["soul_id"]] == lease_before_refresh
    assert len(calls["state"]) == state_reads_before_refresh
    assert "changed fictional" not in calls["token"][-1]["system_instruction"]
    assert "Changed Category" not in calls["token"][-1]["system_instruction"]
    assert calls["token"][-1] == {
        "api_key": "permanent-secret",
        "model": "gemini-2.5-flash-native-audio-preview-12-2025",
        "voice": "Kore",
        "system_instruction": calls["token"][-2]["system_instruction"],
        "mode": "continuous",
    }
    assert calls["route_telemetry"][-1][0][0] == "mentra.token"
    assert calls["route_telemetry"][-1][1]["ok"] is True
    assert calls["route_telemetry"][-1][1]["info"]["wallMs"] >= 0
    assert "permanent-secret" not in json.dumps(calls["route_telemetry"][-1])
    assert calls["token"][-1]["system_instruction"] not in json.dumps(
        calls["route_telemetry"][-1]
    )
    assert client.post(
        "/integration/mentra/session/wrong/heartbeat", json=scope
    ).status_code == 404
    wrong_user_scope = {**scope, "user_id": "Another Fictional User"}
    assert client.post(
        "/integration/mentra/session/sitting-2/heartbeat", json=wrong_user_scope
    ).status_code == 404
    assert client.post(
        "/integration/mentra/session/sitting-2/heartbeat", json=scope
    ).status_code == 200
    stale_append = client.post(
        "/integration/mentra/session/sitting-1/transcripts/append",
        json={
            **scope,
            "events": [{
                "event_id": "sitting-1:1",
                "sequence": 1,
                "event_kind": "transcript",
                "role": "user",
                "content": "A stale fictional line.",
                "status": "complete",
            }],
        },
    )
    assert stale_append.status_code == 404
    assert client.post(
        "/integration/mentra/session/wrong/end", json=scope
    ).status_code == 409
    assert client.post(
        "/integration/mentra/session/sitting-1/end", json=scope
    ).status_code == 409
    assert client.post(
        "/integration/mentra/session/sitting-2/end", json=scope
    ).status_code == 200
    assert client.post(
        "/integration/mentra/session/sitting-2/end", json=scope
    ).status_code == 200


def test_session_end_does_not_wait_for_soul_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    calls["reject_soul_lock"] = True

    response = client.post(
        f"/integration/mentra/session/{sitting_id}/end",
        json={"user_id": START["user_id"], "soul_id": START["soul_id"]},
    )

    assert response.status_code == 200


def test_token_mint_failure_preserves_current_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(
        monkeypatch,
        tmp_path,
        token_results=["initial-token", RuntimeError("upstream")],
    )
    started = client.post("/integration/mentra/session/start", json=START)
    lease_before = mentra_routes._leases[START["soul_id"]]
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}

    failed = client.post(
        f"/integration/mentra/session/{started.json()['session_id']}/token",
        json=scope,
    )

    assert failed.status_code == 502
    assert failed.json() == {"detail": "Gemini session token request failed"}
    assert mentra_routes._leases[START["soul_id"]] == lease_before
    assert calls["route_telemetry"][-1][0][0] == "mentra.token"
    assert calls["route_telemetry"][-1][1]["error"] == "provider_mint_failed"


def test_stop_during_token_mint_returns_no_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, _calls, _ = _session_app(monkeypatch, tmp_path)
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    started = Event()
    release = Event()

    async def slow_mint(**_kwargs: str) -> str:
        started.set()
        await asyncio.to_thread(release.wait, 5)
        return "must-not-return"

    monkeypatch.setattr(mentra_routes, "_mint_gemini_token", slow_mint)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            client.post,
            f"/integration/mentra/session/{sitting_id}/token",
            json=scope,
        )
        assert started.wait(2)
        assert client.post(
            f"/integration/mentra/session/{sitting_id}/end",
            json=scope,
        ).status_code == 200
        release.set()
        refreshed = future.result(timeout=5)

    assert refreshed.status_code == 404
    assert refreshed.json() == {"detail": "Mentra session not found"}


def test_failed_replacement_start_preserves_healthy_sitting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(
        monkeypatch,
        tmp_path,
        token_results=[RuntimeError("upstream"), "token", RuntimeError("upstream")],
    )

    failed = client.post("/integration/mentra/session/start", json=START)
    assert failed.status_code == 502
    failed_call = calls["route_telemetry"][-1]
    assert failed_call[1]["ok"] is False
    assert set(failed_call[1]["info"]) == {
        "setupMs",
        "anchorsMs",
        "stateMs",
        "contextMs",
        "tokenMs",
        "totalMs",
    }
    assert failed_call[1]["error"].startswith("HTTPException:")
    competing = {**START, "device_session_id": "phone-2"}
    healthy = client.post(
        "/integration/mentra/session/start", json=competing
    )
    assert healthy.status_code == 200
    healthy_sitting = healthy.json()["session_id"]
    lease_before = mentra_routes._leases[START["soul_id"]]
    failed_renewal = client.post(
        "/integration/mentra/session/start", json=competing
    )
    assert failed_renewal.status_code == 502
    assert mentra_routes._leases[START["soul_id"]] == lease_before
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    assert client.post(
        f"/integration/mentra/session/{healthy_sitting}/heartbeat",
        json=scope,
    ).status_code == 200


def test_replacement_start_rejects_context_that_changed_during_mint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, _, _ = _session_app(monkeypatch, tmp_path)
    mint_count = 0

    async def mint(**_kwargs: str) -> str:
        nonlocal mint_count
        mint_count += 1
        if mint_count == 2:
            mentra_routes.conversation_sources.persist_mentra_history_snapshot(
                storage_dir=tmp_path,
                user_id=START["user_id"],
                soul_id=START["soul_id"],
                conversation_id=conversation_sources.MENTRA_CONVERSATION_ID,
                history=[
                    {
                        "event_id": "sitting-1:1",
                        "sequence": 1,
                        "event_kind": "transcript",
                        "role": "user",
                        "content": "A fictional late line.",
                        "transcript_status": "complete",
                        "received_at": "2026-08-23T20:00:00.000Z",
                    }
                ],
            )
        return f"token-{mint_count}"

    monkeypatch.setattr(mentra_routes, "_mint_gemini_token", mint)
    healthy = client.post("/integration/mentra/session/start", json=START)
    assert healthy.status_code == 200
    healthy_sitting = healthy.json()["session_id"]

    stale = client.post("/integration/mentra/session/start", json=START)
    assert stale.status_code == 409
    assert stale.json()["detail"] == {"code": "mentra_history_changed"}
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    assert client.post(
        f"/integration/mentra/session/{healthy_sitting}/heartbeat",
        json=scope,
    ).status_code == 200

    refreshed = client.post("/integration/mentra/session/start", json=START)
    assert refreshed.status_code == 200
    assert refreshed.json()["next_transcript_sequence"] == 2


def test_failed_start_claim_cleanup_cannot_remove_newer_claim() -> None:
    mentra_routes._start_claims["Codexia"] = ("newer-sitting", "phone-1")

    asyncio.run(mentra_routes._release_start_claim_if_owned("Codexia", "older-sitting"))

    assert mentra_routes._start_claims["Codexia"] == ("newer-sitting", "phone-1")


def test_second_start_is_rejected_while_same_soul_start_is_in_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    mentra_routes._start_claims[START["soul_id"]] = ("in-progress", "phone-1")

    for device_session_id in ("phone-1", "phone-2"):
        response = client.post(
            "/integration/mentra/session/start",
            json={**START, "device_session_id": device_session_id},
        )
        assert response.status_code == 409
    assert calls["service"] == 0
    assert calls["token"] == []


def test_cancelled_start_always_releases_in_progress_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(
        monkeypatch,
        tmp_path,
        token_results=[asyncio.CancelledError()],
    )

    with pytest.raises((asyncio.CancelledError, FutureCancelledError)):
        client.post("/integration/mentra/session/start", json=START)

    assert START["soul_id"] not in mentra_routes._start_claims


def test_recall_is_sitting_scoped_read_only_compact_and_unlocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    endpoint = f"/integration/mentra/session/{sitting_id}/recall"
    request = {
        "user_id": START["user_id"],
        "soul_id": START["soul_id"],
        "query": "remember the fictional lighthouse",
    }

    invalid = client.post(
        endpoint,
        json={**request, "query": "rejected text", "extra": True},
    )
    assert invalid.status_code == 422
    assert invalid.json()["detail"][0]["type"] == "extra_forbidden"
    assert client.post(endpoint.replace(sitting_id, "wrong"), json=request).status_code == 404
    response = client.post(endpoint, json=request)

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "context": (
            "Dossiers:\n[Fictional Places] A remembered coast.\n\n"
            "Memories:\nKey: [profile] what's said or declared\n"
            "- [profile] The lighthouse keeper preferred cedar tea."
        ),
        "retrieve_ms": 123,
    }
    conversation_id, payload = calls["recalls"][0]
    assert conversation_id == conversation_sources.MENTRA_CONVERSATION_ID
    assert payload["user"] == {
        "user_id": START["user_id"],
        "soul_id": START["soul_id"],
        "conversation_id": conversation_sources.MENTRA_CONVERSATION_ID,
    }
    assert payload["history"][0]["content"] == "A fictional current-chat line."
    assert payload["force_retrieve"] is True
    assert payload["_read_only_retrieve"] is True
    assert payload["mental_health_addon"] is False
    assert "memory-secret-id" not in response.text
    assert calls["lease_locked_during_retrieve"] is False
    assert calls["soul_locked_during_retrieve"] is False
    telemetry_args, telemetry_kwargs = calls["route_telemetry"][-1]
    assert telemetry_args[0] == "mentra.recall"
    assert telemetry_kwargs == {
        "ok": True,
        "info": {
            "wallMs": pytest.approx(0, abs=1000),
            "retrieveMs": 123,
            "resultCounts": {"categories": 1, "items": 1},
        },
    }
    assert "remember the fictional lighthouse" not in json.dumps(calls["route_telemetry"])


def test_recall_failure_is_generic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(
        monkeypatch,
        tmp_path,
        retrieve_result=RuntimeError("private fictional query leaked"),
    )
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    response = client.post(
        f"/integration/mentra/session/{sitting_id}/recall",
        json={
            "user_id": START["user_id"],
            "soul_id": START["soul_id"],
            "query": "private fictional query",
        },
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Mentra memory recall failed"}
    assert "private fictional" not in response.text
    assert calls["route_telemetry"][-1][1] == {
        "ok": False,
        "error": "RuntimeError: private fictional query leaked",
    }


def test_recall_timeout_is_temporary_and_config_failure_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    timeout_client, _, _ = _session_app(
        monkeypatch,
        tmp_path,
        retrieve_result=HTTPException(status_code=504, detail="private timeout"),
    )
    timeout_sitting = timeout_client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    request = {"user_id": START["user_id"], "soul_id": START["soul_id"], "query": "private"}
    timeout = timeout_client.post(
        f"/integration/mentra/session/{timeout_sitting}/recall",
        json=request,
    )
    assert timeout.status_code == 502
    assert timeout.json() == {"detail": "Mentra memory recall failed"}

    mentra_routes._leases.clear()
    config_client, _, _ = _session_app(
        monkeypatch,
        tmp_path,
        retrieve_result=HTTPException(status_code=503, detail="private config"),
    )
    config_sitting = config_client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    config = config_client.post(
        f"/integration/mentra/session/{config_sitting}/recall",
        json=request,
    )
    assert config.status_code == 503
    assert config.json() == {"detail": "Mentra memory recall failed"}


def test_recall_enforces_server_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def stalled_retrieve(
        _conversation_id: str, _payload: dict[str, Any]
    ) -> dict[str, Any]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(mentra_routes, "_RECALL_TIMEOUT_SECONDS", 0.01)
    client, calls, _ = _session_app(
        monkeypatch,
        tmp_path,
        retrieve_hook=stalled_retrieve,
    )
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    response = client.post(
        f"/integration/mentra/session/{sitting_id}/recall",
        json={
            "user_id": START["user_id"],
            "soul_id": START["soul_id"],
            "query": "private fictional query",
        },
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Mentra memory recall failed"}
    assert calls["route_telemetry"][-1][1] == {
        "ok": False,
        "error": "TimeoutError",
    }


def test_slow_recall_does_not_hold_lease_or_soul_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    started = Event()
    release = Event()

    async def slow_retrieve(_conversation_id: str, _payload: dict[str, Any]) -> dict[str, Any]:
        started.set()
        await asyncio.to_thread(release.wait, 5)
        return {"ok": True, "retrieve_ms": 1, "result": {}}

    client, _, _ = _session_app(monkeypatch, tmp_path, retrieve_hook=slow_retrieve)
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    recall_path = f"/integration/mentra/session/{sitting_id}/recall"

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            client.post,
            recall_path,
            json={**scope, "query": "slow fictional recall"},
        )
        assert started.wait(2)
        assert client.post(
            f"/integration/mentra/session/{sitting_id}/heartbeat",
            json=scope,
        ).status_code == 200
        assert client.post(
            f"/integration/mentra/session/{sitting_id}/transcripts/append",
            json={
                **scope,
                "events": [
                    {
                        "event_id": f"{sitting_id}:1",
                        "sequence": 1,
                        "event_kind": "transcript",
                        "role": "user",
                        "content": "A fictional line while recall runs.",
                        "status": "complete",
                    }
                ],
            },
        ).status_code == 200
        release.set()
        assert future.result(timeout=2).status_code == 200


def _recovered_transcripts(count: int = 1) -> dict[str, Any]:
    return {"sitting_id": "previous-sitting", "events": [
        {"event_id": f"previous-sitting:{sequence}", "sequence": sequence,
         "event_kind": "transcript", "role": "user", "status": "interrupted",
         "content": f"Recovered fictional line {sequence}."}
        for sequence in range(1, count + 1)
    ]}


@pytest.mark.parametrize("vision", [False, True])
def test_start_recovery_publishes_complete_tail_before_context_and_replays_exactly(monkeypatch, tmp_path, vision):
    from memu.utils.conversation import format_grouped_chat_history

    client, calls, _ = _session_app(monkeypatch, tmp_path, token_results=["first", "replay", "prefix", "normal"])
    recovery = _recovered_transcripts(20)

    def context(**scope):
        assert calls["soul_lock"].locked()
        rows = conversation_sources.load_mentra_tail(
            storage_dir=tmp_path, since_cursor=-1, recent_fallback_messages=0, **scope,
        )
        return format_grouped_chat_history(rows, soul_name=scope["soul_id"], mark_current_chat=False)

    calls["cross_loader"] = context
    first = client.post("/integration/mentra/session/start", json={**START, "vision": vision, "recovery": recovery})
    assert first.status_code == 200
    assert first.json()["recovery"] == {"sitting_id": "previous-sitting", "ack_sequence": 20}
    assert first.json()["next_transcript_sequence"] == 21
    for event in recovery["events"]:
        assert event["content"] in calls["token"][0]["system_instruction"]
    scope = {"storage_dir": tmp_path, "user_id": START["user_id"],
             "soul_id": START["soul_id"], "conversation_id": "mentra:iris"}
    saved = conversation_sources.load_mentra_history_snapshot(**scope)
    assert [row["event_id"] for row in saved] == [event["event_id"] for event in recovery["events"]]
    assert len(calls["memorize_checks"]) == 1
    replay = client.post("/integration/mentra/session/start", json={**START, "recovery": recovery})
    assert replay.status_code == 200
    assert replay.json()["recovery"] == first.json()["recovery"]
    assert conversation_sources.load_mentra_history_snapshot(**scope) == saved
    assert len(calls["memorize_checks"]) == 2
    prefix = client.post("/integration/mentra/session/start", json={**START,
        "recovery": {**recovery, "events": recovery["events"][:1]}})
    assert prefix.json()["recovery"] == {"sitting_id": "previous-sitting", "ack_sequence": 1}
    assert prefix.json()["next_transcript_sequence"] == 21
    calls["cross_loader"] = None
    normal = client.post("/integration/mentra/session/start", json=START)
    assert normal.status_code == 200 and normal.json()["recovery"] is None
    assert len(calls["memorize_checks"]) == 3


def test_start_recovery_mint_failure_keeps_saved_rows_and_old_lease(monkeypatch, tmp_path):
    client, calls, _ = _session_app(monkeypatch, tmp_path, token_results=["first", RuntimeError("fictional mint"), "retry"])
    client.post("/integration/mentra/session/start", json=START).raise_for_status()
    lease = mentra_routes._leases[START["soul_id"]]
    recovery = _recovered_transcripts()
    failed = client.post("/integration/mentra/session/start", json={**START, "recovery": recovery})
    assert failed.status_code == 502
    assert mentra_routes._leases[START["soul_id"]] is lease
    assert not mentra_routes._start_claims
    saved = conversation_sources.load_mentra_history_snapshot(storage_dir=tmp_path,
        user_id=START["user_id"], soul_id=START["soul_id"], conversation_id="mentra:iris")
    assert saved[0]["event_id"] == recovery["events"][0]["event_id"]
    retry = client.post("/integration/mentra/session/start", json={**START, "recovery": recovery})
    assert retry.status_code == 200
    assert retry.json()["recovery"] == {"sitting_id": "previous-sitting", "ack_sequence": 1}
    assert len(calls["memorize_checks"]) == 2


@pytest.mark.parametrize("failure", ["receipt", "state"])
def test_start_recovery_replay_repairs_partial_publication(monkeypatch, tmp_path, failure):
    client, calls, _ = _session_app(monkeypatch, tmp_path, raise_server_exceptions=False,
        state_results=[RuntimeError("fictional state write"), None] if failure == "state" else [])
    writer = conversation_sources._write_chat_snapshot
    refused = False

    def write(path, data):
        nonlocal refused
        if failure == "receipt" and path.parent.name == ".published" and not refused:
            refused = True
            raise OSError("fictional publication write")
        return writer(path, data)

    monkeypatch.setattr(conversation_sources, "_write_chat_snapshot", write)
    body = {**START, "recovery": _recovered_transcripts()}
    failed = client.post("/integration/mentra/session/start", json=body)
    assert failed.status_code == 500
    assert calls["token"] == [] and not mentra_routes._leases and not mentra_routes._start_claims
    repaired = client.post("/integration/mentra/session/start", json=body)
    assert repaired.status_code == 200
    assert repaired.json()["next_transcript_sequence"] == 2
    assert repaired.json()["recovery"]["ack_sequence"] == 1
    assert len(calls["memorize_checks"]) == 1
    assert len(list((tmp_path / "transcripts" / ".published").glob("*.json"))) == 1


@pytest.mark.parametrize("conflict", ["content", "sitting", "hole", "prefix"])
def test_start_recovery_conflict_does_not_replace_rows_or_publish_a_prefix(monkeypatch, tmp_path, conflict):
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    recovery = _recovered_transcripts()
    client.post("/integration/mentra/session/start", json={**START, "recovery": recovery}).raise_for_status()
    lease = mentra_routes._leases[START["soul_id"]]
    scope = {"storage_dir": tmp_path, "user_id": START["user_id"], "soul_id": START["soul_id"], "conversation_id": "mentra:iris"}
    before = conversation_sources.load_mentra_history_snapshot(**scope)
    if conflict == "content":
        recovery["events"][0]["content"] = "Conflicting fictional words."
    elif conflict == "sitting":
        recovery["sitting_id"] = "different-sitting"
    elif conflict == "hole":
        recovery["events"] = _recovered_transcripts(3)["events"][2:]
    else:
        recovery["events"] = _recovered_transcripts(3)["events"][1:]
        recovery["events"][-1]["event_id"] = "different-sitting:3"
    failed = client.post("/integration/mentra/session/start", json={**START, "recovery": recovery})
    assert failed.status_code == 409
    assert failed.json()["detail"] == {"code": "mentra_recovery_conflict", "expected_sequence": 3 if conflict == "prefix" else 2}
    assert conversation_sources.load_mentra_history_snapshot(**scope) == before
    assert mentra_routes._leases[START["soul_id"]] is lease
    assert len(calls["token"]) == 1 and not mentra_routes._start_claims


@pytest.mark.parametrize("invalid", ["event", "protocol", "json", "empty", "noncontiguous"])
def test_start_recovery_validation_is_redacted_before_side_effects(monkeypatch, tmp_path, invalid):
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    body = {**START, "recovery": _recovered_transcripts()}
    private = "PRIVATE-FICTIONAL-RECOVERY"
    body["recovery"]["events"][0]["content"] = private
    if invalid == "event":
        body["recovery"]["events"][0]["unexpected"] = True
    elif invalid == "protocol":
        body.pop("photo_work_protocol")
    elif invalid == "empty":
        body["recovery"]["events"] = []
    elif invalid == "noncontiguous":
        body["recovery"]["events"] += _recovered_transcripts(3)["events"][2:]
    response = (client.post("/integration/mentra/session/start", content=json.dumps(body)[:-1],
                           headers={"Content-Type": "application/json"}) if invalid == "json"
                else client.post("/integration/mentra/session/start", json=body))
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid session start"}
    assert private not in response.text
    assert calls["service"] == 0 and calls["token"] == [] and calls["state_writes"] == []
    assert not mentra_routes._leases and not mentra_routes._start_claims


@pytest.mark.parametrize("gap", [False, True])
def test_start_recovery_uses_ordinary_memorize_eligibility(monkeypatch, tmp_path, gap):
    scheduled = []
    client, calls, _ = _session_app(monkeypatch, tmp_path,
        prepare_auto_memorize=lambda *_a, **_k: (9000, {"conversation": []}),
        schedule_auto_memorize=lambda *args: scheduled.append(args) or "launched")
    recovery = _recovered_transcripts()
    if gap:
        recovery["events"][0].update(event_kind="transcript_gap", content="Transcript unavailable.", status=None)
    response = client.post("/integration/mentra/session/start", json={**START, "recovery": recovery})
    assert response.status_code == 200
    assert len(calls["memorize_checks"]) == len(scheduled) == (0 if gap else 1)


@pytest.mark.parametrize("rejection", ["pause", "device", "config"])
def test_start_recovery_does_not_publish_before_start_admission(monkeypatch, tmp_path, rejection):
    client, calls, config = _session_app(monkeypatch, tmp_path)
    body = {**START, "recovery": _recovered_transcripts()}
    if rejection == "pause":
        calls["pause_reason"] = "fictional recovery pending"
    elif rejection == "device":
        client.post("/integration/mentra/session/start", json=START).raise_for_status()
        body["device_session_id"] = "another-phone"
    else:
        config["mentra"]["voice"] = ""
    before = len(calls["token"])
    response = client.post("/integration/mentra/session/start", json=body)
    assert response.status_code == (503 if rejection == "config" else 409)
    assert calls["state_writes"] == [] and calls["memorize_checks"] == []
    assert len(calls["token"]) == before and not mentra_routes._start_claims
    assert conversation_sources.load_mentra_history_snapshot(storage_dir=tmp_path,
        user_id=START["user_id"], soul_id=START["soul_id"], conversation_id="mentra:iris") == []


@pytest.mark.parametrize("missing", [True, False])
def test_start_recovery_preserves_published_source_errors(monkeypatch, tmp_path, missing):
    client, calls, _ = _session_app(monkeypatch, tmp_path, raise_server_exceptions=False)
    body = {**START, "recovery": _recovered_transcripts()}
    client.post("/integration/mentra/session/start", json=body).raise_for_status()
    path = conversation_sources._chat_snapshot_path(storage_dir=tmp_path, user_id=START["user_id"],
        soul_id=START["soul_id"], conversation_id="mentra:iris", source_label="mentra")
    if missing:
        path.unlink()
    else:
        path.write_text("fictional corruption")
    response = client.post("/integration/mentra/session/start", json=body)
    assert response.status_code == 500
    assert len(calls["token"]) == 1 and len(calls["state_writes"]) == 1
    assert not path.exists() if missing else path.read_text() == "fictional corruption"


def test_start_recovery_is_scoped_to_its_requested_soul(monkeypatch, tmp_path):
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    response = client.post("/integration/mentra/session/start", json={**START, "soul_id": "Quill",
        "recovery": _recovered_transcripts()})
    assert response.status_code == 200
    assert calls["service_scope"] == {"user_id": START["user_id"], "soul_id": "Quill"}
    for soul, length in [("Quill", 1), (START["soul_id"], 0)]:
        rows = conversation_sources.load_mentra_history_snapshot(storage_dir=tmp_path,
            user_id=START["user_id"], soul_id=soul, conversation_id="mentra:iris")
        assert len(rows) == length


def test_start_recovery_rechecks_pause_before_context_and_mint(monkeypatch, tmp_path):
    def schedule(*_args):
        calls["pause_reason"] = "fictional concurrent recovery"
        return "launched"

    client, calls, _ = _session_app(monkeypatch, tmp_path,
        prepare_auto_memorize=lambda *_a, **_k: (9000, {"conversation": []}), schedule_auto_memorize=schedule)
    response = client.post("/integration/mentra/session/start", json={**START, "recovery": _recovered_transcripts()})
    assert response.status_code == 409 and response.json()["detail"]["code"] == "soul_paused"
    assert calls["cross"] == [] and calls["token"] == []
    assert len(calls["state_writes"]) == 1
    assert not mentra_routes._leases and not mentra_routes._start_claims


def test_start_recovery_rejects_context_changed_by_append_during_mint(monkeypatch, tmp_path):
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    old = client.post("/integration/mentra/session/start", json=START).json()["session_id"]
    old_lease = mentra_routes._leases[START["soul_id"]]
    changed = False

    async def mint(**_kwargs):
        nonlocal changed
        assert not calls["soul_lock"].locked()
        if not changed:
            changed = True
            async with AsyncClient(transport=ASGITransport(app=client.app), base_url="http://test") as http:
                response = await http.post(f"/integration/mentra/session/{old}/transcripts/append", json={
                    "user_id": START["user_id"], "soul_id": START["soul_id"],
                    "events": [{**_recovered_transcripts(2)["events"][-1], "event_id": f"{old}:2"}],
                })
                assert response.status_code == 200
        return "fictional-token"

    monkeypatch.setattr(mentra_routes, "_mint_gemini_token", mint)
    body = {**START, "recovery": _recovered_transcripts()}
    stale = client.post("/integration/mentra/session/start", json=body)
    assert stale.status_code == 409 and stale.json()["detail"] == {"code": "mentra_history_changed"}
    assert mentra_routes._leases[START["soul_id"]] is old_lease
    assert not mentra_routes._start_claims
    retry = client.post("/integration/mentra/session/start", json=body)
    assert retry.status_code == 200 and retry.json()["next_transcript_sequence"] == 3
    assert retry.json()["recovery"] == {"sitting_id": "previous-sitting", "ack_sequence": 1}


def test_start_recovery_refuses_a_worker_failure_during_mint(monkeypatch, tmp_path):
    paused = asyncio.Event()

    async def failed_work():
        assert len(calls["cross"]) == 2
        calls["pause_reason"] = "Fictional background Memorize failed."
        paused.set()

    def schedule(*_args):
        if not paused.is_set():
            asyncio.create_task(failed_work())
        return "launched"

    client, calls, _ = _session_app(monkeypatch, tmp_path,
        prepare_auto_memorize=lambda *_a, **_k: (9000, {"conversation": []}), schedule_auto_memorize=schedule)
    client.post("/integration/mentra/session/start", json=START).raise_for_status()
    old_lease = mentra_routes._leases[START["soul_id"]]

    async def mint(**_kwargs):
        await paused.wait()
        return "fictional-token"

    monkeypatch.setattr(mentra_routes, "_mint_gemini_token", mint)
    response = client.post("/integration/mentra/session/start", json={**START, "recovery": _recovered_transcripts()})
    assert response.status_code == 409 and response.json()["detail"]["code"] == "soul_paused"
    assert mentra_routes._leases[START["soul_id"]] is old_lease
    assert not mentra_routes._start_claims
    scope = {"storage_dir": tmp_path, "user_id": START["user_id"], "soul_id": START["soul_id"], "conversation_id": "mentra:iris"}
    saved = conversation_sources.load_mentra_history_snapshot(**scope)
    assert saved[0]["event_id"] == "previous-sitting:1"
    calls["pause_reason"] = None
    retry = client.post("/integration/mentra/session/start", json={**START, "recovery": _recovered_transcripts()})
    assert retry.status_code == 200
    assert retry.json()["recovery"] == {"sitting_id": "previous-sitting", "ack_sequence": 1}
    assert conversation_sources.load_mentra_history_snapshot(**scope) == saved


def test_transcript_append_is_contiguous_idempotent_and_initializes_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(
        monkeypatch, tmp_path, token_results=["token-1", "token-2", "token-3", "token-4"],
    )
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    endpoint = f"/integration/mentra/session/{sitting_id}/transcripts/append"
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    events = [
        {
            "event_id": f"{sitting_id}:1",
            "sequence": 1,
            "event_kind": "transcript",
            "role": "user",
            "content": "A fictional question.",
            "status": "complete",
        },
        {
            "event_id": f"{sitting_id}:2",
            "sequence": 2,
            "event_kind": "transcript",
            "role": "assistant",
            "content": "A fictional answer.",
            "status": "complete",
        },
    ]

    first = client.post(endpoint, json={**scope, "events": events})
    replay = client.post(endpoint, json={**scope, "events": events})
    suffix = client.post(
        endpoint,
        json={
            **scope,
            "events": [
                events[1],
                {
                    "event_id": f"{sitting_id}:3",
                    "sequence": 3,
                    "event_kind": "sitting_summary",
                    "role": "assistant",
                    "content": "I noticed a fictional shift.",
                },
            ],
        },
    )
    restarted = client.post("/integration/mentra/session/start", json=START)

    assert first.json() == {
        "ok": True,
        "conversation_id": conversation_sources.MENTRA_CONVERSATION_ID,
        "ack_sequence": 2,
        "accepted": 2,
        "duplicates": 0,
        "memorize_check_queued": False,
    }
    assert replay.json()["accepted"] == 0
    assert replay.json()["duplicates"] == 2
    assert suffix.json()["accepted"] == 1
    assert suffix.json()["duplicates"] == 1
    assert restarted.json()["next_transcript_sequence"] == 4
    history = mentra_routes.conversation_sources.load_mentra_history_snapshot(
        storage_dir=tmp_path,
        user_id=START["user_id"],
        soul_id=START["soul_id"],
        conversation_id=conversation_sources.MENTRA_CONVERSATION_ID,
    )
    assert [row["sequence"] for row in history] == [1, 2, 3]
    assert all(row["received_at"].endswith("Z") for row in history)
    assert len(calls["state_writes"]) == 3
    assert calls["state_writes"][0][1]["updates"] == {"memorize_chat": True}
    original_history = history

    session_id = restarted.json()["session_id"]
    client.post(
        f"/integration/mentra/session/{session_id}/end", json=scope,
    ).raise_for_status()
    second_phone = client.post(
        "/integration/mentra/session/start",
        json={**START, "device_session_id": "phone-2"},
    )
    second_phone.raise_for_status()
    assert second_phone.json()["conversation_id"] == conversation_sources.MENTRA_CONVERSATION_ID
    assert second_phone.json()["next_transcript_sequence"] == 4
    second_id = second_phone.json()["session_id"]
    event = {**events[1], "event_id": f"{second_id}:4", "sequence": 4}
    appended = client.post(
        f"/integration/mentra/session/{second_id}/transcripts/append",
        json={**scope, "events": [event]},
    )
    appended.raise_for_status()
    assert appended.json()["ack_sequence"] == 4
    assert calls["state_writes"][-1][0][0] == conversation_sources.MENTRA_CONVERSATION_ID
    assert calls["state"][-1] == (conversation_sources.MENTRA_CONVERSATION_ID, scope)
    assert calls["cross"][-1] == {**scope, "conversation_id": conversation_sources.MENTRA_CONVERSATION_ID}
    client.post(
        f"/integration/mentra/session/{second_id}/recall",
        json={**scope, "query": "A fictional shared memory."},
    ).raise_for_status()
    assert calls["recalls"][-1][0] == conversation_sources.MENTRA_CONVERSATION_ID
    assert calls["current"]["conversation_id"] == conversation_sources.MENTRA_CONVERSATION_ID
    history = conversation_sources.load_mentra_history_snapshot(
        storage_dir=tmp_path, **scope,
        conversation_id=conversation_sources.MENTRA_CONVERSATION_ID,
    )
    assert [row["sequence"] for row in history] == [1, 2, 3, 4]
    assert history[:3] == original_history

    other_scope = {**scope, "soul_id": "TestSoul"}
    other = client.post(
        "/integration/mentra/session/start", json={**START, **other_scope},
    )
    other.raise_for_status()
    assert other.json()["conversation_id"] == conversation_sources.MENTRA_CONVERSATION_ID
    assert other.json()["next_transcript_sequence"] == 1
    other_id = other.json()["session_id"]
    assert client.post(
        f"/integration/mentra/session/{second_id}/heartbeat", json=other_scope,
    ).status_code == 404
    client.post(
        f"/integration/mentra/session/{other_id}/transcripts/append",
        json={**other_scope, "events": [{**events[0], "event_id": f"{other_id}:1"}]},
    ).raise_for_status()
    other_history = conversation_sources.load_mentra_history_snapshot(
        storage_dir=tmp_path, **other_scope,
        conversation_id=conversation_sources.MENTRA_CONVERSATION_ID,
    )
    assert [row["event_id"] for row in other_history] == [f"{other_id}:1"]
    assert conversation_sources.load_mentra_history_snapshot(
        storage_dir=tmp_path, **scope,
        conversation_id=conversation_sources.MENTRA_CONVERSATION_ID,
    ) == history
    assert conversation_sources.load_mentra_history_snapshot(
        storage_dir=tmp_path, **{**scope, "user_id": "Another Fictional User"},
        conversation_id=conversation_sources.MENTRA_CONVERSATION_ID,
    ) == []
    snapshots = [json.loads(path.read_text())
                 for path in (tmp_path / "transcripts").rglob("latest_history.json")]
    assert len(snapshots) == 2
    assert {snapshot["conversation_id"] for snapshot in snapshots} == {
        conversation_sources.MENTRA_CONVERSATION_ID,
    }


def test_transcript_publication_failure_replays_without_duplicating_history(monkeypatch, tmp_path):
    client, calls, _ = _session_app(monkeypatch, tmp_path, raise_server_exceptions=False)
    sitting = client.post("/integration/mentra/session/start", json=START).json()["session_id"]
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    endpoint = f"/integration/mentra/session/{sitting}/transcripts/append"
    event = {"event_id": f"{sitting}:1", "sequence": 1, "event_kind": "transcript",
             "role": "user", "content": "A fictional saved line.", "status": "complete"}
    write_snapshot = conversation_sources._write_chat_snapshot

    def fail_receipt(path, payload):
        if path.parent.name == ".published":
            raise OSError("Fictional receipt write failure")
        write_snapshot(path, payload)

    with monkeypatch.context() as patch:
        patch.setattr(conversation_sources, "_write_chat_snapshot", fail_receipt)
        assert client.post(endpoint, json={**scope, "events": [event]}).status_code == 500
        # A duplicate replay also has to finish publication before acknowledging.
        assert client.post(endpoint, json={**scope, "events": [event]}).status_code == 500
    assert calls["state_writes"] == [] and calls["memorize_checks"] == []
    path = next((tmp_path / "transcripts").rglob("latest_history.json"))
    saved = path.read_bytes()
    replay = client.post(endpoint, json={**scope, "events": [event]})
    assert replay.status_code == 200
    assert replay.json()["accepted"] == 0 and replay.json()["duplicates"] == 1
    assert replay.json()["ack_sequence"] == 1
    assert len(json.loads(path.read_bytes())["history"]) == 1
    assert json.loads(path.read_bytes())["history"] == json.loads(saved)["history"]
    assert len(calls["state_writes"]) == 1
    receipt = next((tmp_path / "transcripts" / ".published").glob("*.json"))
    assert json.loads(receipt.read_text()) == {**scope, "conversation_id": "mentra:iris", "source_label": "mentra"}

    path.unlink()
    status = client.get("/integration/mentra/status", params=scope).json()
    assert status["state"] == "degraded" and status["detail"] == "Transcript status unavailable"
    assert client.post(endpoint, json={**scope, "events": [event]}).status_code == 500
    assert client.post("/integration/mentra/session/start", json=START).status_code == 500
    assert not path.exists() and len(calls["state_writes"]) == 1
    assert len(calls["token"]) == 1


def test_transcript_append_rechecks_due_auto_memorize_on_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scheduled: list[tuple[Any, ...]] = []
    payload = {"conversation": [{"content": "fictional", "memorize_chat": True}]}
    eligibility = iter([(9000, payload), (0, None)])
    client, calls, _ = _session_app(
        monkeypatch,
        tmp_path,
        prepare_auto_memorize=lambda *_a, **_k: next(eligibility),
        schedule_auto_memorize=lambda *args: scheduled.append(args) or "launched",
    )
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    endpoint = f"/integration/mentra/session/{sitting_id}/transcripts/append"
    body = {
        "user_id": START["user_id"],
        "soul_id": START["soul_id"],
        "events": [
            {
                "event_id": f"{sitting_id}:1",
                "sequence": 1,
                "event_kind": "transcript",
                "role": "user",
                "content": "A fictional partial line.",
                "status": "interrupted",
            },
            {
                "event_id": f"{sitting_id}:2",
                "sequence": 2,
                "event_kind": "sitting_summary",
                "role": "assistant",
                "content": "I noticed a fictional shift.",
            },
        ],
    }

    first = client.post(endpoint, json=body)
    replay = client.post(endpoint, json=body)

    assert first.json()["memorize_check_queued"] is True
    assert replay.json()["memorize_check_queued"] is False
    assert len(calls["memorize_checks"]) == 2
    assert len(scheduled) == 1
    assert scheduled[0][0] is payload
    assert scheduled[0][1:4] == (conversation_sources.MENTRA_CONVERSATION_ID, START["user_id"], START["soul_id"])
    projected = calls["memorize_checks"][0][0][5]
    assert [row["source_conversation_index"] for row in projected] == [1, 2]
    assert all(isinstance(row["ts_ms"], int) for row in projected)
    assert [row["speaker"] for row in projected] == [START["user_id"], START["soul_id"]]
    assert projected[0]["content"].endswith("[interrupted]")
    assert projected[1]["content"].startswith("[End-of-sitting reflection]")


@pytest.mark.parametrize("start_recovery", [False, True])
def test_transcript_recovery_uses_real_scheduler_coalescing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, start_recovery: bool,
) -> None:
    from app import main

    async def run() -> None:
        release = asyncio.Event()
        runs, outcomes = [], []
        scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
        marker = main._memorize_lock_key(**scope)

        def schedule(payload, cid, uid, sid, safe, history):
            outcome = main._schedule_auto_memorize(
                payload, main._auto_memorize_scope(cid, uid, sid, safe, history),
            )
            outcomes.append(outcome)
            return outcome

        client, calls, _ = _session_app(
            monkeypatch, tmp_path,
            state_results=[RuntimeError("Fictional publication failure"), None],
            raise_server_exceptions=False,
            prepare_auto_memorize=lambda *_a, **_k: (9000, {"conversation": []}),
            schedule_auto_memorize=schedule,
        )
        monkeypatch.setattr(main, "_BACKGROUND_TASKS", set())
        monkeypatch.setattr(main, "_soul_import_state", lambda *_a: ("", None))
        monkeypatch.setattr(main, "_register_memorize_bounds", lambda *_a, **_k: None)
        monkeypatch.setattr(main, "_sillytavern_source_deleted", lambda *_a: False)

        async def worker(payload):
            if start_recovery and not runs:
                assert len(calls["cross"]) == 2
            runs.append(payload)
            await release.wait()

        monkeypatch.setattr(main, "_run_forced_memorize_from_turn", worker)
        async with AsyncClient(transport=ASGITransport(app=client.app, raise_app_exceptions=False), base_url="http://test") as http:
            sitting = (await http.post("/integration/mentra/session/start", json=START)).json()["session_id"]
            endpoint = f"/integration/mentra/session/{sitting}/transcripts/append"
            body = {**scope, "events": [{
                "event_id": f"{sitting}:1", "sequence": 1, "event_kind": "transcript",
                "role": "user", "content": "A recovered fictional conversation.", "status": "complete",
            }]}
            if start_recovery:
                endpoint = "/integration/mentra/session/start"
                body = {**START, "recovery": {"sitting_id": sitting, "events": body["events"]}}
            try:
                assert (await http.post(endpoint, json=body)).status_code == 500
                assert outcomes == []
                repaired = await http.post(endpoint, json=body)
                assert repaired.status_code == 200
                if start_recovery:
                    assert repaired.json()["recovery"] == {"sitting_id": sitting, "ack_sequence": 1}
                else:
                    assert repaired.json()["accepted"] == 0
                    assert repaired.json()["duplicates"] == 1
                    assert repaired.json()["memorize_check_queued"] is True
                await asyncio.sleep(0)
                replay = await http.post(endpoint, json=body)
                assert replay.status_code == 200
                if start_recovery:
                    assert replay.json()["recovery"] == repaired.json()["recovery"]
                else:
                    assert replay.json()["memorize_check_queued"] is True
                assert outcomes == ["launched", "coalesced"]
                assert len(calls["memorize_checks"]) == 2
                assert len(runs) == 1
                assert marker in main._FORCED_MEMORIZE_RECHECK
                assert len(main._BACKGROUND_TASKS) == 1
            finally:
                release.set()
                await asyncio.gather(*main._BACKGROUND_TASKS, return_exceptions=True)

    asyncio.run(asyncio.wait_for(run(), 5))


def test_sequence_and_duplicate_lookup_tolerate_compacted_prefix() -> None:
    history = [
        {
            "event_id": "fictional-sitting:40",
            "sequence": 40,
            "event_kind": "transcript",
            "role": "user",
            "content": "Retained fictional line.",
            "transcript_status": "complete",
            "received_at": "2026-08-23T10:00:00.000Z",
        },
        {
            "event_id": "fictional-sitting:42",
            "sequence": 42,
            "event_kind": "transcript",
            "role": "assistant",
            "content": "Retained fictional reply.",
            "transcript_status": "complete",
            "received_at": "2026-08-23T10:00:01.000Z",
        },
    ]
    events = mentra_routes.MentraTranscriptBatch.model_validate(
        {
            "user_id": "Fictional User",
            "soul_id": "Codexia",
            "events": [
                {
                    "event_id": "fictional-sitting:42",
                    "sequence": 42,
                    "event_kind": "transcript",
                    "role": "assistant",
                    "content": "Retained fictional reply.",
                    "status": "complete",
                },
                {
                    "event_id": "fictional-sitting:43",
                    "sequence": 43,
                    "event_kind": "transcript",
                    "role": "user",
                    "content": "New fictional line.",
                    "status": "complete",
                },
            ],
        }
    ).events

    merged, accepted, duplicates = mentra_routes._merge_transcript_events(
        history,
        events,
        sitting_id="fictional-sitting",
    )

    assert mentra_routes._next_transcript_sequence(history) == 43
    assert [row["sequence"] for row in merged] == [40, 42, 43]
    assert (accepted, duplicates) == (1, 1)
    assert merged[-1]["received_at"].endswith("Z")


@pytest.mark.parametrize("source", ["phone", "glasses", "bluetooth", "bluetoothClassic", None])
def test_transcript_source_survives_storage_tail_and_label_only_replays(monkeypatch, tmp_path, source):
    client, _, _ = _session_app(monkeypatch, tmp_path)
    sitting = client.post("/integration/mentra/session/start", json=START).json()["session_id"]
    endpoint = f"/integration/mentra/session/{sitting}/transcripts/append"
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    event = {"event_id": f"{sitting}:1", "sequence": 1, "event_kind": "transcript",
             "role": "user", "content": "A fictional view.", "status": "interrupted"}
    if source is not None:
        event["mic_source"] = source
    client.post(endpoint, json={**scope, "events": [event]}).raise_for_status()
    for replay_source in (None, "glasses", "phone"):
        replay = {**event, "mic_source": replay_source}
        response = client.post(endpoint, json={**scope, "events": [replay]})
        response.raise_for_status()
        assert response.json()["duplicates"] == 1
    stored = conversation_sources.load_mentra_history_snapshot(
        storage_dir=tmp_path, **scope, conversation_id="mentra:iris")
    tail = conversation_sources.load_mentra_tail(
        storage_dir=tmp_path, **scope, conversation_id="mentra:iris",
        since_cursor=0, recent_fallback_messages=8)
    assert stored[0].get("mic_source") == tail[0].get("mic_source") == source
    assert stored[0]["content"] == event["content"]
    assert tail[0]["chat_name"] == "Iris"
    assert tail[0]["content"] == "A fictional view. [interrupted]"
    assert tail[0]["event_id"] == event["event_id"]
    assert tail[0]["source_conversation_index"] == 1
    from app.services.turn_contract import build_conversations_block
    from app.services.payload import _normalize_conversation, _normalize_turn_history
    from memu.app.memorize_segments import _render_grouped_chat_messages

    label = "Smartglasses" if source == "glasses" else "Phone" if source else "Iris"
    for rendered in (
        build_conversations_block(history=tail, conversation_id="mentra:iris", soul_name=scope["soul_id"]),
        _render_grouped_chat_messages(_normalize_conversation(_normalize_conversation(tail)),
                                      soul_name=scope["soul_id"]),
        build_conversations_block(history=_normalize_turn_history(tail), conversation_id="mentra:iris",
                                  soul_name=scope["soul_id"]),
    ):
        assert "My Video Conversations:" in rendered
        assert f"[dm][{label}]" in rendered
        assert "A fictional view. [interrupted]" in rendered
    conflict = client.post(endpoint, json={**scope, "events": [{**event, "content": "Different speech."}]})
    assert conflict.status_code == 409


@pytest.mark.parametrize("patch", [
    {"mic_source": "unknown"},
    {"role": "assistant", "mic_source": "phone"},
    {"event_kind": "sitting_summary", "role": "assistant", "status": None, "mic_source": "phone"},
    {"event_kind": "image", "status": None, "content": "Shared a photo.",
     "media_ref": "mentra_media/sitting/photo.jpg", "mic_source": "phone"},
    {"event_kind": "transcript_gap", "status": None, "content": "Transcript unavailable.", "mic_source": "phone"},
])
def test_transcript_source_is_only_valid_on_spoken_user_events(patch):
    with pytest.raises(ValueError):
        mentra_routes.MentraTranscriptEvent.model_validate({
            "event_id": "fictional:1", "sequence": 1, "event_kind": "transcript",
            "role": "user", "content": "Fictional speech.", "status": "complete", **patch,
        })


def test_transcript_chunks_preserve_boundary_whitespace_and_unicode(monkeypatch, tmp_path):
    client, _, _ = _session_app(monkeypatch, tmp_path)
    sitting = client.post("/integration/mentra/session/start", json=START).json()["session_id"]
    pieces = ["first ", " " * 16_000, "\U0001f642" * 16_000, " last"]
    events = [{"event_id": f"{sitting}:{index + 1}", "sequence": index + 1,
               "event_kind": "transcript", "role": "user", "content": text, "status": "complete"}
              for index, text in enumerate(pieces)]
    response = client.post(f"/integration/mentra/session/{sitting}/transcripts/append", json={
        "user_id": START["user_id"], "soul_id": START["soul_id"], "events": events,
    })
    response.raise_for_status()
    stored = mentra_routes.conversation_sources.load_mentra_history_snapshot(storage_dir=tmp_path,
        user_id=START["user_id"], soul_id=START["soul_id"], conversation_id="mentra:iris")
    assert [row["content"] for row in stored] == pieces
    for invalid in ("", "\U0001f642" * 16_001):
        response = client.post(f"/integration/mentra/session/{sitting}/transcripts/append", json={
            "user_id": START["user_id"], "soul_id": START["soul_id"],
            "events": [{**events[0], "event_id": f"{sitting}:5", "sequence": 5, "content": invalid}],
        })
        assert response.status_code == 422


def test_transcript_append_rejects_holes_conflicts_and_redacts_validation_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, _, _ = _session_app(monkeypatch, tmp_path)
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    endpoint = f"/integration/mentra/session/{sitting_id}/transcripts/append"
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}

    hole = client.post(
        endpoint,
        json={
            **scope,
            "events": [{
                "event_id": f"{sitting_id}:2",
                "sequence": 2,
                "event_kind": "transcript",
                "role": "user",
                "content": "Fictional hole.",
                "status": "complete",
            }],
        },
    )
    private_text = "PRIVATE-FICTIONAL-TRANSCRIPT"
    invalid = client.post(
        endpoint,
        json={
            **scope,
            "events": [{
                "event_id": f"{sitting_id}:1",
                "sequence": 1,
                "event_kind": "transcript",
                "role": "user",
                "content": private_text,
                "status": "complete",
                "timestamp": "client-time-is-forbidden",
            }],
        },
    )

    assert hole.status_code == 409
    assert hole.json()["detail"] == {"expected_sequence": 1}
    assert invalid.status_code == 422
    assert invalid.json() == {"detail": "Invalid transcript batch"}
    assert private_text not in invalid.text


def test_transcript_retry_repairs_state_after_snapshot_was_written(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(
        monkeypatch,
        tmp_path,
        state_results=[RuntimeError("fictional state failure"), None],
        raise_server_exceptions=False,
    )
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    endpoint = f"/integration/mentra/session/{sitting_id}/transcripts/append"
    payload = {
        "user_id": START["user_id"],
        "soul_id": START["soul_id"],
        "events": [{
            "event_id": f"{sitting_id}:1",
            "sequence": 1,
            "event_kind": "transcript",
            "role": "user",
            "content": "A recoverable fictional line.",
            "status": "interrupted",
        }],
    }

    failed = client.post(endpoint, json=payload)
    repaired = client.post(endpoint, json=payload)

    assert failed.status_code == 500
    assert repaired.status_code == 200
    assert repaired.json()["accepted"] == 0
    assert repaired.json()["duplicates"] == 1
    assert len(calls["state_writes"]) == 2
    assert len(calls["memorize_checks"]) == 1


def test_snapshot_is_atomic_idempotent_scoped_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, _, _ = _session_app(monkeypatch, tmp_path)
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    endpoint = f"/integration/mentra/session/{sitting_id}/snapshot"
    image = b"\xff\xd8fictional-image\xff\xd9"
    payload = {
        "user_id": START["user_id"],
        "soul_id": START["soul_id"],
        "image_id": "image-1",
        "mime_type": "image/jpeg",
        "data": base64.b64encode(image).decode(),
    }

    stored = client.post(endpoint, json=payload)
    duplicate = client.post(endpoint, json=payload)
    replay = client.post(
        f"{endpoint}/replay",
        json={"user_id": START["user_id"], "soul_id": START["soul_id"], "image_id": "image-1"},
    )
    missing_replay = client.post(
        f"{endpoint}/replay",
        json={"user_id": START["user_id"], "soul_id": START["soul_id"], "image_id": "missing"},
    )
    no_lease_replay = client.post(
        "/integration/mentra/session/missing-sitting/snapshot/replay",
        json={"user_id": START["user_id"], "soul_id": START["soul_id"], "image_id": "image-1"},
    )
    dotted = client.post(
        endpoint,
        json={
            **payload,
            "image_id": "image-1.v2",
            "data": base64.b64encode(b"second-fictional-image").decode(),
        },
    )
    conflict = client.post(
        endpoint,
        json={**payload, "data": base64.b64encode(b"different").decode()},
    )
    private_data = "PRIVATE-FICTIONAL-IMAGE-DATA"
    invalid = client.post(endpoint, json={**payload, "data": private_data})
    large = client.post(
        endpoint,
        json={
            **payload,
            "image_id": "large",
            "data": base64.b64encode(b"x" * (1024 * 1024 + 1)).decode(),
        },
    )

    assert stored.status_code == duplicate.status_code == dotted.status_code == 200
    assert stored.json()["duplicate"] is False
    assert duplicate.json() == {**stored.json(), "duplicate": True}
    assert replay.json() == {"media_ref": stored.json()["media_ref"],
                             "mime_type": "image/jpeg", "data": base64.b64encode(image).decode()}
    assert missing_replay.status_code == 404
    assert no_lease_replay.status_code == 404
    assert stored.json()["media_ref"].startswith("mentra_media/")
    assert (tmp_path / "resources" / stored.json()["media_ref"]).read_bytes() == image
    assert conflict.status_code == 409
    assert invalid.status_code == 422
    assert private_data not in invalid.text
    assert large.status_code == 200
    assert (tmp_path / "resources" / large.json()["media_ref"]).stat().st_size == 1024 * 1024 + 1

    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    client.post(
        f"/integration/mentra/session/{sitting_id}/end", json=scope,
    ).raise_for_status()
    second = client.post(
        "/integration/mentra/session/start",
        json={**START, "device_session_id": "phone-2"},
    )
    second.raise_for_status()
    second_base = f"/integration/mentra/session/{second.json()['session_id']}/snapshot"
    shared = client.post(second_base, json=payload)
    assert shared.json() == {**stored.json(), "duplicate": True}
    assert client.post(
        f"{second_base}/replay", json={**scope, "image_id": "image-1"},
    ).json() == replay.json()

    other_scope = {**scope, "soul_id": "TestSoul"}
    other = client.post(
        "/integration/mentra/session/start", json={**START, **other_scope},
    )
    other.raise_for_status()
    other_base = f"/integration/mentra/session/{other.json()['session_id']}/snapshot"
    assert client.post(
        f"{other_base}/replay", json={**other_scope, "image_id": "image-1"},
    ).status_code == 404
    isolated = client.post(other_base, json={**payload, **other_scope})
    isolated.raise_for_status()
    assert isolated.json()["duplicate"] is False
    assert isolated.json()["media_ref"] != stored.json()["media_ref"]


@pytest.mark.parametrize("owner_kind", ["memorize", "import", "reflection"])
@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled", "waiting_cancelled"])
def test_snapshot_finalize_shares_memory_claim_and_preserves_waiting_auto_memorize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, owner_kind: str, outcome: str,
) -> None:
    from app import main

    async def run() -> None:
        started, release, exited, auto_started = (asyncio.Event() for _ in range(4))
        queue_auto = False
        scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}

        async def photo(_kwargs: dict[str, Any]) -> None:
            assert main._FORCED_MEMORIZE_INFLIGHT[marker] is None
            assert not calls["soul_lock"].locked()
            started.set()
            try:
                await release.wait()
                if outcome == "failure":
                    raise RuntimeError("Fictional photo failure")
            finally:
                exited.set()

        client, calls, _ = _session_app(
            monkeypatch, tmp_path, memorize_hook=photo,
            prepare_auto_memorize=lambda *_a, **_k: (1, {"conversation": []}) if queue_auto else (0, None),
            schedule_auto_memorize=lambda payload, *_a: main._schedule_auto_memorize(
                payload, {**scope, "conversation_id": conversation_sources.MENTRA_CONVERSATION_ID},
            ),
        )
        marker = main._memorize_lock_key(scope["user_id"], scope["soul_id"])
        unrelated = main._memorize_lock_key(scope["user_id"], "Other Fictional Soul")
        monkeypatch.setattr(main, "_BACKGROUND_TASKS", set())
        monkeypatch.setattr(main, "_soul_import_state", lambda *_a: ("", None))
        monkeypatch.setattr(main, "_register_memorize_bounds", lambda *_a, **_k: None)

        async def auto_memorize(_payload: dict[str, Any]) -> None:
            assert exited.is_set()
            assert main._FORCED_MEMORIZE_INFLIGHT[marker] is False
            auto_started.set()
            await main._finish_memorize_claim(marker, True)

        monkeypatch.setattr(main, "_run_forced_memorize_from_turn", auto_memorize)
        owners = main._CONSOLIDATION_RUNNING if owner_kind == "reflection" else main._FORCED_MEMORIZE_INFLIGHT
        owner_key = (scope["user_id"], scope["soul_id"]) if owner_kind == "reflection" else marker
        owners[owner_key] = owner_kind == "import"
        main._FORCED_MEMORIZE_INFLIGHT[unrelated] = True
        async with AsyncClient(transport=ASGITransport(app=client.app), base_url="http://test") as http:
            sitting = (await http.post("/integration/mentra/session/start", json=START)).json()["session_id"]
            base = f"/integration/mentra/session/{sitting}"
            caption = _photo_caption_event(sitting, 1, "shared-claim", "A fictional photo caption.")
            (await http.post(f"{base}/transcripts/append", json={**scope, "events": [caption]})).raise_for_status()
            stored = await http.post(f"{base}/snapshot", json={
                **scope, "image_id": "shared-claim", "mime_type": "image/png", "data": "AQID",
            })
            stored.raise_for_status()
            response = await http.post(f"{base}/snapshot/finalize", json={
                **scope, "image_id": "shared-claim", "caption": caption["content"],
                "caption_event_id": caption["event_id"],
            })
            assert response.json()["queued"] is True
            task = mentra_routes._image_finalize_tasks[(scope["user_id"], scope["soul_id"], stored.json()["media_ref"])][1]
            try:
                await asyncio.sleep(0)
                assert not started.is_set() and calls["image_memorize"] == []
                assert owners[owner_key] is (owner_kind == "import")
                waiting_reply = await asyncio.wait_for(http.post(f"{base}/transcripts/append", json={
                    **scope, "events": [{**caption, "event_id": f"{sitting}:2", "sequence": 2,
                                         "event_kind": "transcript", "status": "complete", "media_ref": None,
                                         "content": "A reply while the photo waits for memory work."}],
                }), 1)
                waiting_reply.raise_for_status()
                assert not started.is_set() and owner_key in owners
                if outcome == "waiting_cancelled":
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    assert owner_key in owners
                else:
                    if owner_kind == "memorize" and outcome == "success":
                        main._MEMORIZE_PROGRESS[marker] = {"active": True, "phase": "accepted"}
                        main._MEMORIZE_CANCEL.add(marker)
                        await main._finish_memorize_claim(marker, False)
                        assert main._MEMORIZE_PROGRESS[marker]["active"] is False
                        assert marker not in main._MEMORIZE_CANCEL
                    else:
                        owners.pop(owner_key)
                    await asyncio.wait_for(started.wait(), 1)
                    assert (await main.memorize_cancel(scope))["status"] == "no_active_memorize"
                    queue_auto = True
                    reply = await asyncio.wait_for(http.post(f"{base}/transcripts/append", json={
                        **scope, "events": [{**caption, "event_id": f"{sitting}:3", "sequence": 3,
                                             "event_kind": "transcript", "status": "complete", "media_ref": None,
                                             "content": "A reply while photo filing holds the claim."}],
                    }), 1)
                    reply.raise_for_status()
                    assert marker in main._FORCED_MEMORIZE_RECHECK and not auto_started.is_set()
                    assert not calls["soul_lock"].locked()
                    if outcome == "cancelled":
                        task.cancel()
                    else:
                        release.set()
                    result = (await asyncio.gather(task, return_exceptions=True))[0]
                    if outcome != "success":
                        assert isinstance(result, RuntimeError if outcome == "failure" else asyncio.CancelledError)
                    await asyncio.wait_for(auto_started.wait(), 1)
                    await asyncio.gather(*main._BACKGROUND_TASKS)
                    assert marker not in main._FORCED_MEMORIZE_INFLIGHT
                    assert marker not in main._FORCED_MEMORIZE_RECHECK
                await asyncio.sleep(0)
                assert main._FORCED_MEMORIZE_INFLIGHT[unrelated] is True
                assert mentra_routes._image_finalize_tasks == {} and calls["background_tasks"] == set()
                assert len(calls["background_errors"]) == (outcome == "failure")
            finally:
                release.set()
                task.cancel()
                await asyncio.gather(task, *main._BACKGROUND_TASKS, return_exceptions=True)

    asyncio.run(asyncio.wait_for(run(), 5))


def test_snapshot_finalize_keeps_reply_publication_available_and_photos_fifo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        started = [asyncio.Event() for _ in range(3)]
        release = [asyncio.Event() for _ in range(3)]

        async def memorize_photo(kwargs: dict[str, Any]) -> None:
            number = int(kwargs["caption"].split()[-1]) - 1
            assert not calls["soul_lock"].locked()
            started[number].set()
            await release[number].wait()

        client, calls, _ = _session_app(monkeypatch, tmp_path, memorize_hook=memorize_photo)
        scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
        unrelated = asyncio.get_running_loop().create_future()
        unrelated_key = (scope["user_id"], "Other Fictional Soul", "other-photo.png")
        mentra_routes._image_finalize_tasks[unrelated_key] = ("Unrelated caption", unrelated, "other-sitting")
        async with AsyncClient(transport=ASGITransport(app=client.app), base_url="http://test") as http:
            sitting = (await http.post("/integration/mentra/session/start", json=START)).json()["session_id"]
            base = f"/integration/mentra/session/{sitting}"
            captions = [_photo_caption_event(sitting, number, f"fifo-{number}", f"Fictional photo {number}")
                        for number in range(1, 5)]
            (await http.post(f"{base}/transcripts/append", json={**scope, "events": captions})).raise_for_status()
            payloads = []
            for number in range(1, 4):
                stored = await http.post(f"{base}/snapshot", json={
                    **scope, "image_id": f"fifo-{number}", "mime_type": "image/png",
                    "data": base64.b64encode(f"fictional-{number}".encode()).decode(),
                })
                stored.raise_for_status()
                payloads.append({**scope, "image_id": f"fifo-{number}",
                                 "caption": captions[number - 1]["content"],
                                 "caption_event_id": captions[number - 1]["event_id"]})
                response = await http.post(f"{base}/snapshot/finalize", json=payloads[-1])
                assert response.json() == {"ok": True, "durable": True, "queued": True, "duplicate": False}
                if number == 1:
                    await asyncio.wait_for(started[0].wait(), 1)
            await asyncio.sleep(0)
            tasks = [entry[1] for key, entry in mentra_routes._image_finalize_tasks.items()
                     if key[:2] == (scope["user_id"], scope["soul_id"])]
            assert len(tasks) == 3
            assert calls["background_tasks"] == set(tasks)
            assert [row["caption"] for row in calls["image_memorize"]] == ["Fictional photo 1"]
            assert not started[1].is_set() and not started[2].is_set()
            duplicate = await http.post(f"{base}/snapshot/finalize", json=payloads[1])
            assert duplicate.json() == {"ok": True, "durable": True, "queued": True, "duplicate": True}
            for _ in range(2):
                retry = await http.post("/integration/mentra/photos/fifo-2/retry", json=scope)
                assert retry.json() == {"ok": True, "durable": True, "queued": True, "duplicate": True}
                assert any(entry[1] is tasks[1] for entry in mentra_routes._image_finalize_tasks.values())
            assert len(calls["image_memorize"]) == 1
            conflict = await http.post(f"{base}/snapshot/finalize", json={
                **payloads[1], "caption": captions[3]["content"], "caption_event_id": captions[3]["event_id"],
            })
            assert conflict.status_code == 409
            assert len(mentra_routes._image_finalize_tasks) == 4
            # Iris publishes its ordinary replies via this actual locked route.
            before_reply = len(calls["state_writes"])
            reply = await asyncio.wait_for(http.post(f"{base}/transcripts/append", json={
                **scope, "events": [{**captions[0], "event_id": f"{sitting}:5", "sequence": 5,
                                    "event_kind": "transcript", "status": "complete", "media_ref": None,
                                    "content": "A reply while photo providers wait."}],
            }), 1)
            reply.raise_for_status()
            assert not tasks[0].done()
            assert len(calls["state_writes"]) == before_reply + 1
            history = conversation_sources.load_mentra_history_snapshot(
                storage_dir=tmp_path, **scope, conversation_id=conversation_sources.MENTRA_CONVERSATION_ID,
            )
            assert history[-1]["content"] == "A reply while photo providers wait."
            recall = await asyncio.wait_for(http.post(f"{base}/recall", json={**scope, "query": "fictional reply recall"}), 1)
            recall.raise_for_status()
            for number in range(3):
                await asyncio.wait_for(started[number].wait(), 1)
                assert [row["caption"] for row in calls["image_memorize"]] == [f"Fictional photo {n + 1}" for n in range(number + 1)]
                release[number].set()
                await asyncio.wait_for(tasks[number], 1)
            await asyncio.sleep(0)
            assert not unrelated.done()
            assert list(mentra_routes._image_finalize_tasks) == [unrelated_key]
            mentra_routes._image_finalize_tasks.pop(unrelated_key)
            assert mentra_routes._image_finalize_tasks == {}
            assert calls["background_tasks"] == set()
            assert calls["background_errors"] == []
            assert len(calls["image_resources"]) == 3
            assert len(calls["memory_items"]) == 3

    asyncio.run(asyncio.wait_for(run(), 5))


@pytest.mark.parametrize("outcome", ["failure", "predecessor_cancelled", "successor_cancelled"])
def test_snapshot_finalize_predecessor_outcomes_do_not_poison_other_photos(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: str,
) -> None:
    async def run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def memorize_photo(kwargs: dict[str, Any]) -> None:
            assert not calls["soul_lock"].locked()
            if kwargs["caption"] == "Fictional predecessor":
                started.set()
                await release.wait()
                if outcome == "failure":
                    raise RuntimeError("fictional predecessor failure")

        client, calls, _ = _session_app(monkeypatch, tmp_path, memorize_hook=memorize_photo)
        scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
        async with AsyncClient(transport=ASGITransport(app=client.app), base_url="http://test") as http:
            sitting = (await http.post("/integration/mentra/session/start", json=START)).json()["session_id"]
            base = f"/integration/mentra/session/{sitting}"
            captions = ["Fictional predecessor", "Fictional successor", "Fictional last photo"]
            events = [_photo_caption_event(sitting, number, f"isolation-{number}", caption)
                      for number, caption in enumerate(captions, 1)]
            (await http.post(f"{base}/transcripts/append", json={**scope, "events": events})).raise_for_status()
            tasks = []
            for number, event in enumerate(events, 1):
                stored = await http.post(f"{base}/snapshot", json={
                    **scope, "image_id": f"isolation-{number}", "mime_type": "image/png",
                    "data": base64.b64encode(f"fictional-{number}".encode()).decode(),
                })
                stored.raise_for_status()
                response = await http.post(f"{base}/snapshot/finalize", json={
                    **scope, "image_id": f"isolation-{number}", "caption": event["content"],
                    "caption_event_id": event["event_id"],
                })
                assert response.json()["queued"] is True
                key = (scope["user_id"], scope["soul_id"], stored.json()["media_ref"])
                tasks.append(mentra_routes._image_finalize_tasks[key][1])
                if number == 1:
                    await asyncio.wait_for(started.wait(), 1)
            await asyncio.sleep(0)
            assert len(calls["image_memorize"]) == 1
            if outcome == "predecessor_cancelled":
                tasks[0].cancel()
            elif outcome == "successor_cancelled":
                tasks[1].cancel()
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert not tasks[0].done() and not tasks[0].cancelling()
                assert len(calls["image_memorize"]) == 1
                release.set()
            else:
                release.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 1)
            await asyncio.sleep(0)
            assert not isinstance(results[2], BaseException)
            if outcome == "failure":
                assert isinstance(results[0], RuntimeError)
                assert len(calls["background_errors"]) == 1
                assert calls["background_errors"][0][1]["code"] == "mentra_image_finalize_failed"
            else:
                assert isinstance(results[0 if outcome == "predecessor_cancelled" else 1], asyncio.CancelledError)
                assert calls["background_errors"] == []
            expected = captions if outcome != "successor_cancelled" else [captions[0], captions[2]]
            assert [row["caption"] for row in calls["image_memorize"]] == expected
            assert mentra_routes._image_finalize_tasks == {}
            assert calls["background_tasks"] == set()

    asyncio.run(asyncio.wait_for(run(), 5))


def test_snapshot_finalize_coalesces_completed_registered_future_until_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(monkeypatch, tmp_path)

    async def run() -> None:
        scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
        async with AsyncClient(transport=ASGITransport(app=client.app), base_url="http://test") as http:
            sitting = (await http.post("/integration/mentra/session/start", json=START)).json()["session_id"]
            base = f"/integration/mentra/session/{sitting}"
            events = [_photo_caption_event(sitting, number, "completed-photo", caption)
                      for number, caption in enumerate(["Fictional caption", "Different caption"], 1)]
            (await http.post(f"{base}/transcripts/append", json={**scope, "events": events})).raise_for_status()
            stored = await http.post(f"{base}/snapshot", json={
                **scope, "image_id": "completed-photo", "mime_type": "image/png",
                "data": base64.b64encode(b"fictional-photo").decode(),
            })
            stored.raise_for_status()
            key = (scope["user_id"], scope["soul_id"], stored.json()["media_ref"])
            completed = asyncio.get_running_loop().create_future()
            completed.set_result(None)
            mentra_routes._image_finalize_tasks[key] = (events[0]["content"], completed, sitting)
            payload = {**scope, "image_id": "completed-photo", "caption": events[0]["content"],
                       "caption_event_id": events[0]["event_id"]}
            duplicate = await http.post(f"{base}/snapshot/finalize", json=payload)
            assert duplicate.json() == {"ok": True, "durable": True, "queued": True, "duplicate": True}
            conflict = await http.post(f"{base}/snapshot/finalize", json={
                **payload, "caption": events[1]["content"], "caption_event_id": events[1]["event_id"],
            })
            assert conflict.status_code == 409
            assert mentra_routes._image_finalize_tasks[key][1] is completed
            assert calls["image_memorize"] == []
            mentra_routes._image_finalize_tasks.pop(key)  # expose the narrow callback-cleanup boundary
            response = await http.post(f"{base}/snapshot/finalize", json=payload)
            assert response.json() == {"ok": True, "durable": True, "queued": True, "duplicate": False}
            await asyncio.wait_for(mentra_routes._image_finalize_tasks[key][1], 1)
            await asyncio.sleep(0)
            assert mentra_routes._image_finalize_tasks == {}
            assert calls["background_tasks"] == set()
            assert len(calls["image_memorize"]) == 1

    asyncio.run(asyncio.wait_for(run(), 5))


@pytest.mark.parametrize("retired", [None, "merged", "superseded"])
def test_snapshot_finalize_queues_existing_workflow_and_conflicts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    retired,
) -> None:
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    base = f"/integration/mentra/session/{sitting_id}/snapshot"
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    client.post(
        base,
        json={
            **scope,
            "image_id": "image-2",
            "mime_type": "image/png",
            "data": base64.b64encode(b"fictional-png").decode(),
        },
    )
    caption_event = _photo_caption_event(sitting_id, 1, "image-2", "A fictional magenta square.")
    client.post(
        f"/integration/mentra/session/{sitting_id}/transcripts/append",
        json={**scope, "events": [caption_event, {
            **caption_event, "event_id": f"{sitting_id}:2", "sequence": 2,
            "content": "A conflicting caption.",
        }]},
    ).raise_for_status()
    conflicting_caption_id = f"{sitting_id}:2"
    client.post(
        f"/integration/mentra/session/{sitting_id}/end", json=scope,
    ).raise_for_status()
    recovered = client.post(
        "/integration/mentra/session/start",
        json={**START, "device_session_id": "phone-2"},
    )
    recovered.raise_for_status()
    sitting_id = recovered.json()["session_id"]
    base = f"/integration/mentra/session/{sitting_id}/snapshot"
    mentra_routes._image_finalize_errors[(scope["user_id"], scope["soul_id"], sitting_id)] = {
        "mentra_media/fictional-scope/older.png"
    }
    payload = {
        **scope, "image_id": "image-2", "caption": caption_event["content"],
        "caption_event_id": caption_event["event_id"],
    }
    assert client.post(
        f"{base}/finalize", json={key: value for key, value in payload.items()
                                  if key != "caption_event_id"},
    ).status_code == 422
    before_finalize = datetime.now(UTC).replace(microsecond=0)
    finalized = client.post(
        f"{base}/finalize",
        json=payload,
    )
    item = calls["memory_items"]["image-item"]
    if retired == "merged":
        item.merged_into = "surviving-memory"
    elif retired == "superseded":
        item.superseded = True
    retry = client.post(
        f"{base}/finalize",
        json=payload,
    )
    conflict = client.post(
        f"{base}/finalize",
        json={**payload, "caption": "A conflicting caption.",
              "caption_event_id": conflicting_caption_id},
    )

    assert finalized.status_code == retry.status_code == 200
    assert finalized.json()["queued"] is True
    assert retry.json()["duplicate"] is True
    assert retry.json()["queued"] is False
    assert conflict.status_code == 409
    assert len(calls["image_memorize"]) == 1
    assert calls["image_memorize"][0]["modality"] == "image"
    assert calls["image_memorize"][0]["caption"] == "A fictional magenta square."
    saved_history = conversation_sources.load_mentra_history_snapshot(
        storage_dir=tmp_path, **scope, conversation_id=conversation_sources.MENTRA_CONVERSATION_ID,
    )
    record_path, = (tmp_path / "resources" / "mentra_media").glob("*/*.filing.json")
    receipt = json.loads(record_path.read_text())
    assert calls["image_memorize"][0]["source_timestamp"] == receipt["source_timestamp"]
    assert before_finalize <= datetime.fromisoformat(receipt["source_timestamp"]) <= datetime.now(UTC)
    assert any(row.get("event_id") == caption_event["event_id"] for row in saved_history)
    assert calls["image_memorize"][0]["user"] == {
        **scope, "conversation_id": conversation_sources.MENTRA_CONVERSATION_ID,
    }
    assert calls["image_lock_during_memorize"] is False
    assert not mentra_routes._lease_lock.locked()
    assert client.post(
        f"/integration/mentra/session/{sitting_id}/heartbeat", json=scope
    ).json() == {
        "ok": True,
        "pause_reason": None,
        "background_error": "Photo memory processing failed; the original remains saved.",
    }


@pytest.mark.parametrize("history_state", ["missing", "blocked", "unrelated"])
@pytest.mark.parametrize("photo_time,expected_time", [
    (None, None),
    ({"timestamp": "2026-01-08T10-17-23", "source": "DateTimeOriginal", "receivedAt": 0,
      "originalTime": "2026:01:08 10:17:23", "offset": "+05:45"}, "2026-01-08T10:17:23+05:45"),
    ({"timestamp": "2026-01-08T10-17-23", "source": "CreateDate", "receivedAt": 0,
      "originalTime": "2026:01:08 10:17:23"}, "2026-01-08T10:17:23"),
    ({"timestamp": "1970-01-01T00-00-00", "source": "receipt", "receivedAt": 0}, "1970-01-01T00:00:00.000Z"),
])
def test_snapshot_finalize_accepts_stored_original_without_caption_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    history_state: str,
    photo_time: dict[str, Any] | None,
    expected_time: str | None,
) -> None:
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START,
    ).json()["session_id"]
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    caption = "A fictional orange kite."
    image_id = "direct-tool-image"
    caption_id = f"{sitting_id}:photo:{image_id}"
    client.post(f"/integration/mentra/session/{sitting_id}/snapshot", json={
        **scope, "image_id": image_id, "mime_type": "image/png",
        "data": base64.b64encode(b"fictional original").decode(),
    }).raise_for_status()
    if history_state == "unrelated":
        conversation_sources.persist_mentra_history_snapshot(
            storage_dir=tmp_path, **scope,
            conversation_id=conversation_sources.MENTRA_CONVERSATION_ID,
            history=[{**_photo_caption_event(sitting_id, 1, "another-image", "Another caption."),
                      "received_at": "2026-10-07T10:00:00.000Z"}],
        )
    if history_state == "blocked":
        def unavailable_history(**_kwargs: Any) -> None:
            raise RuntimeError("Speech reservation prevents timeline publication")
        monkeypatch.setattr(conversation_sources, "load_mentra_history_snapshot", unavailable_history)
    payload = {
        **scope, "image_id": image_id, "caption": caption,
        "caption_event_id": caption_id,
        **({"time": photo_time} if photo_time else {}),
    }
    assert client.post(
        "/integration/mentra/session/wrong/snapshot/finalize", json=payload,
    ).status_code == 404
    before_finalize = datetime.now(UTC).replace(microsecond=0)
    response = client.post(
        f"/integration/mentra/session/{sitting_id}/snapshot/finalize",
        json=payload,
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True, "durable": True, "queued": True, "duplicate": False}
    record_path, = (tmp_path / "resources" / "mentra_media").glob("*/*.filing.json")
    accepted = record_path.read_bytes()
    receipt = json.loads(accepted)
    assert receipt["caption_event_id"] == caption_id
    if expected_time is None:
        assert before_finalize <= datetime.fromisoformat(receipt["source_timestamp"]) <= datetime.now(UTC)
    else:
        assert receipt["source_timestamp"] == expected_time
        assert receipt["time"] == photo_time
        conflict = client.post(f"/integration/mentra/session/{sitting_id}/snapshot/finalize", json={
            **payload, "time": {**photo_time, "receivedAt": photo_time["receivedAt"] + 1000},
        })
        assert conflict.status_code == 409
        assert record_path.read_bytes() == accepted
    assert calls["image_memorize"][0]["source_timestamp"] == receipt["source_timestamp"]
    assert client.post(
        f"/integration/mentra/session/{sitting_id}/snapshot/finalize", json=payload,
    ).json()["duplicate"] is True
    assert client.post(
        f"/integration/mentra/session/{sitting_id}/snapshot/finalize",
        json={**payload, "caption": "A conflicting caption."},
    ).status_code == 409
    assert record_path.read_bytes() == accepted
    assert len(calls["image_memorize"]) == 1
    assert mentra_routes._image_finalize_tasks == {}
    assert calls["background_errors"] == []
    if history_state == "missing":
        image_event = {**_photo_caption_event(sitting_id, 1, image_id, caption),
                       "event_kind": "image", "role": "user", "content": "Shared a photo."}
        caption_event = _photo_caption_event(sitting_id, 2, image_id, caption)
        append_url = f"/integration/mentra/session/{sitting_id}/transcripts/append"
        for invalid_events in (
            [{**image_event, "event_id": f"{sitting_id}:image:{image_id}"}],
            [image_event, {**caption_event, "event_id": caption_id}],
        ):
            assert client.post(append_url, json={
                **scope, "events": invalid_events,
            }).status_code == 409
        events = [image_event, caption_event]
        appended = client.post(append_url, json={**scope, "events": events})
        appended.raise_for_status()
        assert appended.json()["accepted"] == 2
        assert appended.json()["ack_sequence"] == 2
        replayed = client.post(append_url, json={**scope, "events": events})
        replayed.raise_for_status()
        assert replayed.json()["duplicates"] == 2
        assert client.post(
            f"/integration/mentra/session/{sitting_id}/snapshot/finalize", json=payload,
        ).json() == {"ok": True, "durable": True, "queued": False, "duplicate": True}
        history = conversation_sources.load_mentra_history_snapshot(
            storage_dir=tmp_path, **scope, conversation_id=conversation_sources.MENTRA_CONVERSATION_ID,
        )
        assert [row["event_id"] for row in history] == [f"{sitting_id}:1", f"{sitting_id}:2"]
        assert receipt["caption_event_id"] not in {row["event_id"] for row in history}
        assert len(calls["image_memorize"]) == 1
        assert len(calls["image_resources"]) == 1
        assert record_path.read_bytes() == accepted


@pytest.mark.parametrize("original_state", ["missing", "owner", "soul", "ambiguous"])
def test_snapshot_finalize_refuses_missing_or_wrong_scoped_original(monkeypatch, tmp_path, original_state):
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    sitting = client.post("/integration/mentra/session/start", json=START).json()["session_id"]
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    root = tmp_path / "resources"
    saved_scope = dict(scope)
    if original_state == "owner":
        saved_scope["user_id"] = "Another Fictional User"
    elif original_state == "soul":
        saved_scope["soul_id"] = "Another Fictional Soul"
    if original_state != "missing":
        directory = mentra_routes._media_directory(root, **saved_scope)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "direct-tool-image.png").write_bytes(b"fictional original")
        if original_state == "ambiguous":
            (directory / "direct-tool-image.jpg").write_bytes(b"another fictional original")
    service_calls = calls["service"]
    response = client.post(f"/integration/mentra/session/{sitting}/snapshot/finalize", json={
        **scope, "image_id": "direct-tool-image", "caption": "A fictional kite.",
        "caption_event_id": f"{sitting}:photo:direct-tool-image",
    })
    assert response.status_code == 404
    assert calls["service"] == service_calls
    assert calls["image_memorize"] == []
    assert calls["image_resources"] == {}
    assert list(root.glob("mentra_media/*/*.filing.json")) == []
    assert mentra_routes._image_finalize_tasks == {}


def test_snapshot_finalize_rejects_non_multimodal_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, _, config = _session_app(monkeypatch, tmp_path)
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    base = f"/integration/mentra/session/{sitting_id}/snapshot"
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    client.post(
        base,
        json={
            **scope,
            "image_id": "image-disabled",
            "mime_type": "image/png",
            "data": base64.b64encode(b"fictional-png").decode(),
        },
    )
    client.post(
        f"/integration/mentra/session/{sitting_id}/transcripts/append",
        json={**scope, "events": [_photo_caption_event(sitting_id, 1, "image-disabled", "A fictional image.")]},
    ).raise_for_status()
    config["llm"]["embedding"] = {}

    response = client.post(
        f"{base}/finalize",
        json={**scope, "image_id": "image-disabled", "caption": "A fictional image.",
              "caption_event_id": f"{sitting_id}:1"},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Mentra image embedding is not configured"


@pytest.mark.parametrize("observe", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_replacement_photo_check_observes_existing_worker_only_when_requested(monkeypatch, tmp_path, observe, fails):
    async def run():
        started = asyncio.Event()
        release = asyncio.Event()

        async def filing(kwargs):
            calls["image_resources"]["saved-photo"] = SimpleNamespace(
                id="saved-photo", url=kwargs["resource_url"], modality="image", caption=kwargs["caption"],
            )
            calls["memory_items"]["saved-item"] = SimpleNamespace(resource_id="saved-photo")
            started.set()
            await release.wait()
            if fails:
                raise RuntimeError("fictional post-save filing failure")

        client, calls, _ = _session_app(monkeypatch, tmp_path, memorize_hook=filing)
        scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
        async with AsyncClient(transport=ASGITransport(app=client.app), base_url="http://test") as http:
            first = (await http.post("/integration/mentra/session/start", json=START)).json()["session_id"]
            base = f"/integration/mentra/session/{first}"
            caption = "A fictional lighthouse photograph."
            event_id = f"{first}:1"
            (await http.post(f"{base}/transcripts/append", json={**scope, "events": [
                _photo_caption_event(first, 1, "observed-photo", caption),
            ]})).raise_for_status()
            stored = await http.post(f"{base}/snapshot", json={**scope, "image_id": "observed-photo",
                "mime_type": "image/png", "data": base64.b64encode(b"fictional-photo").decode()})
            stored.raise_for_status()
            payload = {**scope, "image_id": "observed-photo", "caption": caption, "caption_event_id": event_id}
            assert (await http.post(f"{base}/snapshot/finalize", json=payload)).json()["queued"] is True
            await asyncio.wait_for(started.wait(), 1)
            key = (scope["user_id"], scope["soul_id"], stored.json()["media_ref"])
            task = mentra_routes._image_finalize_tasks[key][1]
            second = (await http.post("/integration/mentra/session/start", json=START)).json()["session_id"]
            base = f"/integration/mentra/session/{second}"
            error_key = (*key[:2], second)
            if observe:
                response = await http.post(f"{base}/snapshot/finalize", json=payload)
                assert response.json() == {"ok": True, "durable": True, "queued": True, "duplicate": True}
                assert mentra_routes._image_finalize_tasks[key][1] is task
                assert mentra_routes._image_finalize_tasks[key][2] == second
                if not fails:
                    mentra_routes._image_finalize_errors[error_key] = {key[2], "another-photo.png"}
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.sleep(0)
            heartbeat = (await http.post(f"{base}/heartbeat", json=scope)).json()
            assert bool(heartbeat.get("background_error")) == observe
            if observe and not fails:
                assert mentra_routes._image_finalize_errors[error_key] == {"another-photo.png"}
            assert len(calls["image_memorize"]) == 1
            assert key not in mentra_routes._image_finalize_tasks
            assert calls["background_tasks"] == set()
            assert len(calls["background_errors"]) == int(fails)
            history = conversation_sources.load_mentra_history_snapshot(storage_dir=tmp_path,
                **scope, conversation_id=conversation_sources.MENTRA_CONVERSATION_ID)
            assert [row["event_id"] for row in history] == [event_id]

    asyncio.run(asyncio.wait_for(run(), 5))


def test_snapshot_finalize_failure_keeps_bytes_and_reports_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, calls, _ = _session_app(
        monkeypatch, tmp_path, memorize_error=RuntimeError("fictional failure")
    )
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    base = f"/integration/mentra/session/{sitting_id}/snapshot"
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    stored = client.post(
        base,
        json={
            **scope,
            "image_id": "image-3",
            "mime_type": "image/png",
            "data": base64.b64encode(b"fictional-png").decode(),
        },
    )
    client.post(
        f"/integration/mentra/session/{sitting_id}/transcripts/append",
        json={**scope, "events": [_photo_caption_event(sitting_id, 1, "image-3", "A fictional failed image.")]},
    ).raise_for_status()
    finalized = client.post(
        f"{base}/finalize",
        json={**scope, "image_id": "image-3", "caption": "A fictional failed image.",
              "caption_event_id": f"{sitting_id}:1"},
    )

    assert finalized.status_code == 200
    assert (tmp_path / "resources" / stored.json()["media_ref"]).exists()
    assert list(calls["image_resources"]) == ["image-resource"]
    assert calls["background_errors"]
    assert calls["background_errors"][0][1]["code"] == "mentra_image_finalize_failed"
    heartbeat = client.post(
        f"/integration/mentra/session/{sitting_id}/heartbeat", json=scope
    )
    assert heartbeat.json() == {
        "ok": True,
        "pause_reason": None,
        "background_error": "Photo memory processing failed; the original remains saved.",
    }
    assert client.post(
        f"/integration/mentra/session/{sitting_id}/heartbeat", json=scope
    ).json() == heartbeat.json()


def _file_test_photo(client: TestClient, *, image_id: str = "durable-photo") -> tuple[dict[str, str], dict[str, str]]:
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    sitting = client.post("/integration/mentra/session/start", json=START).json()["session_id"]
    base = f"/integration/mentra/session/{sitting}"
    client.post(f"{base}/snapshot", json={**scope, "image_id": image_id,
                "mime_type": "image/png", "data": base64.b64encode(b"fictional image").decode()}).raise_for_status()
    event = _photo_caption_event(sitting, 1, image_id, "A fictional blue notebook.")
    client.post(f"{base}/transcripts/append", json={**scope, "events": [event]}).raise_for_status()
    payload = {**scope, "image_id": image_id, "caption": event["content"], "caption_event_id": event["event_id"]}
    client.post(f"{base}/snapshot/finalize", json=payload).raise_for_status()
    return scope, payload


@pytest.mark.parametrize("history_state", ["missing", "newer_caption"])
def test_zero_item_photo_completion_survives_rediscovery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, history_state) -> None:
    client, calls, _ = _session_app(monkeypatch, tmp_path)
    scope, payload = _file_test_photo(client)
    calls["memory_items"].clear()
    sitting = payload["caption_event_id"].split(":")[0]
    record_path, = (tmp_path / "resources" / "mentra_media").glob("*/*.filing.json")
    accepted = record_path.read_bytes()
    history_path, = (tmp_path / "transcripts").rglob("latest_history.json")
    newer = _photo_caption_event(sitting, 2, "durable-photo", "A different fictional caption.")
    if history_state == "missing":
        history_path.unlink()
    else:
        conversation_sources.persist_mentra_history_snapshot(
            storage_dir=tmp_path, **scope, conversation_id="mentra:iris",
            history=[{**newer, "received_at": "2026-10-20T10:00:00Z"}],
        )
    reply = client.post(f"/integration/mentra/session/{sitting}/snapshot/finalize", json=payload)
    assert reply.json()["durable"] is True and reply.json()["duplicate"] is True
    assert reply.json()["queued"] is False
    conflict = client.post(f"/integration/mentra/session/{sitting}/snapshot/finalize", json={
        **payload, "caption": newer["content"], "caption_event_id": newer["event_id"],
    })
    assert conflict.status_code == 409
    assert record_path.read_bytes() == accepted
    assert len(calls["image_memorize"]) == 1
    if history_state != "missing":
        assert client.get("/integration/mentra/status", params=scope).json()["failed_photos"] == []


@pytest.mark.parametrize("status,mismatch", [
    ("queued", None), ("running", None), ("complete", None), ("failed", None), ("discarded", None),
    ("complete", "user_id"), ("complete", "soul_id"), ("complete", "image_id"), ("complete", "media_ref"),
])
def test_snapshot_replay_exposes_only_matching_receipt_after_reconnect(monkeypatch, tmp_path, status, mismatch):
    client, _, _ = _session_app(monkeypatch, tmp_path)
    scope, payload = _file_test_photo(client)
    sitting = payload["caption_event_id"].split(":")[0]
    client.post(f"/integration/mentra/session/{sitting}/end", json=scope).raise_for_status()
    restarted, calls, _ = _session_app(monkeypatch, tmp_path)
    new_sitting = restarted.post("/integration/mentra/session/start", json=START).json()["session_id"]
    record_path, = (tmp_path / "resources" / "mentra_media").glob("*/*.filing.json")
    record = json.loads(record_path.read_text())
    record["status"] = status
    if mismatch:
        record[mismatch] = "unrelated-fictional-identity"
    mentra_routes._write_photo_filing(record_path, record)
    before = record_path.read_bytes()
    replay = restarted.post(f"/integration/mentra/session/{new_sitting}/snapshot/replay", json={
        **scope, "image_id": payload["image_id"],
    })
    if mismatch:
        assert replay.status_code == 409
        assert payload["caption"] not in replay.text
    else:
        replay.raise_for_status()
        assert replay.json()["filing"] == {
            "caption": payload["caption"], "caption_event_id": payload["caption_event_id"], "status": status,
        }
        assert base64.b64decode(replay.json()["data"]) == b"fictional image"
    assert record_path.read_bytes() == before
    assert calls["image_memorize"] == []
    assert mentra_routes._image_finalize_tasks == {}


def test_photo_retry_clears_the_active_observers_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def fail_once(_kwargs: dict[str, Any]) -> None:
        if len(calls["image_memorize"]) == 1:
            raise RuntimeError("fictional one-time failure")

    client, calls, _ = _session_app(monkeypatch, tmp_path, memorize_hook=fail_once)
    scope, payload = _file_test_photo(client)
    sitting = payload["caption_event_id"].split(":")[0]
    heartbeat = f"/integration/mentra/session/{sitting}/heartbeat"
    assert "background_error" in client.post(heartbeat, json=scope).json()
    with monkeypatch.context() as patch:
        def unavailable_history(**_kwargs):
            raise AssertionError("accepted receipts must precede live-history lookup")
        patch.setattr(conversation_sources, "load_mentra_history_snapshot", unavailable_history)
        response = client.post(f"/integration/mentra/session/{sitting}/snapshot/finalize", json=payload)
        assert response.json() == {"ok": True, "durable": True, "queued": False, "failed": True}
        assert len(calls["image_memorize"]) == 1
    client.post("/integration/mentra/photos/durable-photo/retry", json=scope).raise_for_status()
    assert "background_error" not in client.post(heartbeat, json=scope).json()
    assert len(calls["image_memorize"]) == 2


def test_failed_photo_idle_retry_survives_restart(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client, calls, _ = _session_app(monkeypatch, tmp_path, memorize_error=RuntimeError("fictional failure"))
    scope, payload = _file_test_photo(client)
    sitting = payload["caption_event_id"].split(":")[0]
    client.post(f"/integration/mentra/session/{sitting}/end", json=scope).raise_for_status()
    mentra_routes._image_finalize_errors.clear()
    restarted, restored, _ = _session_app(monkeypatch, tmp_path)
    status = restarted.get("/integration/mentra/status", params=scope).json()
    assert status["failed_photos"] == ["durable-photo"]
    assert restored["image_memorize"] == []
    assert restarted.get("/integration/mentra/status", params=scope).json()["failed_photos"] == ["durable-photo"]
    assert restored["image_memorize"] == []
    response = restarted.post("/integration/mentra/photos/durable-photo/retry", json=scope)
    response.raise_for_status()
    assert response.json()["ok"] is True
    assert len(restored["image_memorize"]) == 1
    assert restarted.get("/integration/mentra/status", params=scope).json()["failed_photos"] == []
    assert calls["background_errors"]


@pytest.mark.parametrize("resource_committed", [False, True])
def test_interrupted_photo_filing_recovers_on_rediscovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, resource_committed: bool,
) -> None:
    async def interrupted(kwargs: dict[str, Any]) -> None:
        if resource_committed:
            calls["image_resources"]["early-resource"] = SimpleNamespace(
                id="early-resource", url=kwargs["resource_url"], modality="image", caption=kwargs["caption"],
            )
        await asyncio.Event().wait()

    client, calls, _ = _session_app(monkeypatch, tmp_path, memorize_hook=interrupted)
    scope, _ = _file_test_photo(client)
    record_path, = (tmp_path / "resources" / "mentra_media").glob("*/*.filing.json")
    assert json.loads(record_path.read_text())["status"] in {"queued", "running"}
    restarted, restored, _ = _session_app(monkeypatch, tmp_path)
    restored["image_resources"].update(calls["image_resources"])
    response = restarted.get("/integration/mentra/status", params=scope)
    response.raise_for_status()
    assert restored["image_memorize"] == []
    mentra_routes._leases.clear()
    restarted.post("/integration/mentra/session/start", json=START).raise_for_status()
    assert len(restored["image_memorize"]) == 1
    assert len(restored["image_resources"]) == 1
    assert restored["image_memorize"][0]["resume_image_filing"] is True
    assert json.loads(record_path.read_text())["status"] == "complete"
    restarted.get("/integration/mentra/status", params=scope).raise_for_status()
    assert len(restored["image_memorize"]) == 1


def test_failed_photo_discard_is_scoped_and_does_not_delete_original(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from app import main
    client, calls, _ = _session_app(monkeypatch, tmp_path, memorize_error=RuntimeError("fictional failure"))
    client.app.add_exception_handler(owner.OwnerIdentityError, main._owner_identity_error)
    scope, _ = _file_test_photo(client)
    for action in ("retry", "discard"):
        assert client.post(f"/integration/mentra/photos/durable-photo/{action}",
                           json={**scope, "soul_id": "Another Soul"}).status_code == 404
        assert client.post(f"/integration/mentra/photos/durable-photo/{action}",
                           json={**scope, "user_id": "Other Owner"}).status_code == 409
    response = client.post("/integration/mentra/photos/durable-photo/discard", json=scope)
    response.raise_for_status()
    assert client.get("/integration/mentra/status", params=scope).json()["failed_photos"] == []
    original, = (tmp_path / "resources" / "mentra_media").glob("*/*.png")
    assert original.read_bytes() == b"fictional image"
    assert len(calls["image_memorize"]) == 1


@pytest.mark.parametrize("event_kind", ["image", "photo_caption"])
def test_image_transcript_event_round_trips_media_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    event_kind: str,
) -> None:
    client, _, _ = _session_app(monkeypatch, tmp_path)
    sitting_id = client.post(
        "/integration/mentra/session/start", json=START
    ).json()["session_id"]
    endpoint = f"/integration/mentra/session/{sitting_id}/transcripts/append"
    scope = {"user_id": START["user_id"], "soul_id": START["soul_id"]}
    event = {
        "event_id": f"{sitting_id}:1",
        "sequence": 1,
        "event_kind": "image",
        "role": "user",
        "content": "Shared a photo.",
        "media_ref": "mentra_media/fictional-scope/image-1.jpg",
    }
    if event_kind == "photo_caption":
        event.update(event_kind="photo_caption", role="assistant", content="A fictional orange kite.")

    accepted = client.post(endpoint, json={**scope, "events": [event]})
    duplicate = client.post(endpoint, json={**scope, "events": [event]})
    invalid = client.post(
        endpoint,
        json={**scope, "events": [{**event, "media_ref": "/private/image.jpg"}]},
    )
    history = mentra_routes.conversation_sources.load_mentra_tail(
        storage_dir=tmp_path,
        user_id=START["user_id"],
        soul_id=START["soul_id"],
        conversation_id=conversation_sources.MENTRA_CONVERSATION_ID,
        since_cursor=-1,
        recent_fallback_messages=0,
    )

    assert accepted.status_code == duplicate.status_code == 200
    assert duplicate.json()["duplicates"] == 1
    assert invalid.status_code == 422
    if event_kind == "photo_caption":
        for patch in ({"role": "user"}, {"status": "complete"}, {"media_ref": None},
                      {"mic_source": "phone"}, {"content": " "}):
            assert client.post(endpoint, json={**scope, "events": [{**event, **patch}]}).status_code == 422
        assert history[0]["content"] == "[Internal visual perception; not spoken aloud] " + event["content"]
        from memu.utils.conversation import format_grouped_chat_history
        rendered = format_grouped_chat_history(history, soul_name=scope["soul_id"])
        assert history[0]["content"] in rendered
        raw = conversation_sources.load_mentra_history_snapshot(
            storage_dir=tmp_path, **scope, conversation_id="mentra:iris",
        )
        assert raw[0]["content"] == event["content"]
        assert "transcript_status" not in raw[0]
    assert history[0]["event_kind"] == event_kind
    assert history[0]["media_ref"] == event["media_ref"]
    assert history[0]["source_conversation_index"] == 1


@pytest.mark.parametrize(
    ("mode", "activity_detection"),
    [
        (
            "continuous",
            {
                "endOfSpeechSensitivity": "END_SENSITIVITY_LOW",
                "silenceDurationMs": 1_500,
            },
        ),
        ("manual", {"disabled": True}),
    ],
)
@pytest.mark.parametrize("model", [
    "gemini-3.8-live", "gemini-3.8-live-extended-thinking", "custom-live-model",
])
def test_token_mint_uses_measured_constrained_wire(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    activity_detection: dict[str, Any],
    model: str,
) -> None:
    captured: dict[str, Any] = {}

    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: Any) -> None:
            self.close()

    def urlopen(request: Any, timeout: int) -> Response:
        captured["request"] = request
        captured["timeout"] = timeout
        return Response(b'{"name":"ephemeral"}')

    monkeypatch.setattr(mentra_routes.urllib.request, "urlopen", urlopen)
    token = asyncio.run(
        mentra_routes._mint_gemini_token(
            api_key="secret",
            model=model,
            voice="Kore",
            system_instruction="fictional context",
            mode=mode,
        )
    )

    assert token == "ephemeral"
    request = captured["request"]
    payload = json.loads(request.data)
    setup = payload["bidiGenerateContentSetup"]
    assert captured["timeout"] == 10
    assert request.get_header("X-goog-api-key") == "secret"
    assert request.full_url == "https://generativelanguage.googleapis.com/v1alpha/auth_tokens"
    assert payload["uses"] == 8
    assert payload["fieldMask"] == (
        "model,generationConfig,systemInstruction,tools,realtimeInputConfig,"
        "inputAudioTranscription,outputAudioTranscription,proactivity,contextWindowCompression,"
        "historyConfig"
    )
    assert setup["model"] == f"models/{model}"
    generation = {
        "responseModalities": ["AUDIO"],
        "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}},
    }
    if model not in ("gemini-3.8-live", "gemini-3.8-live-extended-thinking"):
        generation["enableAffectiveDialog"] = True
    if model == "gemini-3.8-live-extended-thinking":
        generation["thinkingConfig"] = {"thinkingLevel": "LOW"}
    assert setup["generationConfig"] == generation
    assert setup["systemInstruction"] == {"parts": [{"text": "fictional context"}]}
    declarations = setup["tools"][0]["functionDeclarations"]
    assert [tool["name"] for tool in declarations] == [
        "recall_memory", "read_working_state", "save_working_thought", "annul_intention", "save_photo_caption",
        "nothing_to_add",
    ]
    assert all(tool["behavior"] == "NON_BLOCKING" for tool in declarations)
    assert declarations[0]["description"] == mentra_routes._RECALL_TOOL_DESCRIPTION
    assert declarations[0]["parameters"] == {
        "type": "OBJECT", "properties": {"query": {"type": "STRING"}}, "required": ["query"],
    }
    assert declarations[1]["parameters"] == {"type": "OBJECT", "properties": {}}
    assert declarations[2]["parameters"]["required"] == ["thought"]
    assert declarations[3]["parameters"]["required"] == ["intention_id", "status"]
    assert declarations[3]["parameters"]["properties"]["status"]["enum"] == ["completed", "deleted"]
    assert declarations[4]["description"] == (
        "Save your perception of the photo with its exact image_id, so you will remember "
        "what this moment was about later on. Do not speak the caption out loud."
    )
    assert declarations[4]["parameters"] == {
        "type": "OBJECT", "properties": {"image_id": {"type": "STRING"}, "caption": {"type": "STRING"}},
        "required": ["image_id", "caption"],
    }
    assert declarations[5] == {
        "name": "nothing_to_add",
        "description": "Choose not to speak when there is nothing worthwhile to add to the conversation.",
        "behavior": "NON_BLOCKING",
        "parameters": {"type": "OBJECT", "properties": {}},
    }
    assert setup["realtimeInputConfig"] == {
        "automaticActivityDetection": activity_detection,
    }
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert setup["inputAudioTranscription"] == {}
    assert setup["outputAudioTranscription"] == {}
    assert "sessionResumption" not in setup
    assert setup["contextWindowCompression"] == {"slidingWindow": {}}
    assert setup["proactivity"] == {"proactiveAudio": True}
    assert setup["historyConfig"] == {}
    assert set(payload["fieldMask"].split(",")) == set(setup)
    expires = datetime.fromisoformat(payload["expireTime"].replace("Z", "+00:00"))
    new_session = datetime.fromisoformat(
        payload["newSessionExpireTime"].replace("Z", "+00:00")
    )
    now = datetime.now(UTC)
    assert 29 * 60 < (expires - now).total_seconds() <= 30 * 60
    assert 29 * 60 < (new_session - now).total_seconds() <= 30 * 60


@pytest.mark.parametrize("mode", ["continuous", "manual"])
@pytest.mark.parametrize("model", ["gemini-3.8-live", "gemini-3.8-live-extended-thinking"])
@pytest.mark.parametrize("vision", [None, False, True])
def test_start_and_renewal_mint_identical_six_tool_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, model: str, vision: bool | None,
) -> None:
    mint = mentra_routes._mint_gemini_token
    client, _, config = _session_app(monkeypatch, tmp_path)
    payloads: list[dict[str, Any]] = []

    def urlopen(request: Any, timeout: int) -> io.BytesIO:
        payloads.append(json.loads(request.data))
        return io.BytesIO(b'{"name":"ephemeral-offline"}')

    monkeypatch.setattr(mentra_routes, "_mint_gemini_token", mint)
    monkeypatch.setattr(mentra_routes.urllib.request, "urlopen", urlopen)
    started = client.post("/integration/mentra/session/start", json={
        **START, "mode": mode, "model": model, **({"vision": vision} if vision is not None else {}),
    })
    assert started.status_code == 200
    assert started.json()["photo_work_protocol"] == "iris-photo-work-v2"
    config["mentra"]["model"] = "edited-server-model"
    renewed = client.post(
        f"/integration/mentra/session/{started.json()['session_id']}/token",
        json={"user_id": START["user_id"], "soul_id": START["soul_id"]},
    )
    assert renewed.status_code == 200
    assert len(payloads) == 2
    initial, renewal = payloads
    assert initial["fieldMask"] == renewal["fieldMask"] == mentra_routes._TOKEN_SETUP_FIELD_MASK
    assert initial["bidiGenerateContentSetup"] == renewal["bidiGenerateContentSetup"]
    setup = initial["bidiGenerateContentSetup"]
    prompt = setup["systemInstruction"]["parts"][0]["text"]
    assert mentra_routes._leases[START["soul_id"]].system_instruction == prompt
    if vision:
        assert prompt.count(mentra_routes._VISION_GUIDANCE) == 1
        assert mentra_routes._PHOTO_CAPTION_GUIDANCE not in prompt
    else:
        assert prompt.count(mentra_routes._PHOTO_CAPTION_GUIDANCE) == 1
        assert mentra_routes._VISION_GUIDANCE not in prompt
    assert setup["model"] == f"models/{model}"
    assert setup["generationConfig"].get("thinkingConfig") == (
        {"thinkingLevel": "LOW"} if model == "gemini-3.8-live-extended-thinking" else None
    )
    declarations = initial["bidiGenerateContentSetup"]["tools"][0]["functionDeclarations"]
    assert [tool["name"] for tool in declarations] == [
        "recall_memory", "read_working_state", "save_working_thought", "annul_intention",
        "save_photo_caption", "nothing_to_add",
    ]
    assert all(tool["behavior"] == "NON_BLOCKING" for tool in declarations)
    assert declarations[-1]["parameters"] == {"type": "OBJECT", "properties": {}}


def test_partial_mentra_config_update_preserves_omitted_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from app import main

    config = copy.deepcopy(main._CONFIG)
    config["mentra"] = _configured()["mentra"]
    on_disk = copy.deepcopy(config)
    on_disk["mentra"]["public_base_url"] = "http://new-server.invalid"
    on_disk["unrelated_on_disk"] = "keep"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(on_disk), encoding="utf-8")
    saved: dict[str, Any] = {}
    monkeypatch.setattr(main, "_CONFIG", config)
    monkeypatch.setattr(main, "_config_path", lambda: path)
    monkeypatch.setattr(main, "_save_config", lambda value: saved.update(value))
    monkeypatch.setattr(main, "_clear_cached_services", lambda: None)

    response = TestClient(main.app).post("/config", json={"mentra": {"voice": "Puck"}})

    assert response.status_code == 200
    assert saved["mentra"] == {**on_disk["mentra"], "voice": "Puck"}
    assert saved["unrelated_on_disk"] == "keep"
    assert response.json()["config"]["mentra"]["gemini_api_key"] == "***"

    path.write_text("{broken", encoding="utf-8")
    saved.clear()
    assert TestClient(main.app).post("/config", json={"mentra": {"voice": "Kore"}}).status_code == 400
    assert not saved
