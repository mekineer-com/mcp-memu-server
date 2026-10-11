"""Contract test for /memorize: the endpoint's BackgroundTasks must actually run.

History: the endpoint adds `_run_memorize_segments` via `background_tasks.add_task(...)`
but returns a `JSONResponse(status_code=202, ...)` built inline. In FastAPI,
when an endpoint returns a Response object directly, tasks on the injected
`BackgroundTasks` parameter are NOT auto-attached — you have to pass
`background=background_tasks` to the Response. Without it, add_task is a
silent no-op and the memorize batches never run.

Turn-triggered memorize goes through `_run_forced_memorize_from_turn` which
creates its own BackgroundTasks and awaits it manually, so it's unaffected.
Only direct `POST /memorize` calls (Re-memorize chat button, st-api.sh, and
every agent smoke-test driver) hit this hole.

This test is the regression guard: it stubs `_run_memorize_segments` with a
recorder, posts to /memorize, and asserts the task actually ran.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from fastapi import BackgroundTasks, HTTPException

from app import main as main_module
from app.main import app
from app.services import souls, owner
from app.services import memorize_endpoint, state
from tests import SavedBatchService


@pytest.fixture(autouse=True)
def _healthy_paid_state(monkeypatch):
    # These file/archive tests use fake DB bytes; pause behavior has real-DB checks.
    monkeypatch.setattr(main_module, "_paid_work_state", lambda *_args: main_module._soul_state.defaults())
    monkeypatch.setattr(main_module, "_soul_import_state", lambda *_args: (None, {}))


def _state_response() -> dict:
    return {"digest_cursor": 0, "last_memorize_at": None, "pending_segment_ids": []}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_memorize_background_task_runs(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[dict] = []

    async def fake_run(**kwargs) -> None:
        recorded.append({"ran": True, "segment_count": len(kwargs.get("memorize_segments") or [])})

    # Stub the heavy work so we can verify the task fires without LLM calls.
    monkeypatch.setattr(main_module._memorize_endpoint, "run_memorize_segments", fake_run)

    # Stub _get_service_from_payload to avoid real service construction.
    class _FakeSvc:
        pass

    monkeypatch.setattr(main_module, "_get_service_from_payload", lambda *a, **k: _FakeSvc())

    payload = {
        "user": {"user_id": "test_user", "soul_id": "test_soul", "conversation_id": "chat:cid-1"},
        "conversation_id": "chat:cid-1",
        "conversation": [
            {"role": "user", "name": "test_user", "content": "hello"},
            {"role": "assistant", "name": "test_soul", "content": "hi"},
        ],
        "llm_profiles": {"default": {"provider": "openai", "api_key": "x", "base_url": "x", "chat_model": "x", "embed_model": "x"}},
    }
    resp = client.post("/memorize?rebuild=true", json=payload)
    assert resp.status_code == 202, f"unexpected status: {resp.status_code} body={resp.text[:300]}"
    body = resp.json()
    assert body.get("status") == "accepted"
    assert isinstance(body.get("segment_count"), int)
    assert body.get("segment_count", 0) >= 1

    # TestClient blocks until BackgroundTasks complete, so by the time .post()
    # returns, our fake_run should have been invoked.
    assert recorded, (
        "background task did NOT run — JSONResponse is missing background=background_tasks; "
        "check app/main.py memorize endpoint return"
    )
    assert recorded[0]["ran"] is True
    assert recorded[0]["segment_count"] == body.get("segment_count")


def test_memorize_tail_retries_consolidation_when_pending_segments_exist(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    recorded: list[dict] = []

    async def fake_consolidation_task(
        _svc,
        *,
        conversation_id: str,
        soul_id: str,
        uid: str,
        progress_key: str | None = None,
        memorize_progress: dict | None = None,
        release_memorize_claim: bool = False,
    ) -> None:
        recorded.append({"conversation_id": conversation_id, "soul_id": soul_id, "uid": uid})
        assert release_memorize_claim
        await main_module._finish_memorize_claim(main_module._memorize_lock_key(uid, soul_id), True)

    async def fake_run_memorize_segments(**_kwargs) -> None:  # pragma: no cover - should not run in this branch
        raise AssertionError("run_memorize_segments should not run when no tail segments exist")

    class _FakeSvc:
        pass

    def fake_write_conversation_state(*_args, **_kwargs):
        return (
            {
                "last_memorize_at": "2026-05-09T00:00:00+00:00",
                "digest_cursor": 1,  # with 2-message input below, tail is empty
                "pending_segment_ids": ["chat:cid-2:0-1"],
            },
            tmp_path / "Echo.db",
        )

    monkeypatch.setattr(main_module, "_get_service_from_payload", lambda *a, **k: _FakeSvc())
    monkeypatch.setattr(main_module._memorize_endpoint, "run_memorize_segments", fake_run_memorize_segments)
    monkeypatch.setattr(main_module, "_run_consolidation_task", fake_consolidation_task)
    monkeypatch.setattr(main_module, "_write_conversation_state", fake_write_conversation_state)
    monkeypatch.setattr(main_module, "_get_storage_dir", lambda _cfg: tmp_path)

    payload = {
        "user": {"user_id": "test_user", "soul_id": "test_soul", "conversation_id": "chat:cid-2"},
        "conversation_id": "chat:cid-2",
        "conversation": [
            {"role": "user", "name": "test_user", "content": "hello"},
            {"role": "assistant", "name": "test_soul", "content": "hi"},
        ],
        "llm_profiles": {"default": {"provider": "openai", "api_key": "x", "base_url": "x", "chat_model": "x", "embed_model": "x"}},
    }

    resp = client.post("/memorize?tail=true", json=payload)
    assert resp.status_code == 202, f"unexpected status: {resp.status_code} body={resp.text[:300]}"
    body = resp.json()
    assert body.get("pending_segment_retry") is True
    assert recorded == [{"conversation_id": "chat:cid-2", "soul_id": "test_soul", "uid": "test_user"}]


def test_memorize_tail_nothing_to_memorize_sets_terminal_progress(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _FakeSvc:
        pass

    def fake_write_conversation_state(*_args, **_kwargs):
        return (
            {
                "last_memorize_at": "2026-05-09T00:00:00+00:00",
                "digest_cursor": 1,  # with 2-message input below, tail is empty
                "pending_segment_ids": [],
            },
            tmp_path / "Echo.db",
        )

    monkeypatch.setattr(main_module, "_get_service_from_payload", lambda *a, **k: _FakeSvc())
    monkeypatch.setattr(main_module, "_write_conversation_state", fake_write_conversation_state)
    monkeypatch.setattr(main_module, "_get_storage_dir", lambda _cfg: tmp_path)

    payload = {
        "user": {"user_id": "test_user", "soul_id": "test_soul", "conversation_id": "chat:cid-3"},
        "conversation_id": "chat:cid-3",
        "conversation": [
            {"role": "user", "name": "test_user", "content": "hello"},
            {"role": "assistant", "name": "test_soul", "content": "hi"},
        ],
        "llm_profiles": {"default": {"provider": "openai", "api_key": "x", "base_url": "x", "chat_model": "x", "embed_model": "x"}},
    }

    resp = client.post("/memorize?tail=true", json=payload)
    assert resp.status_code == 200, f"unexpected status: {resp.status_code} body={resp.text[:300]}"
    body = resp.json()
    assert body.get("status") == "nothing_to_memorize"

    progress = client.get("/memorize/progress?user_id=test_user&soul_id=test_soul")
    assert progress.status_code == 200
    out = progress.json()
    assert out.get("active") is False
    assert out.get("last_result") == "nothing_to_memorize"


# ---- force vs rebuild split regression tests ----

def _make_stub_payload(uid: str, soul_id: str, cid: str) -> dict:
    return {
        "user": {"user_id": uid, "soul_id": soul_id, "conversation_id": cid},
        "conversation_id": cid,
        "conversation": [
            {"role": "user", "name": uid, "content": "hello"},
            {"role": "assistant", "name": soul_id, "content": "hi"},
        ],
        "llm_profiles": {"default": {"provider": "openai", "api_key": "x", "base_url": "x", "chat_model": "x", "embed_model": "x"}},
    }


async def _noop_run(**kwargs) -> None:
    pass


def test_cross_manifest_ranges_are_non_rebuildable_and_non_overlapping() -> None:
    existing = [{"start": 0, "end": 4}, {"start": 3, "end": 7}]
    start = main_module._memorize_endpoint._next_manifest_start(existing)
    cross_segments = main_module._memorize_endpoint._offset_memorize_segments(
        [("memory://cross", [{"content": "a"}, {"content": "b"}, {"content": "c"}], 0, 2)],
        start=start,
    )

    out = main_module._memorize_endpoint._merge_manifest_segments(
        existing,
        cross_segments,
        rebuildable=False,
    )

    assert cross_segments[0][2:] == (0, 2, (8, 10))
    assert out == [
        {"start": 0, "end": 7},
        {"start": 8, "end": 10, "rebuildable": False},
    ]
    assert main_module._memorize_endpoint._canonical_manifest_segments(out) == [
        {"start": 0, "end": 7},
    ]


def test_context_only_units_do_not_consume_durable_segment_coordinates() -> None:
    segments = main_module._memorize_endpoint._offset_memorize_segments(
        [
            ("memory://context", [{"content": "listen", "memorize_chat": False}], 8, 8),
            ("memory://active", [{"content": "remember", "memorize_chat": True}], 9, 9),
        ],
        start=8,
    )

    assert segments[0][4] is None
    assert segments[1][4] == (8, 8)
    assert main_module._memorize_endpoint._merge_manifest_segments(
        [{"start": 0, "end": 7}],
        segments,
        rebuildable=False,
    ) == [
        {"start": 0, "end": 7},
        {"start": 8, "end": 8, "rebuildable": False},
    ]


def test_failed_run_releases_only_its_manifest_range(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"segments": [{"start": 0, "end": 12}]}),
        encoding="utf-8",
    )

    main_module._memorize_endpoint._remove_manifest_ranges(manifest_path, [(8, 10)])

    assert json.loads(manifest_path.read_text(encoding="utf-8"))["segments"] == [
        {"start": 0, "end": 7},
        {"start": 11, "end": 12},
    ]


def test_message_ts_ms_accepts_received_at_for_segment_filenames() -> None:
    ts_ms = main_module._memorize_endpoint.message_ts_ms(
        {"role": "user", "content": "hello", "received_at": "2026-04-16T12:00:00Z"}
    )

    assert ts_ms == int(datetime(2026, 4, 16, 12, 0, tzinfo=UTC).timestamp() * 1000)


def test_cross_memorize_endpoint_marks_manifest_range(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    recorded: list[tuple[int, int]] = []

    async def fake_run(**kwargs) -> None:
        recorded.extend(
            durable_range
            for _url, _messages, _source_start, _source_end, durable_range in kwargs.get("memorize_segments") or []
            if durable_range is not None
        )

    def fake_write_conversation_state(*_a, **_k):
        return ({"digest_cursor": 7, "last_memorize_at": "2026-06-14T00:00:00+00:00", "pending_segment_ids": []}, tmp_path / "state.db")

    monkeypatch.setattr(main_module, "_get_service_from_payload", lambda *a, **k: object())
    monkeypatch.setattr(main_module._memorize_endpoint, "run_memorize_segments", fake_run)
    monkeypatch.setattr(main_module, "_write_conversation_state", fake_write_conversation_state)
    monkeypatch.setattr(main_module, "_get_storage_dir", lambda _cfg: tmp_path)

    chats_dir = tmp_path / "st_chats"
    chat_dir, _chat_key, _source = main_module._memorize_endpoint.resolve_chat_storage_dir(
        chats_dir,
        "u1",
        "TestSoul",
        "chat:cid-cross",
    )
    chat_dir.mkdir(parents=True)
    (chat_dir / "manifest.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 4}, {"start": 5, "end": 7}]}),
        encoding="utf-8",
    )

    payload = _make_stub_payload("u1", "TestSoul", "chat:cid-cross")
    payload["_cross_memorize"] = True
    payload["conversation"] = [
        {"role": "user", "name": "u1", "content": "one"},
        {"role": "assistant", "name": "TestSoul", "content": "two"},
        {"role": "user", "name": "u1", "content": "three"},
    ]
    for index, row in enumerate(payload["conversation"], 8):
        row.update(source_conversation_id="chat:cid-cross", source_conversation_index=index)
    payload["_final_cursors"] = {"chat:cid-cross": {"cursor": 10, "memory_producing": True}}

    resp = client.post("/memorize", json=payload)

    assert resp.status_code == 202, f"status={resp.status_code} body={resp.text[:300]}"
    assert recorded == [(8, 10)]
    manifest = json.loads((chat_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["segments"] == [
        {"start": 0, "end": 7},
        {"start": 8, "end": 10, "rebuildable": False},
    ]

    payload["conversation"] = [
        {"role": "user", "name": "u1", "content": "listen", "memorize_chat": False,
         "source_conversation_id": "chat:cid-cross", "source_conversation_index": 11},
    ]
    payload["_final_cursors"] = {"chat:cid-cross": {"cursor": 11, "memory_producing": False}}
    resp = client.post("/memorize", json=payload)

    assert resp.status_code == 200
    assert resp.json()["status"] == "nothing_to_memorize"
    assert recorded == [(8, 10)]
    assert json.loads((chat_dir / "manifest.json").read_text(encoding="utf-8"))["segments"] == manifest["segments"]


def test_force_without_rebuild_does_not_archive_db(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """force=True (turn-valve path) must NOT rename / archive the DB file."""
    db_file = tmp_path / "TestSoul.db"
    db_file.write_text("fake-db")
    recorded: list[dict] = []

    def fake_sqlite_current_path(uid: str, soul_id: str):
        return db_file

    def fake_write_conversation_state(*_a, **_k):
        return (_state_response(), tmp_path / "fake_state.db")

    async def fake_run(**kwargs) -> None:
        recorded.append({"segment_count": len(kwargs.get("memorize_segments") or [])})

    monkeypatch.setattr(main_module, "_get_service_from_payload", lambda *a, **k: object())
    monkeypatch.setattr(main_module._memorize_endpoint, "run_memorize_segments", fake_run)
    monkeypatch.setattr(main_module, "_sqlite_current_path", fake_sqlite_current_path)
    monkeypatch.setattr(main_module, "_write_conversation_state", fake_write_conversation_state)
    monkeypatch.setattr(main_module, "_get_storage_dir", lambda _cfg: tmp_path)

    resp = client.post("/memorize?force=true", json=_make_stub_payload("u1", "TestSoul", "chat:cid-f"))
    assert resp.status_code == 202, f"status={resp.status_code} body={resp.text[:300]}"
    assert resp.json()["segment_count"] == 1
    assert recorded == [{"segment_count": 1}]
    # DB file must still exist — force-only must not rename it.
    assert db_file.exists(), "force=True without rebuild must NOT archive the DB file"
    bak_files = list(tmp_path.glob("*.bak-*"))
    assert not bak_files, f"force=True without rebuild must not create .bak files, got: {bak_files}"

    manifests = list((tmp_path / "st_chats").glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["source"]["conversation_id"] == "chat:cid-f"
    assert manifest["segments"] == [{"start": 0, "end": 1}]


def test_rebuild_rejected_during_same_soul_consolidation(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    db_file = tmp_path / "TestSoul.db"
    db_file.write_bytes(b"original database")
    original_files = set(tmp_path.iterdir())
    monkeypatch.setattr(main_module, "_CONSOLIDATION_RUNNING", {("u1", "TestSoul"): False})
    monkeypatch.setattr(main_module, "_sqlite_current_path", lambda *_args: db_file)

    def unexpected_service(*_args):
        pytest.fail("blocked rebuild must not acquire or clear a service")

    monkeypatch.setattr(main_module, "_get_service_from_payload", unexpected_service)
    monkeypatch.setattr(main_module, "_clear_cached_services", unexpected_service)
    resp = client.post("/memorize?rebuild=true", json=_make_stub_payload("u1", "TestSoul", "chat:cid-r"))

    assert resp.status_code == 409
    assert resp.json()["detail"] == "cannot rebuild during consolidation"
    assert db_file.read_bytes() == b"original database"
    assert set(tmp_path.iterdir()) == original_files


@pytest.mark.asyncio
async def test_rebuild_rechecks_consolidation_after_waiting_for_lock(monkeypatch, tmp_path):
    db_file = tmp_path / "TestSoul.db"
    db_file.write_bytes(b"original database")
    running = {}
    lock = asyncio.Lock()
    monkeypatch.setattr(main_module, "_CONSOLIDATION_RUNNING", running)
    monkeypatch.setattr(main_module, "_get_memorize_lock", lambda *_args: lock)
    monkeypatch.setattr(main_module, "_sqlite_current_path", lambda *_args: db_file)

    await lock.acquire()
    task = asyncio.create_task(main_module.memorize(
        _make_stub_payload("u1", "TestSoul", "chat:cid-r"), BackgroundTasks(), rebuild=True,
    ))
    await asyncio.sleep(0)
    assert not task.done()
    running[("u1", "TestSoul")] = False
    lock.release()
    with pytest.raises(HTTPException) as exc:
        await task
    assert exc.value.status_code == 409
    assert db_file.read_bytes() == b"original database"
    assert not list(tmp_path.glob("*.bak-*"))


@pytest.mark.parametrize("running", [{}, {("u2", "TestSoul"): False}, {("u1", "OtherSoul"): False}])
@pytest.mark.parametrize("collision_suffix", [None, "", "-wal", "-shm"])
def test_rebuild_archives_db_and_service_reacquired_after_archive(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    running,
    collision_suffix,
    capsys,
) -> None:
    """rebuild=True must archive the DB and re-acquire the service AFTER the archive."""
    db_file = tmp_path / "TestSoul.db"
    souls.publish_soul_db(db_file, "DifferentHuman")
    with sqlite3.connect(db_file) as con:
        con.execute("CREATE TABLE rebuild_sentinel (value TEXT)")
        con.execute("INSERT INTO rebuild_sentinel VALUES ('original data')")
    con.close()
    original_bytes = db_file.read_bytes()

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 10, 12, 0, 0, tzinfo=tz)

    archive_ids = iter((UUID(int=1), UUID(int=2)))
    monkeypatch.setattr(memorize_endpoint, "datetime", FrozenDateTime)
    monkeypatch.setattr(memorize_endpoint, "uuid4", lambda: next(archive_ids))
    monkeypatch.setattr(main_module, "_CONSOLIDATION_RUNNING", running)

    call_order: list[str] = []
    publish = memorize_endpoint.publish_soul_db

    def record_publish(path, user_name):
        call_order.append("published")
        return publish(path, user_name)

    monkeypatch.setattr(memorize_endpoint, "publish_soul_db", record_publish)

    def fake_sqlite_current_path(uid: str, soul_id: str):
        return db_file

    def fake_clear_cached_services() -> None:
        call_order.append("cleared")

    def fake_get_service(payload):
        call_order.append("get_service")
        return object()

    def fake_write_conversation_state(*_a, **_k):
        return (_state_response(), tmp_path / "fake_state.db")

    monkeypatch.setattr(main_module, "_get_service_from_payload", fake_get_service)
    monkeypatch.setattr(main_module._memorize_endpoint, "run_memorize_segments", _noop_run)
    monkeypatch.setattr(main_module, "_sqlite_current_path", fake_sqlite_current_path)
    monkeypatch.setattr(main_module, "_clear_cached_services", fake_clear_cached_services)
    monkeypatch.setattr(main_module, "_write_conversation_state", fake_write_conversation_state)
    monkeypatch.setattr(main_module, "_get_storage_dir", lambda _cfg: tmp_path)

    archives = []
    for index in (1, 2):
        source_bytes = db_file.read_bytes()
        sidecars = {suffix: f"rebuild {index}{suffix}".encode() for suffix in ("-wal", "-shm")}
        for suffix, contents in sidecars.items():
            db_file.with_name(db_file.name + suffix).write_bytes(contents)
        archive = db_file.with_suffix(f".bak-261010-120000-{UUID(int=index).hex}")
        if collision_suffix is not None:
            collision = archive.with_name(archive.name + collision_suffix)
            collision.write_bytes(b"existing backup")

        resp = client.post("/memorize?rebuild=true", json=_make_stub_payload("u1", "TestSoul", "chat:cid-r"))
        if collision_suffix is not None:
            assert resp.status_code == 500, resp.text
            error = capsys.readouterr().err
            assert "FileExistsError" in error
            assert str(collision) in error
            for suffix in ("", "-wal", "-shm"):
                if suffix == collision_suffix:
                    break
                assert archive.with_name(archive.name + suffix).samefile(db_file.with_name(db_file.name + suffix))
            assert collision.read_bytes() == b"existing backup"
            assert db_file.read_bytes() == original_bytes
            for suffix, contents in sidecars.items():
                assert db_file.with_name(db_file.name + suffix).read_bytes() == contents
            assert call_order == []
            return

        assert resp.status_code in (200, 202), f"status={resp.status_code} body={resp.text[:300]}"
        assert archive.read_bytes() == source_bytes
        for suffix, contents in sidecars.items():
            assert archive.with_name(archive.name + suffix).read_bytes() == contents
            assert not db_file.with_name(db_file.name + suffix).exists()
        assert souls.read_soul_name(db_file) == souls.read_soul_name(archive) == "DifferentHuman"
        assert call_order == ["published", "cleared", "get_service"] * index
        archives.append(archive)

    assert archives[0].read_bytes() == original_bytes
    assert archives[0].with_name(archives[0].name + "-wal").read_bytes() == b"rebuild 1-wal"
    assert archives[0].with_name(archives[0].name + "-shm").read_bytes() == b"rebuild 1-shm"
    assert set(tmp_path.glob("TestSoul.bak-*")) == {
        path.with_name(path.name + suffix) for path in archives for suffix in ("", "-wal", "-shm")
    }


def test_rebuild_preserves_display_name_with_real_service_reacquisition(client, monkeypatch):
    owner.create_owner(main_module._CONFIG, "TestOwner")
    souls.create_soul(main_module._CONFIG, souls.SoulCreate(soul_id="TestSoul", use_existing=False, user_name="DifferentHuman"))
    db_path = main_module._sqlite_current_path("TestOwner", "TestSoul")
    monkeypatch.setattr(main_module._memorize_endpoint, "run_memorize_segments", _noop_run)
    response = client.post("/memorize?rebuild=true", json=_make_stub_payload("TestOwner", "TestSoul", "chat:rebuild"))
    assert response.status_code == 202, response.text
    assert souls.read_soul_name(db_path) == "DifferentHuman"
    assert len(list(db_path.parent.glob("TestSoul.bak-*"))) == 1
    service = main_module._get_service_from_payload({"user": {"user_id": "TestOwner", "soul_id": "TestSoul"}})
    assert service.user_config.user_name == "DifferentHuman"


@pytest.mark.parametrize("positions", [(0, 1, 2, 3), (0, 3, 7, 9)])
def test_rebuild_includes_open_sleep_tail_with_original_positions_and_name(client, monkeypatch, positions):
    uid, sid, cid = "TestOwner", "TestSoul", "chat:rebuild-sleep-tail"
    owner.create_owner(main_module._CONFIG, uid)
    souls.create_soul(main_module._CONFIG, souls.SoulCreate(soul_id=sid, use_existing=False, user_name="DifferentHuman"))
    db_path = main_module._sqlite_current_path(uid, sid)
    main_module._write_conversation_state(cid, user_id=uid, soul_id=sid, updates={
        "digest_cursor": positions[-1], "last_memorize_at": "2026-10-02T10:01:00+00:00",
    })
    times = ["2026-10-01T20:00:00+00:00", "2026-10-01T20:01:00+00:00",
             "2026-10-02T10:00:00+00:00", "2026-10-02T10:01:00+00:00"]
    rows = [{"role": "user" if i % 2 == 0 else "soul",
             "name": "DifferentHuman" if i % 2 == 0 else sid, "content": f"fictional saved row {i}",
             "source_conversation_index": position, "source_message_id": str(position),
             "ts_ms": int(datetime.fromisoformat(timestamp).timestamp() * 1000)}
            for i, (position, timestamp) in enumerate(zip(positions, times, strict=True))]
    monkeypatch.setattr(memorize_endpoint, "server_timezone", lambda: UTC)
    monkeypatch.setattr(main_module, "_MIN_CHUNK_TOKENS", 1)
    monkeypatch.setattr(main_module, "_SLEEP_SPLIT_MIN_LULL_SECONDS", 3600)
    assert memorize_endpoint.split_indices_by_sleep(rows, UTC, 3600)[0] == [2]
    recorded = []

    async def capture(**kwargs):
        assert kwargs["svc"].user_config.user_name == "DifferentHuman"
        recorded.extend(row for segment in kwargs["memorize_segments"] for row in segment[1])
        return True

    monkeypatch.setattr(memorize_endpoint, "run_memorize_segments", capture)
    payload = _make_stub_payload(uid, sid, cid)
    payload.update(conversation=rows, intent="manual", source_message_count=positions[-1] + 1, chat_name=sid)
    response = client.post("/memorize?rebuild=true", json=payload)
    assert response.status_code == 202, response.text
    assert response.json()["expected_cursor"] == positions[-1]
    for field in ("content", "name", "ts_ms", "source_conversation_index", "source_message_id"):
        assert [row[field] for row in recorded] == [row[field] for row in rows]
    archive, = db_path.parent.glob("TestSoul.bak-*")
    assert souls.read_soul_name(archive) == souls.read_soul_name(db_path) == "DifferentHuman"


def test_rebuild_refuses_missing_binding_before_archiving(client, monkeypatch, tmp_path):
    path = tmp_path / "Unbound.db"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE existing (value TEXT)")
    con.execute("PRAGMA journal_mode=WAL")
    con.close()
    before = path.read_bytes()
    monkeypatch.setattr(main_module, "_sqlite_current_path", lambda *_args: path)
    monkeypatch.setattr(main_module, "_get_service_from_payload", lambda *_args: pytest.fail("service acquired after missing binding"))
    response = client.post("/memorize?rebuild=true", json=_make_stub_payload("TestOwner", "Unbound", "chat:unbound"))
    assert response.status_code == 409
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.bak-*"))
    assert not list(tmp_path.glob("*-wal")) and not list(tmp_path.glob("*-shm"))


@pytest.mark.asyncio
async def test_frozen_preparation_uses_mutated_rows_under_existing_lock(monkeypatch, tmp_path):
    original_files = set(tmp_path.iterdir())
    lock = asyncio.Lock()
    bounds = {"chat:frozen": {"end": 99, "ceiling": 99}}
    progress = {"owner:soul": {"_st_cursor_ceiling": bounds}}
    payload = _make_stub_payload("TestOwner", "TestSoul", "chat:frozen")
    payload.update(_cross_memorize=True, _st_cursor_ceiling=bounds)
    retained = [{"role": "user", "content": "captured survivor", "source_conversation_index": 7}]
    final = {"chat:frozen": {"cursor": 7, "memory_producing": True}}
    recorded = []

    def prepare(safe, uid, sid):
        assert lock.locked()
        assert (uid, sid) == ("TestOwner", "TestSoul")
        assert safe["_st_cursor_ceiling"] is bounds
        assert set(tmp_path.iterdir()) == original_files
        safe["conversation"] = retained
        safe["_final_cursors"] = final
        return True

    async def run(**kwargs):
        recorded.append(kwargs)

    base = replace(main_module._make_memorize_context(),
                   get_memorize_lock=lambda _key: lock, memorize_lock_key=lambda *_args: "owner:soul",
                   memorize_progress=progress,
                   write_conversation_state=lambda *_args, **_kwargs: (_state_response(), tmp_path / "state.db"))
    endpoint = replace(main_module._make_memorize_endpoint_context(), base=base,
                       safe_payload=lambda value: value, get_service_from_payload=lambda _safe: object(),
                       get_storage_dir=lambda _cfg: tmp_path, run_memorize_segments=run,
                       prepare_frozen_payload=prepare)
    tasks = BackgroundTasks()
    await lock.acquire()
    pending = asyncio.create_task(memorize_endpoint.memorize_endpoint(payload, tasks, True, endpoint_ctx=endpoint))
    await asyncio.sleep(0)
    assert not pending.done()
    bounds["chat:frozen"]["ceiling"] = 49
    lock.release()
    response = await pending
    assert response.status_code == 202
    assert response.background is tasks
    await tasks()
    assert recorded[0]["memorize_segments"][0][1][0]["content"] == "captured survivor"
    assert recorded[0]["memorize_segments"][0][1][0]["source_conversation_index"] == 7
    assert recorded[0]["final_cursors"] is final
    assert recorded[0]["safe"]["_st_cursor_ceiling"] is bounds
    assert progress["owner:soul"]["_st_cursor_ceiling"] is bounds


@pytest.mark.asyncio
@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("unresolved", [False, True])
@pytest.mark.parametrize("empty_original", [False, True])
async def test_frozen_no_work_has_no_endpoint_mutations(tmp_path, automatic, unresolved, empty_original):
    original_files = set(tmp_path.iterdir())
    payload = _make_stub_payload("TestOwner", "TestSoul", "chat:empty")
    payload.update(_cross_memorize=True, _automatic_memorize=automatic)
    if empty_original:
        payload["conversation"] = []
    key = main_module._memorize_lock_key("TestOwner", "TestSoul")
    bounds, processed = {}, []
    progress = {key: {"active": True, "phase": "accepted", "_st_cursor_ceiling": bounds, "_processed_rows": processed}}
    original_progress = dict(progress[key])

    def prepare(safe, _uid, _sid):
        safe["conversation"] = []
        if unresolved:
            raise HTTPException(status_code=409, detail="Unfinished source checkpoints")
        return False

    def forbidden(*_args, **_kwargs):
        pytest.fail("empty frozen work must not acquire services, write state, or normalize")

    base = replace(main_module._make_memorize_context(), memorize_progress=progress,
                   get_memorize_lock=lambda _key: asyncio.Lock(), write_conversation_state=forbidden)
    endpoint = replace(main_module._make_memorize_endpoint_context(), base=base,
                       safe_payload=lambda value: value, get_service_from_payload=forbidden,
                       normalize_conversation=forbidden, get_storage_dir=lambda _cfg: tmp_path,
                       prepare_frozen_payload=prepare)
    tasks = BackgroundTasks()
    if unresolved:
        with pytest.raises(HTTPException, match="Unfinished source checkpoints"):
            await memorize_endpoint.memorize_endpoint(payload, tasks, True, endpoint_ctx=endpoint)
        assert progress[key] == original_progress
    else:
        response = await memorize_endpoint.memorize_endpoint(payload, tasks, True, endpoint_ctx=endpoint)
        status = "deferred" if automatic else "nothing_to_memorize"
        assert json.loads(response.body)["status"] == status
        public = memorize_endpoint.memorize_progress_endpoint(
            "TestOwner", "TestSoul", memorize_lock_key=base.memorize_lock_key, memorize_progress=progress,
        )
        assert public["active"] is False
        assert public["last_result"] == status
        assert "phase" not in public
        assert "_st_cursor_ceiling" not in public
        assert "_processed_rows" not in public
        assert progress[key]["_st_cursor_ceiling"] is bounds
        assert progress[key]["_processed_rows"] is processed
    assert tasks.tasks == []
    assert set(tmp_path.iterdir()) == original_files


@pytest.mark.asyncio
async def test_historical_endpoint_bypasses_frozen_preparation(tmp_path):
    payload = _make_stub_payload("TestOwner", "TestSoul", "import:test")
    imported = {"history_end_index": 2, "memorize_cursor": -1, "pending_segment_ids": []}
    base = replace(main_module._make_memorize_context(),
                   get_memorize_lock=lambda _key: asyncio.Lock(), memorize_progress={},
                   write_conversation_state=lambda *_args, **_kwargs: ({"import_state": imported}, tmp_path / "state.db"))

    def forbidden(*_args):
        pytest.fail("historical preparation must not use ordinary frozen admission")

    endpoint = replace(main_module._make_memorize_endpoint_context(), base=base,
                       safe_payload=lambda value: value, get_service_from_payload=lambda _safe: object(),
                       get_storage_dir=lambda _cfg: tmp_path, prepare_frozen_payload=forbidden)
    tasks = BackgroundTasks()
    response = await memorize_endpoint.memorize_endpoint(payload, tasks, True, endpoint_ctx=endpoint, historical=True)
    assert response.status_code == 202
    assert tasks.tasks[0].kwargs["historical"] is True
    assert tasks.tasks[0].kwargs["memorize_segments"][0][2:4] == (0, 1)


async def _run_disposable_segment(tmp_path, svc, rows, safe, final, *, progress=None):
    progress = {} if progress is None else progress
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    async def no_consolidation(*_args, **_kwargs):
        return {}

    background = set()
    run_ctx = replace(main_module._make_memorize_run_context(),
                      base=replace(main_module._make_memorize_context(), memorize_progress=progress),
                      run_consolidation_task=no_consolidation, background_tasks_set=background)
    producing = any(row.get("memorize_chat") is not False for row in rows)
    result = await memorize_endpoint.run_memorize_segments(
        memorize_segments=[("unused", rows, 0, len(rows) - 1, (0, len(rows) - 1) if producing else None)],
        svc=svc, scope={"user_id": "TestOwner", "soul_id": "TestSoul"},
        conversation_id="chat:frozen", soul_id="TestSoul", uid="TestOwner", processed_cursor=8,
        safe=safe, resource_url=str(tmp_path), chat_key=None, merged_len=len(rows), force=True,
        sleep_stats=None, segments_dir=segments_dir, cross_memorize=True, final_cursors=final, run_ctx=run_ctx,
    )
    if background:
        await asyncio.gather(*background)
    return result, progress


@pytest.mark.asyncio
@pytest.mark.parametrize("producing", [False, True])
@pytest.mark.parametrize(("ceiling", "newer_cursor", "expected"), [(49, None, 49), (-1, None, -1), (49, 70, 70)])
async def test_active_rewind_ceiling_survives_regrowth_and_committed_progress(tmp_path, producing, ceiling, newer_cursor, expected):
    cid = "chat:frozen"
    field = "digest_cursor" if producing else "rolling_summary_cursor_id"
    stamp = "2026-10-01T00:00:00+00:00"
    main_module._write_conversation_state(cid, user_id="TestOwner", soul_id="TestSoul",
                                          updates={field: 8, "last_memorize_at": stamp})
    bounds = {cid: {"end": 99, "ceiling": 99}}
    key = main_module._memorize_lock_key("TestOwner", "TestSoul")
    progress = {key: {"_st_cursor_ceiling": bounds}}
    rows = [{"role": "user", "name": "TestOwner", "content": "original captured text",
             "source_conversation_id": cid, "source_conversation_index": 99, "memorize_chat": producing}]

    class Service(SavedBatchService):
        async def memorize_segments_batch(self, **kwargs):
            original = kwargs["on_segment_saved"]

            def before_commit(session, url, ids):
                original(session, url, ids)
                assert "_processed_rows" not in progress[key]
            kwargs["on_segment_saved"] = before_commit
            return await super().memorize_segments_batch(**kwargs)

        async def extract(self, **kwargs):
            assert json.loads(kwargs["segments"][0]["raw_text"]) == [
                {**row, "name": "TestHuman", "speaker": "TestHuman", "source_owner_id": "TestOwner"}
                for row in rows
            ]
            if producing:
                assert json.loads(Path(kwargs["segments"][0]["local_path"]).read_text()) == rows
            assert progress[key]["_st_cursor_ceiling"] is bounds
            assert "_processed_rows" not in progress[key]
            bounds[cid]["ceiling"] = ceiling
            main_module._write_conversation_state(cid, user_id="TestOwner", soul_id="TestSoul",
                                                  updates={field: ceiling if newer_cursor is None else newer_cursor})
            # Saved length regrows to 60; the old bound stays 49 (or -1 for an empty rewind).
            assert bounds[cid]["end"] == 99
            return [{}]

    service = Service()
    success, progress = await _run_disposable_segment(tmp_path, service, rows, {"_st_cursor_ceiling": bounds},
                                                     {cid: {"cursor": 99, "memory_producing": producing}}, progress=progress)
    assert success
    saved, _, _ = main_module._load_turn_state_and_soul_card(cid, user_id="TestOwner", soul_id="TestSoul")
    # Background state written by the simulated concurrent operation is retained.
    assert saved[field] == expected
    assert saved["digest_cursor"] == (expected if producing else 0)
    assert saved["pending_segment_ids"] == (["chat:frozen:0-0"] if producing else [])
    if producing:
        assert progress[key]["_processed_rows"] == rows
    else:
        assert "_processed_rows" not in progress[key]
    assert progress[key]["_st_cursor_ceiling"] is bounds
    with service._sqlite_write_session(service.database) as session:
        resources = session.connection().exec_driver_sql("SELECT url FROM resources").all()
    assert len(resources) == int(producing)


@pytest.mark.asyncio
async def test_rolled_back_job_does_not_publish_processed_rows(tmp_path):
    key = main_module._memorize_lock_key("TestOwner", "TestSoul")
    progress = {}
    rows = [{"role": "user", "content": "uncommitted text", "source_conversation_id": "chat:frozen",
             "source_conversation_index": 99}]

    class Service(SavedBatchService):
        async def memorize_segments_batch(self, **kwargs):
            original = kwargs["on_segment_saved"]

            def abort(session, url, ids):
                original(session, url, ids)
                assert "_processed_rows" not in progress[key]
                raise RuntimeError("disposable rollback")
            kwargs["on_segment_saved"] = abort
            return await super().memorize_segments_batch(**kwargs)

        async def extract(self, **kwargs):
            return [{}]

    with pytest.raises(RuntimeError, match="disposable rollback"):
        await _run_disposable_segment(tmp_path, Service(), rows, {},
                                     {"chat:frozen": {"cursor": 99, "memory_producing": True}}, progress=progress)
    assert "_processed_rows" not in progress[key]
    assert not list((tmp_path / "segments").iterdir())
    saved, _, _ = main_module._load_turn_state_and_soul_card("chat:frozen", user_id="TestOwner", soul_id="TestSoul")
    assert saved["memorize_failure"]["targets"]["chat:frozen"]["cursor"] == 99
    assert saved["pending_segment_ids"] == []


@pytest.mark.asyncio
async def test_rewind_commit_does_not_shorten_unresolved_failure_targets(tmp_path):
    cid = "chat:frozen"
    main_module._write_conversation_state(cid, user_id="TestOwner", soul_id="TestSoul", updates={
        "digest_cursor": 49, "last_memorize_at": "2026-10-01T00:00:00+00:00",
        "memorize_failure": {"conversation_id": cid, "error": "restore captured source", "paused": True,
                             "targets": {cid: {"cursor": 99, "memory_producing": True}}},
    })
    rows = [{"role": "user", "content": "original captured text", "source_conversation_id": cid,
             "source_conversation_index": 99}]

    class Service(SavedBatchService):
        async def extract(self, **kwargs):
            return [{}]

    service = Service()
    with pytest.raises(RuntimeError, match="unfinished source checkpoints"):
        await _run_disposable_segment(tmp_path, service, rows,
                                     {"_st_cursor_ceiling": {cid: {"end": 99, "ceiling": 49}}},
                                     {cid: {"cursor": 99, "memory_producing": True}})
    saved, _, _ = main_module._load_turn_state_and_soul_card(cid, user_id="TestOwner", soul_id="TestSoul")
    assert saved["digest_cursor"] == 49
    assert saved["memorize_failure"]["targets"][cid]["cursor"] == 99
    assert saved["pending_segment_ids"] == ["chat:frozen:0-0"]
    assert len(list((tmp_path / "segments").iterdir())) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("summary", "updated_at"), [
    ("captured summary", "2026-10-01T00:00:00+00:00"),
    ("new summary", "2026-10-01T00:00:00+00:00"),
    ("captured summary", "2026-10-02T00:00:00+00:00"),
])
async def test_memorize_preserves_rollup_owned_summary_and_cursor(tmp_path, summary, updated_at):
    cid = "chat:background"
    stamp = "2026-10-01T00:00:00+00:00"
    main_module._write_conversation_state(cid, user_id="TestOwner", soul_id="TestSoul", updates={
        "memorize_chat": False, "digest_cursor": 7,
        "rolling_summary": "captured summary", "rolling_summary_updated_at": stamp, "rolling_summary_cursor_id": 8,
    })
    rows = [
        {"role": "user", "content": "primary", "source_conversation_id": "chat:frozen", "source_conversation_index": 0},
        {"role": "user", "content": "background", "source_conversation_id": cid,
         "source_conversation_index": 9, "memorize_chat": False},
    ]

    class Service(SavedBatchService):
        async def extract(self, **kwargs):
            assert kwargs["segments"][0]["segment"]["segment_background_context_rows"][0]["summary"] == "captured summary"
            main_module._write_conversation_state(cid, user_id="TestOwner", soul_id="TestSoul", updates={
                "rolling_summary": summary, "rolling_summary_updated_at": updated_at,
            })
            return [{}]

    success, _ = await _run_disposable_segment(tmp_path, Service(), rows, {"_background_rolling_summaries": {
        cid: {"summary": "captured summary", "updated_at": stamp},
    }}, {"chat:frozen": {"cursor": 0, "memory_producing": True}, cid: {"cursor": 9, "memory_producing": False}})
    saved, _, _ = main_module._load_turn_state_and_soul_card(cid, user_id="TestOwner", soul_id="TestSoul")
    assert success
    assert saved["rolling_summary"] == summary
    assert saved["rolling_summary_updated_at"] == updated_at
    assert saved["rolling_summary_cursor_id"] == 8 and saved["digest_cursor"] == 7
    assert saved["pending_segment_ids"] == []
    primary, _, _ = main_module._load_turn_state_and_soul_card("chat:frozen", user_id="TestOwner", soul_id="TestSoul")
    assert primary["pending_segment_ids"] == ["chat:frozen:0-1"]


def test_empty_rewind_round_trip_keeps_timestamp_and_row_zero_eligible():
    cid = "chat:empty-rewind"
    stamp = "2026-10-01T00:00:00+00:00"
    write = lambda updates: main_module._write_conversation_state(
        cid, user_id="TestOwner", soul_id="TestSoul", updates=updates,
    )[0]
    write({"digest_cursor": 8, "last_memorize_at": stamp})
    empty = write({"digest_cursor": -1, "rolling_summary_cursor_id": -1})
    assert empty["last_memorize_at"] == stamp
    assert empty["digest_cursor"] == empty["rolling_summary_cursor_id"] == -1
    assert state.effective_digest_cursor_from_row(empty) == -1
    saved = write(memorize_endpoint._cursor_updates_for_unit(cursor=0, now_iso=stamp))
    assert state.effective_digest_cursor_from_row(saved) == 0
