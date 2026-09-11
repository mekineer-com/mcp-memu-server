from __future__ import annotations

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException

from app import config, main
from app.db import sqlite_ensure_conversation_state_schema
from app.services import free_turn, owner
from app.services.state import write_conversation_state


def _config(tmp_path):
    return {
        "storage": {
            "sqlite_dir": str(tmp_path),
            "metadata_store": {"dsn": f"sqlite:///{tmp_path / 'memu.db'}"},
        }
    }


def test_owner_is_create_once_and_persists(tmp_path) -> None:
    cfg = _config(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: owner.create_owner(cfg, " Marcos "), range(2)))

    assert sorted(created for _, created in results) == [False, True]
    assert owner.read_owner(cfg) == "Marcos"
    assert owner.require_owner(cfg, "Marcos") == "Marcos"
    with pytest.raises(owner.OwnerMismatchError):
        owner.require_owner(cfg, "marcos")
    assert owner.create_owner(cfg, "Marcos") == ("Marcos", False)
    with pytest.raises(owner.OwnerMismatchError):
        owner.create_owner(cfg, "marcos")

    owner.owner_path(cfg).write_text("\n", encoding="utf-8")
    with pytest.raises(owner.OwnerStorageError):
        owner.read_owner(cfg)


def test_scoped_database_requires_forwarded_owner(tmp_path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    seen: list[str] = []
    monkeypatch.setattr(owner, "require_owner", lambda _cfg, user_id: seen.append(user_id))

    config.sqlite_dsn_for_scope(
        cfg,
        cfg["storage"]["metadata_store"]["dsn"],
        {"user_id": "Marcos", "soul_id": "Codexia"},
    )

    assert seen == ["Marcos"]


def test_scoped_database_rejects_missing_user_before_creation(tmp_path) -> None:
    cfg = _config(tmp_path)
    owner.create_owner(cfg, "user")

    with pytest.raises(owner.OwnerMismatchError, match="valid OpenAlma owner"):
        config.sqlite_dsn_for_scope(
            cfg,
            cfg["storage"]["metadata_store"]["dsn"],
            {"soul_id": "Codexia"},
        )

    assert not (tmp_path / "Codexia.db").exists()


def test_conversation_owner_cannot_be_replaced(tmp_path) -> None:
    db_path = tmp_path / "Codexia.db"
    with sqlite3.connect(db_path) as con:
        sqlite_ensure_conversation_state_schema(con)
    kwargs = {
        "sqlite_current_path": lambda _user, _soul: db_path,
        "soul_id": "Codexia",
        "user_id": "Marcos",
    }
    write_conversation_state("chat:test", **kwargs, updates={"digest_cursor": 1})

    with pytest.raises(HTTPException, match="another owner"):
        write_conversation_state(
            "chat:test",
            **{**kwargs, "user_id": "marcos"},
            updates={"digest_cursor": 2},
        )


def test_deferred_followup_rejects_wrong_owner_before_turn(tmp_path) -> None:
    calls: list[str] = []

    async def should_not_run(*_args, **_kwargs):
        calls.append("turn")
        return {}

    asyncio.run(
        free_turn._run_free_turn_followup(
            {"id": "f1", "user_id": "Wrong", "soul_id": "Codexia"},
            tmp_path / "Codexia.db",
            mark_inflight=lambda _pool, _marker: True,
            free_turn_follow_up_inflight=set(),
            conversation_retrieve=should_not_run,
            conversation_turn=should_not_run,
            build_prompt_override_payload=lambda _result: {},
            insert_whatsapp_outbound=lambda **_kwargs: "",
            mark_free_turn_followup=lambda _path, _id, **kwargs: calls.append(kwargs["status"]),
            clear_inflight=lambda _pool, _marker: None,
            require_owner=lambda _user_id: (_ for _ in ()).throw(owner.OwnerMismatchError("wrong")),
            logger=type("Logger", (), {"exception": lambda *_args, **_kwargs: None})(),
        )
    )

    assert calls == ["failed"]


def test_scheduler_does_not_claim_before_owner_creation(monkeypatch) -> None:
    monkeypatch.setattr(main._owner, "read_owner", lambda _config: None)
    monkeypatch.setattr(
        main._free_turn,
        "_run_due_free_turn_followups_once",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("claimed before onboarding")),
    )

    assert asyncio.run(main._run_due_free_turn_followups_once()) == 0
