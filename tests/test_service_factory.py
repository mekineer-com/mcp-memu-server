import logging

import pytest
from fastapi import HTTPException
from pydantic import BaseModel

from app.services import service_factory


def test_validated_step_models_warns_on_unknown_key(caplog: pytest.LogCaptureFixture) -> None:
    llm_profiles = {
        "default": {},
        "preprocess": {},
        "memory_extract": {},
        "category_update": {},
        "reflection": {},
        "ranking": {},
        "consolidation": {},
    }
    with caplog.at_level(logging.WARNING):
        out = service_factory._validated_step_models(
            {"typo_step": "gpt-4o-mini", "preprocess": "gpt-4o-mini"},
            llm_profiles=llm_profiles,
            logger=logging.getLogger("test.step_models"),
        )

    assert out == {"preprocess": "gpt-4o-mini"}
    assert "ignoring unrecognized llm.step_models key: typo_step" in caplog.text


def test_validated_step_models_raises_when_profile_missing() -> None:
    llm_profiles = {
        "default": {},
    }
    with pytest.raises(HTTPException, match="llm.step_models.preprocess is configured but profile 'preprocess' is missing"):
        service_factory._validated_step_models(
            {"preprocess": "gpt-4o-mini"},
            llm_profiles=llm_profiles,
            logger=logging.getLogger("test.step_models"),
        )


def test_get_service_from_payload_passes_claude_code_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeService:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    class _DummyUserModel(BaseModel):
        text: str = ""

    monkeypatch.setattr(service_factory, "MemoryService", _FakeService)
    service_factory._SERVICES.clear()
    service_factory._SERVICE_STORAGE_FP.clear()

    cfg = {
        "llm": {"step_models": {}},
        "claude_code": True,
        "claude_code_model": "claude-opus-4-7",
        "claude_code_effort": "medium",
        "claude_code_permission_mode": "bypassPermissions",
        "claude_code_settings": "/home/marcos/.config/memu/siri-claude-settings.json",
        "claude_code_workspace": "/home/marcos/Desktop/siri",
        "claude_code_timeout_seconds": 3600,
    }

    out = service_factory._get_service_from_payload(
        {"user": {"user_id": "u", "soul_id": "echo"}},
        config=cfg,
        default_llm_profiles_from_server_config=lambda _cfg: {
            "default": {
                "provider": "openai",
                "api_key": "k",
                "base_url": "https://example.com/v1",
                "chat_model": "m",
                "embed_model": "e",
            },
            "embedding": {
                "provider": "openai",
                "api_key": "k",
                "base_url": "https://example.com/v1",
                "chat_model": "m",
                "embed_model": "e",
            },
        },
        database_config_from_cfg=lambda _cfg, scope=None: {"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        blob_config_from_cfg=lambda _cfg: {"resources_dir": "./resources"},
        categories_from_cfg=lambda _cfg: [],
        normalize_sqlite_dsn=lambda dsn: dsn,
        sqlite_dsn_for_scope=lambda _cfg, base, _scope: base,
        sqlite_file_from_dsn=lambda _dsn: None,
        extract_scope=lambda payload: payload.get("user"),
        payload_signature=lambda _payload: "sig",
        episode_items_per_segment=1,
        min_chunk_tokens=4000,
        log_prompts=False,
        prompt_log_before=lambda *a, **k: None,
        prompt_log_after=lambda *a, **k: None,
        prompt_log_on_error=lambda *a, **k: None,
        st_user_model=_DummyUserModel,
        logger=logging.getLogger("test.service_factory"),
    )

    assert isinstance(out, _FakeService)
    assert captured["claude_code"] is True
    assert captured["claude_code_model"] == "claude-opus-4-7"
    assert captured["claude_code_effort"] == "medium"
    assert captured["claude_code_permission_mode"] == "bypassPermissions"
    assert captured["claude_code_settings"] == "/home/marcos/.config/memu/siri-claude-settings.json"
    assert captured["claude_code_workspace"] == "/home/marcos/Desktop/siri"
    assert captured["claude_code_timeout_seconds"] == 3600
    assert captured["memorize_config"]["min_chunk_tokens"] == 4000


def test_client_llm_profiles_suppress_server_step_model_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeService:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    class _DummyUserModel(BaseModel):
        text: str = ""

    monkeypatch.setattr(service_factory, "MemoryService", _FakeService)
    service_factory._SERVICES.clear()
    service_factory._SERVICE_STORAGE_FP.clear()

    out = service_factory._get_service_from_payload(
        {
            "user": {"user_id": "u", "soul_id": "echo"},
            "llm_profiles": {
                "default": {
                    "provider": "openai",
                    "api_key": "client-key",
                    "base_url": "https://client.example/v1",
                    "chat_model": "client-model",
                    "embed_model": "client-embed",
                },
            },
        },
        config={"llm": {"step_models": {"memory_extract": "server-heavy", "reflection": "server-reflect"}}},
        default_llm_profiles_from_server_config=lambda _cfg: {
            "default": {"chat_model": "server-default"},
            "embedding": {"chat_model": "server-default"},
            "memory_extract": {"chat_model": "server-heavy"},
            "reflection": {"chat_model": "server-reflect"},
        },
        database_config_from_cfg=lambda _cfg, scope=None: {"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        blob_config_from_cfg=lambda _cfg: {"resources_dir": "./resources"},
        categories_from_cfg=lambda _cfg: [],
        normalize_sqlite_dsn=lambda dsn: dsn,
        sqlite_dsn_for_scope=lambda _cfg, base, _scope: base,
        sqlite_file_from_dsn=lambda _dsn: None,
        extract_scope=lambda payload: payload.get("user"),
        payload_signature=lambda _payload: "sig-client",
        episode_items_per_segment=1,
        min_chunk_tokens=4000,
        log_prompts=False,
        prompt_log_before=lambda *a, **k: None,
        prompt_log_after=lambda *a, **k: None,
        prompt_log_on_error=lambda *a, **k: None,
        st_user_model=_DummyUserModel,
        logger=logging.getLogger("test.service_factory"),
    )

    assert isinstance(out, _FakeService)
    assert captured["llm_profiles"]["default"]["chat_model"] == "client-model"
    assert "memory_extract" not in captured["llm_profiles"]
    assert "reflection" not in captured["llm_profiles"]
    assert "memory_extract_llm_profile" not in captured["memorize_config"]
    assert "sufficiency_check_llm_profile" not in captured["retrieve_config"]
