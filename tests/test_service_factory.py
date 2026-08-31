import logging

import pytest
from fastapi import HTTPException
from pydantic import BaseModel

from app.config import database_config_from_cfg, default_llm_profiles_from_server_config
from app.services import service_factory


def test_server_config_separates_embedding_provider_and_profile_guard(tmp_path) -> None:
    cfg = {
        "llm": {
            "provider": "openai",
            "api_key": "chat-key",
            "base_url": "https://chat.example/v1",
            "chat_model": "chat-model",
            "embed_model": "legacy-model",
            "embedding": {
                "provider": "gemini",
                "api_key": "embed-key",
                "base_url": "https://generativelanguage.googleapis.com/",
                "embed_model": "gemini-embedding-2",
            },
        },
        "storage": {
            "sqlite_dir": str(tmp_path),
            "metadata_store": {
                "provider": "sqlite",
                "dsn": f"sqlite:///{tmp_path / 'base.db'}",
                "embedding_profile": "gemini-embedding-2:3072",
            },
        },
    }
    profiles = default_llm_profiles_from_server_config(cfg)
    database = database_config_from_cfg(cfg, {"soul_id": "test"})

    assert profiles["default"]["provider"] == "openai"
    assert profiles["embedding"] == {
        "provider": "gemini",
        "api_key": "embed-key",
        "base_url": "https://generativelanguage.googleapis.com/",
        "chat_model": "chat-model",
        "embed_model": "gemini-embedding-2",
        "endpoint_overrides": {},
    }
    assert database["metadata_store"]["embedding_profile"] == "gemini-embedding-2:3072"


def test_embedding_profile_must_match_configured_model(tmp_path) -> None:
    cfg = {
        "llm": {"embedding": {"embed_model": "different-model"}},
        "storage": {
            "metadata_store": {
                "provider": "sqlite",
                "dsn": f"sqlite:///{tmp_path / 'base.db'}",
                "embedding_profile": "gemini-embedding-2:3072",
            }
        },
    }
    with pytest.raises(RuntimeError, match="must match"):
        database_config_from_cfg(cfg, {"soul_id": "test"})


def test_embedding_profile_defaults_to_configured_model(tmp_path) -> None:
    cfg = {
        "llm": {"embedding": {"embed_model": "gemini-embedding-2"}},
        "storage": {
            "metadata_store": {
                "provider": "sqlite",
                "dsn": f"sqlite:///{tmp_path / 'base.db'}",
            }
        },
    }

    database = database_config_from_cfg(cfg, {"soul_id": "test"})

    assert database["metadata_store"]["embedding_profile"] == "gemini-embedding-2:3072"


def test_embedding_profile_requires_a_model(tmp_path) -> None:
    cfg = {"storage": {"metadata_store": {"dsn": f"sqlite:///{tmp_path / 'base.db'}"}}}

    with pytest.raises(RuntimeError, match="embed_model is required"):
        database_config_from_cfg(cfg, {"soul_id": "test"})


def test_validated_step_models_warns_on_unknown_key(caplog: pytest.LogCaptureFixture) -> None:
    llm_profiles = {
        "default": {},
        "preprocess": {},
        "memory_extract": {},
        "category_update": {},
        "reflection": {},
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

        def require_dossier_cutover_ready(self, scope) -> None:
            captured["cutover_scope"] = scope

    class _DummyUserModel(BaseModel):
        text: str = ""

    monkeypatch.setattr(service_factory, "MemoryService", _FakeService)
    service_factory._SERVICES.clear()
    service_factory._SERVICE_STORAGE_FP.clear()

    cfg = {
        "llm": {"step_models": {}},
        "categories": {"category_summary_target_words": 275},
        "memorize": {"background_extra_messages_tokens": 321},
        "claude_code": True,
        "claude_code_model": "claude-opus-4-7",
        "claude_code_effort": "medium",
        "claude_code_permission_mode": "bypassPermissions",
        "claude_code_settings": "/home/marcos/.config/memu/siri-claude-settings.json",
        "claude_code_workspace": "/home/marcos/Desktop/siri",
        "claude_code_timeout_seconds": 3600,
    }

    out = service_factory._get_service_from_payload(
        {
            "user": {"user_id": "u", "soul_id": "echo"},
            "database_config": {"metadata_store": {"dsn": "sqlite:///:memory:"}},
        },
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
        normalize_sqlite_dsn=lambda dsn: dsn,
        sqlite_dsn_for_scope=lambda _cfg, base, _scope: base,
        sqlite_file_from_dsn=lambda _dsn: None,
        extract_scope=lambda payload: payload.get("user"),
        payload_signature=lambda _payload: "sig",
        episodes_per_segment=1,
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
    assert captured["memorize_config"]["background_extra_messages_tokens"] == 321
    assert captured["memorize_config"]["dynamic_category_cluster_size"] == 10
    assert captured["memorize_config"]["category_summary_target_words"] == 275
    assert captured["cutover_scope"] == {"user_id": "u", "soul_id": "echo"}
    assert captured["database_config"]["metadata_store"]["embedding_profile"] == "e:3072"


def test_client_llm_profiles_suppress_server_step_model_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeService:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def require_dossier_cutover_ready(self, scope) -> None:
            captured["cutover_scope"] = scope

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
            "embedding": {"chat_model": "server-default", "embed_model": "server-embed"},
            "memory_extract": {"chat_model": "server-heavy"},
            "reflection": {"chat_model": "server-reflect"},
        },
        database_config_from_cfg=lambda _cfg, scope=None: {"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        blob_config_from_cfg=lambda _cfg: {"resources_dir": "./resources"},
        normalize_sqlite_dsn=lambda dsn: dsn,
        sqlite_dsn_for_scope=lambda _cfg, base, _scope: base,
        sqlite_file_from_dsn=lambda _dsn: None,
        extract_scope=lambda payload: payload.get("user"),
        payload_signature=lambda _payload: "sig-client",
        episodes_per_segment=1,
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


def test_server_step_models_inject_when_client_profiles_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeService:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def require_dossier_cutover_ready(self, scope) -> None:
            captured["cutover_scope"] = scope

    class _DummyUserModel(BaseModel):
        text: str = ""

    monkeypatch.setattr(service_factory, "MemoryService", _FakeService)
    service_factory._SERVICES.clear()
    service_factory._SERVICE_STORAGE_FP.clear()

    out = service_factory._get_service_from_payload(
        {"user": {"user_id": "u", "soul_id": "echo"}},
        config={"llm": {"step_models": {"memory_extract": "server-heavy", "reflection": "server-reflect"}}},
        default_llm_profiles_from_server_config=lambda _cfg: {
            "default": {"chat_model": "server-default"},
            "embedding": {"chat_model": "server-embed", "embed_model": "server-embed"},
            "memory_extract": {"chat_model": "server-heavy"},
            "reflection": {"chat_model": "server-reflect"},
        },
        database_config_from_cfg=lambda _cfg, scope=None: {"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        blob_config_from_cfg=lambda _cfg: {"resources_dir": "./resources"},
        normalize_sqlite_dsn=lambda dsn: dsn,
        sqlite_dsn_for_scope=lambda _cfg, base, _scope: base,
        sqlite_file_from_dsn=lambda _dsn: None,
        extract_scope=lambda payload: payload.get("user"),
        payload_signature=lambda _payload: "sig-server",
        episodes_per_segment=1,
        min_chunk_tokens=4000,
        log_prompts=False,
        prompt_log_before=lambda *a, **k: None,
        prompt_log_after=lambda *a, **k: None,
        prompt_log_on_error=lambda *a, **k: None,
        st_user_model=_DummyUserModel,
        logger=logging.getLogger("test.service_factory"),
    )

    assert isinstance(out, _FakeService)
    assert captured["llm_profiles"]["memory_extract"]["chat_model"] == "server-heavy"
    assert captured["llm_profiles"]["reflection"]["chat_model"] == "server-reflect"
    assert captured["memorize_config"]["memory_extract_llm_profile"] == "memory_extract"
    assert captured["retrieve_config"]["sufficiency_check_llm_profile"] == "reflection"
