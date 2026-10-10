import logging
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import BaseModel

from app.config import database_config_from_cfg, default_llm_profiles_from_server_config
from app.services import owner, service_factory, souls
from app.services.payload import _payload_signature
from app.services.consolidation import consolidation_input_budget
from memu.app.settings import LLMConfig


def test_capacity_follows_effective_model_and_cli_without_inheriting_other_models():
    cfg = {"llm": {"chat_model": "large", "context_window_tokens": 1_000_000, "max_tokens": 100_000,
                   "step_models": {"category_update": "small", "consolidation": "large"},
                   "step_context_window_tokens": {"category_update": 200_000}}}
    profiles = default_llm_profiles_from_server_config(cfg)
    svc = SimpleNamespace(llm_profiles=SimpleNamespace(profiles={
        name: LLMConfig(**profile) for name, profile in profiles.items()
    }))
    assert consolidation_input_budget(svc, None) == 720_000
    assert consolidation_input_budget(svc, "category_update") == 80_000
    assert consolidation_input_budget(svc, "consolidation") == 720_000
    same = service_factory._merge_llm_profiles(profiles, {"default": {"chat_model": "large"}})
    assert same["default"]["context_window_tokens"] == 1_000_000
    changed = service_factory._merge_llm_profiles(profiles, {"default": {"chat_model": "another"}})
    svc.llm_profiles.profiles["default"] = LLMConfig(**changed["default"])
    with pytest.raises(ValueError, match="context_window_tokens is required"):
        consolidation_input_budget(svc, None)
    del cfg["llm"]["step_context_window_tokens"]["category_update"]
    assert "context_window_tokens" not in default_llm_profiles_from_server_config(cfg)["category_update"]
    svc._claude_code = True
    svc._claude_code_model = "cli-model"
    svc._claude_code_context_window_tokens = 200_000
    assert consolidation_input_budget(svc, "category_update") == 160_000
    svc._claude_code_context_window_tokens = None
    with pytest.raises(ValueError, match="cli-model"):
        consolidation_input_budget(svc, None)


def test_storage_fingerprint_treats_omitted_provider_as_sqlite(tmp_path) -> None:
    database = tmp_path / "soul.db"
    database.touch()

    fingerprint = service_factory._service_storage_fingerprint(
        {"metadata_store": {"dsn": f"sqlite:///{database}"}},
        sqlite_file_from_dsn=lambda _dsn: database,
        logger=logging.getLogger("test.service_factory"),
    )

    assert fingerprint["provider"] == "sqlite"
    assert fingerprint["ino"] == database.stat().st_ino


def test_server_config_separates_embedding_provider_and_profile_guard(tmp_path) -> None:
    cfg = {
        "llm": {
            "provider": "openai",
            "api_key": "chat-key",
            "base_url": "https://chat.example/v1",
            "chat_model": "chat-model",
            "embedding": {
                "api_key": "embed-key",
                "base_url": "https://generativelanguage.googleapis.com/",
            },
        },
        "storage": {
            "sqlite_dir": str(tmp_path),
            "metadata_store": {
                "provider": "sqlite",
                "dsn": f"sqlite:///{tmp_path / 'base.db'}",
            },
        },
    }
    owner.create_owner(cfg, "test-user")
    souls.publish_soul_db(tmp_path / "test.db")
    profiles = default_llm_profiles_from_server_config(cfg)
    database = database_config_from_cfg(cfg, {"user_id": "test-user", "soul_id": "test"})

    assert profiles["default"]["provider"] == "openai"
    assert "embed_model" not in profiles["default"]
    assert profiles["embedding"] == {
        "provider": "gemini",
        "api_key": "embed-key",
        "base_url": "https://generativelanguage.googleapis.com/",
        "embed_model": "gemini-embedding-2",
        "endpoint_overrides": {},
    }
    assert database["metadata_store"]["embedding_profile"] == "gemini-embedding-2:3072"


def test_embedding_profile_is_managed_by_openalma(tmp_path) -> None:
    cfg = {
        "storage": {
            "metadata_store": {
                "provider": "sqlite",
                "dsn": f"sqlite:///{tmp_path / 'base.db'}",
            }
        },
    }
    owner.create_owner(cfg, "test-user")
    souls.publish_soul_db(tmp_path / "test.db")

    database = database_config_from_cfg(cfg, {"user_id": "test-user", "soul_id": "test"})

    assert database["metadata_store"]["embedding_profile"] == "gemini-embedding-2:3072"


def test_semantic_dedupe_threshold_uses_profile_default_or_operator_override() -> None:
    assert service_factory._semantic_dedupe_threshold("text-embedding-3-large:3072") == 0.89
    assert service_factory._semantic_dedupe_threshold("gemini-embedding-2:3072") == 0.90
    assert service_factory._semantic_dedupe_threshold("gemini-embedding-2:3072", 0.95) == 0.95
    with pytest.raises(HTTPException, match="No semantic dedupe threshold") as exc_info:
        service_factory._semantic_dedupe_threshold("unknown:3072")
    assert exc_info.value.status_code == 503
    with pytest.raises(RuntimeError, match="default.*number"):
        service_factory._semantic_dedupe_threshold("gemini-embedding-2:3072", True)
    assert service_factory._semantic_dedupe_settings(
        "gemini-embedding-2:3072",
        {"semantic_dedupe_enabled": False, "semantic_dedupe_similarity_threshold": True},
    ) == (False, None)
    with pytest.raises(HTTPException, match="No semantic dedupe threshold"):
        service_factory._semantic_dedupe_settings(
            "unknown:3072", {"semantic_dedupe_enabled": False}
        )


def test_entity_similarity_floor_follows_embedding_profile() -> None:
    assert service_factory._entity_similarity_floor("text-embedding-3-large:3072") == 0.30
    assert service_factory._entity_similarity_floor("gemini-embedding-2:3072") == 0.62
    with pytest.raises(HTTPException, match="No entity similarity floor"):
        service_factory._entity_similarity_floor("unknown:3072")


@pytest.mark.parametrize("setting", ["step_models", "step_context_window_tokens"])
def test_validated_step_models_warns_on_unknown_key(caplog: pytest.LogCaptureFixture, setting) -> None:
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
            {"typo_step": "gpt-4o-mini", "preprocess": "gpt-4o-mini"} if setting == "step_models"
            else {"typo_step": 200_000, "preprocess": 200_000},
            llm_profiles=llm_profiles,
            logger=logging.getLogger("test.step_models"),
            setting=setting,
        )

    assert out == {"preprocess": "gpt-4o-mini" if setting == "step_models" else "200000"}
    assert f"ignoring unrecognized llm.{setting} key: typo_step" in caplog.text
    if setting == "step_context_window_tokens":
        for invalid in (0, -1, "not-a-number", True, 1.5):
            with pytest.raises(HTTPException, match="llm.step_context_window_tokens.preprocess must be a positive integer"):
                service_factory._validated_step_models(
                    {"preprocess": invalid}, llm_profiles=llm_profiles,
                    logger=logging.getLogger("test.step_models"), setting=setting,
                )


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


@pytest.mark.parametrize("server_mode,request_mode", [
    (False, None), (True, None), (True, False), (False, True),
])
def test_get_service_from_payload_passes_claude_code_settings(monkeypatch: pytest.MonkeyPatch, server_mode, request_mode) -> None:
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
        "memorize": {
            "background_extra_messages_tokens": 321,
            "semantic_dedupe_enabled": False,
        },
        "claude_code": server_mode,
        "claude_code_model": "claude-opus-4-7",
        "claude_code_context_window_tokens": 200_000,
        "claude_code_effort": "medium",
        "claude_code_permission_mode": "bypassPermissions",
        "claude_code_settings": "/tmp/test-claude-settings.json",
        "claude_code_workspace": "/tmp/test-soul",
        "claude_code_timeout_seconds": 3600,
    }

    payload = {
        "user": {"user_id": "u", "soul_id": "echo"},
        "database_config": {},
        "memorize_config": {"semantic_dedupe_similarity_threshold": "banana"},
        **({"claude_code": request_mode} if request_mode is not None else {}),
    }
    kwargs = dict(
        config=cfg,
        default_llm_profiles_from_server_config=lambda _cfg: {
            "default": {
                "provider": "openai",
                "api_key": "k",
                "base_url": "https://example.com/v1",
                "chat_model": "m",
                "embed_model": "gemini-embedding-2",
            },
            "embedding": {
                "provider": "openai",
                "api_key": "k",
                "base_url": "https://example.com/v1",
                "chat_model": "m",
                "embed_model": "gemini-embedding-2",
            },
        },
        database_config_from_cfg=lambda _cfg, scope=None: {"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        blob_config_from_cfg=lambda _cfg: {"resources_dir": "./resources"},
        normalize_sqlite_dsn=lambda dsn: dsn,
        sqlite_dsn_for_scope=lambda _cfg, base, _scope: base,
        sqlite_file_from_dsn=lambda _dsn: None,
        extract_scope=lambda payload: payload.get("user"),
        payload_signature=_payload_signature,
        min_chunk_tokens=4000,
        log_prompts=False,
        prompt_log_before=lambda *a, **k: None,
        prompt_log_after=lambda *a, **k: None,
        prompt_log_on_error=lambda *a, **k: None,
        st_user_model=_DummyUserModel,
        logger=logging.getLogger("test.service_factory"),
    )
    out = service_factory._get_service_from_payload(payload, **kwargs)

    assert isinstance(out, _FakeService)
    effective_mode = server_mode if request_mode is None else request_mode
    assert captured["claude_code"] is effective_mode
    assert captured["claude_code_model"] == "claude-opus-4-7"
    assert captured["claude_code_context_window_tokens"] == 200_000
    assert captured["claude_code_effort"] == "medium"
    assert captured["claude_code_permission_mode"] == "bypassPermissions"
    assert captured["claude_code_settings"] == "/tmp/test-claude-settings.json"
    assert captured["claude_code_workspace"] == "/tmp/test-soul"
    assert captured["claude_code_timeout_seconds"] == 3600
    assert captured["memorize_config"]["min_chunk_tokens"] == 4000
    assert captured["memorize_config"]["background_extra_messages_tokens"] == 321
    assert captured["memorize_config"]["dynamic_category_cluster_size"] == 10
    assert captured["memorize_config"]["category_summary_target_words"] == 275
    assert captured["memorize_config"]["semantic_dedupe_enabled"] is False
    assert "semantic_dedupe_similarity_threshold" not in captured["memorize_config"]
    assert captured["cutover_scope"] == {"user_id": "u", "soul_id": "echo"}
    assert captured["database_config"]["metadata_store"]["embedding_profile"] == "gemini-embedding-2:3072"

    opposite_payload = {**payload, "claude_code": not effective_mode}
    opposite = service_factory._get_service_from_payload(opposite_payload, **kwargs)
    assert opposite is not out
    assert captured["claude_code"] is not effective_mode
    assert service_factory._get_service_from_payload(opposite_payload, **kwargs) is opposite
    for invalid_mode in (None, "false", 0, 1):
        with pytest.raises(HTTPException, match="claude_code must be a Boolean"):
            service_factory._get_service_from_payload({**payload, "claude_code": invalid_mode}, **kwargs)


def test_changed_endpoint_cannot_inherit_server_credentials():
    defaults = {"default": {"base_url": "https://server.example/v1", "api_key": "server-key"}}
    for override in (
        {"base_url": "https://client.example/v1"},
        {"endpoint_overrides": {"chat": "https://client.example/chat"}},
        {"endpoint_overrides": {"summary": "https://client.example/summary"}},
    ):
        with pytest.raises(HTTPException, match="api_key is required when changing endpoints"):
            service_factory._merge_llm_profiles(defaults, {"default": override})
    with pytest.raises(HTTPException, match="api_key is required when changing endpoints"):
        service_factory._merge_llm_profiles({"default": {
            "provider": "openai", "base_url": "https://api.openai.com/v1", "api_key": "server-key",
        }}, {"default": {"provider": "grok"}})
    assert service_factory._merge_llm_profiles(defaults, {"default": {"chat_model": "client-model"}})["default"]["api_key"] == "server-key"
    for key in ("client-key", ""):
        merged = service_factory._merge_llm_profiles(defaults, {"default": {
            "base_url": "https://client.example/v1", "api_key": key,
        }})
        assert merged["default"]["api_key"] == key


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
                "embedding": {
                    "provider": "openai",
                    "embed_model": "client-embed",
                },
            },
        },
        config={"llm": {"step_models": {"memory_extract": "server-heavy", "reflection": "server-reflect"}}},
        default_llm_profiles_from_server_config=lambda _cfg: {
            "default": {"chat_model": "server-default"},
            "embedding": {"chat_model": "server-default", "embed_model": "gemini-embedding-2"},
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
    assert captured["llm_profiles"]["embedding"]["embed_model"] == "gemini-embedding-2"
    assert captured["retrieve_config"]["graph"]["min_entity_similarity"] == 0.62
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
            "embedding": {"chat_model": "server-embed", "embed_model": "gemini-embedding-2"},
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
