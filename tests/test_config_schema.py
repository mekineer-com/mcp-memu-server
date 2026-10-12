"""Checks for the installable config example, defaults, and atomic saves."""

from __future__ import annotations

import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from app import config as app_config
from app.config import default_config
import run as server_run

_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLE_CONFIG = _ROOT / "config.example.json"


@pytest.fixture(autouse=True)
def _isolated_config_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_config, "config_path", lambda: tmp_path / "config" / "config.json")


@pytest.mark.parametrize("engine_present", [False, True])
def test_missing_config_bootstraps_repo_local_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine_present: bool,
) -> None:
    home = tmp_path / "home"
    legacy_memu = home / "apps" / "memu"
    legacy_memu.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(home)
    server_dir = app_config.config_dir()
    memu_dir = server_dir.parent / "memu"
    if engine_present:
        (memu_dir / "src").mkdir(parents=True)

    cfg = app_config.load_config()

    assert cfg["storage"]["resources_dir"] == str(memu_dir / "resources")
    assert (memu_dir / "resources").is_dir()
    assert cfg["pid_file"] == str(server_dir / ".memu-server.pid")
    expected_memu = str(memu_dir / "src") if engine_present else ""
    assert cfg["memu"]["path"] == expected_memu
    assert app_config.default_config()["memu"]["path"] == expected_memu
    assert not (legacy_memu / "resources").exists()


def test_example_is_valid_json_and_non_empty() -> None:
    example = json.loads(_EXAMPLE_CONFIG.read_text())
    assert isinstance(example, dict)
    assert example, "config.example.json must be non-empty"
    # setup_install.py fills paths inside these four sections.
    for required in ("llm", "storage", "memu", "listen"):
        assert required in example, f"config.example.json missing top-level '{required}'"
    assert "embed_model" not in example["llm"]["embedding"]
    assert example["storage"]["metadata_store"]["dsn"]


def _mapping_config() -> dict:
    return {"claude_code": False, "llm": {
        "provider": "openai", "base_url": "https://fictional.example/v1", "api_key": "base-secret",
        "chat_model": "custom-base", "context_window_tokens": 100000,
        "embedding": {"api_key": "embedding-secret", "base_url": "https://embedding.example/"},
        "reply_models": {"srmm": {"source": "configured", "selection": {"provider": "openrouter", "model": "fictional-reply"},
            "resolved": {"provider": "openai", "base_url": "https://reply.example/v1", "api_key": "reply-secret",
                         "chat_model": "fictional-reply", "context_window_tokens": 50000, "max_tokens": 2000}},
            "other_apps": {"source": "st_native"}},
        "st_selected": {"selection": {"provider": "openai", "model": "fictional-native"}, "resolved": {
            "provider": "openai", "base_url": "https://native.example/v1", "api_key": "native-secret",
            "chat_model": "fictional-native", "context_window_tokens": 60000}},
        "step_models": {"preprocess": {"source": "config"}, "memory_extract": {"source": "srmm"},
                        "category_update": {"source": "st_native"}, "consolidation": {"source": "srmm"},
                        "reflection": {"source": "config"}},
    }, "storage": {"resources_dir": "untouched"}, "mentra": {"gemini_api_key": "voice-secret", "model": "unchanged-voice"}}


def test_model_mapping_normalization_preserves_scalar_model_and_matching_capacity() -> None:
    cfg = _mapping_config()
    cfg["llm"]["step_models"] = {"memory_extract": "old-model", "preprocess": "another-model"}
    cfg["llm"]["step_context_window_tokens"] = {"memory_extract": 40000}
    normalized = app_config.normalize_model_mappings(cfg)
    assert cfg["llm"]["step_models"]["memory_extract"] == "old-model"
    rows = normalized["llm"]["step_models"]
    assert rows["memory_extract"]["resolved"]["chat_model"] == "old-model"
    assert rows["memory_extract"]["resolved"]["context_window_tokens"] == 40000
    assert "context_window_tokens" not in rows["preprocess"]["resolved"]
    assert rows["reflection"] == {"source": "config"}


def test_load_config_normalizes_without_rewriting_existing_file(monkeypatch) -> None:
    cfg = _mapping_config()
    cfg["llm"]["step_models"] = {"memory_extract": "saved-scalar-model"}
    cfg["llm"]["step_context_window_tokens"] = {"memory_extract": 70000}
    path = app_config.config_path()
    path.parent.mkdir(parents=True)
    original = json.dumps(cfg)
    path.write_text(original)
    monkeypatch.setattr(app_config, "ensure_storage_paths", lambda _cfg: None)
    loaded = app_config.load_config()
    assert loaded["llm"]["step_models"]["memory_extract"]["resolved"]["context_window_tokens"] == 70000
    assert path.read_text() == original


@pytest.mark.parametrize("operation,names", [
    ("st_reply", set()), ("other_reply", set()), ("retrieve", {"reflection"}), ("rollup", {"preprocess"}),
    ("memorize", {"preprocess", "memory_extract", "category_update", "consolidation"}),
    ("consolidation", {"category_update", "consolidation"}), (None, set(app_config.MODEL_STEPS)),
])
def test_model_mapping_operation_profiles(operation, names) -> None:
    cfg = _mapping_config()
    cfg["llm"]["step_context_window_tokens"] = {"memory_extract": 70000, "srmm": 40000}
    profiles = app_config.default_llm_profiles_from_server_config(cfg, operation=operation)
    assert set(profiles) == {"default", "embedding"} | names
    expected = "fictional-reply" if operation == "st_reply" else "fictional-native" if operation == "other_reply" else "custom-base"
    assert profiles["default"]["chat_model"] == expected
    assert profiles["embedding"]["api_key"] == "embedding-secret"
    if "memory_extract" in profiles:
        assert profiles["memory_extract"]["api_key"] == "reply-secret"
        assert profiles["memory_extract"]["context_window_tokens"] == 70000
    if "consolidation" in profiles:
        assert profiles["consolidation"]["context_window_tokens"] == 50000
    if operation == "st_reply":
        assert profiles["default"]["context_window_tokens"] == 50000
    if "category_update" in profiles:
        assert profiles["category_update"]["chat_model"] == "fictional-native"


@pytest.mark.parametrize("operation", [None, "other_reply", "retrieve", "rollup", "memorize", "consolidation"])
def test_cli_skips_unresolved_hidden_api_mappings(operation) -> None:
    cfg = _mapping_config()
    cfg["claude_code"] = True
    del cfg["llm"]["st_selected"]
    cfg["llm"]["step_models"]["preprocess"] = {"source": "configured"}
    assert set(app_config.default_llm_profiles_from_server_config(cfg, operation=operation)) == {"default", "embedding"}
    assert app_config.default_llm_profiles_from_server_config(cfg, operation="st_reply")["default"]["chat_model"] == "fictional-reply"


def test_unused_native_mapping_does_not_block_retrieval_or_normalization() -> None:
    cfg = _mapping_config()
    del cfg["llm"]["st_selected"]
    app_config.normalize_model_mappings(cfg)
    assert app_config.default_llm_profiles_from_server_config(cfg, operation="retrieve")["reflection"]["chat_model"] == "custom-base"
    with pytest.raises(ValueError, match="resolved st_native"):
        app_config.default_llm_profiles_from_server_config(cfg, operation="other_reply")


@pytest.mark.parametrize("capacity", [0, -1, True, "64000"])
def test_selected_model_capacity_must_be_positive_integer(capacity) -> None:
    cfg = _mapping_config()
    cfg["llm"]["reply_models"]["srmm"]["resolved"]["context_window_tokens"] = capacity
    with pytest.raises(ValueError, match="positive integer"):
        app_config.default_llm_profiles_from_server_config(cfg, operation="st_reply")


@pytest.mark.parametrize("source,reply_source", [
    ("config", None), ("configured", None), ("st_native", None),
    ("srmm", "config"), ("srmm", "configured"), ("srmm", "st_native"),
])
@pytest.mark.parametrize("selected_capacity,override", [
    (None, None), (50000, None), (None, 70000), (50000, 70000), (0, 70000),
])
def test_named_step_capacity_overlays_selected_profile_without_leaking(
    source, reply_source, selected_capacity, override,
) -> None:
    cfg = _mapping_config()
    llm = cfg["llm"]
    selected_source = reply_source or source
    if selected_source == "config":
        selected = llm
    elif selected_source == "st_native":
        selected = llm["st_selected"]["resolved"]
    else:
        selected = llm["reply_models"]["srmm"]["resolved"]
    selected.pop("context_window_tokens", None)
    if selected_capacity is not None:
        selected["context_window_tokens"] = selected_capacity
    row = {"source": source}
    if source == "configured":
        row["resolved"] = copy.deepcopy(selected)
    if source == "srmm":
        llm["reply_models"]["srmm"] = {
            "source": reply_source,
            **({"resolved": selected} if reply_source == "configured" else {}),
        }
    llm["step_models"]["preprocess"] = row
    llm["step_context_window_tokens"] = {"preprocess": override}
    original = copy.deepcopy(cfg)
    profiles = app_config.default_llm_profiles_from_server_config(cfg, operation="rollup")
    expected = override if override is not None else selected_capacity
    assert profiles["preprocess"].get("context_window_tokens") == expected
    assert profiles["preprocess"]["chat_model"] == selected["chat_model"]
    assert profiles["preprocess"]["api_key"] == selected["api_key"]
    assert profiles["default"].get("context_window_tokens") == llm.get("context_window_tokens")
    assert cfg == original
    profiles["preprocess"]["api_key"] = "changed-copy"
    assert profiles["default"]["api_key"] != "changed-copy"
    assert cfg == original


@pytest.mark.parametrize("override,expected", [(None, 40000), (70000, 70000)])
def test_named_capacity_preserves_config_source_srmm_capacity_contribution(override, expected) -> None:
    cfg = _mapping_config()
    cfg["llm"]["reply_models"]["srmm"] = {"source": "config"}
    cfg["llm"]["step_context_window_tokens"] = {"srmm": 40000, "memory_extract": override}
    profiles = app_config.default_llm_profiles_from_server_config(cfg, operation="memorize")
    assert profiles["memory_extract"]["context_window_tokens"] == expected
    assert profiles["consolidation"]["context_window_tokens"] == 40000
    reply = app_config.default_llm_profiles_from_server_config(cfg, operation="st_reply")
    assert reply["default"]["context_window_tokens"] == 40000


def test_named_capacity_replaces_invalid_config_source_srmm_contribution() -> None:
    cfg = _mapping_config()
    cfg["llm"]["reply_models"]["srmm"] = {"source": "config"}
    cfg["llm"]["step_models"]["preprocess"] = {"source": "srmm"}
    cfg["llm"]["step_context_window_tokens"] = {"srmm": 0, "preprocess": 70000}
    profiles = app_config.default_llm_profiles_from_server_config(cfg, operation="rollup")
    assert profiles["preprocess"]["context_window_tokens"] == 70000
    assert cfg["llm"]["step_context_window_tokens"]["srmm"] == 0


def test_invalid_named_step_capacity_fails_final_validation() -> None:
    cfg = _mapping_config()
    cfg["llm"]["step_models"]["preprocess"] = {"source": "srmm"}
    cfg["llm"]["step_context_window_tokens"] = {"preprocess": 0}
    with pytest.raises(ValueError, match=r"Model mapping preprocess.context_window_tokens.*positive integer"):
        app_config.default_llm_profiles_from_server_config(cfg, operation="rollup")


def test_named_capacity_does_not_bypass_inherited_output_cap_validation() -> None:
    cfg = _mapping_config()
    cfg["llm"]["reply_models"]["srmm"]["resolved"].update({"context_window_tokens": 0, "max_tokens": 0})
    cfg["llm"]["step_models"]["preprocess"] = {"source": "srmm"}
    cfg["llm"]["step_context_window_tokens"] = {"preprocess": 70000}
    with pytest.raises(ValueError, match=r"Model mapping preprocess.max_tokens.*positive integer"):
        app_config.default_llm_profiles_from_server_config(cfg, operation="rollup")


@pytest.mark.parametrize("field", ["context_window_tokens", "max_tokens"])
@pytest.mark.parametrize("capacity", [0, -1, True, "64000"])
def test_base_model_capacity_must_be_positive_integer(field, capacity) -> None:
    cfg = _mapping_config()
    cfg["llm"]["reply_models"]["srmm"] = {"source": "config"}
    cfg["llm"][field] = capacity
    with pytest.raises(ValueError, match="positive integer"):
        app_config.default_llm_profiles_from_server_config(cfg, operation="st_reply")


def test_effective_generic_api_mode_preserves_steps_under_global_cli() -> None:
    cfg = _mapping_config()
    cfg["claude_code"] = True
    effective = {**cfg, "claude_code": False}
    profiles = app_config.default_llm_profiles_from_server_config(effective)
    assert profiles["memory_extract"]["chat_model"] == "fictional-reply"
    assert cfg["claude_code"] is True


def test_config_model_default_and_native_inheritance_are_not_self_references() -> None:
    cfg = _mapping_config()
    cfg["llm"]["reply_models"]["srmm"] = {"source": "config"}
    profiles = app_config.default_llm_profiles_from_server_config(cfg, operation="memorize")
    assert profiles["memory_extract"]["chat_model"] == "custom-base"
    cfg["llm"]["reply_models"]["srmm"] = {"source": "srmm"}
    with pytest.raises(ValueError, match="advanced steps"):
        app_config.normalize_model_mappings(cfg)


def test_model_config_merge_preserves_secrets_hidden_rows_and_other_settings() -> None:
    current = _mapping_config()
    current["storage"] = {
        "resources_dir": "selected-resources", "sqlite_dir": "selected-sqlite",
        "metadata_store": {"provider": "sqlite", "dsn": "sqlite:///selected.db", "ddl_mode": "create"},
    }
    body = {"llm": {"reply_models": {"srmm": {"source": "config"}}, "embedding": {"base_url": "https://new-embedding.example/"}},
            "claude_code": True}
    result = app_config.merge_model_config(current, body)
    assert result["llm"]["reply_models"]["srmm"] == {"source": "config"}
    assert result["llm"]["reply_models"]["other_apps"] == current["llm"]["reply_models"]["other_apps"]
    assert result["llm"]["step_models"] == current["llm"]["step_models"]
    assert result["llm"]["st_selected"] == current["llm"]["st_selected"]
    assert result["llm"]["api_key"] == "base-secret" and result["llm"]["embedding"]["api_key"] == "embedding-secret"
    assert result["storage"] == current["storage"] and result["mentra"] == current["mentra"]
    assert current["claude_code"] is False and body["llm"]["reply_models"]["srmm"] == {"source": "config"}
    for patch in ({"resources_dir": "new-resources"}, {"metadata_store": {"ddl_mode": "validate"}}):
        updated = app_config.merge_model_config(current, {"storage": patch})["storage"]
        assert updated["sqlite_dir"] == "selected-sqlite"
        assert updated["resources_dir"] == patch.get("resources_dir", "selected-resources")
        assert updated["metadata_store"] == {
            **current["storage"]["metadata_store"], **patch.get("metadata_store", {}),
        }
    for replacement in (None, "", []):
        assert app_config.merge_model_config(current, {"storage": replacement})["storage"] == replacement
        updated = app_config.merge_model_config(current, {"storage": {"metadata_store": replacement}})["storage"]
        assert updated == {**current["storage"], "metadata_store": replacement}
    assert current["storage"]["resources_dir"] == "selected-resources"
    assert current["storage"]["metadata_store"]["ddl_mode"] == "create"


def test_model_mapping_masks_every_nested_profile_without_mutating_config() -> None:
    cfg = _mapping_config()
    masked = app_config.mask_config(cfg)
    assert masked["llm"]["api_key"] == "base...cret"
    for holder in [masked["llm"]["embedding"], masked["llm"]["reply_models"]["srmm"]["resolved"], masked["llm"]["st_selected"]["resolved"]]:
        assert holder["api_key"] == "***"
    assert cfg["llm"]["api_key"] == "base-secret"
    assert app_config.mask_config(cfg, include_llm_secret=True)["llm"]["st_selected"]["resolved"]["api_key"] == "native-secret"
    assert masked["mentra"]["gemini_api_key"] == "***"


@pytest.mark.parametrize("secret", ["***", "abcd...wxyz"])
def test_model_config_rejects_masked_secret_round_trip(secret) -> None:
    cfg = _mapping_config()
    with pytest.raises(ValueError, match="Masked api_key"):
        app_config.merge_model_config(cfg, {"llm": {"reply_models": {"srmm": {"source": "configured", "resolved": {"api_key": secret}}}}})


@pytest.mark.parametrize("cli", [False, True])
def test_model_mapping_rejects_native_noncompatible_protocol(cli) -> None:
    cfg = _mapping_config()
    cfg["claude_code"] = cli
    row = cfg["llm"]["reply_models"]["srmm"]
    row["resolved"]["provider"] = "anthropic"
    with pytest.raises(ValueError, match="OpenAI-compatible"):
        app_config.default_llm_profiles_from_server_config(cfg, operation="st_reply")
    with pytest.raises(ValueError, match="OpenAI-compatible"):
        app_config.merge_model_config(_mapping_config(), {"llm": {"reply_models": {"srmm": row}}})
    with pytest.raises(ValueError, match="OpenAI-compatible"):
        app_config.merge_model_config(cfg, {"llm": {"st_selected": {"resolved": row["resolved"]}}})


def test_default_whatsapp_paths_point_at_channels_data() -> None:
    hermes = default_config()["hermes"]

    assert hermes["home"].endswith("/hermes-channels/data")
    assert hermes["state_db_path"].endswith("/hermes-channels/data/state.db")
    assert hermes["sessions_index_path"].endswith("/hermes-channels/data/sessions/sessions.json")
    assert hermes["whatsapp_web_source_db"].endswith("/hermes-channels/data/whatsapp/web_source.db")
    assert "whatsapp_history_source" not in hermes


def test_default_dynamic_category_cluster_size_matches_engine() -> None:
    assert default_config()["categories"]["dynamic_category_cluster_size"] == 10


def test_default_mentra_contract_is_keyless_and_loopback() -> None:
    config = default_config()
    assert config["listen"]["host"] == "127.0.0.1"
    assert config["mentra"]["enabled"] is False
    assert "integration_bearer_token" not in config["mentra"]
    assert "gemini_api_key" in config["mentra"]


@pytest.mark.parametrize("operation", ["create", "replace", "load"])
def test_config_writes_are_atomic_and_private(operation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    path = app_config.config_path()
    cfg = {"llm": {"api_key": "fictional-key"}, "mentra": {"public_base_url": "http://phone.example.invalid"}}
    expected = app_config.normalize_model_mappings(cfg) if operation == "load" else cfg
    original = b'{"llm": {"api_key": "fictional-old-key"}}'
    if operation == "replace":
        path.parent.mkdir()
        path.write_bytes(original)
        path.chmod(0o644)
    staged_paths = []
    real_replace = os.replace

    def replace(src, dst):
        staged = Path(src)
        assert Path(dst) == path
        assert staged.parent == path.parent
        assert staged.read_text(encoding="utf-8") == json.dumps(expected, indent=2)
        if os.name == "posix":
            assert staged.stat().st_mode & 0o777 == 0o600
        if operation == "replace":
            assert path.read_bytes() == original
        else:
            assert not path.exists()
        staged_paths.append(staged)
        real_replace(src, dst)

    monkeypatch.setattr(app_config.os, "replace", replace)
    if operation == "load":
        monkeypatch.setattr(app_config, "default_config", lambda: cfg)
        assert app_config.load_config() == expected
    else:
        app_config.save_config(cfg)
    assert len(staged_paths) == 1
    assert json.loads(path.read_text(encoding="utf-8")) == expected
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("existing", [False, True])
def test_config_read_preserves_missing_bootstrap_and_valid_objects(existing, monkeypatch):
    path = app_config.config_path()
    cfg = {"llm": {"api_key": "fictional-key"}}
    monkeypatch.setattr(app_config, "default_config", lambda: cfg)
    initialized = []
    monkeypatch.setattr(app_config, "ensure_storage_paths", initialized.append)
    if existing:
        path.parent.mkdir()
        path.write_text(json.dumps(cfg), encoding="utf-8")
    assert server_run._read_json(path) == (cfg if existing else {})
    assert path.exists() is existing
    original = path.read_text() if existing else None
    normalized = app_config.normalize_model_mappings(cfg)
    assert app_config.load_config() == normalized
    assert initialized == [normalized]
    assert server_run._read_json(path) == (cfg if existing else normalized)
    if existing:
        assert path.read_text() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("loader", ["runtime", "runner", "reload"])
@pytest.mark.parametrize("failure", ["malformed", "blank", "non_object", "unreadable", "encoding"])
async def test_invalid_existing_config_fails_without_side_effects(loader, failure, monkeypatch):
    from app import main

    path = app_config.config_path()
    path.parent.mkdir()
    original = {
        "malformed": b'{"llm":', "blank": b" \n", "non_object": b"[]",
        "unreadable": b'{"llm":{"api_key":"fictional-key"}}', "encoding": b"\xff",
    }[failure]
    path.write_bytes(original)
    error = {
        "non_object": ValueError, "unreadable": PermissionError, "encoding": UnicodeDecodeError,
    }.get(failure, json.JSONDecodeError)
    if failure == "unreadable":
        read_text = Path.read_text

        def unreadable(self, *args, **kwargs):
            if self == path:
                raise PermissionError("fictional unreadable config")
            return read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", unreadable)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid config must fail before storage/runtime/service initialization")

    monkeypatch.setattr(app_config, "ensure_storage_paths", forbidden)
    old_config = main._CONFIG
    cache = {"fictional-service": object()}
    monkeypatch.setattr(main._service_factory, "_SERVICES", cache)
    monkeypatch.setattr(main, "_refresh_runtime_limits", forbidden)
    monkeypatch.setattr(main, "_clear_cached_services", forbidden)
    if loader == "runner":
        monkeypatch.chdir(path.parent)
        monkeypatch.setattr(server_run, "ROOT", path.parent)
        monkeypatch.setattr(server_run, "_reexec_into_venv_if_requested", forbidden)
    with pytest.raises(error):
        if loader == "reload":
            await main.reload_config()
        elif loader == "runner":
            server_run.main()
        else:
            app_config.load_config()
    assert main._CONFIG is old_config
    assert main._service_factory._SERVICES is cache
    assert list(cache) == ["fictional-service"]
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("failure", ["serialize", "write", "fsync", "replace"])
def test_failed_save_preserves_credentials_and_cleans_tempfile(failure: str, monkeypatch: pytest.MonkeyPatch) -> None:
    path = app_config.config_path()
    path.parent.mkdir()
    original = b'{"llm":{"api_key":"fictional-chat-key","embedding":{"api_key":"fictional-embedding-key"}},"mentra":{"gemini_api_key":"fictional-voice-key"}}'
    path.write_bytes(original)
    original_mode = path.stat().st_mode
    cfg = {"llm": {"api_key": "fictional-new-key"}}
    error = OSError
    if failure == "serialize":
        cfg["invalid"] = object()
        error = TypeError
    else:
        def fail(*args, **kwargs):
            if failure == "write":
                args[1].write('{"llm":')
            raise OSError("fictional save failure")

        target, name = (app_config.json, "dump") if failure == "write" else (app_config.os, failure)
        monkeypatch.setattr(target, name, fail)
    with pytest.raises(error):
        app_config.save_config(cfg)
    assert path.read_bytes() == original
    assert path.stat().st_mode == original_mode
    assert list(path.parent.iterdir()) == [path]


def test_overlapping_saves_use_distinct_tempfiles(monkeypatch: pytest.MonkeyPatch) -> None:
    path = app_config.config_path()
    configs = [
        {"llm": {"api_key": "fictional-key"}, "mentra": {"public_base_url": f"http://phone-{number}.example.invalid"}}
        for number in (1, 2)
    ]
    ready = Barrier(2)
    staged_paths = []
    real_replace = os.replace

    def replace(src, dst):
        staged = Path(src)
        assert staged.parent == path.parent
        assert json.loads(staged.read_text(encoding="utf-8")) in configs
        staged_paths.append(staged)
        ready.wait(timeout=5)
        real_replace(src, dst)

    monkeypatch.setattr(app_config.os, "replace", replace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(app_config.save_config, configs))
    assert len(set(staged_paths)) == 2
    assert json.loads(path.read_text(encoding="utf-8")) in configs
    assert list(path.parent.iterdir()) == [path]
