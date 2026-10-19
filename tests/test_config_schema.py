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


def test_mental_health_defaults_off_without_resetting_saved_choice() -> None:
    cfg = default_config()
    assert cfg["retrieve"]["mental_health_query"] is False
    cfg["retrieve"]["mental_health_query"] = True
    app_config.save_config(cfg)
    assert app_config.load_config()["retrieve"]["mental_health_query"] is True
    cfg["retrieve"]["top_k"] = 7
    updated = app_config.merge_model_config(cfg, {"retrieve": {"mental_health_query": False}})
    assert updated["retrieve"] == {**cfg["retrieve"], "mental_health_query": False}


def _mapping_config() -> dict:
    connection = {"provider": "openai", "base_url": "https://steps.example/v1", "api_key": "step-secret",
                  "chat_model": "step-model", "context_window_tokens": 50000, "max_tokens": 2000}
    return {"claude_code": False, "llm": {
        "provider": "openai", "base_url": "https://fictional.example/v1", "api_key": "base-secret",
        "chat_model": "custom-base", "context_window_tokens": 100000,
        "embedding": {"api_key": "embedding-secret", "base_url": "https://embedding.example/"},
        "step_models": {"memory_extract": connection, "category_update": "custom-category",
                        "consolidation": copy.deepcopy(connection)},
    }, "storage": {"resources_dir": "untouched"},
        "mentra": {"gemini_api_key": "voice-secret", "model": "unchanged-voice"}}


def test_model_mapping_normalization_preserves_scalar_model_and_matching_capacity() -> None:
    cfg = _mapping_config()
    cfg["llm"]["step_models"] = {"memory_extract": "old-model", "preprocess": "another-model"}
    cfg["llm"]["step_context_window_tokens"] = {"memory_extract": 40000}
    assert app_config.normalize_model_mappings(cfg) == cfg
    profiles = app_config.default_llm_profiles_from_server_config(cfg, operation="memorize")
    assert profiles["memory_extract"]["chat_model"] == "old-model"
    assert profiles["memory_extract"]["context_window_tokens"] == 40000
    assert "context_window_tokens" not in profiles["preprocess"]


# Transition cleanup expires 2027-01-17: remove this old-mapping upgrade handling.
def test_load_config_cleans_old_mapping_layout_once(monkeypatch) -> None:
    cfg = _mapping_config()
    cfg["llm"].update({"reply_models": {"srmm": {"source": "config"}},
                       "st_selected": {"resolved": {"api_key": "obsolete-secret"}},
                       "step_context_window_tokens": {"srmm": 40000, "memory_extract": 70000}})
    cfg["llm"]["step_models"]["reflection"] = {"source": "configured", "resolved": {"api_key": "old"}}
    cfg["llm"]["step_models"]["preprocess"] = {"source": "config"}
    path = app_config.config_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(cfg))
    save = app_config.save_config
    calls = []
    monkeypatch.setattr(app_config, "save_config", lambda value: (calls.append(copy.deepcopy(value)), save(value)))
    monkeypatch.setattr(app_config, "ensure_storage_paths", lambda value: value["storage"].update({"dsn": "runtime-only"}))
    app_config.load_config()
    saved = json.loads(path.read_text())
    assert "reply_models" not in saved["llm"] and "st_selected" not in saved["llm"]
    assert "reflection" not in saved["llm"]["step_models"] and "preprocess" not in saved["llm"]["step_models"]
    assert saved["llm"]["step_context_window_tokens"] == {"memory_extract": 70000}
    assert saved["llm"]["step_models"]["memory_extract"] == cfg["llm"]["step_models"]["memory_extract"]
    assert saved["llm"]["step_models"]["category_update"] == "custom-category"
    assert saved["llm"]["embedding"] == cfg["llm"]["embedding"]
    assert saved["storage"] == cfg["storage"]
    app_config.load_config()
    assert len(calls) == 1


def test_load_corrected_config_does_not_rewrite_hand_edits(monkeypatch) -> None:
    cfg = _mapping_config()
    path = app_config.config_path()
    path.parent.mkdir(parents=True)
    original = json.dumps(cfg, indent=4) + "\n"
    path.write_text(original)
    monkeypatch.setattr(app_config, "ensure_storage_paths", lambda _cfg: None)
    monkeypatch.setattr(app_config, "save_config", lambda _cfg: pytest.fail("rewrote corrected config"))
    assert app_config.load_config() == cfg
    assert path.read_text() == original


@pytest.mark.parametrize("operation,names", [
    ("st_reply", set()), ("other_reply", set()), ("retrieve", {"reflection"}), ("rollup", {"preprocess"}),
    ("memorize", {"preprocess", "memory_extract", "category_update", "consolidation"}),
    ("consolidation", {"category_update", "consolidation"}), (None, set(app_config.MODEL_STEPS)),
])
def test_model_mapping_operation_profiles(operation, names) -> None:
    cfg = _mapping_config()
    cfg["llm"]["step_context_window_tokens"] = {"memory_extract": 70000}
    profiles = app_config.default_llm_profiles_from_server_config(cfg, operation=operation)
    assert set(profiles) == {"default", "embedding"} | names
    assert profiles["default"]["chat_model"] == "custom-base"
    assert profiles["embedding"]["api_key"] == "embedding-secret"
    if "memory_extract" in profiles:
        assert profiles["memory_extract"]["api_key"] == "step-secret"
        assert profiles["memory_extract"]["context_window_tokens"] == 70000
    if "consolidation" in profiles:
        assert profiles["consolidation"]["context_window_tokens"] == 50000
    if "category_update" in profiles:
        assert profiles["category_update"]["chat_model"] == "custom-category"
        assert "context_window_tokens" not in profiles["category_update"]


@pytest.mark.parametrize("operation", [None, "other_reply", "retrieve", "rollup", "memorize", "consolidation"])
def test_cli_skips_unused_api_profiles(operation) -> None:
    cfg = _mapping_config()
    cfg["claude_code"] = True
    assert set(app_config.default_llm_profiles_from_server_config(cfg, operation=operation)) == {"default", "embedding"}


@pytest.mark.parametrize("field", ["context_window_tokens", "max_tokens"])
@pytest.mark.parametrize("capacity", [0, -1, True, "64000"])
def test_selected_model_capacity_must_be_positive_integer(field, capacity) -> None:
    cfg = _mapping_config()
    cfg["llm"]["step_models"]["memory_extract"][field] = capacity
    with pytest.raises(ValueError, match="positive integer"):
        app_config.default_llm_profiles_from_server_config(cfg, operation="memorize")


@pytest.mark.parametrize("source", ["base", "string", "object"])
@pytest.mark.parametrize("override", [None, 70000])
def test_named_step_capacity_overlays_selected_profile_without_leaking(source, override) -> None:
    cfg = _mapping_config()
    cfg["llm"]["step_models"].pop("preprocess", None)
    if source == "string":
        cfg["llm"]["step_models"]["preprocess"] = "another-model"
    elif source == "object":
        cfg["llm"]["step_models"]["preprocess"] = copy.deepcopy(cfg["llm"]["step_models"]["memory_extract"])
    if override is not None:
        cfg["llm"]["step_context_window_tokens"] = {"preprocess": override}
    original = copy.deepcopy(cfg)
    profiles = app_config.default_llm_profiles_from_server_config(cfg, operation="rollup")
    assert profiles["preprocess"].get("context_window_tokens") == (override if override else {
        "base": 100000, "string": None, "object": 50000,
    }[source])
    profiles["preprocess"]["api_key"] = "changed-copy"
    assert cfg == original and profiles["default"]["api_key"] == "base-secret"


def test_independent_step_requires_its_own_key_and_does_not_inherit_capacity() -> None:
    cfg = _mapping_config()
    row = cfg["llm"]["step_models"]["memory_extract"]
    del row["api_key"]
    with pytest.raises(ValueError, match="explicit api_key"):
        app_config.default_llm_profiles_from_server_config(cfg, operation="memorize")
    row["api_key"] = ""
    del row["context_window_tokens"]
    profile = app_config.default_llm_profiles_from_server_config(cfg, operation="memorize")["memory_extract"]
    assert profile["api_key"] == "" and "context_window_tokens" not in profile


def test_named_capacity_overrides_invalid_connection_capacity_but_not_output_cap() -> None:
    cfg = _mapping_config()
    connection = cfg["llm"]["step_models"]["memory_extract"]
    connection["context_window_tokens"] = 0
    cfg["llm"]["step_context_window_tokens"] = {"memory_extract": 70000}
    profile = app_config.default_llm_profiles_from_server_config(cfg, operation="memorize")["memory_extract"]
    assert profile["context_window_tokens"] == 70000
    connection["max_tokens"] = 0
    with pytest.raises(ValueError, match="max_tokens.*positive integer"):
        app_config.default_llm_profiles_from_server_config(cfg, operation="memorize")


def test_model_config_merge_preserves_secrets_and_other_settings_and_deletes_only_requested_entries() -> None:
    current = _mapping_config()
    current["storage"] = {
        "resources_dir": "selected-resources", "sqlite_dir": "selected-sqlite",
        "metadata_store": {"provider": "sqlite", "dsn": "sqlite:///selected.db", "ddl_mode": "create"},
    }
    current["llm"]["step_context_window_tokens"] = {"category_update": 8000, "reflection": 12000}
    before = copy.deepcopy(current)
    patch = {"llm": {"step_models": {"category_update": None},
                     "step_context_window_tokens": {"category_update": None},
                     "embedding": {"base_url": "https://new-embedding.example/"}}}
    result = app_config.merge_model_config(current, patch)
    assert result["llm"]["step_models"] == {k: v for k, v in current["llm"]["step_models"].items() if k != "category_update"}
    assert result["llm"]["step_context_window_tokens"] == {"reflection": 12000}
    assert result["llm"]["embedding"]["api_key"] == "embedding-secret"
    assert result["llm"]["api_key"] == "base-secret"
    assert result["storage"] == current["storage"] and result["mentra"] == current["mentra"]
    assert current == before
    cleared = app_config.merge_model_config(current, {"llm": {
        "step_models": {key: None for key in current["llm"]["step_models"]},
        "step_context_window_tokens": {key: None for key in current["llm"]["step_context_window_tokens"]},
    }})
    assert "step_models" not in cleared["llm"] and "step_context_window_tokens" not in cleared["llm"]
    for storage_patch in ({"resources_dir": "new-resources"}, {"metadata_store": {"ddl_mode": "validate"}}):
        updated = app_config.merge_model_config(current, {"storage": storage_patch})["storage"]
        assert updated["sqlite_dir"] == "selected-sqlite"
        assert updated["resources_dir"] == storage_patch.get("resources_dir", "selected-resources")
        assert updated["metadata_store"] == {
            **current["storage"]["metadata_store"], **storage_patch.get("metadata_store", {}),
        }
    for replacement in (None, "", []):
        assert app_config.merge_model_config(current, {"storage": replacement})["storage"] == replacement
        updated = app_config.merge_model_config(current, {"storage": {"metadata_store": replacement}})["storage"]
        assert updated == {**current["storage"], "metadata_store": replacement}
    for body in ({"retrieve": {"mental_health_query": False}}, {"mentra": {"gemini_api_key": "new-voice"}},
                 {"storage": {"metadata_store": {"ddl_mode": "validate"}}}):
        updated = app_config.merge_model_config(current, body)
        for field in ("llm", "mentra", "retrieve", "storage"):
            if field not in body:
                assert updated.get(field) == current.get(field)


def test_model_mapping_masks_every_nested_profile_without_mutating_config() -> None:
    cfg = _mapping_config()
    masked = app_config.mask_config(cfg)
    assert masked["llm"]["api_key"] == "base...cret"
    assert masked["llm"]["step_models"]["memory_extract"]["api_key"] == "***"
    assert masked["llm"]["embedding"]["api_key"] == "***"
    assert cfg["llm"]["step_models"]["memory_extract"]["api_key"] == "step-secret"
    assert masked["mentra"]["gemini_api_key"] == "***"
    unmasked_llm = app_config.mask_config(cfg, include_llm_secret=True)
    assert unmasked_llm["llm"]["api_key"] == cfg["llm"]["api_key"]
    assert unmasked_llm["mentra"]["gemini_api_key"] == "***"
    assert app_config.mask_config({"llm": {"api_key": ""}})["llm"]["api_key"] == ""


@pytest.mark.parametrize("secret", ["***", "abcd...wxyz"])
@pytest.mark.parametrize("field", ["llm", "mentra"])
def test_model_config_rejects_masked_secret_round_trip(secret, field) -> None:
    cfg = _mapping_config()
    before = copy.deepcopy(cfg)
    body = ({"llm": {"step_models": {"memory_extract": {"api_key": secret}}}}
            if field == "llm" else {"mentra": {"gemini_api_key": secret}})
    with pytest.raises(ValueError, match="Masked .*api_key"):
        app_config.merge_model_config(cfg, body)
    assert cfg == before


@pytest.mark.parametrize("cli", [False, True])
def test_model_mapping_rejects_native_noncompatible_protocol(cli) -> None:
    cfg = _mapping_config()
    cfg["claude_code"] = cli
    cfg["llm"]["step_models"]["memory_extract"]["provider"] = "anthropic"
    with pytest.raises(ValueError, match="OpenAI-compatible"):
        app_config.normalize_model_mappings(cfg)


def test_default_whatsapp_paths_point_at_channels_data() -> None:
    hermes = default_config()["hermes"]

    assert hermes["home"] == "../hermes-channels/data"
    assert hermes["state_db_path"] == "../hermes-channels/data/state.db"
    assert hermes["sessions_index_path"] == "../hermes-channels/data/sessions/sessions.json"
    assert hermes["whatsapp_web_source_db"] == "../hermes-channels/data/whatsapp/web_source.db"
    assert "whatsapp_history_source" not in hermes


@pytest.mark.parametrize("paths", ["generated", "absolute_override", "legacy_absolute", "relative_override"])
def test_channels_paths_follow_config_location_without_rebasing_explicit_paths(tmp_path, monkeypatch, paths):
    from app import main
    server = tmp_path / "before" / "mcp-memu-server"
    monkeypatch.setattr(app_config, "config_dir", lambda: server)
    hermes = default_config()["hermes"]
    external = tmp_path / ("explicit" if paths == "absolute_override" else "before") / "hermes-channels" / "data"
    if paths in {"absolute_override", "legacy_absolute"}:
        hermes = {"home": str(external), "state_db_path": str(external / "state.db"),
                  "sessions_index_path": str(external / "sessions/sessions.json"),
                  "whatsapp_web_source_db": str(external / "whatsapp/web_source.db")}
    elif paths == "relative_override":
        hermes = {"home": "../custom-data", "state_db_path": "../custom-data/state.db",
                  "sessions_index_path": "../custom-data/sessions/sessions.json",
                  "whatsapp_web_source_db": "../custom-data/whatsapp/web_source.db"}
    saved = copy.deepcopy(hermes)
    monkeypatch.setattr(main, "_CONFIG", {"hermes": hermes, "storage": {"resources_dir": "../memu/resources"}})
    unrelated = tmp_path / "unrelated-working-directory"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    for location in ("before", "after"):
        server = tmp_path / location / "mcp-memu-server"
        expected = external if paths in {"absolute_override", "legacy_absolute"} else server.parent / (
            "custom-data" if paths == "relative_override" else "hermes-channels/data")
        expected.mkdir(parents=True, exist_ok=True)
        (expected / "channel_directory.json").write_text(json.dumps({"platforms": {"whatsapp": [
            {"id": "fictional@g.us", "name": f"Fictional {location}"}]}}))
        _, home, sessions, state = main._resolve_cross_source_paths()
        assert (home, sessions, state, main._resolve_whatsapp_web_source_config()) == (
            expected, expected / "sessions/sessions.json", expected / "state.db", expected / "whatsapp/web_source.db")
        assert main._message_log._load_whatsapp_directory_names()["fictional@g.us"] == f"Fictional {location}"
        assert hermes == saved


def test_empty_channels_paths_keep_existing_environment_fallback(tmp_path, monkeypatch):
    from app import main
    monkeypatch.setattr(main, "_CONFIG", {"hermes": {}, "storage": {"resources_dir": str(tmp_path)}})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "explicit-legacy-env"))
    monkeypatch.setenv("CHANNELS_HOME", str(tmp_path / "explicit-env"))
    assert main._resolve_cross_source_paths()[1:] == (None, None, None)
    assert main._resolve_whatsapp_web_source_config() is None
    assert main._message_log._load_whatsapp_directory_names() == {}
    assert main._conversation_sources._hermes_base() == tmp_path / "explicit-legacy-env"


def test_default_dynamic_category_cluster_size_matches_engine() -> None:
    assert default_config()["categories"]["dynamic_category_cluster_size"] == 10


def test_default_mentra_contract_is_keyless_and_loopback() -> None:
    config = default_config()
    assert config["listen"]["host"] == "127.0.0.1"
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
