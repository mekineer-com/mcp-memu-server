"""Checks for the installable config example, defaults, and atomic saves."""

from __future__ import annotations

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
    monkeypatch.setattr(app_config, "home_dir", lambda: tmp_path / "home")


def test_example_is_valid_json_and_non_empty() -> None:
    example = json.loads(_EXAMPLE_CONFIG.read_text())
    assert isinstance(example, dict)
    assert example, "config.example.json must be non-empty"
    # setup_install.py fills paths inside these four sections.
    for required in ("llm", "storage", "memu", "listen"):
        assert required in example, f"config.example.json missing top-level '{required}'"
    assert "embed_model" not in example["llm"]["embedding"]
    assert example["storage"]["metadata_store"]["dsn"]


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
        assert staged.read_text(encoding="utf-8") == json.dumps(cfg, indent=2)
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
        assert app_config.load_config() == cfg
    else:
        app_config.save_config(cfg)
    assert len(staged_paths) == 1
    assert json.loads(path.read_text(encoding="utf-8")) == cfg
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
    assert app_config.load_config() == cfg
    assert initialized == [cfg]
    assert server_run._read_json(path) == cfg


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
