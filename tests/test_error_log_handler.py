"""Verify that _build_uvicorn_log_config wires an ERROR-level RotatingFileHandler
to the root logger so errors from all modules land in errors.log."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

import run as server_run


class _FakeUvicornConfig:
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(message)s",
            },
        },
        "handlers": {},
        "loggers": {},
    }


class _FakeUvicorn:
    config = _FakeUvicornConfig


def test_error_handler_wired_to_root_logger(tmp_path: Path) -> None:
    """_build_uvicorn_log_config must include memu_errors in root logger handlers."""
    cfg = {"debug": {"log_file": str(tmp_path / "server.log")}}
    log_cfg = server_run._build_uvicorn_log_config(_FakeUvicorn, cfg)

    handlers = log_cfg.get("handlers", {})
    assert "memu_errors" in handlers, "memu_errors handler must be defined"
    assert "RotatingFileHandler" in handlers["memu_file"]["class"]

    err_handler_cfg = handlers["memu_errors"]
    assert "RotatingFileHandler" in err_handler_cfg.get("class", ""), "must be RotatingFileHandler"
    assert err_handler_cfg.get("level") == "ERROR", "handler level must be ERROR"
    assert err_handler_cfg.get("filename") == str(server_run.ROOT / "errors.log")
    assert err_handler_cfg.get("formatter") == "memu_error"

    err_formatter_cfg = log_cfg.get("formatters", {}).get("memu_error", {})
    assert "%(asctime)s" in err_formatter_cfg.get("format", "")

    root_cfg = log_cfg.get("loggers", {}).get("", {})
    assert "memu_errors" in (root_cfg.get("handlers") or []), "memu_errors must be on root logger"

    stream_cfg = log_cfg.get("loggers", {}).get("memu.llm.http_client", {})
    assert stream_cfg.get("level") == "INFO"
    assert "memu_file" in (stream_cfg.get("handlers") or [])
    assert "quiet_access" in log_cfg["loggers"]["uvicorn.access"]["filters"]


def test_quiet_access_filter_limits_mentra_errors_without_hiding_other_errors(monkeypatch):
    now = 0.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    quiet = server_run._QuietAccessFilter()

    def access(path, status, port=12345):
        return logging.LogRecord("uvicorn.access", logging.INFO, "", 0,
                                 '%s - "%s %s HTTP/%s" %d',
                                 (f"127.0.0.1:{port}", "GET", path, "1.1", status), None)

    for status in (200, 204, 307):
        assert not quiet.filter(access("/integration/mentra/status", status))
    for status in (401, 409, 503):
        assert quiet.filter(access("/integration/mentra/status?user_id=fictional", status))
        for port in range(100):
            assert not quiet.filter(access(f"/integration/mentra/session/fictional-{port}/heartbeat", status, port))
        assert quiet.filter(access("/health", status))
        assert quiet.filter(access("/health", status))
    now = 59.9
    for status in (401, 409, 503):
        assert not quiet.filter(access("/integration/mentra/health", status))
    now = 60.0
    for status in (401, 409, 503):
        assert quiet.filter(access("/integration/mentra/health", status))
    assert quiet.filter(logging.LogRecord("uvicorn.error", logging.ERROR, "", 0, "Application exception", (), None))


def test_resolve_errors_log_path_migrates_legacy_logs_subfolder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_run, "ROOT", tmp_path)
    legacy = tmp_path / "logs" / "errors.log"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("legacy\n", encoding="utf-8")

    resolved = server_run._resolve_errors_log_path()

    assert resolved == (tmp_path / "errors.log").resolve()
    assert resolved.exists()
    assert resolved.read_text(encoding="utf-8") == "legacy\n"
    assert not legacy.exists()
