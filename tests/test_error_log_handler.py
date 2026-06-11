"""Verify that _build_uvicorn_log_config wires an ERROR-level RotatingFileHandler
to the root logger so errors from all modules land in logs/errors.log."""
from __future__ import annotations

import logging
import logging.handlers
import types
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

    err_handler_cfg = handlers["memu_errors"]
    assert "RotatingFileHandler" in err_handler_cfg.get("class", ""), "must be RotatingFileHandler"
    assert err_handler_cfg.get("level") == "ERROR", "handler level must be ERROR"
    assert "errors.log" in err_handler_cfg.get("filename", ""), "must write to errors.log"

    root_cfg = log_cfg.get("loggers", {}).get("", {})
    assert "memu_errors" in (root_cfg.get("handlers") or []), "memu_errors must be on root logger"


def test_error_handler_writes_errors_to_file(tmp_path: Path) -> None:
    """An ERROR log message reaches logs/errors.log when the handler is attached."""
    errors_log = tmp_path / "logs" / "errors.log"
    errors_log.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        filename=str(errors_log),
        maxBytes=512_000,
        backupCount=2,
        encoding="utf-8",
    )
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)

    test_logger = logging.getLogger("test_error_log_handler_sentinel")
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.DEBUG)
    try:
        test_logger.error("sentinel error message for test")
        handler.flush()
    finally:
        test_logger.removeHandler(handler)
        handler.close()

    content = errors_log.read_text(encoding="utf-8")
    assert "sentinel error message for test" in content
