from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

import pytest

import run as server_run


def _cfg_with_pid_file(path: Path) -> dict:
    return {"pid_file": str(path)}


def test_enforce_single_instance_rejects_live_owned_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pid_file = tmp_path / ".memu-server.pid"
    pid_file.write_text("2222\n", encoding="utf-8")

    monkeypatch.setattr(server_run, "_is_pid_alive", lambda _pid: True)
    monkeypatch.setattr(server_run, "_is_our_server_process", lambda _pid: True)
    monkeypatch.setattr(server_run.os, "getpid", lambda: 9999)

    with pytest.raises(SystemExit):
        server_run._enforce_single_instance(_cfg_with_pid_file(pid_file))

    assert pid_file.exists()


def test_enforce_single_instance_clears_live_foreign_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pid_file = tmp_path / ".memu-server.pid"
    pid_file.write_text("3333\n", encoding="utf-8")

    monkeypatch.setattr(server_run, "_is_pid_alive", lambda _pid: True)
    monkeypatch.setattr(server_run, "_is_our_server_process", lambda _pid: False)
    monkeypatch.setattr(server_run.os, "getpid", lambda: 9999)

    server_run._enforce_single_instance(_cfg_with_pid_file(pid_file))
    assert not pid_file.exists()


def test_server_process_identity_accepts_relative_runner(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server_run, "_proc_cmdline", lambda _pid: "python run.py")
    monkeypatch.setattr(server_run, "_proc_cwd", lambda _pid: server_run.ROOT)

    assert server_run._is_our_server_process(2222) is True


def test_single_instance_rejects_real_relative_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = tmp_path / "run.py"
    runner.write_text("import time; time.sleep(10)\n", encoding="utf-8")
    process = subprocess.Popen([sys.executable, "run.py"], cwd=tmp_path)
    try:
        time.sleep(0.1)
        pid_file = tmp_path / "server.pid"
        pid_file.write_text(str(process.pid), encoding="utf-8")
        monkeypatch.setattr(server_run, "ROOT", tmp_path)

        with pytest.raises(SystemExit):
            server_run._enforce_single_instance(_cfg_with_pid_file(pid_file))
    finally:
        process.terminate()
        process.wait(timeout=2)


def test_quiet_access_filter_suppresses_successful_request() -> None:
    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='127.0.0.1:54446 - "GET /api/canvas/global HTTP/1.1" 200',
        args=(),
        exc_info=None,
    )

    assert server_run._QuietAccessFilter().filter(record) is False


def test_quiet_access_filter_keeps_failed_request() -> None:
    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='127.0.0.1:53774 - "GET /api/canvas/global HTTP/1.1" 500',
        args=(),
        exc_info=None,
    )

    assert server_run._QuietAccessFilter().filter(record) is True


def test_quiet_access_filter_keeps_application_log() -> None:
    record = logging.LogRecord(
        name="uvicorn.error",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Application startup complete.",
        args=(),
        exc_info=None,
    )

    assert server_run._QuietAccessFilter().filter(record) is True
