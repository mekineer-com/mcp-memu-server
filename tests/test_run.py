from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

import run as server_run


def _cfg_with_pid_file(path: Path) -> dict:
    return {"pid_file": str(path)}


@pytest.mark.parametrize("contents", ["9999\n", "3333\n", "invalid\n", "", None])
def test_pidfile_cleanup_only_removes_own_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents):
    pid_file = tmp_path / ".memu-server.pid"
    callbacks = []
    monkeypatch.setattr(server_run.os, "getpid", lambda: 9999)
    monkeypatch.setattr(server_run.atexit, "register", callbacks.append)

    server_run._write_pidfile(_cfg_with_pid_file(pid_file))
    assert pid_file.read_text(encoding="utf-8") == "9999\n"
    if contents is None:
        pid_file.unlink()
    else:
        pid_file.write_text(contents, encoding="utf-8")

    callbacks[0]()
    if contents in ("9999\n", None):
        assert not pid_file.exists()
    else:
        assert pid_file.read_text(encoding="utf-8") == contents


def test_enforce_single_instance_clears_live_foreign_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pid_file = tmp_path / ".memu-server.pid"
    pid_file.write_text("3333\n", encoding="utf-8")

    monkeypatch.setattr(server_run, "_is_pid_alive", lambda _pid: True)
    monkeypatch.setattr(server_run, "_is_our_server_process", lambda _pid: False)
    monkeypatch.setattr(server_run.os, "getpid", lambda: 9999)

    server_run._enforce_single_instance(_cfg_with_pid_file(pid_file))
    assert not pid_file.exists()


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
        assert pid_file.exists()
    finally:
        process.terminate()
        process.wait(timeout=2)
