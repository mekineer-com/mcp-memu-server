from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

import run as server_run


def _cfg_with_pid_file(path: Path) -> dict:
    return {"pid_file": str(path)}


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
