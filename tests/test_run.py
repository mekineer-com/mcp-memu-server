from __future__ import annotations

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
