# SPDX-License-Identifier: Apache-2.0
"""Ownership, immutable active profile and real systemd unit syntax checks."""

import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from app.config import Config
from deploy import runtime, services


def test_native_command_keeps_current_capacity_and_context_contract():
    cfg = Config.load()
    cmd = runtime.native_command("llama", cfg, Path("/profile.json"))
    assert cmd[cmd.index("-c") + 1] == str(cfg.llm.context_tokens)
    assert cmd[cmd.index("-t") + 1] == "4" and cmd[cmd.index("-np") + 1] == "1"
    assert cmd[cmd.index("--cache-ram") + 1] == "256"
    assert json.loads(cmd[cmd.index("--chat-template-kwargs") + 1]) == {
        "enable_thinking": False
    }
    asr = runtime.native_command("asr", cfg, Path("/profile.json"))
    assert asr[asr.index("--cache-ram") + 1] == "64" and asr[asr.index("-t") + 1] == "2"
    cfg.backends.llm_parallel = 2
    with pytest.raises(ValueError):
        cfg.validate()


def test_real_systemd_accepts_generated_units_and_literal_paths(tmp_path, monkeypatch):
    cfg = Config.load()
    cfg.backends.stack_root = str(tmp_path / "stack with space")
    python = runtime.paths(cfg)["app_python"]
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    monkeypatch.setattr(services, "ROOT", tmp_path / "project with space%literal")
    paths = []
    for name in services.ORDER:
        file = tmp_path / services.unit(name)
        file.write_text(services.render(name, cfg))
        paths.append(str(file))
    target = tmp_path / (services.PREFIX + ".target")
    target.write_text("[Unit]\nDescription=Test group\n")
    paths.append(str(target))
    result = subprocess.run(
        ["systemd-analyze", "--user", "verify", *paths], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_occupied_unowned_port_is_not_stopped_or_replaced(monkeypatch):
    cfg = Config.load()
    with socket.socket() as owned:
        owned.bind(("127.0.0.1", 0))
        owned.listen()
        cfg.llm.base_url = f"http://127.0.0.1:{owned.getsockname()[1]}"
        monkeypatch.setattr(
            services,
            "properties",
            lambda name: {"cgroup_pids": [], "ActiveState": "inactive"},
        )
        monkeypatch.setattr(
            runtime,
            "verify",
            lambda *a: pytest.fail("Artifacts must not mask port conflict"),
        )
        with pytest.raises(RuntimeError, match="occupied outside"):
            services.preflight(["llama"], cfg)
        assert owned.fileno() >= 0


def install_fixture(tmp_path, monkeypatch):
    cfg = Config.load()
    cfg.backends.stack_root = str(tmp_path / "stack")
    link = tmp_path / "units"
    link.mkdir()
    monkeypatch.setattr(services, "LINK_DIR", link)
    monkeypatch.setattr(services, "properties", lambda name: {"cgroup_pids": []})
    monkeypatch.setattr(
        services,
        "command",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    return cfg, link


def test_install_refuses_unrelated_same_name_unit(tmp_path, monkeypatch):
    cfg, link = install_fixture(tmp_path, monkeypatch)
    file = link / services.unit("llama")
    file.write_text("user-owned")
    with pytest.raises(RuntimeError, match="unrelated"):
        services.install(cfg)
    assert file.read_text() == "user-owned"


def test_active_profile_cannot_be_replaced_and_local_unit_edits_survive(
    tmp_path, monkeypatch
):
    cfg, link = install_fixture(tmp_path, monkeypatch)
    services.install(cfg)
    profile = services.profile_path(cfg)
    before = profile.read_bytes()
    monkeypatch.setattr(services, "properties", lambda name: {"cgroup_pids": [123]})
    cfg.llm.temperature = 0.25
    with pytest.raises(RuntimeError, match="Stop managed"):
        services.install(cfg)
    assert profile.read_bytes() == before
    monkeypatch.setattr(services, "properties", lambda name: {"cgroup_pids": []})
    unit = services.directory(cfg) / services.unit("llama")
    unit.write_text(unit.read_text() + "# local change\n")
    with pytest.raises(RuntimeError, match="local edits"):
        services.install(cfg)
    assert unit.read_text().endswith("# local change\n")


def test_health_from_an_unowned_listener_is_not_ready(tmp_path, monkeypatch):
    cfg = Config.load()
    cfg.backends.stack_root = str(tmp_path)
    monkeypatch.setattr(
        services,
        "properties",
        lambda name: {
            "main_pid": 123,
            "MainPID": "123",
            "cgroup_pids": [123],
            "ActiveState": "active",
            "FragmentPath": str(services.directory(cfg) / services.unit(name)),
        },
    )
    monkeypatch.setattr(services, "health", lambda *a: {"ready": True})
    monkeypatch.setattr(services, "owns_port", lambda *a: False)
    assert services.state(["llama"], cfg)["status"] == "DEGRADED"


def test_artifact_mismatch_does_not_launch_anything(tmp_path, monkeypatch):
    cfg = Config.load()
    cfg.backends.stack_root = str(tmp_path)
    python = runtime.paths(cfg)["app_python"]
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    artifact = tmp_path / "weights"
    artifact.write_bytes(b"changed")
    monkeypatch.setattr(
        runtime,
        "artifact_files",
        lambda *a: {str(artifact): hashlib.sha256(b"original").hexdigest()},
    )
    with pytest.raises(RuntimeError, match="hash mismatch"):
        runtime.verify(cfg, ["llama"])


def test_failed_systemd_observation_is_not_reported_as_stopped(monkeypatch):
    monkeypatch.setattr(
        services,
        "command",
        lambda *a, **kw: SimpleNamespace(
            returncode=1, stdout="", stderr="Message recipient disconnected"
        ),
    )
    with pytest.raises(RuntimeError, match="Cannot observe systemd"):
        services.properties("app")


def test_failed_unit_without_pids_is_still_degraded(tmp_path, monkeypatch):
    cfg = Config.load()
    cfg.backends.stack_root = str(tmp_path)
    monkeypatch.setattr(
        services,
        "properties",
        lambda name: {
            "main_pid": 0,
            "cgroup_pids": [],
            "ActiveState": "failed",
            "LoadState": "loaded",
            "FragmentPath": str(services.directory(cfg) / services.unit(name)),
        },
    )
    assert services.state(["app"], cfg)["status"] == "DEGRADED"


def test_start_failure_preserves_preexisting_service(monkeypatch):
    cfg = Config.load()
    calls = []
    monkeypatch.setattr(services, "preflight", lambda *a: {})
    monkeypatch.setattr(services, "install", lambda *a: None)
    monkeypatch.setattr(services, "require_headroom", lambda: None)
    monkeypatch.setattr(services, "command", lambda argv, **kw: calls.append(argv))
    monkeypatch.setattr(
        services,
        "properties",
        lambda name: {"ActiveState": "active" if name == "llama" else "inactive"},
    )

    def readiness(names, *args):
        if names == ["asr"]:
            raise RuntimeError("injected startup failure")
        return {"status": "READY"}

    monkeypatch.setattr(services, "wait_ready", readiness)
    with pytest.raises(RuntimeError, match="injected startup failure"):
        services.start(["llama", "asr"], cfg, 1)
    stopped = [argv[-1] for argv in calls if "stop" in argv]
    assert stopped == [services.unit("asr")]


def test_low_memory_does_not_start_another_model_or_stop_existing_one(monkeypatch):
    import psutil

    cfg = Config.load()
    calls = []
    monkeypatch.setattr(services, "preflight", lambda *a: {})
    monkeypatch.setattr(services, "install", lambda *a: None)
    monkeypatch.setattr(services, "command", lambda argv, **kw: calls.append(argv))
    monkeypatch.setattr(
        services,
        "properties",
        lambda name: {"ActiveState": "active" if name == "llama" else "inactive"},
    )
    monkeypatch.setattr(services, "wait_ready", lambda *a: {"status": "READY"})
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=1024**3)
    )
    with pytest.raises(RuntimeError, match="another model service was not admitted"):
        services.start(["llama", "asr"], cfg, 1)
    assert all(not any(word.endswith(".service") for word in argv) for argv in calls)
