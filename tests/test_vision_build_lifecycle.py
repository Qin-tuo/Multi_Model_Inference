# SPDX-License-Identifier: Apache-2.0
"""A failed candidate build must restore only the originally active services."""

from contextlib import nullcontext
import fcntl
import json
from types import SimpleNamespace

import pytest

from deploy import operations as workflow


def setup(tmp_path, monkeypatch):
    calls = []
    cfg = SimpleNamespace()
    state = {
        "status": "READY",
        "services": {
            n: {
                "ActiveState": "active" if n in ("llama", "app") else "inactive",
                "cgroup_pids": [10] if n in ("llama", "app") else [],
                "managed_by_project": True,
                "health": {"phase": "idle"},
            }
            for n in workflow.services.ORDER
        },
    }
    monkeypatch.setattr(workflow.services, "operation", lambda _: nullcontext())
    monkeypatch.setattr(workflow.services, "state", lambda *a: state)
    monkeypatch.setattr(workflow.runtime, "paths", lambda _: {"stack": tmp_path})

    def stop(names, actual):
        assert actual is cfg
        calls.append(("stop", names))
        return {"status": "STOPPED"}

    def start(names, actual, timeout):
        assert actual is cfg
        calls.append(("start", names))
        return {"status": "READY"}

    monkeypatch.setattr(workflow.services, "stop", stop)
    monkeypatch.setattr(workflow.services, "start", start)
    return cfg, state, calls


def test_failure_restores_exact_active_components_and_releases_lease(
    tmp_path, monkeypatch
):
    cfg, state, calls = setup(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="candidate failure"):
        with workflow.exclusive_build(cfg, tmp_path / "report"):
            raise RuntimeError("candidate failure")
    assert calls[-1] == ("start", ["llama", "app"])
    with (tmp_path / "cache/ort-build.owner.lock").open("a+b") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_unrelated_owner_is_rejected_before_stopping_anything(tmp_path, monkeypatch):
    cfg, state, calls = setup(tmp_path, monkeypatch)
    state["services"]["app"]["managed_by_project"] = False
    with pytest.raises(RuntimeError, match="unrelated"):
        with workflow.exclusive_build(cfg, tmp_path / "report"):
            pytest.fail("Build must not run")
    assert calls == []


def test_busy_application_is_not_interrupted_by_build(tmp_path, monkeypatch):
    cfg, state, calls = setup(tmp_path, monkeypatch)
    state["services"]["app"]["health"]["phase"] = "generating"
    with pytest.raises(RuntimeError, match="current application turn"):
        with workflow.exclusive_build(cfg, tmp_path / "report"):
            pytest.fail("Build must not run")
    assert calls == []


def test_failed_exclusive_admission_restores_service_state(tmp_path, monkeypatch):
    cfg, state, calls = setup(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    cache.mkdir()
    with (cache / "ort-build.owner.lock").open("a+b") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            with workflow.exclusive_build(cfg, tmp_path / "report"):
                pytest.fail("Build must not run")
    assert calls[-1] == ("start", ["llama", "app"])


def failed_stop():
    return {
        "status": "DEGRADED",
        "services": {
            name: {
                "ActiveState": "failed" if name == "app" else "inactive",
                "Result": "timeout" if name == "app" else "success",
                "cgroup_pids": [], "main_pid": 0, "managed_by_project": True,
            }
            for name in workflow.services.ORDER
        },
    }


def test_diagnostic_keeps_timeout_evidence_after_processes_exit(tmp_path, monkeypatch):
    cfg, _, calls = setup(tmp_path, monkeypatch)
    stopped = failed_stop()
    monkeypatch.setattr(workflow.services, "stop", lambda *args: stopped)
    folder = tmp_path / "report"
    with workflow.exclusive_build(cfg, folder, allow_failed_stop=True):
        assert json.loads((folder / "services-stopped.json").read_text()) == stopped
        warning = json.loads((folder / "shutdown-warning.json").read_text())
        assert warning["services"]["app"]["Result"] == "timeout"
    assert calls[-1] == ("start", ["llama", "app"])


@pytest.mark.parametrize("field,value", [
    ("cgroup_pids", [42]), ("main_pid", 42),
    ("ActiveState", "activating"), ("managed_by_project", False),
    ("cgroup_pids", None),
])
def test_diagnostic_cannot_bypass_live_or_unconfirmed_processes(
    tmp_path, monkeypatch, field, value
):
    cfg, _, calls = setup(tmp_path, monkeypatch)
    stopped = failed_stop()
    stopped["services"]["app"][field] = value
    monkeypatch.setattr(workflow.services, "stop", lambda *args: stopped)
    with pytest.raises(RuntimeError, match="have not stopped"):
        with workflow.exclusive_build(cfg, tmp_path / "report", allow_failed_stop=True):
            pytest.fail("Unconfirmed shutdown must not admit profiling")
    assert calls[-1] == ("start", ["llama", "app"])


def test_ordinary_measurement_still_requires_clean_shutdown(tmp_path, monkeypatch):
    cfg, _, calls = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(workflow.services, "stop", lambda *args: failed_stop())
    with pytest.raises(RuntimeError, match="not clean"):
        with workflow.exclusive_build(cfg, tmp_path / "report"):
            pytest.fail("Ordinary baselines must reject shutdown failures")
    assert calls[-1] == ("start", ["llama", "app"])
