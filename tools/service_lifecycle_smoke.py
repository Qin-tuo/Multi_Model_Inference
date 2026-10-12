"""Real user-systemd ownership, rollback and bounded-stop probes."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy import services
from tools.project import CACHE, stamp, write_json


def show(name):
    result = services.command(["systemctl", "--user", "show", name, "--property=ActiveState",
        "--property=MainPID", "--property=Result", "--property=NRestarts", "--property=ControlGroup",
        "--property=Restart", "--property=StartLimitBurst"], check=False)
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def main():
    if services.properties("camera").get("ActiveState") == "active":
        raise RuntimeError("An existing managed camera session is active; leave it intact before running this probe")
    checks = {}
    active = stopped = None
    try:
        active = services.start(["camera"], timeout=30)
        initial_pid = active["components"]["camera"]["main_pid"]
        checks["real_camera_ready"] = active["backend_status"] == "READY"
        with patch.object(services, "probe", return_value={"ready": False, "reason": "injected health failure"}):
            try: services.start(["camera"], timeout=.5)
            except RuntimeError: pass
            else: raise RuntimeError("Injected readiness failure was ignored")
        checks["preexisting_unit_preserved_on_failure"] = services.properties("camera")["main_pid"] == initial_pid
        stopped = services.stop(["camera"])
        checks["real_camera_group_empty_after_stop"] = all(not row["main_pid"] and not row["cgroup_pids"]
            for row in stopped["components"].values())
        with patch.object(services, "probe", return_value={"ready": False, "reason": "injected health failure"}):
            try: services.start(["camera"], timeout=.5)
            except RuntimeError: pass
            else: raise RuntimeError("Injected readiness failure was ignored")
        checks["new_unit_rolled_back_on_failure"] = services.properties("camera")["main_pid"] == 0
    finally:
        services.stop(["camera"])

    suffix = uuid.uuid4().hex[:8]
    hanging = f"mmassistant-test-hanging-{suffix}.service"
    failing = f"mmassistant-test-restart-{suffix}.service"
    fixture = CACHE / "fixtures/service-lifecycle"
    fixture.mkdir(parents=True, exist_ok=True)
    script = fixture / "hanging.py"
    script.write_text("import signal,subprocess,time\nsignal.signal(signal.SIGINT,signal.SIG_IGN)\nsubprocess.Popen(['/usr/bin/python3','-c','import signal,time;signal.signal(signal.SIGINT,signal.SIG_IGN);time.sleep(120)'])\ntime.sleep(120)\n")
    details = {}
    try:
        services.command(["systemd-run", "--user", "--unit=" + hanging, "--property=Type=exec", "--property=KillMode=control-group",
            "--property=KillSignal=SIGINT", "--property=TimeoutStopSec=2", "/usr/bin/python3", str(script)])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            before = show(hanging)
            group = Path("/sys/fs/cgroup") / before.get("ControlGroup", "").lstrip("/")
            pids = (group / "cgroup.procs").read_text().split() if (group / "cgroup.procs").exists() else []
            if len(pids) >= 2: break
            time.sleep(.05)
        checks["fixture_parent_and_child_observed"] = len(pids) >= 2
        started = time.monotonic()
        services.command(["systemctl", "--user", "stop", hanging], timeout=8)
        elapsed = time.monotonic() - started
        remaining = (group / "cgroup.procs").read_text().split() if (group / "cgroup.procs").exists() else []
        checks["bounded_force_stop_clears_children"] = not remaining and elapsed < 5
        details["forced_stop_seconds"] = elapsed
        services.command(["systemd-run", "--user", "--unit=" + failing, "--property=Type=exec",
            "--property=Restart=on-failure", "--property=RestartSec=100ms",
            "--property=StartLimitIntervalSec=10", "--property=StartLimitBurst=3", "/usr/bin/false"])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            limited = show(failing)
            if limited.get("ActiveState") == "failed" and int(limited.get("NRestarts", 0)) >= 3: break
            time.sleep(.1)
        time.sleep(.8)  # Eight configured restart intervals; the count must stop growing.
        stable = show(failing)
        checks["restart_storm_limited"] = (stable.get("ActiveState") == "failed" and stable.get("Restart") == "on-failure"
            and stable.get("StartLimitBurst") == "3" and stable.get("NRestarts") == limited.get("NRestarts") == "3"
            and stable.get("MainPID") == "0")
        details["restart_stable_after_seconds"] = .8
        details["restart_limit"] = limited
    finally:
        for name in (hanging, failing):
            services.command(["systemctl", "--user", "stop", name], check=False)
            services.command(["systemctl", "--user", "reset-failed", name], check=False)
    if services.build_running():
        try: services.preflight(["llama"])
        except RuntimeError as error: checks["live_gpu_build_blocks_model_start"] = "build" in str(error)
        else: checks["live_gpu_build_blocks_model_start"] = False
    report = {"created_at": stamp(), "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "real user systemd and camera lifecycle; readiness failure and hanging/restarting processes are labelled fixtures; no model inference, 8h or full assistant acceptance",
        "checks": checks, "camera_active": active, "camera_stopped": stopped, "fixtures": details}
    write_json(ROOT / "benchmarks/results/service-lifecycle-smoke.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
