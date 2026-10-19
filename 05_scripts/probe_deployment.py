#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rehearse managed deployment, complete-app cycles and a real legacy rollback.

Requires the owned legacy stack to be running and new services to be stopped.
Always restores the initially active legacy components, including fresh RGB/depth.
Raw files stay outside Git. This is a functional probe, not B0 or an endurance run.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_src"))
from deploy import runtime, services  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.cycles <= 20:
        raise ValueError("Choose 1..20 complete-app cycles")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    legacy = args.legacy_root.resolve()
    cfg = runtime.config()
    if cfg.audio.input_mode != "file" or cfg.audio.output_mode != "file":
        raise ValueError("Probe requires the software/file audio profile")
    report = {
        "created_at": runtime.stamp(),
        "scope": "Prepared deployment and rollback; no physical audio, B0 or 8h claim",
        "source_sha256": {
            str(p.relative_to(ROOT)): runtime.sha256(p)
            for folder in ["01_src/app", "01_src/deploy", "05_scripts"]
            for p in (ROOT / folder).rglob("*.py")
        },
        "legacy_root": str(legacy),
        "steps": [],
        "cycles": [],
        "status": "RUNNING",
    }

    def save():
        runtime.write_json(out / "report.json", report)

    def run(label, argv, *, cwd=ROOT, env=None):
        print(label, flush=True)
        start = time.monotonic()
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(
                [str(a) for a in argv],
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1200,
            )
        report["steps"].append(
            {
                "label": label,
                "command": [str(a) for a in argv],
                "returncode": result.returncode,
                "elapsed_seconds": time.monotonic() - start,
            }
        )
        save()
        if result.returncode:
            raise RuntimeError(f"{label} failed; inspect {out / (label + '.log')}")

    def old(action, components="all"):
        return [
            "/usr/bin/python3",
            legacy / "tools/project.py",
            action,
            "--components",
            components,
        ]

    report["legacy_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=legacy, text=True
    ).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=legacy, text=True
    ).strip():
        raise RuntimeError("Rollback source must be clean")
    initial = services.state(services.ORDER, cfg, False)
    if initial["status"] != "STOPPED":
        raise RuntimeError("New managed services must be stopped before rehearsal")
    run("legacy-before", old("status"), cwd=legacy)
    before = json.loads((out / "legacy-before.log").read_text())
    if before["backend_status"] != "READY":
        raise RuntimeError("Legacy stack is not ready; no cutover performed")
    report["legacy_before"] = before
    active = ",".join(before["components"])
    report["legacy_unit_sha256"] = {
        name: runtime.sha256(Path(row["FragmentPath"]).resolve())
        for name, row in before["components"].items()
    }
    report["effective_config"] = cfg.effective()
    save()
    failed = None
    try:
        run("legacy-down", old("down", active), cwd=legacy)
        print("new-up", flush=True)
        with services.operation(cfg):
            report["new_ready"] = services.start(list(services.ORDER), cfg, 240)
        save()
        python = runtime.paths(cfg)["app_python"]
        for label, args_ in [
            ("text", ["--text", "只回答：连接成功"]),
            (
                "camera",
                ["--camera", "--text", "用一句话描述当前画面，不执行任何工具。"],
            ),
        ]:
            run(
                label,
                [
                    python,
                    ROOT / "05_scripts/probe_web.py",
                    *args_,
                    "--output",
                    out / (label + ".json"),
                ],
            )
        # Known synthetic Chinese fixture; no user microphone/audio is recorded.
        import httpx

        response = httpx.post(
            cfg.tts.base_url + "/audio/speech",
            json={
                "model": cfg.tts.model,
                "voice": cfg.tts.voice,
                "input": "请只回答，测试成功。",
                "speed": cfg.tts.speed,
                "response_format": "pcm",
            },
            timeout=90,
            trust_env=False,
        )
        response.raise_for_status()
        with wave.open(str(out / "input.wav"), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(response.content)
        run(
            "file-speech",
            [
                python,
                ROOT / "05_scripts/probe_speech.py",
                "--input",
                out / "input.wav",
                "--output",
                out / "reply.wav",
            ],
        )
        # Dependencies remain resident: exercise the full application lifecycle,
        # including camera, all enabled small models, VAD and action scheduler.
        for index in range(args.cycles):
            print(f"app cycle {index + 1}/{args.cycles}", flush=True)
            started = time.monotonic()
            with services.operation(cfg):
                stopped = services.stop(["app"], cfg)
                if stopped["status"] != "STOPPED":
                    raise RuntimeError("Application did not fully stop")
                ready = services.start(["app"], cfg, 240)
            row = ready["services"]["app"]
            report["cycles"].append(
                {
                    "cycle": index + 1,
                    "stopped": stopped,
                    "ready": ready,
                    "elapsed_seconds": time.monotonic() - started,
                    "process_status": {
                        str(pid): (Path("/proc") / str(pid) / "status").read_text()
                        for pid in row["cgroup_pids"]
                    },
                }
            )
            save()
        run(
            "after-cycles",
            [
                python,
                ROOT / "05_scripts/probe_web.py",
                "--text",
                "只回答：恢复成功",
                "--output",
                out / "after-cycles.json",
            ],
        )
        # Kill the real app process via its pidfd after verifying its cgroup
        # and command. systemd, not this probe, must restart the failed app.
        before_crash = services.state(["app"], cfg)
        group = before_crash["services"]["app"]
        candidates = []
        for pid in group["cgroup_pids"]:
            argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
            if str(ROOT / "run_web_vision_chat.py").encode() in argv:
                candidates.append(pid)
        if len(candidates) != 1 or before_crash["status"] != "READY":
            raise RuntimeError("Cannot identify one owned ready app process")
        pid = candidates[0]
        fd = os.pidfd_open(pid)
        try:
            if pid not in services.properties("app")["cgroup_pids"]:
                raise RuntimeError("App ownership changed before crash injection")
            signal.pidfd_send_signal(fd, signal.SIGKILL)
        finally:
            os.close(fd)
        deadline = time.monotonic() + 120
        while True:
            after_crash = services.state(["app"], cfg)
            new_group = after_crash["services"]["app"]
            if (
                after_crash["status"] == "READY"
                and new_group["main_pid"] != group["main_pid"]
                and pid not in new_group["cgroup_pids"]
            ):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Systemd did not recover the crashed application")
            time.sleep(0.25)
        report["app_crash_recovery"] = {
            "method": "SIGKILL to pidfd of owned real app, no probe restart",
            "killed_pid": pid,
            "before": before_crash,
            "after": after_crash,
        }
        save()
        run(
            "after-crash",
            [
                python,
                ROOT / "05_scripts/probe_web.py",
                "--text",
                "只回答：恢复成功",
                "--output",
                out / "after-crash.json",
            ],
        )
        report["software_status"] = "PASS"
    except BaseException as exc:
        failed = exc
        report["error"] = repr(exc)
    finally:
        print("restore legacy stack", flush=True)
        try:
            with services.operation(cfg):
                report["new_stopped"] = services.stop(list(services.ORDER), cfg)
            run("legacy-up", old("up", active), cwd=legacy)
            run("legacy-restored", old("status", active), cwd=legacy)
            restored = json.loads((out / "legacy-restored.log").read_text())
            if restored["backend_status"] != "READY":
                raise RuntimeError("Legacy readiness was not restored")
            report["legacy_restored"] = restored
            env = dict(
                os.environ,
                ROS_DOMAIN_ID="42",
                ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST",
            )
            run(
                "legacy-camera",
                [
                    "bash",
                    "-c",
                    'source /opt/ros/jazzy/setup.bash && exec /usr/bin/python3 "$1" "$2"',
                    "probe",
                    ROOT / "05_scripts/probe_legacy_camera.py",
                    out / "legacy-camera.json",
                ],
                env=env,
            )
            report["rollback_status"] = "PASS"
        except BaseException as exc:
            report["rollback_error"] = repr(exc)
            failed = failed or exc
        report["status"] = "FAIL" if failed else "PASS"
        report["finished_at"] = runtime.stamp()
        save()
    if failed:
        raise failed
    print(json.dumps({"status": report["status"], "report": str(out / "report.json")}))


if __name__ == "__main__":
    main()
