#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise native RGBD + B1 + a real text/4B/TTS turn, restoring prior services.

This is a functional coexistence probe, not a performance baseline. It does
not enroll camera subjects or claim physical microphone/speaker validation.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_src"))
from deploy import runtime, services


def http(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def traced_stop(cfg, output):
    """Observe delayed teardown without signalling a finalizing interpreter.

    Python can unregister faulthandler before native destructors return. A user
    signal then would terminate the process and corrupt the evidence.
    """
    done = threading.Event()

    def inspect():
        if done.wait(8):
            return
        rows = []
        for pid in services.properties("app")["cgroup_pids"]:
            proc = Path("/proc") / str(pid)
            try:
                argv = proc.joinpath("cmdline").read_bytes().split(b"\0")
                if os.fsencode(ROOT / "run_web_vision_chat.py") not in argv:
                    continue
                row = {"pid": pid}
                for name in ("wchan", "status", "stat", "io", "syscall"):
                    try:
                        row[name] = proc.joinpath(name).read_text()
                    except (PermissionError, FileNotFoundError) as exc:
                        row[name] = repr(exc)
                rows.append(row)
            except (FileNotFoundError, ProcessLookupError):
                continue
        runtime.write_json(output / "shutdown-stall.json", rows)

    watcher = threading.Thread(target=inspect, name="shutdown-probe", daemon=True)
    watcher.start()
    try:
        return services.stop(list(services.ORDER), cfg)
    finally:
        done.set()
        watcher.join(5)


def wait_turn(base, turn_id, seconds=90):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = http(base, "/api/state")
        result = value.get("last_result") or {}
        if result.get("turn_id") == turn_id and value.get("phase") == "idle":
            return value
        time.sleep(0.25)
    raise TimeoutError("Turn did not settle: " + turn_id)


def probe(output, webdriver=None):
    cfg = runtime.config()
    cfg = runtime.config(services.profile_path(cfg))
    b1 = runtime.config(ROOT / "02_config/settings.orin-nx-b1.yaml")
    # Automated greetings could take admission while the probe submits a turn.
    # This test-specific override is saved in services-started.json.
    b1.gestures.greetings_enabled = False
    base = "http://127.0.0.1:" + str(runtime.ports(b1)["app"])
    report = {"created_at": runtime.stamp(), "status": "RUNNING", "scope": __doc__}
    with services.operation(cfg):
        before = services.state(list(services.ORDER), cfg)
        runtime.write_json(output / "services-before.json", before)
        for name, row in before["services"].items():
            if row.get("cgroup_pids") and not row.get("managed_by_project"):
                raise RuntimeError("Probe will not stop an unrelated service: " + name)
        if before["services"]["app"].get("health", {}).get("phase") not in (
            None,
            "idle",
        ):
            raise RuntimeError("Wait for the current application turn")
        active = [
            n
            for n in services.ORDER
            if before["services"][n].get("ActiveState") == "active"
        ]
        started = False
        try:
            stopped = services.stop(list(services.ORDER), cfg)
            runtime.write_json(output / "services-stopped-before.json", stopped)
            if any(r["cgroup_pids"] for r in stopped["services"].values()):
                raise RuntimeError("Managed processes remain")
            runtime.write_json(
                output / "services-started.json",
                services.start(list(services.ORDER), b1, 240),
            )
            started = True
            samples = []
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                value = http(base, "/api/state")
                samples.append(value)
                if (
                    value["camera"].get("depth_ready")
                    and value["perception"].get("ready")
                    and value["perception"].get("processed", 0) >= 10
                ):
                    break
                time.sleep(0.5)
            runtime.write_json(output / "scene-samples.json", samples)
            if not (
                value["camera"].get("depth_ready") and value["perception"].get("ready")
            ):
                raise RuntimeError("Native depth or B1 perception not ready")
            report["identities"] = http(base, "/api/vision/identities")
            response = http(
                base,
                "/api/chat",
                {"text": "只回复：框架检查完成。不要调用工具。", "use_image": False},
            )
            result = wait_turn(base, response["turn_id"])
            runtime.write_json(output / "turn-state.json", result)
            turn = result["last_result"]
            if turn.get("status") != "complete" or not turn.get("audio_url"):
                raise RuntimeError("Text/4B/TTS turn did not complete: " + str(turn))
            with urlopen(base + turn["audio_url"], timeout=5) as response:
                wav = response.read()
            if wav[:4] != b"RIFF" or len(wav) <= 44:
                raise RuntimeError("Published speech WAV is invalid")
            (output / "reply.wav").write_bytes(wav)
            # Feed the generated fixture through the actual VAD/ASR/4B/TTS path.
            request = Request(
                base + "/api/audio", data=wav, headers={"Content-Type": "audio/wav"}
            )
            with urlopen(request, timeout=5) as response:
                accepted = json.load(response)
            audio_state = wait_turn(base, accepted["turn_id"])
            runtime.write_json(output / "audio-turn-state.json", audio_state)
            audio_turn = audio_state["last_result"]
            if (
                audio_turn["status"] != "complete"
                or not audio_turn.get("asr")
                or not audio_turn.get("audio_url")
            ):
                raise RuntimeError("Audio-file/ASR/4B/TTS turn did not complete")
            report["audio_input_turn"] = audio_turn["turn_id"]
            accepted = http(
                base,
                "/api/chat",
                {
                    "text": "请详细解释十种编程学习方法，每一种写出五个具体步骤。不要调用工具。"
                },
            )
            time.sleep(1)
            report["cancellation"] = http(
                base, "/api/cancel", {"turn_id": accepted["turn_id"]}
            )
            cancelled = wait_turn(base, accepted["turn_id"])
            runtime.write_json(output / "cancel-state.json", cancelled)
            if cancelled["last_result"]["status"] != "cancelled" or cancelled[
                "last_result"
            ].get("audio_url"):
                raise RuntimeError("Cancellation did not suppress turn publication")
            if webdriver:
                from probe_b1_browser import probe as browser_probe

                report["browser_status"] = browser_probe(base, webdriver, output)[
                    "status"
                ]
            report.update(
                status="FUNCTIONAL_COEXISTENCE_PASS_NOT_B1_BASELINE",
                turn_id=turn["turn_id"],
                audio_bytes=len(wav),
            )
        except BaseException as exc:
            report.update(status="FAILED", error=repr(exc))
            raise
        finally:
            try:
                stopped = (
                    traced_stop(b1, output)
                    if started
                    else services.stop(list(services.ORDER), b1)
                )
                runtime.write_json(output / "services-stopped-after.json", stopped)
                report["clean_shutdown"] = stopped["status"] == "STOPPED"
                if not report["clean_shutdown"]:
                    report["status"] = "FAILED_SHUTDOWN"
            finally:
                restored = (
                    services.start(active, cfg, 240)
                    if active
                    else services.state(list(services.ORDER), cfg, False)
                )
                runtime.write_json(output / "services-restored.json", restored)
                runtime.write_json(output / "report.json", report)
    if report["status"] != "FUNCTIONAL_COEXISTENCE_PASS_NOT_B1_BASELINE":
        raise RuntimeError("Probe did not pass; inspect " + str(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--webdriver", help="Optional local Firefox WebDriver URL")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(json.dumps(probe(output, args.webdriver), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
