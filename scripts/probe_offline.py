#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cold-start prepared native artifacts in a loopback-only network namespace.

The outer process stops/restores only our four managed services. Root is used
only to create the temporary network namespace; inference runs as the original
user. Host interfaces/routes are never modified. This is not a clean rebuild,
machine reboot or a test of systemd itself inside the namespace.
"""

import argparse
import errno
import json
import os
from pathlib import Path
import pwd
import signal
import socket
import subprocess
import sys
import time

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy import runtime, services  # noqa: E402


def network_state():
    return {
        "namespace": os.readlink("/proc/self/ns/net"),
        "interfaces": json.loads(
            subprocess.check_output(["/usr/sbin/ip", "-j", "address"], text=True)
        ),
        "ipv4_routes": json.loads(
            subprocess.check_output(["/usr/sbin/ip", "-j", "route"], text=True)
        ),
        "ipv6_routes": json.loads(
            subprocess.check_output(["/usr/sbin/ip", "-j", "-6", "route"], text=True)
        ),
    }


def topology(value):
    """DHCP lifetimes can decrease while the actual host topology is unchanged."""
    if isinstance(value, dict):
        return {
            k: topology(v)
            for k, v in value.items()
            if k not in {"valid_life_time", "preferred_life_time", "expires"}
        }
    if isinstance(value, list):
        return [topology(v) for v in value]
    return value


def remaining_namespace(namespace):
    if namespace == os.readlink("/proc/self/ns/net"):
        raise RuntimeError("Refusing cleanup of the host namespace")
    result = []
    for process in psutil.process_iter(["pid", "uids"]):
        if not process.info["uids"] or process.info["uids"].effective != os.geteuid():
            continue
        try:
            if os.readlink(f"/proc/{process.pid}/ns/net") == namespace:
                result.append(process)
        except (FileNotFoundError, PermissionError, psutil.NoSuchProcess):
            continue
    return result


def cleanup_namespace(namespace):
    """Only our unique namespace/user, even if its launcher timed out or died."""
    leftovers = remaining_namespace(namespace)
    initial = [p.pid for p in leftovers]
    for sig in (signal.SIGINT, signal.SIGKILL):
        for process in leftovers:
            try:
                handle = os.pidfd_open(process.pid)
                try:
                    if (
                        process.is_running()
                        and os.readlink(f"/proc/{process.pid}/ns/net") == namespace
                    ):
                        signal.pidfd_send_signal(handle, sig)
                finally:
                    os.close(handle)
            except (ProcessLookupError, FileNotFoundError, psutil.NoSuchProcess):
                pass
        deadline = time.monotonic() + 10
        while leftovers and time.monotonic() < deadline:
            time.sleep(0.1)
            leftovers = remaining_namespace(namespace)
        if not leftovers:
            break
    if leftovers:
        raise RuntimeError("Owned namespace processes did not exit")
    return initial


def inside(args):
    out = args.output
    cfg = runtime.config(out / "settings.json")
    initial = json.loads((out / "outer.json").read_text())
    network = network_state()
    if (
        os.geteuid() != initial["uid"]
        or network["namespace"] == initial["network"]["namespace"]
        or {x["ifname"] for x in network["interfaces"]} != {"lo"}
        or network["ipv4_routes"]
    ):
        raise RuntimeError("Network isolation or unprivileged user was not established")
    with socket.socket() as connection:
        connection.settimeout(1)
        unreachable = connection.connect_ex(("198.51.100.1", 80))
    if unreachable != errno.ENETUNREACH:
        raise RuntimeError("Expected kernel ENETUNREACH for a non-loopback address")
    report = {
        "status": "RUNNING",
        "created_at": runtime.stamp(),
        "uid": os.geteuid(),
        "network": network,
        "external_connect_errno": unreachable,
        "components": {},
        "checks": [],
    }
    children = {}
    logs = []
    error = None

    def save():
        runtime.write_json(out / "inside.json", report)

    save()

    def check(label, argv):
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(
                [str(x) for x in argv],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=240,
            )
        report["checks"].append(
            {
                "label": label,
                "command": [str(x) for x in argv],
                "returncode": result.returncode,
            }
        )
        save()
        if result.returncode:
            raise RuntimeError(label + " failed")

    try:
        for name in services.ORDER:
            print("offline start: " + name, flush=True)
            services.require_headroom()
            log = (out / (name + ".log")).open("w")
            logs.append(log)
            child = subprocess.Popen(
                [
                    sys.executable,
                    ROOT / "deploy/runtime.py",
                    name,
                    "--profile",
                    out / "settings.json",
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children[name] = child
            deadline = time.monotonic() + 240
            while True:
                if child.poll() is not None:
                    raise RuntimeError(name + " exited during offline startup")
                try:
                    health = services.health(name, cfg)
                    if health["ready"]:
                        report["components"][name] = {
                            "pid": child.pid,
                            "namespace": os.readlink(f"/proc/{child.pid}/ns/net"),
                            "health": health,
                        }
                        if (
                            report["components"][name]["namespace"]
                            != network["namespace"]
                        ):
                            raise RuntimeError(
                                "Inference process escaped the test namespace"
                            )
                        break
                except (httpx.HTTPError, OSError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError(name + " offline startup timed out")
                time.sleep(0.25)
            save()
        python = runtime.paths(cfg)["app_python"]
        for label, extra in [
            ("text", ["--text", "只回答：离线运行成功"]),
            (
                "camera",
                ["--camera", "--text", "用一句话描述当前画面，不执行任何工具。"],
            ),
        ]:
            check(
                label,
                [
                    python,
                    ROOT / "scripts/probe_web.py",
                    *extra,
                    "--output",
                    out / (label + ".json"),
                ],
            )
        check(
            "file-speech",
            [
                python,
                ROOT / "scripts/probe_speech.py",
                "--input",
                args.input,
                "--output",
                out / "reply.wav",
            ],
        )
        report["status"] = "PASS"
    except BaseException as exc:
        error = exc
        report["status"] = "FAIL"
        report["error"] = repr(exc)
    finally:
        report["shutdown"] = {}
        for name, child in reversed(children.items()):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(10)
                    error = error or RuntimeError(
                        name + " required forced offline cleanup"
                    )
            report["shutdown"][name] = child.returncode
            if child.returncode not in {0, -signal.SIGINT, 130}:
                error = error or RuntimeError(name + " exited uncleanly")
        for log in logs:
            log.close()
        report["remaining_owned_processes"] = [
            p.pid for p in psutil.Process().children(recursive=True) if p.is_running()
        ]
        if report["remaining_owned_processes"]:
            error = error or RuntimeError("Offline child processes remain")
        report["status"] = "FAIL" if error else "PASS"
        report["finished_at"] = runtime.stamp()
        save()
    if error:
        raise error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inside:
        return inside(args)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = runtime.config(services.profile_path(runtime.config()))
    before = services.state(services.ORDER, cfg)
    if before["status"] != "READY":
        raise RuntimeError("The owned managed stack must be ready")
    if before["services"]["app"]["health"]["phase"] != "idle" or any(
        before["services"][name]["health"]["processing"] for name in ("llama", "asr")
    ):
        raise RuntimeError("Wait for the active application/model request to finish")
    report = {
        "status": "RUNNING",
        "created_at": runtime.stamp(),
        "scope": __doc__,
        "uid": os.getuid(),
        "network": network_state(),
        "before_services": before,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "probe_sha256": runtime.sha256(__file__),
        "input_sha256": runtime.sha256(args.input),
    }
    runtime.write_json(out / "outer.json", report)
    isolated = runtime.config(services.profile_path(cfg))
    isolated.actions.database = str(out / "actions.sqlite")
    isolated.audio.output_directory = str(out / "audio")
    runtime.write_json(out / "settings.json", isolated.effective())
    # Preflight sudo without changing the running stack.
    subprocess.run(["sudo", "-n", "true"], check=True)
    error = None
    try:
        with services.operation(cfg):
            report["managed_stopped"] = services.stop(list(services.ORDER), cfg)
        username = pwd.getpwuid(os.getuid()).pw_name
        command = [
            "sudo",
            "-n",
            "unshare",
            "--net",
            "bash",
            "-c",
            '/usr/sbin/ip link set lo up && exec sudo -n -H -u "$1" -- "${@:2}"',
            "namespace",
            username,
            sys.executable,
            __file__,
            "--inside",
            "--output",
            str(out),
            "--input",
            str(args.input.resolve()),
        ]
        report["namespace_command"] = command
        with (out / "namespace.log").open("w") as log:
            result = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1500
            )
        report["namespace_exit_code"] = result.returncode
        if result.returncode:
            raise RuntimeError("Offline namespace check failed; inspect namespace.log")
        report["inside"] = json.loads((out / "inside.json").read_text())
        if report["inside"]["status"] != "PASS":
            raise RuntimeError("Incomplete offline check")
    except BaseException as exc:
        error = exc
        report["error"] = repr(exc)
    finally:
        if (out / "inside.json").is_file():
            try:
                inside_report = json.loads((out / "inside.json").read_text())
                remaining = cleanup_namespace(inside_report["network"]["namespace"])
                report["outer_namespace_cleanup"] = remaining
                if remaining:
                    error = error or RuntimeError(
                        "Offline processes needed outer cleanup"
                    )
            except BaseException as exc:
                error = error or exc
                report["namespace_cleanup_error"] = repr(exc)
        report["network_after"] = network_state()
        if topology(report["network_after"]) != topology(report["network"]):
            error = error or RuntimeError(
                "Host network snapshot changed; inspect before/after"
            )
        try:
            with services.operation(cfg):
                report["restored"] = services.start(list(services.ORDER), cfg, 240)
        except BaseException as exc:
            error = error or exc
            report["restore_error"] = repr(exc)
        report["status"] = "FAIL" if error else "PASS"
        report["finished_at"] = runtime.stamp()
        runtime.write_json(out / "report.json", report)
    if error:
        raise error
    print(json.dumps({"status": "PASS", "report": str(out / "report.json")}))


if __name__ == "__main__":
    main()
