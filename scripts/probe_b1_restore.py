#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify a restored B1 checkout offline with both source runtime trees hidden.

Reuses the native offline probe. Original services are stopped/restored under
their operation lock; the restored checkout never registers systemd units.
Host mounts/routes are unchanged and inference runs as the original user.
This is a same-host restore test, not a clean-OS or cross-Jetson certification.
"""

import argparse
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_src"))
from deploy import runtime, services
from project_paths import LEGACY_STACK_DIR
from probe_offline import cleanup_namespace, network_state, topology


def probe(checkout, output):
    hidden = [ROOT, LEGACY_STACK_DIR]
    if any(
        checkout.is_relative_to(source) or output.is_relative_to(source)
        for source in hidden
    ):
        raise ValueError(
            "Restored checkout and output must be outside both hidden source trees"
        )
    python = checkout / "05_runtime/environments/edge-assistant/bin/python"
    script = checkout / "scripts/probe_offline.py"
    audio = checkout / "05_runtime/datasets/b0-sw-v1/zh-20.wav"
    profile = checkout / "02_config/settings.orin-nx.yaml"
    if not all(p.is_file() for p in (python, script, audio, profile)):
        raise ValueError("Prepare the Git checkout and unpack its runtime first")
    # Read defaults using the restored project itself, without importing two
    # different app packages into this controller's interpreter.
    config_json = subprocess.check_output(
        [
            str(python),
            "-c",
            "import json,sys; sys.path.insert(0,'01_src'); from app.config import Config; print(json.dumps(Config.load(sys.argv[1]).effective()))",
            str(profile),
        ],
        cwd=checkout,
        text=True,
    )
    isolated = json.loads(config_json)
    if Path(isolated["backends"]["stack_root"]) != checkout / "05_runtime":
        raise RuntimeError("Restored configuration points at another runtime")
    output.mkdir(parents=True, exist_ok=False)
    isolated["actions"]["database"] = str(output / "actions.sqlite")
    isolated["audio"]["output_directory"] = str(output / "audio")
    gallery = (
        checkout
        / "05_runtime/runtime-data/restore-probe"
        / output.name
        / "identities.sqlite"
    )
    if gallery.exists():
        raise FileExistsError("Use a new isolated identity gallery")
    isolated["perception"]["identity_gallery"] = str(gallery)
    runtime.write_json(output / "settings.json", isolated)
    empty = output / "empty-source"
    empty.mkdir()
    subprocess.run(["sudo", "-n", "true"], check=True)
    prior = runtime.config(services.profile_path(runtime.config()))
    with services.operation(prior):
        before = services.state(services.ORDER, prior)
        if (
            before["status"] != "READY"
            or before["services"]["app"]["health"]["phase"] != "idle"
        ):
            raise RuntimeError("Original managed stack must be ready and idle")
        if any(
            before["services"][name]["health"]["processing"]
            for name in ("llama", "asr")
        ):
            raise RuntimeError("Wait for the active backend request")
        report = {
            "status": "RUNNING",
            "created_at": runtime.stamp(),
            "scope": __doc__,
            "uid": os.getuid(),
            "network": network_state(),
            "mount_namespace": os.readlink("/proc/self/ns/mnt"),
            "hidden_sources": [str(p) for p in hidden],
            "checkout": str(checkout),
            "before_services": before,
            "checkout_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
            ).strip(),
            "restored_manifest_sha256": runtime.sha256(
                checkout / "05_runtime/bundle.manifest.json"
            ),
            "probe_sha256": runtime.sha256(__file__),
            "inside_probe_sha256": runtime.sha256(script),
        }
        runtime.write_json(output / "outer.json", report)
        command = [
            "sudo",
            "-n",
            "unshare",
            "--net",
            "--mount",
            "--propagation",
            "private",
            "bash",
            "-c",
            'mount --bind "$1" "$2" && mount -o remount,bind,ro "$2" && mount --bind "$1" "$3" && mount -o remount,bind,ro "$3" && /usr/sbin/ip link set lo up && exec sudo -n -H -u "$4" -- "${@:5}"',
            "namespace",
            str(empty),
            str(ROOT),
            str(LEGACY_STACK_DIR),
            pwd.getpwuid(os.getuid()).pw_name,
            str(python),
            str(script),
            "--inside",
            "--output",
            str(output),
            "--input",
            str(audio),
        ]
        report["namespace_command"] = command
        error = None
        try:
            stopped = services.stop(list(services.ORDER), prior)
            report["managed_stopped"] = stopped
            if any(row["cgroup_pids"] for row in stopped["services"].values()):
                raise RuntimeError("Original service processes remain")
            with (output / "namespace.log").open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=checkout,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=1500,
                )
            report["namespace_exit_code"] = result.returncode
            if result.returncode:
                raise RuntimeError(
                    "Restored offline probe failed; inspect namespace.log"
                )
            report["inside"] = json.loads((output / "inside.json").read_text())
            if report["inside"]["status"] != "PASS" or not report["inside"].get("b1"):
                raise RuntimeError("Incomplete restored B1 check")
        except BaseException as exc:
            error = exc
            report["error"] = repr(exc)
        finally:
            try:
                if (output / "inside.json").is_file():
                    inner = json.loads((output / "inside.json").read_text())
                    leftovers = cleanup_namespace(inner["network"]["namespace"])
                    report["outer_namespace_cleanup"] = leftovers
                    if leftovers:
                        raise RuntimeError("Restored processes required outer cleanup")
                report["network_after"] = network_state()
                if topology(report["network_after"]) != topology(report["network"]):
                    raise RuntimeError("Host network topology changed")
                if os.readlink("/proc/self/ns/mnt") != report[
                    "mount_namespace"
                ] or not all(any(p.iterdir()) for p in hidden):
                    raise RuntimeError("Host source mount state changed")
            except BaseException as exc:
                error = error or exc
                report["cleanup_error"] = repr(exc)
            try:
                report["restored"] = services.start(list(services.ORDER), prior, 240)
            except BaseException as exc:
                error = error or exc
                report["restore_error"] = repr(exc)
            report.update(
                status="FAIL" if error else "PASS", finished_at=runtime.stamp()
            )
            runtime.write_json(output / "report.json", report)
        if error:
            raise error
    print(json.dumps({"status": "PASS", "report": str(output / "report.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe(args.checkout.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
