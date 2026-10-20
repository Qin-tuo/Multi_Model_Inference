#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a pinned B1 candidate while exclusively owning the heavy-build lease.

Only this project's managed model services are stopped. Their frozen profile
and previously active components are restored even when the build fails.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_src"))
from deploy import runtime, services
from deploy.operations import exclusive_build
from prepare_vision_b1 import MANIFEST, prepare




def build(role, entry, output, timeout, workspace_mib=512, *, prepare_candidate=None):
    import psutil

    (prepare_candidate or prepare)(role, entry, check_only=True)
    folder = output / role
    folder.mkdir(parents=True, exist_ok=True)
    engine = folder / "model.fp32.engine"
    report = folder / "build.json"
    if engine.exists() or report.exists():
        raise FileExistsError("Use a new build output directory: " + str(folder))
    specs = entry["observed_onnx_contract"]
    shapes = {}
    for value in specs["inputs"]:
        shape = value["shape"]
        if any(not isinstance(d, int) for d in shape[1:]):
            raise ValueError("Specify a concrete spatial contract before building")
        shapes[value["name"]] = [1 if not isinstance(d, int) else d for d in shape]
    argv = [
        "/usr/src/tensorrt/bin/trtexec",
        "--onnx=" + str(ROOT / entry["path"]),
        "--saveEngine=" + str(engine),
        "--noTF32",
        f"--memPoolSize=workspace:{workspace_mib}",
        "--skipInference",
        "--profilingVerbosity=detailed",
    ]
    if any(not isinstance(d, int) for v in specs["inputs"] for d in v["shape"]):
        value = ",".join(
            name + ":" + "x".join(map(str, shape)) for name, shape in shapes.items()
        )
        argv += [
            "--" + name + "=" + value
            for name in ("minShapes", "optShapes", "maxShapes")
        ]
    available = psutil.virtual_memory().available
    if available < 6 * 1024**3:
        raise RuntimeError("Less than 6 GiB available for this candidate build")
    record = {
        "created_at": runtime.stamp(),
        "role": role,
        "status": "BUILDING",
        "onnx_sha256": entry["sha256"],
        "argv": argv,
        "input_shapes": shapes,
        "available_before_bytes": available,
        "precision": "FP32; TF32 disabled",
        "scope": "Build feasibility, not B1 performance or functional acceptance",
        "workspace_mib": workspace_mib,
        "build_script_sha256": runtime.sha256(Path(__file__)),
    }
    runtime.write_json(report, record)
    start = time.monotonic()
    process = None
    try:
        with (folder / "build.log").open("w") as log:
            process = subprocess.Popen(
                argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            record["build_pid"] = process.pid
            runtime.write_json(report, record)
            # Observe host pressure during compiler/tactic selection, keeping
            # these build samples separate from runtime performance evidence.
            with (folder / "build-memory.jsonl").open("w") as samples:
                while process.poll() is None:
                    if time.monotonic() - start > timeout:
                        raise subprocess.TimeoutExpired(argv, timeout)
                    memory = psutil.virtual_memory()
                    try:
                        rss = psutil.Process(process.pid).memory_info().rss
                    except psutil.NoSuchProcess:
                        rss = 0
                    samples.write(
                        json.dumps(
                            {
                                "elapsed_seconds": time.monotonic() - start,
                                "available_bytes": memory.available,
                                "build_rss_bytes": rss,
                            }
                        )
                        + "\n"
                    )
                    samples.flush()
                    time.sleep(1)
            returncode = process.returncode
        record["returncode"] = returncode
        if returncode != 0 or not engine.is_file():
            raise RuntimeError(
                "TensorRT build failed; inspect " + str(folder / "build.log")
            )
        record.update(
            status="BUILT_NOT_ACCEPTED",
            engine_path=str(engine),
            engine_sha256=runtime.sha256(engine),
            engine_bytes=engine.stat().st_size,
        )
    except BaseException as exc:
        record.update(status="BUILD_FAILED", error=str(exc))
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        record["elapsed_seconds"] = time.monotonic() - start
        runtime.write_json(report, record)
    return record


def main():
    entries = json.loads(MANIFEST.read_text())["models"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", nargs="+", choices=tuple(entries))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument(
        "--workspace-mib", type=int, choices=(512, 1024, 2048, 4096), default=512
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = runtime.config()
    profile = services.profile_path(cfg)
    if profile.exists():
        cfg = runtime.config(profile)
    selected = (
        args.role
        if args.role
        else [
            role
            for role, item in entries.items()
            if item.get("phase") != "DEFERRED_EXPERIMENT"
        ]
    )
    failures = []
    with exclusive_build(cfg, output):
        for role in selected:
            try:
                result = build(
                    role, entries[role], output, args.timeout, args.workspace_mib
                )
                print(role, result["status"], result["engine_bytes"], flush=True)
            except Exception as exc:
                failures.append(role)
                print(role, "FAILED:", exc, flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
