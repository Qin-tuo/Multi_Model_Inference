#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build selected current models while exclusively owning the heavy-build lease.

Previously active services are restored even when a build fails. The generated
candidate manifest stays in the output directory until separately validated.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "01_src"))
from deploy import runtime, services
from deploy.operations import exclusive_build
from deploy.vision.artifacts import MANIFEST, prepare
from app.vision_models import contract_hash




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
        "scope": "Build feasibility; model parity and performance are separate checks",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--role", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument(
        "--workspace-mib", type=int, choices=(512, 1024, 2048, 4096), default=512
    )
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())["models"]
    selected = args.role or list(entries)
    if set(selected) - set(entries):
        parser.error("Unknown role for selected model manifest")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = runtime.config()
    profile = services.profile_path(cfg)
    if profile.exists():
        cfg = runtime.config(profile)
    candidate = deepcopy(json.loads((ROOT / cfg.perception.model_manifest).read_text()))
    source = args.manifest.resolve()
    candidate["source_manifest"] = str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source)
    candidate["status"] = "BUILT_REFERENCE_AND_INTEGRATION_CHECKS_PENDING"
    failures = []
    with exclusive_build(cfg, output):
        for role in selected:
            try:
                result = build(
                    role, entries[role], output, args.timeout, args.workspace_mib
                )
                engine = Path(result["engine_path"])
                candidate["models"][role] = {
                    "role": role,
                    "engine_path": str(engine.relative_to(ROOT)) if engine.is_relative_to(ROOT) else str(engine),
                    "engine_sha256": result["engine_sha256"],
                    "onnx_sha256": result["onnx_sha256"],
                    "input_shapes": result["input_shapes"],
                    "engine_bytes": result["engine_bytes"],
                    "precision": result["precision"],
                    "candidate_contract_sha256": contract_hash(entries[role]),
                    "source_build": str(output / role / "build.json"),
                }
                print(role, result["status"], result["engine_bytes"], flush=True)
            except Exception as exc:
                failures.append(role)
                print(role, "FAILED:", exc, flush=True)
    if not failures:
        runtime.write_json(output / "runtime-candidate.json", candidate)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
