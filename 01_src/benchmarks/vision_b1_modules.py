#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""B1 isolated-module baseline: paired wall-clock and CUDA stream measurements."""

import argparse
import csv
import json
import os
from pathlib import Path
import resource
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "01_src"), str(ROOT / "scripts")]
from app.vision_models import ROLES, VisionModels
from benchmarks.baseline_support import (
    Events,
    Resources,
    distribution,
    environment,
    run_worker,
)
from benchmarks.vision_b1_support import (
    PROTOCOL,
    arguments,
    outputs_match,
    plain_call,
    validate,
)
from benchmarks.vision_b1_timing import TensorRTTrace
from build_vision_b1 import exclusive_build
from deploy import runtime, services


def summarize(output):
    run = json.loads((output / "report.json").read_text())
    if run["status"] != "MEASURED_MODULES_ONLY_PIPELINE_PENDING":
        raise ValueError("Cannot summarize an incomplete measurement")
    summary = {
        "created_at": runtime.stamp(),
        "status": "MODULE_BASELINE_MEASURED_PIPELINE_PENDING",
        "raw_directory": str(output.relative_to(ROOT)),
        "protocol_sha256": run["protocol_sha256"],
        "scope": "One engine per process, fixed 640x480 public inputs, five warmups and twenty alternating plain/instrumented pairs per input; no pipeline FPS estimate",
        "units": "milliseconds; percentiles use nearest-rank",
        "stage_rules": "Host pre/native/post spans partition an instrumented adapter call. CUDA stream spans overlap host native time and include host-submit gaps; do not sum separate percentiles or add CUDA to host durations.",
        "resource_rules": "1s host/process/tegrastats samples include harness validation. Short GPU calls may be missed by sampling; RSS high-water includes model loading, not just steady inference. CPU/GPU share board RAM.",
        "modules": {},
    }
    for role in run["modules"]:
        folder = output / role
        report = json.loads((folder / "report.json").read_text())
        monitor = json.loads((folder / "monitor.json").read_text())
        if report["status"] != "MEASURED" or monitor["errors"]:
            raise ValueError("Incomplete module or resource samples: " + role)
        samples = [
            json.loads(line)
            for line in (folder / "events.jsonl").read_text().splitlines()
        ]
        pairs = [r for r in samples if r["phase"] == "paired-measurement"]
        if not pairs or not all(r["parity"] for r in pairs):
            raise ValueError("Missing paired output parity: " + role)
        with (folder / "resources.csv").open() as stream:
            resources = list(csv.DictReader(stream))
        deltas = [r["instrumented"]["adapter_ms"] - r["plain_ms"] for r in pairs]
        summary["modules"][role] = {
            "paired_count": len(pairs),
            "plain_adapter_ms": distribution([r["plain_ms"] for r in pairs]),
            "instrumented": {
                key: distribution([r["instrumented"][key] for r in pairs])
                for key in pairs[0]["instrumented"]
            },
            "observer_delta_ms": {
                "min": min(deltas),
                "median": statistics.median(deltas),
                "max": max(deltas),
                "meaning": "signed instrumented minus paired plain; contains measurement noise",
            },
            "fixtures": report["fixtures"],
            "load": report["loaded"],
            "rss_high_water_bytes": report["rss_high_water_bytes"],
            "host_min_available_bytes_sampled": min(
                int(r["mem_available"]) for r in resources
            )
            if resources
            else None,
            "sampled_rss_max_bytes": max(int(r["module_rss"]) for r in resources)
            if resources
            else None,
            "resource_sample_count": len(resources),
            "raw_hashes": {
                name: runtime.sha256(folder / name)
                for name in (
                    "events.jsonl",
                    "resources.csv",
                    "tegrastats.log",
                    "report.json",
                )
            },
        }
    runtime.write_json(output / "summary.json", summary)
    return summary


def save_prediction(path, prediction):
    """Preserve a structured reference without pickle or expanding mask arrays."""
    import numpy as np

    arrays = {}

    def visit(value):
        if isinstance(value, np.ndarray):
            key = "array_" + str(len(arrays))
            arrays[key] = value
            return {"array": key}
        if isinstance(value, dict):
            return {k: visit(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [visit(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    tree = visit(prediction)
    np.savez(path, metadata=np.asarray(json.dumps(tree)), **arrays)


def worker(role, output):
    protocol, cfg = validate()
    owner = VisionModels()
    resources = Resources(output, {"module": [os.getpid()]}).start()
    events = Events(output / "events.jsonl")
    report = {
        "role": role,
        "fixtures": [],
        "errors": [],
        "scope": "Isolated selected TensorRT engine; plain adapter calls and paired stream instrumentation",
    }
    error = None
    try:
        owner.load([role])
        model = owner.models[role]
        report["loaded"] = owner.snapshot()[role]
        with TensorRTTrace(model) as observer:
            for case in protocol["module_cases"][role]:
                name = case["input"]
                resources.phase = name + ":load-input"
                args = arguments(protocol, case)
                resources.phase = name + ":first-call"
                first, first_ms = plain_call(model, args)
                save_prediction(output / (name + "-prediction.npz"), first)
                del first
                events.write(phase="first-call", input=name, adapter_ms=first_ms)
                resources.phase = name + ":warmup"
                for i in range(protocol["measurement"]["warmup_per_case"]):
                    _, elapsed = plain_call(model, args)
                    events.write(
                        phase="warmup", input=name, index=i, adapter_ms=elapsed
                    )
                resources.phase = name + ":paired-measurement"
                rows = []
                for i in range(protocol["measurement"]["paired_samples_per_case"]):
                    row, outputs = {"index": i, "input": name}, {}
                    for traced in (False, True) if i % 2 == 0 else (True, False):
                        if traced:
                            outputs["instrumented"], row["instrumented"] = (
                                observer.call(*args)
                            )
                        else:
                            outputs["plain"], row["plain_ms"] = plain_call(model, args)
                    row["parity"] = outputs_match(
                        outputs["plain"], outputs["instrumented"]
                    )
                    events.write(phase="paired-measurement", **row)
                    if not row["parity"]:
                        raise RuntimeError("Observer altered model output: " + name)
                    rows.append(row)
                summary = {
                    "input": name,
                    "first_call_ms": first_ms,
                    "paired_count": len(rows),
                    "plain_adapter_ms": distribution([r["plain_ms"] for r in rows]),
                    "instrumented": {
                        k: distribution([r["instrumented"][k] for r in rows])
                        for k in rows[0]["instrumented"]
                    },
                    "paired_observer_delta_ms": [
                        r["instrumented"]["adapter_ms"] - r["plain_ms"] for r in rows
                    ],
                }
                report["fixtures"].append(summary)
                runtime.write_json(output / "report.json", report)
                print(role, name, "measured", flush=True)
    except BaseException as exc:
        error = exc
        report["errors"].append(repr(exc))
    finally:
        resources.phase = "release"
        for close in (owner.close, resources.close, events.close):
            try:
                close()
            except Exception as exc:
                error = error or exc
                report["errors"].append(repr(exc))
        report["rss_high_water_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        report["status"] = "FAILED" if error else "MEASURED"
        runtime.write_json(output / "report.json", report)
    if error:
        raise error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=ROLES, nargs="+")
    parser.add_argument("--worker", choices=ROLES, help=argparse.SUPPRESS)
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Summarize an existing successful run without inference",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if args.summarize:
        summarize(output)
        print(output / "summary.json")
        return
    if args.worker:
        worker(args.worker, output)
        return
    output.mkdir(parents=True, exist_ok=False)
    protocol, cfg = validate()
    environment(output)
    runtime.write_json(output / "protocol-used.json", protocol)
    before = runtime.config()
    before = runtime.config(services.profile_path(before))
    report = {
        "created_at": runtime.stamp(),
        "status": "RUNNING",
        "protocol_sha256": runtime.sha256(PROTOCOL),
        "modules": {},
    }
    try:
        with exclusive_build(before, output):
            for role in args.role or ROLES:
                folder = output / role
                folder.mkdir()
                with (folder / "worker.log").open("w") as log:
                    code = run_worker(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            role,
                            "--output",
                            str(folder),
                        ],
                        log,
                        seconds=300,
                    )
                result = (
                    json.loads((folder / "report.json").read_text())
                    if (folder / "report.json").exists()
                    else {}
                )
                report["modules"][role] = {
                    "returncode": code,
                    "status": result.get("status"),
                    "report": str((folder / "report.json").relative_to(ROOT)),
                }
                runtime.write_json(output / "report.json", report)
                if code or result.get("status") != "MEASURED":
                    raise RuntimeError("B1 module measurement failed: " + role)
                print(role, "MEASURED", flush=True)
        report["status"] = "MEASURED_MODULES_ONLY_PIPELINE_PENDING"
    except BaseException as exc:
        report.update(status="FAILED", error=repr(exc))
        raise
    finally:
        runtime.write_json(output / "report.json", report)
    summarize(output)


if __name__ == "__main__":
    main()
