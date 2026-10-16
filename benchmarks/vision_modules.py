#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V1: unchanged native vision models, paired plain/instrumented calls."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.baseline_support import (  # noqa: E402
    CallTimes,
    Events,
    Resources,
    distribution,
    environment,
    instrument,
    jsonable,
    matching,
    run_worker,
    validate,
)
from deploy import runtime, services  # noqa: E402


def worker(module, out):
    import cv2
    from app.body_detector import BodyDetector
    from app.face_detector import FaceDetector
    from app.perception import HandDetector
    from deploy.resource_guard import hold_model_lease

    protocol, cfg, fixtures = validate()
    _lease = hold_model_lease(runtime.paths(cfg)["stack"] / "cache")
    resources = Resources(out, {module: [os.getpid()]}).start()
    events = Events(out / "events.jsonl")
    manifest = json.loads((ROOT / "models/perception.json").read_text())
    model = None
    report = {
        "module": module,
        "opencv_threads": cv2.getNumThreads(),
        "fixtures": [],
        "errors": [],
        "scope": "Isolated module process; hands loads the production palm+landmark pair. Native call times include provider's synchronous work, not pure kernel durations.",
    }
    try:
        start = time.perf_counter()
        if module == "body":
            model = BodyDetector(
                manifest["body"],
                cfg.perception.body_threshold,
                cfg.perception.body_iou,
                cfg.perception.max_people,
            )
            call = model.infer
        elif module == "face":
            model = FaceDetector()
            if not model.load():
                raise RuntimeError(model.last_error)
            call = model.detect
        else:
            model = HandDetector(
                manifest,
                cfg.perception.palm_threshold,
                cfg.perception.hand_threshold,
                cfg.perception.max_hands,
            )
            call = model.infer
        report["load_seconds"] = time.perf_counter() - start
        for entry in [x for x in fixtures["files"] if x["kind"] == "image"]:
            image = cv2.imread(
                str(Path(protocol["input_directory"]) / entry["filename"])
            )
            image.flags.writeable = False
            row = {
                "input": entry["filename"],
                "input_sha256": entry["sha256"],
                "pairs": [],
                "stages_ms": {},
            }
            report["fixtures"].append(row)
            resources.phase = entry["filename"] + ":warmup"
            # First invocation after fixture change plus the five planned warmups.
            for index in range(6):
                started = time.perf_counter()
                try:
                    output = jsonable(call(image))
                    events.write(
                        phase="first" if index == 0 else "warmup",
                        module=module,
                        input=entry["filename"],
                        index=index,
                        ms=(time.perf_counter() - started) * 1000,
                        output=output,
                    )
                except Exception as exc:
                    report["errors"].append(repr(exc))
                    events.write(
                        phase="warmup_error",
                        module=module,
                        input=entry["filename"],
                        index=index,
                        error=repr(exc),
                    )
            resources.phase = entry["filename"] + ":measured"
            for index in range(20):
                pair = {"index": index, "parity": False}
                outputs = {}
                for measured in [False, True] if index % 2 == 0 else [True, False]:
                    label = "instrumented" if measured else "plain"
                    times = CallTimes()
                    try:
                        if measured:
                            with instrument(module, model, times):
                                start = time.perf_counter()
                                output = call(image)
                                end = time.perf_counter()
                        else:
                            start = time.perf_counter()
                            output = call(image)
                            end = time.perf_counter()
                        pair[label + "_ms"] = (end - start) * 1000
                        output = jsonable(output)
                        outputs[label] = output
                        if measured and module == "face":
                            setup = next(
                                c
                                for c in times.calls
                                if c["label"] == "face.native_config"
                            )
                            native = next(
                                c
                                for c in times.calls
                                if c["label"] == "face.native_detect"
                            )
                            times.calls.extend(
                                [
                                    {
                                        "label": "face.python_pre_and_lock",
                                        "ms": (setup["start"] - start) * 1000,
                                    },
                                    {
                                        "label": "face.python_post",
                                        "ms": (end - native["end"]) * 1000,
                                    },
                                ]
                            )
                        for timing in times.calls:
                            row["stages_ms"].setdefault(timing["label"], []).append(
                                timing["ms"]
                            )
                        events.write(
                            phase="measured",
                            module=module,
                            input=entry["filename"],
                            pair=index,
                            variant=label,
                            ms=pair[label + "_ms"],
                            calls=times.calls,
                            output=output,
                        )
                    except Exception as exc:
                        pair[label + "_error"] = repr(exc)
                        report["errors"].append(repr(exc))
                        events.write(
                            phase="measured_error",
                            input=entry["filename"],
                            pair=index,
                            variant=label,
                            error=repr(exc),
                        )
                pair["parity"] = len(outputs) == 2 and matching(
                    outputs["plain"], outputs["instrumented"]
                )
                if not pair["parity"]:
                    report["errors"].append("Pair output mismatch or failed call")
                row["pairs"].append(pair)
            row["plain_ms"] = distribution(
                [p["plain_ms"] for p in row["pairs"] if "plain_ms" in p]
            )
            row["instrumented_ms"] = distribution(
                [p["instrumented_ms"] for p in row["pairs"] if "instrumented_ms" in p]
            )
            row["stage_summary_ms"] = {
                k: distribution(v) for k, v in row["stages_ms"].items()
            }
            # Signed paired differences retained; never subtract noisy medians to
            # manufacture corrected native timings or claim negative overhead.
            row["paired_overhead_ms"] = [
                p["instrumented_ms"] - p["plain_ms"]
                for p in row["pairs"]
                if "instrumented_ms" in p and "plain_ms" in p
            ]
            runtime.write_json(out / "result.json", report)
        resources.phase = "resident_after_measurement"
        time.sleep(2)
    except BaseException as exc:
        report["errors"].append(repr(exc))
        raise
    finally:
        try:
            if model:
                model.unload() if module == "face" else model.close()
        except Exception as exc:
            report["errors"].append("model cleanup: " + repr(exc))
        try:
            resources.close()
        except Exception as exc:
            report["errors"].append("monitor cleanup: " + repr(exc))
        events.close()
        report["monitor_errors"] = resources.errors
        report["status"] = (
            "MEASURED"
            if not report["errors"] and not resources.errors
            else "MEASURED_WITH_FAILURES"
        )
        runtime.write_json(out / "result.json", report)
    if report["status"] != "MEASURED":
        raise RuntimeError("Module measurement recorded failures; see result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=["body", "face", "hands"])
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker, args.output)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    protocol, cfg, _ = validate()
    before = services.state(services.ORDER, cfg)
    if (
        before["status"] != "READY"
        or before["services"]["app"]["health"]["phase"] != "idle"
    ):
        raise RuntimeError("Managed stack must be ready and idle")
    environment(out)
    runtime.write_json(out / "effective-config.json", cfg.effective())
    report = {
        "created_at": runtime.stamp(),
        "status": "RUNNING",
        "before": before,
        "protocol_sha256": runtime.sha256(
            ROOT / "benchmarks/manifests/reachy-b0-sw-v1.json"
        ),
        "modules": {},
    }
    error = None
    try:
        with services.operation(cfg):
            report["stopped"] = services.stop(list(services.ORDER), cfg)
        for module in ("body", "face", "hands"):
            print("V1 measuring: " + module, flush=True)
            folder = out / module
            folder.mkdir()
            with (folder / "process.log").open("w") as log:
                exit_code = run_worker(
                    [
                        sys.executable,
                        __file__,
                        "--worker",
                        module,
                        "--output",
                        str(folder),
                    ],
                    log,
                )
            report["modules"][module] = {
                "exit_code": exit_code,
                "result": str(folder / "result.json"),
            }
            runtime.write_json(out / "report.json", report)
            if exit_code:
                raise RuntimeError("Module measurement failed: " + module)
    except BaseException as exc:
        error = exc
        report["error"] = repr(exc)
    finally:
        try:
            with services.operation(cfg):
                report["restored"] = services.start(list(services.ORDER), cfg, 240)
        except BaseException as exc:
            error = error or exc
            report["restore_error"] = repr(exc)
        report["status"] = "FAIL" if error else "MEASURED"
        report["finished_at"] = runtime.stamp()
        runtime.write_json(out / "report.json", report)
    if error:
        raise error
    print(json.dumps({"status": report["status"], "report": str(out / "report.json")}))


if __name__ == "__main__":
    main()
