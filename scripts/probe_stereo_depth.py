#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deferred experiment: neural stereo depth, not part of native-depth B1.

This is a sensor-domain feasibility check. SDK depth is not ground truth, and a
short static scene cannot qualify person-distance gates or broad task accuracy.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_src"))
from benchmarks.stereo_depth_experiment import StereoDepth
from deploy.operations import exclusive_build
from deploy import runtime, services

# Fixed before examining sensor-domain outputs. No metric includes SDK holes.
CRITERIA = {
    "reference_min_m": 0.3,
    "reference_max_m": 5.0,
    "minimum_reference_pixels": 1000,
    "minimum_overlap_fraction": 0.5,
    "maximum_median_relative_error": 0.10,
    "maximum_p95_relative_error": 0.30,
}


def probe(capture_root, build_root, output):
    capture_path = capture_root / "report.json"
    capture = json.loads(capture_path.read_text())
    if capture["status"] != "CAPTURE_PASS_LIMITED_WINDOW":
        raise ValueError("A successful stereo capture probe is required")
    calibration = capture["calibration"]
    build = json.loads((build_root / "depth/build.json").read_text())
    parity = json.loads((build_root / "parity/depth/report.json").read_text())
    if (
        parity["status"] != "PARITY_PASS_LIMITED_FIXTURES"
        or parity["engine_sha256"] != build["engine_sha256"]
    ):
        raise ValueError("Depth engine numerical parity has not passed")
    report = {
        "created_at": runtime.stamp(),
        "status": "RUNNING",
        "scope": __doc__,
        "criteria": CRITERIA,
        "capture_report_sha256": runtime.sha256(capture_path),
        "engine_sha256": build["engine_sha256"],
        "onnx_sha256": build["onnx_sha256"],
        "script_sha256": runtime.sha256(Path(__file__)),
        "implementation_sha256": {
            name: runtime.sha256(ROOT / "01_src/benchmarks" / name)
            for name in ("stereo_depth_experiment.py", "stereo_geometry.py")
        },
        "cases": [],
        "timing_scope": "Single calls on three fixed sensor frames; not a B1 latency/FPS benchmark",
        "reference": "D435i SDK depth, same-frame timestamp and identical IR/depth calibration; not ground truth",
    }

    def save():
        runtime.write_json(output / "report.json", report)

    save()
    try:
        # Prove SDK depth and left IR use identical geometry before comparing.
        if (
            calibration["profiles"]["left_ir"]
            != calibration["profiles"]["sdk_depth_reference"]
        ):
            raise ValueError(
                "SDK depth requires separate registration before comparison"
            )
        extrinsics = calibration["extrinsics"]["left_ir_to_sdk_depth_reference"]
        if not np.allclose(
            extrinsics["rotation"], np.eye(3).ravel()
        ) or not np.allclose(extrinsics["translation_m"], 0):
            raise ValueError("SDK reference is in a different optical frame")
        with StereoDepth(build) as model:
            for item in capture["saved"]:
                path = capture_root / item["file"]
                if runtime.sha256(path) != item["sha256"]:
                    raise ValueError("Captured fixture checksum mismatch")
                provenance = item["frame"]
                left, ref = (
                    provenance["streams"][name]
                    for name in ("left_ir", "sdk_depth_reference")
                )
                if (
                    left["domain"] != ref["domain"]
                    or abs(left["timestamp_ms"] - ref["timestamp_ms"]) > 0.1
                ):
                    raise ValueError("SDK depth reference is not time-matched")
                with np.load(path, allow_pickle=False) as fixture:
                    values = model.infer(
                        fixture["left_ir"],
                        fixture["right_ir"],
                        calibration,
                        register_rgb=provenance["correspondence"]["rgb_time_matched"],
                    )
                    reference = (
                        fixture["sdk_depth_reference"].astype(np.float32)
                        * calibration["depth_units_m"]
                    )
                reference_valid = (reference >= CRITERIA["reference_min_m"]) & (
                    reference <= CRITERIA["reference_max_m"]
                )
                joint = reference_valid & values["valid"]
                reference_count, overlap_count = (
                    int(reference_valid.sum()),
                    int(joint.sum()),
                )
                error = np.abs(values["depth_m"][joint] - reference[joint])
                relative = error / reference[joint]
                metrics = {
                    "reference_pixels": reference_count,
                    "overlap_pixels": overlap_count,
                    "overlap_fraction": overlap_count / max(1, reference_count),
                    "median_absolute_error_m": float(np.median(error))
                    if overlap_count
                    else None,
                    "p95_absolute_error_m": float(np.quantile(error, 0.95))
                    if overlap_count
                    else None,
                    "median_relative_error": float(np.median(relative))
                    if overlap_count
                    else None,
                    "p95_relative_error": float(np.quantile(relative, 0.95))
                    if overlap_count
                    else None,
                }
                passed = bool(
                    reference_count >= CRITERIA["minimum_reference_pixels"]
                    and overlap_count
                    and metrics["overlap_fraction"]
                    >= CRITERIA["minimum_overlap_fraction"]
                    and metrics["median_relative_error"]
                    <= CRITERIA["maximum_median_relative_error"]
                    and metrics["p95_relative_error"]
                    <= CRITERIA["maximum_p95_relative_error"]
                )
                output_path = output / item["file"]
                np.savez(
                    output_path,
                    **{k: v for k, v in values.items() if isinstance(v, np.ndarray)},
                )
                report["cases"].append(
                    {
                        "input": item["file"],
                        "input_sha256": item["sha256"],
                        "frame": provenance,
                        "output_sha256": runtime.sha256(output_path),
                        "metrics": metrics,
                        "passed": passed,
                        "transform": values["transform"],
                        "stages_ms": values["stages_ms"],
                        "validity_scope": values["validity_scope"],
                        "rgb_valid_fraction": float(values["rgb_valid"].mean())
                        if "rgb_valid" in values
                        else None,
                    }
                )
                save()
                print(item["file"], json.dumps(metrics), "passed", passed, flush=True)
        report["status"] = (
            "SENSOR_REFERENCE_PASS_LIMITED_SCENE"
            if all(row["passed"] for row in report["cases"])
            else "SENSOR_REFERENCE_FAILED"
        )
    except Exception as exc:
        report.update(status="ERROR", error=repr(exc))
        raise
    finally:
        save()
    return report["status"] == "SENSOR_REFERENCE_PASS_LIMITED_SCENE"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = runtime.config()
    if services.profile_path(cfg).exists():
        cfg = runtime.config(services.profile_path(cfg))
    with exclusive_build(cfg, output):
        passed = probe(args.capture_root.resolve(), args.build_root.resolve(), output)
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
