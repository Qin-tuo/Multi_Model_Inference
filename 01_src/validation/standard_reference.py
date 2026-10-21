# SPDX-License-Identifier: Apache-2.0
"""Check current 2D pose, ArcFace and OSNet engines against pinned native fixtures."""

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01_src"))
from app.appearance import AppearanceFeatures, prepare_appearance
from app.face_tasks import ArcFaceIdentity, aligned_bgr, sface_transform
from app.pose_geometry import prepare_pose
from app.wholebody_pose2d import WholeBodyPose2D
from deploy import runtime, services
from deploy.operations import exclusive_build
from deploy.validation_inputs import input_path

ROLES = ("body_hand_pose", "face_identity", "person_reid")


def parity(role, entry, built, output):
    spec = {**built, "engine_path": str(ROOT / built["engine_path"])}
    if role == "body_hand_pose":
        model = WholeBodyPose2D(spec, entry)
    elif role == "face_identity":
        model = ArcFaceIdentity(spec)
    else:
        model = AppearanceFeatures(spec, role)
    exported = json.loads(input_path(ROOT / entry["export_report"], ROOT).read_text())
    report = {
        "role": role,
        "status": "RUNNING",
        "cases": [],
        "criteria": {
            "raw_atol": 1e-3,
            "raw_rtol": 1e-3,
            "min_cosine": 0.999,
            "max_pose_input_pixel_error": 0.51,
            "max_crop_mae": 0.05,
        },
        "scope": "Fixed-input native/ONNX/TensorRT and adapter parity; no field-accuracy claim",
    }
    path = output / "parity.json"
    try:
        for case in exported["cases"]:
            fixture_path = input_path(ROOT / case["fixture"], ROOT)
            if runtime.sha256(fixture_path) != case["sha256"]:
                raise ValueError("Native fixture changed")
            with np.load(fixture_path, allow_pickle=False) as f:
                fixture = dict(f)
            input_name = next(iter(entry["input_shapes"]))
            raw = model.model.infer({input_name: fixture["input"]})
            row = {"input": case["input"], "raw": {}}
            for index, name in enumerate(entry["output_names"]):
                expected = fixture[f"output_{index}"]
                actual = raw[name]
                passed = bool(np.allclose(actual, expected, atol=1e-3, rtol=1e-3))
                row["raw"][name] = {
                    "max_abs_error": float(np.abs(actual - expected).max()),
                    "passed": passed,
                }
            image, arguments = fixture["source"], case["arguments"]
            tick = time.perf_counter()
            observed = model.infer(image, *arguments)
            row["functional_call_ms"] = (time.perf_counter() - tick) * 1000
            if role == "body_hand_pose":
                prepared, _ = prepare_pose(
                    image, arguments[0], model.geometry, model.input_size
                )
                row["preprocess_max_error"] = float(
                    np.abs(prepared["input"] - fixture["input"]).max()
                )
                error = np.abs(
                    np.asarray(observed["keypoints"]) - fixture["decoded_xy"][0]
                )
                error = error * np.array(model.input_size) / fixture["scale"]
                row["max_pose_input_pixel_error"] = float(error.max())
                row["dimensions"] = observed["coordinate_dimensions"]
                row["adapter_pass"] = bool(
                    row["preprocess_max_error"] <= 1e-6
                    and error.max() <= 0.51
                    and observed["coordinate_dimensions"] == 2
                    and all(
                        np.asarray(h["landmarks"]).shape == (21, 2)
                        for h in observed["hands"]
                    )
                )
            else:
                expected = fixture["output_0"].reshape(-1)
                expected = expected / np.linalg.norm(expected)
                row["native_adapter_cosine"] = float(expected @ observed["embedding"])
                if role == "face_identity":
                    crop = aligned_bgr(image, sface_transform(arguments[0]))
                    row["crop_mae"] = float(
                        np.abs(crop.astype(float) - fixture["aligned_bgr"]).mean()
                    )
                    row["adapter_pass"] = row["crop_mae"] <= 0.05
                    row["embedding_dimensions"] = len(observed["embedding"])
                else:
                    prepared, _ = prepare_appearance(image, arguments[0], role)
                    row["preprocess_max_error"] = float(
                        np.abs(prepared["images"] - fixture["input"]).max()
                    )
                    row["adapter_pass"] = row["preprocess_max_error"] <= 1e-6
                row["adapter_pass"] = bool(
                    row["adapter_pass"] and row["native_adapter_cosine"] >= 0.999
                )
            report["cases"].append(row)
            runtime.write_json(path, report)
            if not row["adapter_pass"] or not all(
                v["passed"] for v in row["raw"].values()
            ):
                raise ValueError(
                    "Native/TensorRT/adapter comparison failed for " + role
                )
        report["status"] = "NATIVE_TRT_ADAPTER_PARITY_PASS_LIMITED_FIXTURES"
    except Exception as exc:
        report.update(status="FAILED", error=str(exc))
        raise
    finally:
        model.close()
        runtime.write_json(path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=ROLES, nargs="+", default=list(ROLES))
    parser.add_argument("--manifest", type=Path, default=ROOT / "03_models/vision-runtime.json")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected = json.loads(args.manifest.read_text())
    sources = json.loads((ROOT / selected["source_manifest"]).read_text())["models"]
    cfg = runtime.config()
    if services.profile_path(cfg).exists():
        cfg = runtime.config(services.profile_path(cfg))
    results = {}
    with exclusive_build(cfg, output):
        for role in args.role:
            spec, entry = selected["models"][role], sources[role]
            if spec["onnx_sha256"] != entry["sha256"]:
                raise ValueError("Engine/model source mismatch: " + role)
            folder = output / role
            folder.mkdir()
            results[role] = parity(role, entry, spec, folder)
    runtime.write_json(output / "report.json", {"status": "NATIVE_PARITY_PASS_LIMITED_FIXTURES", "models": results})


if __name__ == "__main__":
    main()
