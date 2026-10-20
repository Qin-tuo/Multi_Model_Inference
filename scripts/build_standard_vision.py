#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build three B2 engines, verify native fixtures, and prepare a separate manifest."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "01_src"), str(ROOT / "scripts")]
from app.appearance import AppearanceFeatures, prepare_appearance
from app.face_tasks import ArcFaceIdentity, aligned_bgr, sface_transform
from app.vision_models import contract_hash
from app.pose_geometry import prepare_pose
from app.wholebody_pose2d import WholeBodyPose2D
from build_vision_b1 import build
from deploy.operations import exclusive_build
from deploy import runtime, services
from deploy.validation_inputs import input_path

MANIFEST = ROOT / "03_models/candidates/vision-b2.json"
ROLES = ("body_hand_pose", "face_identity", "person_reid")


def checked_candidate(role, entry, check_only=True):
    path = (ROOT / entry["path"]).resolve()
    if (
        role not in ROLES
        or not path.is_relative_to(ROOT / "05_runtime/models/b2")
        or runtime.sha256(path) != entry["sha256"]
        or path.stat().st_size != entry["bytes"]
    ):
        raise ValueError("B2 candidate is not the pinned exported graph")


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
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--role", choices=ROLES, nargs="+", default=list(ROLES))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = runtime.config()
    if services.profile_path(cfg).exists():
        cfg = runtime.config(services.profile_path(cfg))
    source = json.loads(MANIFEST.read_text())
    previous = ROOT / "03_models/vision-runtime.json"
    if not previous.exists():
        previous = ROOT / "03_models/vision-b1-runtime.json"
    selected = deepcopy(json.loads(previous.read_text()))
    selected.update(
        source_manifest=str(MANIFEST.relative_to(ROOT)), model_set="B2-standard-2d"
    )
    records = {}
    with exclusive_build(cfg, output):
        for role in args.role:
            entry = source["models"][role]
            built = build(
                role, entry, output, args.timeout, prepare_candidate=checked_candidate
            )
            checked = parity(role, entry, built, output / role)
            destination = (
                ROOT
                / f"05_runtime/models/b2/runtime/{role}-{entry['sha256'][:12]}.fp32.engine"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copyfile(ROOT / built["engine_path"], destination)
            if runtime.sha256(destination) != built["engine_sha256"]:
                raise ValueError("Copied engine changed")
            selected["models"][role] = {
                "role": role,
                "engine_path": str(destination.relative_to(ROOT)),
                "engine_sha256": built["engine_sha256"],
                "onnx_sha256": entry["sha256"],
                "input_shapes": built["input_shapes"],
                "engine_bytes": destination.stat().st_size,
                "precision": "FP32; TF32 disabled",
                "candidate_contract_sha256": contract_hash(entry),
                "source_build": str((output / role / "build.json").relative_to(ROOT)),
            }
            records[role] = checked
            print(role, checked["status"], flush=True)
    selected["status"] = "REPLACEMENTS_REFERENCE_CHECKED_INTEGRATION_PENDING"
    runtime.write_json(ROOT / "03_models/vision-runtime.json", selected)
    runtime.write_json(
        output / "summary.json",
        {
            "status": selected["status"],
            "changed_roles": list(ROLES),
            "rebuilt_roles": args.role,
            "retained_roles": [r for r in selected["models"] if r not in ROLES],
            "models": records,
        },
    )


if __name__ == "__main__":
    main()
