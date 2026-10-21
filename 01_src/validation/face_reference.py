#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check real public-image ONNX/TensorRT parity; timings are functional diagnostics.

The reference is the exact pinned ONNX. Original PyTorch/export equivalence,
task accuracy, camera calibration and end-to-end acceptance are separate gates.
"""

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01_src"))
from app.tensorrt_runner import TensorRTModel
from app.face_trt import prepare_face, decode_yunet
from deploy.operations import exclusive_build
from deploy import runtime, services
from deploy.vision.artifacts import MANIFEST, prepare

# Fixed before observing conversion results, and recorded with every run.
CRITERIA = {
    "face_atol": 1e-4,
    "face_rtol": 1e-4,
    "face_decoded_max_error_px": 0.05,
    "face_score_max_error": 1e-4,
}


def inputs_for(role):
    if role != "face":
        raise ValueError("Only the current YuNet face graph is supported")
    path = ROOT / "05_runtime/datasets/b0-sw-v1/public-original.png"
    image = cv2.imread(str(path))
    for label, photo in (
        ("public-face", image),
        ("public-face-mirrored", image[:, ::-1]),
        ("black-negative", np.zeros_like(image)),
    ):
        tensors, transform = prepare_face(photo)
        yield (
            label,
            tensors,
            {
                "source": str(path.relative_to(ROOT)),
                "source_sha256": runtime.sha256(path),
                "mirrored": label.endswith("mirrored"),
                "synthetic_black": label == "black-negative",
                "transform": transform,
                "preprocess_reference": "OpenCV FaceDetectorYN BGR blobFromImage without normalization",
            },
        )


def compare(role, entry, build_root, output):
    import onnxruntime as ort

    prepare(role, entry, check_only=True)
    build = json.loads((build_root / role / "build.json").read_text())
    if (
        build["status"] != "BUILT_NOT_ACCEPTED"
        or build["onnx_sha256"] != entry["sha256"]
    ):
        raise ValueError("No matching successful build for " + role)
    folder = output / role
    folder.mkdir(parents=True, exist_ok=False)
    report = {
        "role": role,
        "status": "RUNNING",
        "criteria": CRITERIA,
        "cases": [],
        "scope": "Conversion feasibility only; one execution per input, not a performance or task-accuracy claim",
        "onnx_sha256": entry["sha256"],
        "engine_sha256": build["engine_sha256"],
        "reference": "ONNX Runtime CPUExecutionProvider",
        "provider": "TensorRT",
        "normalization": entry["preprocessing"],
        "source_sha256": runtime.sha256(Path(__file__)),
    }
    runtime.write_json(folder / "report.json", report)
    cases = list(inputs_for(role))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(ROOT / entry["path"]), options, providers=["CPUExecutionProvider"]
    )
    names = [item.name for item in session.get_outputs()]
    references = []
    for label, tensors, provenance in cases:
        np.savez(folder / (label + "-inputs.npz"), **tensors)
        start = time.monotonic()
        values = dict(zip(names, session.run(None, tensors)))
        seconds = time.monotonic() - start
        if any(not np.isfinite(value).all() for value in values.values()):
            raise RuntimeError("Non-finite reference output")
        np.savez(folder / (label + "-reference.npz"), **values)
        references.append(values)
        report["cases"].append(
            {"name": label, "provenance": provenance, "reference_seconds": seconds}
        )
        runtime.write_json(folder / "report.json", report)
        print(role, label, "reference complete", flush=True)
    del session
    gc.collect()
    with TensorRTModel(
        build["engine_path"], build["engine_sha256"], build["input_shapes"]
    ) as model:
        report["provider"] = model.provider
        report["io_buffer_bytes_each_host_device"] = model.io_bytes
        for row, (_, tensors, _), reference in zip(report["cases"], cases, references):
            start = time.monotonic()
            actual = model.infer(tensors)
            row["synchronous_trt_call_seconds"] = time.monotonic() - start
            np.savez(folder / (row["name"] + "-tensorrt.npz"), **actual)
            if set(actual) != set(reference):
                raise ValueError("Output tensor names differ")
            row["outputs"] = {}
            for name, expected in reference.items():
                observed = actual[name]
                assert (
                    expected.shape == observed.shape
                    and expected.dtype == observed.dtype
                )
                error = np.abs(expected.astype(np.float64) - observed)
                detail = {
                    "shape": list(expected.shape),
                    "reference_min": float(expected.min()),
                    "reference_max": float(expected.max()),
                    "mean_abs_error": float(error.mean()),
                    "p99_abs_error": float(np.quantile(error, 0.99)),
                    "max_abs_error": float(error.max()),
                }
                detail["passed"] = bool(np.allclose(
                    expected, observed, atol=CRITERIA["face_atol"], rtol=CRITERIA["face_rtol"]
                ))
                row["outputs"][name] = detail
            row["passed"] = all(value["passed"] for value in row["outputs"].values())
            if role == "face":
                # Independent official implementation validates the translated
                # decoder, not just equality of raw neural-network tensors.
                detector = cv2.FaceDetectorYN.create(
                    str(ROOT / entry["path"]), "", (640, 640), 0.5, 0.3, 5000
                )
                photo = np.ascontiguousarray(
                    tensors["input"][0].transpose(1, 2, 0), dtype=np.uint8
                )
                _, expected_faces = detector.detect(photo)
                expected_faces = (
                    expected_faces
                    if expected_faces is not None
                    else np.empty((0, 15), np.float32)
                )
                faces = decode_yunet(actual)
                same_count = len(faces) == len(expected_faces)
                coordinate_error = (
                    float(np.abs(faces[:, :14] - expected_faces[:, :14]).max())
                    if same_count and len(faces)
                    else 0.0
                )
                score_error = (
                    float(np.abs(faces[:, 14] - expected_faces[:, 14]).max())
                    if same_count and len(faces)
                    else 0.0
                )
                presence = (
                    len(faces) == 0
                    if row["name"] == "black-negative"
                    else len(faces) >= 1
                )
                decoded_pass = (
                    same_count
                    and presence
                    and coordinate_error <= CRITERIA["face_decoded_max_error_px"]
                    and score_error <= CRITERIA["face_score_max_error"]
                )
                row["decoded"] = {
                    "reference": "OpenCV " + cv2.__version__ + " FaceDetectorYN",
                    "reference_faces": len(expected_faces),
                    "actual_faces": len(faces),
                    "max_coordinate_error_px": coordinate_error if same_count else None,
                    "max_score_error": score_error if same_count else None,
                    "passed": bool(decoded_pass),
                }
                np.savez(
                    folder / (row["name"] + "-decoded.npz"),
                    expected=expected_faces,
                    tensorrt=faces,
                )
                row["passed"] = row["passed"] and decoded_pass
            runtime.write_json(folder / "report.json", report)
            print(role, row["name"], "parity", row["passed"], flush=True)
    report["status"] = (
        "PARITY_PASS_LIMITED_FIXTURES"
        if all(row["passed"] for row in report["cases"])
        else "PARITY_FAILED"
    )
    runtime.write_json(folder / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("face",), default="face")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = runtime.config()
    if services.profile_path(cfg).exists():
        cfg = runtime.config(services.profile_path(cfg))
    entries = json.loads(MANIFEST.read_text())["models"]
    failures = []
    with exclusive_build(cfg, output):
        for role in [args.role]:
            try:
                result = compare(role, entries[role], args.build_root.resolve(), output)
                if result["status"] != "PARITY_PASS_LIMITED_FIXTURES":
                    failures.append(role)
            except Exception as exc:
                runtime.write_json(
                    output / role / "error.json",
                    {"type": type(exc).__name__, "error": str(exc)},
                )
                failures.append(role)
                print(role, "FAILED", repr(exc), flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
