#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check real public-image ONNX/TensorRT parity; this is not a B1 benchmark.

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_src"))
from app.tensorrt_runner import TensorRTModel
from app.face_trt import prepare_face, decode_yunet
from build_vision_b1 import exclusive_build
from deploy import runtime, services
from prepare_vision_b1 import MANIFEST, prepare

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
# Fixed before observing conversion results, and recorded with every run.
CRITERIA = {
    "pose_atol": 1e-3,
    "pose_rtol": 1e-3,
    "pose_max_argmax_bin_shift": 1,
    "depth_p99_abs_disparity_px": 0.1,
    "depth_max_abs_disparity_px": 0.5,
    "face_atol": 1e-4,
    "face_rtol": 1e-4,
    "face_decoded_max_error_px": 0.05,
    "face_score_max_error": 1e-4,
}


def normalized(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1)[None])


def inputs_for(role):
    if role == "face":
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
    elif role == "body_hand_pose":
        path = ROOT / "05_runtime/datasets/b0-sw-v1/public-original.png"
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        sx, sy = w * 1.25, h * 1.25
        if sx / sy > 288 / 384:
            sy = sx * 384 / 288
        else:
            sx = sy * 288 / 384
        matrix = np.array(
            [
                [288 / sx, 0, 144 - w / 2 * 288 / sx],
                [0, 384 / sy, 192 - h / 2 * 384 / sy],
            ],
            np.float32,
        )
        # Same zero-rotation center/scale convention as MMPose TopdownAffine.
        for name, photo in (
            ("public-person", image),
            ("public-person-mirrored", image[:, ::-1]),
        ):
            crop = cv2.warpAffine(photo, matrix, (288, 384), flags=cv2.INTER_LINEAR)
            yield (
                name,
                {"input": normalized(crop)},
                {
                    "source": str(path.relative_to(ROOT)),
                    "source_sha256": runtime.sha256(path),
                    "affine_source_to_input": matrix.tolist(),
                    "bbox": [0, 0, w, h],
                    "bbox_padding": 1.25,
                    "mirrored": name.endswith("mirrored"),
                    "preprocess_reference": "MMPose pinned RTMW3D config and zero-rotation TopdownAffine",
                },
            )
    elif role == "depth":
        folder = ROOT / "05_runtime/frameworks/b1/fast-foundationstereo/demo_data"
        paths = [folder / name for name in ("left.png", "right.png")]
        images = [cv2.imread(str(p)) for p in paths]
        assert images[0].shape == images[1].shape
        h, w = images[0].shape[:2]
        scale = min(736 / w, 480 / h)
        shape = (round(w * scale), round(h * scale))
        for label, factor in (
            ("upstream-stereo", 1.0),
            ("upstream-stereo-dimmed", 0.75),
        ):
            tensors = {}
            for name, image in zip(("left_image", "right_image"), images):
                photo = (
                    np.rint(image.astype(np.float32) * factor)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                photo = cv2.resize(photo, shape, interpolation=cv2.INTER_AREA)
                photo = cv2.copyMakeBorder(
                    photo, 0, 480 - shape[1], 0, 736 - shape[0], cv2.BORDER_REPLICATE
                )
                tensors[name] = normalized(photo)
            yield (
                label,
                tensors,
                {
                    "sources": {
                        str(p.relative_to(ROOT)): runtime.sha256(p) for p in paths
                    },
                    "source_revision": "476f4249561f7c79ca707326954f9255643412a6",
                    "resize_scale_xy": [shape[0] / w, shape[1] / h],
                    "unpad_hw": [shape[1], shape[0]],
                    "brightness_factor": factor,
                    "preprocess_reference": "Pinned scripts/make_single_onnx.py: normalization is external to ONNX",
                },
            )
    else:
        raise ValueError("No reviewed input fixture contract for " + role)


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
                if role == "body_hand_pose":
                    bins = np.abs(expected.argmax(-1) - observed.argmax(-1))
                    detail["argmax_equal_fraction"] = float((bins == 0).mean())
                    detail["max_argmax_bin_shift"] = int(bins.max())
                    detail["passed"] = bool(
                        np.allclose(
                            expected,
                            observed,
                            atol=CRITERIA["pose_atol"],
                            rtol=CRITERIA["pose_rtol"],
                        )
                        and bins.max() <= CRITERIA["pose_max_argmax_bin_shift"]
                    )
                elif role == "face":
                    detail["passed"] = bool(
                        np.allclose(
                            expected,
                            observed,
                            atol=CRITERIA["face_atol"],
                            rtol=CRITERIA["face_rtol"],
                        )
                    )
                else:
                    detail["passed"] = (
                        detail["p99_abs_error"]
                        <= CRITERIA["depth_p99_abs_disparity_px"]
                        and detail["max_abs_error"]
                        <= CRITERIA["depth_max_abs_disparity_px"]
                    )
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
    parser.add_argument("--role", choices=("body_hand_pose", "depth", "face"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = runtime.config()
    if services.profile_path(cfg).exists():
        cfg = runtime.config(services.profile_path(cfg))
    entries = json.loads(MANIFEST.read_text())["models"]
    failures = []
    with exclusive_build(cfg, output):
        for role in [args.role] if args.role else ("body_hand_pose", "face"):
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
