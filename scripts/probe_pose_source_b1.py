#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare official RTMW3D weights/preprocessing/codec with ONNX and TensorRT.

CPU references run in separate processes while project services are stopped.
Results establish limited conversion parity, not real-scene pose accuracy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_src"))
from deploy import runtime, services

CRITERIA = {
    "input_atol": 1e-6,
    "raw_atol": 1e-3,
    "raw_rtol": 1e-3,
    "max_argmax_bin_shift": 1,
    "decoded_input_xy_atol": 0.5001,
    "decoded_relative_z_atol": 2 * 2.1744869 / 576 + 1e-6,
}
OUTPUT_NAMES = ("output", "1554", "1556")


def fixtures(output):
    rows = []
    path = ROOT / "05_runtime/datasets/b0-sw-v1/public-original.png"
    image = cv2.imread(str(path))
    for name, photo in (("public-person", image), ("public-mirrored", image[:, ::-1])):
        rows.append((name, photo, [0, 0, image.shape[1], image.shape[0]], path))
    path = (
        ROOT
        / "05_runtime/validation/b1-segmentation-20261019/parity-opencv411/reference/bus.npz"
    )
    with np.load(path, allow_pickle=False) as values:
        image, boxes = values["source"], values["boxes"]
    for i in np.flatnonzero(boxes[:, 5] == 0)[:2]:
        rows.append(("bus-person-" + str(i), image, boxes[i, :4], path))
    metadata = []
    for name, image, box, source in rows:
        np.savez(
            output / (name + "-source.npz"), image=image, box=np.array(box, np.float32)
        )
        metadata.append(
            {
                "name": name,
                "source": str(source.relative_to(ROOT)),
                "source_sha256": runtime.sha256(source),
                "box": list(map(float, box)),
            }
        )
    runtime.write_json(output / "fixtures.json", metadata)
    return metadata


def worker(output, kind):
    if kind == "native":
        import torch
        from deploy.vision.pose_reference import load_reference

        model, prepare, codec, provenance = load_reference()
        from importlib.metadata import version

        provenance["runtime_versions"] = {
            name: version(name)
            for name in (
                "torch",
                "mmengine",
                "mmcv-lite",
                "numpy",
                "opencv-python-headless",
            )
        }
        runtime.write_json(output / "native-provenance.json", provenance)
    else:
        import onnxruntime as ort

        entry = json.loads((ROOT / "03_models/candidates/vision-b1.json").read_text())[
            "models"
        ]["body_hand_pose"]
        if runtime.sha256(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("ONNX changed")
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
        session = ort.InferenceSession(
            str(ROOT / entry["path"]),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    for row in json.loads((output / "fixtures.json").read_text()):
        name = row["name"]
        if kind == "native":
            with np.load(output / (name + "-source.npz"), allow_pickle=False) as source:
                tensor, center, scale = prepare(source["image"], source["box"])
            with torch.inference_mode():
                raw = [value.numpy() for value in model(tensor)]
            points, bins, scores = codec.decode(*raw)
            points[..., :2] = (
                points[..., :2] / np.array([288, 384]) * scale + center - 0.5 * scale
            )
            np.savez(
                output / (name + "-native.npz"),
                input=tensor.numpy(),
                points=points,
                scores=scores,
                center=center,
                scale=scale,
                **dict(zip(OUTPUT_NAMES, raw)),
            )
        else:
            with np.load(output / (name + "-native.npz"), allow_pickle=False) as source:
                raw = session.run(list(OUTPUT_NAMES), {"input": source["input"]})
            np.savez(output / (name + "-onnx.npz"), **dict(zip(OUTPUT_NAMES, raw)))
        print(kind, name, "saved", flush=True)
    if kind == "native":
        # A small deterministic regression contract independent of public photos.
        y, x = np.indices((1080, 810))
        image = np.stack(
            [(5 * x + 7 * y) % 256, (11 * x + 3 * y) % 256, (19 * x + 2 * y) % 256],
            axis=-1,
        ).astype(np.uint8)
        box = json.loads((output / "fixtures.json").read_text())[-1]["box"]
        tensor, center, scale = prepare(image, box)
        runtime.write_json(
            output / "preprocess-regression.json",
            {
                "reference": "Original MMPose GetBBoxCenterScale, TopdownAffine, PoseDataPreprocessor",
                "revision": provenance["revision"],
                "opencv": provenance["runtime_versions"]["opencv-python-headless"],
                "image_hw": [1080, 810],
                "bgr_formula": [[5, 7], [11, 3], [19, 2]],
                "box": box,
                "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "normalized_tensor_sha256": hashlib.sha256(
                    tensor.numpy().tobytes()
                ).hexdigest(),
                "center": center.tolist(),
                "scale": scale.tolist(),
            },
        )


def compare(output, rows):
    from app.wholebody_pose import WholeBodyPose, prepare_pose

    entries = json.loads((ROOT / "03_models/candidates/vision-b1.json").read_text())[
        "models"
    ]
    spec = json.loads((ROOT / "03_models/vision-b1-runtime.json").read_text())[
        "models"
    ]["body_hand_pose"]
    spec["engine_path"] = str(ROOT / spec["engine_path"])
    model = WholeBodyPose(spec, entries["body_hand_pose"])
    results = []
    try:
        for row in rows:
            name = row["name"]
            with np.load(output / (name + "-source.npz"), allow_pickle=False) as f:
                image, box = f["image"], f["box"]
            with np.load(output / (name + "-native.npz"), allow_pickle=False) as f:
                native = dict(f)
            with np.load(output / (name + "-onnx.npz"), allow_pickle=False) as f:
                onnx = dict(f)
            inputs, matrix = prepare_pose(image, box)
            raw = model.model.infer({"input": native["input"]})
            application = model.infer(image, box)
            np.savez(output / (name + "-trt.npz"), **raw)
            runtime.write_json(output / (name + "-application.json"), application)
            comparisons = {}
            for backend, tensors in (("onnx", onnx), ("tensorrt", raw)):
                for key in OUTPUT_NAMES:
                    error = np.abs(tensors[key] - native[key])
                    shift = np.abs(tensors[key].argmax(-1) - native[key].argmax(-1))
                    comparisons[backend + ":" + key] = {
                        "max_abs_error": float(error.max()),
                        "max_argmax_bin_shift": int(shift.max()),
                        "passed": bool(
                            np.allclose(
                                tensors[key],
                                native[key],
                                atol=CRITERIA["raw_atol"],
                                rtol=CRITERIA["raw_rtol"],
                            )
                            and shift.max() <= CRITERIA["max_argmax_bin_shift"]
                        ),
                    }
            point_error = np.abs(
                np.array(application["keypoints"]) - native["points"][0]
            )
            xy_input_error = point_error[:, :2] * np.array([288, 384]) / native["scale"]
            input_error = float(np.abs(inputs["input"] - native["input"]).max())
            result = {
                "name": name,
                "input_max_error": input_error,
                "raw": comparisons,
                "decoded_original_xy_max_error_px": float(point_error[:, :2].max()),
                "decoded_input_xy_max_error_px": float(xy_input_error.max()),
                "decoded_relative_z_max_error": float(point_error[:, 2].max()),
            }
            result["passed"] = bool(
                input_error <= CRITERIA["input_atol"]
                and all(c["passed"] for c in comparisons.values())
                and xy_input_error.max() <= CRITERIA["decoded_input_xy_atol"]
                and point_error[:, 2].max() <= CRITERIA["decoded_relative_z_atol"]
            )
            results.append(result)
            runtime.write_json(output / "comparison.json", results)
            print(name, "PASS" if result["passed"] else "FAIL", flush=True)
    finally:
        model.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("native", "onnx"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        worker(output, args.worker)
        return
    output.mkdir(parents=True, exist_ok=False)
    from build_vision_b1 import exclusive_build

    cfg = runtime.config()
    cfg = runtime.config(services.profile_path(cfg))
    report = {
        "created_at": runtime.stamp(),
        "status": "RUNNING",
        "scope": __doc__,
        "criteria": CRITERIA,
    }
    runtime.write_json(output / "report.json", report)
    try:
        with exclusive_build(cfg, output):
            rows = fixtures(output)
            for kind, python in (
                ("native", ROOT / "05_runtime/builds/vision-export/bin/python"),
                ("onnx", Path(sys.executable)),
            ):
                argv = [
                    str(python),
                    str(Path(__file__).resolve()),
                    "--worker",
                    kind,
                    "--output",
                    str(output),
                ]
                with (output / (kind + ".log")).open("w") as log:
                    subprocess.run(
                        argv,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=600,
                    )
            report["cases"] = compare(output, rows)
            if not all(r["passed"] for r in report["cases"]):
                raise RuntimeError(
                    "Original pose reference parity failed; inspect comparison.json"
                )
            report["status"] = (
                "ORIGINAL_CHECKPOINT_ONNX_TRT_PARITY_PASS_LIMITED_FIXTURES"
            )
    except BaseException as exc:
        report.update(status="FAILED", error=repr(exc))
        raise
    finally:
        runtime.write_json(output / "report.json", report)
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
