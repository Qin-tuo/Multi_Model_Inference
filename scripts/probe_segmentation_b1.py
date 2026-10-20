#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate TensorRT segmentation against original Ultralytics and ONNX outputs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_src"))
CRITERIA = {
    "raw_atol": 1e-3,
    "raw_rtol": 1e-3,
    "min_box_iou": 0.999,
    "min_mask_iou": 0.99,
    "max_score_error": 1e-4,
}


def reference(output):
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "05_runtime/builds/ultralytics-config")
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from deploy import runtime
    from export_segmentation_b1 import WEIGHTS, WEIGHT_SHA, VERSIONS
    import importlib.metadata

    if {
        name: importlib.metadata.version(name) for name in VERSIONS
    } != VERSIONS or runtime.sha256(WEIGHTS) != WEIGHT_SHA:
        raise ValueError(
            "Reference model/tool versions differ from the export contract"
        )
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = YOLO(str(WEIGHTS))
    public = ROOT / "05_runtime/datasets/b0-sw-v1/public-original.png"
    bus = Path(ultralytics.__file__).parent / "assets/bus.jpg"
    rows = []
    for name, path, resize, blank in (
        ("public-person", public, None, False),
        ("bus", bus, None, False),
        ("bus-odd-shape", bus, (799, 601), False),
        ("blank-negative", public, (640, 480), True),
    ):
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        if resize:
            image = cv2.resize(image, resize)
        if blank:
            image = np.zeros_like(image)
        result = model.predict(
            source=image,
            imgsz=640,
            rect=False,
            conf=0.25,
            iou=0.45,
            max_det=10,
            retina_masks=True,
            device="cpu",
            verbose=False,
        )[0]
        padded = LetterBox(new_shape=(640, 640), auto=False, scaleup=True)(image=image)
        inputs = (
            torch.from_numpy(
                np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None])
            ).float()
            / 255
        )
        boxes = result.boxes.data.cpu().numpy()
        masks = (
            result.masks.data.cpu().numpy().astype(bool)
            if result.masks is not None
            else np.empty((0, *image.shape[:2]), bool)
        )
        np.savez(
            output / (name + ".npz"),
            source=image,
            input=inputs.numpy(),
            boxes=boxes,
            masks=masks,
        )
        rows.append(
            {
                "name": name,
                "source": str(path),
                "source_sha256": runtime.sha256(path),
                "resize_wh": resize,
                "synthetic_blank": blank,
                "reference_detections": len(boxes),
                "reference_classes": boxes[:, 5].astype(int).tolist(),
                "fixture_sha256": runtime.sha256(output / (name + ".npz")),
            }
        )
    runtime.write_json(
        output / "reference.json",
        {
            "versions": VERSIONS,
            "weights_sha256": WEIGHT_SHA,
            "cases": rows,
            "parameters": {
                "imgsz": 640,
                "rect": False,
                "conf": 0.25,
                "iou": 0.45,
                "max_det": 10,
                "retina_masks": True,
            },
        },
    )


def compare(build_root, output):
    import numpy as np
    import onnxruntime as ort
    from app.segmentation import (
        DetectionSegmentation,
        prepare_segmentation,
        decode_segmentation,
    )
    from deploy import runtime
    from prepare_vision_b1 import MANIFEST, prepare

    entry = json.loads(MANIFEST.read_text())["models"]["detection_segmentation"]
    prepare("detection_segmentation", entry, check_only=True)
    build = json.loads((build_root / "detection_segmentation/build.json").read_text())
    if (
        build["status"] != "BUILT_NOT_ACCEPTED"
        or build["onnx_sha256"] != entry["sha256"]
    ):
        raise ValueError("Mismatched segmentation build")
    references = json.loads((output / "reference/reference.json").read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(ROOT / entry["path"]), options, providers=["CPUExecutionProvider"]
    )
    model = DetectionSegmentation(build, entry)
    report = {
        "created_at": runtime.stamp(),
        "status": "RUNNING",
        "criteria": CRITERIA,
        "onnx_sha256": entry["sha256"],
        "engine_sha256": build["engine_sha256"],
        "script_sha256": runtime.sha256(Path(__file__)),
        "adapter_sha256": runtime.sha256(ROOT / "01_src/app/segmentation.py"),
        "reference": references,
        "cases": [],
        "scope": "Four fixed-input conversion/semantic checks; not COCO accuracy or B1 performance",
    }

    def save():
        runtime.write_json(output / "report.json", report)

    save()
    try:
        for row in references["cases"]:
            fixture = output / "reference" / (row["name"] + ".npz")
            if runtime.sha256(fixture) != row["fixture_sha256"]:
                raise ValueError("Reference fixture changed")
            with np.load(fixture, allow_pickle=False) as f:
                image = f["source"]
                expected_boxes = f["boxes"]
                expected_masks = f["masks"]
                reference_input = f["input"]
            inputs, transform = prepare_segmentation(image)
            input_pass = bool(
                np.allclose(inputs["images"], reference_input, atol=1e-7, rtol=0)
            )
            reference_raw = dict(
                zip([x.name for x in session.get_outputs()], session.run(None, inputs))
            )
            actual = model.model.infer(inputs)
            raw_pass = all(
                np.allclose(
                    actual[k], v, atol=CRITERIA["raw_atol"], rtol=CRITERIA["raw_rtol"]
                )
                for k, v in reference_raw.items()
            )
            boxes, masks = decode_segmentation(actual, transform)
            used = set()
            pairs = []
            for i, wanted in enumerate(expected_boxes):
                choices = []
                for j, observed in enumerate(boxes):
                    if j in used or observed[5] != wanted[5]:
                        continue
                    wh = np.maximum(
                        0,
                        np.minimum(wanted[2:4], observed[2:4])
                        - np.maximum(wanted[:2], observed[:2]),
                    )
                    intersection = float(wh.prod())
                    union = float(
                        np.prod(wanted[2:4] - wanted[:2])
                        + np.prod(observed[2:4] - observed[:2])
                        - intersection
                    )
                    choices.append((intersection / max(union, 1e-12), j))
                if not choices:
                    continue
                box_iou, j = max(choices)
                used.add(j)
                mask_union = np.logical_or(expected_masks[i], masks[j]).sum()
                pairs.append(
                    {
                        "box_iou": box_iou,
                        "mask_iou": float(
                            np.logical_and(expected_masks[i], masks[j]).sum()
                            / max(1, mask_union)
                        ),
                        "score_error": float(abs(wanted[4] - boxes[j, 4])),
                        "class_id": int(wanted[5]),
                    }
                )
            semantic_pass = len(boxes) == len(expected_boxes) == len(pairs) and all(
                p["box_iou"] >= CRITERIA["min_box_iou"]
                and p["mask_iou"] >= CRITERIA["min_mask_iou"]
                and p["score_error"] <= CRITERIA["max_score_error"]
                for p in pairs
            )
            presence = (
                (len(boxes) == 0)
                if row["synthetic_blank"]
                else any(box[5] == 0 for box in boxes)
            )
            api = model.infer(image)
            api_pass = len(api) == len(boxes) and all(
                v["mask"].shape == image.shape[:2] and not v["mask"].flags.writeable
                for v in api
            )
            case = {
                "name": row["name"],
                "input_pass": input_pass,
                "raw_pass": bool(raw_pass),
                "raw_max_abs_error": max(
                    float(np.abs(actual[k] - v).max()) for k, v in reference_raw.items()
                ),
                "reference_count": len(expected_boxes),
                "actual_count": len(boxes),
                "matches": pairs,
                "passed": bool(
                    input_pass and raw_pass and semantic_pass and presence and api_pass
                ),
            }
            report["cases"].append(case)
            np.savez(
                output / (row["name"] + "-tensorrt.npz"),
                boxes=boxes,
                masks=masks,
                **actual,
            )
            save()
            print(
                row["name"], "passed", case["passed"], "objects", len(boxes), flush=True
            )
        report["status"] = (
            "PARITY_PASS_LIMITED_FIXTURES"
            if all(c["passed"] for c in report["cases"])
            else "PARITY_FAILED"
        )
    except Exception as exc:
        report.update(status="ERROR", error=repr(exc))
        raise
    finally:
        model.close()
        save()
    return report["status"] == "PARITY_PASS_LIMITED_FIXTURES"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-python",
        type=Path,
        default=ROOT / "05_runtime/builds/vision-export/bin/python",
    )
    parser.add_argument(
        "--reference-worker", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if args.reference_worker:
        reference(output)
        return 0
    if not args.reference_python.is_file():
        parser.error("Provide the pinned CPU reference environment")
    from deploy.operations import exclusive_build
    from deploy import runtime, services

    output.mkdir(parents=True, exist_ok=False)
    folder = output / "reference"
    folder.mkdir()
    cfg = runtime.config()
    if services.profile_path(cfg).exists():
        cfg = runtime.config(services.profile_path(cfg))
    with exclusive_build(cfg, output):
        with (folder / "reference.log").open("w") as log:
            result = subprocess.run(
                [
                    str(args.reference_python),
                    str(Path(__file__).resolve()),
                    "--reference-worker",
                    "--build-root",
                    str(args.build_root),
                    "--output",
                    str(folder),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
        if result.returncode:
            raise RuntimeError("Ultralytics reference failed; inspect reference.log")
        passed = compare(args.build_root.resolve(), output)
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
