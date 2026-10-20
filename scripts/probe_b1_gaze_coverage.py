#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Attribute input coverage on frozen B1 images; not an accuracy benchmark."""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_src"))
from app.vision_models import VisionModels
from benchmarks.vision_b1_support import validate
from deploy.operations import exclusive_build
from deploy import runtime, services


def probe(output):
    protocol, cfg = validate()
    models = VisionModels()
    report = {
        "created_at": runtime.stamp(),
        "status": "RUNNING",
        "faces": [],
        "scope": "Frozen 640x480 fixture/face ROI coverage; actual TensorRT adapters, no field-accuracy claim",
    }
    try:
        models.load(("face", "head_pose", "gaze_landmarks", "gaze"))
        gaze = models.models["gaze"]
        original = gaze.model.infer
        calls = [0]

        def observed(values):
            calls[0] += 1
            return original(values)

        gaze.model.infer = observed
        for name, spec in protocol["inputs"].items():
            image = np.load(ROOT / spec["path"], allow_pickle=False)
            for index, face in enumerate(
                models.run("face", image)[: cfg.perception.max_face_targets]
            ):
                head = models.run("head_pose", image, face["box_xyxy"])
                points = models.run("gaze_landmarks", image, face["box_xyxy"])[
                    "landmarks"
                ]
                eyes = []
                for pair in (points[:2], points[2:4]):
                    side = int(1.8 * np.linalg.norm(pair[0] - pair[1]))
                    center = ((pair[0] + pair[1]) / 2).astype(np.int32)
                    x, y = int(center[0]) - side // 2, int(center[1]) - side // 2
                    eyes.append(
                        {
                            "side_pixels": side,
                            "box_xyxy": [x, y, x + side, y + side],
                            "inside_image": x >= 0
                            and y >= 0
                            and x + side <= image.shape[1]
                            and y + side <= image.shape[0],
                        }
                    )
                row = {"input": name, "face_index": index, "face": face, "eyes": eyes}
                before = calls[0]
                try:
                    row["gaze"] = models.run(
                        "gaze", image, points, head["angles_degrees"]
                    )
                    row["status"] = "INFERRED"
                except ValueError as exc:
                    row.update(status="INPUT_REJECTED", reason=str(exc))
                    assert any(
                        e["side_pixels"] < 4 or not e["inside_image"] for e in eyes
                    )
                    assert calls[0] == before
                row["gaze_engine_calls"] = calls[0] - before
                report["faces"].append(row)
        report["status"] = "GAZE_INPUT_COVERAGE_INSPECTED"
    except BaseException as exc:
        report.update(status="FAILED", error=repr(exc))
        raise
    finally:
        models.close()
        runtime.write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = runtime.config()
    cfg = runtime.config(services.profile_path(cfg))
    with exclusive_build(cfg, output):
        report = probe(output)
    print(
        report["status"],
        [(r["input"], r["face_index"], r["status"]) for r in report["faces"]],
        flush=True,
    )


if __name__ == "__main__":
    main()
