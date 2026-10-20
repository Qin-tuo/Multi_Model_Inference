#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Re-infer saved gesture previews with the replacement pose model.

JPEG previews are lossy. Recorded detector selections/ranges and frame times
are reused explicitly; this is pose/rule regression, not full RGBD or live
throughput acceptance. No dialogue, audio or tool actions are dispatched.
"""

import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "01_src"), str(ROOT / "scripts")]
from app.config import Config
from app.gestures import GestureRecognizer
from app.vision_models import VisionModels
from build_vision_b1 import exclusive_build
from deploy import runtime, services


def replay(recording, models, cfg, output):
    record = json.loads(recording.read_text())
    images = {
        (row["frame"]["source_epoch"], row["frame"]["frame_id"]): row
        for row in record["preview"]["frames"]
    }
    recognizer = GestureRecognizer(
        cfg.gestures, cfg.perception.hand_threshold, cfg.perception.pose_joint_threshold
    )
    report = {
        "scope": __doc__,
        "recording": str(recording.relative_to(ROOT)),
        "recording_sha256": runtime.sha256(recording),
        "frames": [],
        "events": [],
        "missing_previews": [],
        "recorded_actions": record["actions"],
    }
    for sample in record["samples"]:
        original = sample["perception"]
        frame = original["frame"]
        key = frame["source_epoch"], frame["frame_id"]
        item = images.get(key)
        if item is None:
            report["missing_previews"].append(frame["frame_id"])
            # Absence is a continuity break, never an invented observation.
            recognizer.update({"ready": False}, now=sample["observed_monotonic"])
            continue
        path = (ROOT / item["path"]).resolve()
        if (
            not path.is_relative_to(ROOT / "05_runtime/validation")
            or runtime.sha256(path) != item["sha256"]
        ):
            raise ValueError("Preview source changed or is outside local validation")
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (frame["height"], frame["width"]):
            raise ValueError("Preview image and frame geometry differ")
        targets = {pose["target_id"] for pose in original.get("poses", [])}
        hands = []
        for person in original["persons"]:
            if person["track_id"] not in targets:
                continue
            pose = models.run(
                "body_hand_pose",
                image,
                person["box_xyxy"],
                joint_threshold=cfg.perception.pose_joint_threshold,
            )
            hands.extend(
                dict(hand, person_track_id=person["track_id"]) for hand in pose["hands"]
            )
        events = recognizer.update(
            {"ready": original["ready"], "frame": frame, "hands": hands},
            now=sample["observed_monotonic"],
        )
        report["events"].extend(events)
        # snapshot() would compare historical frame times to today's clock.
        report["frames"].append(
            {
                "frame": frame,
                "hands": hands,
                "gesture_state": recognizer._state,
                "observed_monotonic": sample["observed_monotonic"],
            }
        )
    report["status"] = "RECORDED_MODEL_GESTURE_OBSERVATIONS"
    runtime.write_json(output / (recording.stem + ".json"), report)
    print(
        recording.name,
        [(e["kind"], e["frame_id"]) for e in report["events"]],
        "missing_previews",
        len(report["missing_previews"]),
        flush=True,
    )
    return {key: value for key, value in report.items() if key != "frames"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "02_config/settings.orin-nx-b2.yaml"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = Config.load(args.config)
    prior = runtime.config()
    prior = runtime.config(services.profile_path(prior))
    results = []
    with exclusive_build(prior, output):
        models = VisionModels(cfg.perception.model_manifest)
        try:
            models.load(roles=["body_hand_pose"])
            for recording in args.recording:
                results.append(replay(recording.resolve(), models, cfg, output))
        finally:
            models.close()
    runtime.write_json(
        output / "summary.json",
        {
            "scope": __doc__,
            "model_manifest": cfg.perception.model_manifest,
            "recordings": results,
        },
    )


if __name__ == "__main__":
    main()
