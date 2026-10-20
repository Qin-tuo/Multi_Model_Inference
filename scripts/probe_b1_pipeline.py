#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the complete B1 model set on public RGB with explicit synthetic depth.

The depth plane exercises gates and coordinate ownership only. This is real
TensorRT execution, but not sensor-distance/gesture-accuracy or B1 performance
qualification. Managed services are restored after the probe.
"""

import argparse
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_src"))
from app.camera import FrameSnapshot
from app.face_tracker import FaceTracker
from app.perception_b1 import B1Perception
from app.gestures import GestureRecognizer
from app.identity_gallery import IdentityGallery
from build_vision_b1 import exclusive_build
from benchmarks.vision_b1_support import validate
from deploy import runtime, services


def probe(output):
    protocol, cfg = validate()
    cfg.perception.identity_gallery = (
        "05_runtime/runtime-data/vision-probes/" + output.name + "/identities.sqlite"
    )
    camera = SimpleNamespace(
        source_epoch="public-person", stale_after=cfg.vision.stale_after
    )
    tracker = FaceTracker(camera, None)
    perception = B1Perception(camera, cfg.perception, tracker)
    gestures = GestureRecognizer(
        cfg.gestures, cfg.perception.hand_threshold, cfg.perception.pose_joint_threshold
    )
    report = {
        "created_at": runtime.stamp(),
        "status": "RUNNING",
        "scope": __doc__,
        "available_before_bytes": psutil.virtual_memory().available,
        "frames": [],
        "input_protocol": "benchmarks/manifests/b1-vision-measurements-v1.json",
    }
    try:
        perception.load()
        report["after_load_available_bytes"] = psutil.virtual_memory().available
        for name in ("public-person", "bus"):
            # Frozen inputs ship with the runtime bundle. A fresh checkout must
            # not depend on another experiment's ignored validation directory.
            image = np.load(ROOT / protocol["inputs"][name]["path"], allow_pickle=False)
            camera.source_epoch = name
            tracker._reset()
            events_seen = []
            for index in range(7 if name == "public-person" else 3):
                stamp = time.monotonic()
                _, jpeg = cv2.imencode(".jpg", image)
                frame = FrameSnapshot(
                    index + 1,
                    name,
                    stamp,
                    image,
                    jpeg.tobytes(),
                    invalidated=threading.Event(),
                    depth_m=np.full(image.shape[:2], 1.5, np.float32),
                    depth_valid=np.ones(image.shape[:2], bool),
                    sensor={"depth_kind": "synthetic_uniform_1.5m_for_gate_tests"},
                )
                scene = perception.process_frame(frame)
                with perception._lock:
                    perception._state = scene
                    perception._preview = frame
                events = gestures.update(scene)
                events_seen.extend(events)
                if name == "public-person" and index == 1:
                    report["enrollment"] = perception.enroll(
                        scene["faces"][0]["track_id"], "public-fixture"
                    )
                    report["selected_target"] = perception.select_target(
                        scene["persons"][0]["track_id"]
                    )
                if name == "public-person" and index >= 2:
                    assert scene["faces"][0]["identity"]["label"] == "public-fixture"
                    assert scene["selected_target"]["state"] == "visible"
                row = {
                    "name": name,
                    "frame_id": frame.frame_id,
                    "elapsed_ms": scene["elapsed_ms"],
                    "persons": len(scene["persons"]),
                    "objects": len(scene["objects"]),
                    "faces": len(scene["faces"]),
                    "poses": len(scene["poses"]),
                    "hands": len(scene["hands"]),
                    "attributes": len(scene["attributes"]),
                    "track_ids": [x["track_id"] for x in scene["persons"]],
                    "gesture_events": events,
                    "gesture_state": gestures.snapshot(),
                    "model_calls": {
                        k: v["calls"] for k, v in scene["model_metrics"].items()
                    },
                    "task_errors": scene["task_errors"],
                }
                report["frames"].append(row)
                runtime.write_json(output / "report.json", report)
                runtime.write_json(output / (name + f"-{index + 1}.json"), scene)
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "name",
                                "frame_id",
                                "elapsed_ms",
                                "poses",
                                "attributes",
                            )
                        }
                    ),
                    flush=True,
                )
                # Distinct synthetic observations exercise tracker initialization;
                # this delay is not part of a throughput measurement.
                time.sleep(0.15)
            if name == "public-person":
                if not any(e["kind"] == "stop" for e in events_seen):
                    raise RuntimeError(
                        "Repeated public open-palm pose did not trigger stop"
                    )
                report["public_palm_events"] = events_seen
        report["models"] = perception.networks.snapshot()
        if not all(row["calls"] > 0 for row in report["models"].values()):
            raise RuntimeError("Some B1 models were never exercised")
        report["status"] = "MODELS_EXECUTED_PERSISTENCE_CHECK_PENDING"
    except Exception as exc:
        report.update(status="FAILED", error=repr(exc))
        raise
    finally:
        try:
            perception.close()
        finally:
            report["available_after_close_bytes"] = psutil.virtual_memory().available
            runtime.write_json(output / "report.json", report)
    reopened = IdentityGallery(ROOT / cfg.perception.identity_gallery)
    try:
        report["reopened_gallery"] = reopened.list()
        assert report["reopened_gallery"][0]["label"] == "public-fixture"
        report["status"] = "COMPLETE_PIPELINE_PUBLIC_REPLAY_PASS_LIMITED_SCOPE"
    except Exception as exc:
        report.update(status="FAILED", error=repr(exc))
        raise
    finally:
        reopened.close()
        runtime.write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = runtime.config()
    cfg = runtime.config(services.profile_path(cfg))
    with exclusive_build(cfg, output):
        probe(output)


if __name__ == "__main__":
    main()
