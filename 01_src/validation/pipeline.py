#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the complete current model set on public RGB with explicit synthetic depth.

The depth plane exercises gates and coordinate ownership only. This is real
TensorRT execution, but not sensor-distance/gesture-accuracy or performance
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01_src"))
from app.camera import FrameSnapshot
from app.config import Config
from app.face_state import FaceState
from app.vision_pipeline import VisionPipeline
from app.gestures import GestureRecognizer
from app.identity_gallery import IdentityGallery
from deploy.operations import exclusive_build
from benchmarks.vision_protocol import PROTOCOL, validate
from deploy import runtime, services
from deploy.validation_inputs import input_path, is_local_input


def held_palm_images(fixture_path):
    """Load an independently labeled segment; never select frames by predictions."""
    fixture = json.loads(fixture_path.read_text())
    recording = input_path(ROOT / fixture["recording"], ROOT)
    if runtime.sha256(recording) != fixture["recording_sha256"]:
        raise ValueError("Held-palm recording changed")
    record = json.loads(recording.read_text())
    first, last = fixture["frame_interval_inclusive"]
    rows = [
        row
        for row in record["preview"]["frames"]
        if first <= row["frame"]["frame_id"] <= last
    ]
    if len(rows) != fixture["frame_count"] or not 5 <= len(rows) <= 300:
        raise ValueError("Held-palm segment is incomplete or unbounded")
    images = []
    for row in rows:
        path = input_path(ROOT / row["path"], ROOT).resolve()
        if (
            not is_local_input(path, ROOT)
            or runtime.sha256(path) != row["sha256"]
        ):
            raise ValueError("Held-palm preview changed")
        image = cv2.imread(str(path))
        if image is None or image.shape != (480, 640, 3):
            raise ValueError("Held-palm preview geometry changed")
        images.append(image)
    return images, {
        "path": str(fixture_path.relative_to(ROOT)),
        "sha256": runtime.sha256(fixture_path),
        "frames": len(images),
        "scope": fixture["scope"],
    }


def probe(output, protocol_path=PROTOCOL, config_path=None, gesture_fixture=None):
    if config_path is None:
        protocol, cfg = validate(protocol_path)
    else:
        # Candidate integration precedes baseline freezing. Reuse only the
        # checked input pixels; do not claim the old runtime lock is applicable.
        protocol = json.loads(Path(protocol_path).read_text())
        cfg = Config.load(config_path)
        for value in protocol["inputs"].values():
            if runtime.sha256(ROOT / value["path"]) != value["sha256"]:
                raise ValueError("Functional probe input bytes changed")
    cfg.perception.identity_gallery = (
        "05_runtime/runtime-data/vision-probes/" + output.name + "/identities.sqlite"
    )
    camera = SimpleNamespace(
        source_epoch="public-person", stale_after=cfg.vision.stale_after
    )
    tracker = FaceState(camera)
    perception = VisionPipeline(camera, cfg.perception, tracker)
    gestures = GestureRecognizer(
        cfg.gestures, cfg.perception.hand_threshold, cfg.perception.pose_joint_threshold
    )
    report = {
        "created_at": runtime.stamp(),
        "status": "RUNNING",
        "scope": __doc__,
        "available_before_bytes": psutil.virtual_memory().available,
        "frames": [],
        "input_protocol": str(Path(protocol_path).resolve().relative_to(ROOT)),
        "config_override": str(config_path) if config_path else None,
        "runtime_scope": "Candidate model integration; inherited input pixels only"
        if config_path
        else "Frozen runtime and inputs",
    }
    try:
        perception.load()
        report["after_load_available_bytes"] = psutil.virtual_memory().available
        sequences = []
        for name in ("public-person", "bus"):
            # Frozen inputs ship with the runtime bundle. A fresh checkout must
            # not depend on another experiment's ignored validation directory.
            image = np.load(ROOT / protocol["inputs"][name]["path"], allow_pickle=False)
            sequences.append((name, [image] * (7 if name == "public-person" else 3)))
        if gesture_fixture:
            images, report["gesture_fixture"] = held_palm_images(gesture_fixture)
            sequences.append(("recorded-held-palm", images))
            report["public_palm_scope"] = (
                "Historical B1 response retained as a comparison, not a gesture "
                "label: the posed photo has bent fingers and a cropped hand. "
                "The separately labeled complete held-palm segment must emit stop "
                "and no wave. Gesture rules and thresholds are unchanged."
            )
        for name, images in sequences:
            camera.source_epoch = name
            tracker._reset()
            events_seen = []
            for index, image in enumerate(images):
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
                report["public_palm_events"] = events_seen
                if not gesture_fixture and not any(
                    e["kind"] == "stop" for e in events_seen
                ):
                    raise RuntimeError(
                        "Repeated public open-palm pose did not trigger stop"
                    )
            if name == "recorded-held-palm":
                report["recorded_palm_events"] = events_seen
                if not any(e["kind"] == "stop" for e in events_seen) or any(
                    e["kind"] == "wave" for e in events_seen
                ):
                    raise RuntimeError("Labeled held-palm segment must stop, not wave")
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
    policy = perception.networks.entries["face_identity"].get("gallery_policy", {})
    reopened = IdentityGallery(
        ROOT / cfg.perception.identity_gallery,
        **{k: policy[k] for k in ("dimensions", "threshold", "margin") if k in policy},
    )
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
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument(
        "--config",
        type=Path,
        help="Candidate integration using only the protocol's input pixels",
    )
    parser.add_argument(
        "--gesture-fixture",
        type=Path,
        help="Pinned operator-confirmed held-palm segment for explicit gesture acceptance",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = runtime.config()
    cfg = runtime.config(services.profile_path(cfg))
    with exclusive_build(cfg, output):
        probe(
            output,
            args.protocol,
            args.config,
            args.gesture_fixture.resolve() if args.gesture_fixture else None,
        )


if __name__ == "__main__":
    main()
