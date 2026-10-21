#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise public motion -> current TensorRT -> real greeting/cancellation.

Annotated public demonstration videos retain their original media timing and
are letterboxed to 640x480. Uniform 1.5 m depth is explicitly synthetic. This
checks integration coverage, not real-camera distance or gesture accuracy.
No model, threshold, runtime frequency or serving parameter is tuned here.
"""

import argparse
import bisect
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageSequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01_src"))
from app.camera import Camera
from app.face_state import FaceState
from app.gestures import GestureRecognizer
from app.vision_pipeline import VisionPipeline
from benchmarks.baseline_support import Events
from benchmarks.conversation_load import ConversationLoad
from benchmarks.vision_protocol import PROTOCOL, validate
from deploy import runtime, services
from deploy.resource_guard import hold_model_lease
from benchmarks.freeze import letterbox

SOURCES = (
    (
        "whc-output-1.mp4",
        "2189b99d1ac679837f7fa17404c5d10077a5d883ea2a33248ec69f872ffeee82",
    ),
    (
        "handwave-example.gif",
        "a5fff6bf197cc11593e90d066eab1d81da603e5b50aee8546006d951e7dd3114",
    ),
    (
        "whc-output-2.mp4",
        "632db3ea724285c85461e364e47ea2f5d746756352a13e464e8d7fba320d3351",
    ),
)


def decode(path, capture_fps):
    """Keep at most capture_fps source samples/second; preserve GIF durations."""
    rows, next_due = [], 0.0

    def take(image, timestamp, index):
        nonlocal next_due
        if timestamp + 1e-6 >= next_due:
            value = letterbox(image)
            value.flags.writeable = False
            rows.append((timestamp, index, value))
            next_due += 1 / capture_fps

    if path.suffix == ".gif":
        stamp = 0.0
        with Image.open(path) as movie:
            for index, frame in enumerate(ImageSequence.Iterator(movie)):
                duration = frame.info["duration"] / 1000
                if duration <= 0:
                    raise ValueError("GIF frame has no valid duration")
                take(
                    cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR),
                    stamp,
                    index,
                )
                stamp += duration
        duration = stamp
    else:
        movie = cv2.VideoCapture(str(path))
        fps = movie.get(cv2.CAP_PROP_FPS)
        expected = int(movie.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        try:
            if not movie.isOpened() or fps <= 0:
                raise ValueError("Video cannot be decoded")
            while True:
                ok, frame = movie.read()
                if not ok:
                    break
                take(frame, count / fps, count)
                count += 1
        finally:
            movie.release()
        if count != expected:
            raise ValueError("Video ended before the declared frame count")
        duration = count / fps
    if not rows or duration > 60:
        raise ValueError("Expected a nonempty clip of at most 60 seconds")
    return rows, duration


class MovieCamera(Camera):
    """Replace only the capture source; use production JPEG/history/expiry."""

    def __init__(self, cfg, rows, duration, name):
        super().__init__(
            **{
                key: getattr(cfg, key)
                for key in (
                    "width",
                    "height",
                    "jpeg_quality",
                    "capture_fps",
                    "stale_after",
                    "buffer_bytes_limit",
                    "reconnect_seconds",
                    "v4l_select_timeout",
                    "backend",
                    "serial",
                )
            },
            device=cfg.camera_device,
        )
        self.rows, self.duration, self.name = rows, duration, name
        self.times = [r[0] for r in rows]
        self.origin = None
        self.ended = threading.Event()
        self.depth = np.full((480, 640), 1.5, np.float32)
        self.valid = np.ones((480, 640), bool)
        self.depth.flags.writeable = self.valid.flags.writeable = False

    def _open(self):
        if self.origin is None:
            self.origin = time.monotonic()
        return True

    def _read_frame(self):
        received = time.monotonic()
        media = received - self.origin
        if media >= self.duration:
            self.ended.set()
            # Publish no artificial held frames. Let the owner finish its
            # in-flight inference before Camera.close invalidates the source.
            self._stop.wait()
            return False, None, None
        stamp, index, image = self.rows[
            max(0, bisect.bisect_right(self.times, media) - 1)
        ]
        return (
            True,
            image,
            (
                self.depth,
                self.valid,
                {
                    "received_monotonic": received,
                    "depth_provider": "SYNTHETIC uniform aligned 1.5m; not D435i depth",
                    "source_kind": "public_media_replay",
                    "source": self.name,
                    "media_timestamp_seconds": stamp,
                    "media_frame_index": index,
                },
            ),
        )


class RecordedMovieCamera(MovieCamera):
    """Replay a complete recorded selection with consumer backpressure.

    This fixes the RGB observations used for a functional before/after check.
    It deliberately prevents latest-frame dropping and is not a real-time or
    throughput test. Actual delivery lag is recorded, never hidden as speedup.
    """

    def __init__(self, cfg, rows, duration, name, observations):
        super().__init__(cfg, rows, duration, name)
        available = {index: (stamp, image) for stamp, index, image in rows}
        self.selection = []
        seen = set()
        for line in observations.read_text().splitlines():
            row = json.loads(line)
            if row["kind"] != "scene" or not row.get("frame"):
                continue
            frame = row["frame"]
            key = frame["source_epoch"], frame["frame_id"]
            if key in seen:
                continue
            seen.add(key)
            sensor = frame["sensor"]
            if sensor["source"] != name or sensor["media_frame_index"] not in available:
                raise ValueError("Recorded observation does not belong to this source")
            if self.selection and (
                frame["source_epoch"] != self.selection[0]["source_epoch"]
                or frame["received_monotonic"]
                <= self.selection[-1]["received_monotonic"]
            ):
                raise ValueError("Recorded source must be one ordered epoch")
            self.selection.append(frame)
        if not self.selection:
            raise ValueError("No recorded source observations")
        self.available = available
        self.source_origin = self.selection[0]["received_monotonic"]
        self.cursor, self.awaiting_frame = 0, None
        self.consumed = threading.Event()
        self.consumed.set()

    def mark_processed(self, frame_id):
        if frame_id == self.awaiting_frame:
            self.consumed.set()

    def _read_frame(self):
        while not self.consumed.wait(0.05):
            if self._stop.is_set():
                return False, None, None
        if self.cursor == len(self.selection):
            self.ended.set()
            self._stop.wait()
            return False, None, None
        selected = self.selection[self.cursor]
        due = self.origin + selected["received_monotonic"] - self.source_origin
        if self._stop.wait(max(0, due - time.monotonic())):
            return False, None, None
        received = time.monotonic()
        sensor = dict(selected["sensor"])
        stamp, image = self.available[sensor["media_frame_index"]]
        sensor.update(
            received_monotonic=received,
            source_kind="recorded_selection_with_backpressure",
            media_timestamp_seconds=stamp,
            delivery_lag_ms=(received - due) * 1000,
            original_frame_id=selected["frame_id"],
        )
        self.awaiting_frame = self._frame_id + 1
        self.cursor += 1
        self.consumed.clear()
        return True, image, (self.depth, self.valid, sensor)


def case(output, cfg, name, rows, duration, expect_stop=False, recorded=None):
    cfg = deepcopy(cfg)
    task = ROOT / "05_runtime/runtime-data/motion-probe" / output.parent.name / name
    cfg.perception.identity_gallery = str(
        (task / "identities.sqlite").relative_to(ROOT)
    )
    cfg.actions.database, cfg.audio.output_directory = (
        str(task / "actions.sqlite"),
        str(task / "audio"),
    )
    cfg.validate()
    runtime.write_json(output / "effective-config.json", cfg.effective())
    events = Events(output / "events.jsonl")
    camera = (
        RecordedMovieCamera(cfg.vision, rows, duration, name, recorded)
        if recorded
        else MovieCamera(cfg.vision, rows, duration, name)
    )
    tracker = FaceState(camera)
    gestures = GestureRecognizer(
        cfg.gestures, cfg.perception.hand_threshold, cfg.perception.pose_joint_threshold
    )
    controls = None
    decisions, results, scene_errors = [], {}, []

    def observed(scene):
        produced = gestures.update(scene)
        events.write(
            kind="scene",
            frame=scene["frame"],
            hands=scene["hands"],
            gesture_state=gestures.snapshot(),
            task_counts=scene["task_counts"],
            task_errors=scene["task_errors"],
        )
        for event in produced:
            decision = controls.control.handle_gesture(event)
            decisions.append(decision)
            events.write(kind="gesture_dispatch", decision=decision)
        if recorded and scene.get("frame") and scene.get("ready"):
            camera.mark_processed(scene["frame"]["frame_id"])

    perception = VisionPipeline(camera, cfg.perception, tracker, observed)
    report = {
        "name": name,
        "status": "RUNNING",
        "media_seconds": duration,
        "source_samples": len(rows),
        "replay_mode": "recorded_selection_with_backpressure"
        if recorded
        else "real_time_latest_frame",
        "expectation": "stationary_palm_cancel" if expect_stop else "wave_greeting",
    }
    try:
        perception.load()
        controls = ConversationLoad(
            cfg, camera, perception, events, lambda: "functional", {}
        )
        if not camera.start():
            raise RuntimeError(camera.last_error)
        if expect_stop:
            report["submitted_turn"] = controls.control.submit(
                "请详细分点说明如何分析一个复杂程序的性能，给出至少十条具体建议，不调用工具。"
            )
        perception.start()
        deadline = time.monotonic() + duration + 90
        while True:
            state = controls.control.state()
            last = state.get("last_result") or {}
            if last.get("turn_id") and state["phase"] == "idle":
                results[last["turn_id"]] = deepcopy(last)
            scene = perception.snapshot()
            if recorded and scene["failures"]:
                raise RuntimeError("A required recorded observation failed inference")
            if scene["callback_error"]:
                scene_errors.append(scene["callback_error"])
                raise RuntimeError(scene["callback_error"])
            if camera.ended.is_set() and state["phase"] == "idle":
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Media replay/conversation did not finish")
            time.sleep(0.05)
        report.update(
            decisions=decisions,
            results=results,
            perception_failures=perception.failures,
            processed=perception.processed,
            model_metrics=perception.networks.snapshot(),
            camera=camera.status(),
            scene_errors=scene_errors,
        )
        if expect_stop:
            turn_id = report["submitted_turn"]["turn_id"]
            passed = any(
                d["reason"] == "turn_cancelled" and d.get("target_turn_id") == turn_id
                for d in decisions
            )
            passed = (
                passed
                and results.get(turn_id, {}).get("status") == "cancelled"
                and not results[turn_id].get("audio_url")
            )
        else:
            admitted = [
                d["turn_id"] for d in decisions if d["reason"] == "greeting_started"
            ]
            passed = any(
                results.get(t, {}).get("status") == "complete"
                and results[t].get("audio_url")
                for t in admitted
            )
        report["status"] = (
            "EXPECTED_INTERACTION_PASS"
            if passed and not perception.failures
            else "EXPECTED_INTERACTION_NOT_PROVEN"
        )
    except BaseException as exc:
        report.update(
            status="FAILED", error=repr(exc), decisions=decisions, results=results
        )
        raise
    finally:
        errors = []
        for obj in (perception, controls, camera):
            if obj:
                try:
                    obj.close()
                except Exception as exc:
                    errors.append(repr(exc))
        tracker.stop()
        events.close()
        report["cleanup_errors"] = errors
        if errors:
            report["status"] = "FAILED_CLEANUP"
        runtime.write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument(
        "--recorded-observations",
        type=Path,
        help="Reproduce a complete prior event log's input selection; requires one video case; not a throughput test",
    )
    case_names = [Path(name).stem for name, _ in SOURCES] + ["stationary-public-palm"]
    parser.add_argument(
        "--case",
        choices=case_names,
        action="append",
        help="Run a recorded subset; default runs all cases",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol, cfg = validate(args.protocol.resolve())
    root = ROOT / "05_runtime/datasets/b1-motion-candidates"
    selected = set(args.case or case_names)
    if args.recorded_observations and (
        len(selected) != 1 or "stationary-public-palm" in selected
    ):
        raise ValueError("Recorded observations require exactly one video case")
    clips = []
    for name, digest in SOURCES:
        path = root / name
        if path.stem not in selected:
            continue
        if runtime.sha256(path) != digest:
            raise ValueError("Public motion input differs from recorded source")
        rows, duration = decode(path, cfg.vision.capture_fps)
        clips.append((path.stem, rows, duration, False))
    palm = np.load(
        ROOT / protocol["inputs"]["public-person"]["path"], allow_pickle=False
    )
    palm.flags.writeable = False
    if "stationary-public-palm" in selected:
        clips.append(("stationary-public-palm", [(0.0, 0, palm)], 5.0, True))
    prior = runtime.config(services.profile_path(runtime.config()))
    report = {
        "status": "RUNNING",
        "created_at": runtime.stamp(),
        "scope": __doc__,
        "cases": {},
        "probe_sha256": runtime.sha256(__file__),
        "runtime_protocol": str(args.protocol.resolve().relative_to(ROOT)),
        "requested_cases": sorted(selected),
    }
    if args.recorded_observations:
        report["recorded_selection"] = {
            "path": str(args.recorded_observations),
            "sha256": runtime.sha256(args.recorded_observations),
            "scope": RecordedMovieCamera.__doc__,
        }
    (output / "probe-source.py").write_bytes(Path(__file__).read_bytes())
    runtime.write_json(
        output / "sources.json", json.loads((root / "sources.json").read_text())
    )
    with services.operation(prior):
        before = services.state(list(services.ORDER), prior)
        if (
            before["status"] != "READY"
            or before["services"]["app"]["health"]["phase"] != "idle"
        ):
            raise RuntimeError("Managed services must be ready and idle")
        if any(before["services"][n]["health"]["processing"] for n in ("llama", "asr")):
            raise RuntimeError("Wait for the active backend request")
        runtime.write_json(output / "services-before.json", before)
        lease = None
        try:
            services.stop(list(services.ORDER), prior)
            runtime.write_json(
                output / "services-started.json",
                services.start(["llama", "asr", "speech"], cfg, 240),
            )
            lease = hold_model_lease()
            for name, rows, duration, stopping in clips:
                print("Motion probe:", name, flush=True)
                folder = output / name
                folder.mkdir()
                report["cases"][name] = case(
                    folder,
                    cfg,
                    name,
                    rows,
                    duration,
                    stopping,
                    args.recorded_observations,
                )
                runtime.write_json(output / "report.json", report)
            if not all(
                r["status"] == "EXPECTED_INTERACTION_PASS"
                for r in report["cases"].values()
            ):
                report["status"] = "PUBLIC_MOTION_COVERAGE_INCOMPLETE"
            elif args.recorded_observations:
                report["status"] = "RECORDED_OBSERVATION_CHAIN_PASS"
            elif selected == set(case_names):
                report["status"] = "PUBLIC_MOTION_INTERACTIONS_PASS"
            else:
                report["status"] = "PUBLIC_MOTION_SUBSET_PASS"
        except BaseException as exc:
            report.update(status="FAILED", error=repr(exc))
            raise
        finally:
            if lease:
                lease.close()
            services.stop(list(services.ORDER), cfg)
            report["restored"] = services.start(list(services.ORDER), prior, 240)[
                "status"
            ]
            runtime.write_json(output / "report.json", report)
    print(report["status"], flush=True)
    if report["status"] not in {
        "PUBLIC_MOTION_INTERACTIONS_PASS",
        "PUBLIC_MOTION_SUBSET_PASS",
        "RECORDED_OBSERVATION_CHAIN_PASS",
    }:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
