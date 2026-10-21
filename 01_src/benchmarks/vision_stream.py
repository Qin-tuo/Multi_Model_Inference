#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Continuous visual pipeline and backend coexistence; no runtime tuning."""

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01_src"))
from app.camera import Camera
from app.face_state import FaceState
from app.gestures import GestureRecognizer
from app.vision_pipeline import VisionPipeline
from benchmarks.baseline_support import (
    Events,
    Resources,
    environment,
    run_worker,
)
from benchmarks.conversation_load import ConversationLoad
from benchmarks.vision_protocol import validate
from benchmarks.conversation_load import BackgroundLLM
from deploy import runtime, services
from deploy.resource_guard import hold_model_lease
from deploy.operations import backend_identity

WORKLOAD = ROOT / "benchmarks/manifests/vision-stream.json"


def protocol_path(spec):
    # Older workload bytes stay unchanged. Each revision identifies its own
    # runtime lock; a newer implementation cannot silently reuse an old lock.
    return ROOT / spec["module_protocol"]


def workload():
    spec = json.loads(WORKLOAD.read_text())
    if runtime.sha256(protocol_path(spec)) != spec["module_protocol_sha256"]:
        raise ValueError("Frozen module protocol changed")
    audio = spec["audio_input"]
    if runtime.sha256(ROOT / audio["path"]) != audio["sha256"]:
        raise ValueError("Background speech fixture changed")
    return spec


class MeasuredCamera(Camera):
    """Production camera publication/JPEG/history with a controlled RGBD source."""

    def __init__(self, config, protocol, spec, events, phase, live):
        super().__init__(
            device=config.camera_device,
            width=config.width,
            height=config.height,
            jpeg_quality=config.jpeg_quality,
            capture_fps=config.capture_fps,
            stale_after=config.stale_after,
            buffer_bytes_limit=config.buffer_bytes_limit,
            reconnect_seconds=config.reconnect_seconds,
            v4l_select_timeout=config.v4l_select_timeout,
            backend=config.backend,
            serial=config.serial,
        )
        self.events, self.phase, self.live, self.spec = events, phase, live, spec
        self.origin = None
        self.last_source = -1
        self.missed_source = 0
        self.read_meta = None
        self.images = {}
        if not live:
            for name in spec["replay_inputs"]:
                image = np.load(
                    ROOT / protocol["inputs"][name]["path"], allow_pickle=False
                )
                image.flags.writeable = False
                self.images[name] = image
            self.depth = np.full((480, 640), 1.5, np.float32)
            self.valid = np.ones((480, 640), bool)
            self.depth.flags.writeable = self.valid.flags.writeable = False

    def _open(self):
        if self.live:
            return super()._open()
        if self.origin is None:
            self.origin = time.monotonic()
        return True

    def _read_frame(self):
        start = time.monotonic()
        if self.live:
            ok, image, rgbd = super()._read_frame()
        else:
            index = max(
                self.last_source + 1, int((start - self.origin) * self.capture_fps)
            )
            due = self.origin + index / self.capture_fps
            if self._stop.wait(max(0, due - time.monotonic())):
                return False, None, None
            self.missed_source += max(0, index - self.last_source - 1)
            self.last_source = index
            name = self.spec["replay_inputs"][
                (index // self.spec["hold_source_frames"])
                % len(self.spec["replay_inputs"])
            ]
            received = time.monotonic()
            meta = {
                "received_monotonic": received,
                "input": name,
                "source_index": index,
                "scheduled_arrival_monotonic": due,
                "source_missed_total": self.missed_source,
                "depth_provider": "SYNTHETIC uniform aligned 1.5m; not sensor depth",
                "depth_unit": "meter",
                "depth_valid_fraction": 1.0,
            }
            ok, image, rgbd = True, self.images[name], (self.depth, self.valid, meta)
        if ok and rgbd:
            self.read_meta = rgbd[2]
            self.read_meta["benchmark_read_ms"] = (time.monotonic() - start) * 1000
        return ok, image, rgbd

    def _encode_frame_bytes(self, image):
        start = time.monotonic()
        value = super()._encode_frame_bytes(image)
        end = time.monotonic()
        if self.read_meta:
            self.read_meta["benchmark_jpeg_complete_monotonic"] = end
            self.read_meta["benchmark_jpeg_ms"] = (end - start) * 1000
        self.events.write(
            kind="jpeg",
            phase=self.phase(),
            source_epoch=self.source_epoch,
            frame_id=self._frame_id + 1,
            started_monotonic=start,
            completed_monotonic=end,
            ms=(end - start) * 1000,
            bytes=len(value) if value else None,
        )
        return value


def worker(condition, output):
    spec = workload()
    protocol, frozen = validate(protocol_path(spec))
    cfg = deepcopy(frozen)
    task = (
        ROOT
        / "05_runtime/runtime-data/vision-benchmark"
        / output.parent.name
        / condition
    )
    cfg.perception.identity_gallery = str(
        (task / "identities.sqlite").relative_to(ROOT)
    )
    cfg.actions.database = str(task / "actions.sqlite")
    cfg.audio.output_directory = str(task / "audio")
    cfg.validate()
    runtime.write_json(output / "effective-config.json", cfg.effective())
    lease = hold_model_lease()
    phase = "setup"
    events = Events(output / "events.jsonl")
    live = condition.startswith("live")
    resident = condition != "replay-only"
    identities = (
        {name: backend_identity(name, cfg) for name in ("llama", "asr", "speech")}
        if resident
        else {}
    )
    groups = {
        "worker": [os.getpid()],
        **{name: row["stable_pids"] for name, row in identities.items()},
    }
    resources = Resources(output, groups).start()
    report = {
        "condition": condition,
        "errors": [],
        "periodic_samples": [],
        "backend_identity_before": identities,
        "status": "RUNNING",
        "scope": "Production Camera/JPEG/history, VisionPipeline, FaceState and gesture rules; rules logged without dispatch to pressure requests. GUI/websocket transport excluded. Resident conditions include production conversation clients/VAD/actions/owner.",
    }
    camera = tracker = perception = controls = background = None
    failure = None

    class MeasuredPerception(VisionPipeline):
        def process_frame(self, frame):
            started = time.monotonic()
            try:
                result = super().process_frame(frame)
            except Exception as exc:
                expired = "expired" in str(exc)
                events.write(
                    kind="frame_failure",
                    phase=phase,
                    frame=frame.metadata(),
                    started_monotonic=started,
                    completed_monotonic=time.monotonic(),
                    error=repr(exc),
                    expired=expired,
                )
                if not expired:
                    report["errors"].append(repr(exc))
                raise
            completed = time.monotonic()
            sensor = frame.sensor or {}
            jpeg_end = sensor.get("benchmark_jpeg_complete_monotonic")
            events.write(
                kind="perception",
                phase=phase,
                frame=frame.metadata(),
                started_monotonic=started,
                completed_monotonic=completed,
                received_to_start_ms=(started - frame.received_monotonic) * 1000,
                post_jpeg_wait_ms=(started - jpeg_end) * 1000 if jpeg_end else None,
                processing_ms=result["elapsed_ms"],
                received_to_complete_ms=(completed - frame.received_monotonic) * 1000,
                source_schedule_to_complete_ms=(
                    completed - sensor["scheduled_arrival_monotonic"]
                )
                * 1000
                if "scheduled_arrival_monotonic" in sensor
                else None,
                stages=result["stages"],
                task_counts=result["task_counts"],
                task_errors=result["task_errors"],
                persons=len(result["persons"]),
                faces=len(result["faces"]),
                objects=len(result["objects"]),
                hands=len(result["hands"]),
            )
            return result

    try:
        camera = MeasuredCamera(cfg.vision, protocol, spec, events, lambda: phase, live)
        tracker = FaceState(
            camera,
            dead_zone=cfg.face.dead_zone,
            good_frame_zone=cfg.face.good_frame_zone,
            min_face_size=cfg.face.min_face_size,
            stable_frames=cfg.face.stable_frames,
        )
        gestures = GestureRecognizer(
            cfg.gestures,
            cfg.perception.hand_threshold,
            cfg.perception.pose_joint_threshold,
        )

        def observed(scene):
            for event in gestures.update(scene):
                events.write(kind="gesture", phase=phase, event=event)

        perception = MeasuredPerception(camera, cfg.perception, tracker, observed)
        perception.load()
        if not camera.start():
            raise RuntimeError(camera.last_error)
        perception.start()
        if resident:
            controls = ConversationLoad(
                cfg, camera, perception, events, lambda: phase, spec
            )
        if condition == "replay-4b":
            background = BackgroundLLM(cfg, events, lambda: phase)
            background.start()
        elif condition.endswith("conversation") or condition == "replay-vision-chat":
            controls.start("vision" if condition == "replay-vision-chat" else "audio")
            background = controls
        phase = resources.phase = "warmup"
        started = time.monotonic()
        while time.monotonic() - started < spec["warmup_min_seconds"] or (
            background and background.completed < spec["warmup_background_completions"]
        ):
            if (background and background.errors) or report["errors"]:
                raise RuntimeError("Warmup recorded inference errors")
            if time.monotonic() - started > spec["warmup_deadline_seconds"]:
                raise TimeoutError("Warmup did not finish")
            time.sleep(0.1)
        if not live:
            cycle = (
                len(spec["replay_inputs"])
                * spec["hold_source_frames"]
                / cfg.vision.capture_fps
            )
            start = (
                camera.origin
                + math.ceil((time.monotonic() - camera.origin) / cycle) * cycle
            )
            time.sleep(max(0, start - time.monotonic()))
        else:
            start = time.monotonic()
        report["warmup_seconds"] = time.monotonic() - started
        phase = resources.phase = "measured"
        report["window_start_monotonic"] = start
        report["camera_before"] = camera.status()
        report["perception_before"] = {
            key: perception.snapshot()[key]
            for key in ("processed", "failures", "skipped_capture_frames")
        }
        seconds, rate = spec["window_seconds"], spec["sample_hz"]
        for index in range(seconds * rate):
            due = start + index / rate
            time.sleep(max(0, due - time.monotonic()))
            tick = time.monotonic()
            row = {
                "index": index,
                "scheduled_monotonic": due,
                "sampled_monotonic": tick,
                "lag_ms": (tick - due) * 1000,
                "missing_reason": None,
            }
            if tick - due >= 1 / rate:
                row.update(
                    missing_reason="observer_missed_deadline", latest_age_ms=None
                )
            else:
                scene = perception.snapshot()
                snapshot_at = time.monotonic()
                frame = scene.get("frame")
                row.update(
                    ready=scene["ready"],
                    error=scene.get("error"),
                    callback_error=scene["callback_error"],
                    snapshot_monotonic=snapshot_at,
                    latest_age_ms=(snapshot_at - frame["received_monotonic"]) * 1000
                    if scene["ready"] and frame
                    else None,
                    processed=scene["processed"],
                    failures=scene["failures"],
                    skipped_capture_frames=scene["skipped_capture_frames"],
                    camera=camera.status(),
                    background_busy=bool(background and background.busy),
                    observer_ms=(time.monotonic() - tick) * 1000,
                )
                if scene["callback_error"]:
                    raise RuntimeError(scene["callback_error"])
            events.write(kind="periodic_sample", phase=phase, **row)
            report["periodic_samples"].append(row)
            if report["errors"] or (background and background.errors):
                raise RuntimeError("Measurement recorded inference errors")
        time.sleep(max(0, start + seconds - time.monotonic()))
        report["window_end_monotonic"] = time.monotonic()
        report["camera_after"] = camera.status()
        final = perception.snapshot()
        report["perception_after"] = {
            key: final[key]
            for key in ("processed", "failures", "skipped_capture_frames")
        }
        report["models"] = final["model_metrics"]
        report["replay_missed_source_arrivals"] = (
            camera.missed_source if not live else None
        )
        report["task_counts_at_end"] = final.get("task_counts")
        phase = resources.phase = "finish"
        if background:
            background.close()
            report["background_completed_total"] = background.completed
            report["errors"].extend(background.errors)
            if background is controls:
                controls = None
            background = None
        report["backend_identity_after"] = {
            name: backend_identity(name, cfg) for name in identities
        }
        if any(
            row["identity"] != report["backend_identity_after"][name]["identity"]
            for name, row in identities.items()
        ):
            raise RuntimeError("Backend identity changed during measurement")
    except BaseException as exc:
        failure = exc
        report["errors"].append(repr(exc))
    finally:
        phase = resources.phase = "release"
        objects = []
        for obj in (background, controls, perception, camera):
            if obj is not None and obj not in objects:
                objects.append(obj)
        for obj in objects:
            try:
                obj.close()
            except Exception as exc:
                failure = failure or exc
                report["errors"].append(repr(exc))
        if tracker:
            tracker.stop()
        try:
            resources.close()
        finally:
            events.close()
            lease.close()
        report["errors"].extend(resources.errors)
        report["rss_high_water_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        report["status"] = "MEASURED" if not report["errors"] else "FAILED"
        runtime.write_json(output / "result.json", report)
    if failure or report["errors"]:
        raise RuntimeError(
            "Vision stream measurement failed; inspect result.json"
        ) from failure


def main():
    global WORKLOAD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument(
        "--workload",
        type=Path,
        default=WORKLOAD,
        help="Recorded workload JSON; use a separate file for bounded soak runs",
    )
    args = parser.parse_args()
    WORKLOAD = args.workload.resolve()
    output = args.output.resolve()
    spec = workload()
    conditions = args.conditions or spec["conditions"]
    if set(conditions) - set(spec["conditions"]) or (
        args.worker and args.worker not in spec["conditions"]
    ):
        raise ValueError("Unknown stream condition")
    if args.worker:
        worker(args.worker, output)
        return
    output.mkdir(parents=True, exist_ok=False)
    _, cfg = validate(protocol_path(spec))
    environment(output)
    runtime.write_json(output / "workload.json", spec)
    prior = runtime.config()
    prior = runtime.config(services.profile_path(prior))
    report = {
        "created_at": runtime.stamp(),
        "status": "RUNNING",
        "conditions": {},
        "workload_sha256": runtime.sha256(WORKLOAD),
        "module_protocol": str(protocol_path(spec).relative_to(ROOT)),
    }
    with services.operation(prior):
        before = services.state(list(services.ORDER), prior)
        runtime.write_json(output / "services-before.json", before)
        if before["services"]["app"].get("health", {}).get("phase") not in (
            None,
            "idle",
        ):
            raise RuntimeError("Wait for the active application turn")
        for name, row in before["services"].items():
            if row["cgroup_pids"] and not row["managed_by_project"]:
                raise RuntimeError("Unrelated service: " + name)
        active = [
            n
            for n in services.ORDER
            if before["services"][n].get("ActiveState") == "active"
        ]
        try:
            for condition in conditions:
                stopped = services.stop(list(services.ORDER), prior)
                if any(r["cgroup_pids"] for r in stopped["services"].values()):
                    raise RuntimeError("Managed service processes remain")
                folder = output / condition
                folder.mkdir()
                runtime.write_json(folder / "services-stopped.json", stopped)
                if condition != "replay-only":
                    runtime.write_json(
                        folder / "services-started.json",
                        services.start(["llama", "asr", "speech"], cfg, 240),
                    )
                print("Measuring vision:", condition, flush=True)
                with (folder / "worker.log").open("w") as log:
                    code = run_worker(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            condition,
                            "--output",
                            str(folder),
                            "--workload",
                            str(WORKLOAD),
                        ],
                        log,
                        seconds=spec["worker_timeout_seconds"],
                    )
                report["conditions"][condition] = {
                    "exit_code": code,
                    "result": str((folder / "result.json").relative_to(ROOT)),
                }
                runtime.write_json(output / "report.json", report)
                if code:
                    raise RuntimeError("Condition failed: " + condition)
            report["status"] = "MEASURED"
        except BaseException as exc:
            report.update(status="FAILED", error=repr(exc))
            raise
        finally:
            try:
                services.stop(list(services.ORDER), cfg)
                restored = (
                    services.start(active, prior, 240)
                    if active
                    else services.state(list(services.ORDER), prior, False)
                )
                runtime.write_json(output / "services-restored.json", restored)
            finally:
                runtime.write_json(output / "report.json", report)


if __name__ == "__main__":
    main()
