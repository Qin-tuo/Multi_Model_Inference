#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V2: production Camera/Perception, fixed-rate replay and native 4B coexistence.

The replay replaces only the camera driver read, keeping the production capture,
JPEG ring, frame ownership and perception worker. It has an independent arrival
clock and skips already missed arrivals rather than slowing down for inference.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.baseline_support import (  # noqa: E402
    Events,
    Resources,
    distribution,
    environment,
    run_worker,
    validate,
)
from deploy import runtime, services  # noqa: E402

CONDITIONS = ("replay-only", "replay-resident", "replay-4b", "live-resident")
PROMPT = "用中文分点介绍如何分析程序的性能，先测量再定位瓶颈，回答尽量具体。"


class ReplayCapture:
    def __init__(self, images, stop, events, phase, rate=10):
        self.images, self.stop, self.events, self.phase = images, stop, events, phase
        self.rate = rate
        self.origin = time.monotonic()
        self.last = -1
        self.missed = 0
        self.delivered = 0
        self.closed = False

    def read(self):
        now = time.monotonic()
        index = max(self.last + 1, int((now - self.origin) * self.rate))
        due = self.origin + index / self.rate
        if self.stop.wait(max(0, due - now)) or self.closed:
            return False, None
        self.missed += max(0, index - self.last - 1)
        self.last = index
        self.delivered += 1
        name, image = self.images[(index // 20) % len(self.images)]
        self.events.write(
            kind="source_arrival",
            phase=self.phase(),
            source_index=index,
            delivered_index=self.delivered,
            scheduled_monotonic=due,
            delivered_monotonic=time.monotonic(),
            input=name,
            missed_arrivals_total=self.missed,
        )
        return True, image

    def release(self):
        self.closed = True


class BackgroundLLM:
    def __init__(self, cfg, events, phase):
        from app.llm import LLM

        self.model = LLM(
            **{k: v for k, v in vars(cfg.llm).items() if k != "system_prompt_no_rag"}
        )
        self.events, self.phase = events, phase
        self.cancel = threading.Event()
        self.thread = None
        self.completed = 0
        self.errors = []
        self.busy = False

    def start(self):
        if not self.model.load():
            raise RuntimeError(self.model.last_error)
        self.thread = threading.Thread(
            target=self._loop, name="baseline-4b", daemon=True
        )
        self.thread.start()

    def _loop(self):
        index = 0
        while not self.cancel.is_set():
            index += 1
            start = time.monotonic()
            first = None
            chars = 0
            metadata = {}
            self.busy = True
            text = PROMPT + f"这是第{index}个请求。"
            error = None
            try:
                for chunk, meta in self.model.generate_stream(
                    text, cancel_event=self.cancel
                ):
                    metadata.update(meta)
                    if chunk:
                        first = first or time.monotonic()
                        chars += len(chunk)
                if not metadata.get("done") and not self.cancel.is_set():
                    raise RuntimeError("Background generation did not complete")
                if metadata.get("done"):
                    self.completed += 1
            except Exception as exc:
                error = repr(exc)
                if not self.cancel.is_set():
                    self.errors.append(error)
            finally:
                self.busy = False
                self.events.write(
                    kind="background_llm",
                    phase=self.phase(),
                    request=index,
                    prompt=text,
                    started_monotonic=start,
                    completed_monotonic=time.monotonic(),
                    first_content_seconds=first - start if first else None,
                    chars=chars,
                    metadata=metadata,
                    window_end_cancel=self.cancel.is_set(),
                    error=error,
                )
            if self.errors:
                return
            deadline = time.monotonic() + 30
            while not self.cancel.is_set() and not self.model.idle():
                if time.monotonic() > deadline:
                    self.errors.append("Background native slot did not release")
                    return
                self.cancel.wait(0.02)

    def close(self):
        self.cancel.set()
        self.model.cancel(self.cancel)
        if self.thread:
            self.thread.join(10)
            if self.thread.is_alive():
                raise RuntimeError("Background model client did not release")
        deadline = time.monotonic() + 30
        while not self.model.idle():
            if time.monotonic() > deadline:
                raise RuntimeError("Background native model did not release")
            time.sleep(0.02)
        self.model.unload()


def worker(condition, out):
    import cv2
    from app.camera import Camera
    from app.face_detector import FaceDetector
    from app.face_tracker import FaceTracker
    from app.perception import Perception
    from deploy.resource_guard import hold_model_lease
    from scripts.probe_coexistence import backend_identity

    protocol, cfg, _ = validate()
    _lease = hold_model_lease(runtime.paths(cfg)["stack"] / "cache")
    phase = "warmup"
    events = Events(out / "events.jsonl")
    groups = {"vision": [os.getpid()]}
    identities = {}
    if condition != "replay-only":
        identities = {
            name: backend_identity(name, cfg) for name in ("llama", "asr", "speech")
        }
        groups.update({name: row["stable_pids"] for name, row in identities.items()})
    resources = Resources(out, groups).start()
    report = {
        "condition": condition,
        "status": "RUNNING",
        "errors": [],
        "backend_identity_before": identities,
        "opencv_threads": cv2.getNumThreads(),
        "periodic_samples": [],
        "scope": "Production Camera + FaceTracker + Perception only; GUI, gestures, VAD and action control are in E1, not this controlled vision comparison.",
    }
    camera = face = tracker = perception = background = None
    try:

        class BenchCamera(Camera):
            def _open(self):
                if condition == "live-resident":
                    result = super()._open()
                    if result:
                        # A local driver proxy times read while forwarding release.
                        from benchmarks.baseline_support import NativeProxy

                        class ReadTimes:
                            def wrap(self, label, function):
                                def read(*args, **kwargs):
                                    start = time.monotonic()
                                    value = function(*args, **kwargs)
                                    events.write(
                                        kind=label,
                                        phase=phase,
                                        ms=(time.monotonic() - start) * 1000,
                                        ok=bool(value[0]),
                                    )
                                    return value

                                return read

                        self._cap = NativeProxy(
                            self._cap, ReadTimes(), {"read": "driver_read"}
                        )
                    return result
                images = []
                for name in ("public-camera-shape.png", "blank-camera-shape.png"):
                    image = cv2.imread(str(Path(protocol["input_directory"]) / name))
                    image.flags.writeable = False
                    images.append((name, image))
                self._cap = ReplayCapture(
                    images, self._stop, events, lambda: phase, cfg.vision.capture_fps
                )
                self.replay = self._cap
                return True

            def _encode_frame_bytes(self, frame):
                start = time.monotonic()
                value = super()._encode_frame_bytes(frame)
                events.write(
                    kind="jpeg_encode",
                    phase=phase,
                    ms=(time.monotonic() - start) * 1000,
                    bytes=len(value) if value is not None else 0,
                )
                return value

            def snapshot(self, **kwargs):
                start = time.monotonic()
                value = super().snapshot(**kwargs)
                events.write(
                    kind="frame_fetch",
                    phase=phase,
                    ms=(time.monotonic() - start) * 1000,
                    available=value is not None,
                )
                return value

        camera = BenchCamera(
            device=cfg.vision.camera_device,
            width=cfg.vision.width,
            height=cfg.vision.height,
            jpeg_quality=cfg.vision.jpeg_quality,
            capture_fps=cfg.vision.capture_fps,
            stale_after=cfg.vision.stale_after,
            buffer_bytes_limit=cfg.vision.buffer_bytes_limit,
            reconnect_seconds=cfg.vision.reconnect_seconds,
            v4l_select_timeout=cfg.vision.v4l_select_timeout,
        )
        face = FaceDetector()
        if not face.load():
            raise RuntimeError(face.last_error)
        tracker = FaceTracker(
            camera,
            face,
            fps=cfg.face.fps,
            dead_zone=cfg.face.dead_zone,
            good_frame_zone=cfg.face.good_frame_zone,
            min_face_size=cfg.face.min_face_size,
            stable_frames=cfg.face.stable_frames,
        )
        previous = None

        def completed(scene):
            nonlocal previous
            key = (scene.get("completed_monotonic"), scene["failures"])
            if key == previous:
                return
            previous = key
            events.write(
                kind="perception",
                phase=phase,
                **{
                    k: scene.get(k)
                    for k in (
                        "ready",
                        "error",
                        "frame",
                        "completed_monotonic",
                        "elapsed_ms",
                        "stages",
                        "processed",
                        "skipped_capture_frames",
                        "failures",
                    )
                },
                persons=len(scene["persons"]),
                hands=len(scene["hands"]),
                face=scene["face"]["face_detected"],
            )

        perception = Perception(camera, cfg.perception, tracker, completed)
        perception.load()
        if not camera.start():
            raise RuntimeError(camera.last_error)
        perception.start()
        if condition == "replay-4b":
            background = BackgroundLLM(cfg, events, lambda: phase)
            background.start()
        warm_start = time.monotonic()
        while time.monotonic() - warm_start < 20 or (
            background and background.completed < 5
        ):
            if background and background.errors:
                raise RuntimeError(str(background.errors))
            if time.monotonic() - warm_start > 120:
                raise RuntimeError("Vision/background warmup did not finish")
            time.sleep(0.1)
        report["warmup_seconds"] = time.monotonic() - warm_start
        if condition != "live-resident":
            # Align every measured replay to the same 40-frame input cycle.
            cycle_seconds = 40 / cfg.vision.capture_fps
            start = (
                camera.replay.origin
                + math.ceil((time.monotonic() - camera.replay.origin) / cycle_seconds)
                * cycle_seconds
            )
            time.sleep(max(0, start - time.monotonic()))
        else:
            start = time.monotonic()
        phase = resources.phase = "measured"
        report["window_start_monotonic"] = start
        report["camera_before"] = camera.status()
        rate, count = 10, 600
        for index in range(count):
            due = start + index / rate
            time.sleep(max(0, due - time.monotonic()))
            tick = time.monotonic()
            if tick - due >= 1 / rate:
                row = {
                    "index": index,
                    "scheduled_monotonic": due,
                    "sampled_monotonic": None,
                    "missing_reason": "observer_missed_deadline",
                    "latest_age_ms": None,
                }
            else:
                scene = perception.snapshot()
                frame = scene.get("frame")
                row = {
                    "index": index,
                    "scheduled_monotonic": due,
                    "sampled_monotonic": tick,
                    "lag_ms": (tick - due) * 1000,
                    "ready": scene["ready"],
                    "latest_age_ms": (tick - frame["received_monotonic"]) * 1000
                    if scene["ready"] and frame
                    else None,
                    "scene_error": scene.get("error"),
                    "camera": camera.status(),
                    "captured_count": camera._frame_id,
                    "capture_fps_since_open": camera.actual_fps,
                    "processed": scene["processed"],
                    "skipped_capture_frames": scene["skipped_capture_frames"],
                    "failures": scene["failures"],
                    "background_api_busy": bool(background and background.busy),
                }
            report["periodic_samples"].append(row)
            events.write(kind="periodic_sample", phase=phase, **row)
        time.sleep(max(0, start + 60 - time.monotonic()))
        report["window_end_monotonic"] = time.monotonic()
        report["camera_after"] = camera.status()
        report["replay_missed_arrivals"] = getattr(
            getattr(camera, "replay", None), "missed", None
        )
        phase = resources.phase = "finish"
        if background:
            background.close()
            report["background_completed"] = background.completed
            report["errors"].extend(background.errors)
            background = None
        report["backend_identity_after"] = {
            name: backend_identity(name, cfg) for name in identities
        }
        if any(
            row["identity"] != report["backend_identity_after"][name]["identity"]
            for name, row in identities.items()
        ):
            report["errors"].append(
                "Native backend identity changed during measurement"
            )
    except BaseException as exc:
        report["errors"].append(repr(exc))
        raise
    finally:
        for name, obj in [
            ("background", background),
            ("perception", perception),
            ("camera", camera),
        ]:
            if obj:
                try:
                    obj.close()
                except Exception as exc:
                    report["errors"].append(name + " cleanup: " + repr(exc))
        if tracker:
            tracker.stop()
        if face:
            face.unload()
        resources.close()
        events.close()
        report["errors"].extend(resources.errors)
        finite = [
            x["latest_age_ms"]
            for x in report["periodic_samples"]
            if x["latest_age_ms"] is not None
        ]
        missing = len(report["periodic_samples"]) - len(finite)
        report["finite_age_ms"] = distribution(finite)
        rank = math.ceil(0.95 * len(report["periodic_samples"])) - 1
        report["age_p95_including_missing_ms"] = (
            sorted(finite)[rank] if 0 <= rank < len(finite) else None
        )
        report["no_result_fraction"] = (
            missing / len(report["periodic_samples"])
            if report["periodic_samples"]
            else 1
        )
        report["status"] = (
            "MEASURED" if not report["errors"] else "MEASURED_WITH_FAILURES"
        )
        runtime.write_json(out / "result.json", report)
    if report["errors"]:
        raise RuntimeError("Continuous vision measurement recorded failures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=CONDITIONS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker, args.output)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    _, cfg, _ = validate()
    before = services.state(services.ORDER, cfg)
    if (
        before["status"] != "READY"
        or before["services"]["app"]["health"]["phase"] != "idle"
    ):
        raise RuntimeError("Managed services must be ready and idle")
    environment(out)
    runtime.write_json(out / "effective-config.json", cfg.effective())
    workload = {
        "seconds_each": 60,
        "sample_hz": 10,
        "replay_arrival_hz": cfg.vision.capture_fps,
        "replay_pattern": "public-camera-shape.png for source indices 0..19, blank-camera-shape.png for 20..39, repeat on absolute source clock",
        "background_prompt": PROMPT,
        "background_suffix": "这是第{index}个请求。",
        "background_policy": "Single backlogged caller, same native config, no history/tools; next call after actual slot idle; cancel in-flight call at window end. This is interference load, not fixed-arrival LLM capacity.",
        "warmup": "At least 20 seconds for full JPEG ring; with background 4B also wait for five completed model calls.",
    }
    runtime.write_json(out / "workload.json", workload)
    report = {
        "created_at": runtime.stamp(),
        "status": "RUNNING",
        "before": before,
        "conditions": {},
        "workload_sha256": runtime.sha256(out / "workload.json"),
    }
    error = None
    try:
        with services.operation(cfg):
            services.stop(list(services.ORDER), cfg)
        for index, condition in enumerate(CONDITIONS):
            if index == 1:
                with services.operation(cfg):
                    services.start(["llama", "asr", "speech"], cfg, 240)
            print("V2 measuring: " + condition, flush=True)
            folder = out / condition
            folder.mkdir()
            with (folder / "process.log").open("w") as log:
                code = run_worker(
                    [
                        sys.executable,
                        __file__,
                        "--worker",
                        condition,
                        "--output",
                        str(folder),
                    ],
                    log,
                    360,
                )
            report["conditions"][condition] = {
                "exit_code": code,
                "result": str(folder / "result.json"),
            }
            runtime.write_json(out / "report.json", report)
            if code:
                raise RuntimeError("Vision condition failed: " + condition)
    except BaseException as exc:
        error = exc
        report["error"] = repr(exc)
    finally:
        try:
            with services.operation(cfg):
                report["restored"] = services.start(list(services.ORDER), cfg, 240)
        except BaseException as exc:
            error = error or exc
            report["restore_error"] = repr(exc)
        report["status"] = "FAIL" if error else "MEASURED"
        report["finished_at"] = runtime.stamp()
        runtime.write_json(out / "report.json", report)
    if error:
        raise error
    print(json.dumps({"status": report["status"], "report": str(out / "report.json")}))


if __name__ == "__main__":
    main()
