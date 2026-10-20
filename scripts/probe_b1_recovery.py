#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify B1 distance gates, stale frames, SDK restarts and repeated release.

Public-image tests use explicit synthetic depth. Live tests reopen the SDK
stream; they do not claim physical USB disconnect or real-person accuracy.
This is a functional/lifecycle probe, not a performance baseline.
"""

import argparse
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
from app.camera import Camera, FrameSnapshot
from app.config import Config
from app.vision_pipeline import VisionPipeline
from deploy.operations import exclusive_build
from deploy import runtime, services


def wait_for(predicate, seconds=15):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.025)
    raise TimeoutError("Recovery condition did not become true")


def expect_rejected(fn, message):
    try:
        fn()
    except (ValueError, RuntimeError) as exc:
        return str(exc)
    raise AssertionError(message)


def calls(owner):
    return {role: row["calls"] for role, row in owner.networks.snapshot().items()}


def assert_released(owner, engines):
    assert not owner.networks.models and owner.gallery is None
    assert not owner._thread and not owner._face_vectors
    assert not owner.entities.known_ids() and not owner.faces.known_ids()
    for engine in engines:
        assert (
            engine.context is None and engine.engine is None and engine.runtime is None
        )
        assert engine.stream is None and not engine.inputs and not engine.outputs


def replay_gates(cfg, output):
    camera = SimpleNamespace(
        source_epoch="synthetic-rgbd-recovery", stale_after=cfg.vision.stale_after
    )
    owner = VisionPipeline(camera, cfg.perception)
    image = cv2.imread(str(ROOT / "05_runtime/datasets/b0-sw-v1/public-original.png"))
    _, jpeg = cv2.imencode(".jpg", image)
    index = 0

    def frame(depth=1.5, valid=True, epoch=None, age=0):
        nonlocal index
        index += 1
        return FrameSnapshot(
            index,
            epoch or camera.source_epoch,
            time.monotonic() - age,
            image,
            jpeg.tobytes(),
            invalidated=threading.Event(),
            depth_m=np.full(image.shape[:2], depth, np.float32)
            if depth is not None
            else None,
            depth_valid=np.full(image.shape[:2], valid, bool)
            if depth is not None
            else None,
            sensor={"depth_kind": "synthetic_uniform_for_gate_tests"},
        )

    def publish(observation):
        scene = owner.process_frame(observation)
        with owner._lock:
            owner._state, owner._preview = scene, observation
        return scene

    report = {}
    try:
        owner.load()
        engines = [m.model for m in owner.networks.models.values()]
        publish(frame())
        scene = publish(frame())
        assert scene["poses"] and scene["attributes"]
        target = scene["faces"][0]["track_id"]
        owner.enroll(target, "public-recovery-fixture")
        owner.select_target(scene["persons"][0]["track_id"])
        before = calls(owner)
        for name, depth, valid in (
            ("too-near", 0.2, True),
            ("too-far", 5.0, True),
            ("invalid", np.nan, False),
            ("missing", None, False),
        ):
            scene = publish(frame(depth, valid))
            assert not scene["poses"] and not scene["hands"] and not scene["attributes"]
            assert all(
                f["identity"]["status"] == "not_triggered" for f in scene["faces"]
            )
            after = calls(owner)
            assert all(
                after[k] == before[k]
                for k in before
                if k not in ("face", "detection_segmentation")
            )
            report[name] = {"task_counts": scene["task_counts"], "model_calls": after}
        report["old_epoch"] = expect_rejected(
            lambda: publish(frame(epoch="old-source")), "Old frame accepted"
        )
        report["expired"] = expect_rejected(
            lambda: publish(frame(age=cfg.vision.stale_after + 1)),
            "Expired frame accepted",
        )
        invalid = frame()
        invalid.invalidated.set()
        report["invalidated"] = expect_rejected(
            lambda: publish(invalid), "Invalidated frame accepted"
        )
        assert calls(owner) == after

        observation = frame()
        original = owner.networks.run

        def invalidate_after_detector(role, *args, **kwargs):
            value = original(role, *args, **kwargs)
            if role == "detection_segmentation":
                observation.invalidated.set()
            return value

        owner.networks.run = invalidate_after_detector
        report["invalidated_during_inference"] = expect_rejected(
            lambda: publish(observation), "Invalidated result published"
        )
        owner.networks.run = original
        scene = publish(frame())
        assert scene["ready"] and scene["poses"]
        wait_for(lambda: not owner.snapshot()["ready"], cfg.vision.stale_after + 2)
        report["expired_enroll"] = expect_rejected(
            lambda: owner.enroll(scene["faces"][0]["track_id"], "must-not-save"),
            "Stale face enrolled",
        )
        report["expired_selection"] = expect_rejected(
            lambda: owner.select_target(scene["persons"][0]["track_id"]),
            "Stale target selected",
        )
        assert all(r["label"] != "must-not-save" for r in owner.gallery.list())
        owner.close()
        assert_released(owner, engines)
        report["released"] = True
    finally:
        owner.close()
        runtime.write_json(output / "replay-gates.json", report)
    return report


def live_recovery(cfg, output, cycles=3):
    camera = Camera(
        backend="realsense",
        capture_fps=cfg.vision.capture_fps,
        stale_after=cfg.vision.stale_after,
    )
    owner = VisionPipeline(camera, cfg.perception)
    report = {
        "cycles": [],
        "scope": "Controlled SDK close/start and injected worker exception; no physical USB disconnect",
    }
    process = psutil.Process()
    try:
        for cycle in range(cycles):
            assert camera.start(), camera.last_error
            owner.load()
            assert all(
                r["calls"] == 0 and r["p50_ms"] is None
                for r in owner.networks.snapshot().values()
            )
            engines = [m.model for m in owner.networks.models.values()]
            owner.start()
            initial = owner.processed
            wait_for(
                lambda: owner.processed >= initial + 10 and owner.snapshot()["ready"]
            )
            prior = camera.snapshot()
            observed_ids = {
                r["track_id"]
                for r in owner.snapshot().get("objects", [])
                if r["track_id"]
            }
            camera.close()
            assert prior.invalidated.is_set()
            assert (
                camera.snapshot() is None
                and not camera.status()["capture_worker_alive"]
            )
            assert not owner.snapshot()["ready"]
            assert owner.preview()[0] is None
            assert camera.start(), camera.last_error
            new_epoch = camera.source_epoch
            assert new_epoch != prior.source_epoch
            wait_for(
                lambda: owner.snapshot().get("ready")
                and owner.snapshot()["frame"]["source_epoch"] == new_epoch
            )
            current_ids = {
                r["track_id"]
                for r in owner.snapshot().get("objects", [])
                if r["track_id"]
            }
            assert not observed_ids.intersection(current_ids)

            original = owner.networks.run
            injected = threading.Event()
            failed_before = owner.failures
            errors = []

            def fault(role, *args, **kwargs):
                if role == "face" and not injected.is_set():
                    injected.set()
                    raise RuntimeError("probe injected face failure")
                return original(role, *args, **kwargs)

            owner.on_result = (
                lambda scene: errors.append(scene["error"])
                if scene.get("error") == "probe injected face failure"
                and len(errors) < 4
                else None
            )
            owner.networks.run = fault
            wait_for(lambda: injected.is_set() and owner.failures > failed_before)
            owner.networks.run = original
            wait_for(lambda: owner.snapshot()["ready"] and errors)
            owner.on_result = None
            row = {
                "cycle": cycle,
                "prior_epoch": prior.source_epoch,
                "new_epoch": new_epoch,
                "camera": camera.status(),
                "processed": owner.processed,
                "failures": owner.failures,
                "injected_failure_observed": errors,
                "model_calls": calls(owner),
                "rss_loaded": process.memory_info().rss,
                "fds_loaded": process.num_fds(),
            }
            owner.close()
            camera.close()
            assert_released(owner, engines)
            assert not camera.status()["capture_worker_alive"]
            row.update(
                rss_after_close=process.memory_info().rss,
                fds_after_close=process.num_fds(),
                models_released=True,
            )
            report["cycles"].append(row)
            runtime.write_json(output / "live-recovery.json", report)
            print("live cycle", cycle, "PASS", flush=True)
        report["status"] = (
            "LIVE_SDK_RESTART_WORKER_RECOVERY_RELEASE_PASS_LIMITED_CYCLES"
        )
    finally:
        try:
            owner.close()
        finally:
            camera.close()
            runtime.write_json(output / "live-recovery.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = Config.load(str(ROOT / "02_config/settings.orin-nx.yaml"))
    cfg.perception.identity_gallery = (
        "05_runtime/runtime-data/vision-probes/" + output.name + "/identities.sqlite"
    )
    previous = runtime.config()
    previous = runtime.config(services.profile_path(previous))
    report = {
        "created_at": runtime.stamp(),
        "status": "RUNNING",
        "scope": __doc__,
        "script_sha256": runtime.sha256(Path(__file__)),
    }
    try:
        with exclusive_build(previous, output):
            replay_gates(cfg, output)
            live_recovery(cfg, output)
        report["status"] = "B1_RECOVERY_PROBE_PASS_LIMITED_SCOPE"
    except BaseException as exc:
        report.update(status="FAILED", error=repr(exc))
        raise
    finally:
        runtime.write_json(output / "report.json", report)
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
