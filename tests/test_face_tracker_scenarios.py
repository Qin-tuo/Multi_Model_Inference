# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import numpy as np
from app.camera import Camera
from app.face_tracker import FaceTracker


class Detector:
    box = (40, 40, 60, 60)
    calls = 0
    def detect(self, frame):
        self.calls += 1
        return self.box
    def health_check(self): return True


def publish(camera, frame_id, timestamp=None):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    camera._latest = (timestamp or time.monotonic(), image)
    camera._latest_jpeg = camera._encode_frame_bytes(image)
    camera._frame_id = frame_id
    camera._alive = True


def test_stability_requires_distinct_frames_and_resets_after_loss():
    camera = Camera(); detector = Detector(); tracker = FaceTracker(camera, detector, stable_frames=2)
    publish(camera, 1); tracker._tick(); tracker._tick()
    assert tracker.face_detected and not tracker.stable and detector.calls == 1
    publish(camera, 2); tracker._tick()
    assert tracker.stable
    detector.box = None; publish(camera, 3); tracker._tick()
    assert not tracker.face_detected and not tracker.stable and tracker.last_face_box is None


def test_camera_restart_resets_stability_and_old_result():
    camera = Camera(); tracker = FaceTracker(camera, Detector(), stable_frames=2)
    publish(camera, 1); tracker._tick(); publish(camera, 2); tracker._tick()
    assert tracker.stable
    camera.source_epoch = "restarted"
    assert not tracker.face_detected
    publish(camera, 1); tracker._tick()
    assert tracker.face_detected and not tracker.stable


def test_expired_face_is_not_reported_as_live():
    camera = Camera(stale_after=0.01); tracker = FaceTracker(camera, Detector())
    publish(camera, 1); tracker._tick()
    time.sleep(0.02)
    assert not tracker.face_detected
    tracker._tick()
    assert tracker.snapshot()["frame"] is None


def test_off_center_face_does_not_require_a_motor_controller():
    camera = Camera(); detector = Detector(); detector.box = (5, 5, 25, 25)
    tracker = FaceTracker(camera, detector)
    publish(camera, 1); tracker._tick()
    assert tracker.face_detected and not tracker.centered
    assert tracker.snapshot()["frame"]["frame_id"] == 1


def test_preview_pairs_pixels_and_box_from_same_frame():
    camera = Camera(); tracker = FaceTracker(camera, Detector())
    publish(camera, 10); tracker._tick()
    publish(camera, 11)
    image, state = tracker.preview()
    assert image.frame_id == state["frame"]["frame_id"] == 10
    assert state["face_box"] == (40, 40, 60, 60)
    camera.source_epoch = "new"
    assert tracker.preview()[0] is None


def test_bad_artifact_reload_does_not_leave_detector_ready(monkeypatch):
    import app.face_detector as module
    detector = module.FaceDetector()
    detector._detector = object()
    def bad(): raise ValueError("hash mismatch")
    monkeypatch.setattr(module, "_resolve_model_path", bad)
    assert not detector.load() and not detector.health_check()
