# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import numpy as np
from app.camera import Camera
from app.face_state import FaceState


def publish(camera, frame_id, timestamp=None):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    camera._latest = (timestamp or time.monotonic(), image)
    camera._latest_jpeg = camera._encode_frame_bytes(image)
    camera._frame_id = frame_id
    camera._alive = True


def test_stability_requires_distinct_frames_and_resets_after_loss():
    camera = Camera()
    tracker = FaceState(camera, stable_frames=2)
    publish(camera, 1)
    tracker.process_box(camera.snapshot(), (40, 40, 60, 60))
    tracker.process_box(camera.snapshot(), (40, 40, 60, 60))
    assert tracker.face_detected and not tracker.stable
    publish(camera, 2)
    tracker.process_box(camera.snapshot(), (40, 40, 60, 60))
    assert tracker.stable
    publish(camera, 3)
    tracker.process_box(camera.snapshot(), None)
    assert (
        not tracker.face_detected
        and not tracker.stable
        and tracker.last_face_box is None
    )


def test_camera_restart_resets_stability_and_old_result():
    camera = Camera()
    tracker = FaceState(camera, stable_frames=2)
    publish(camera, 1)
    tracker.process_box(camera.snapshot(), (40, 40, 60, 60))
    publish(camera, 2)
    tracker.process_box(camera.snapshot(), (40, 40, 60, 60))
    assert tracker.stable
    camera.source_epoch = "restarted"
    assert not tracker.face_detected
    publish(camera, 1)
    tracker.process_box(camera.snapshot(), (40, 40, 60, 60))
    assert tracker.face_detected and not tracker.stable


def test_expired_face_is_not_reported_as_live():
    camera = Camera(stale_after=0.01)
    tracker = FaceState(camera)
    publish(camera, 1)
    tracker.process_box(camera.snapshot(), (40, 40, 60, 60))
    time.sleep(0.02)
    assert not tracker.face_detected
    tracker.process_box(camera.snapshot(), (40, 40, 60, 60))
    assert tracker.snapshot()["frame"] is None


def test_off_center_face_does_not_require_a_motor_controller():
    camera = Camera()
    tracker = FaceState(camera)
    publish(camera, 1)
    tracker.process_box(camera.snapshot(), (5, 5, 25, 25))
    assert tracker.face_detected and not tracker.centered
    assert tracker.snapshot()["frame"]["frame_id"] == 1


def test_preview_pairs_pixels_and_box_from_same_frame():
    camera = Camera()
    tracker = FaceState(camera)
    publish(camera, 10)
    tracker.process_box(camera.snapshot(), (40, 40, 60, 60))
    publish(camera, 11)
    image, state = tracker.preview()
    assert image.frame_id == state["frame"]["frame_id"] == 10
    assert state["face_box"] == (40, 40, 60, 60)
    camera.source_epoch = "new"
    assert tracker.preview()[0] is None
