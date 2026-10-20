# SPDX-License-Identifier: Apache-2.0
"""Controlled RGBD input must still use production publication and invalidation."""

import time

import numpy as np

from app.config import Config
from benchmarks.vision_stream import MeasuredCamera


def test_replay_preserves_camera_ownership_jpeg_and_explicit_synthetic_depth(tmp_path):
    image = np.zeros((480, 640, 3), np.uint8)
    image[:, :, 1] = 100
    path = tmp_path / "input.npy"
    np.save(path, image, allow_pickle=False)
    cfg = Config.load("02_config/settings.orin-nx.yaml")
    protocol = {"inputs": {"fixed": {"path": str(path)}}}
    spec = {"replay_inputs": ["fixed"], "hold_source_frames": 20}

    class Events:
        def __init__(self):
            self.rows = []

        def write(self, **value):
            self.rows.append(value)

    events = Events()
    camera = MeasuredCamera(cfg.vision, protocol, spec, events, lambda: "test", False)
    try:
        assert camera.start()
        first = camera.snapshot()
        assert first is not None and first.jpeg[:2] == b"\xff\xd8"
        assert not first.bgr.flags.writeable and not first.depth_m.flags.writeable
        assert np.all(first.depth_m == 1.5) and first.depth_valid.all()
        assert first.sensor["depth_provider"].startswith("SYNTHETIC")
        assert (
            first.received_monotonic
            <= first.sensor["benchmark_jpeg_complete_monotonic"]
        )
        assert camera.status()["depth_ready"]
        deadline = time.monotonic() + 2
        while (
            camera.snapshot().frame_id == first.frame_id and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert camera.snapshot().frame_id > first.frame_id
        assert camera.snapshot().source_epoch == first.source_epoch
        assert events.rows and all(r["kind"] == "jpeg" for r in events.rows)
    finally:
        camera.close()
    assert first.invalidated.is_set()
    assert camera.snapshot() is None and not camera.status()["capture_worker_alive"]
