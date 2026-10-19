# SPDX-License-Identifier: Apache-2.0
"""Native depth must stay attached to fresh RGB and preserve invalid pixels."""

from types import SimpleNamespace
import time

import numpy as np

from app.camera import Camera
from app.realsense_source import SensorFrame, correspondence


def group(index, matched=True):
    streams = {
        "color": {
            "frame_number": index + 100,
            "timestamp_ms": index * 100,
            "domain": "timestamp_domain.global_time",
        },
        "sdk_depth": {
            "frame_number": index,
            "timestamp_ms": index * 100 + (0 if matched else 100),
            "domain": "timestamp_domain.global_time",
        },
    }
    return SensorFrame(
        "source",
        index,
        time.monotonic(),
        {
            "color": np.full((2, 3, 3), [12, 34, 56], np.uint8),
            "aligned_depth": np.array([[1000, 0, 2000], [3000, 4000, 5000]], np.uint16),
        },
        streams,
        correspondence(streams, 33),
    )


def test_rgbd_rejects_wrong_time_and_repeats_then_preserves_units_and_rgb_order():
    camera = Camera(backend="realsense")
    values = iter([group(1, False), group(2), group(2), group(3)])
    camera._source = SimpleNamespace(
        read=lambda **_: next(values), calibration={"depth_units_m": 0.001}
    )
    ok, bgr, (depth, valid, metadata) = camera._read_frame()
    assert ok and camera._rgbd_rejected == 1
    assert bgr[0, 0].tolist() == [56, 34, 12]
    assert depth[0, 0] == 1 and depth[0, 2] == 2
    assert np.isnan(depth[0, 1]) and not valid[0, 1]
    assert not depth.flags.writeable and not valid.flags.writeable
    assert metadata["group_id"] == 2
    assert camera._read_frame()[2][2]["group_id"] == 3
    assert camera._rgbd_rejected == 2


def test_source_invalidation_clears_depth_with_rgb():
    camera = Camera(backend="realsense")
    camera._latest_depth = np.ones((2, 3), np.float32)
    camera._latest_depth_valid = np.ones((2, 3), bool)
    camera._latest_sensor = {"group_id": 1}
    old = camera._source_invalidated
    camera._invalidate("retrying")
    assert old.is_set()
    assert (
        camera._latest_depth
        is camera._latest_depth_valid
        is camera._latest_sensor
        is None
    )
    assert not camera.status()["depth_ready"]
