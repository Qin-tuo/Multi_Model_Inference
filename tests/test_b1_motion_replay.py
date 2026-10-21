# SPDX-License-Identifier: Apache-2.0
"""A recorded functional comparison must not silently pick different frames."""

import json
import time

import numpy as np

from app.config import Config
from benchmarks.motion import RecordedMovieCamera


def test_recorded_camera_waits_for_consumer_and_keeps_selected_media_frames(tmp_path):
    cfg = Config.load("02_config/settings.orin-nx.yaml")
    log = tmp_path / "events.jsonl"
    rows = []
    for i, source in enumerate((0, 2)):
        rows.append(
            {
                "kind": "scene",
                "frame": {
                    "source_epoch": "original",
                    "frame_id": 10 + i,
                    "received_monotonic": 1000 + i * 0.1,
                    "sensor": {
                        "source": "fixture",
                        "media_frame_index": source,
                        "media_timestamp_seconds": source * 0.1,
                        "depth_provider": "SYNTHETIC",
                    },
                },
            }
        )
    log.write_text("\n".join(json.dumps(r) for r in rows))
    frames = [(i * 0.1, i, np.full((480, 640, 3), i, np.uint8)) for i in range(3)]
    camera = RecordedMovieCamera(cfg.vision, frames, 0.3, "fixture", log)
    try:
        assert camera.start()
        first = camera.snapshot()
        time.sleep(0.15)
        assert camera.snapshot().frame_id == first.frame_id
        camera.mark_processed(first.frame_id)
        deadline = time.monotonic() + 2
        while (
            camera.snapshot().frame_id == first.frame_id and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        second = camera.snapshot()
        assert second.frame_id != first.frame_id
        assert second.sensor["media_frame_index"] == 2
        assert np.all(second.bgr == 2)
        assert second.sensor["source_kind"] == "recorded_selection_with_backpressure"
        assert second.sensor["delivery_lag_ms"] >= 0
        camera.mark_processed(second.frame_id)
        assert camera.ended.wait(1)
    finally:
        camera.close()
    assert not camera.status()["capture_worker_alive"]
