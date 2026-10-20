# SPDX-License-Identifier: Apache-2.0
"""Preview evidence must retain frame ownership without overwriting or growing."""

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest

from probe_gestures import PreviewCapture


def frame_event(index=1, epoch="camera-a"):
    ok, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))
    assert ok
    return {
        "type": "frame",
        "frame": {"source_epoch": epoch, "frame_id": index},
        "data": base64.b64encode(encoded).decode("ascii"),
    }


def test_preview_retains_bytes_and_epoch_frame_identity(tmp_path):
    recorder = PreviewCapture(tmp_path / "observation.json")
    first = frame_event()
    recorder.write(first)
    recorder.write(first)
    recorder.write(frame_event(epoch="camera-b"))
    rows = recorder.report["frames"]
    assert len(rows) == 2
    assert rows[0]["frame"] == first["frame"]
    assert Path(rows[0]["path"]).read_bytes() == base64.b64decode(first["data"])
    assert rows[1]["frame"]["source_epoch"] == "camera-b"
    with pytest.raises(FileExistsError):
        PreviewCapture(tmp_path / "observation.json")


def test_preview_budget_and_invalid_payload_never_create_partial_images(tmp_path):
    event = frame_event()
    size = len(base64.b64decode(event["data"]))
    recorder = PreviewCapture(tmp_path / "observation.json", max_bytes=size)
    recorder.write(event)
    recorder.write(frame_event(index=2))
    recorder.write({**frame_event(index=3), "data": "not base64"})
    recorder.write({**frame_event(index=4), "data": base64.b64encode(b"text").decode()})
    recorder.write({**frame_event(index=5), "frame": {"source_epoch": "camera-a"}})
    assert recorder.report["limit_rejections"] == 1
    assert recorder.report["invalid_frames"] == 3
    assert recorder.report["bytes"] == size
    assert len(list(recorder.directory.iterdir())) == 1
