# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Face detection — YuNet (MIT, OpenCV Zoo) via cv2.FaceDetectorYN.

A single-purpose detector used to locate the most prominent face in a
camera frame. Detection is independent of a robot body.
No identity, emotion, or expression classification.
"""

import hashlib
import json
import threading
from pathlib import Path
from project_paths import MODEL_DIR, load_artifact_manifest
from typing import Optional

import cv2
import numpy as np

_MANIFEST = MODEL_DIR / "perception.json"

_SCORE_THRESH = 0.5
_NMS_THRESH = 0.3
_TOP_K = 5000
_MIN_FACE_PX = 10

# YuNet on full 1080p frames is slow on CPU and adds control-loop lag.
# Detect on a downscaled copy and scale the box back to original coords.
_DETECT_MAX_W = 640

FaceBox = tuple[int, int, int, int]


def _resolve_model_path() -> Path:
    """Use a prepared, pinned local artifact; runtime never downloads weights."""
    entry = load_artifact_manifest(_MANIFEST)["face"]
    path = Path(entry["directory"]).expanduser() / entry["filename"]
    expected = entry["files"][entry["filename"]]["sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("YuNet artifact hash mismatch")
    return path


def _pick_best_face(faces: Optional[np.ndarray]) -> Optional[FaceBox]:
    """Return the highest-confidence face as (x1, y1, x2, y2), or None.

    YuNet returns an Nx15 array per face where columns 0..3 are the
    bounding box (x, y, w, h) and the last column is the score.
    """
    if faces is None or len(faces) == 0:
        return None
    best = faces[faces[:, -1].argmax()]
    x, y, w, h = (int(best[0]), int(best[1]), int(best[2]), int(best[3]))
    if w < _MIN_FACE_PX or h < _MIN_FACE_PX:
        return None
    return (x, y, x + w, y + h)


class FaceDetector:
    """Thread-safe YuNet face detector (CPU, via OpenCV)."""

    def __init__(self):
        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._lock = threading.Lock()
        self.last_error = None

    def load(self) -> bool:
        self.last_error = None
        self.unload()
        try:
            model_path = _resolve_model_path()
            self._detector = cv2.FaceDetectorYN.create(
                str(model_path),
                "",
                (320, 320),
                score_threshold=_SCORE_THRESH,
                nms_threshold=_NMS_THRESH,
                top_k=_TOP_K,
            )
            return True
        except Exception as e:
            self.last_error = str(e)
            print(f"FaceDetector: load error — {e}")
            return False

    @property
    def backend(self) -> str:
        return "YuNet (OpenCV CPU)"

    def detect(self, frame: np.ndarray) -> Optional[FaceBox]:
        """Detect the most prominent face in a raw BGR frame.

        Returns (x1, y1, x2, y2) in the original frame's pixel coordinates,
        or None. Detection runs on a downscaled copy for speed. Thread-safe.
        """
        if frame is None:
            return None
        h, w = frame.shape[:2]

        scale = _DETECT_MAX_W / w if w > _DETECT_MAX_W else 1.0
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame
        sh, sw = small.shape[:2]

        with self._lock:
            if self._detector is None:
                return None
            self._detector.setInputSize((sw, sh))
            _, faces = self._detector.detect(small)

        box = _pick_best_face(faces)
        if box is None or scale >= 1.0:
            return box
        inv = 1.0 / scale
        return (
            int(box[0] * inv),
            int(box[1] * inv),
            int(box[2] * inv),
            int(box[3] * inv),
        )

    def health_check(self) -> bool:
        return self._detector is not None

    def unload(self) -> None:
        with self._lock:
            self._detector = None
