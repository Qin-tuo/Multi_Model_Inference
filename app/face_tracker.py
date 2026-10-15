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

"""YuNet detection state from the upstream tracker, with all motor logic removed.

Tracks anonymous face presence/position only. Consecutive *distinct* frames
establish framing stability; this is not identity tracking or authentication.
"""
import threading
import time
from app.camera import Camera
from app.face_detector import FaceDetector


class FaceTracker:
    def __init__(self, camera: Camera, detector: FaceDetector, fps=5.0,
                 dead_zone=0.12, good_frame_zone=0.18, min_face_size=0.06,
                 stable_frames=2):
        self._camera, self._detector = camera, detector
        self._period = 1.0 / fps
        self._dead_zone, self._good_frame_zone = dead_zone, good_frame_zone
        self._min_face_size, self._stable_frames_required = min_face_size, stable_frames
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._thread = None
        self._stable_count = 0
        self._last_frame = None
        self._state = self._empty()
        self._preview = None

    @staticmethod
    def _empty(error=None):
        return {"face_detected": False, "face_box": None, "centered": False,
                "stable": False, "error": error, "frame": None, "completed_monotonic": None}

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        if not self._detector.health_check():
            raise RuntimeError("Face detector is not ready")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="face-tracker", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("Face inference has not released")
        self._thread = None
        self._reset()

    def _reset(self, error=None):
        with self._state_lock:
            self._state = self._empty(error)
            self._stable_count = 0
            self._last_frame = None
            self._preview = None

    def preview(self):
        with self._state_lock:
            frame, state = self._preview, dict(self._state)
        if frame is None or frame.source_epoch != self._camera.source_epoch or time.monotonic() - frame.received_monotonic > self._camera.stale_after:
            return None, self._empty("Preview expired")
        return frame, state

    def snapshot(self):
        with self._state_lock:
            result = dict(self._state)
        frame = result["frame"]
        if frame and (frame["source_epoch"] != self._camera.source_epoch or
                      time.monotonic() - frame["received_monotonic"] > self._camera.stale_after):
            return self._empty("Face result expired")
        return result

    @property
    def face_detected(self): return self.snapshot()["face_detected"]
    @property
    def centered(self): return self.snapshot()["centered"]
    @property
    def stable(self): return self.snapshot()["stable"]
    @property
    def last_face_box(self): return self.snapshot()["face_box"]
    @property
    def is_tracking(self): return self.face_detected

    def wait_until_stable(self, timeout, poll_interval=0.03):
        deadline = time.monotonic() + max(0, timeout)
        while time.monotonic() < deadline and not self._stop.is_set():
            if self.stable: return True
            self._stop.wait(poll_interval)
        return self.stable

    def _loop(self):
        while not self._stop.is_set():
            start = time.monotonic()
            try:
                self._tick()
            except Exception as exc:
                self._reset(str(exc))
            self._stop.wait(max(0, self._period - (time.monotonic() - start)))

    def _tick(self):
        frame = self._camera.snapshot()
        if frame is None:
            self._reset("No fresh camera frame")
            return
        key = (frame.source_epoch, frame.frame_id)
        if key == self._last_frame:
            return
        box = self._detector.detect(frame.bgr)
        with self._state_lock:
            if self._last_frame and self._last_frame[0] != frame.source_epoch:
                self._stable_count = 0
            self._last_frame = key
            state = self._empty()
            state.update(frame=frame.metadata(), completed_monotonic=time.monotonic())
            if box is not None:
                # Preserve upstream normalized framing and minimum-size rules.
                h, w = frame.bgr.shape[:2]
                err_x = ((box[0] + box[2]) / 2 - w / 2) / (w / 2)
                err_y = ((box[1] + box[3]) / 2 - h / 2) / (h / 2)
                size = max((box[2] - box[0]) / w, (box[3] - box[1]) / h)
                good = size >= self._min_face_size and max(abs(err_x), abs(err_y)) < self._good_frame_zone
                self._stable_count = self._stable_count + 1 if good else 0
                state.update(face_detected=True, face_box=box,
                             centered=good or max(abs(err_x), abs(err_y)) < self._dead_zone,
                             stable=self._stable_count >= self._stable_frames_required)
            else:
                self._stable_count = 0
            self._state = state
            self._preview = frame
