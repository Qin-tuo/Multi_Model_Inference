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

"""Passive face framing state; the shared visual pipeline owns all inference."""

import threading
import time
from app.camera import Camera


class FaceState:
    def __init__(
        self,
        camera: Camera,
        *,
        dead_zone=0.12,
        good_frame_zone=0.18,
        min_face_size=0.06,
        stable_frames=2,
    ):
        self._camera = camera
        self._dead_zone, self._good_frame_zone = dead_zone, good_frame_zone
        self._min_face_size, self._stable_frames_required = min_face_size, stable_frames
        self._state_lock = threading.Lock()
        self._stable_count = 0
        self._last_frame = None
        self._state = self._empty()
        self._preview = None

    def stop(self):
        self._reset()

    @staticmethod
    def _empty(error=None):
        return {
            "face_detected": False,
            "face_box": None,
            "centered": False,
            "stable": False,
            "error": error,
            "frame": None,
            "completed_monotonic": None,
        }

    def _reset(self, error=None):
        with self._state_lock:
            self._state = self._empty(error)
            self._stable_count = 0
            self._last_frame = None
            self._preview = None

    def preview(self):
        with self._state_lock:
            frame, state = self._preview, dict(self._state)
        if (
            frame is None
            or frame.source_epoch != self._camera.source_epoch
            or time.monotonic() - frame.received_monotonic > self._camera.stale_after
        ):
            return None, self._empty("Preview expired")
        return frame, state

    def snapshot(self):
        with self._state_lock:
            result = dict(self._state)
        frame = result["frame"]
        if frame and (
            frame["source_epoch"] != self._camera.source_epoch
            or time.monotonic() - frame["received_monotonic"] > self._camera.stale_after
        ):
            return self._empty("Face result expired")
        return result

    @property
    def face_detected(self):
        return self.snapshot()["face_detected"]

    @property
    def centered(self):
        return self.snapshot()["centered"]

    @property
    def stable(self):
        return self.snapshot()["stable"]

    @property
    def last_face_box(self):
        return self.snapshot()["face_box"]

    @property
    def is_tracking(self):
        return self.face_detected

    def process_box(self, frame, box):
        """Accept a same-frame box from the shared vision worker without re-inference."""
        if (
            frame is None
            or frame.source_epoch != self._camera.source_epoch
            or (frame.invalidated and frame.invalidated.is_set())
            or time.monotonic() - frame.received_monotonic > self._camera.stale_after
        ):
            self._reset("No fresh face observation")
            return
        key = (frame.source_epoch, frame.frame_id)
        with self._state_lock:
            if key == self._last_frame:
                return
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
                good = (
                    size >= self._min_face_size
                    and max(abs(err_x), abs(err_y)) < self._good_frame_zone
                )
                self._stable_count = self._stable_count + 1 if good else 0
                state.update(
                    face_detected=True,
                    face_box=box,
                    centered=good or max(abs(err_x), abs(err_y)) < self._dead_zone,
                    stable=self._stable_count >= self._stable_frames_required,
                )
            else:
                self._stable_count = 0
            self._state = state
            self._preview = frame
