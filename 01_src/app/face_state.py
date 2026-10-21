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

"""消费共享视觉管线给出的人脸框，维护居中与稳定状态，不重复执行人脸推理。"""

import threading
import time
from app.camera import Camera


class FaceState:
    """基于已检测人脸框维护居中、稳定和预览状态，不拥有额外推理线程。"""

    def __init__(
        self,
        camera: Camera,
        *,
        dead_zone=0.12,
        good_frame_zone=0.18,
        min_face_size=0.06,
        stable_frames=2,
    ):
        """绑定相机并设置居中区域、最小脸尺寸和连续稳定帧要求。"""
        self._camera = camera
        self._dead_zone, self._good_frame_zone = dead_zone, good_frame_zone
        self._min_face_size, self._stable_frames_required = min_face_size, stable_frames
        self._state_lock = threading.Lock()
        self._stable_count = 0
        self._last_frame = None
        self._state = self._empty()
        self._preview = None

    def stop(self):
        """清空当前人脸观察和稳定计数，停止对旧结果提供可用状态。"""
        self._reset()

    @staticmethod
    def _empty(error=None):
        """构造无人脸、未稳定的空状态，必要时附带不可用原因。"""
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
        """在锁内清除帧身份、预览和稳定证据。"""
        with self._state_lock:
            self._state = self._empty(error)
            self._stable_count = 0
            self._last_frame = None
            self._preview = None

    def preview(self):
        """返回同一观察的人脸状态与源帧；帧已过期或来自旧连接时返回空值。"""
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
        """取得人脸状态副本，并在读取时检查来源批次和帧龄。"""
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
        """返回当前新鲜观察是否存在人脸。"""
        return self.snapshot()["face_detected"]

    @property
    def centered(self):
        """返回当前人脸是否满足归一化居中规则。"""
        return self.snapshot()["centered"]

    @property
    def stable(self):
        """返回当前人脸是否已连续满足足够帧数的稳定要求。"""
        return self.snapshot()["stable"]

    @property
    def last_face_box(self):
        """返回当前有效人脸框；观察无效时不沿用旧框。"""
        return self.snapshot()["face_box"]

    @property
    def is_tracking(self):
        """以当前人脸是否可见表示跟踪状态，不额外启动跟踪线程。"""
        return self.face_detected

    def process_box(self, frame, box):
        """接收共享管线的同帧人脸框，更新归一化位置和连续稳定计数。

        重复帧不重复累积证据；来源变化、过期或显式失效会清除旧状态。
        """
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
