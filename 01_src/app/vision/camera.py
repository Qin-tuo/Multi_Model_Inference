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

"""相机唯一采集者与共享 RGBD 快照。

最新原始图像供推理使用，语音回看窗口只保存有字节预算的 JPEG；消费者不竞争底层设备读取。
"""

import base64
import threading
import time
import uuid
import os
from collections import deque
from typing import Optional

import cv2
import numpy as np

from app.contracts import FrameSnapshot

MAX_SPEECH_SECS = 16
PRE_SPEECH_SECS = 0.5


class Camera:
    """统一管理 V4L2 或 RealSense 的采集、重连、帧失效和有界图像缓存。"""

    _backend_config_lock = threading.Lock()
    _configured_select_timeout = None

    def __init__(
        self,
        device: int | str = 0,
        width: int = 640,
        height: int = 480,
        jpeg_quality: int = 80,
        capture_fps: float = 3.0,
        stale_after: float = 1.0,
        buffer_bytes_limit: int = 16 * 1024 * 1024,
        reconnect_seconds: float = 1.0,
        v4l_select_timeout: int = 2,
        backend: str = "v4l2",
        serial: str | None = None,
    ):
        """保存采集模式、帧龄和缓存预算，初始化来源批次及线程状态；此时不打开设备。"""
        if min(width, height, capture_fps, stale_after, buffer_bytes_limit) <= 0:
            raise ValueError("Camera sizes, rates and limits must be positive")
        self.stale_after = stale_after
        if backend not in {"v4l2", "realsense"}:
            raise ValueError("Unknown camera backend")
        self.backend, self.serial = backend, serial
        self._source = None
        self._latest_depth = self._latest_depth_valid = self._latest_sensor = None
        self._last_rgbd_counters = None
        self._rgbd_rejected = 0
        self.buffer_bytes_limit = buffer_bytes_limit
        if (
            not 0.1 <= reconnect_seconds <= 10
            or type(v4l_select_timeout) is not int
            or not 1 <= v4l_select_timeout <= 3
        ):
            raise ValueError("Invalid camera recovery timing")
        self.reconnect_seconds, self.v4l_select_timeout = (
            reconnect_seconds,
            v4l_select_timeout,
        )
        self.source_epoch = uuid.uuid4().hex
        self._source_invalidated = threading.Event()
        self._frame_id = 0
        self.last_error = None
        self._stop = threading.Event()
        self.device = device
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality
        self.capture_fps = capture_fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._lifecycle_lock = threading.RLock()
        self._requested_size = (width, height)
        self._state = "stopped"
        self._open_attempts = self._failures = self._reconnects = 0
        self._ever_streamed = False
        # 相机可能返回比请求更大的图像；以原始帧保存长回看窗口会占用大量板载内存。
        # 因此历史只存 JPEG，原始 RGBD 仅保留最新一帧，并同时限制帧数和字节数。
        self._ring: deque[tuple[float, bytes]] = deque(
            maxlen=max(1, int(capture_fps * (MAX_SPEECH_SECS + PRE_SPEECH_SECS)))
        )
        self._latest: Optional[tuple[float, np.ndarray]] = None
        self._latest_jpeg: Optional[bytes] = None
        self._lock = threading.Lock()
        self._alive = False
        self._thread: Optional[threading.Thread] = None
        self._actual_fps: float = 0.0

    def _open(self) -> bool:
        """按显式后端打开设备并检查资源状态；RealSense 打开失败时回收其半初始化对象。"""
        if self.backend == "realsense":
            if self._source is None:
                from app.vision.realsense_source import RealSenseSource

                self._source = RealSenseSource(
                    self.width, self.height, serial=self.serial
                )
                try:
                    self._source.open()
                    self._last_rgbd_counters = None
                except BaseException:
                    self._source.close()
                    self._source = None
                    raise
            return True
        if self._cap is not None and self._cap.isOpened():
            return True

        # Explicit V4L2 device only: no SDK autodetection or process cleanup.
        self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            self.last_error = (
                f"Cannot open V4L2 camera {self.device}; check device and ownership"
            )
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_size[1])
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._sync_actual_frame_geometry()
        return True

    def _read_frame(self):
        """读取一帧 BGR 或经过时间匹配的 RGBD，返回成功标记、图像和可选深度信息。

        RealSense 拒绝重复计数和失配帧，将深度单位转换为米，无效值设为 NaN。
        """
        if self.backend == "v4l2":
            ok, frame = self._cap.read()
            return ok, frame, None
        deadline = time.monotonic() + 4
        while not self._stop.is_set() and time.monotonic() < deadline:
            group = self._source.read(timeout_ms=1000)
            matched = group.correspondence["rgb_depth_time_matched"]
            if not matched or "aligned_depth" not in group.arrays:
                self._rgbd_rejected += 1
                continue
            counters = tuple(
                group.streams[name]["frame_number"] for name in ("color", "sdk_depth")
            )
            if self._last_rgbd_counters and any(
                a <= b for a, b in zip(counters, self._last_rgbd_counters)
            ):
                self._rgbd_rejected += 1
                continue
            color, raw_depth = group.arrays["color"], group.arrays["aligned_depth"]
            if raw_depth.shape != color.shape[:2] or raw_depth.dtype != np.uint16:
                raise ValueError("Aligned depth geometry or dtype mismatch")
            scale = self._source.calibration["depth_units_m"]
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("Invalid SDK depth units")
            depth = raw_depth.astype(np.float32) * scale
            valid = (raw_depth > 0) & np.isfinite(depth)
            depth[~valid] = np.nan
            depth.flags.writeable = valid.flags.writeable = False
            self._last_rgbd_counters = counters
            metadata = group.metadata()
            metadata.update(
                depth_provider="D435i native depth / RealSense SDK",
                alignment="rs.align(stream.color)",
                depth_unit="meter",
                depth_valid_fraction=float(valid.mean()),
                calibration=self._source.calibration,
            )
            return (
                True,
                cv2.cvtColor(color, cv2.COLOR_RGB2BGR),
                (depth, valid, metadata),
            )
        return False, None, None

    def _release_capture(self):
        """关闭当前 RealSense/V4L2 句柄并清除引用，由采集生命周期的所有者调用。"""
        if self._source is not None:
            self._source.close()
            self._source = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _sync_actual_frame_geometry(self) -> None:
        """读取 V4L2 实际接受的宽高，避免把请求尺寸误当作真实采集尺寸。"""
        if self._cap is None:
            return
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width > 0:
            self.width = actual_width
        if actual_height > 0:
            self.height = actual_height

    def _invalidate(self, state, error=None):
        """使当前来源的所有快照失效，更新批次标识并清空 RGBD/JPEG 缓存。"""
        with self._lock:
            self._source_invalidated.set()
            self._source_invalidated = threading.Event()
            self.source_epoch = uuid.uuid4().hex
            self._frame_id = 0
            self._ring.clear()
            self._latest = self._latest_jpeg = None
            self._latest_depth = self._latest_depth_valid = self._latest_sensor = None
            self._actual_fps = 0
            self._state = state
            if error is not None:
                self.last_error = error

    def start(self) -> bool:
        """启动唯一采集线程，并在限定时间内等待首个新鲜帧。

        统一设置进程内 V4L2 等待时限；旧句柄未释放或启动超时会拒绝进入可用状态。
        """
        with self._lifecycle_lock:
            if self._thread and self._thread.is_alive():
                return self.health_check()
            if self._cap is not None or self._source is not None:
                raise RuntimeError(
                    "Previous camera release must complete before restart"
                )
            # OpenCV 4.11 reads this once in its V4L2 ioctl/select path. The app
            # has one capture owner; changing this setting requires a new process.
            with Camera._backend_config_lock:
                if Camera._configured_select_timeout not in (
                    None,
                    self.v4l_select_timeout,
                ):
                    raise RuntimeError(
                        "V4L2 timeout change requires an application restart"
                    )
                os.environ["OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT"] = str(
                    self.v4l_select_timeout
                )
                Camera._configured_select_timeout = self.v4l_select_timeout
            self.last_error = None
            self._invalidate("opening")
            self._stop.clear()
            self._alive = True
            self._thread = threading.Thread(
                target=self._capture_loop, name="camera-capture", daemon=True
            )
            self._thread.start()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and self._alive:
                if self.snapshot() is not None:
                    return True
                self._stop.wait(0.02)
            self.last_error = (
                self.last_error or "Camera produced no fresh frame during startup"
            )
            self.close()
            return False

    def _capture_loop(self):
        """由同一线程完成打开、读取、JPEG 缓存、失败重连和设备释放。

        原始数组只发布最新帧，JPEG 历史同时受帧数和字节预算约束；重连会更换来源批次。
        """
        interval = 1.0 / self.capture_fps
        try:
            while not self._stop.is_set():
                try:
                    self._open_attempts += 1
                    if not self._open():
                        raise RuntimeError(self.last_error)
                    self._invalidate("awaiting_frame")
                    n_frames = 0
                    t_start = time.monotonic()
                    while not self._stop.is_set():
                        tick = time.monotonic()
                        ret, frame, rgbd = self._read_frame()
                        if self._stop.is_set():
                            break
                        if not ret:
                            raise RuntimeError("Camera produced no matched fresh frame")
                        received = (
                            rgbd[2]["received_monotonic"] if rgbd else time.monotonic()
                        )
                        jpg = self._encode_frame_bytes(frame)
                        if jpg is None or len(jpg) > self.buffer_bytes_limit:
                            raise RuntimeError(
                                "Camera JPEG invalid or exceeds buffer budget"
                            )
                        frame.flags.writeable = False
                        with self._lock:
                            if self._stop.is_set():
                                break
                            if n_frames == 0:
                                if self._ever_streamed:
                                    self._reconnects += 1
                                self._ever_streamed = True
                            self._frame_id += 1
                            self._latest = (received, frame)
                            self._latest_jpeg = jpg
                            if rgbd:
                                (
                                    self._latest_depth,
                                    self._latest_depth_valid,
                                    self._latest_sensor,
                                ) = rgbd
                            self._ring.append((received, jpg))
                            while (
                                sum(len(data) for _, data in self._ring)
                                > self.buffer_bytes_limit
                            ):
                                self._ring.popleft()
                            self._state = "streaming"
                            self.last_error = None
                        n_frames += 1
                        self._actual_fps = n_frames / max(
                            time.monotonic() - t_start, 1e-6
                        )
                        self._stop.wait(max(0, interval - (time.monotonic() - tick)))
                except Exception as exc:
                    self._failures += 1
                    self._invalidate("retrying", str(exc))
                finally:
                    # 仅采集所有者在 read 返回后释放设备，避免关闭与读取相互竞争。
                    self._release_capture()
                if self._stop.wait(self.reconnect_seconds):
                    break
        except Exception as exc:
            self.last_error = str(exc)
        finally:
            self._alive = False
            self._invalidate("stopped" if self._stop.is_set() else "failed")

    def snapshot(self, *, copy: bool = False) -> Optional[FrameSnapshot]:
        """在锁内取得仍新鲜的 RGBD 快照；copy=True 时复制图像和深度数组，否则共享只读数据。"""
        with self._lock:
            if self._stop.is_set() or self._latest is None or self._latest_jpeg is None:
                return None
            timestamp, frame = self._latest
            if time.monotonic() - timestamp > self.stale_after:
                return None
            return FrameSnapshot(
                self._frame_id,
                self.source_epoch,
                timestamp,
                frame.copy() if copy else frame,
                self._latest_jpeg,
                invalidated=self._source_invalidated,
                depth_m=self._latest_depth.copy()
                if copy and self._latest_depth is not None
                else self._latest_depth,
                depth_valid=self._latest_depth_valid.copy()
                if copy and self._latest_depth_valid is not None
                else self._latest_depth_valid,
                sensor=self._latest_sensor,
            )

    def _encode_frame_bytes(self, frame: np.ndarray) -> Optional[bytes]:
        """按配置质量编码 JPEG 字节，编码失败返回 None。"""
        ok, jpg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            return None
        return jpg.tobytes()

    def _encode_frame(self, frame: np.ndarray) -> Optional[str]:
        """把 BGR 图像编码成 JPEG 的 Base64 字符串，供需要文本图像载荷的接口使用。"""
        jpg = self._encode_frame_bytes(frame)
        if jpg is None:
            return None
        return base64.b64encode(jpg).decode("ascii")

    @staticmethod
    def _base64_jpeg(jpg: bytes) -> str:
        """将已编码的 JPEG 字节转成 ASCII Base64 文本。"""
        return base64.b64encode(jpg).decode("ascii")

    def get_speech_frames(
        self,
        speech_start: float,
        speech_end: float,
        max_frames: int = 3,
    ) -> list[str]:
        """从指定语音时间窗口选取 JPEG，并返回 Base64 列表。

        只取一帧时选窗口内最新图像；多帧模式包含预语音回看区间并抽样，来源失效或窗口无图像时返回空列表。
        """
        if max_frames > 1:
            window_start = speech_start - PRE_SPEECH_SECS
        else:
            window_start = speech_start

        with self._lock:
            invalidated = self._source_invalidated
            candidates = [
                (t, jpg) for t, jpg in self._ring if window_start <= t <= speech_end
            ]

        # 所需时间窗口没有图像时返回空，不借用另一窗口或重连前的旧图。
        if not self.health_check() or not candidates:
            return []

        if max_frames == 1:
            selected = [candidates[-1][1]]
        elif len(candidates) <= max_frames:
            selected = [jpg for _, jpg in candidates]
        else:
            step = len(candidates) / max_frames
            selected = [candidates[int(i * step)][1] for i in range(max_frames)]

        encoded = [self._base64_jpeg(jpg) for jpg in selected]
        return [] if invalidated.is_set() else encoded

    def capture_single(self) -> Optional[str]:
        """返回当前新鲜帧的 Base64 JPEG，没有可用帧则返回 None。"""
        frame = self.snapshot()
        return self._base64_jpeg(frame.jpeg) if frame else None

    def latest_raw(self, *, copy: bool = True) -> Optional[np.ndarray]:
        """返回最新 BGR 数组，默认复制以免调用方修改共享帧。"""
        frame = self.snapshot(copy=copy)
        return frame.bgr if frame else None

    def read_live(self) -> Optional[str]:
        """为实时预览提供最新 Base64 JPEG，复用同一个采集缓存。"""
        return self.capture_single()

    def read_raw_live(self) -> Optional[np.ndarray]:
        """为原始图像消费者返回最新帧副本，不直接读取底层设备。"""
        return self.latest_raw()

    @property
    def buffer_count(self) -> int:
        """在锁内返回语音回看 JPEG 环形缓冲区的当前帧数。"""
        with self._lock:
            return len(self._ring)

    @property
    def actual_fps(self) -> float:
        """返回当前连接周期内依据真实采集计数计算的平均 FPS。"""
        return self._actual_fps

    @property
    def buffer_bytes(self) -> int:
        """统计回看缓冲区实际占用的 JPEG 字节数。"""
        with self._lock:
            return sum(len(data) for _, data in self._ring)

    def health_check(self) -> bool:
        """仅在采集所有者存活且能够取得新鲜帧时报告健康。"""
        return self._alive and self.snapshot() is not None

    def status(self):
        """汇总当前采集状态、帧龄、重连次数、缓存用量和深度可用性。"""
        with self._lock:
            age = time.monotonic() - self._latest[0] if self._latest else None
            ready = (
                self._alive
                and not self._stop.is_set()
                and age is not None
                and age <= self.stale_after
            )
            return {
                "ready": ready,
                "state": "stale"
                if self._state == "streaming" and not ready
                else self._state,
                "error": self.last_error,
                "source_epoch": self.source_epoch,
                "frame_id": self._frame_id,
                "frame_age_seconds": age,
                "open_attempts": self._open_attempts,
                "failures": self._failures,
                "reconnects": self._reconnects,
                "buffer_count": len(self._ring),
                "buffer_bytes": sum(len(data) for _, data in self._ring),
                "capture_worker_alive": bool(self._thread and self._thread.is_alive()),
                "backend": self.backend,
                "rgbd_rejected_groups": self._rgbd_rejected,
                "depth_ready": ready and self._latest_depth is not None,
                "depth_valid_fraction": self._latest_sensor.get("depth_valid_fraction")
                if self._latest_sensor
                else None,
            }

    def close(self):
        """通知采集退出并使旧帧失效，确认线程结束后释放设备。

        线程仍阻塞时标记关闭未确认并报错，避免新所有者与旧读取操作竞争。
        """
        with self._lifecycle_lock:
            self._stop.set()
            self._invalidate("stopping")
            if self._thread is not None:
                self._thread.join(timeout=5)
                if self._thread.is_alive():
                    self._state = "close_unconfirmed"
                    raise RuntimeError(
                        "Camera read has not released; close is unconfirmed"
                    )
                self._thread = None
            self._release_capture()
            self._alive = False
            self._invalidate("stopped")
