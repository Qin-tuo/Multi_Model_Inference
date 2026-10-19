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

"""Camera — USB webcam with shared latest frame and VLM ring buffer.

The background thread owns hardware reads and publishes frames into a
timestamped ring buffer. Consumers such as the web UI, face tracker, and VLM
capture use the latest buffered frame instead of competing for VideoCapture.
"""

import base64
import threading
import time
import uuid
import os
from dataclasses import dataclass
from collections import deque
from typing import Optional

import cv2
import numpy as np

MAX_SPEECH_SECS = 16
PRE_SPEECH_SECS = 0.5


@dataclass(frozen=True)
class FrameSnapshot:
    frame_id: int
    source_epoch: str
    received_monotonic: float
    bgr: np.ndarray
    jpeg: bytes
    time_kind: str = "host_receive_monotonic"
    invalidated: threading.Event | None = None

    def metadata(self):
        return {
            "frame_id": self.frame_id,
            "source_epoch": self.source_epoch,
            "received_monotonic": self.received_monotonic,
            "time_kind": self.time_kind,
            "width": self.bgr.shape[1],
            "height": self.bgr.shape[0],
            "stride": self.bgr.strides[0],
            "encoding": "bgr8",
        }


class Camera:
    """V4L2 USB webcam with a background capture thread and timestamped
    ring buffer sized to cover the maximum speech duration plus lookback."""

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
    ):
        if min(width, height, capture_fps, stale_after, buffer_bytes_limit) <= 0:
            raise ValueError("Camera sizes, rates and limits must be positive")
        self.stale_after = stale_after
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
        # The camera can ignore the requested 640x480 mode and return
        # 1920x1080 BGR frames. Keeping a 16.5 second raw ring at 10 FPS would
        # retain about 979 MiB. Preserve the temporal window as JPEG bytes and
        # keep only the newest frame raw for face tracking.
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

    def _sync_actual_frame_geometry(self) -> None:
        """Record the mode accepted by the camera and calibrate for that mode."""
        if self._cap is None:
            return
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width > 0:
            self.width = actual_width
        if actual_height > 0:
            self.height = actual_height

    def _invalidate(self, state, error=None):
        with self._lock:
            self._source_invalidated.set()
            self._source_invalidated = threading.Event()
            self.source_epoch = uuid.uuid4().hex
            self._frame_id = 0
            self._ring.clear()
            self._latest = self._latest_jpeg = None
            self._actual_fps = 0
            self._state = state
            if error is not None:
                self.last_error = error

    def start(self) -> bool:
        """One owner opens, reads, releases and retries; start needs a fresh frame."""
        with self._lifecycle_lock:
            if self._thread and self._thread.is_alive():
                return self.health_check()
            if self._cap is not None:
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
                        ret, frame = self._cap.read()
                        if self._stop.is_set():
                            break
                        if not ret:
                            raise RuntimeError("V4L2 camera read failed")
                        received = time.monotonic()
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
                    # Only this owner releases the handle, after read returns.
                    if self._cap is not None:
                        self._cap.release()
                        self._cap = None
                if self._stop.wait(self.reconnect_seconds):
                    break
        except Exception as exc:
            self.last_error = str(exc)
        finally:
            self._alive = False
            self._invalidate("stopped" if self._stop.is_set() else "failed")

    def snapshot(self, *, copy: bool = False) -> Optional[FrameSnapshot]:
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
            )

    def _encode_frame_bytes(self, frame: np.ndarray) -> Optional[bytes]:
        ok, jpg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            return None
        return jpg.tobytes()

    def _encode_frame(self, frame: np.ndarray) -> Optional[str]:
        jpg = self._encode_frame_bytes(frame)
        if jpg is None:
            return None
        return base64.b64encode(jpg).decode("ascii")

    @staticmethod
    def _base64_jpeg(jpg: bytes) -> str:
        return base64.b64encode(jpg).decode("ascii")

    def get_speech_frames(
        self,
        speech_start: float,
        speech_end: float,
        max_frames: int = 3,
    ) -> list[str]:
        """Return frames from the speech window [speech_start, speech_end].

        When max_frames == 1, returns the most recent frame (best context).
        When max_frames > 1, evenly samples across the window including
        PRE_SPEECH_SECS lookback for temporal context.
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

        # No fallback to a different time window or stale pre-restart image.
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
        frame = self.snapshot()
        return self._base64_jpeg(frame.jpeg) if frame else None

    def latest_raw(self, *, copy: bool = True) -> Optional[np.ndarray]:
        frame = self.snapshot(copy=copy)
        return frame.bgr if frame else None

    def read_live(self) -> Optional[str]:
        return self.capture_single()

    def read_raw_live(self) -> Optional[np.ndarray]:
        return self.latest_raw()

    @property
    def buffer_count(self) -> int:
        with self._lock:
            return len(self._ring)

    @property
    def actual_fps(self) -> float:
        return self._actual_fps

    @property
    def buffer_bytes(self) -> int:
        with self._lock:
            return sum(len(data) for _, data in self._ring)

    def health_check(self) -> bool:
        return self._alive and self.snapshot() is not None

    def status(self):
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
            }

    def close(self):
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
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._alive = False
            self._invalidate("stopped")
