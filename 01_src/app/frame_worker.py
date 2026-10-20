# SPDX-License-Identifier: Apache-2.0
"""Latest-frame ownership, freshness, bounded cadence and worker teardown.

Model loading and frame processing belong to the concrete perception pipeline.
"""

import threading
import time


class FrameWorker:
    """One writer publishes results from its own source frame; subclasses infer."""

    def __init__(self, camera, config, tracker=None, on_result=None):
        self.camera, self.config, self.tracker = camera, config, tracker
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._preview = None
        self._state = self._empty()
        self.processed = self.skipped = self.failures = 0
        self.on_result = on_result
        self.callback_error = None

    @staticmethod
    def _empty(error=None):
        return {
            "ready": False,
            "error": error,
            "frame": None,
            "persons": [],
            "hands": [],
            "face": {"face_detected": False, "face_box": None},
            "completed_monotonic": None,
        }

    def _loop(self):
        # 一轮包括处理和等待，目标周期为 1/fps。处理变快会增加等待时间；
        # 这也是单模块耗时降低后，配置为 5 FPS 的输出不自动升高的原因。
        previous = None
        while not self._stop.is_set():
            start = time.monotonic()
            frame = self.camera.snapshot()
            if frame is None:
                with self._lock:
                    self._state = self._empty("No fresh camera frame")
                    self._preview = None
            else:
                key = (frame.source_epoch, frame.frame_id)
                if key != previous:
                    if previous and previous[0] == key[0]:
                        self.skipped += max(0, key[1] - previous[1] - 1)
                    previous = key
                    try:
                        result = self.process_frame(frame)
                        with self._lock:
                            self._state = result
                            self._preview = frame
                        self.processed += 1
                    except Exception as exc:
                        self.failures += 1
                        with self._lock:
                            self._state = self._empty(str(exc))
                            self._preview = None
            if self.on_result:
                try:
                    self.on_result(self.snapshot())
                    self.callback_error = None
                except Exception as exc:
                    self.callback_error = str(exc)
            self._stop.wait(max(0, 1 / self.config.fps - (time.monotonic() - start)))

    def snapshot(self):
        with self._lock:
            result = dict(self._state)
        frame = result["frame"]
        if frame and (
            frame["source_epoch"] != self.camera.source_epoch
            or time.monotonic() - frame["received_monotonic"] > self.camera.stale_after
        ):
            result = self._empty("Perception result expired")
        return {
            **result,
            "processed": self.processed,
            "skipped_capture_frames": self.skipped,
            "failures": self.failures,
            "callback_error": self.callback_error,
        }

    def preview(self):
        with self._lock:
            frame = self._preview
            result = dict(self._state)
        if (
            frame is None
            or frame.source_epoch != self.camera.source_epoch
            or time.monotonic() - frame.received_monotonic > self.camera.stale_after
        ):
            return None, self._empty("Perception preview expired")
        return frame, result

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("Perception worker has not released")
            self._thread = None
        with self._lock:
            self._state = self._empty()
            self._preview = None

    def _fresh(self, frame):
        return (
            not self._stop.is_set()
            and frame.source_epoch == self.camera.source_epoch
            and not (frame.invalidated and frame.invalidated.is_set())
            and time.monotonic() - frame.received_monotonic <= self.camera.stale_after
        )

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="perception", daemon=True
        )
        self._thread.start()
