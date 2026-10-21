# SPDX-License-Identifier: Apache-2.0
"""最新帧工作线程的公共生命周期：去重、限频、失效检查和发布状态。

具体模型加载与单帧算法由子类实现，公共层只管理线程及结果归属。
"""

import threading
import time


class FrameWorker:
    """由一个工作线程写入结果，所有发布结果均关联其实际处理的源帧。"""

    def __init__(self, camera, config, tracker=None, on_result=None):
        """绑定相机、配置和结果回调，初始化线程、状态锁及处理/跳帧/失败计数。"""
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
        """构造未就绪的空感知状态，用于缺帧、过期、失败或关闭时替换旧结果。"""
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
        """反复读取最新帧，按来源批次和帧号去重后处理并发布，再等待剩余周期。

        跳过的采集帧单独计数；推理异常清空本帧状态，回调异常单独记录，不积累无限待处理队列。
        """
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
        """返回状态副本及累计计数，读取时再次屏蔽过期或相机重连前的结果。"""
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
        """成对返回预览帧和对应结果；来源批次变化或帧龄超限时返回空预览。"""
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
        """通知线程退出并等待结束，确认退出后清空状态；超时则报错并保留资源所有权。"""
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
        """检查帧的来源批次、显式失效事件和主机接收时间，判断能否继续使用。"""
        return (
            not self._stop.is_set()
            and frame.source_epoch == self.camera.source_epoch
            and not (frame.invalidated and frame.invalidated.is_set())
            and time.monotonic() - frame.received_monotonic <= self.camera.stale_after
        )

    def start(self):
        """启动唯一感知工作线程；已有存活线程时直接返回，避免重复消费者。"""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="perception", daemon=True
        )
        self._thread.start()
