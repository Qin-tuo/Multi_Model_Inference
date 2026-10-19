# SPDX-License-Identifier: Apache-2.0
"""Bounded shared-frame body/face/hand inference using pinned existing backends."""

import json
import threading
import time
from pathlib import Path
import numpy as np
from app.body_detector import BodyDetector, checked_file, load_source


_MANIFEST = Path(__file__).resolve().parents[1] / "models/perception.json"


class HandDetector:
    def __init__(self, manifest, palm_threshold=0.5, hand_threshold=0.8, max_hands=2):
        self.max_hands = max_hands
        self.palm = self.hand = None
        for role, class_name in [("palm", "MPPalmDet"), ("hand", "MPHandPose")]:
            entry = manifest[role]
            filename = entry.get("runtime_source_file", entry["source_file"])
            source = entry["files"][filename]
            module = load_source(source["path"], source["sha256"], "reachy_" + role)
            weight_name = Path(entry["model_path"]).name
            model = checked_file(
                entry["model_path"], entry["files"][weight_name]["sha256"]
            )
            kwargs = (
                {"scoreThreshold": palm_threshold}
                if role == "palm"
                else {"confThreshold": hand_threshold}
            )
            setattr(self, role, getattr(module, class_name)(str(model), **kwargs))
        self.provider = "OpenCV CPU"

    def infer(self, image):
        if self.palm is None or self.hand is None:
            raise RuntimeError("Hand models are closed")
        palms = self.palm.infer(image)
        if palms.ndim != 2 or palms.shape[1] != 19 or not np.isfinite(palms).all():
            raise ValueError("Invalid palm model output")
        selected = sorted(palms, key=lambda p: float(p[-1]), reverse=True)[
            : self.max_hands
        ]
        hands = []
        invalid_rois = 0
        height, width = image.shape[:2]
        for palm in selected:
            x1, y1, x2, y2 = palm[:4]
            if min(x2, width) <= max(x1, 0) or min(y2, height) <= max(y1, 0):
                invalid_rois += 1
                continue
            output = self.hand.infer(image, palm)
            if output is None:
                continue
            if output.shape != (132,) or not np.isfinite(output).all():
                raise ValueError("Invalid hand landmark output")
            if not 0 <= output[-1] <= 1 or not 0 <= output[-2] <= 1:
                raise ValueError("Invalid hand confidence or handedness")
            # Upstream screen z is a model estimate relative to the wrist, not
            # a RealSense depth measurement. No per-keypoint confidence exists.
            hands.append(
                {
                    "type": "hand",
                    "box_xyxy": output[:4].tolist(),
                    "landmarks": output[4:67].reshape(21, 3).tolist(),
                    "score": float(output[131]),
                    "handedness_score": float(output[130]),
                    "palm_score": float(palm[18]),
                    "landmark_order": "MediaPipe Hands 21",
                    "z_kind": "model_relative_wrist_pixels",
                    "landmark_confidence": None,
                }
            )
        return {
            "hands": hands,
            "palms_detected": len(palms),
            "palms_selected": len(selected),
            "invalid_rois": invalid_rois,
        }

    def close(self):
        self.palm = self.hand = None


class Perception:
    """一个 worker 顺序运行人体、脸和手部模型，结果绑定同一原始帧。

    不积压待识别图像：每个周期读取最新帧。模型没有检出目标是有效结果；
    相机断开或结果过期则是不可用结果，二者不能混为“没有人”。
    """

    def __init__(self, camera, config, tracker=None, on_result=None):
        self.camera, self.config, self.tracker = camera, config, tracker
        self.body = self.hands = None
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

    def load(self):
        manifest = json.loads(_MANIFEST.read_text())
        try:
            if self.config.body_enabled:
                self.body = BodyDetector(
                    manifest["body"],
                    threshold=self.config.body_threshold,
                    iou=self.config.body_iou,
                    max_detections=self.config.max_people,
                )
            if self.config.hands_enabled:
                self.hands = HandDetector(
                    manifest,
                    self.config.palm_threshold,
                    self.config.hand_threshold,
                    self.config.max_hands,
                )
        except Exception:
            self.close()
            raise

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        if self.config.body_enabled and self.body is None:
            raise RuntimeError("Body model is not loaded")
        if self.config.hands_enabled and self.hands is None:
            raise RuntimeError("Hand models are not loaded")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="perception", daemon=True
        )
        self._thread.start()

    def process_frame(self, frame):
        """处理一帧并保留各阶段计时；发布前再次确认来源与新鲜度。"""
        started = time.monotonic()
        stages = {}
        persons, hands, face = (
            [],
            [],
            {"face_detected": False, "face_box": None, "frame": frame.metadata()},
        )
        if self.body:
            tick = time.monotonic()
            persons = self.body.infer(frame.bgr)
            stages["body_ms"] = (time.monotonic() - tick) * 1000
        if self.tracker:
            tick = time.monotonic()
            self.tracker.process_frame(frame)
            face = self.tracker.snapshot()
            stages["face_ms"] = (time.monotonic() - tick) * 1000
        hand_stats = {}
        if self.hands:
            tick = time.monotonic()
            hand_stats = self.hands.infer(frame.bgr)
            hands = hand_stats.pop("hands")
            stages["hands_ms"] = (time.monotonic() - tick) * 1000
        finished = time.monotonic()
        if (
            frame.source_epoch != self.camera.source_epoch
            or finished - frame.received_monotonic > self.camera.stale_after
        ):
            raise RuntimeError("Perception result expired before publication")
        return {
            "ready": True,
            "error": None,
            "frame": frame.metadata(),
            "persons": persons,
            "hands": hands,
            "face": face,
            "hand_stats": hand_stats,
            "completed_monotonic": finished,
            "elapsed_ms": (finished - started) * 1000,
            "stages": stages,
            "providers": {
                "body": self.body.provider if self.body else None,
                "face": "YuNet OpenCV CPU" if self.tracker else None,
                "hands": self.hands.provider if self.hands else None,
            },
            "body_stages_ms": dict(getattr(self.body, "last_timings", {})),
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
        if self.body:
            self.body.close()
            self.body = None
        if self.hands:
            self.hands.close()
            self.hands = None
        with self._lock:
            self._state = self._empty()
            self._preview = None
