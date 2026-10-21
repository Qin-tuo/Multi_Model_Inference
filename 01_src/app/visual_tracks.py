# SPDX-License-Identifier: Apache-2.0
"""基于 Norfair 的有界轨迹关联，组合框几何与可选外观特征，管理来源和时间失效。"""

import uuid
import numpy as np
from norfair import Detection, Tracker
from norfair.distances import iou


def appearance_distance(first, second):
    """同一特征空间内计算余弦距离；任一特征缺失或空间不同则返回 None。"""
    if (
        first.embedding is None
        or second.embedding is None
        or first.data.get("space") != second.data.get("space")
    ):
        return None
    return float(1 - np.clip(np.dot(first.embedding, second.embedding), -1, 1))


class VisualTracks:
    """将同帧检测交给 Norfair 关联，并限制轨迹容量与跨帧有效时间。"""

    def __init__(self, namespace, capacity=20, ttl=2.0):
        """设置轨迹命名空间、数量上限和有效期，并创建初始跟踪器。"""
        self.namespace, self.capacity, self.ttl = namespace, capacity, ttl
        self.reset()

    def reset(self):
        """重建 Norfair 跟踪器及随机代次标识，避免重连后复用旧轨迹身份。"""
        self.epoch = self.stamp = self.frame_id = None
        self.generation = uuid.uuid4().hex[:8]
        self.tracker = Tracker(
            distance_function=self._distance,
            distance_threshold=0.7,
            hit_counter_max=8,
            initialization_delay=1,
            past_detections_length=1,
            reid_hit_counter_max=8,
            reid_distance_function=self._reid_distance,
            reid_distance_threshold=0.2,
        )

    @staticmethod
    def _distance(detection, tracked):
        """组合框的 IoU 距离和兼容外观距离，明显不相符的候选直接拒绝关联。"""
        geometric = float(
            iou(detection.points.reshape(1, 4), tracked.estimate.reshape(1, 4))[0, 0]
        )
        if geometric >= 0.95:
            return 1.0
        appearance = appearance_distance(detection, tracked.last_detection)
        if appearance is not None:
            if appearance > 0.75:
                return 1.0
            return 0.5 * geometric + 0.5 * appearance
        return geometric

    def _reid_distance(self, first, second):
        """为同类别的新旧轨迹计算外观重识别距离，并拒绝存在近似替代候选的歧义匹配。"""
        a, b = first.last_detection, second.last_detection
        if a.label != b.label:
            return 1.0
        current = (self.epoch, self.frame_id)
        if (a.data["key"] == current) == (b.data["key"] == current):
            return 1.0
        new, old = (a, b) if a.data["key"] == current else (b, a)
        distance = appearance_distance(new, old)
        if distance is None:
            return 1.0
        alternatives = []
        for obj in self.tracker.tracked_objects:
            other = obj.last_detection
            if other is old or other.data["key"] == current or other.label != new.label:
                continue
            value = appearance_distance(new, other)
            if value is not None:
                alternatives.append(value)
        # Ambiguous appearance must not silently steal an existing identity.
        if alternatives and min(alternatives) - distance < 0.05:
            return 1.0
        return distance

    def update(self, rows, frame):
        """将当前帧目标送入 Norfair，返回带轨迹 ID 和确认标记的目标副本。

        来源变化或长时间中断会重置；乱序帧报错，旧轨迹按时间失效并受容量限制。
        """
        epoch, index, stamp = (
            frame.source_epoch,
            frame.frame_id,
            frame.received_monotonic,
        )
        if epoch != self.epoch or (
            self.stamp is not None and stamp - self.stamp > self.ttl
        ):
            self.reset()
        if self.stamp is not None and (stamp <= self.stamp or index <= self.frame_id):
            raise ValueError("Out-of-order tracking observation")
        self.epoch, self.frame_id, self.stamp = epoch, index, stamp
        # Wall-clock expiry is independent of processing/capture frame rate.
        self.tracker.tracked_objects = [
            o
            for o in self.tracker.tracked_objects
            if stamp - o.last_detection.data["stamp"] <= self.ttl
        ]
        detections = []
        for i, row in enumerate(rows[: self.capacity]):
            detections.append(
                Detection(
                    points=np.asarray(row["box_xyxy"], np.float32).reshape(2, 2),
                    scores=np.full(2, row["score"]),
                    label=row.get("class_id", "face"),
                    embedding=row.get("_embedding"),
                    data={
                        "index": i,
                        "key": (epoch, index),
                        "stamp": stamp,
                        "space": row.get("_embedding_space"),
                    },
                )
            )
        active = self.tracker.update(detections)
        current = {}
        for obj in active:
            data = obj.last_detection.data
            if data["key"] == (epoch, index):
                current[data["index"]] = self.key(obj)
        self.tracker.tracked_objects = sorted(
            self.tracker.tracked_objects,
            # 容量裁剪优先保留近期确实被观察到的轨迹。
            key=lambda o: o.last_detection.data["stamp"],
            reverse=True,
        )[: self.capacity]
        return [
            dict(row, track_id=current.get(i), track_confirmed=i in current)
            for i, row in enumerate(rows)
        ]

    def key(self, obj):
        """组合来源批次、跟踪代次、命名空间和 Norfair ID，生成不会跨重置混用的标识。"""
        return f"{self.epoch}:{self.generation}:{self.namespace}:{obj.id}"

    def known_ids(self):
        """返回仍在跟踪器内且已分配 ID 的轨迹标识集合。"""
        return {self.key(o) for o in self.tracker.tracked_objects if o.id is not None}
