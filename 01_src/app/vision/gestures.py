# SPDX-License-Identifier: Apache-2.0
"""依据手部关键点执行有界的挥手/张掌时序规则。

开掌判定改编自 OpenCV Zoo 的五指角度与腕距规则；连续证据按人物、手和帧来源隔离。
"""

from collections import deque
import threading
import time
import uuid
import numpy as np


def open_palm(points):
    """用五指角度和指尖相对腕距判定开掌，退化向量或非有限角度按未开掌处理。"""
    xy = np.asarray(points, dtype=float)[:, :2]
    vectors = [
        (xy[0] - xy[base], xy[joint] - xy[tip])
        for base, joint, tip in [
            (2, 3, 4),
            (6, 7, 8),
            (10, 11, 12),
            (14, 15, 16),
            (18, 19, 20),
        ]
    ]
    for first, second in vectors:
        length = np.linalg.norm(first) * np.linalg.norm(second)
        if length <= 1e-8:
            return False
        angle = np.degrees(np.arccos(np.clip(np.dot(first, second) / length, -1, 1)))
        if not np.isfinite(angle) or angle >= 49:
            return False
    return all(
        np.linalg.norm(xy[tip] - xy[0]) > np.linalg.norm(xy[base] - xy[0])
        for base, tip in [(5, 4), (6, 8), (10, 12), (14, 16), (18, 20)]
    )


class _Track:
    """保存单只手的最近观察、运动证据、静止证据和一次触发锁存状态。"""

    def __init__(self, sample, stamp):
        """为新手部观察分配身份，保存掌心/尺度/所属人物并初始化触发状态。"""
        self.id = uuid.uuid4().hex
        self.center, self.scale, self.handedness = (
            sample["center"],
            sample["scale"],
            sample["handedness"],
        )
        self.seen = stamp
        self.owner = sample.get("owner")
        self.latched = False
        self.neutral_since = None
        self.samples = deque(maxlen=32)
        self.clear_evidence()

    def clear_evidence(self):
        """清除挥动历史与静止证据，不把中断前的动作接到新观察上。"""
        self.samples.clear()
        self.clear_stationary()

    def clear_stationary(self):
        """清空静止锚点、起始时间和有效帧数，重新积累举掌停止证据。"""
        self.stationary_anchor = None
        self.stationary_since = None
        self.stationary_count = 0


class GestureRecognizer:
    """感知线程作为单写入者，按帧更新手势证据并向其他线程提供状态副本。"""

    def __init__(self, config, min_hand_score=0.8, min_pose_joint_score=0.2):
        """保存配置和质量阈值，初始化手部轨迹、帧序及线程安全状态。"""
        self.config, self.min_hand_score = config, min_hand_score
        self.min_pose_joint_score = min_pose_joint_score
        self._lock = threading.Lock()
        self._tracks = {}
        self._epoch = None
        self._last_frame = -1
        self._last_stamp = None
        self._state = {"ready": False, "reason": "awaiting_frames", "tracks": []}
        self._last_event = None

    def snapshot(self):
        """返回当前手势摘要，超过事件有效期时标为陈旧，同时保留最近事件供观察。"""
        with self._lock:
            frame = self._state.get("frame")
            if (
                frame
                and time.monotonic() - frame["received_monotonic"]
                > self.config.event_ttl_seconds
            ):
                return {
                    "ready": False,
                    "reason": "stale_frame",
                    "tracks": [],
                    "last_event": self._last_event,
                }
            return {**self._state, "last_event": self._last_event}

    def update(self, scene, now=None):
        """在锁内把场景交给时序状态机，默认使用板端单调时钟。"""
        with self._lock:
            return self._update(scene, time.monotonic() if now is None else now)

    def _invalid(self, reason, now):
        """使当前证据失效并清理超时手轨迹，返回不触发任何动作的空事件列表。"""
        for track in self._tracks.values():
            track.clear_evidence()
        self._tracks = {
            key: t
            for key, t in self._tracks.items()
            if now - t.seen <= self.config.max_gap_seconds
        }
        self._state = {"ready": False, "reason": reason, "tracks": []}
        return []

    def _sample(self, hand, frame):
        """校验 21 点、得分、左右手类别、图像边界和掌宽，生成可关联的手部观察。

        SimCC 峰值按关键点阈值检查，不当作概率；人物轨迹 ID 用于阻止跨人继承证据。
        """
        points = np.asarray(hand["landmarks"], dtype=float)
        if points.shape not in ((21, 2), (21, 3)) or not np.isfinite(points).all():
            return None, "invalid_landmarks"
        if hand.get("score_kind") == "simcc_peak":
            scores = np.asarray(hand.get("landmark_scores"), dtype=float)
            if (
                scores.shape != (21,)
                or not np.isfinite(scores).all()
                or scores.min() < self.min_pose_joint_score
            ):
                return None, "low_pose_confidence"
            if hand.get("side") not in {"left", "right"}:
                return None, "invalid_handedness"
            # An anatomical topology label is categorical, not a probability.
            handedness = 0.0 if hand["side"] == "left" else 1.0
        else:
            if not self.min_hand_score <= hand["score"] <= 1:
                return None, "low_confidence"
            if not 0 <= hand["handedness_score"] <= 1:
                return None, "invalid_handedness"
            handedness = hand["handedness_score"]
        if (
            points[:, 0].min() < 0
            or points[:, 0].max() >= frame["width"]
            or points[:, 1].min() < 0
            or points[:, 1].max() >= frame["height"]
        ):
            return None, "clipped_hand"
        scale = float(np.linalg.norm(points[5, :2] - points[17, :2]))
        if scale < self.config.min_palm_pixels:
            return None, "small_hand"
        return {
            "points": points[:, :2],
            "center": points[[0, 5, 9, 13, 17], :2].mean(axis=0),
            "motion_x": float(points[:, :2].mean(axis=0)[0]),
            "scale": scale,
            "handedness": handedness,
            "owner": hand.get("person_track_id"),
            "open": open_palm(points),
        }, None

    def _associate(self, samples):
        """按人物归属、左右手、尺度和位置做双向最近关联；歧义候选不继承旧证据。"""
        tracks = list(self._tracks.values())
        candidates = {}
        reverse = {}
        for i, sample in enumerate(samples):
            for track in tracks:
                if sample.get("owner") != track.owner:
                    continue
                # Strongly conflicting handedness is not one continuous track.
                if (sample["handedness"] > 0.8 and track.handedness < 0.2) or (
                    sample["handedness"] < 0.2 and track.handedness > 0.8
                ):
                    continue
                if (
                    max(sample["scale"], track.scale)
                    / min(sample["scale"], track.scale)
                    > 2
                ):
                    continue
                distance = float(
                    np.linalg.norm(sample["center"] - track.center)
                    / max(sample["scale"], track.scale)
                )
                if distance <= self.config.association_palm_widths:
                    candidates.setdefault(i, []).append((distance, track.id))
                    reverse.setdefault(track.id, []).append((distance, i))

        def clear_best(options):
            """只返回与第二候选有足够距离差的最佳匹配，无候选或歧义时返回 None。"""
            options = sorted(options)
            if not options or (
                len(options) > 1 and options[1][0] - options[0][0] < 0.35
            ):
                return None
            return options[0][1]

        result = {}
        for i, options in candidates.items():
            key = clear_best(options)
            if key is not None and clear_best(reverse[key]) == i:
                result[i] = self._tracks[key]
        return result

    def _wave(self, samples):
        # Natural waving can briefly turn the palm away from the camera. Its
        # valid motion still counts, but the gesture needs enough actual open
        # observations and a currently open hand. A closed fist cannot greet.
        """检查最近运动是否包含足够横向摆幅和反向次数，且有足够开掌观察及当前开掌状态。"""
        if (
            not samples[-1]["open"]
            or sum(s["open"] for s in samples) < self.config.min_observations
        ):
            return False
        amplitude = self.config.wave_swing_palm_widths * float(
            np.median([x["scale"] for x in samples])
        )
        extreme = samples[0]["x"]
        direction = 0
        reversals = 0
        for sample in samples:
            change = sample["x"] - extreme
            if direction == 0:
                if abs(change) >= amplitude:
                    direction = 1 if change > 0 else -1
                    extreme = sample["x"]
            elif direction * change > 0:
                extreme = sample["x"]
            elif direction * change <= -amplitude:
                direction = -direction
                extreme = sample["x"]
                reversals += 1
        return reversals >= self.config.wave_reversals

    def _update(self, scene, now):
        """校验帧新鲜度/顺序后关联双手，累计挥手或静止张掌证据并生成带有效期的事件。

        同一帧不重复计数；触发后锁存，经过中性间隔才重新启用；同帧停止事件排序在问候之前。
        """
        cfg = self.config
        frame = scene.get("frame")
        if not cfg.enabled:
            return self._invalid("disabled", now)
        if not scene.get("ready") or not frame:
            return self._invalid("perception_unavailable", now)
        stamp = frame["received_monotonic"]
        epoch = frame["source_epoch"]
        index = frame["frame_id"]
        if not np.isfinite(stamp) or stamp > now or now - stamp > cfg.event_ttl_seconds:
            return self._invalid("stale_frame", now)
        if epoch != self._epoch:
            self._tracks.clear()
            self._epoch = epoch
            self._last_frame = -1
            self._last_stamp = None
        if index == self._last_frame:
            return []  # Polling one frame cannot grow evidence.
        if index < self._last_frame or (
            self._last_stamp is not None and stamp <= self._last_stamp
        ):
            return self._invalid("out_of_order_frame", now)
        if (
            self._last_stamp is not None
            and stamp - self._last_stamp > cfg.max_gap_seconds
        ):
            for track in self._tracks.values():
                track.clear_evidence()
        self._last_frame = index
        self._last_stamp = stamp
        self._tracks = {
            key: t
            for key, t in self._tracks.items()
            if stamp - t.seen <= cfg.max_gap_seconds
        }
        samples = []
        rejected = {}
        for hand in scene.get("hands", []):
            sample, reason = self._sample(hand, frame)
            if sample is not None:
                samples.append(sample)
            else:
                rejected[reason] = rejected.get(reason, 0) + 1
        if len(samples) > cfg.max_tracks:
            return self._invalid("too_many_hands", now)
        matches = self._associate(samples)
        seen = set()
        events = []
        status = []
        for i, sample in enumerate(samples):
            track = matches.get(i)
            if track is None:
                if len(self._tracks) >= cfg.max_tracks:
                    continue
                track = _Track(sample, stamp)
                self._tracks[track.id] = track
            seen.add(track.id)
            track.center, track.scale, track.handedness = (
                sample["center"],
                sample["scale"],
                sample["handedness"],
            )
            track.seen = stamp
            # A neutral interval rearms a latched gesture and also discards
            # earlier motion. This check includes the first reopened frame.
            if (
                track.neutral_since is not None
                and stamp - track.neutral_since >= cfg.rearm_seconds
            ):
                track.clear_evidence()
                track.latched = False
            if not track.latched:
                track.samples.append(
                    {
                        "time": stamp,
                        "frame_id": index,
                        "x": sample["motion_x"],
                        "scale": sample["scale"],
                        "open": sample["open"],
                    }
                )
                while (
                    track.samples
                    and stamp - track.samples[0]["time"] > cfg.wave_window_seconds
                ):
                    track.samples.popleft()
            if not sample["open"]:
                # Stop must be continuously open and stationary. Wave keeps
                # only a brief, observed turn of the palm; missing or invalid
                # observations still clear both histories below.
                track.clear_stationary()
                if track.neutral_since is None:
                    track.neutral_since = stamp
                status.append(
                    {
                        "source_id": track.id,
                        "open_palm": False,
                        "latched": track.latched,
                    }
                )
                continue
            track.neutral_since = None
            if not track.latched:
                if (
                    track.stationary_anchor is None
                    or np.max(
                        np.linalg.norm(
                            sample["points"] - track.stationary_anchor, axis=1
                        )
                    )
                    > cfg.stationary_palm_widths * sample["scale"]
                ):
                    track.stationary_anchor = sample["points"]
                    track.stationary_since = stamp
                    track.stationary_count = 0
                track.stationary_count += 1
                kind = None
                if len(track.samples) >= cfg.min_observations and self._wave(
                    track.samples
                ):
                    kind = "wave"
                elif (
                    track.stationary_count >= cfg.min_observations
                    and stamp - track.stationary_since >= cfg.stop_hold_seconds
                ):
                    kind = "stop"
                if kind:
                    first = (
                        track.samples[0]
                        if kind == "wave"
                        else next(
                            s
                            for s in track.samples
                            if s["time"] >= track.stationary_since
                        )
                    )
                    event = {
                        "event_id": uuid.uuid4().hex,
                        "kind": kind,
                        "source_id": track.id,
                        "source_epoch": epoch,
                        "frame_id": index,
                        "first_frame_id": first["frame_id"],
                        "evidence_start_monotonic": first["time"],
                        "evidence_end_monotonic": stamp,
                        "confirmed_monotonic": now,
                        "expires_monotonic": stamp + cfg.event_ttl_seconds,
                        "time_kind": frame.get("time_kind", "host_receive_monotonic"),
                    }
                    events.append(event)
                    self._last_event = event
                    track.latched = True
            status.append(
                {
                    "source_id": track.id,
                    "open_palm": True,
                    "latched": track.latched,
                    "stationary_seconds": stamp - track.stationary_since
                    if track.stationary_since is not None
                    else 0,
                }
            )
        for key, track in self._tracks.items():
            if key not in seen:
                track.clear_evidence()
                track.neutral_since = None
        self._state = {
            "ready": True,
            "reason": None,
            "frame": frame,
            "tracks": status,
            "rejected_hands": rejected,
        }
        # A valid stop from either anonymous hand takes precedence in this frame.
        # 同一帧多只手同时触发时，先处理停止，再处理问候。
        return sorted(events, key=lambda e: e["kind"] != "stop")
