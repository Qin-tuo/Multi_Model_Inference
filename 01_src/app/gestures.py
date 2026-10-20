# SPDX-License-Identifier: Apache-2.0
"""OpenCV Zoo open-hand predicate and bounded anonymous temporal gestures.

open_palm adapts GestureClassification's 'Five' predicate from the pinned
OpenCV Zoo handpose demo (models/perception.json). Temporal rules are local
interaction policy. They require real positive/negative gesture validation.
"""

from collections import deque
import threading
import time
import uuid
import numpy as np


def open_palm(points):
    """The upstream five-finger angle + wrist-distance rule, with NaN guards."""
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
    def __init__(self, sample, stamp):
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
        self.samples.clear()
        self.clear_stationary()

    def clear_stationary(self):
        self.stationary_anchor = None
        self.stationary_since = None
        self.stationary_count = 0


class GestureRecognizer:
    """One writer (perception); readers only receive compact status copies."""

    def __init__(self, config, min_hand_score=0.8, min_pose_joint_score=0.2):
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
        with self._lock:
            return self._update(scene, time.monotonic() if now is None else now)

    def _invalid(self, reason, now):
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
        """Mutual nearest matches; ambiguous crossings restart evidence."""
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
        return sorted(events, key=lambda e: e["kind"] != "stop")
