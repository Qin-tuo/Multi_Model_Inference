#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Diagnose recorded motion without models, backend calls or production edits.

Counterfactuals isolate open-palm continuity from motion/quality gates.
They are diagnostic hypotheses, not accepted runtime behavior or accuracy
results. The original observations and all rejected examples remain intact.
"""

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_src"))
from app.config import Config
from app.gestures import GestureRecognizer as CurrentGestureRecognizer
from deploy import runtime

# Replaying a diagnostic after a fix must still use the measured original
# algorithm. Load the trusted repository snapshot and verify its frozen hash.
BASELINE_REVISION = "before-gesture-wave-fix-20261020"
BASELINE_SOURCE = subprocess.check_output(
    ["git", "show", BASELINE_REVISION + ":01_src/app/gestures.py"], cwd=ROOT
)
BASELINE_SHA = hashlib.sha256(BASELINE_SOURCE).hexdigest()
expected = json.loads(
    (ROOT / "benchmarks/manifests/b1-vision-measurements-v1.json").read_text()
)["locked_files"]["01_src/app/gestures.py"]
if BASELINE_SHA != expected:
    raise ValueError("Diagnostic source differs from the measured original")
baseline_module = types.ModuleType("b1_original_gestures")
exec(
    compile(BASELINE_SOURCE, BASELINE_REVISION + "/gestures.py", "exec"),
    baseline_module.__dict__,
)
GestureRecognizer = baseline_module.GestureRecognizer


class AssumeOpen(GestureRecognizer):
    """Upper bound only: keep confidence/geometry gates, ignore hand opening."""

    def _sample(self, hand, frame):
        sample, reason = super()._sample(hand, frame)
        if sample is not None:
            sample["open"] = True
        return sample, reason


class BridgeOpening(GestureRecognizer):
    """Diagnostic history restoration after a brief valid-but-closed sample.

    Retain only actual open observations for wave. Stationary evidence is
    still cleared by the original implementation. Missing, low-confidence,
    stale, ambiguous or owner-changed tracks cannot receive restored evidence.
    """

    def _update(self, scene, now):
        old = {key: list(track.samples) for key, track in self._tracks.items()}
        events = super()._update(scene, now)
        if not self._state.get("ready"):
            return events
        stamp = scene["frame"]["received_monotonic"]
        for key, track in self._tracks.items():
            if (
                key in old
                and track.seen == stamp
                and track.neutral_since is not None
                and stamp - track.neutral_since < self.config.rearm_seconds
                and not track.latched
            ):
                track.samples.extend(old[key])
        return events


class MotionWithOpening(AssumeOpen):
    """Diagnostic motion with the existing minimum number of open observations.

    Valid closed-hand positions can describe a brief turn of the palm, but a
    wave requires a currently open hand and at least min_observations actual
    open samples. A closed interval reaching rearm_seconds discards history.
    Stationary stops are disabled for this diagnostic only.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.open_samples = {}

    def _sample(self, hand, frame):
        sample, reason = GestureRecognizer._sample(self, hand, frame)
        if sample is not None:
            key = (frame["frame_id"], sample["motion_x"], sample["scale"])
            self.open_samples[key] = sample["open"]
            self.open_samples = {
                k: v
                for k, v in self.open_samples.items()
                if k[0] >= frame["frame_id"] - 64
            }
            sample["open"] = True
        return sample, reason

    def _wave(self, samples):
        samples = list(samples)
        opened = [
            self.open_samples.get((s["frame_id"], s["x"], s["scale"]), False)
            for s in samples
        ]
        if not opened[-1]:
            return False
        closed_since, after = None, 0
        for i, (sample, is_open) in enumerate(zip(samples, opened)):
            if not is_open and closed_since is None:
                closed_since = sample["time"]
            if (
                closed_since is not None
                and sample["time"] - closed_since >= self.config.rearm_seconds
            ):
                after = i if is_open else i + 1
            if is_open:
                closed_since = None
        samples, opened = samples[after:], opened[after:]
        return sum(opened) >= self.config.min_observations and super()._wave(samples)


def analyze(folder):
    cfg = Config.load(folder / "effective-config.json")
    rows = [
        json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()
    ]
    rows = [r for r in rows if r["kind"] == "scene"]
    policies = {
        "current_runtime": CurrentGestureRecognizer(
            cfg.gestures,
            cfg.perception.hand_threshold,
            cfg.perception.pose_joint_threshold,
        ),
        "original": GestureRecognizer(
            cfg.gestures,
            cfg.perception.hand_threshold,
            cfg.perception.pose_joint_threshold,
        ),
        "brief_closed_wave_history_hypothesis": BridgeOpening(
            cfg.gestures,
            cfg.perception.hand_threshold,
            cfg.perception.pose_joint_threshold,
        ),
        "qualified_motion_with_open_observations_hypothesis": MotionWithOpening(
            replace(cfg.gestures, stop_hold_seconds=999),
            cfg.perception.hand_threshold,
            cfg.perception.pose_joint_threshold,
        ),
        "all_eligible_hands_open_no_stop_upper_bound": AssumeOpen(
            replace(cfg.gestures, stop_hold_seconds=999),
            cfg.perception.hand_threshold,
            cfg.perception.pose_joint_threshold,
        ),
    }
    results = {}
    for name, policy in policies.items():
        events, rejection_counts = [], Counter()
        sample_gates, evidence_windows = Counter(), []
        longest = []
        seen = set()
        for row in rows:
            frame = row["frame"]
            if not frame:
                continue
            key = frame["source_epoch"], frame["frame_id"]
            if key in seen:
                continue
            seen.add(key)
            scene = {
                "ready": row["gesture_state"]["ready"],
                "frame": frame,
                "hands": row["hands"],
            }
            produced = policy.update(scene, now=row["observed_monotonic"])
            if name == "current_runtime":
                # Inspect the unchanged runtime's own decisions. Retain every
                # window, including failures; this is not a threshold sweep.
                for hand in row["hands"]:
                    sample, reason = policy._sample(hand, frame)
                    sample_gates["hand_observations"] += 1
                    gate = reason or ("open" if sample["open"] else "non_open")
                    sample_gates[gate] += 1
                windows = []
                for track in policy._tracks.values():
                    samples = list(track.samples)
                    if (
                        track.seen != frame["received_monotonic"]
                        or not samples
                        or samples[-1]["frame_id"] != frame["frame_id"]
                    ):
                        continue
                    windows.append(
                        {
                            "handedness": track.handedness,
                            "latched": track.latched,
                            "open_observations": sum(s["open"] for s in samples),
                            "required_open_observations": cfg.gestures.min_observations,
                            "wave_predicate": policy._wave(samples),
                            "samples": samples,
                        }
                    )
                evidence_windows.append(
                    {
                        "frame_id": frame["frame_id"],
                        "media_frame_index": frame["sensor"].get("media_frame_index"),
                        "media_seconds": frame["sensor"].get("media_timestamp_seconds"),
                        "windows": windows,
                    }
                )
            for event in produced:
                events.append(
                    {
                        "kind": event["kind"],
                        "frame_id": event["frame_id"],
                        "media_seconds": frame["sensor"].get("media_timestamp_seconds"),
                        "evidence_seconds": event["evidence_end_monotonic"]
                        - event["evidence_start_monotonic"],
                    }
                )
            rejection_counts.update(policy._state.get("rejected_hands", {}))
            for track in policy._tracks.values():
                if len(track.samples) > len(longest):
                    longest = [dict(s) for s in track.samples]
        results[name] = {
            "events": events,
            "unique_frames": len(seen),
            "rejected_hand_observations": dict(rejection_counts),
            "longest_wave_evidence": longest,
        }
        if name == "current_runtime":
            results[name]["sample_gate_counts"] = dict(sample_gates)
            results[name]["evidence_windows"] = evidence_windows
    return {
        "input_directory": str(folder.relative_to(ROOT)),
        "events_sha256": runtime.sha256(folder / "events.jsonl"),
        "policies": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-directory", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep previous diagnostic evidence")
    report = {
        "scope": __doc__,
        "status": "DIAGNOSTIC_COUNTERFACTUALS_ONLY",
        "source_sha256": runtime.sha256(__file__),
        "baseline_gesture_revision": BASELINE_REVISION,
        "baseline_gesture_sha256": BASELINE_SHA,
        "current_gesture_sha256": runtime.sha256(ROOT / "01_src/app/gestures.py"),
        "cases": [analyze(path.resolve()) for path in args.case_directory],
        "limits": [
            "No greeting, tool, audio or backend is executed.",
            "Assume-open also suppresses stationary-stop solely to inspect whether any eligible motion could meet the existing wave reversals/amplitude gate.",
            "The brief-closed hypothesis leaves all confidence, size, association and wave thresholds unchanged; a real implementation would need negative/regression evidence and a new runtime baseline lock.",
        ],
    }
    runtime.write_json(args.output, report)
    for case in report["cases"]:
        print(
            case["input_directory"],
            {k: v["events"] for k, v in case["policies"].items()},
        )


if __name__ == "__main__":
    main()
