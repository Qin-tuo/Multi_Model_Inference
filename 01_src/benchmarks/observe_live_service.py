#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read the running app's metrics; never load models or control services.

Run on the Jetson itself: frame ages use its monotonic clock. HTTP snapshots
can miss frames. Compare unique observed frames with the app's completion
counter and report gaps rather than claiming complete-window percentiles.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
from pathlib import Path
import time


def distribution(values):
    values = sorted(v for v in values if isinstance(v, (float, int))
                    and math.isfinite(v) and v >= 0)
    return {
        "count": len(values),
        "p50": values[math.ceil(len(values) * .50) - 1] if values else None,
        "p95": values[math.ceil(len(values) * .95) - 1] if values else None,
        "max": values[-1] if values else None,
        "method": "nearest-rank",
    }


def observe(state, tick, request_ms):
    """Keep numeric observations, not images, identities or conversation text."""
    scene = state.get("perception", {})
    frame = scene.get("frame") or {}
    camera = state.get("camera", {})
    diagnostics = state.get("diagnostics", {})
    received = frame.get("received_monotonic")
    completed = scene.get("completed_monotonic")
    ready = bool(scene.get("ready") and received is not None
                 and frame.get("source_epoch") == camera.get("source_epoch"))
    return {
        "observed_monotonic": tick,
        "request_ms": request_ms,
        "session": diagnostics.get("session"),
        "epoch": camera.get("source_epoch"),
        "model_set": scene.get("model_set"),
        "phase": state.get("phase"),
        "ready": ready,
        "processed": scene.get("processed"),
        "failures": scene.get("failures"),
        "skipped": scene.get("skipped_capture_frames"),
        "camera_frame_id": camera.get("frame_id"),
        "camera_reconnects": camera.get("reconnects"),
        "rgbd_rejected_groups": camera.get("rgbd_rejected_groups"),
        "frame_id": frame.get("frame_id"),
        "received_monotonic": received,
        "completed_monotonic": completed,
        "latest_result_age_ms": (tick - received) * 1000 if ready else None,
        "received_to_complete_ms": (completed - received) * 1000
        if ready and completed is not None else None,
        "processing_ms": scene.get("elapsed_ms") if ready else None,
        "stages_ms": scene.get("stages", {}) if ready else {},
        "task_counts": scene.get("task_counts", {}) if ready else {},
        "scene_counts": {k: len(scene.get(k, []))
                         for k in ("persons", "objects", "poses", "faces")},
        "diagnostics_sequence": diagnostics.get("sequence"),
        "resource_sample_monotonic": diagnostics.get("sampled_monotonic"),
        "resources": diagnostics.get("resources", {}),
        "turn": diagnostics.get("turn", {}),
    }


class Observation:
    def __init__(self, first, condition):
        self.first = self.last = first
        self.condition = condition
        self.frames = {}
        self.samples = []
        self.resources = {}
        self.turns = {}
        self.issues = set()
        self.http_errors = []

    def add(self, row):
        for key in ("session", "epoch", "model_set"):
            if row[key] != self.first[key]:
                self.issues.add(key + "_changed")
        for key in ("processed", "failures", "skipped", "camera_frame_id",
                    "camera_reconnects", "rgbd_rejected_groups"):
            if (not isinstance(row[key], int) or not isinstance(self.last[key], int)
                    or row[key] < self.last[key]):
                self.issues.add(key + "_counter_invalid_or_reset")
        self.last = row
        self.samples.append(row)
        new_frame = None
        if (row["ready"] and row["frame_id"] is not None
                and isinstance(row["processed"], int)
                and row["processed"] > self.first["processed"]):
            key = (row["epoch"], row["frame_id"])
            if key not in self.frames:
                new_frame = {k: row[k] for k in (
                    "epoch", "frame_id", "processed", "received_monotonic",
                    "completed_monotonic", "processing_ms",
                    "received_to_complete_ms", "stages_ms", "task_counts",
                    "scene_counts", "phase")}
                self.frames[key] = new_frame
        if row["diagnostics_sequence"] is not None:
            self.resources[(row["session"], row["diagnostics_sequence"])] = {
                "sampled_monotonic": row["resource_sample_monotonic"],
                **row["resources"],
            }
        last_turn = row["turn"].get("last") or {}
        if last_turn.get("turn_id"):
            self.turns[last_turn["turn_id"]] = last_turn
        return new_frame

    def report(self, interrupted=False):
        elapsed = self.last["observed_monotonic"] - self.first["observed_monotonic"]
        changes = {}
        for key in ("processed", "failures", "skipped", "camera_frame_id",
                    "camera_reconnects", "rgbd_rejected_groups"):
            first, last = self.first[key], self.last[key]
            changes[key] = last - first if isinstance(last, int) and last >= first else None
        completed = changes["processed"]
        frames = list(self.frames.values())
        observed = len(frames)
        issues = set(self.issues)
        phases = Counter(r["phase"] for r in self.samples)
        if self.condition == "resident-idle" and any(k != "idle" for k in phases):
            issues.add("non_idle_phase_observed")
        if not completed or observed != completed:
            issues.add("frame_coverage_incomplete")
        if not frames:
            issues.add("no_completed_frames_observed")
        if self.http_errors:
            issues.add("http_observation_errors")
        if interrupted:
            issues.add("interrupted")
        for r in frames:
            timings = [r["processing_ms"], r["received_to_complete_ms"],
                       r["stages_ms"].get("detection_segmentation_ms")]
            if any(not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0
                   for v in timings):
                issues.add("invalid_frame_timing")
        stages = sorted({k for r in frames for k in r["stages_ms"]})
        task_totals = Counter()
        for r in frames:
            task_totals.update({k: v for k, v in r["task_counts"].items()
                                if isinstance(v, (int, float))})
        initial_turn = (self.first["turn"].get("last") or {}).get("turn_id")
        return {
            "status": "OBSERVATION_WITH_LIMITATIONS" if issues
            else "OBSERVED_COUNTER_WINDOW_COMPLETE",
            "condition": self.condition,
            "scope": "Live app HTTP observation on the Jetson; percentiles use "
                     "unique observed completed frames. No model workload generated. "
                     "Completion-counter coverage is checked, not guaranteed by polling.",
            "elapsed_seconds_between_snapshots": elapsed,
            "counter_changes": changes,
            "unique_observed_frames": observed,
            "unobserved_completed_frames": max(0, completed - observed)
            if completed is not None else None,
            "frame_coverage_fraction": observed / completed if completed else None,
            "processing_fps_counter": completed / elapsed
            if completed is not None and elapsed > 0 else None,
            "frame_metrics_ms": {
                "detection_segmentation": distribution(
                    r["stages_ms"].get("detection_segmentation_ms") for r in frames),
                "processing": distribution(r["processing_ms"] for r in frames),
                "received_to_complete": distribution(
                    r["received_to_complete_ms"] for r in frames),
            },
            "stages_ms": {k: distribution(r["stages_ms"].get(k) for r in frames)
                          for k in stages},
            "task_counts_total_observed": dict(task_totals),
            "observed_frames_with_person": sum(r["scene_counts"]["persons"] > 0
                                               for r in frames),
            "phase_samples": dict(phases),
            "http_requests_after_anchor": len(self.samples),
            "http_roundtrip_ms": distribution(r["request_ms"] for r in self.samples),
            "latest_result_age_available_samples_ms": distribution(
                r["latest_result_age_ms"] for r in self.samples),
            "unavailable_result_samples": sum(not r["ready"] for r in self.samples),
            "resource_samples": list(self.resources.values()),
            "observed_new_turns": [r for key, r in self.turns.items()
                                   if key != initial_turn],
            "turn_scope": "Latest-result polling can miss turns; not a complete turn latency benchmark.",
            "issues": sorted(issues),
            "http_errors": self.http_errors,
            "limits": [
                "Frame age is host-receive to local HTTP observation, not sensor exposure to browser display.",
                "Sampled frame-age percentiles exclude unavailable results; inspect their count separately.",
                "Resource and turn diagnostics are cached and may precede the observation window.",
                "HTTP observer adds workload; request duration does not measure all observer overhead.",
                "No detection-quality or long-duration stability qualification.",
            ],
        }


def request_state(connection):
    start = time.monotonic()
    connection.request("GET", "/api/state")
    response = connection.getresponse()
    body = response.read(2 * 1024 * 1024 + 1)
    end = time.monotonic()
    if response.status != 200 or len(body) > 2 * 1024 * 1024:
        raise RuntimeError(f"State read failed: HTTP {response.status} or oversized response")
    state = json.loads(body)
    return state, observe(state, end, (end - start) * 1000)


def validate_first(row, model_set, condition):
    if not row["ready"] or row["model_set"] != model_set or not row["session"]:
        raise ValueError("App perception is not ready with the expected model set")
    for key in ("processed", "failures", "skipped", "camera_frame_id",
                "camera_reconnects", "rgbd_rejected_groups"):
        if not isinstance(row[key], int):
            raise ValueError("Missing counter: " + key)
    if condition == "resident-idle" and row["phase"] != "idle":
        raise ValueError("Wait for the current conversation to finish")


def markdown(report):
    def value(number):
        return "无数据" if number is None else f"{number:.3f}"
    rows = [
        "# 正式服务观察结果", "",
        f"- 状态：`{report['status']}`",
        f"- 场景：`{report['condition']}`",
        f"- 观察窗口：{value(report['elapsed_seconds_between_snapshots'])} 秒",
        f"- 完成计数增量：{report['counter_changes']['processed']}；"
        f"采到唯一帧：{report['unique_observed_frames']}；"
        f"未采到完成帧：{report['unobserved_completed_frames']}",
        f"- 完成率：{value(report['processing_fps_counter'])} FPS",
        f"- 整帧失败增量：{report['counter_changes']['failures']}；"
        f"跳过采集帧增量：{report['counter_changes']['skipped']}；"
        f"RGBD 拒绝组增量：{report['counter_changes']['rgbd_rejected_groups']}",
        f"- 观察到人体的帧：{report['observed_frames_with_person']}；"
        f"下游任务计数：`{report['task_counts_total_observed']}`",
        f"- 问题：{', '.join(report['issues']) or '未发现计数覆盖问题'}", "",
        "| 观察到的完整调用 | 样本数 | P50 ms | P95 ms |",
        "|---|---:|---:|---:|",
    ]
    for name, metric in report["frame_metrics_ms"].items():
        rows.append(f"| {name} | {metric['count']} | {value(metric['p50'])} | {value(metric['p95'])} |")
    age = report["latest_result_age_available_samples_ms"]
    rows.extend([
        "", f"可用结果的周期观察帧龄 P95：{value(age['p95'])} ms；"
        f"不可用结果样本：{report['unavailable_result_samples']}。",
        "",
        "以上来自正在运行的 app。若有漏采，分位数仅代表观察到的帧。"
        "即使计数覆盖完整，仍需检查观察器开销、场景和工作量；本结果不证明检测质量或长期稳定性。",
        "帧龄从主机接收开始，到板端 HTTP 观察为止，不包含传感器曝光到浏览器显示的完整延迟。",
        "",
        "逐次观察见 observations.jsonl；去重帧见 frames.jsonl；全部指标与限制见 report.json。",
    ])
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--hz", type=float, default=10)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--condition", default="resident-idle")
    parser.add_argument("--note", required=True, help="Describe people, distance, lighting and consumers")
    parser.add_argument("--expect-model-set", default="B2-standard-2d")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 3600 or not 1 <= args.hz <= 20:
        parser.error("seconds must be 1..3600 and hz must be 1..20")
    if args.output.exists():
        parser.error("output must be a new directory")
    connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=2)
    interrupted = False
    try:
        state, first = request_state(connection)
        validate_first(first, args.expect_model_set, args.condition)
        args.output.mkdir(parents=True, exist_ok=False)
        effective = state.get("effective_config", {})
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "endpoint": f"http://127.0.0.1:{args.port}/api/state",
            "condition": args.condition, "note": args.note,
            "seconds_requested": args.seconds, "poll_hz_requested": args.hz,
            "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "session": first["session"], "model_set": first["model_set"],
            "effective_config": {k: effective.get(k) for k in ("vision", "perception", "face")},
            "anchor": first,
        }
        (args.output / "environment.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        observation = Observation(first, args.condition)
        end_at = first["observed_monotonic"] + args.seconds
        next_at = first["observed_monotonic"]
        progress_at = next_at + 10
        print(f"Observing existing app for {args.seconds:g}s -> {args.output}", flush=True)
        with (args.output / "observations.jsonl").open("w") as raw, \
                (args.output / "frames.jsonl").open("w") as frames:
            try:
                while time.monotonic() < end_at:
                    next_at += 1 / args.hz
                    time.sleep(max(0, min(next_at, end_at) - time.monotonic()))
                    try:
                        _, row = request_state(connection)
                    except (OSError, http.client.HTTPException, ValueError, RuntimeError) as exc:
                        connection.close()
                        error = {"observed_monotonic": time.monotonic(), "error": str(exc)}
                        observation.http_errors.append(error)
                        raw.write(json.dumps(error, ensure_ascii=False) + "\n")
                    else:
                        new = observation.add(row)
                        raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                        if new:
                            frames.write(json.dumps(new, ensure_ascii=False) + "\n")
                    now = time.monotonic()
                    next_at = max(next_at, now)  # Do not burst requests after a stall.
                    if now >= progress_at:
                        print(f"Observed {len(observation.frames)} unique frames", flush=True)
                        progress_at = now + 10
            except KeyboardInterrupt:
                interrupted = True
        report = observation.report(interrupted)
        (args.output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        (args.output / "report.md").write_text(markdown(report))
        print(f"{report['status']}: {args.output / 'report.md'}", flush=True)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
