#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize recorded vision stream windows without running models or services."""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import statistics
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01_src"))
from benchmarks.baseline_support import distribution
from deploy import runtime


def read(path):
    return json.loads(path.read_text())


def measured_event(row, start, end):
    return (
        start <= row.get("completed_monotonic", row.get("observed_monotonic", -1)) < end
    )


def board_telemetry(folder, monitor, start, end):
    anchor = monitor["clock_anchor"]
    rows = []
    for line in (folder / "tegrastats.log").read_text().splitlines():
        match = re.match(r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}) ", line)
        if not match:
            continue
        wall = (
            datetime.strptime(match[1], "%m-%d-%Y %H:%M:%S")
            .replace(tzinfo=ZoneInfo("Asia/Shanghai"))
            .timestamp()
        )
        stamp = anchor["monotonic"] + wall - anchor["wall_time_unix"]
        if not start <= stamp < end:
            continue
        values = {}
        for name, pattern, divisor in (
            ("ram_used_mb_reported", r" RAM (\d+)/", 1),
            ("cpu_temp_c", r" cpu@([\d.]+)C", 1),
            ("gpu_temp_c", r" gpu@([\d.]+)C", 1),
            ("module_input_w", r" VDD_IN (\d+)mW", 1000),
        ):
            found = re.search(pattern, line)
            values[name] = float(found[1]) / divisor if found else None
        frequencies = [float(x) for x in re.findall(r"\d+%@(\d+)", line)]
        values["cpu_freq_max_mhz"] = max(frequencies) if frequencies else None
        rows.append(values)
    return {
        "sample_count": len(rows),
        "timestamp_scope": "Tegrastats local Asia/Shanghai timestamps, 1-second resolution, mapped to monitor monotonic anchor",
        "metrics": {
            key: {
                "mean": statistics.mean(v),
                "max": max(v),
                "min": min(v),
                "count": len(v),
            }
            for key in (rows[0] if rows else [])
            if (v := [r[key] for r in rows if r[key] is not None])
        },
        "gpu_clock_mhz": None,
        "gpu_clock_note": "This board's recorded tegrastats lines omit GPU MHz; do not infer clock from utilization.",
    }


def condition_summary(folder):
    result, monitor = read(folder / "result.json"), read(folder / "monitor.json")
    if result["status"] != "MEASURED" or result["errors"] or monitor["errors"]:
        raise ValueError("Condition or resource monitor failed: " + folder.name)
    start, end = result["window_start_monotonic"], result["window_end_monotonic"]
    seconds = end - start
    events = [
        json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()
    ]
    completed = [
        r for r in events if r["kind"] == "perception" and measured_event(r, start, end)
    ]
    interior = [r for r in completed if r["started_monotonic"] >= start]
    failures = [
        r
        for r in events
        if r["kind"] == "frame_failure" and measured_event(r, start, end)
    ]
    samples = result["periodic_samples"]
    finite = [r["latest_age_ms"] for r in samples if r["latest_age_ms"] is not None]
    rank = math.ceil(0.95 * len(samples)) - 1
    camera_before, camera_after = result["camera_before"], result["camera_after"]
    stable_source = camera_before["source_epoch"] == camera_after["source_epoch"]
    published = (
        camera_after["frame_id"] - camera_before["frame_id"] if stable_source else None
    )
    before, after = result["perception_before"], result["perception_after"]
    with (folder / "resources.csv").open() as f:
        resource_rows = [
            r for r in csv.DictReader(f) if start <= float(r["monotonic"]) < end
        ]
    resources = {}
    for key in resource_rows[0] if resource_rows else []:
        if key in ("monotonic", "phase"):
            continue
        values = [float(r[key]) for r in resource_rows if r[key] != ""]
        if values:
            resources[key] = {
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
            if key.endswith("_cpu_seconds") and len(resource_rows) > 1:
                span = float(resource_rows[-1]["monotonic"]) - float(
                    resource_rows[0]["monotonic"]
                )
                resources[key]["cpu_seconds_delta"] = values[-1] - values[0]
                resources[key]["average_cpu_cores"] = (values[-1] - values[0]) / span
                resources[key]["counter_note"] = (
                    "Values are lifetime CPU-second counters; use delta/interval for average occupied CPU cores."
                )
    stages = {}
    for key in sorted({k for r in interior for k in r["stages"]}):
        values = [r["stages"][key] for r in interior if key in r["stages"]]
        stages[key] = distribution(values)
    turns = [
        r
        for r in events
        if r["kind"] in ("background_turn", "background_llm")
        and r.get("started_monotonic", -1) >= start
        and r.get("completed_monotonic", end + 1) < end
        and not r.get("window_end_cancel")
    ]
    turn_latencies = [
        (r["completed_monotonic"] - r["started_monotonic"]) * 1000 for r in turns
    ]
    trace_stages = {}
    trace_marks = {}
    for row in turns:
        trace = row.get("trace") or {}
        for name, stage in trace.get("stages", {}).items():
            trace_stages.setdefault(name, []).append(stage["duration_ms"])
        for name, value in trace.get("marks_ms", {}).items():
            trace_marks.setdefault(name, []).append(value)
    metadata = [
        r.get("result", {}).get("metadata", r.get("metadata", {})) for r in turns
    ]
    tokens = [r["eval_count"] for r in metadata if r.get("eval_count") is not None]
    cached = [
        r["usage"]["prompt_tokens_details"]["cached_tokens"]
        for r in metadata
        if r.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens")
        is not None
    ]
    pressure = [r for r in events if r["kind"] in ("background_turn", "background_llm")]
    censored = [r for r in pressure if r.get("window_end_cancel")]
    return {
        "condition": result["condition"],
        "seconds": seconds,
        "capture_fps": published / seconds if published is not None else None,
        "processing_fps_counter": (after["processed"] - before["processed"]) / seconds,
        "completion_fps_events": len(completed) / seconds,
        "completed_frames": len(completed),
        "frames_used_for_latency": len(interior),
        "latency_ms": {
            key: distribution([r[key] for r in interior if r.get(key) is not None])
            for key in (
                "received_to_start_ms",
                "post_jpeg_wait_ms",
                "processing_ms",
                "received_to_complete_ms",
                "source_schedule_to_complete_ms",
            )
        },
        "periodic_latest_age_ms": distribution(finite),
        "age_p95_including_unavailable_ms": sorted(finite)[rank]
        if 0 <= rank < len(finite)
        else None,
        "no_result_samples": len(samples) - len(finite),
        "periodic_samples": len(samples),
        "no_result_fraction": (len(samples) - len(finite)) / len(samples)
        if samples
        else 1,
        "observer_missed_deadlines": sum(
            r.get("missing_reason") == "observer_missed_deadline" for r in samples
        ),
        "observer_ms": distribution(
            [r["observer_ms"] for r in samples if "observer_ms" in r]
        ),
        "skipped_captured_frames": after["skipped_capture_frames"]
        - before["skipped_capture_frames"],
        "frame_failure_counter_delta": after["failures"] - before["failures"],
        "frame_failure_events": len(failures),
        "expired_frames": sum(r["expired"] for r in failures),
        "unexpected_frame_errors": [r for r in failures if not r["expired"]],
        "camera_reconnects": camera_after["reconnects"] - camera_before["reconnects"],
        "rgbd_groups_rejected": camera_after["rgbd_rejected_groups"]
        - camera_before["rgbd_rejected_groups"],
        "camera_buffer_peak_bytes_sampled": max(
            (r["camera"]["buffer_bytes"] for r in samples if "camera" in r),
            default=None,
        ),
        "model_adapter_counters_total_including_warmup": result["models"],
        "model_counter_note": "Adapter errors include input rejection; they are not automatically engine failures. Per-frame failures/expiry are counted separately.",
        "stages_ms": stages,
        "resources_1s_window": resources,
        "worker_rss_high_water_bytes_entire_lifetime": result["rss_high_water_bytes"],
        "board_1s_window": board_telemetry(folder, monitor, start, end),
        "background": {
            "complete_turns_entirely_inside_window": len(turns),
            "turn_latency_ms": distribution(turn_latencies),
            "controller_stage_ms": {
                k: distribution(v) for k, v in trace_stages.items()
            },
            "controller_marks_ms": {k: distribution(v) for k, v in trace_marks.items()},
            "output_tokens": distribution(tokens),
            "cached_prompt_tokens": distribution(cached),
            "stage_scope": "Production TurnTrace wall times, potentially overlapping LLM/TTS spans; not kernel times. Caller total includes up to 25ms polling delay after owner release. Dequeued mark reports admission-to-worker queue delay.",
            "completed_total_including_warmup": result.get(
                "background_completed_total"
            ),
            "busy_fraction_sampled": statistics.mean(
                [r["background_busy"] for r in samples if "background_busy" in r]
            )
            if any("background_busy" in r for r in samples)
            else None,
            "end_window_censored_requests": len(censored),
            "latency_scope": "Only requests starting and completing inside the measured window; raw events retain warmup, boundary and cancelled requests. Small request counts are reported explicitly.",
        },
        "raw_hashes": {
            name: runtime.sha256(folder / name)
            for name in (
                "result.json",
                "events.jsonl",
                "resources.csv",
                "tegrastats.log",
                "monitor.json",
            )
        },
    }


def summarize(output):
    run = read(output / "report.json")
    spec = read(output / "workload.json")
    if run["status"] != "MEASURED":
        raise ValueError("Stream run did not complete")
    results = {}
    for name, row in run["conditions"].items():
        if row["exit_code"]:
            raise ValueError("Worker failed: " + name)
        results[name] = condition_summary(output / name)
    summary = {
        "created_at": runtime.stamp(),
        "status": "STREAM_CONDITIONS_MEASURED"
        if set(results) == set(spec["conditions"])
        else "MEASURED_SUBSET",
        "raw_directory": str(output.relative_to(ROOT)),
        "workload_sha256": run["workload_sha256"],
        "conditions": results,
        "scope": spec["scope"],
        "gesture_policy": spec["gesture_policy"],
        "comparison_rules": [
            "Replay conditions use identical frozen images/synthetic depth; live scenes are not paired equivalent inputs.",
            "Processing target is capped at 5 FPS. Latest-frame skips are intentional bounded-backlog behavior, distinct from capture faults and expired/failed results.",
            "Receive-to-start includes JPEG/dispatch; post-JPEG wait isolates the interval after encoding. Monotonic receive is not sensor exposure time.",
            "Reported stage percentiles cannot be added to obtain end-to-end percentiles.",
            "Resource peaks are 1s sampled within the window, except worker RSS high-water which covers its entire lifetime; CPU/GPU share board memory.",
            "GUI/browser/websocket not measured here; full browser/service functional probes are separate.",
        ],
    }
    runtime.write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.output.resolve())
    print(value["status"], args.output / "summary.json")


if __name__ == "__main__":
    main()
