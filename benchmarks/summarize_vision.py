#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize frozen V1/V2 evidence; never rerun inference or omit failures."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.baseline_support import distribution  # noqa: E402
from deploy import runtime  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def artifacts(folder):
    return [
        {
            "file": str(p.relative_to(folder)),
            "bytes": p.stat().st_size,
            "sha256": runtime.sha256(p),
        }
        for p in sorted(folder.rglob("*"))
        if p.is_file() and "source" not in p.relative_to(folder).parts
    ]


def resource_summary(path, phase):
    rows = [r for r in csv.DictReader(path.open()) if phase(r["phase"])]
    if not rows:
        return {
            "sample_count": 0,
            "error": "No resource observations in requested phase",
        }
    result = {
        "sample_count": len(rows),
        "minimum_mem_available_gib": min(float(r["mem_available"]) for r in rows)
        / 2**30,
        "maximum_swap_used_gib": max(float(r["swap_used"]) for r in rows) / 2**30,
        "host_cpu_percent": distribution([float(r["cpu_percent"]) for r in rows]),
        "gpu_load_snapshot_percent": distribution(
            [float(r["gpu_load_percent"]) for r in rows if r["gpu_load_percent"]]
        ),
    }
    for name in rows[0]:
        if name.endswith(("_pss", "_rss", "_fds", "_threads")):
            values = [float(r[name]) for r in rows]
            result[name] = {
                "first": values[0],
                "last": values[-1],
                "sampled_max": max(values),
            }
        if name.endswith("_cpu_seconds") and len(rows) > 1:
            result[name.replace("_seconds", "_percent_one_core")] = (
                (float(rows[-1][name]) - float(rows[0][name]))
                / (float(rows[-1]["monotonic"]) - float(rows[0]["monotonic"]))
                * 100
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tests-xml", type=Path)
    args = parser.parse_args()
    if runtime.sha256(args.v1 / "effective-config.json") != runtime.sha256(
        args.v2 / "effective-config.json"
    ):
        raise ValueError("V1 and V2 have different effective configurations")
    modules, streams = {}, {}
    for module in ("body", "face", "hands"):
        folder = args.v1 / module
        value = read(folder / "result.json")
        rows = []
        for fixture in value["fixtures"]:
            events = [
                json.loads(line)
                for line in (folder / "events.jsonl").read_text().splitlines()
            ]
            first = next(
                (
                    e
                    for e in events
                    if e.get("input") == fixture["input"] and e["phase"] == "first"
                ),
                None,
            )
            rows.append(
                {
                    "input": fixture["input"],
                    "plain_ms": fixture["plain_ms"],
                    "instrumented_ms": fixture["instrumented_ms"],
                    "stage_ms": fixture["stage_summary_ms"],
                    "pair_count": len(fixture["pairs"]),
                    "parity_matches": sum(p["parity"] for p in fixture["pairs"]),
                    "signed_paired_difference_median_ms": statistics.median(
                        fixture["paired_overhead_ms"]
                    ),
                    "first_after_fixture_change_ms": first["ms"] if first else None,
                }
            )
        modules[module] = {
            "status": value["status"],
            "load_seconds": value["load_seconds"],
            "opencv_threads": value["opencv_threads"],
            "fixtures": rows,
            "errors": value["errors"],
            "resources": resource_summary(
                folder / "resources.csv", lambda s: s.endswith(":measured")
            ),
            "monitor": read(folder / "monitor.json"),
        }
    for condition in ("replay-only", "replay-resident", "replay-4b", "live-resident"):
        folder = args.v2 / condition
        value = read(folder / "result.json")
        events = [
            json.loads(line)
            for line in (folder / "events.jsonl").read_text().splitlines()
        ]
        start, end = value["window_start_monotonic"], value["window_end_monotonic"]
        completed = [
            e
            for e in events
            if e["kind"] == "perception"
            and e["ready"]
            and start <= e["completed_monotonic"] < end
        ]
        samples = value["periodic_samples"]
        finite_samples = [s for s in samples if s.get("sampled_monotonic") is not None]
        capture_rate = (
            finite_samples[-1]["captured_count"] - finite_samples[0]["captured_count"]
        ) / (
            finite_samples[-1]["sampled_monotonic"]
            - finite_samples[0]["sampled_monotonic"]
        )
        stages = {}
        for event in completed:
            for stage, ms in event["stages"].items():
                stages.setdefault(stage, []).append(ms)
        bg = [
            e
            for e in events
            if e["kind"] == "background_llm"
            and e["started_monotonic"] < end
            and e["completed_monotonic"] >= start
        ]
        streams[condition] = {
            "status": value["status"],
            "window_seconds": end - start,
            "samples": len(samples),
            "captured_rate_hz": capture_rate,
            "perception_completions": len(completed),
            "perception_rate_hz": len(completed) / (end - start),
            "latest_age_ms": value["finite_age_ms"],
            "age_p95_including_missing_ms": value["age_p95_including_missing_ms"],
            "no_result_fraction": value["no_result_fraction"],
            "observer_missed_deadlines": len(samples) - len(finite_samples),
            "perception_errors_cumulative": max(s["failures"] for s in finite_samples),
            "camera_after": value["camera_after"],
            "frame_processing_ms": distribution([e["elapsed_ms"] for e in completed]),
            "perception_stage_ms": {
                stage: distribution(v) for stage, v in stages.items()
            },
            "source_read_ms": distribution(
                [
                    e["ms"]
                    for e in events
                    if e["kind"] == "driver_read" and e["phase"] == "measured"
                ]
            ),
            "jpeg_encode_ms": distribution(
                [
                    e["ms"]
                    for e in events
                    if e["kind"] == "jpeg_encode" and e["phase"] == "measured"
                ]
            ),
            "frame_fetch_ms": distribution(
                [
                    e["ms"]
                    for e in events
                    if e["kind"] == "frame_fetch" and e["phase"] == "measured"
                ]
            ),
            "replay_missed_arrivals_with_warmup": value["replay_missed_arrivals"],
            "replay_delivery_lag_ms": distribution(
                [
                    (e["delivered_monotonic"] - e["scheduled_monotonic"]) * 1000
                    for e in events
                    if e["kind"] == "source_arrival"
                    and start <= e["delivered_monotonic"] < end
                ]
            ),
            "background_requests_overlapping_window": len(bg),
            "background_requests_completed": sum(
                bool(e["metadata"].get("done")) for e in bg
            ),
            "background_end_of_window_cancelled": sum(
                e["window_end_cancel"] for e in bg
            ),
            "resources": resource_summary(
                folder / "resources.csv", lambda s: s == "measured"
            ),
            "errors": value["errors"],
            "monitor": read(folder / "monitor.json"),
        }
    report = {
        "created_at": runtime.stamp(),
        "status": "VISION_COMPONENTS_COLLECTED_FULL_B0_PENDING",
        "summarizer_sha256": runtime.sha256(__file__),
        "modules": modules,
        "streams": streams,
        "raw_runs": [
            {
                "directory": str(folder.resolve()),
                "report_status": read(folder / "report.json")["status"],
                "restored": read(folder / "report.json")["restored"]["status"],
                "files": artifacts(folder),
            }
            for folder in (args.v1, args.v2)
        ],
        "boundaries": [
            "20 plain/instrumented pairs per module and fixture; three fixed inputs, not a representative accuracy dataset. Empty valid detection outputs are distinct from unavailable results.",
            "V1 total call timing excludes input PNG decoding and output JSON serialization. Face native detect includes OpenCV internal preprocessing/inference/NMS. TensorRT native call includes transfers, execution and synchronization.",
            "Hand pipeline timings include palm and up to two landmark ROIs; landmark stage distributions are per actual ROI call. Blank input has no landmark calls, not zero-cost landmark inference.",
            "Paired overhead differences can be negative due to runtime noise and dynamic clocks; values are retained without subtraction. V2 event/resource observation overhead is included and consistent across conditions, not independently removed.",
            "V2 age is host receipt to independent sample, not sensor exposure or display latency. Missing samples rank after finite ages. Only replay conditions share fixed input workload; live camera is a separate observational window.",
            "One ordered 60-second run per condition with current MAXN/dynamic clocks. Differences include thermal/scheduling history; no repeated-run statistical significance or causal bottleneck attribution claimed.",
            "1Hz GPU-load snapshots can miss brief TensorRT work; zero sampled load is not proof of zero GPU execution. Raw tegrastats clocks/temperature cover the whole worker including warmup; no precise per-phase alignment claimed for those raw logs.",
            "PSS, RSS and system MemAvailable have different meanings on shared-memory Orin; do not add them as GPU allocation. Resource peaks are sampled, not exact allocation maxima.",
            "Background 4B is a single backlogged adapter client with unchanged context/cache/output limits; GUI/voice/actions/gesture timing belongs to E1. The final in-flight request is intentionally cancelled at the window boundary.",
            "ASR, LLM/VLM, TTS and complete-application B0 measurements and handoff guide remain pending. Physical audio and user-deferred real gesture samples remain unqualified.",
        ],
    }
    if args.tests_xml:
        suite = ET.parse(args.tests_xml).find("testsuite")
        report["automatic_tests"] = {
            k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")
        }
        report["automatic_tests"].update(
            xml_path=str(args.tests_xml), xml_sha256=runtime.sha256(args.tests_xml)
        )
    runtime.write_json(args.output, report)
    print(
        json.dumps(
            {"status": report["status"], "summary": str(args.output)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
