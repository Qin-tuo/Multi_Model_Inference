#!/usr/bin/env python3
"""Join E1 client/server records by turn ID; retain failures and routing issues."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.baseline_support import distribution  # noqa: E402
from benchmarks.summarize_vision import artifacts, resource_summary  # noqa: E402
from deploy import runtime  # noqa: E402


def rows(path):
    return [json.loads(line) for line in path.open()]


def summarize(run):
    report = json.loads((run / "report.json").read_text())
    server = rows(run / "server-1/server-events.jsonl")
    states = rows(run / "state-samples.jsonl")
    by_turn = defaultdict(list)
    for event in server:
        if event.get("turn_id"):
            by_turn[event["turn_id"]].append(event)
    classes = defaultdict(list)
    timelines = []
    irrelevant_reads = []
    unexpected_writes = []
    for row in report["requests"]:
        key = row.get("turn_id")
        observed = by_turn[key]
        final = next(
            (e["result"] for e in observed if e["kind"] == "final_result"),
            row.get("result"),
        )
        sample = {
            k: row.get(k)
            for k in (
                "class",
                "case",
                "request_id",
                "turn_id",
                "outcome",
                "http_status",
                "first_client_content_seconds",
                "submit_to_done_seconds",
                "submit_to_idle_seconds",
            )
        }
        sample["admission_ack_seconds"] = row["response_received"] - row["submitted"]
        sample["server_result"] = final
        sample["stages"] = []
        for event in observed:
            if event["kind"] == "dequeued":
                sample["queue_wait_seconds"] = (
                    event["dequeued_monotonic"] - event["admitted_monotonic"]
                )
            if event["kind"] in {
                "file_vad",
                "asr_call",
                "llm_call",
                "tts_call",
                "image_base64",
                "tool_completed",
                "audio_published",
            }:
                sample["stages"].append(
                    {
                        "kind": event["kind"],
                        "start_seconds": event["started"] - row["submitted"],
                        "end_seconds": event["ended"] - row["submitted"],
                        "elapsed_seconds": event["ended"] - event["started"],
                        "name": event.get("name"),
                        "round": event.get("round"),
                        "error": event.get("error"),
                    }
                )
        tts = [
            e
            for e in observed
            if e["kind"] == "tts_call"
            and e.get("first_pcm_estimated_monotonic") is not None
        ]
        published = [e for e in observed if e["kind"] == "audio_published"]
        sample["first_pcm_estimate_seconds"] = (
            min(e["first_pcm_estimated_monotonic"] for e in tts) - row["submitted"]
            if tts
            else None
        )
        sample["audio_publication_seconds"] = (
            min(e["ended"] for e in published) - row["submitted"] if published else None
        )
        sample["model_round_count"] = len(
            [e for e in observed if e["kind"] == "llm_call"]
        )
        sample["tool_calls"] = [
            {
                "name": e["name"],
                "elapsed_seconds": e["ended"] - e["started"],
                "result": e["result"],
            }
            for e in observed
            if e["kind"] == "tool_completed"
        ]
        sample["model_call_sum_seconds"] = sum(
            s["elapsed_seconds"] for s in sample["stages"] if s["kind"] == "llm_call"
        )
        if row["class"] == "actions":
            sample["verified_notes"] = row.get("verified_notes")
            committed = [
                e
                for e in observed
                if e["kind"] == "tool_completed" and e["name"] == "create_note"
            ]
            sample["first_note_receipt_seconds"] = (
                min(e["ended"] for e in committed) - row["submitted"]
                if committed
                else None
            )
        if row["class"] in {"cancellation", "overload_incumbent"}:
            cancel = next(
                (
                    e
                    for e in observed
                    if e["kind"] == "server_event"
                    and e["message"]["type"] == "cancelled"
                ),
                None,
            )
            sample.update(
                target_stage=row.get("target_stage"),
                cancel_to_idle_seconds=row.get(
                    "cancel_to_idle_seconds", row["idle_received"] - row["cancel_sent"]
                ),
                cancel_to_server_flag_seconds=cancel["emitted_monotonic"]
                - row["cancel_sent"]
                if cancel
                else None,
                cancelled_audio_status=row.get("cancelled_audio_status"),
                late_output=row.get("late_output"),
                target_stage_not_observed=row.get("target_stage_not_observed", False),
            )
        # These corpus cases ask about the image, not stored records. Keep the
        # unrequested read calls as observed model-routing limitations.
        if row["case"] in {"audio_image-03", "audio_image-04", "audio_image-05"} or row[
            "case"
        ].startswith("warm-image"):
            if sample["tool_calls"]:
                irrelevant_reads.append(
                    {
                        "case": row["case"],
                        "turn_id": key,
                        "tools": [t["name"] for t in sample["tool_calls"]],
                        "answer": final.get("text") if final else None,
                    }
                )
        authorized = row["class"] == "actions" or row["case"] in {
            "audio-08",
            "audio-10",
            "audio-11",
            "audio_image-08",
            "audio_image-10",
            "audio_image-11",
        }
        if not authorized:
            for tool in sample["tool_calls"]:
                if tool["name"] in {
                    "create_note",
                    "create_reminder",
                    "cancel_reminder",
                }:
                    unexpected_writes.append({"case": row["case"], "tool": tool})
        classes[row["class"]].append(sample)
        if row["case"] in {"audio_image-03", "audio_image-04", "action-01"}:
            timelines.append(sample)
    aggregates = {}
    metric_names = [
        "admission_ack_seconds",
        "queue_wait_seconds",
        "first_client_content_seconds",
        "first_pcm_estimate_seconds",
        "audio_publication_seconds",
        "submit_to_done_seconds",
        "submit_to_idle_seconds",
        "model_call_sum_seconds",
        "first_note_receipt_seconds",
        "cancel_to_idle_seconds",
        "cancel_to_server_flag_seconds",
    ]
    for kind, values in classes.items():
        samples = [s for s in states if s["phase"] == kind]
        finite = [
            s["latest_age_ms"] for s in samples if s.get("latest_age_ms") is not None
        ]
        aggregates[kind] = {
            "requests": len(values),
            "outcomes": dict(Counter(s["outcome"] for s in values)),
            "metrics": {
                key: distribution(
                    [s[key] for s in values if s.get(key) is not None and s[key] >= 0]
                )
                for key in metric_names
            },
            "model_round_counts": dict(Counter(s["model_round_count"] for s in values)),
            "tool_call_counts": dict(
                Counter(t["name"] for s in values for t in s["tool_calls"])
            ),
            "resources": resource_summary(
                run / "resources.csv", lambda phase: phase == kind
            )
            if kind not in {"warmup", "overload_incumbent"}
            else None,
            "vision_state_samples": len(samples),
            "finite_latest_age_ms": distribution(finite),
            "unavailable_state_or_result_samples": len(samples) - len(finite),
            "observer_missed_deadlines": sum(
                s.get("missing") == "observer_missed_deadline" for s in samples
            ),
            "perception_unready_observations": sum(
                s.get("perception", {}).get("ready") is False for s in samples
            ),
            "age_p95_including_unavailable_ms": sorted(finite)[
                math.ceil(0.95 * len(samples)) - 1
            ]
            if samples and math.ceil(0.95 * len(samples)) <= len(finite)
            else None,
        }
    overload = report["overload"]
    snapshots_equal = (
        report["notes_before_restart"]
        == report["notes_after_process_exit"]
        == report["notes_after_restart"]
    )
    return {
        "created_at": runtime.stamp(),
        "status": "E1_SOFTWARE_COLLECTED_WITH_DECLARED_MODEL_LIMITATIONS",
        "summary_script_sha256": runtime.sha256(__file__),
        "classes": aggregates,
        "overload": {
            "arrivals": len(overload),
            "http_status_counts": dict(Counter(s["http_status"] for s in overload)),
            "send_lateness_seconds": distribution(
                [max(0, s["sent"] - s["scheduled"]) for s in overload]
            ),
            "response_seconds": distribution(
                [s["received"] - s["sent"] for s in overload]
            ),
        },
        "functional_checks": {
            "checks_failed": report["checks_failed"],
            "note_count": len(report["notes_before_restart"]),
            "notes_survive_exit_and_restart": snapshots_equal,
            "post_restart_api": json.loads(
                (run / "restart-api-verification.json").read_text()
            ),
            "unexpected_mutating_calls": unexpected_writes,
        },
        "known_model_routing_issues": irrelevant_reads,
        "trace_examples": timelines,
        "requests": dict(classes),
        "raw_directory": str(run.resolve()),
        "raw_status": report["status"],
        "restored": report["restored"]["status"],
        "server_runs": report["server_runs"],
        "monitor_errors": report.get("monitor_errors"),
        "sampler_errors": report.get("sampler_errors"),
        "files": artifacts(run),
        "workload": json.loads((run / "workload.json").read_text()),
        "boundaries": [
            "Production app classes and all enabled functionality; only camera-driver input is fixed public-image replay. E1 is not physical D435i acquisition. Real camera observations are separately available in V2 and functional probes.",
            "Normal classes retain history within each class, clear at boundaries, retain durable state throughout. Fifteen representative non-mutating warmups precede measurement. Stateful commands are not silently repeated for warmup.",
            "Client times start before HTTP submission/file reading, not microphone speech end; first text is receipt over WebSocket, not browser paint. First PCM is estimated from TTS wrapper entry plus the production first-block duration, not audible output.",
            "Model adapters currently buffer text when tools are enabled; end-to-end first content is distinct from raw model first token. Preserve actual rounds, cache counts and unrelated read-tool calls.",
            "Per-role spans are host call boundaries and may include I/O/waits. They are not CUDA kernel traces and their percentile values must not be added to infer total latency. Bench-only observation overhead remains included.",
            "Server send-hook records include send attempts; delivered-output assertions use client WebSocket events. Intended cancellation exceptions are not ordinary model failures.",
            "Periodic state/result age measures host receipt, with observer loss retained; empty valid detections are not missing results. 1Hz resource samples can miss short GPU activity and exact allocation peaks.",
            "No extra writes observed outside explicit note/reminder requests. Unrequested read tools did occur for image questions; these remain an unresolved model-routing issue, not a claim of perfect tool selection or semantic accuracy.",
            "One initial workload, not 8h endurance or five-run optimization evidence. Physical audio and real gesture negative/occlusion/multi-hand acceptance remain deferred/pending; user mastery still requires user execution.",
            "Performance source was archived before collection. The later harness adds a bounded overload-busy wait and post-restart API checks; those checks are outside the measured windows and do not change production inference.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tests-xml", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run)
    suite = ET.parse(args.tests_xml).find("testsuite")
    report["automatic_tests"] = {
        **{k: int(suite.get(k, 0)) for k in ("tests", "errors", "failures", "skipped")},
        "xml_sha256": runtime.sha256(args.tests_xml),
    }
    runtime.write_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
