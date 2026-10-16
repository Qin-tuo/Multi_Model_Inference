#!/usr/bin/env python3
"""Summarize S1/T1/L1 with cold/input-first, warm and quality scopes intact."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.baseline_support import distribution  # noqa: E402
from benchmarks.summarize_vision import artifacts, resource_summary  # noqa: E402
from deploy import runtime  # noqa: E402


def summarize(rows, role):
    valid = [row for row in rows if not row.get("error")]
    result = {
        "attempts": len(rows),
        "valid_calls": len(valid),
        "errors": [r for r in rows if r.get("error")],
    }
    keys = {
        "s1": [
            "vad_seconds",
            "asr_seconds",
            "total_seconds",
            "input_seconds",
            "segmented_audio_seconds",
            "asr_rtf_original_file",
            "asr_rtf_submitted_audio",
        ],
        "t1": ["first_pcm_seconds", "call_seconds", "duration", "synthesis_rtf"],
        "l1-text": [
            "first_content_seconds",
            "total_seconds",
            "image_base64_seconds",
            "preflight_http_seconds",
            "generation_http_first_content_seconds",
            "client_completion_tokens_per_total_second",
        ],
        "l1-image": [
            "first_content_seconds",
            "total_seconds",
            "image_base64_seconds",
            "preflight_http_seconds",
            "generation_http_first_content_seconds",
            "client_completion_tokens_per_total_second",
        ],
    }[role]
    result["metrics"] = {
        key: distribution([r[key] for r in valid if r.get(key) is not None])
        for key in keys
    }
    if role == "s1":
        references = sum(r["prompt_agreement"]["reference_characters"] for r in valid)
        edits = sum(r["prompt_agreement"]["edit_distance"] for r in valid)
        result["prompt_agreement"] = {
            "reference_characters": references,
            "edits": edits,
            "weighted_cer": edits / references if references else None,
            "exact_normalized_matches": sum(
                not r["prompt_agreement"]["edit_distance"] for r in valid
            ),
            "differences": [
                {
                    "case": r["case"],
                    "prompt": r["reference_prompt"],
                    "transcript": r["text"],
                    "score": r["prompt_agreement"],
                }
                for r in valid
                if r["prompt_agreement"]["edit_distance"]
            ],
        }
    if role.startswith("l1"):
        result["native_decode_tokens_per_second"] = distribution(
            [
                r["native_timings"]["predicted_per_second"]
                for r in valid
                if r.get("native_timings")
                and r["native_timings"].get("predicted_per_second") is not None
            ]
        )
        result["completion_tokens"] = distribution(
            [r["metadata"]["usage"]["completion_tokens"] for r in valid]
        )
        result["prompt_tokens"] = distribution(
            [r["metadata"]["usage"]["prompt_tokens"] for r in valid]
        )
        result["cached_tokens"] = distribution(
            [
                r["metadata"]["usage"]
                .get("prompt_tokens_details", {})
                .get("cached_tokens", 0)
                for r in valid
            ]
        )
        result["preflight_actual_token_deltas"] = sorted(
            {
                r["metadata"]["usage"]["prompt_tokens"]
                - r["metadata"]["input_tokens_preflight"]
                for r in valid
            }
        )
        result["finish_reasons"] = dict(
            Counter(r["metadata"].get("finish_reason") for r in valid)
        )
        result["slowest"] = max(
            (
                {
                    "case": r["case"],
                    "total_seconds": r["total_seconds"],
                    "completion_tokens": r["metadata"]["usage"]["completion_tokens"],
                    "finish_reason": r["metadata"].get("finish_reason"),
                }
                for r in valid
            ),
            key=lambda r: r["total_seconds"],
            default=None,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tests-xml", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads((args.run / "report.json").read_text())
    roles = {}
    for role in ("s1", "t1", "l1-text", "l1-image"):
        folder = args.run / role
        value = json.loads((folder / "result.json").read_text())
        first = [c["first"] for c in value["cases"]]
        measured = [
            next(r for r in c["measurements"] if r["phase"] == "measured")
            for c in value["cases"]
        ]
        roles[role] = {
            "status": value["status"],
            "case_count": len(value["cases"]),
            "first_input_call": summarize(first, role),
            "measured_after_warmup": summarize(measured, role),
            "warmup_calls": sum(len(c["warmups"]) for c in value["cases"]),
            "errors_all_phases": value["errors"],
            "resources_active_including_warmup": resource_summary(
                folder / "resources.csv",
                lambda phase: phase not in {"setup", "resident_after_measurement"},
            ),
            "monitor": json.loads((folder / "monitor.json").read_text()),
            "backend_before": value["backend_before"],
            "backend_after": value["backend_after"],
        }
        if role.startswith("l1"):
            roles[role]["observer_validation"] = {
                "pairs": len(value["cases"]),
                "equal_outputs": sum(c["paired_output_equal"] for c in value["cases"]),
                "signed_adjacent_pair_difference_median_seconds": statistics.median(
                    c["signed_observer_difference_seconds"]
                    for c in value["cases"]
                    if "signed_observer_difference_seconds" in c
                ),
                "note": "Alternating plain/instrumented pair order; signed differences retain runtime noise and are not subtracted from latency.",
            }
    suite = ET.parse(args.tests_xml).find("testsuite")
    report = {
        "created_at": runtime.stamp(),
        "status": "S1_T1_L1_COLLECTED_END_TO_END_B0_PENDING",
        "summarizer_sha256": runtime.sha256(__file__),
        "roles": roles,
        "raw_directory": str(args.run.resolve()),
        "raw_status": raw["status"],
        "restored": raw["restored"]["status"],
        "workload": json.loads((args.run / "workload.json").read_text()),
        "files": artifacts(args.run),
        "automatic_tests": {
            **{
                k: int(suite.get(k, 0))
                for k in ("tests", "failures", "errors", "skipped")
            },
            "xml_sha256": runtime.sha256(args.tests_xml),
        },
        "boundaries": [
            "20 fixed cases per role, one first call, five warmups and one measured call per case; L1 also has an adjacent plain call. All 600 calls including warmups are retained. Only measured-call distributions are the warm sample metrics.",
            "First input call is not guaranteed cache-empty: earlier cases can share prefixes, and native startup may warm its model. Actual prompt/cache token counts are retained. L1 has empty conversational history and no tools; full app latency is separate.",
            "S1 uses frozen synthetic WAV files and production file VAD. Its padding/offsets and prompt-agreement CER do not measure real microphone endpoint latency or independently annotated ASR accuracy. Transcript differences are retained, not normalized away after observation.",
            "S1/T1 include the production OVS profile with both ASR and TTS resident, one role active. 4B and perception are stopped. L1 runs only the native 4B service. Native model parameters/cache/thread/power settings unchanged.",
            "TTS first PCM is the first block delivered by iter_bytes(4096), not necessarily first wire byte. Output is PCM/WAV, no audible playback timing.",
            "L1 client first-content includes native input-token preflight and stream setup. Native predicted_per_second is reported by pinned llama.cpp and has a different denominator from completion_tokens/client_total_seconds.",
            "Image PNG decode/JPEG encoding is fixture preparation outside L1 inference timing; each call separately records base64 time. Exact JPEG and request hashes and SSE bytes are preserved. Image semantic correctness is not scored here.",
            "Resource samples include input-first/warmup/measured calls and release checks, with stable native/listener PIDs. 1Hz snapshots can miss short GPU activity; sampled maxima are not exact allocation peaks. New clock anchors permit time alignment, subject to wall-clock adjustments.",
            "One development workload/run, not 5-run optimization statistics, production QoS, long-run stability or complete B0. E1 full application and handoff remain pending.",
        ],
    }
    runtime.write_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
