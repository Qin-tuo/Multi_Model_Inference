#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""S1/T1/L1 through production adapters, frozen inputs and native services.

Speech profile keeps ASR and TTS resident but sends requests to one role at a
time. L1 runs only llama.cpp. All first calls/warmups/errors are retained; no
model parameters, prompt-cache settings, power mode or clocks are adjusted.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_src"))
# Shared native-process identity checks live with the functional probes.
sys.path.insert(0, str(ROOT / "scripts"))
from benchmarks.baseline_support import (  # noqa: E402
    Events,
    Resources,
    environment,
    run_worker,
    validate,
)
from benchmarks.http_observer import Observer  # noqa: E402
from benchmarks.text_metrics import character_errors  # noqa: E402
from deploy import runtime, services  # noqa: E402
from deploy.operations import backend_identity  # noqa: E402

ROLES = ("s1", "t1", "l1-text", "l1-image")


def wait_idle(adapter, seconds=30):
    start = time.monotonic()
    last_error = None
    while time.monotonic() - start < seconds:
        try:
            if adapter.idle():
                return {
                    "seconds": time.monotonic() - start,
                    "last_probe_error": last_error,
                }
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(0.02)
    raise RuntimeError("Native role did not release: " + str(last_error))


def speech_case(role, cfg, model, vad, case, phase, out):
    from app.audio import pcm_wav, wav_pcm
    from app.speech.vad import segment_pcm

    row = {"case": case["id"], "phase": phase, "reference_prompt": case["text"]}
    if role == "s1":
        # Input file I/O is outside VAD/ASR latency, exactly the frozen PCM.
        pcm = wav_pcm(Path(case["path"]).read_bytes(), cfg.stt.max_seconds)
        row["input_seconds"] = len(pcm) / 32000
        start = time.monotonic()
        segments = list(segment_pcm(pcm, cfg.vad, vad))
        row["vad_seconds"] = time.monotonic() - start
        row["segments"] = [
            {
                "start_file_offset": s.start_time,
                "end_file_offset": s.end_time,
                "duration": s.duration,
                "end_reason": s.end_reason,
            }
            for s in segments
        ]
        if not segments:
            raise ValueError("No VAD segment for frozen speech file")
        transcripts = []
        asr_start = time.monotonic()
        for segment in segments:
            value = model.transcribe(segment.audio)
            transcripts.append(value)
            if value.get("error"):
                raise RuntimeError(value["error"])
        row["asr_seconds"] = time.monotonic() - asr_start
        row["total_seconds"] = time.monotonic() - start
        row["transcripts"] = transcripts
        row["text"] = "".join(t["text"] for t in transcripts)
        row["segmented_audio_seconds"] = sum(t["duration"] for t in transcripts)
        row["asr_rtf_original_file"] = row["asr_seconds"] / row["input_seconds"]
        row["asr_rtf_submitted_audio"] = (
            row["asr_seconds"] / row["segmented_audio_seconds"]
        )
        row["prompt_agreement"] = character_errors(case["text"], row["text"])
    else:
        start = time.monotonic()
        value = model.synthesize(case["text"])
        row["call_seconds"] = time.monotonic() - start
        if value.get("error") or value.get("audio") is None:
            raise RuntimeError(value.get("error", "No PCM returned"))
        pcm = value.pop("audio").tobytes()
        row.update(value)
        row["synthesis_rtf"] = row["call_seconds"] / row["duration"]
        row["pcm_bytes"] = len(pcm)
        row["pcm_sha256"] = hashlib.sha256(pcm).hexdigest()
        if phase in {"first", "measured"}:
            path = out / f"{case['id']}-{phase}.wav"
            path.write_bytes(pcm_wav(pcm))
            row["output_file"] = path.name
    return row


def llm_case(model, cfg, case, phase, out, observed):
    row = {
        "case": case["id"],
        "phase": phase,
        "prompt": case["prompt"],
        "image": case.get("image"),
    }
    tick = time.monotonic()
    images = (
        [base64.b64encode(case["jpeg"]).decode("ascii")] if "jpeg" in case else None
    )
    row["image_base64_seconds"] = time.monotonic() - tick if images else None
    row["image_jpeg_sha256"] = case.get("jpeg_sha256")
    observer = Observer() if observed else None
    from contextlib import nullcontext

    first = None
    metadata = {}
    content = []
    started = None
    try:
        with observer.capture() if observer else nullcontext():
            started = time.monotonic()
            for text, meta in model.generate_stream(case["prompt"], images_b64=images):
                metadata.update(meta)
                if text:
                    first = first or time.monotonic()
                    content.append(text)
            ended = time.monotonic()
        row.update(
            total_seconds=ended - started,
            first_content_seconds=first - started if first else None,
            text="".join(content),
            metadata=metadata,
        )
        if not metadata.get("done") or not row["text"]:
            raise RuntimeError("Incomplete model response")
        tokens = metadata.get("usage", {}).get("completion_tokens")
        row["client_completion_tokens_per_total_second"] = (
            tokens / row["total_seconds"] if tokens else None
        )
        if observed:
            rows = observer.export(out, f"{case['id']}-{phase}")
            row["http"] = rows
            if any(r["capture_truncated"] for r in rows):
                raise RuntimeError("Benchmark capture exceeded byte budget")
            preflight = next(r for r in rows if r["path"].endswith("input_tokens"))
            generation = next(r for r in rows if r["path"] == "/v1/chat/completions")
            row["preflight_http_seconds"] = (
                preflight["closed_monotonic"] - preflight["sent_monotonic"]
            )
            row["generation_http_first_content_seconds"] = (
                generation["first_content_received_monotonic"]
                - generation["sent_monotonic"]
            )
            row["native_timings"] = generation["native_timings"]
        return row
    except BaseException as exc:
        row.update(error=repr(exc), text="".join(content), metadata=metadata)
        if observer:
            try:
                row["http"] = observer.export(out, f"{case['id']}-{phase}-error")
            except Exception as trace_error:
                row["trace_error"] = repr(trace_error)
        return row


def worker(role, out):
    protocol, cfg, fixtures = validate()
    names = ["llama"] if role.startswith("l1") else ["asr", "speech"]
    before = {name: backend_identity(name, cfg) for name in names}
    groups = {name: value["stable_pids"] for name, value in before.items()}
    groups["client"] = [os.getpid()]
    resources = Resources(out, groups).start()
    events = Events(out / "events.jsonl")
    report = {
        "role": role,
        "status": "RUNNING",
        "backend_before": before,
        "cases": [],
        "errors": [],
    }
    model = vad = None
    try:
        if role == "s1":
            from app.stt import STT
            from app.speech.vad import SileroVAD

            start = time.monotonic()
            vad = SileroVAD()
            report["vad_load_seconds"] = time.monotonic() - start
            model = STT(**vars(cfg.stt))
        elif role == "t1":
            from app.tts import LocalTTS

            model = LocalTTS(
                **{
                    k: v
                    for k, v in vars(cfg.tts).items()
                    if k not in {"first_chunk_chars", "max_chunk_chars"}
                }
            )
        else:
            from app.llm import LLM

            model = LLM(
                **{
                    k: v
                    for k, v in vars(cfg.llm).items()
                    if k != "system_prompt_no_rag"
                }
            )
        if not model.load():
            raise RuntimeError(model.last_error)
        if role.startswith("l1"):
            import cv2

            cases = [
                c
                for c in json.loads(
                    (ROOT / "benchmarks/corpora/b0-llm-v1.json").read_text()
                )["cases"]
                if c["kind"] == role.removeprefix("l1-")
            ]
            for case in cases:
                if case.get("image"):
                    tick = time.monotonic()
                    image = cv2.imread(
                        str(Path(protocol["input_directory"]) / case["image"])
                    )
                    ok, jpeg = cv2.imencode(
                        ".jpg",
                        image,
                        [cv2.IMWRITE_JPEG_QUALITY, cfg.vision.jpeg_quality],
                    )
                    if not ok:
                        raise RuntimeError("Input JPEG encoding failed")
                    case["jpeg"] = jpeg.tobytes()
                    case["jpeg_sha256"] = hashlib.sha256(case["jpeg"]).hexdigest()
                    case["fixture_decode_encode_seconds"] = time.monotonic() - tick
                    (out / (case["id"] + ".jpg")).write_bytes(case["jpeg"])
        else:
            cases = json.loads(
                (ROOT / "benchmarks/corpora/b0-speech-v1.json").read_text()
            )["cases"]
            for case in cases:
                case["path"] = str(
                    Path(protocol["input_directory"]) / (case["id"] + ".wav")
                )
        for index, case in enumerate(cases):
            print(f"{role}: {index + 1}/{len(cases)} {case['id']}", flush=True)
            item = {
                "id": case["id"],
                "first": None,
                "warmups": [],
                "measurements": [],
                "input": {k: v for k, v in case.items() if k != "jpeg"},
            }
            report["cases"].append(item)
            phases = ["first", *[f"warmup-{n}" for n in range(5)]]
            if role.startswith("l1"):
                phases += (
                    ["plain", "measured"] if index % 2 == 0 else ["measured", "plain"]
                )
            else:
                phases += ["measured"]
            for phase in phases:
                wait_idle(model)
                resources.phase = phase
                tick = time.monotonic()
                try:
                    if role.startswith("l1"):
                        row = llm_case(
                            model, cfg, case, phase, out, phase == "measured"
                        )
                    else:
                        row = speech_case(role, cfg, model, vad, case, phase, out)
                except Exception as exc:
                    row = {
                        "case": case["id"],
                        "phase": phase,
                        "error": repr(exc),
                        "observed_seconds": time.monotonic() - tick,
                    }
                row["native_release"] = wait_idle(model)
                if row.get("error"):
                    report["errors"].append(
                        {"case": case["id"], "phase": phase, "error": row["error"]}
                    )
                events.write(**row)
                if phase == "first":
                    item["first"] = row
                elif phase.startswith("warmup"):
                    item["warmups"].append(row)
                else:
                    item["measurements"].append(row)
            if role.startswith("l1"):
                plain = next(r for r in item["measurements"] if r["phase"] == "plain")
                measured = next(
                    r for r in item["measurements"] if r["phase"] == "measured"
                )
                item["paired_output_equal"] = (
                    plain.get("text") == measured.get("text")
                    and not plain.get("error")
                    and not measured.get("error")
                )
                if "total_seconds" in plain and "total_seconds" in measured:
                    item["signed_observer_difference_seconds"] = (
                        measured["total_seconds"] - plain["total_seconds"]
                    )
            runtime.write_json(out / "result.json", report)
        resources.phase = "resident_after_measurement"
        time.sleep(2)
        report["backend_after"] = {name: backend_identity(name, cfg) for name in names}
        if any(
            report["backend_after"][name]["identity"] != row["identity"]
            for name, row in before.items()
        ):
            report["errors"].append({"error": "Backend changed during measurement"})
    except BaseException as exc:
        report["errors"].append({"error": repr(exc)})
        raise
    finally:
        if model:
            model.unload()
        resources.close()
        events.close()
        report["monitor_errors"] = resources.errors
        report["status"] = (
            "MEASURED"
            if not report["errors"] and not resources.errors
            else "MEASURED_WITH_FAILURES"
        )
        runtime.write_json(out / "result.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=ROLES)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker, args.output)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    _, cfg, _ = validate()
    before = services.state(services.ORDER, cfg)
    if (
        before["status"] != "READY"
        or before["services"]["app"]["health"]["phase"] != "idle"
    ):
        raise RuntimeError("Managed services must be ready and idle")
    environment(out)
    runtime.write_json(out / "effective-config.json", cfg.effective())
    for filename in ("b0-llm-v1.json", "b0-speech-v1.json"):
        (out / filename).write_bytes(
            (ROOT / "benchmarks/corpora" / filename).read_bytes()
        )
    workload = {
        "speech_profile": "ASR and TTS resident together; one role requested, other idle; no 4B/perception",
        "llm_profile": "Only native 4B server; production adapter, no tools/history, current 128-token limit and caches unchanged",
        "per_case": "First input call separately, five warmups, one measured call; L1 additionally one adjacent unobserved call, alternating pair order",
        "timestamps": "File PCM is already available; VAD offsets can include appended silence; no physical endpoint or playback metric",
        "tts_first_pcm": "First nonempty block from production iter_bytes(4096), not necessarily first byte on wire",
        "corpus_sha256": {
            name: runtime.sha256(out / name)
            for name in ("b0-llm-v1.json", "b0-speech-v1.json")
        },
    }
    runtime.write_json(out / "workload.json", workload)
    report = {
        "created_at": runtime.stamp(),
        "status": "RUNNING",
        "before": before,
        "roles": {},
        "workload_sha256": runtime.sha256(out / "workload.json"),
    }
    error = None
    try:
        with services.operation(cfg):
            services.stop(list(services.ORDER), cfg)
            services.start(["asr", "speech"], cfg, 240)
        for index, role in enumerate(ROLES):
            if index == 2:
                with services.operation(cfg):
                    services.stop(["asr", "speech"], cfg)
                    services.start(["llama"], cfg, 240)
            print("Measuring " + role, flush=True)
            folder = out / role
            folder.mkdir()
            with (folder / "process.log").open("w") as log:
                code = run_worker(
                    [
                        sys.executable,
                        __file__,
                        "--worker",
                        role,
                        "--output",
                        str(folder),
                    ],
                    log,
                    2400,
                )
            report["roles"][role] = {
                "exit_code": code,
                "result": str(folder / "result.json"),
            }
            runtime.write_json(out / "report.json", report)
            if code:
                raise RuntimeError("Module worker failed: " + role)
    except BaseException as exc:
        error = exc
        report["error"] = repr(exc)
    finally:
        try:
            with services.operation(cfg):
                report["restored"] = services.start(list(services.ORDER), cfg, 240)
        except BaseException as exc:
            error = error or exc
            report["restore_error"] = repr(exc)
        report["status"] = "FAIL" if error else "COLLECTED_REVIEW_RESULTS"
        report["finished_at"] = runtime.stamp()
        runtime.write_json(out / "report.json", report)
    if error:
        raise error
    print(json.dumps({"status": report["status"], "report": str(out / "report.json")}))


if __name__ == "__main__":
    main()
