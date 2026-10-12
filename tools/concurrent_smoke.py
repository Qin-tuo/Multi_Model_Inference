"""Repeat real file-speech turns while the managed visual services keep running."""
import asyncio
import json
from pathlib import Path
import re
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.voice_file import run_file
from tools.project import DATA, meminfo, stamp, write_json


def normalize(text): return re.sub(r"[\s，。！？,.!?]", "", text)


async def verify_output(path):
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        with path.open("rb") as stream:
            response = await client.post("http://127.0.0.1:8621/v1/audio/transcriptions",
                data={"model": "qwen3-asr"}, files={"file": (path.name, stream, "audio/wav")})
        response.raise_for_status()
        return response.json()["text"]


async def main():
    results, memory_samples = [], []
    stop = asyncio.Event()
    async def sample():
        while not stop.is_set():
            data = meminfo()
            memory_samples.append({"monotonic_ns": time.monotonic_ns(), "available": data["MemAvailable"],
                                   "swap_used": data["SwapTotal"] - data["SwapFree"]})
            await asyncio.sleep(.1)
    sampling = asyncio.create_task(sample())
    try:
        for index in range(5):
            path = DATA / f"fixtures/concurrent-voice-{index}.wav"
            started = time.monotonic_ns()
            result = await run_file(DATA / "fixtures/roundtrip/request-0.wav", path)
            result.update(request_started_ns=started, request_finished_ns=time.monotonic_ns())
            results.append(result)
            print(json.dumps({"case": index, "answer": result["answer"],
                              "latency": result["vad_stopped_to_first_pcm_seconds"]}, ensure_ascii=False), flush=True)
    finally:
        stop.set(); await sampling
    # Verification requests are separate from the measured five-turn workload.
    for result in results:
        result["output_audio_transcript"] = await verify_output(Path(result["output_wav"]))
    checks = {"five_real_file_turns": len(results) == 5,
              "input_recognized": all(normalize(r["transcripts"][0]) == "一加一等于几" for r in results),
              "math_answers": all(normalize(r["answer"]) in {"一加一等于二", "一加一等于2", "2", "二"} for r in results),
              "output_content_readback": all(normalize(r["output_audio_transcript"]) in {"一加一等于二", "一加一等于2", "2", "二"} for r in results),
              "nonempty_audio": all(r["output_audio_seconds"] > 0 for r in results)}
    report = {"created_at": stamp(), "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "five synthetic file-audio turns with visual services running; output readback is outside the measured workload; not physical playback/AEC, held-out quality, or GPU-overlap trace acceptance",
        "checks": checks, "cases": results, "memory": {
            "sample_count": len(memory_samples), "minimum_available_bytes": min(x["available"] for x in memory_samples),
            "maximum_swap_used_bytes": max(x["swap_used"] for x in memory_samples),
            "swap_used_start_bytes": memory_samples[0]["swap_used"], "swap_used_end_bytes": memory_samples[-1]["swap_used"]}}
    write_json(ROOT / "benchmarks/results/concurrent-file-speech.json", report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(asyncio.run(main()))
