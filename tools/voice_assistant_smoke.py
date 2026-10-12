"""Real persistent speech/memory/tool/image/notification integration on synthetic input."""
import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import time
import uuid
import wave

import httpx
from loguru import logger
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.adapters.llama import LlamaConfig
from app.memory import Mem0Adapter
from app.records import Records
from app.reminders import ReminderService, _NOTIFICATIONS
from app.vision_frames import VisionFrameStore
from app.voice_assistant import VoiceAssistant
from tools.project import CACHE, DATA, sha256, stamp, write_json


async def synthesize(text, path):
    async with httpx.AsyncClient(timeout=40, trust_env=False) as client:
        response = await client.post("http://127.0.0.1:8621/v1/audio/speech", json={
            "model": "matcha-icefall-zh-en", "voice": "0", "input": text, "response_format": "pcm"})
        response.raise_for_status()
        # OVS's streaming WAV header declares an unknown length. A finite
        # fixture needs a finalized RIFF header, so wrap its actual PCM bytes.
        if not response.content or len(response.content) % 2: raise ValueError("Invalid synthetic PCM fixture")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
            wav.writeframes(response.content)


async def readback(path):
    async with httpx.AsyncClient(timeout=40, trust_env=False) as client:
        with path.open("rb") as stream:
            response = await client.post("http://127.0.0.1:8621/v1/audio/transcriptions",
                data={"model": "qwen3-asr"}, files={"file": (path.name, stream, "audio/wav")})
        response.raise_for_status()
        return response.json()["text"]


async def main():
    logger.remove(); logger.add(sys.stderr, level="WARNING")
    artifacts = DATA / "fixtures" / ("voice-assistant-" + uuid.uuid4().hex)
    artifacts.mkdir(parents=True)
    prompts = ["请记住，我喜欢茉莉花茶。", "我喜欢喝什么茶？", "现在几点？",
               "请看画面，主要是什么颜色？", "十秒钟后提醒我喝水。", "一加一等于几？"]
    for i, prompt in enumerate(prompts): await synthesize(prompt, artifacts / f"input-{i}.wav")
    frames = VisionFrameStore()
    stop = asyncio.Event()
    async def feed_images():
        while not stop.is_set():
            now = time.monotonic_ns()
            frames.add(source_time_ns=now, capture_time_ns=now, arrival_time_ns=now, width=640, height=480,
                rgb_step=1920, rgb_encoding="rgb8", rgb=bytes([0, 0, 255]) * (640 * 480),
                depth_step=1280, depth=bytes(640 * 480 * 2), optical_frame="SYNTHETIC_BLUE_RGB_FIXTURE")
            await asyncio.sleep(.03)
    feeder = asyncio.create_task(feed_images())
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="assistant-voice-state-") as temporary, ExitStack() as stack:
            data = Path(temporary)
            memory = stack.enter_context(Mem0Adapter(data / "memory", bge_path=CACHE / "models/bge-small-zh-v1.5-onnx", embedding_backend="onnx"))
            reminders = stack.enter_context(ReminderService(data / "reminders.sqlite"))
            records = stack.enter_context(Records(data / "records.sqlite"))
            voice = VoiceAssistant(LlamaConfig.from_manifest(ROOT / "models/manifest.json"), memory, reminders, records,
                vision_frames=frames, output_path=artifacts / "answers-and-reminders.wav")
            try:
                await voice.start()
                original_worker = voice.session.bot.worker
                for i, prompt in enumerate(prompts):
                    result = await voice.run_file(artifacts / f"input-{i}.wav")
                    result.update(expected_input=prompt, input_sha256=sha256(artifacts / f"input-{i}.wav"))
                    results.append(result)
                    print(json.dumps({"case": i, "transcript": result["transcript"], "answer": result["response"],
                        "receipt": result["speech_receipt"]["completion"]}, ensure_ascii=False), flush=True)
                deadline = time.monotonic() + 25
                while not voice.notifications.results:
                    if time.monotonic() > deadline: raise RuntimeError("The real scheduled reminder was not delivered")
                    await asyncio.sleep(.1)
                with reminders.engine.connect() as db:
                    notices = [dict(row) for row in db.execute(select(_NOTIFICATIONS)).mappings()]
                checks = {
                    "one_persistent_model_worker": original_worker is voice.session.bot.worker,
                    "six_complete_audio_outputs": len(results) == 6 and all(r["speech_receipt"]["bytes_written"] > 0 for r in results),
                    "memory_written_by_spoken_tool": any(c["tool"] == "remember" and c["outcome"]["ok"] for c in results[0]["calls"]),
                    "spoken_memory_recall": "茉莉" in results[1]["response"],
                    "spoken_time_tool": any(c["tool"] == "get_time" and c["outcome"]["ok"] for c in results[2]["calls"]),
                    "spoken_image_question": results[3]["image"] is not None and "蓝" in results[3]["response"],
                    "spoken_reminder_created": any(c["tool"] == "create_reminder" and c["outcome"]["ok"] for c in results[4]["calls"]),
                    "following_turn_works": "2" in results[5]["response"] or "二" in results[5]["response"],
                    "reminder_confirmed_as_file_not_played": len(notices) == 1 and notices[0]["status"] == "fixture_delivered" and notices[0]["played_utc"] is None,
                    "history_only_after_file_receipts": len(records.recent("owner")) == 6 and all(r["output_channel"] == "fixture" for r in records.recent("owner")),
                    "native_budget_matches_usage": all(r["prompt_tokens"] == [x["prompt_tokens"] for x in r["model_usage"]] for r in results),
                    "no_pipeline_errors": not voice.errors,
                }
                notification_receipts = list(voice.notifications.results)
            finally: await voice.close()
    finally:
        stop.set(); await feeder
    # Read back each known answer from its actual ordered file-output interval.
    with wave.open(str(artifacts / "answers-and-reminders.wav")) as source:
        for i, result in enumerate(results):
            receipt = result["speech_receipt"]
            source.setpos(receipt["first_sample"])
            pcm = source.readframes(receipt["bytes_written"] // 2)
            path = artifacts / f"answer-{i}.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000); wav.writeframes(pcm)
            result["audio_readback"] = await readback(path)
            result["output_audio_sha256"] = sha256(path)
    checks["audio_readback_has_memory_and_color"] = "茉莉" in results[1]["audio_readback"] and "蓝" in results[3]["audio_readback"]
    checks["all_audio_readbacks_nonempty"] = all(r["audio_readback"].strip() for r in results)
    report = {"created_at": stamp(), "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "scope": "six synthetic file utterances through real Silero/ASR/BGE/Mem0/llama/TTS and native output queue; synthetic RGB visual input; real APScheduler reminder delivered to WAV, not speaker; no physical AEC or acoustic acceptance",
        "artifacts": str(artifacts), "cases": results, "notification_receipts": notification_receipts,
        "notification_statuses": [{k: row[k] for k in ("status", "output_channel", "played_utc", "delivery_attempts")} for row in notices]}
    write_json(ROOT / "benchmarks/results/voice-assistant-smoke.json", report)
    print(json.dumps({"status": report["status"], "checks": checks}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(asyncio.run(main()))
