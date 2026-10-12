"""Real ASR/LLM/TTS with injected response delays; no simulated transcripts.

Synthetic spoken inputs exercise segmented-input completeness and cancellation.
No microphone, acoustic timing, GPU cancellation or broad quality claim is made.
"""
import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import uuid
import wave

from loguru import logger
from pipecat.frames.frames import InputAudioRawFrame

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.adapters.asr import INPUT_EPOCH
from app.adapters.llama import LlamaConfig
from app.memory import Mem0Adapter
from app.records import Records
from app.reminders import ReminderService
from app.voice_assistant import VoiceAssistant
from tools.voice_assistant_smoke import synthesize, readback
from tools.project import CACHE, DATA, stamp, write_json


async def wait_for(predicate, seconds=15):
    async with asyncio.timeout(seconds):
        while not predicate(): await asyncio.sleep(.02)


async def feed(voice, path):
    with wave.open(str(path)) as wav: pcm = wav.readframes(wav.getnframes())
    audio = bytes(9600) + pcm + bytes(22400)
    epoch = voice.input_epoch
    for offset in range(0, len(audio), 640):
        frame = InputAudioRawFrame(audio[offset:offset+640], 16000, 1)
        frame.metadata[INPUT_EPOCH] = epoch
        await voice.worker.queue_frame(frame)
        await asyncio.sleep(.02)


def delayed_actual_asr(voice):
    actual = voice.stt._transcribe
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    transcripts = []
    async def delay(audio):
        response = await actual(audio)
        index = len(transcripts)
        transcripts.append(response.text)
        if index >= 2: raise RuntimeError("Unexpected extra speech segment")
        entered[index].set()
        await release[index].wait()
        return response
    voice.stt._transcribe = delay
    return entered, release, transcripts


async def main():
    logger.remove(); logger.add(sys.stderr, level="WARNING")
    artifacts = DATA / "fixtures" / ("asr-lifecycle-" + uuid.uuid4().hex)
    artifacts.mkdir(parents=True)
    for name, text in [("part1", "请记住。"), ("part2", "我喜欢茉莉花茶。"),
                       ("cancelled", "请记住，我喜欢薄荷茶。"), ("current", "一加一等于几？")]:
        await synthesize(text, artifacts / f"{name}.wav")
    reports = []
    with tempfile.TemporaryDirectory(prefix="assistant-asr-lifecycle-") as temporary, ExitStack() as stack:
        data = Path(temporary)
        memory = stack.enter_context(Mem0Adapter(data / "memory", bge_path=CACHE / "models/bge-small-zh-v1.5-onnx", embedding_backend="onnx"))
        reminders = stack.enter_context(ReminderService(data / "reminders.sqlite"))
        records = stack.enter_context(Records(data / "records.sqlite"))
        config = LlamaConfig.from_manifest(ROOT / "models/manifest.json")
        for phase in ("paused_utterance", "cancelled_callback"):
            voice = VoiceAssistant(config, memory, reminders, records, output_path=artifacts / f"{phase}.wav")
            entered, release, transcripts = delayed_actual_asr(voice)
            try:
                await voice.start()
                await feed(voice, artifacts / ("part1.wav" if phase == "paused_utterance" else "cancelled.wav"))
                await asyncio.wait_for(entered[0].wait(), 10)
                if phase == "cancelled_callback": await voice.cancel_input("injected_cancel_before_asr_callback")
                await feed(voice, artifacts / ("part2.wav" if phase == "paused_utterance" else "current.wav"))
                await wait_for(lambda: voice.stt.status()["queued_segments"] == 1)
                release[0].set()
                await asyncio.wait_for(entered[1].wait(), 10)
                await asyncio.sleep(.2)
                no_early_request = voice.answer_task is None and not voice.session.busy.locked() and not voice.session.budget.measurements and not voice.results
                release[1].set()
                await wait_for(lambda: len(voice.results) == 1, seconds=30)
                result = voice.results[0]
                saved = memory.list_memories("owner")
                checks = {"no_request_before_current_asr_complete": no_early_request,
                    "one_completed_turn": len(voice.results) == 1,
                    "audio_receipt": result["speech_receipt"]["bytes_written"] > 0,
                    "no_pipeline_errors": not voice.errors,
                    "bounded_segments_drained": not voice.stt.pending and voice.stt.status()["queued_segments"] == 0}
                if phase == "paused_utterance":
                    checks.update(full_transcript="记住" in result["transcript"] and "茉莉" in result["transcript"],
                        actual_memory_tool=any(c["tool"] == "remember" and c["outcome"]["ok"] for c in result["calls"]),
                        complete_fact_saved="茉莉" in json.dumps(saved, ensure_ascii=False))
                else:
                    checks.update(cancelled_preference_absent="薄荷" not in json.dumps(saved, ensure_ascii=False),
                        cancelled_text_absent="薄荷" not in json.dumps(records.recent("owner"), ensure_ascii=False),
                        stale_asr_discarded=voice.stt.counts["obsolete_results_discarded"] >= 1,
                        current_question_only="一加一" in result["transcript"] and "薄荷" not in result["transcript"],
                        next_answer_correct=any(x in result["response"] for x in ("2", "二")))
                reports.append({"phase": phase, "checks": checks, "actual_asr_transcripts": transcripts,
                    "result": result, "asr_state": voice.stt.status()})
                print(json.dumps({"phase": phase, "checks": checks}, ensure_ascii=False), flush=True)
            finally:
                for event in release: event.set()
                await voice.close()
            reports[-1]["audio_readback"] = await readback(artifacts / f"{phase}.wav")
    report = {"created_at": stamp(), "status": "PASS" if all(all(r["checks"].values()) for r in reports) else "FAIL",
        "scope": "two injected late-ASR-callback cases with real Silero/ASR/LLM/BGE/Mem0/TTS and WAV output; source audio is synthetic; no acoustic, physical-device, broad quality or GPU release acceptance",
        "artifacts": str(artifacts), "cases": reports}
    write_json(ROOT / "benchmarks/results/voice-asr-lifecycle-smoke.json", report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(asyncio.run(main()))
