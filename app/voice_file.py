"""Real file-audio → Silero → STT → Pipecat context/LLM → TTS → WAV probe.

This uses Pipecat's native turn strategies and services. Saved PCM is not
physical speaker playback, and this probe makes no acoustic interruption claim.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time
import wave

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import EndFrame, ErrorFrame, InputAudioRawFrame, OutputAudioRawFrame, TranscriptionFrame, VADUserStoppedSpeakingFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response import LLMFullResponseAggregator
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from app.adapters.llama import LlamaConfig, create_llama_service
from app.adapters.speech import OVSTTSService, create_stt_service, validate_speech_capabilities
from app.pipecat_bot import PROJECT_ROOT, SYSTEM_MESSAGE


class EvidenceTap(FrameProcessor):
    def __init__(self, evidence, *, audio=False):
        super().__init__()
        self.evidence = evidence
        self.audio = audio

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            self.evidence["transcripts"].append(frame.text)
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self.evidence.setdefault("vad_stopped_at", time.monotonic())
        if self.audio and isinstance(frame, OutputAudioRawFrame):
            if frame.sample_rate != 16000 or frame.num_channels != 1: raise ValueError("Unexpected output PCM format")
            self.evidence.setdefault("first_pcm_at", time.monotonic())
            if len(self.evidence["pcm"]) + len(frame.audio) > 10*1024*1024: raise ValueError("Audio output bound exceeded")
            self.evidence["pcm"].extend(frame.audio)
        await self.push_frame(frame, direction)


async def run_file(input_path: Path, output_path: Path):
    await validate_speech_capabilities()
    with wave.open(str(input_path), "rb") as wav:
        if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth()) != (16000,1,2):
            raise ValueError("Input must be PCM16 mono WAV at 16 kHz")
        if wav.getnframes() > 16000*30: raise ValueError("File utterance must be at most 30 seconds")
        pcm = wav.readframes(wav.getnframes())
    evidence = {"transcripts":[],"pcm":bytearray()}
    stt = create_stt_service()
    context = LLMContext(messages=[{"role":"system","content":SYSTEM_MESSAGE}])
    aggregators = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(sample_rate=16000,params=VADParams(stop_secs=0.4)),
        user_turn_strategies=UserTurnStrategies(stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.1)]),
    ))
    llm = create_llama_service(LlamaConfig.from_manifest(PROJECT_ROOT / "models/manifest.json"))
    responses = LLMFullResponseAggregator()
    tts = OVSTTSService()
    pipeline = Pipeline([stt,EvidenceTap(evidence),aggregators.user(),llm,responses,tts,
                         EvidenceTap(evidence,audio=True),aggregators.assistant()])
    worker = PipelineWorker(pipeline, params=PipelineParams(audio_in_sample_rate=16000,audio_out_sample_rate=16000,
                                                          enable_metrics=True,enable_usage_metrics=True),
                            idle_timeout_secs=None,enable_rtvi=False)
    errors = []
    @responses.event_handler("on_completion")
    async def completion(_aggregator,text,completed):
        if completed and text.strip():
            evidence["answer"] = text
            await worker.queue_frame(EndFrame())
    @worker.event_handler("on_pipeline_error")
    async def error(_worker,frame: ErrorFrame):
        errors.append(frame.error)
        await worker.queue_frame(EndFrame())
    runner = WorkerRunner(handle_sigint=False,handle_sigterm=False)
    await runner.add_workers(worker)
    async def feed():
        # Real Silero VAD sees the PCM; silence is padding, not synthetic VAD events.
        audio = bytes(16000//2*2) + pcm + bytes(16000*2)
        await asyncio.sleep(0.1)
        for start in range(0,len(audio),640):
            if worker.has_finished(): break
            await worker.queue_frame(InputAudioRawFrame(audio[start:start+640],16000,1))
            await asyncio.sleep(0.02)
    started = time.monotonic()
    async with asyncio.timeout(60):
        await asyncio.gather(runner.run(),feed())
    if errors: raise RuntimeError("; ".join(errors))
    if not evidence["pcm"] or not evidence.get("answer"): raise RuntimeError("Pipeline produced no complete speech output")
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(output_path),"wb") as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(evidence["pcm"])
    return {"status":"PASS","scope":"paced file-audio pipeline with real Silero/STT/LLM/TTS; physical playback/AEC not tested",
            "framework":"pipecat-ai 1.12.0","transcripts":evidence["transcripts"],"answer":evidence["answer"],
            "output_audio_seconds":len(evidence["pcm"])/32000,"total_seconds":time.monotonic()-started,
            "vad_stopped_to_first_pcm_seconds":(evidence["first_pcm_at"]-evidence["vad_stopped_at"])
                if "vad_stopped_at" in evidence else None,"output_wav":str(output_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output-wav",required=True,type=Path)
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    result=asyncio.run(run_file(args.input,args.output_wav))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
