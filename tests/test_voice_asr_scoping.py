"""Native voice lifecycle with controlled ASR timing, fixture VAD/model/audio.

These tests deliberately delay a complete STT response across user-turn events.
They do not measure microphone, ASR quality or acoustic interruption latency.
"""
import asyncio
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from pipecat.frames.frames import (InputAudioRawFrame, LLMContextFrame, TranscriptionFrame,
                                  VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)
from app.adapters.llama import LlamaConfig
from app.adapters.speech import create_stt_service
from app.adapters.asr import ASR_GENERATION
from app.memory import Mem0Adapter
from app.records import Records
from app.reminders import ReminderService
from app.voice_assistant import VoiceAssistant
from test_mem0_adapter import actual_sdk
from test_pipecat_pipeline import FrameworkFakeLLM


class InputRecordingLLM(FrameworkFakeLLM):
    def __init__(self):
        super().__init__()
        self.inputs = []

    async def process_frame(self, frame, direction):
        if isinstance(frame, LLMContextFrame):
            self.inputs.append(frame.context.get_messages()[-1]["content"])
        await super().process_frame(frame, direction)


class VoiceASRScopeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.memory = Mem0Adapter(root / "memory", sdk=actual_sdk(root / "memory"), embedding_scope="FIXTURE")
        self.records = Records(root / "records.sqlite")
        self.reminders = ReminderService(root / "reminders.sqlite")
        self.stt = create_stt_service()
        self.entered = [asyncio.Event(), asyncio.Event()]
        self.release = [asyncio.Event(), asyncio.Event()]
        self.requests = 0
        async def transcribe(_audio):
            index = self.requests
            self.requests += 1
            self.entered[index].set()
            await self.release[index].wait()
            return SimpleNamespace(text=["第一段发言", "第二段补充"][index])
        self.stt._transcribe = transcribe
        self.llm = InputRecordingLLM()
        self.voice = VoiceAssistant(LlamaConfig("FIXTURE"), self.memory, self.reminders, self.records,
            output_path=root / "output.wav", stt=self.stt, llm=self.llm)
        self.voice.session.budget.count = AsyncMock(return_value=100)
        self.evidence = patch("app.session.memory_message", return_value=({"role": "user", "content": "FIXTURE []"},
            {"injected_memory_ids": [], "memory_tokens": 3, "retrieved_count": 0, "injected_count": 0}))
        self.evidence.start()
        await self.voice.tts._http.aclose()
        self.voice.tts._http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"\x01\x00" * 480)))
        await self.voice.start(check_services=False)

    async def asyncTearDown(self):
        for event in self.release: event.set()
        await self.voice.close()
        await self.stt._client.close()
        self.evidence.stop()
        self.reminders.close(); self.records.close(); self.memory.close(); self.temp.cleanup()

    async def wait_for(self, predicate, seconds=4):
        async with asyncio.timeout(seconds):
            while not predicate(): await asyncio.sleep(.01)

    async def utterance(self, *, new_turn=True):
        generation = self.voice.generation
        await self.voice.worker.queue_frame(VADUserStartedSpeakingFrame())
        if new_turn: await self.wait_for(lambda: self.voice.generation > generation)
        else: await self.wait_for(lambda: self.stt._user_speaking)
        await self.voice.worker.queue_frame(InputAudioRawFrame(bytes(640), 16000, 1))
        await asyncio.sleep(.05)
        await self.voice.worker.queue_frame(VADUserStoppedSpeakingFrame(stop_secs=.4))

    async def test_first_segment_cannot_end_turn_while_later_audio_is_pending(self):
        await self.utterance()
        await asyncio.wait_for(self.entered[0].wait(), 3)
        await self.utterance(new_turn=False)
        self.release[0].set()
        await asyncio.wait_for(self.entered[1].wait(), 3)
        await asyncio.sleep(.15)
        self.assertEqual(self.llm.inputs, [])
        self.release[1].set()
        await self.wait_for(lambda: bool(self.voice.results))
        self.assertEqual(len(self.llm.inputs), 1)
        self.assertIn("第一段发言", self.voice.results[-1]["transcript"])
        self.assertIn("第二段补充", self.voice.results[-1]["transcript"])

    async def test_delayed_current_result_remains_valid_without_new_user_speech(self):
        await self.utterance()
        await asyncio.wait_for(self.entered[0].wait(), 3)
        await asyncio.sleep(.8)
        self.release[0].set()
        await self.wait_for(lambda: bool(self.voice.results))
        self.assertEqual(self.voice.results[-1]["transcript"], "第一段发言")

    async def test_cancelled_asr_result_cannot_start_tools_and_next_input_recovers(self):
        await self.utterance()
        await asyncio.wait_for(self.entered[0].wait(), 3)
        await self.voice.cancel_input()
        await self.utterance()
        self.release[0].set()
        await asyncio.wait_for(self.entered[1].wait(), 3)
        await asyncio.sleep(.05)
        self.assertEqual(self.llm.inputs, [])
        self.release[1].set()
        await self.wait_for(lambda: bool(self.voice.results))
        self.assertEqual(self.voice.results[0]["transcript"], "第二段补充")
        self.assertNotIn("第一段发言", str(self.records.recent("owner")))
        self.assertGreaterEqual(self.stt.counts["obsolete_results_discarded"], 1)

    async def test_final_consumer_refuses_an_already_queued_obsolete_transcript(self):
        old_generation = self.voice.generation
        await self.voice.cancel_input()
        frame = TranscriptionFrame("旧语音不能写入记忆", "owner", "FIXTURE")
        frame.metadata[ASR_GENERATION] = old_generation
        await self.voice.user_aggregator.queue_frame(frame)
        await asyncio.sleep(.05)
        self.assertEqual(self.llm.inputs, [])
        self.assertEqual(self.stt.counts["obsolete_at_aggregator"], 1)

    async def test_queue_overflow_refuses_the_whole_utterance_without_partial_execution(self):
        await self.utterance()
        await asyncio.wait_for(self.entered[0].wait(), 3)
        await self.utterance(new_turn=False)
        await self.wait_for(lambda: self.stt.status()["queued_segments"] == 1)
        await self.utterance(new_turn=False)
        await self.wait_for(lambda: "ASR_PENDING_QUEUE_FULL" in self.voice.errors)
        self.assertLessEqual(self.stt.status()["active_and_pending_segments"], 2)
        self.release[0].set()
        await self.wait_for(lambda: not self.stt.pending)
        self.assertEqual(self.llm.inputs, [])
        self.assertEqual(self.records.recent("owner"), [])

    async def test_capture_buffer_limit_is_finite_and_does_not_submit_truncated_audio(self):
        await self.utterance()
        await asyncio.wait_for(self.entered[0].wait(), 3)
        self.stt._audio_buffer = bytearray(self.stt.max_pcm_bytes)
        await self.voice.worker.queue_frame(InputAudioRawFrame(bytes(640), 16000, 1))
        await self.wait_for(lambda: "ASR_AUDIO_BUFFER_LIMIT" in self.voice.errors)
        self.assertEqual(len(self.stt._audio_buffer), 0)
        self.assertEqual(self.requests, 1)

    async def test_asr_deadline_fails_without_partial_execution_and_recovers(self):
        self.stt.request_timeout = .1
        await self.utterance()
        await self.wait_for(lambda: "ASR_REQUEST_TIMEOUT" in self.voice.errors)
        if self.voice.input_cancel_task: await self.voice.input_cancel_task
        self.assertEqual(self.llm.inputs, [])
        self.stt.request_timeout = 12
        self.release[1].set()
        await self.utterance()
        await self.wait_for(lambda: bool(self.voice.results))
        self.assertEqual(self.voice.results[0]["transcript"], "第二段补充")

    async def test_cancel_during_capture_ends_occupancy_and_allows_new_input(self):
        await self.voice.worker.queue_frame(VADUserStartedSpeakingFrame())
        await self.wait_for(lambda: self.voice.user_speaking)
        await self.voice.cancel_input()
        await self.wait_for(lambda: not self.voice.occupied())
        self.assertEqual(self.requests, 0)
        self.release[0].set()
        await self.utterance()
        await self.wait_for(lambda: bool(self.voice.results))
        self.assertEqual(len(self.voice.results), 1)

    async def test_delayed_or_repeated_turn_events_do_not_admit_a_second_request(self):
        self.release[0].set()
        await self.utterance()
        await self.wait_for(lambda: bool(self.voice.results))
        for text in ("第一段发言", "请记住别的旧内容"):
            await self.voice.user_aggregator._call_event_handler("on_user_turn_stopped", self.voice.stop_strategy,
                SimpleNamespace(content=text, timestamp="FIXTURE", user_id="owner"))
        await asyncio.sleep(.05)
        self.assertEqual(len(self.llm.inputs), 1)
        self.assertGreaterEqual(self.stt.counts["obsolete_or_duplicate_turn_events"], 2)

    async def test_queue_wait_counts_toward_the_segment_deadline(self):
        self.stt.request_timeout = 2
        await self.utterance()
        await asyncio.wait_for(self.entered[0].wait(), 3)
        self.stt.request_timeout = .05
        await self.utterance(new_turn=False)
        await self.wait_for(lambda: self.stt.status()["queued_segments"] == 1)
        await asyncio.sleep(.1)
        self.release[0].set()
        await self.wait_for(lambda: "ASR_REQUEST_TIMEOUT" in self.voice.errors)
        self.assertEqual(self.requests, 1)
        self.assertEqual(self.stt.counts["expired_before_request"], 1)
        self.assertEqual(self.llm.inputs, [])


if __name__ == "__main__": unittest.main()
