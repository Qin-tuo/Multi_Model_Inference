"""Native speech output ordering; TTS HTTP and audio device are fixtures."""
import asyncio
from pathlib import Path
import tempfile
import unittest
import wave

import httpx
from pipecat.frames.frames import EndFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner
from app.adapters.speech import OVSTTSService
from app.notifications import NotificationFrameGate, PipecatNotificationSink
from app.reminders import ReminderService
from app.speech_output import ConversationSpeechGate, ConversationSpeechSink, SPEECH_TOKEN, WavAssistantOutput


class DelayedOutput(WavAssistantOutput):
    async def drain_speech_audio(self):
        self.at_drain.set()
        await self.allow_drain.wait()
        await super().drain_speech_audio()


class SpeechOutputTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.reminders = ReminderService(root / "reminders.sqlite")
        self.sink = ConversationSpeechSink(timeout=5)
        self.notices = PipecatNotificationSink(self.reminders)
        self.tts = OVSTTSService(speech_sink=self.sink, notification_sink=self.notices)
        await self.tts._http.aclose()
        self.http_status = 200
        self.tts._http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(self.http_status, content=b"\x01\x00" * 480)))
        self.path = root / "output.wav"
        self.output = DelayedOutput(self.path, TransportParams(audio_out_enabled=True,
            audio_out_sample_rate=16000, audio_out_channels=1, audio_out_10ms_chunks=2,
            audio_out_end_silence_secs=0), self.sink, self.notices)
        self.output.at_drain = asyncio.Event(); self.output.allow_drain = asyncio.Event()
        self.worker = PipelineWorker(Pipeline([NotificationFrameGate(self.notices), ConversationSpeechGate(self.sink),
            self.tts, self.output]), params=PipelineParams(audio_out_sample_rate=16000), idle_timeout_secs=None, enable_rtvi=False)
        self.sink.bind_worker(self.worker); self.notices.worker = self.worker
        self.runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
        await self.runner.add_workers(self.worker)
        self.running = asyncio.create_task(self.runner.run())

    async def asyncTearDown(self):
        self.output.allow_drain.set()
        await self.worker.queue_frame(EndFrame())
        await asyncio.wait_for(self.running, 10)
        self.reminders.close(); self.temp.cleanup()

    async def test_completion_waits_for_output_flush_and_records_fixture_scope(self):
        task = asyncio.create_task(self.sink.deliver("fixture", turn_id="one"))
        await asyncio.wait_for(self.output.at_drain.wait(), 4)
        self.assertFalse(task.done())
        self.output.allow_drain.set()
        receipt = await task
        self.assertEqual(receipt["channel"], "fixture")
        self.assertEqual(receipt["completion"], "wav_file_flushed")
        with wave.open(str(self.path)) as wav:
            self.assertEqual(wav.readframes(480), b"\x01\x00" * 480)

    async def test_interrupt_during_drain_rejects_receipt_and_late_old_pcm(self):
        task = asyncio.create_task(self.sink.deliver("fixture", turn_id="old"))
        await asyncio.wait_for(self.output.at_drain.wait(), 4)
        self.sink.interrupt("fixture_interrupt")
        await self.sink.interrupt_pipeline()
        self.output.allow_drain.set()
        with self.assertRaises(asyncio.CancelledError): await task
        samples = self.output.samples_written
        old = TTSAudioRawFrame(bytes(640), 16000, 1, context_id="obsolete")
        old.metadata[SPEECH_TOKEN] = "old"
        await self.worker.queue_frame(old)
        await asyncio.sleep(.05)
        self.assertEqual(self.output.samples_written, samples)
        receipt = await self.sink.deliver("next", turn_id="new")
        self.assertGreater(receipt["bytes_written"], 0)

    async def test_tts_failure_does_not_confirm_and_next_delivery_recovers(self):
        self.http_status = 503
        with self.assertRaises(RuntimeError): await self.sink.deliver("failure", turn_id="bad")
        self.assertEqual(self.output.samples_written, 0)
        self.http_status = 200; self.output.allow_drain.set()
        receipt = await self.sink.deliver("recovered", turn_id="good")
        self.assertGreater(receipt["bytes_written"], 0)


if __name__ == "__main__": unittest.main()
