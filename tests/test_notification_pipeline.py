"""Real Pipecat output ordering and SQLite receipts; HTTP/audio are fixtures."""
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

import httpx
from sqlalchemy import select
from pipecat.frames.frames import EndFrame, InterruptionFrame, TTSAudioRawFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner

from app.adapters.speech import OVSTTSService
from app.notifications import (NotificationFrameGate, NotificationOutputMixin,
                               NotificationPump, PipecatNotificationSink, NOTICE_TOKEN)
from app.reminders import ReminderService, _NOTIFICATIONS


class FixtureDevice(BaseOutputTransport):
    def __init__(self, sink):
        super().__init__(TransportParams(audio_out_enabled=True, audio_out_sample_rate=16000,
            audio_out_channels=1, audio_out_10ms_chunks=2, audio_out_end_silence_secs=0))
        self.bind_notifications(sink)
        self.written = []
        self.failed_write = False
        self.at_drain = asyncio.Event()
        self.allow_drain = asyncio.Event()

    async def start(self, frame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame):
        if self.failed_write: return False
        self.written.append(frame.audio)
        await asyncio.sleep(0)
        return True

    async def drain_notification_audio(self):
        self.at_drain.set()
        await self.allow_drain.wait()


class ReceiptOutput(NotificationOutputMixin, FixtureDevice):
    notification_channel = "fixture"
    drain_notification_audio = FixtureDevice.drain_notification_audio


class NotificationPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = ReminderService(Path(self.temp.name) / "reminders.sqlite", retry_delay_seconds=0)
        self.occupied = False
        self.sink = PipecatNotificationSink(self.service, occupied=lambda: self.occupied, timeout_seconds=5)
        self.tts = OVSTTSService(notification_sink=self.sink)
        await self.tts._http.aclose()
        self.http_status = 200
        self.tts._http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(self.http_status, content=b"\x01\x00" * 480)))
        self.output = ReceiptOutput(self.sink)
        self.worker = PipelineWorker(Pipeline([NotificationFrameGate(self.sink), self.tts, self.output]),
            params=PipelineParams(audio_out_sample_rate=16000), idle_timeout_secs=None, enable_rtvi=False)
        self.sink.worker = self.worker
        self.runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
        await self.runner.add_workers(self.worker)
        self.runner_task = asyncio.create_task(self.runner.run())
        self.pump = NotificationPump(self.service, "owner", self.sink, occupied=lambda: self.occupied, poll_seconds=.02)
        self.pump_task = None

    async def asyncTearDown(self):
        self.output.allow_drain.set()
        await self.pump.stop()
        if self.pump_task:
            await asyncio.wait_for(self.pump_task, 5)
        await self.worker.queue_frame(EndFrame())
        await asyncio.wait_for(self.runner_task, 10)
        self.service.close()
        self.temp.cleanup()

    def schedule(self):
        return self.service.create("owner", "提醒输出合同测试", datetime.now(timezone.utc) + timedelta(seconds=.05), request_id="notice")

    def row(self):
        with self.service.engine.connect() as connection:
            row = connection.execute(select(_NOTIFICATIONS)).mappings().first()
            return dict(row) if row else None

    async def wait_for(self, predicate, seconds=4):
        async with asyncio.timeout(seconds):
            while not predicate(): await asyncio.sleep(.01)

    def start_consumer(self):
        self.pump_task = asyncio.create_task(self.pump.run())

    async def test_ack_waits_for_native_audio_queue_and_output_drain(self):
        self.schedule(); self.start_consumer()
        await asyncio.wait_for(self.output.at_drain.wait(), 4)
        actual = b"".join(self.output.written)
        expected = b"\x01\x00" * 480
        self.assertGreaterEqual(len(actual), len(expected))
        self.assertEqual(actual[:len(expected)], expected)
        self.assertEqual(self.row()["status"], "playing")
        self.assertIsNone(self.row()["delivered_utc"])
        self.output.allow_drain.set()
        await self.wait_for(lambda: self.row()["status"] == "fixture_delivered")
        self.assertIsNone(self.row()["played_utc"])
        self.assertEqual(self.row()["delivery_attempts"], 1)

    async def test_cancel_while_drain_pending_rejects_late_receipt(self):
        reminder = self.schedule(); self.start_consumer()
        await asyncio.wait_for(self.output.at_drain.wait(), 4)
        self.service.cancel("owner", reminder["reminder_id"], request_id="cancel")
        self.output.allow_drain.set()
        await self.wait_for(lambda: self.sink.current is None)
        self.assertEqual(self.row()["status"], "cancelled")
        self.assertIsNone(self.row()["delivered_utc"])

    async def test_interruption_never_confirms_partial_output(self):
        self.schedule(); self.start_consumer()
        await asyncio.wait_for(self.output.at_drain.wait(), 4)
        await self.worker.queue_frame(InterruptionFrame())
        await self.wait_for(lambda: self.row()["status"] == "uncertain")
        self.output.allow_drain.set()
        self.assertIsNone(self.row()["delivered_utc"])
        self.assertIsNone(self.service.claim_next("owner"))
        count = len(self.output.written)
        stale = TTSAudioRawFrame(b"\x01\x00" * 320, 16000, 1, context_id="obsolete-context")
        stale.metadata[NOTICE_TOKEN] = self.row()["attempt_id"]
        await self.worker.queue_frame(stale)
        await asyncio.sleep(.08)
        self.assertEqual(len(self.output.written), count)
        await self.worker.queue_frame(TTSSpeakFrame("后续普通输出"))
        await self.wait_for(lambda: len(self.output.written) > count)

    async def test_failed_device_write_is_not_played(self):
        self.output.failed_write = True
        self.schedule(); self.start_consumer()
        await self.wait_for(lambda: self.row() is not None and self.row()["status"] == "uncertain")
        self.assertFalse(self.output.written)
        self.assertFalse(self.output.at_drain.is_set())
        self.assertIsNone(self.row()["delivered_utc"])

    async def test_mute_and_foreground_hold_until_released(self):
        self.service.set_muted("owner", True)
        self.occupied = True
        self.schedule(); self.start_consumer()
        await self.wait_for(lambda: self.row() is not None)
        await asyncio.sleep(.08)
        self.assertEqual(self.row()["delivery_attempts"], 0)
        self.service.set_muted("owner", False)
        await asyncio.sleep(.08)
        self.assertEqual(self.row()["delivery_attempts"], 0)
        self.occupied = False
        self.output.allow_drain.set()
        await self.wait_for(lambda: self.row()["status"] == "fixture_delivered")

    async def test_tts_failure_has_no_audio_receipt_and_retries_are_finite(self):
        self.http_status = 503
        self.schedule(); self.start_consumer()
        try:
            await self.wait_for(lambda: self.row() is not None and self.row()["status"] == "failed")
        except TimeoutError:
            self.fail(f"Failed notification did not reach its retry limit: {self.row()}")
        self.assertEqual(self.row()["failure_count"], 3)
        self.assertFalse(self.output.written)
        self.assertIsNone(self.row()["delivered_utc"])
        self.http_status = 200
        await self.worker.queue_frame(TTSSpeakFrame("故障恢复后的普通输出"))
        await self.wait_for(lambda: bool(self.output.written))


if __name__ == "__main__": unittest.main()
