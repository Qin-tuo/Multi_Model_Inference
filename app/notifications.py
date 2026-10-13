"""Deliver APScheduler outbox entries through existing text/Pipecat outputs.

The scheduler owns due times. This bounded consumer admits one notice at a
time, and the chosen output adapter must confirm completion of that attempt.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Callable

from pipecat.frames.frames import (CancelFrame, ErrorFrame, InterruptionFrame,
    OutputTransportMessageFrame, TTSSpeakFrame)
from pipecat.processors.frame_processor import FrameProcessor

NOTICE_TOKEN = "assistant_notification_attempt"
NOTICE_MARKER = "assistant_notification_marker"


@dataclass
class DeliveryReceipt:
    confirmed: bool
    evidence: str


class NotificationPump:
    def __init__(self, service, user_id, sink, *, occupied: Callable[[], bool], poll_seconds=.2):
        if not .01 <= poll_seconds <= 5: raise ValueError("Notification poll interval must be .01..5 seconds")
        self.service, self.user_id, self.sink = service, user_id, sink
        self.occupied, self.poll_seconds = occupied, poll_seconds
        self.stop_event = asyncio.Event()
        self.active = None
        self.results = []
        self.before_claim = None

    async def run(self):
        while not self.stop_event.is_set():
            if self.before_claim:
                await asyncio.to_thread(self.before_claim)
            notice = await asyncio.to_thread(self.service.claim_next, self.user_id, occupied=self.occupied())
            if notice:
                if self.stop_event.is_set():
                    self.service.release_notification(notice["id"], attempt_id=notice["attempt_id"])
                    break
                self.active = notice
                try:
                    # A foreground turn may have begun while the DB claim ran.
                    if self.occupied() or self.service.is_muted(self.user_id):
                        self.service.release_notification(notice["id"], attempt_id=notice["attempt_id"])
                    else:
                        receipt = await self.sink.deliver(notice)
                        if receipt.confirmed:
                            acknowledged = await asyncio.to_thread(self.service.acknowledge_delivery,
                                notice["id"], attempt_id=notice["attempt_id"], evidence=receipt.evidence)
                            self.results.append({"notification_id": notice["id"], "confirmed": acknowledged,
                                                 "evidence": receipt.evidence})
                            if not acknowledged:
                                await asyncio.to_thread(self.service.fail_output, notice['id'],
                                    attempt_id=notice['attempt_id'], reason='completion_no_longer_valid')
                        else:
                            await asyncio.to_thread(self.service.fail_output, notice["id"],
                                attempt_id=notice["attempt_id"], reason=receipt.evidence)
                except asyncio.CancelledError:
                    await self.sink.interrupt("consumer_cancelled", signal_pipeline=True)
                    await asyncio.to_thread(self.service.fail_output, notice["id"],
                        attempt_id=notice["attempt_id"], reason="consumer_cancelled")
                    raise
                except Exception as error:
                    await asyncio.to_thread(self.service.fail_output, notice["id"],
                        attempt_id=notice["attempt_id"], reason=type(error).__name__)
                finally:
                    self.active = None
                    self.results = self.results[-100:]
            try: await asyncio.wait_for(self.stop_event.wait(), self.poll_seconds)
            except TimeoutError: pass

    async def stop(self):
        self.stop_event.set()
        await self.sink.interrupt("consumer_stopped", signal_pipeline=True)


class TextNotificationSink:
    def __init__(self, service, emit, *, occupied=lambda: False):
        self.service, self.emit, self.occupied = service, emit, occupied

    async def deliver(self, notice):
        if self.occupied() or self.service.is_muted(notice["user_id"]):
            self.service.release_notification(notice["id"], attempt_id=notice["attempt_id"])
            return DeliveryReceipt(False, "foreground_or_mute")
        if not await asyncio.to_thread(self.service.begin_output, notice["id"], attempt_id=notice["attempt_id"], channel="text"):
            return DeliveryReceipt(False, "notice_no_longer_allowed")
        if self.occupied() or not self.service.attempt_active(notice):
            self.service.release_unwritten_output(notice["id"], attempt_id=notice["attempt_id"])
            return DeliveryReceipt(False, "deferred_before_text_write")
        await self.emit(notice["text"])
        return DeliveryReceipt(True, "text_sink_write_completed")

    async def interrupt(self, reason, *, signal_pipeline=False):
        # Text sinks provide their own bounded write. There is no pending audio
        # buffer here and no acoustic completion is asserted.
        pass


@dataclass
class AudioAttempt:
    notice: dict
    future: asyncio.Future
    invalidated: bool = False
    started: bool = False
    bytes_written: int = 0
    failed: bool = False


class PipecatNotificationSink:
    """Use native TTS serialization and output-queue markers for one utterance."""
    def __init__(self, service, worker=None, *, occupied=lambda: False, timeout_seconds=60):
        self.service, self.worker, self.occupied = service, worker, occupied
        self.timeout_seconds = timeout_seconds
        self.current = None

    def allowed(self, token):
        item = self.current
        return (item is not None and item.notice["attempt_id"] == token and not item.invalidated
                and not self.occupied() and self.service.attempt_active(item.notice)
                and (item.started or item.notice["expires_utc"] > datetime.now(timezone.utc).isoformat()))

    async def interrupt(self, reason="interrupted", *, signal_pipeline=False):
        if self.current and not self.current.invalidated:
            self.current.invalidated = True
            if not self.current.future.done(): self.current.future.set_result(DeliveryReceipt(False, reason))
            if signal_pipeline and self.worker is not None:
                await self.worker.queue_frame(InterruptionFrame())

    async def deliver(self, notice):
        if self.current is not None or self.worker is None:
            raise RuntimeError("Notification sink is busy or is not attached to a Pipecat worker")
        attempt = AudioAttempt(notice, asyncio.get_running_loop().create_future())
        self.current = attempt
        token = notice["attempt_id"]
        speak = TTSSpeakFrame(notice["text"], append_to_context=False)
        speak.metadata[NOTICE_TOKEN] = token
        try:
            await self.worker.queue_frames([
                OutputTransportMessageFrame({NOTICE_MARKER: "begin", NOTICE_TOKEN: token}), speak,
                OutputTransportMessageFrame({NOTICE_MARKER: "end", NOTICE_TOKEN: token})])
            return await asyncio.wait_for(asyncio.shield(attempt.future), self.timeout_seconds)
        except (TimeoutError, asyncio.CancelledError):
            await self.interrupt("notification_timeout_or_cancel", signal_pipeline=True)
            raise
        finally:
            self.current = None


class NotificationFrameGate(FrameProcessor):
    """Invalidate a reserved notice on framework interruption/error/shutdown."""
    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        error_token = frame.metadata.get(NOTICE_TOKEN) if isinstance(frame, ErrorFrame) else None
        stale_error = bool(error_token and (self.sink.current is None or self.sink.current.notice["attempt_id"] != error_token))
        if isinstance(frame, (InterruptionFrame, CancelFrame, ErrorFrame)) and not stale_error:
            await self.sink.interrupt(type(frame).__name__)
        if isinstance(frame, TTSSpeakFrame) and (token := frame.metadata.get(NOTICE_TOKEN)):
            if not self.sink.allowed(token):
                await self.sink.interrupt("foreground_or_mute_before_synthesis")
                return
        await self.push_frame(frame, direction)


class NotificationOutputMixin:
    """Extend a native BaseOutputTransport; its audio queue still owns ordering.

    Subclasses implement ``drain_notification_audio`` using the actual device
    API. Successful synthesis or a generic BotStoppedSpeakingFrame is not a
    playback receipt. Test sinks use channel='fixture' explicitly.
    """
    notification_channel = "audio"

    def bind_notifications(self, sink):
        self.notification_sink = sink
        self._notice_output = None

    async def process_frame(self, frame, direction):
        error_token = frame.metadata.get(NOTICE_TOKEN) if isinstance(frame, ErrorFrame) else None
        stale_error = bool(error_token and (self.notification_sink.current is None
            or self.notification_sink.current.notice["attempt_id"] != error_token))
        if isinstance(frame, (InterruptionFrame, CancelFrame, ErrorFrame)) and not stale_error:
            await self.notification_sink.interrupt(type(frame).__name__)
            self._notice_output = None
        token = frame.metadata.get(NOTICE_TOKEN)
        if token and not self.notification_sink.allowed(token):
            return
        await super().process_frame(frame, direction)

    async def send_message(self, frame):
        message = frame.message
        if not isinstance(message, dict) or NOTICE_MARKER not in message:
            return await super().send_message(frame)
        token = message.get(NOTICE_TOKEN)
        # An error may already have resolved the producer's future. Its queued
        # end marker must still clear that output span without touching a newer one.
        if message[NOTICE_MARKER] == "end" and self._notice_output is not None:
            if self._notice_output.notice["attempt_id"] == token:
                self._notice_output = None
        item = self.notification_sink.current
        if item is None or item.notice["attempt_id"] != token:
            return
        if message[NOTICE_MARKER] == "begin":
            self._notice_output = item
            return
        if message[NOTICE_MARKER] != "end": return
        try:
            if item.invalidated or item.failed or item.bytes_written == 0:
                receipt = DeliveryReceipt(False, "interrupted_failed_or_empty_output")
            else:
                await asyncio.wait_for(self.drain_notification_audio(), 5)
                receipt = DeliveryReceipt(self.notification_sink.allowed(token),
                    "audio_device_drained" if self.notification_channel == "audio" else "fixture_output_drained")
            if not item.future.done(): item.future.set_result(receipt)
        except Exception as error:
            if isinstance(error, TimeoutError):
                await self.push_error("Notification output drain timed out", exception=error, force_treat_as_permanent=True)
            if not item.future.done(): item.future.set_result(DeliveryReceipt(False, type(error).__name__))
        finally:
            self._notice_output = None

    async def write_audio_frame(self, frame):
        item = self._notice_output
        if item:
            token = item.notice["attempt_id"]
            if not self.notification_sink.allowed(token):
                await self.notification_sink.interrupt("notice_no_longer_allowed", signal_pipeline=True)
                return False
            if not item.started:
                item.started = await asyncio.to_thread(self.notification_sink.service.begin_output,
                    item.notice["id"], attempt_id=token, channel=self.notification_channel)
                if not item.started: return False
                if not self.notification_sink.allowed(token):
                    await asyncio.to_thread(self.notification_sink.service.release_unwritten_output,
                        item.notice["id"], attempt_id=token)
                    await self.notification_sink.interrupt("deferred_before_audio_write", signal_pipeline=True)
                    return False
            try:
                written = await super().write_audio_frame(frame)
            except BaseException:
                item.failed = True
                raise
            if written: item.bytes_written += len(frame.audio)
            else: item.failed = True
            return written
        return await super().write_audio_frame(frame)

    async def drain_notification_audio(self):
        raise NotImplementedError("An actual output-drain implementation is required")


def local_notification_output(py_audio, params, sink):
    """Pipecat's local output with PortAudio drain confirmation.

    Requires Pipecat's optional local/PyAudio dependency and an explicit device.
    Physical ReSpeaker/AEC qualification remains a separate hardware test.
    """
    from pipecat.transports.local.audio import LocalAudioOutputTransport
    class Output(NotificationOutputMixin, LocalAudioOutputTransport):
        async def drain_notification_audio(self):
            if self._out_stream is None: raise RuntimeError("Audio device is unavailable")
            def drain():
                # PortAudio Pa_StopStream waits for queued output to finish.
                self._out_stream.stop_stream()
                self._out_stream.start_stream()
            await self.get_event_loop().run_in_executor(self._executor, drain)
    transport = Output(py_audio, params)
    transport.bind_notifications(sink)
    return transport
