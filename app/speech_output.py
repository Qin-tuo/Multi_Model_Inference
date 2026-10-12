"""Conversation audio receipts on Pipecat's ordered output transport queue."""
import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import time
import wave

from pipecat.frames.frames import (CancelFrame, ErrorFrame, InterruptionFrame,
                                  OutputTransportMessageFrame, TTSSpeakFrame)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.base_output import BaseOutputTransport
from app.notifications import NotificationOutputMixin, NOTICE_TOKEN

SPEECH_TOKEN = "assistant_speech_turn"
SPEECH_MARKER = "assistant_speech_marker"


@dataclass
class SpeechAttempt:
    token: str
    future: asyncio.Future
    invalidated: bool = False
    bytes_written: int = 0
    first_pcm_ns: int | None = None
    first_sample: int | None = None


class ConversationSpeechSink:
    def __init__(self, worker=None, *, timeout=60):
        self.worker, self.timeout = None, timeout
        self.current = None
        self._interruption_events = {}
        if worker is not None: self.bind_worker(worker)

    def bind_worker(self, worker):
        self.worker = worker
        worker.add_reached_downstream_filter((InterruptionFrame,))
        @worker.event_handler("on_frame_reached_downstream")
        async def reached(_worker, frame):
            if frame.id in self._interruption_events: self._interruption_events[frame.id].set()

    async def interrupt_pipeline(self):
        if self.worker is None or self.worker.has_finished(): return
        frame = InterruptionFrame()
        event = self._interruption_events[frame.id] = asyncio.Event()
        try:
            await self.worker.queue_frame(frame)
            await asyncio.wait_for(event.wait(), 5)
        finally: self._interruption_events.pop(frame.id, None)

    def allowed(self, token):
        return self.current is not None and self.current.token == token and not self.current.invalidated

    def interrupt(self, reason, *, failed=False):
        item = self.current
        if item:
            item.invalidated = True
            if not item.future.done():
                item.future.set_exception(RuntimeError(reason) if failed else asyncio.CancelledError())

    async def deliver(self, text, *, turn_id):
        if self.current is not None or self.worker is None:
            raise RuntimeError("Speech output is busy or has no native worker")
        item = SpeechAttempt(turn_id, asyncio.get_running_loop().create_future())
        self.current = item
        speak = TTSSpeakFrame(text, append_to_context=False)
        speak.metadata[SPEECH_TOKEN] = turn_id
        try:
            await self.worker.queue_frames([
                OutputTransportMessageFrame({SPEECH_MARKER: "begin", SPEECH_TOKEN: turn_id}), speak,
                OutputTransportMessageFrame({SPEECH_MARKER: "end", SPEECH_TOKEN: turn_id})])
            return await asyncio.wait_for(asyncio.shield(item.future), self.timeout)
        except BaseException:
            self.interrupt("speech_delivery_aborted")
            if item.future.done() and not item.future.cancelled(): item.future.exception()
            cleanup = asyncio.create_task(self.interrupt_pipeline())
            try: await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
            raise
        finally:
            self.current = None


class ConversationSpeechGate(FrameProcessor):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        token = frame.metadata.get(SPEECH_TOKEN)
        if token and not self.sink.allowed(token): return
        if isinstance(frame, (InterruptionFrame, CancelFrame)):
            self.sink.interrupt(type(frame).__name__)
        elif isinstance(frame, ErrorFrame) and self.sink.current and not frame.metadata.get(NOTICE_TOKEN):
            self.sink.interrupt(frame.error, failed=True)
        await self.push_frame(frame, direction)


class ConversationOutputMixin:
    def bind_speech(self, sink):
        self.speech_sink = sink
        self._speech_output = None

    async def process_frame(self, frame, direction):
        if isinstance(frame, (InterruptionFrame, CancelFrame)):
            self.speech_sink.interrupt(type(frame).__name__)
            self._speech_output = None
        token = frame.metadata.get(SPEECH_TOKEN)
        if token and not self.speech_sink.allowed(token): return
        await super().process_frame(frame, direction)

    async def send_message(self, frame):
        message = frame.message
        if not isinstance(message, dict) or SPEECH_MARKER not in message:
            return await super().send_message(frame)
        token = message.get(SPEECH_TOKEN)
        if not self.speech_sink.allowed(token):
            if self._speech_output and self._speech_output.token == token: self._speech_output = None
            return
        item = self.speech_sink.current
        if message[SPEECH_MARKER] == "begin":
            self._speech_output = item
        elif message[SPEECH_MARKER] == "end":
            try:
                if not item.bytes_written: raise RuntimeError("TTS completed without output audio")
                await asyncio.wait_for(self.drain_speech_audio(), 5)
                if self.speech_sink.allowed(token) and not item.future.done():
                    item.future.set_result({"channel": self.speech_channel, "bytes_written": item.bytes_written,
                        "first_pcm_ns": item.first_pcm_ns, "first_sample": item.first_sample,
                        "completion": "device_drained" if self.speech_channel == "audio" else "wav_file_flushed"})
            except Exception as error:
                if not item.future.done(): item.future.set_exception(error)
            finally: self._speech_output = None

    async def write_audio_frame(self, frame):
        item = self._speech_output
        if item and not self.speech_sink.allowed(item.token): return False
        try:
            if item and item.first_sample is None: item.first_sample = getattr(self, "samples_written", None)
            written = await super().write_audio_frame(frame)
            if item:
                if not written: raise RuntimeError("Audio output write failed")
                if item.first_pcm_ns is None: item.first_pcm_ns = time.monotonic_ns()
                item.bytes_written += len(frame.audio)
                if item.bytes_written > 10 * 1024 * 1024: raise RuntimeError("Speech output exceeded 10 MiB")
            return written
        except Exception as error:
            self.speech_sink.interrupt(str(error), failed=True)
            raise


class WavOutputDevice(BaseOutputTransport):
    """A file transport for reproducible input; no physical playback is claimed."""
    speech_channel = "fixture"

    def __init__(self, path, params):
        super().__init__(params)
        self.path = Path(path)
        self.stream = self.wav = None
        self.samples_written = 0

    async def start(self, frame):
        await super().start(frame)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Do not overwrite a user's previous recording.
        self.stream = self.path.open("xb")
        self.wav = wave.open(self.stream, "wb")
        self.wav.setnchannels(1); self.wav.setsampwidth(2); self.wav.setframerate(16000)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame):
        if (frame.sample_rate, frame.num_channels) != (16000, 1) or len(frame.audio) % 2:
            raise ValueError("Output must be PCM16 mono at 16 kHz")
        if self.samples_written * 2 + len(frame.audio) > 256 * 1024 * 1024:
            raise ValueError("File output exceeded 256 MiB; start a new output file")
        self.wav.writeframesraw(frame.audio)
        self.samples_written += len(frame.audio) // 2
        return True

    async def drain_speech_audio(self):
        self.wav.writeframes(b"")  # Patch the WAV length before confirming output.
        self.stream.flush()
        os.fsync(self.stream.fileno())

    drain_notification_audio = drain_speech_audio

    async def cleanup(self):
        await super().cleanup()
        if self.wav: self.wav.close(); self.wav = None
        if self.stream: self.stream.close(); self.stream = None


class WavAssistantOutput(NotificationOutputMixin, ConversationOutputMixin, WavOutputDevice):
    notification_channel = "fixture"
    drain_notification_audio = WavOutputDevice.drain_notification_audio

    def __init__(self, path, params, speech_sink, notification_sink):
        super().__init__(path, params)
        self.bind_speech(speech_sink)
        self.bind_notifications(notification_sink)


def local_assistant_output(py_audio, params, speech_sink, notification_sink):
    from pipecat.transports.local.audio import LocalAudioOutputTransport
    class Output(NotificationOutputMixin, ConversationOutputMixin, LocalAudioOutputTransport):
        speech_channel = "audio"

        async def drain_speech_audio(self):
            if self._out_stream is None: raise RuntimeError("Audio device is unavailable")
            def drain():
                self._out_stream.stop_stream()
                self._out_stream.start_stream()
            await self.get_event_loop().run_in_executor(self._executor, drain)

        drain_notification_audio = drain_speech_audio
    output = Output(py_audio, params)
    output.bind_speech(speech_sink)
    output.bind_notifications(notification_sink)
    return output
