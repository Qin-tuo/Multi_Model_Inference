"""Bound and scope Pipecat's native segmented OpenAI-compatible STT service.

The upstream service still owns WAV segmentation and its single transcription
task. These hooks carry segment ownership and prevent premature turn completion.
"""
import asyncio
from collections import Counter
from dataclasses import dataclass
import time

import httpx
from pipecat.frames.frames import (ErrorFrame, InputAudioRawFrame, TranscriptionFrame,
                                  UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy

ASR_GENERATION = "assistant_asr_generation"
ASR_SEQUENCE = "assistant_asr_segment"
INPUT_EPOCH = "assistant_audio_input_epoch"
ABORT_GENERATION = "assistant_abort_input_generation"


@dataclass(frozen=True)
class Segment:
    audio: bytes
    generation: int
    sequence: int
    deadline_ns: int


class SegmentQueue(asyncio.Queue):
    """One pending native segment; overflow fails the whole current utterance."""
    def __init__(self, owner):
        super().__init__(maxsize=1)
        self.owner = owner

    async def put(self, audio):
        if audio is None:
            return await super().put(None)
        owner = self.owner
        scope = owner.generation()
        # Obsolete pending work can be discarded before the native consumer
        # takes it. The already-running request remains single-flight.
        if self.full():
            queued = self.get_nowait()
            if queued is not None and queued.generation == scope:
                self.put_nowait(queued)
                await owner.reject(scope, "ASR_PENDING_QUEUE_FULL")
                return
            if queued: owner.pending.pop(queued.sequence, None)
            owner.counts["obsolete_pending_discarded"] += 1
        owner.sequence += 1
        segment = Segment(audio, scope, owner.sequence,
                          time.monotonic_ns() + int(owner.request_timeout * 1e9))
        owner.latest = (scope, segment.sequence)
        owner.pending[segment.sequence] = (scope, len(audio))
        self.put_nowait(segment)
        owner.counts["segments_submitted"] += 1
        owner.peak_segment_bytes = max(owner.peak_segment_bytes, sum(size for _, size in owner.pending.values()))


class ScopedSTTService(OpenAISTTService):
    def __init__(self, *, generation, input_epoch=None, request_timeout=12, **kwargs):
        self.generation = generation
        self.input_epoch = input_epoch
        self.request_timeout = request_timeout
        self.counts = Counter()
        self.sequence = self.peak_segment_bytes = 0
        self.latest = None
        self.pending = {}
        self.completed = set()
        self.rejected_generation = None
        self.aborted_segment = False
        # 30s speech plus the native 1s VAD prefix and 0.5s trailing silence.
        self.max_pcm_bytes = 16000 * 2 * 31
        self.max_segment_bytes = self.max_pcm_bytes + 16000 + 44
        super().__init__(http_client=httpx.AsyncClient(timeout=request_timeout, trust_env=False,
            limits=httpx.Limits(max_connections=1)), **kwargs)
        self._client.max_retries = 0
        self._segment_queue = SegmentQueue(self)

    def accepts(self, generation):
        return generation == self.generation() and generation != self.rejected_generation

    def complete_through(self, sequence):
        scope = self.generation()
        return (self.accepts(scope) and self.latest is not None and self.latest[0] == scope
                and sequence >= self.latest[1] and not any(g == scope and seq not in self.completed
                    for seq, (g, _) in self.pending.items()))

    async def reject(self, scope, reason):
        self.rejected_generation = scope
        self.counts[reason] += 1
        error = ErrorFrame(error=reason)
        error.metadata[ASR_GENERATION] = scope
        await self.push_error_frame(error)

    def invalidate(self):
        self._audio_buffer.clear()
        self._user_speaking = False
        self.aborted_segment = True
        while not self._segment_queue.empty():
            item = self._segment_queue.get_nowait()
            if item: self.pending.pop(item.sequence, None)
        self.counts["input_invalidated"] += 1

    async def process_frame(self, frame, direction):
        if (isinstance(frame, InputAudioRawFrame) and self.input_epoch
                and frame.metadata.get(INPUT_EPOCH, self.input_epoch()) != self.input_epoch()):
            self.counts["obsolete_input_frames_discarded"] += 1
            return
        if isinstance(frame, InputAudioRawFrame) and len(frame.audio) > 6400:
            await self.reject(self.generation(), "ASR_AUDIO_FRAME_LIMIT")
            return
        await super().process_frame(frame, direction)

    async def process_audio_frame(self, frame, direction):
        scope = self.generation()
        if self.rejected_generation == scope:
            self._audio_buffer.clear()
            return
        if len(self._audio_buffer) + len(frame.audio) > self.max_pcm_bytes:
            self._audio_buffer.clear()
            await self.reject(scope, "ASR_AUDIO_BUFFER_LIMIT")
            return
        await super().process_audio_frame(frame, direction)

    async def _handle_user_started_speaking(self, frame):
        self.aborted_segment = False
        await super()._handle_user_started_speaking(frame)

    async def _handle_user_stopped_speaking(self, frame):
        if self.aborted_segment or not self.accepts(self.generation()):
            self._audio_buffer.clear()
            self._user_speaking = False
            return
        await super()._handle_user_stopped_speaking(frame)

    async def run_stt(self, segment):
        if not isinstance(segment, Segment):
            raise ValueError("ASR segment has no capture ownership")
        try:
            if not self.accepts(segment.generation):
                self.counts["obsolete_before_request"] += 1
                return
            if len(segment.audio) > self.max_segment_bytes:
                await self.reject(segment.generation, "ASR_SEGMENT_LIMIT")
                return
            remaining = (segment.deadline_ns - time.monotonic_ns()) / 1e9
            if remaining <= 0:
                self.counts["expired_before_request"] += 1
                await self.reject(segment.generation, "ASR_REQUEST_TIMEOUT")
                return
            async with asyncio.timeout(remaining):
                had_result = False
                async for frame in super().run_stt(segment.audio):
                    had_result = True
                    self.completed.add(segment.sequence)
                    if not self.accepts(segment.generation):
                        self.counts["obsolete_results_discarded"] += 1
                        continue
                    if frame:
                        frame.metadata.update({ASR_GENERATION: segment.generation, ASR_SEQUENCE: segment.sequence})
                        yield frame
                if not had_result and self.accepts(segment.generation):
                    await self.reject(segment.generation, "ASR_EMPTY_TRANSCRIPT")
        except TimeoutError:
            if self.accepts(segment.generation):
                await self.reject(segment.generation, "ASR_REQUEST_TIMEOUT")
        finally:
            self.pending.pop(segment.sequence, None)
            self.completed.discard(segment.sequence)

    def status(self):
        return {"counts": dict(self.counts), "queued_segments": self._segment_queue.qsize(),
            "active_and_pending_segments": len(self.pending),
            "segment_bytes": sum(size for _, size in self.pending.values()),
            "peak_segment_bytes": self.peak_segment_bytes, "pcm_buffer_bytes": len(self._audio_buffer),
            "max_pcm_bytes": self.max_pcm_bytes, "max_segment_bytes": self.max_segment_bytes,
            "max_pending_segments": 1, "request_timeout_seconds": self.request_timeout}

    async def cleanup(self):
        await super().cleanup()
        await self._client.close()


class ScopedUserAggregator(LLMUserAggregator):
    def __init__(self, context, *, asr, stop_strategy, **kwargs):
        super().__init__(context, **kwargs)
        self.asr = asr
        self.scoped_stop_strategy = stop_strategy
        self.source_generation = None
        self.source_parts = []
        self.source_bytes = 0
        self.admitted_source = None

    def consume_current_transcript(self, text):
        """Qualify the asynchronous turn event against consumed ASR segments."""
        if self.source_generation != self.asr.generation() or not self.source_parts or self.asr._user_speaking:
            return False
        sequence = self.source_parts[-1][0]
        source = (self.source_generation, sequence)
        if self.admitted_source == source or not self.asr.complete_through(sequence): return False
        expected = "".join(part for _, part in self.source_parts)
        if "".join(expected.split()) != "".join(text.split()): return False
        self.admitted_source = source
        return True

    async def process_frame(self, frame, direction):
        if isinstance(frame, UserStoppedSpeakingFrame) and ABORT_GENERATION in frame.metadata:
            if frame.metadata[ABORT_GENERATION] == self.asr.generation():
                # Execute reset inside the aggregator's native frame processing
                # queue, serialized with VAD/start events. This is not an async
                # callback reaching in and clearing a newer user's input.
                stopped = VADUserStoppedSpeakingFrame()
                stopped.metadata[ABORT_GENERATION] = frame.metadata[ABORT_GENERATION]
                # This is an explicitly labelled input-reset control signal,
                # not a detected acoustic endpoint or a latency observation.
                await super().process_frame(stopped, direction)
                await self.reset()
                await self.scoped_stop_strategy.trigger_user_turn_stopped()
            await self.push_frame(frame, direction)
            return
        # Check at the actual aggregation consumer, after any intervening
        # queueing or new VAD turn. A provider-only check is insufficient.
        if isinstance(frame, TranscriptionFrame) and not self.asr.accepts(frame.metadata.get(ASR_GENERATION)):
            self.asr.counts["obsolete_at_aggregator"] += 1
            return
        if isinstance(frame, TranscriptionFrame):
            scope, sequence = frame.metadata.get(ASR_GENERATION), frame.metadata.get(ASR_SEQUENCE)
            if scope != self.source_generation:
                self.source_generation, self.source_parts, self.source_bytes = scope, [], 0
            if type(sequence) is not int or sequence <= 0 or sequence > self.asr.sequence or (self.source_parts and sequence <= self.source_parts[-1][0]):
                self.asr.counts["invalid_or_duplicate_transcription"] += 1
                return
            self.source_bytes += len(frame.text.encode("utf-8"))
            if self.source_bytes > 8192:
                await self.asr.reject(scope, "ASR_TEXT_BUFFER_LIMIT")
                return
            self.source_parts.append((sequence, frame.text))
        await super().process_frame(frame, direction)


class CompleteASRTurnStop(SpeechTimeoutUserTurnStopStrategy):
    """Use native stop timers, but never finalize a prefix of queued speech."""
    def __init__(self, asr, **kwargs):
        super().__init__(**kwargs)
        self.asr = asr
        self.received_sequence = 0

    async def handle_user_turn_started(self):
        self.received_sequence = 0
        await super().handle_user_turn_started()

    async def _handle_vad_user_stopped_speaking(self, frame):
        if ABORT_GENERATION in frame.metadata:
            # The aggregator has already checked this reset's generation. It
            # updates native VAD state, but is not a new acoustic endpoint.
            self.received_sequence = 0
            await self.handle_user_turn_stopped()
            return
        # Pipecat 1.12 restarts the speech timer on every stop, but replaces the
        # STT timer handle without cancelling its previous task. Duplicate stop
        # events otherwise orphan that task and its later turn-stop callback.
        if self._stt_timeout_task:
            await self.task_manager.cancel_task(self._stt_timeout_task)
            self._stt_timeout_task = None
        await super()._handle_vad_user_stopped_speaking(frame)

    async def _handle_transcription(self, frame):
        self.received_sequence = max(self.received_sequence, frame.metadata.get(ASR_SEQUENCE, 0))
        await super()._handle_transcription(frame)

    async def _maybe_trigger_user_turn_stopped(self):
        if self.asr.complete_through(self.received_sequence):
            await super()._maybe_trigger_user_turn_stopped()
