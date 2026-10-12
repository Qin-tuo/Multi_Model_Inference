"""Native Pipecat speech I/O connected to the persistent assistant session.

Pipecat owns VAD, turn detection, STT, TTS, interruption and audio queues.
This adapter forwards finalized user turns and waits for actual output receipts.
"""
from __future__ import annotations

import asyncio
import time
import wave

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (EndFrame, ErrorFrame, InputAudioRawFrame, InterruptionFrame,
                                  LLMContextFrame, VADUserStoppedSpeakingFrame)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from app.adapters.speech import OVSTTSService, create_stt_service, validate_speech_capabilities
from app.notifications import NotificationFrameGate, NotificationPump, PipecatNotificationSink, NOTICE_TOKEN
from app.session import AssistantSession
from app.speech_output import (ConversationSpeechGate, ConversationSpeechSink, SPEECH_TOKEN,
                               WavAssistantOutput, local_assistant_output)


class VoiceContextBridge(FrameProcessor):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            # VAD reports after its silence window. Use the inferred end of
            # speech for picture selection and preserve detection time separately.
            self.owner.vad_detected_ns = time.monotonic_ns()
            self.owner.question_time_ns = self.owner.vad_detected_ns - int(frame.stop_secs * 1e9)
        if isinstance(frame, LLMContextFrame):
            # The finalized transcript event is the input to AssistantSession,
            # whose native model pipeline adds memory/tools/visual context.
            frame.context.set_messages([])
            return
        await self.push_frame(frame, direction)


class VoiceAssistant:
    def __init__(self, config, memory, reminders, records, *, user_id="owner", vision_frames=None,
                 output_path=None, input_device=None, output_device=None, session_id=None,
                 context_tokens=2048, stt=None, llm=None, tts=None):
        self.user_id, self.reminders = user_id, reminders
        self.user_speaking = False
        self.answer_task = self.interrupt_task = self.runner_task = self.notification_task = None
        self.generation = 0
        self.question_time_ns = self.vad_detected_ns = None
        self.result_future = None
        self.results, self.errors = [], []
        self.ready = asyncio.Event()
        self.interruption_events = {}
        self.file_lock = asyncio.Lock()
        self.closed = False
        self.file_mode = output_path is not None
        self.py_audio = None
        self.output_path = output_path
        self.speech = ConversationSpeechSink()
        self.notices = PipecatNotificationSink(reminders, occupied=self.occupied)
        self.session = AssistantSession(config, memory, reminders, records, user_id=user_id,
            session_id=session_id, vision_frames=vision_frames, llm=llm, context_tokens=context_tokens,
            output_channel="fixture" if self.file_mode else "audio")
        params = dict(audio_out_enabled=True, audio_out_sample_rate=16000, audio_out_channels=1,
                      audio_out_10ms_chunks=2, audio_out_end_silence_secs=0)
        processors = []
        if self.file_mode:
            output = WavAssistantOutput(output_path, TransportParams(**params), self.speech, self.notices)
        else:
            if input_device is None or output_device is None:
                raise ValueError("Local speech requires explicit input and output device indexes")
            import pyaudio
            from pipecat.transports.local.audio import LocalAudioInputTransport, LocalAudioTransportParams
            self.py_audio = pyaudio.PyAudio()
            local_params = LocalAudioTransportParams(**params, audio_in_enabled=True,
                audio_in_sample_rate=16000, audio_in_channels=1,
                input_device_index=input_device, output_device_index=output_device)
            processors.append(LocalAudioInputTransport(self.py_audio, local_params))
            output = local_assistant_output(self.py_audio, local_params, self.speech, self.notices)
        self.output = output
        context = LLMContext()
        self.aggregators = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(sample_rate=16000, params=VADParams(stop_secs=.4)),
            user_turn_strategies=UserTurnStrategies(stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=.1)])))
        self.tts = tts or OVSTTSService(notification_sink=self.notices, speech_sink=self.speech)
        self.worker = PipelineWorker(Pipeline([*processors, stt or create_stt_service(),
            self.aggregators.user(), VoiceContextBridge(self), NotificationFrameGate(self.notices),
            ConversationSpeechGate(self.speech), self.tts, output, self.aggregators.assistant()]),
            params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=16000,
                                  enable_metrics=True, enable_usage_metrics=True),
            idle_timeout_secs=None, enable_rtvi=False)
        self.speech.bind_worker(self.worker)
        self.notices.worker = self.worker
        self.runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
        self.notifications = NotificationPump(reminders, user_id, self.notices, occupied=self.occupied)
        self.worker.add_reached_downstream_filter((InterruptionFrame,))

        @self.worker.event_handler("on_pipeline_started")
        async def started(_worker, _frame): self.ready.set()

        @self.worker.event_handler("on_frame_reached_downstream")
        async def reached(_worker, frame):
            if frame.id in self.interruption_events: self.interruption_events[frame.id].set()

        @self.worker.event_handler("on_pipeline_error")
        async def error(_worker, frame):
            token = frame.metadata.get(SPEECH_TOKEN)
            if frame.metadata.get(NOTICE_TOKEN) or (token and not self.speech.allowed(token)): return
            self.errors.append(frame.error)
            self.errors = self.errors[-100:]
            if self.result_future and not self.result_future.done():
                self.result_future.set_exception(RuntimeError(frame.error))

        @self.aggregators.user().event_handler("on_user_turn_started")
        async def user_started(_aggregator, _strategy):
            self.user_speaking = True
            self.generation += 1
            self.question_time_ns = self.vad_detected_ns = None
            self.speech.interrupt("new_user_turn")
            if self.answer_task and not self.answer_task.done() and not self.answer_task.cancelling():
                self.answer_task.cancel()
            self._ensure_interruption()

        @self.aggregators.user().event_handler("on_user_turn_stopped")
        async def user_stopped(_aggregator, _strategy, message):
            self.user_speaking = False
            if not message.content or self.closed: return
            if len(message.content.encode()) > 8192:
                if self.result_future and not self.result_future.done():
                    self.result_future.set_exception(ValueError("Transcript exceeds 8 KiB"))
                return
            previous = self.answer_task
            if previous and not previous.done() and not previous.cancelling(): previous.cancel()
            self.answer_task = asyncio.create_task(self._answer(message.content, self.generation, previous,
                self.result_future, self.question_time_ns, self.vad_detected_ns))

    def occupied(self):
        return self.user_speaking or self.session.busy.locked() or (self.answer_task is not None and not self.answer_task.done())

    def _ensure_interruption(self):
        if self.interrupt_task is None or self.interrupt_task.done():
            self.interrupt_task = asyncio.create_task(self._interrupt())
        return self.interrupt_task

    async def _interrupt(self):
        frame = InterruptionFrame()
        event = self.interruption_events[frame.id] = asyncio.Event()
        try:
            await self.worker.queue_frame(frame)
            await asyncio.wait_for(event.wait(), 5)
            await self.session.cancel()
        finally: self.interruption_events.pop(frame.id, None)

    async def _answer(self, text, generation, previous, future, question_time_ns, vad_detected_ns):
        try:
            if self.interrupt_task: await asyncio.shield(self.interrupt_task)
            if previous: await asyncio.gather(previous, return_exceptions=True)
            if self.closed or generation != self.generation: raise asyncio.CancelledError()
            receipt = None
            async def emit(answer):
                nonlocal receipt
                receipt = await self.speech.deliver(answer, turn_id=self.session.turn.turn_id)
            result = await self.session.ask(text, emit=emit, question_time_ns=question_time_ns)
            result.update(transcript=text, speech_receipt=receipt, vad_detected_ns=vad_detected_ns)
            if receipt and vad_detected_ns:
                result["vad_detection_to_first_pcm_seconds"] = (receipt["first_pcm_ns"] - vad_detected_ns) / 1e9
                result["inferred_speech_end_to_first_pcm_seconds"] = (receipt["first_pcm_ns"] - question_time_ns) / 1e9
            self.results.append(result); self.results = self.results[-100:]
            if future and not future.done(): future.set_result(result)
        except asyncio.CancelledError:
            if future and not future.done(): future.set_exception(asyncio.CancelledError())
        except Exception as error:
            self.errors.append(str(error)); self.errors = self.errors[-100:]
            if future and not future.done(): future.set_exception(error)

    async def start(self, *, check_services=True):
        if check_services: await validate_speech_capabilities()
        await self.session.start()
        await self.runner.add_workers(self.worker)
        self.runner_task = asyncio.create_task(self.runner.run())
        await asyncio.wait_for(self.ready.wait(), 15)
        self.notification_task = asyncio.create_task(self.notifications.run())
        return self

    async def run_file(self, path, *, timeout=90):
        if not self.file_mode: raise ValueError("run_file requires the file output transport")
        if self.file_lock.locked(): raise RuntimeError("A file utterance is already in flight")
        async with self.file_lock:
            with wave.open(str(path), "rb") as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                    raise ValueError("Input must be PCM16 mono WAV at 16 kHz")
                if not 0 < wav.getnframes() <= 16000 * 30: raise ValueError("File utterance must be 0..30 seconds")
                pcm = wav.readframes(wav.getnframes())
            future = self.result_future = asyncio.get_running_loop().create_future()
            try:
                async with asyncio.timeout(timeout):
                    audio = bytes(16000) + pcm + bytes(32000)
                    for start in range(0, len(audio), 640):
                        if self.worker.has_finished(): raise RuntimeError("Speech pipeline stopped")
                        await self.worker.queue_frame(InputAudioRawFrame(audio[start:start+640], 16000, 1))
                        await asyncio.sleep(.02)
                    return await asyncio.shield(future)
            except BaseException:
                self.speech.interrupt("file_request_aborted")
                await asyncio.shield(self._ensure_interruption())
                raise
            finally:
                if future.done() and not future.cancelled(): future.exception()
                else: future.cancel()
                self.result_future = None

    async def close(self):
        self.closed = True
        await self.notifications.stop()
        if self.notification_task: await self.notification_task
        self.speech.interrupt("voice_assistant_closed")
        if self.answer_task and not self.answer_task.done(): self.answer_task.cancel()
        if self.answer_task: await asyncio.gather(self.answer_task, return_exceptions=True)
        if self.interrupt_task: await asyncio.gather(self.interrupt_task, return_exceptions=True)
        if self.runner_task:
            if not self.worker.has_finished(): await self.worker.queue_frame(EndFrame())
            await asyncio.wait_for(self.runner_task, 15)
        await self.session.close()
        if self.py_audio: self.py_audio.terminate()
