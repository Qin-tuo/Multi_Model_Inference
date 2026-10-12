"""Local OVS adapters using Pipecat's maintained STT and TTS service base."""
from __future__ import annotations

from collections.abc import AsyncGenerator
import httpx
import json
from pathlib import Path
from urllib.parse import urlsplit

from pipecat.frames.frames import CancelFrame, ErrorFrame, Frame, InterruptionFrame, TTSAudioRawFrame, TTSSpeakFrame
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.tts_service import TTSService
from pipecat.services.settings import TTSSettings
from pipecat.transcriptions.language import Language


def speech_settings():
    settings = json.loads((Path(__file__).resolve().parents[2] / "configs/runtime.yaml").read_text())["speech"]
    endpoint = urlsplit(settings["base_url"])
    if endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Speech service must use local HTTP")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment or endpoint.path.rstrip("/") != "/v1":
        raise ValueError("Speech endpoint must be an uncredentialed local /v1 URL")
    if settings["sample_rate"] != 16000 or settings["channels"] != 1:
        raise ValueError("The qualified OVS profile requires 16 kHz mono")
    return settings


async def validate_speech_capabilities():
    config = speech_settings()
    root = config["base_url"].rstrip("/").removesuffix("/v1")
    async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
        response = await client.get(root + "/v1/capabilities")
        response.raise_for_status()
        caps = response.json()
    for kind in ("asr", "tts"):
        if not caps[kind]["ready"] or caps[kind]["model_id"] != config[kind + "_model"]:
            raise ValueError(f"The configured {kind} model is not ready")
        audio = caps[kind]["audio"]
        if audio["sample_rate"] != config["sample_rate"] or audio["channels"] != config["channels"]:
            raise ValueError(f"The {kind} audio format differs from the configured transport")
    if "pcm" not in caps["tts"]["audio"]["response_formats"]:
        raise ValueError("OVS must provide PCM for the Pipecat transport")
    return caps


def create_stt_service():
    config = speech_settings()
    return OpenAISTTService(
        api_key="local-no-key", base_url=config["base_url"],
        settings=OpenAISTTService.Settings(model=config["asr_model"], language=Language.ZH),
        sample_rate=16000, include_prob_metrics=False,
    )


class OVSTTSService(TTSService):
    """OVS numeric voices/16 kHz PCM atop native Pipecat aggregation/cancellation.

    The upstream OpenAITTSService validates OpenAI's named voices and 24 kHz.
    This transport adapter preserves OVS's actual voice and audio format.
    """

    def __init__(self, *, notification_sink=None, speech_sink=None):
        self._local_config = speech_settings()
        self._notification_sink = notification_sink
        self._speech_sink = speech_sink
        self._delivery_submission = None
        self._delivery_contexts = {}
        super().__init__(sample_rate=16000, push_start_frame=True, push_stop_frames=True,
                         max_consecutive_zero_audio_contexts=0,
                         settings=TTSSettings(model=self._local_config["tts_model"], voice=self._local_config["voice"], language=None))
        self._http = httpx.AsyncClient(timeout=30, trust_env=False, limits=httpx.Limits(max_connections=1))

    async def process_frame(self, frame, direction):
        from app.notifications import NOTICE_TOKEN
        from app.speech_output import SPEECH_TOKEN
        delivery = None
        if isinstance(frame, TTSSpeakFrame):
            for key, sink in ((NOTICE_TOKEN, self._notification_sink), (SPEECH_TOKEN, self._speech_sink)):
                if (token := frame.metadata.get(key)) and sink is not None:
                    if not sink.allowed(token): return
                    delivery = (key, token)
        if delivery:
            self._delivery_submission = delivery
            try: await super().process_frame(frame, direction)
            finally:
                if self._delivery_submission == delivery: self._delivery_submission = None
        else:
            await super().process_frame(frame, direction)
            if isinstance(frame, (InterruptionFrame, CancelFrame)):
                self._delivery_contexts.clear()

    async def on_turn_context_created(self, context_id):
        if self._delivery_submission:
            self._delivery_contexts[context_id] = self._delivery_submission
        await super().on_turn_context_created(context_id)

    async def push_frame(self, frame, direction=None):
        from pipecat.processors.frame_processor import FrameDirection
        context_id = getattr(frame, "context_id", None)
        delivery = self._delivery_contexts.get(context_id)
        if delivery: frame.metadata[delivery[0]] = delivery[1]
        await super().push_frame(frame, direction or FrameDirection.DOWNSTREAM)

    async def push_error_frame(self, error, force_treat_as_permanent=False):
        delivery = self._delivery_contexts.get(self.get_active_audio_context_id())
        if delivery: error.metadata.setdefault(delivery[0], delivery[1])
        await super().push_error_frame(error, force_treat_as_permanent=force_treat_as_permanent)

    async def on_audio_context_completed(self, context_id):
        self._delivery_contexts.pop(context_id, None)
        await super().on_audio_context_completed(context_id)

    async def on_audio_context_interrupted(self, context_id):
        self._delivery_contexts.pop(context_id, None)
        await super().on_audio_context_interrupted(context_id)

    async def tts_process_generator(self, context_id, generator):
        await super().tts_process_generator(context_id, generator)
        # This is a finite HTTP response even when it failed or returned no
        # PCM. The base heuristic otherwise treats zero-audio as a websocket
        # context and waits for a timeout before closing it.
        self._is_yielding_frames_synchronously = True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        delivery = self._delivery_contexts.get(context_id)
        byte_count = 0
        await self.start_ttfb_metrics()
        try:
            async with self._http.stream("POST", self._local_config["base_url"].rstrip("/") + "/audio/speech", json={
                "model": self._local_config["tts_model"], "voice": self._local_config["voice"], "input": text, "response_format": "pcm",
            }) as response:
                response.raise_for_status()
                await self.start_tts_usage_metrics(text)
                async for chunk in response.aiter_bytes(640):
                    if chunk:
                        byte_count += len(chunk)
                        if byte_count > 10 * 1024 * 1024: raise ValueError("TTS response exceeded 10 MiB")
                        if len(chunk) % 2: raise ValueError("OVS returned a partial PCM16 sample")
                        await self.stop_ttfb_metrics()
                        frame = TTSAudioRawFrame(chunk, 16000, 1, context_id=context_id)
                        if delivery: frame.metadata[delivery[0]] = delivery[1]
                        yield frame
        except (httpx.HTTPError, ValueError) as error:
            frame = ErrorFrame(error=f"Local OVS TTS failed: {error}")
            if delivery: frame.metadata[delivery[0]] = delivery[1]
            yield frame

    async def cleanup(self):
        await self._http.aclose()
        await super().cleanup()
