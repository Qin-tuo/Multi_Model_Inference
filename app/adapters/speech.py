"""Local OVS adapters using Pipecat's maintained STT and TTS service base."""
from __future__ import annotations

from collections.abc import AsyncGenerator
import httpx
import json
from pathlib import Path
from urllib.parse import urlsplit

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
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

    def __init__(self):
        self._local_config = speech_settings()
        super().__init__(sample_rate=16000, settings=TTSSettings(model=self._local_config["tts_model"], voice=self._local_config["voice"]))
        self._http = httpx.AsyncClient(timeout=30, trust_env=False, limits=httpx.Limits(max_connections=1))

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        await self.start_ttfb_metrics()
        try:
            async with self._http.stream("POST", self._local_config["base_url"].rstrip("/") + "/audio/speech", json={
                "model": self._local_config["tts_model"], "voice": self._local_config["voice"], "input": text, "response_format": "pcm",
            }) as response:
                response.raise_for_status()
                await self.start_tts_usage_metrics(text)
                async for chunk in response.aiter_bytes(640):
                    if chunk:
                        if len(chunk) % 2: raise ValueError("OVS returned a partial PCM16 sample")
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(chunk, 16000, 1, context_id=context_id)
        except (httpx.HTTPError, ValueError) as error:
            yield ErrorFrame(error=f"Local OVS TTS failed: {error}")

    async def cleanup(self):
        await self._http.aclose()
        await super().cleanup()
