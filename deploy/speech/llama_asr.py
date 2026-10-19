"""An OVS backend adapter to the pinned llama.cpp Qwen3-ASR service."""

from __future__ import annotations

import base64
import json
import re
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from server.core.asr_backend import ASRBackend, ASRCapability, TranscriptionResult
from server.core.profile_loader import current_profile


class LlamaASRBackend(ASRBackend):
    supports_hot_reload = False

    def __init__(self):
        settings = current_profile().get("local_llama_asr", {})
        self.url = settings.get("base_url", "http://127.0.0.1:8089").rstrip("/")
        endpoint = urlsplit(self.url)
        if endpoint.scheme != "http" or endpoint.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("ASR endpoint must be local HTTP")
        self.model = settings.get("model", "assistant-asr")
        self.context = settings.get("context", "")
        if not isinstance(self.context, str) or len(self.context) > 1024:
            raise ValueError("ASR context must be a bounded string")

    @property
    def name(self):
        return "llama_qwen3_asr"

    @property
    def sample_rate(self):
        return 16000

    @property
    def capabilities(self):
        return {
            ASRCapability.OFFLINE,
            ASRCapability.MULTI_LANGUAGE,
            ASRCapability.LANGUAGE_ID,
        }

    def _request(self, path, value=None, timeout=60):
        data = json.dumps(value).encode() if value is not None else None
        with urlopen(
            Request(self.url + path, data, {"Content-Type": "application/json"}),
            timeout=timeout,
        ) as response:
            return json.load(response)

    def is_ready(self):
        try:
            return self._request("/health", timeout=2).get("status") == "ok"
        except (OSError, ValueError):
            return False

    def preload(self):
        if not self.is_ready():
            raise RuntimeError("Local Qwen3-ASR llama service is not ready")

    def transcribe(self, audio_bytes, language="auto"):
        if not audio_bytes or len(audio_bytes) > 8 * 1024 * 1024:
            raise ValueError("ASR audio must be nonempty and at most 8 MiB")
        started = time.monotonic()
        response = self._request(
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.context},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": base64.b64encode(audio_bytes).decode(),
                                    "format": "wav",
                                },
                            }
                        ],
                    },
                ],
                "temperature": 0,
                "max_tokens": 256,
            },
        )
        raw = response["choices"][0]["message"].get("content", "")
        match = re.fullmatch(r"language\s+([^<]+)<asr_text>(.*)", raw, re.DOTALL)
        if not match:
            raise RuntimeError("Qwen3-ASR returned an unexpected transcript envelope")
        detected, text = match.group(1).strip(), match.group(2).strip()
        return TranscriptionResult(
            text,
            language=detected,
            inference_time=time.monotonic() - started,
            backend="llama.cpp_cuda",
            streaming=False,
            remote_resource_release_on_cancel="not_yet_verified",
        )
