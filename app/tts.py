# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Upstream TTS contract adapted to local Matcha/Vocos with bounded PCM output."""
import time
from pathlib import Path
import httpx
from app.request_io import RequestIO
import numpy as np
from app.audio import local_speech_url, speech_capability, pcm_wav


class LocalTTS:
    def __init__(self, voice="0", speed=1.0, lang="zh", model="matcha-icefall-zh-en",
                 base_url="http://127.0.0.1:8621/v1", backend="matcha_trt", timeout=60.0,
                 max_audio_seconds=30.0):
        self.voice, self.speed, self.lang = voice, speed, lang
        self.model, self.backend_name = model, backend
        self.base_url = local_speech_url(base_url)
        self.timeout, self.max_audio_seconds = timeout, max_audio_seconds
        self._client = None
        self.last_error = None
        self.capability = None
        self.requests = RequestIO()

    def load(self):
        self.unload()
        self._client = httpx.Client(timeout=self.timeout, trust_env=False,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0))
        try:
            self.capability = speech_capability(self.base_url, "tts", self.model, self.backend_name,self.timeout)
            if "pcm" not in self.capability["audio"]["response_formats"]:
                raise ValueError("TTS service does not support PCM")
            if self.voice not in {str(v["payload"]) for v in self.capability["voices"]["items"]}:
                raise ValueError("Requested TTS voice is unavailable")
            self.last_error = None
            return True
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            self.last_error = str(exc); self.unload(); return False

    def synthesize(self, text, cancel_event=None):
        if self._client is None: return {"audio": None, "error": "TTS is not loaded"}
        start = time.monotonic(); first = None
        try:
            if not isinstance(text, str) or not text.strip() or len(text) > 500:
                raise ValueError("TTS input must contain 1–500 characters")
            with self.requests.operation(cancel_event,timeout=self.timeout) as operation:
                data = bytearray()
                with self._client.stream("POST", self.base_url + "/audio/speech", json={
                    "model": self.model, "voice": self.voice, "input": text.strip(),
                    "speed": self.speed, "response_format": "pcm",
                }, extensions=operation.extensions) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes(4096):
                        operation.check()
                        if chunk:
                            first = first or time.monotonic()
                            data.extend(chunk)
                            if len(data) > self.max_audio_seconds * 32000:
                                raise ValueError("TTS PCM exceeds response budget")
                if not data or len(data) % 2: raise ValueError("TTS returned empty or truncated PCM16")
                self.last_error = None
                return {"audio": np.frombuffer(bytes(data), dtype='<i2'), "sample_rate": 16000,
                        "duration": len(data)/32000, "elapsed_seconds": time.monotonic()-start,
                        "first_pcm_seconds": first-start if first else None}
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            self.last_error = str(exc)
            return {"audio": None, "error": str(exc)}

    def synthesize_to_file(self, text, path):
        result = self.synthesize(text)
        if result.get("audio") is None: return False
        Path(path).write_bytes(pcm_wav(result["audio"].tobytes()))
        return True

    def health_check(self):
        if self._client is None: return False
        try:
            speech_capability(self.base_url, "tts", self.model, self.backend_name,min(5,self.timeout))
            return True
        except (httpx.HTTPError, ValueError, KeyError): return False

    def cancel(self, event):
        self.requests.cancel(event)

    def idle(self):
        if not self.requests.idle(): return False
        capability=speech_capability(self.base_url,"tts",self.model,self.backend_name,.5)
        active=capability.get('concurrency',{}).get('active')
        if type(active) is not int or active<0: raise ValueError('Invalid speech concurrency observation')
        return active==0

    def unload(self):
        if self._client: self._client.close()
        self._client = None
        self.capability = None


def create_tts(**kwargs):
    return LocalTTS(**kwargs)
