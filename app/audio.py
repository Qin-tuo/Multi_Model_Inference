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

"""Audio formats, local service capabilities and scoped device discovery."""

import subprocess
import io
import wave
from urllib.parse import urlsplit
from pathlib import Path
from typing import Optional


def find_alsa_device(
    name_hint: str = "USB Audio",
    direction: str = "input",
) -> Optional[tuple[int, int, str]]:
    """Find ALSA device by name substring. Returns (card, device, name) or None."""
    cmd = "arecord" if direction == "input" else "aplay"
    try:
        r = subprocess.run([cmd, "-l"], capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    needle = name_hint.lower()
    for line in r.stdout.splitlines():
        if "card" in line.lower() and needle in line.lower():
            try:
                parts = line.split()
                card = int(parts[parts.index("card") + 1].rstrip(":,"))
                dev = int(parts[parts.index("device") + 1].rstrip(":,"))
                name = line.split("[")[1].split("]")[0] if "[" in line else name_hint
                return card, dev, name
            except (ValueError, IndexError):
                continue
    return None


def local_speech_url(url: str) -> str:
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") != "/v1"):
        raise ValueError("Speech endpoint must be local HTTP /v1")
    return url.rstrip("/")


def speech_capability(client, url, kind, model, backend):
    response = client.get(url + "/capabilities")
    response.raise_for_status()
    capability = response.json()[kind]
    if capability.get("ready") is not True or capability.get("model_id") != model or capability.get("backend") != backend:
        raise ValueError(f"Configured {kind} model/backend is not ready")
    audio = capability["audio"]
    if (audio["sample_rate"], audio["channels"], audio["sample_format"]) != (16000, 1, "pcm_s16le"):
        raise ValueError(f"Unsupported {kind} audio contract")
    return capability


def wav_pcm(data: bytes, max_seconds: float = 15) -> bytes:
    """Strict PCM16/16kHz/mono; never silently reinterpret or resample bytes."""
    if len(data) > int(max_seconds * 32000) + 4096:
        raise ValueError("WAV exceeds input size limit")
    try:
        with wave.open(io.BytesIO(data), "rb") as source:
            if (source.getframerate(), source.getnchannels(), source.getsampwidth(), source.getcomptype()) != (16000, 1, 2, "NONE"):
                raise ValueError("Expected PCM16 16kHz mono WAV")
            frames = source.getnframes()
            if not 0 < frames <= max_seconds * 16000:
                raise ValueError("WAV duration exceeds the configured limit or is empty")
            pcm = source.readframes(frames)
            if len(pcm) != frames * 2:
                raise ValueError("WAV data is truncated")
    except (wave.Error, EOFError) as exc:
        raise ValueError("Invalid PCM WAV container") from exc
    return pcm


def pcm_wav(pcm: bytes) -> bytes:
    if not pcm or len(pcm) % 2:
        raise ValueError("Invalid PCM16 sample boundary")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000)
        output.writeframes(pcm)
    return buffer.getvalue()
