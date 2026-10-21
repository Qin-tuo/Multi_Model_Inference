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

"""音频容器、采样格式校验、本机语音能力检查和 ALSA 设备查询。"""

import subprocess
import io
import wave
import numpy as np
from urllib.parse import urlsplit
from typing import Optional


def find_alsa_device(
    name_hint: str = "USB Audio",
    direction: str = "input",
) -> Optional[tuple[int, int, str]]:
    """按名称片段查找 ALSA 输入或输出设备，返回声卡号、设备号和名称；找不到时返回 None。"""
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
    """校验语音服务只能使用无凭据的本机 HTTP /v1 地址，并规范末尾斜杠。"""
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1"
    ):
        raise ValueError("Speech endpoint must be local HTTP /v1")
    return url.rstrip("/")


def speech_capability(url, kind, model, backend, timeout):
    """读取受时限与大小约束的能力接口，确认指定模型、后端及 16 kHz 单声道 PCM16 格式就绪。"""
    from app.inference.request_io import control_json

    payload = control_json(url + "/capabilities", timeout)
    if not isinstance(payload, dict) or not isinstance(payload.get(kind), dict):
        raise ValueError(f"Invalid {kind} capability response")
    capability = payload[kind]
    if (
        capability.get("ready") is not True
        or capability.get("model_id") != model
        or capability.get("backend") != backend
    ):
        raise ValueError(f"Configured {kind} model/backend is not ready")
    audio = capability.get("audio")
    if not isinstance(audio, dict) or (
        audio.get("sample_rate"),
        audio.get("channels"),
        audio.get("sample_format"),
    ) != (16000, 1, "pcm_s16le"):
        raise ValueError(f"Unsupported {kind} audio contract")
    return capability


def wav_pcm(data: bytes, max_seconds: float = 15) -> bytes:
    """从 WAV 容器提取严格的 16 kHz、单声道、PCM16 数据。

    限制文件大小和时长，拒绝截断数据或其他采样格式，不隐式重采样。
    """
    if len(data) > int(max_seconds * 32000) + 4096:
        raise ValueError("WAV exceeds input size limit")
    try:
        with wave.open(io.BytesIO(data), "rb") as source:
            if (
                source.getframerate(),
                source.getnchannels(),
                source.getsampwidth(),
                source.getcomptype(),
            ) != (16000, 1, 2, "NONE"):
                raise ValueError("Expected PCM16 16kHz mono WAV")
            frames = source.getnframes()
            if not 0 < frames <= max_seconds * 16000:
                raise ValueError(
                    "WAV duration exceeds the configured limit or is empty"
                )
            pcm = source.readframes(frames)
            if len(pcm) != frames * 2:
                raise ValueError("WAV data is truncated")
    except (wave.Error, EOFError) as exc:
        raise ValueError("Invalid PCM WAV container") from exc
    return pcm


def pcm_wav(pcm: bytes) -> bytes:
    """将非空、按双字节样本对齐的 PCM16 包装成 16 kHz 单声道 WAV 字节。"""
    if not pcm or len(pcm) % 2:
        raise ValueError("Invalid PCM16 sample boundary")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm)
    return buffer.getvalue()


def chunk_rms(raw: bytes) -> float:
    """把 PCM16 音频归一化到浮点范围，计算此块的均方根幅值。"""
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(pcm**2)))
