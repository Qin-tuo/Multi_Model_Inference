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
"""本机 TTS 适配器：接收文本并收集受大小、时长和取消约束的 PCM16 音频。"""

import time
from pathlib import Path
import httpx
from app.request_io import RequestIO
import numpy as np
from app.audio import local_speech_url, speech_capability, pcm_wav


class LocalTTS:
    """拥有单个语音合成 HTTP 客户端，实际合成由语音服务承载。"""

    def __init__(
        self,
        voice="0",
        speed=1.0,
        lang="zh",
        model="matcha-icefall-zh-en",
        base_url="http://127.0.0.1:8621/v1",
        backend="matcha_trt",
        timeout=60.0,
        max_audio_seconds=30.0,
    ):
        """保存语音模型、音色、速度和单次音频时长上限，初始化请求控制器。"""
        self.voice, self.speed, self.lang = voice, speed, lang
        self.model, self.backend_name = model, backend
        self.base_url = local_speech_url(base_url)
        self.timeout, self.max_audio_seconds = timeout, max_audio_seconds
        self._client = None
        self.last_error = None
        self.capability = None
        self.requests = RequestIO()

    def load(self):
        """连接语音服务并确认模型、后端、PCM 输出和目标音色均可用。"""
        self.unload()
        self._client = httpx.Client(
            timeout=self.timeout,
            trust_env=False,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        )
        try:
            self.capability = speech_capability(
                self.base_url, "tts", self.model, self.backend_name, self.timeout
            )
            if "pcm" not in self.capability["audio"]["response_formats"]:
                raise ValueError("TTS service does not support PCM")
            if self.voice not in {
                str(v["payload"]) for v in self.capability["voices"]["items"]
            }:
                raise ValueError("Requested TTS voice is unavailable")
            self.last_error = None
            return True
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            self.last_error = str(exc)
            self.unload()
            return False

    def synthesize(self, text, cancel_event=None):
        """流式读取合成的 PCM16 字节，记录首块到达时间并在总时限与音频预算内收集结果。

        返回一维 int16 数组、采样率和耗时；主动取消异常向上传播，已捕获的格式或网络错误写入 error 字段。
        """
        if self._client is None:
            return {"audio": None, "error": "TTS is not loaded"}
        start = time.monotonic()
        first = None
        try:
            if not isinstance(text, str) or not text.strip() or len(text) > 500:
                raise ValueError("TTS input must contain 1–500 characters")
            with self.requests.operation(
                cancel_event, timeout=self.timeout
            ) as operation:
                data = bytearray()
                with self._client.stream(
                    "POST",
                    self.base_url + "/audio/speech",
                    json={
                        "model": self.model,
                        "voice": self.voice,
                        "input": text.strip(),
                        "speed": self.speed,
                        "response_format": "pcm",
                    },
                    extensions=operation.extensions,
                ) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes(4096):
                        operation.check()
                        if chunk:
                            first = first or time.monotonic()
                            data.extend(chunk)
                            if len(data) > self.max_audio_seconds * 32000:
                                raise ValueError("TTS PCM exceeds response budget")
                if not data or len(data) % 2:
                    raise ValueError("TTS returned empty or truncated PCM16")
                self.last_error = None
                return {
                    "audio": np.frombuffer(bytes(data), dtype="<i2"),
                    "sample_rate": 16000,
                    "duration": len(data) / 32000,
                    "elapsed_seconds": time.monotonic() - start,
                    "first_pcm_seconds": first - start if first else None,
                }
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            self.last_error = str(exc)
            return {"audio": None, "error": str(exc)}

    def synthesize_to_file(self, text, path):
        """合成文本后直接写出 WAV，并返回是否成功；轮次级暂存/提交由 SpeechOutput 另行管理。"""
        result = self.synthesize(text)
        if result.get("audio") is None:
            return False
        Path(path).write_bytes(pcm_wav(result["audio"].tobytes()))
        return True

    def health_check(self):
        """在短时限内检查 TTS 能力与格式契约，避免把接口可连接等同于模型正确。"""
        if self._client is None:
            return False
        try:
            speech_capability(
                self.base_url,
                "tts",
                self.model,
                self.backend_name,
                min(5, self.timeout),
            )
            return True
        except (httpx.HTTPError, ValueError, KeyError):
            return False

    def cancel(self, event):
        """中断与指定事件对应的合成 HTTP 操作。"""
        self.requests.cancel(event)

    def idle(self):
        """同时确认本地 HTTP 操作结束和语音服务 active 计数为零。"""
        if not self.requests.idle():
            return False
        capability = speech_capability(
            self.base_url, "tts", self.model, self.backend_name, 0.5
        )
        active = capability.get("concurrency", {}).get("active")
        if type(active) is not int or active < 0:
            raise ValueError("Invalid speech concurrency observation")
        return active == 0

    def unload(self):
        """关闭本地客户端并清空能力缓存，不停止独立 TTS 服务进程。"""
        if self._client:
            self._client.close()
        self._client = None
        self.capability = None


def create_tts(**kwargs):
    """使用统一构造入口创建当前的本机语音合成适配器。"""
    return LocalTTS(**kwargs)
