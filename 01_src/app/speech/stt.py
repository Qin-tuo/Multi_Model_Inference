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

"""通过本机语音服务执行受时限约束的整句 ASR，严格统一 WAV/PCM 输入格式。"""

from pathlib import Path
import time
import wave
import httpx
from app.inference.request_io import RequestIO, llama_slots_idle
import numpy as np
from app.speech.audio import local_speech_url, speech_capability, pcm_wav, wav_pcm


class STT:
    """拥有一个语音识别 HTTP 客户端，原生 ASR 模型由独立服务管理。"""

    def __init__(
        self,
        model="qwen3-asr",
        base_url="http://127.0.0.1:8621/v1",
        backend="llama_qwen3_asr",
        language="zh",
        timeout=60.0,
        max_seconds=15,
        native_base_url="http://127.0.0.1:8089",
    ):
        """保存识别模型、接口地址、语言和输入时长限制，初始化单请求控制器。"""
        self.model_name, self.backend, self.language = model, backend, language
        self.base_url = local_speech_url(base_url)
        self.native_base_url = local_speech_url(
            native_base_url.rstrip("/") + "/v1"
        ).removesuffix("/v1")
        self.timeout, self.max_seconds = timeout, max_seconds
        self._client = None
        self.last_error = None
        self.capability = None
        self.requests = RequestIO()

    def load(self):
        """建立本机 HTTP 客户端并核验识别能力、模型和音频契约，失败时回收客户端。"""
        self.unload()
        self._client = httpx.Client(
            timeout=self.timeout,
            trust_env=False,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        )
        try:
            self.capability = speech_capability(
                self.base_url, "asr", self.model_name, self.backend, self.timeout
            )
            self.last_error = None
            return True
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            self.last_error = str(exc)
            self.unload()
            return False

    def transcribe(self, audio, sample_rate=16000, cancel_event=None):
        """将 WAV 文件或一维 PCM/归一化浮点数组转为 16 kHz 单声道输入，提交整句识别。

        限制输入时长与输出文本长度，传播轮次取消，并以错误字段返回格式或网络失败。
        """
        if self._client is None:
            return {"text": "", "error": "ASR is not loaded"}
        start = time.monotonic()
        try:
            if sample_rate != 16000:
                raise ValueError("ASR expects 16kHz mono input")
            if isinstance(audio, (str, Path)):
                path = Path(audio)
                if path.stat().st_size > self.max_seconds * 32000 + 4096:
                    raise ValueError("WAV exceeds input limit")
                pcm = wav_pcm(path.read_bytes(), self.max_seconds)
            else:
                array = np.asarray(audio)
                if array.ndim != 1 or not 0 < array.size <= self.max_seconds * 16000:
                    raise ValueError("ASR requires a bounded nonempty mono array")
                if array.dtype == np.int16:
                    pcm = array.astype("<i2", copy=False).tobytes()
                elif (
                    np.issubdtype(array.dtype, np.floating)
                    and np.all(np.isfinite(array))
                    and np.max(np.abs(array)) <= 1
                ):
                    pcm = np.rint(array * 32767).astype("<i2").tobytes()
                else:
                    raise ValueError("ASR expects normalized floats or int16 PCM")
            with self.requests.operation(
                cancel_event, timeout=self.timeout
            ) as operation:
                response = self._client.post(
                    self.base_url + "/audio/transcriptions",
                    files={"file": ("utterance.wav", pcm_wav(pcm), "audio/wav")},
                    data={
                        "model": self.model_name,
                        "language": self.language,
                        "response_format": "json",
                    },
                    extensions=operation.extensions,
                )
                operation.check()
                response.raise_for_status()
                text = response.json()["text"]
                if not isinstance(text, str) or not text.strip() or len(text) > 4000:
                    raise ValueError("ASR returned an empty or oversized transcript")
                self.last_error = None
                return {
                    "text": text.strip(),
                    "language": self.language,
                    "duration": len(pcm) / 32000,
                    "elapsed_seconds": time.monotonic() - start,
                    "backend": self.backend,
                }
        except (
            httpx.HTTPError,
            ValueError,
            OSError,
            KeyError,
            wave.Error,
            EOFError,
        ) as exc:
            self.last_error = str(exc)
            return {"text": "", "error": str(exc)}

    def get_info(self):
        """返回识别后端、模型及整句文件接口的采样格式信息。"""
        return {
            "backend": self.backend,
            "model": self.model_name,
            "mode": "utterance_file",
            "sample_rate": 16000,
            "channels": 1,
        }

    def health_check(self):
        """以较短时限重新检查语音能力接口，不创建新的识别请求。"""
        if self._client is None:
            return False
        try:
            speech_capability(
                self.base_url,
                "asr",
                self.model_name,
                self.backend,
                min(5, self.timeout),
            )
            return True
        except (httpx.HTTPError, ValueError, KeyError):
            return False

    def cancel(self, event):
        """中断属于指定取消事件的在途识别 HTTP 操作。"""
        self.requests.cancel(event)

    def idle(self):
        """确认本地请求、语音服务并发计数及原生 llama.cpp 槽位均空闲后才允许释放完成。"""
        if not self.requests.idle():
            return False
        capability = speech_capability(
            self.base_url, "asr", self.model_name, self.backend, 0.5
        )
        active = capability.get("concurrency", {}).get("active")
        if type(active) is not int or active < 0:
            raise ValueError("Invalid speech concurrency observation")
        return active == 0 and llama_slots_idle(self.native_base_url)

    def unload(self):
        """关闭本地 HTTP 客户端并清空能力缓存，不终止原生 ASR 服务。"""
        if self._client:
            self._client.close()
        self._client = None
        self.capability = None
