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

"""Configuration — loads settings.yaml into typed dataclasses."""

import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, get_args, get_origin, get_type_hints, Union
from pathlib import Path
import yaml


@dataclass
class LLMConfig:
    model: str = ""
    base_url: str = "http://localhost:8080"
    backend: str = "openai"
    max_tokens: int = 512
    temperature: float = 0.7
    timeout: float = 120.0
    context_tokens: int = 2048
    chat_template_kwargs: Dict[str, bool] = field(default_factory=lambda: {"enable_thinking": False})
    system_prompt: str = "You are a helpful AI assistant."
    system_prompt_no_rag: str = "You are a helpful AI assistant. Answer from your own knowledge."


@dataclass
class STTConfig:
    model: str = "base.en"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "en"
    beam_size: int = 1


@dataclass
class TTSConfig:
    voice: str = "af_sarah"
    speed: float = 1.0
    lang: str = "en-us"
    first_chunk_words: int = 3
    max_chunk_words: int = 8


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 2
    input_device: Optional[str] = "Reachy Mini Audio"
    output_device: Optional[str] = None
    echo_cancellation: bool = True


@dataclass
class VADConfig:
    speech_threshold: float = 0.008
    silence_duration_ms: int = 500
    lookback_ms: int = 250
    max_speech_secs: int = 15
    chunk_ms: int = 30
    min_utterance_secs: float = 0.3
    min_utterance_rms: float = 0.001
    silero_threshold: float = 0.5


@dataclass
class VisionConfig:
    camera_device: Union[int, str] = 0
    width: int = 640
    height: int = 480
    jpeg_quality: int = 80
    frames: int = 3
    capture_fps: float = 10.0
    stale_after: float = 1.0
    buffer_bytes_limit: int = 16 * 1024 * 1024
    system_prompt: str = (
        "You are a vision assistant on an NVIDIA Jetson device with a live camera. "
        "Answer in one to two sentences. Be direct and concise."
    )
    few_shot: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class FaceConfig:
    enabled: bool = True
    fps: float = 5.0
    dead_zone: float = 0.12
    good_frame_zone: float = 0.18
    min_face_size: float = 0.06
    stable_frames: int = 2


@dataclass
class WebConfig:
    ui_fps: float = 10.0
    host: str = "0.0.0.0"
    port: int = 8090


_SECTIONS = [
    ("llm", "llm", LLMConfig),
    ("stt", "stt", STTConfig),
    ("tts", "tts", TTSConfig),
    ("audio", "audio", AudioConfig),
    ("vad", "vad", VADConfig),
    ("vision", "vision", VisionConfig),
    ("face", "face", FaceConfig),
    ("web", "web", WebConfig),
]


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    web: WebConfig = field(default_factory=WebConfig)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        default_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        profile_path = config_path or os.environ.get("REACHY_ASSISTANT_CONFIG")
        config_paths = [default_path]
        if profile_path and Path(profile_path).resolve() != default_path.resolve():
            config_paths.append(Path(profile_path))

        # A named profile must exist; invalid config must never silently boot
        # an English/robot default. Preserve upstream dataclasses and overlays.
        if not profile_path:
            config_paths.append(default_path.with_name("settings.orin-nx-headless.yaml"))
        config = cls()
        sections = {key: (attr, kind) for key, attr, kind in _SECTIONS}
        for path in config_paths:
            with path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict) or set(data) - sections.keys():
                raise ValueError(f"Unknown or invalid configuration section in {path}")
            for section, values in data.items():
                attr, kind = sections[section]
                if not isinstance(values, dict):
                    raise ValueError(f"{section} must be a mapping")
                obj = getattr(config, attr)
                hints = get_type_hints(kind)
                for key, value in values.items():
                    if key not in hints or not _matches_type(value, hints[key]):
                        raise ValueError(f"Invalid configuration value: {section}.{key}")
                    setattr(obj, key, value)
        config.validate()
        return config

    def validate(self):
        from urllib.parse import urlparse
        url = urlparse(self.llm.base_url)
        if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
                or url.username or url.password or url.query or url.fragment or url.path not in {"", "/"}):
            raise ValueError("llm.base_url must be a local HTTP server root")
        if self.llm.backend != "openai" or not self.llm.model.strip():
            raise ValueError("This profile requires an explicit local OpenAI-compatible model")
        if not 0 < self.llm.max_tokens + 16 < self.llm.context_tokens:
            raise ValueError("Output reservation exceeds context capacity")
        if not 0 <= self.llm.temperature <= 2 or self.llm.timeout <= 0:
            raise ValueError("Invalid LLM sampling or timeout setting")
        if min(self.vision.width, self.vision.height, self.vision.capture_fps, self.web.ui_fps) <= 0:
            raise ValueError("Camera geometry and rates must be positive")
        if self.vision.frames != 1:
            raise ValueError("The initial 4B profile supports one image per request")
        if not 1 <= self.web.port <= 65535:
            raise ValueError("Invalid web port")
        if min(self.vision.stale_after, self.vision.buffer_bytes_limit, self.face.fps, self.face.stable_frames) <= 0:
            raise ValueError("Perception timing and buffer limits must be positive")
        if not 0 < self.face.dead_zone <= self.face.good_frame_zone <= 1:
            raise ValueError("Invalid face framing zones")
        if not 0 < self.face.min_face_size <= 1:
            raise ValueError("Invalid face size threshold")

    def effective(self) -> dict:
        return asdict(self)


def _matches_type(value, expected) -> bool:
    origin, args = get_origin(expected), get_args(expected)
    if origin is Union:
        return any(_matches_type(value, arg) for arg in args)
    if expected is type(None):
        return value is None
    if origin is list:
        return isinstance(value, list) and all(_matches_type(v, args[0]) for v in value)
    if origin is dict:
        return isinstance(value, dict) and all(
            _matches_type(k, args[0]) and _matches_type(v, args[1]) for k, v in value.items())
    if expected is float:
        return type(value) in (int, float)
    return type(value) is expected
