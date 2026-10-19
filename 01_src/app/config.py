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
from project_paths import CONFIG_DIR, RUNTIME_DIR, ROOT, artifact_path
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
    chat_template_kwargs: Dict[str, bool] = field(
        default_factory=lambda: {"enable_thinking": False}
    )
    system_prompt: str = "You are a helpful AI assistant."
    system_prompt_no_rag: str = (
        "You are a helpful AI assistant. Answer from your own knowledge."
    )


@dataclass
class STTConfig:
    model: str = "qwen3-asr"
    base_url: str = "http://127.0.0.1:8621/v1"
    backend: str = "llama_qwen3_asr"
    native_base_url: str = "http://127.0.0.1:8089"
    language: str = "zh"
    timeout: float = 60.0
    max_seconds: float = 15.0


@dataclass
class TTSConfig:
    model: str = "matcha-icefall-zh-en"
    base_url: str = "http://127.0.0.1:8621/v1"
    backend: str = "matcha_trt"
    voice: str = "0"
    speed: float = 1.0
    lang: str = "zh"
    timeout: float = 60.0
    max_audio_seconds: float = 30.0
    first_chunk_chars: int = 8
    max_chunk_chars: int = 40


@dataclass
class AudioConfig:
    enabled: bool = True
    input_mode: str = "file"
    output_mode: str = "file"
    sample_rate: int = 16000
    channels: int = 1
    input_device: Optional[str] = "ReSpeaker Lite"
    output_device: Optional[str] = None
    echo_cancellation: bool = True
    output_directory: str = str(RUNTIME_DIR / "runtime-data/reachy-audio")
    max_output_seconds: float = 120.0
    retained_files: int = 10
    voice_interruptions: bool = True
    pending_voice_seconds: float = 30.0
    input_stale_seconds: float = 1.0


@dataclass
class VADConfig:
    speech_threshold: float = 0.008
    silence_duration_ms: int = 500
    lookback_ms: int = 250
    max_speech_secs: int = 15
    chunk_ms: int = 32
    min_utterance_secs: float = 0.3
    min_utterance_rms: float = 0.001
    silero_threshold: float = 0.5
    confirm_speech_ms: int = 160


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
    reconnect_seconds: float = 1.0
    v4l_select_timeout: int = 2
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
class PerceptionConfig:
    enabled: bool = True
    body_enabled: bool = True
    hands_enabled: bool = True
    fps: float = 5.0
    body_threshold: float = 0.5
    body_iou: float = 0.45
    max_people: int = 10
    palm_threshold: float = 0.5
    hand_threshold: float = 0.8
    max_hands: int = 2


@dataclass
class GestureConfig:
    enabled: bool = True
    greetings_enabled: bool = True
    stop_hold_seconds: float = 1.0
    min_observations: int = 5
    max_gap_seconds: float = 0.6
    event_ttl_seconds: float = 0.75
    stationary_palm_widths: float = 0.2
    wave_swing_palm_widths: float = 0.6
    wave_window_seconds: float = 2.0
    wave_reversals: int = 2
    rearm_seconds: float = 0.4
    greeting_cooldown_seconds: float = 5.0
    min_palm_pixels: float = 20.0
    association_palm_widths: float = 2.0
    max_tracks: int = 4


@dataclass
class SessionConfig:
    enabled: bool = True
    max_turns: int = 4


@dataclass
class ActionsConfig:
    enabled: bool = False
    database: str = str(RUNTIME_DIR / "runtime-data/reachy-actions/actions.sqlite")
    max_calls_per_turn: int = 2
    max_pending_reminders: int = 100
    max_records: int = 1000
    max_requests: int = 10000
    notification_ttl_seconds: int = 3600
    timezone: str = "Asia/Shanghai"


@dataclass
class WebConfig:
    ui_fps: float = 10.0
    host: str = "0.0.0.0"
    port: int = 8090
    max_clients: int = 4
    queue_events: int = 16
    queue_bytes: int = 4 * 1024 * 1024
    bridge_events: int = 64
    bridge_bytes: int = 4 * 1024 * 1024
    message_bytes: int = 2 * 1024 * 1024
    json_body_bytes: int = 32768
    ws_message_bytes: int = 4096
    body_readers: int = 4
    body_timeout: float = 10.0
    send_timeout: float = 2.0
    read_workers: int = 2
    read_timeout: float = 2.0


@dataclass
class BackendsConfig:
    stack_root: str = str(RUNTIME_DIR)
    llm_threads: int = 4
    llm_gpu_layers: int = 99
    llm_parallel: int = 1
    llm_cache_ram_mib: int = 256
    asr_threads: int = 2
    asr_gpu_layers: int = 99
    asr_context_tokens: int = 2048
    asr_cache_ram_mib: int = 64
    asr_model_id: str = "assistant-asr"
    # Preserve the already running ASR hint; changes belong to later experiments.
    asr_context: str = "Jetson, TensorRT, TensorFlow, CUDA, Pipecat, Mem0"


_SECTIONS = [
    ("backends", "backends", BackendsConfig),
    ("llm", "llm", LLMConfig),
    ("stt", "stt", STTConfig),
    ("tts", "tts", TTSConfig),
    ("audio", "audio", AudioConfig),
    ("vad", "vad", VADConfig),
    ("vision", "vision", VisionConfig),
    ("face", "face", FaceConfig),
    ("web", "web", WebConfig),
    ("perception", "perception", PerceptionConfig),
    ("gestures", "gestures", GestureConfig),
    ("session", "session", SessionConfig),
    ("actions", "actions", ActionsConfig),
]


@dataclass
class Config:
    backends: BackendsConfig = field(default_factory=BackendsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    web: WebConfig = field(default_factory=WebConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    actions: ActionsConfig = field(default_factory=ActionsConfig)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """加载基础配置，再叠加指定 profile；默认使用 Orin NX 无机身配置。

        启动管理器会保存最终有效配置；运行中的服务共同读取该冻结副本。
        """
        default_path = CONFIG_DIR / "settings.yaml"
        profile_path = config_path or os.environ.get("REACHY_ASSISTANT_CONFIG")
        config_paths = [default_path]
        if profile_path and Path(profile_path).resolve() != default_path.resolve():
            config_paths.append(Path(profile_path))

        # A named profile must exist; invalid config must never silently boot
        # an English/robot default. Preserve upstream dataclasses and overlays.
        if not profile_path:
            config_paths.append(
                default_path.with_name("settings.orin-nx-headless.yaml")
            )
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
                        raise ValueError(
                            f"Invalid configuration value: {section}.{key}"
                        )
                    setattr(obj, key, value)
        # Rebase historical filesystem locations; keep model parameters intact.
        for section, field in (
            ("backends", "stack_root"),
            ("audio", "output_directory"),
            ("actions", "database"),
        ):
            obj = getattr(config, section)
            value = getattr(obj, field)
            if not value.strip():
                raise ValueError("Runtime paths must not be empty")
            path = artifact_path(value)
            setattr(obj, field, str(path if path.is_absolute() else ROOT / path))
        config.validate()
        return config

    def validate(self):
        b = self.backends
        if not (
            1 <= b.llm_threads <= 8
            and 1 <= b.asr_threads <= 8
            and b.llm_parallel == 1
            and 0 <= b.llm_gpu_layers <= 999
            and 0 <= b.asr_gpu_layers <= 999
            and 0 <= b.llm_cache_ram_mib <= 1024
            and 0 <= b.asr_cache_ram_mib <= 1024
            and 512 <= b.asr_context_tokens <= 8192
            and b.asr_model_id
            and len(b.asr_context) <= 1024
            and Path(b.stack_root).expanduser().is_absolute()
        ):
            raise ValueError("Invalid native backend capacity or paths")
        if not (
            self.vad.max_speech_secs + 1 <= self.audio.pending_voice_seconds <= 60
            and 0.032 <= self.audio.input_stale_seconds <= 2
            and 32 <= self.vad.confirm_speech_ms <= 1000
        ):
            raise ValueError("Invalid live voice timing bounds")
        if not (
            0.1 <= self.vision.reconnect_seconds <= 10
            and 1 <= self.vision.v4l_select_timeout <= 3
        ):
            raise ValueError("Invalid camera recovery timing")
        w = self.web
        if not (
            1 <= w.max_clients <= 4
            and 4 <= w.queue_events <= 64
            and 4 <= w.bridge_events <= 256
            and 65536 <= w.message_bytes <= 4 * 1024 * 1024
            and w.message_bytes <= w.queue_bytes <= 16 * 1024 * 1024
            and w.message_bytes <= w.bridge_bytes <= 16 * 1024 * 1024
            and 24576 <= w.json_body_bytes <= 65536
            and 512 <= w.ws_message_bytes <= 65536
            and 1 <= w.body_readers <= 4
            and 1 <= w.read_workers <= 2
        ):
            raise ValueError("Invalid Web resource bounds")
        if not (
            0 < w.body_timeout <= 30
            and 0 < w.send_timeout <= 10
            and 0 < w.read_timeout <= 10
        ):
            raise ValueError("Invalid Web timeouts")
        from zoneinfo import ZoneInfo

        ZoneInfo(self.actions.timezone)
        if not (
            1 <= self.actions.max_calls_per_turn <= 2
            and 1 <= self.actions.max_pending_reminders <= 100
            and self.actions.max_pending_reminders <= self.actions.max_records <= 10000
            and 1 <= self.actions.max_requests <= 100000
            and 1 <= self.actions.notification_ttl_seconds <= 86400
        ):
            raise ValueError("Invalid action capacity")
        if not 1 <= self.session.max_turns <= 16:
            raise ValueError("Session history must contain at most 1–16 turns")
        from urllib.parse import urlparse

        url = urlparse(self.llm.base_url)
        if (
            url.scheme != "http"
            or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise ValueError("llm.base_url must be a local HTTP server root")
        if self.llm.backend != "openai" or not self.llm.model.strip():
            raise ValueError(
                "This profile requires an explicit local OpenAI-compatible model"
            )
        if not 0 < self.llm.max_tokens + 16 < self.llm.context_tokens:
            raise ValueError("Output reservation exceeds context capacity")
        if not 0 <= self.llm.temperature <= 2 or self.llm.timeout <= 0:
            raise ValueError("Invalid LLM sampling or timeout setting")
        if (
            min(
                self.vision.width,
                self.vision.height,
                self.vision.capture_fps,
                self.web.ui_fps,
            )
            <= 0
        ):
            raise ValueError("Camera geometry and rates must be positive")
        if self.vision.frames != 1:
            raise ValueError("The initial 4B profile supports one image per request")
        from app.audio import local_speech_url

        local_speech_url(self.stt.native_base_url.rstrip("/") + "/v1")
        for audio_config in (self.stt, self.tts):
            local_speech_url(audio_config.base_url)
            if audio_config.timeout <= 0:
                raise ValueError("Speech timeout must be positive")
        if (self.audio.sample_rate, self.audio.channels) != (16000, 1):
            raise ValueError("Speech requires 16kHz mono")
        if self.audio.input_mode not in {
            "file",
            "device",
        } or self.audio.output_mode not in {"file", "device"}:
            raise ValueError("Unknown audio mode")
        if self.audio.output_mode == "device" and not self.audio.output_device:
            raise ValueError("Explicit output device required")
        if self.audio.input_mode == "device" and not self.audio.input_device:
            raise ValueError("Explicit input device required")
        if not (1 <= self.stt.max_seconds <= 30 and 0.25 <= self.tts.speed <= 4):
            raise ValueError("Invalid ASR duration or TTS speed")
        if not 1 <= self.tts.first_chunk_chars <= self.tts.max_chunk_chars <= 200:
            raise ValueError("Invalid TTS text chunk bounds")
        if not 0 < self.tts.max_audio_seconds <= self.audio.max_output_seconds <= 120:
            raise ValueError("Invalid generated audio duration limits")
        if not 1 <= self.audio.retained_files <= 100:
            raise ValueError("Invalid audio file retention")
        if not (
            0 < self.vad.silero_threshold < 1
            and 0
            <= self.vad.min_utterance_secs
            < self.vad.max_speech_secs
            <= self.stt.max_seconds
        ):
            raise ValueError("Invalid VAD thresholds or segment duration")
        if self.vad.silence_duration_ms < 32 or not 0 <= self.vad.lookback_ms <= 1000:
            raise ValueError("Invalid VAD endpoint/lookback settings")
        if self.vad.chunk_ms != 32:
            raise ValueError("Silero v6 requires 32ms/512-sample frames")
        if not 1 <= self.web.port <= 65535:
            raise ValueError("Invalid web port")
        if (
            min(
                self.vision.stale_after,
                self.vision.buffer_bytes_limit,
                self.face.fps,
                self.face.stable_frames,
            )
            <= 0
        ):
            raise ValueError("Perception timing and buffer limits must be positive")
        if not 0 < self.face.dead_zone <= self.face.good_frame_zone <= 1:
            raise ValueError("Invalid face framing zones")
        if not 0 < self.face.min_face_size <= 1:
            raise ValueError("Invalid face size threshold")

        if (
            self.perception.fps <= 0
            or not 1 <= self.perception.max_hands <= 4
            or not 1 <= self.perception.max_people <= 100
        ):
            raise ValueError("Invalid perception rate/capacity")
        if not all(
            0 < value < 1
            for value in (
                self.perception.body_threshold,
                self.perception.body_iou,
                self.perception.palm_threshold,
                self.perception.hand_threshold,
            )
        ):
            raise ValueError("Invalid perception threshold")
        g = self.gestures
        import math

        if not all(
            math.isfinite(v) for v in vars(g).values() if type(v) in (int, float)
        ):
            raise ValueError("Gesture parameters must be finite")
        if not (
            0
            < g.stationary_palm_widths
            < g.wave_swing_palm_widths
            < g.association_palm_widths
            <= 4
        ):
            raise ValueError("Invalid gesture motion thresholds")
        if not (
            0 < g.rearm_seconds <= g.stop_hold_seconds <= g.wave_window_seconds <= 5
        ):
            raise ValueError("Invalid gesture evidence windows")
        if not (
            0 < g.max_gap_seconds < g.stop_hold_seconds
            and 0 < g.event_ttl_seconds <= self.vision.stale_after
        ):
            raise ValueError("Invalid gesture freshness bounds")
        if not (
            3 <= g.min_observations <= 32
            and 1 <= g.wave_reversals <= 4
            and 1 <= g.max_tracks <= 8
        ):
            raise ValueError("Invalid gesture capacities")
        if g.min_palm_pixels < 5 or g.greeting_cooldown_seconds < g.wave_window_seconds:
            raise ValueError("Invalid gesture size/cooldown")

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
            _matches_type(k, args[0]) and _matches_type(v, args[1])
            for k, v in value.items()
        )
    if expected is float:
        return type(value) in (int, float)
    return type(value) is expected
