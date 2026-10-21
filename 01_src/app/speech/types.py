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

"""采集帧、语音起点与完整语句的数据结构，统一携带时间和输入失效信息。"""

import threading
from dataclasses import dataclass
import numpy as np


@dataclass
class SpeechSegment:
    """VAD 完成的一段语音，包含音频、原始块、时长、输入批次与结束原因。"""

    audio: np.ndarray
    raw_chunks: list
    duration: float
    rms: float
    start_time: float
    end_time: float
    input_epoch: int = 0
    utterance_id: str | None = None
    input_invalidated: threading.Event | None = None
    end_reason: str = "silence"


@dataclass(frozen=True)
class SpeechOnset:
    """已确认的语音起点；记录起始/确认时刻及话语身份，用于申请轮次和打断。"""

    utterance_id: str
    started_monotonic: float
    confirmed_monotonic: float
    input_epoch: int = 0
    input_invalidated: threading.Event | None = None


@dataclass(frozen=True)
class AudioChunk:
    """一块带顺序号与输入批次的 PCM 数据；失效事件可使下游丢弃已排队的旧输入。"""

    data: bytes
    input_epoch: int
    received_monotonic: float
    sequence: int = 0
    input_invalidated: threading.Event | None = None
