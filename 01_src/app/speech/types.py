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

"""Timestamped audio frames and speech boundaries.

Extracted from the upstream-derived audio pipeline; behavior is preserved.
"""

import threading
from dataclasses import dataclass
import numpy as np


@dataclass
class SpeechSegment:
    """A completed speech utterance from the VAD."""

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
    utterance_id: str
    started_monotonic: float
    confirmed_monotonic: float
    input_epoch: int = 0
    input_invalidated: threading.Event | None = None


@dataclass(frozen=True)
class AudioChunk:
    data: bytes
    input_epoch: int
    received_monotonic: float
    sequence: int = 0
    input_invalidated: threading.Event | None = None
