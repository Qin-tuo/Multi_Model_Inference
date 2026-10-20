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

"""Shared audio sample geometry and routing identifiers.

Extracted from the upstream-derived audio pipeline; behavior is preserved.
"""

SAMPLE_RATE = 16000


SILERO_CHUNK_SAMPLES = 512  # Silero VAD requires exactly 512 samples (32ms) at 16kHz


CHANNELS = 1


TTS_BREAKS = frozenset(".,;:!?\n")


AEC_SOURCE_NAME = "assistant_aec_source"


AEC_SINK_NAME = "assistant_aec_sink"
