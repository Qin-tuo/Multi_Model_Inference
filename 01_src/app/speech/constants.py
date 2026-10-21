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

"""语音采样和路由常量。

输入输出采用 16 kHz 单声道；Silero 每次接收 512 个样本，即 32 ms。AEC 名称作为每个实例的命名前缀。
"""

SAMPLE_RATE = 16000


SILERO_CHUNK_SAMPLES = 512  # 16 kHz 下每次必须输入 512 个样本，即 32 ms。


CHANNELS = 1


TTS_BREAKS = frozenset(".,;:!?\n")


AEC_SOURCE_NAME = "assistant_aec_source"


AEC_SINK_NAME = "assistant_aec_sink"
