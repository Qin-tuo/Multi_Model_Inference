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

"""可取消的物理音频播放，始终只管理当前轮次拥有的播放器进程。"""

import time
import subprocess
import threading
from typing import Optional
import numpy as np


class Playback:
    """以同一把锁协调播放器创建与中断，避免取消操作误伤其他轮次或系统播放器。"""

    def __init__(self, event):
        """绑定取消事件，初始化进程引用和所有权锁。"""
        self.event = event
        self._lock = threading.Lock()
        self._process = None

    def start(self, cmd):
        """在锁内检查取消状态后启动播放器，返回持有 stdin 管道的子进程。"""
        from app.request_io import TurnCancelled

        with self._lock:
            if self.event.is_set():
                raise TurnCancelled("Playback cancelled")
            self._process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return self._process

    def cancel(self):
        """请求终止当前拥有的播放进程；取消事件由上层设置，不操作其他播放器。"""
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass

    def clear(self, process):
        """仅当传入进程仍是本实例所有者时清除引用，避免旧清理覆盖新进程。"""
        with self._lock:
            if self._process is process:
                self._process = None


def play_audio(
    audio: np.ndarray,
    sample_rate: int,
    sink: Optional[str] = None,
    playback=None,
    on_start=None,
):
    """将 16 kHz 单声道 PCM16 送往明确指定的 ALSA/PulseAudio 输出，并等待受时限约束的进程回执。

    持续响应取消，只回收自己启动的进程；进程完成不等于已经验证扬声器实际可听。
    """
    if not sink:
        raise ValueError("Explicit speaker device is required")
    if audio.ndim != 1 or audio.dtype != np.int16 or sample_rate != 16000:
        raise ValueError("Playback requires mono PCM16 at 16kHz")
    if sink.startswith(("hw:", "plughw:")):
        cmd = [
            "aplay",
            "-D",
            sink,
            "-f",
            "S16_LE",
            "-r",
            str(sample_rate),
            "-c",
            "1",
            "-t",
            "raw",
            "-q",
        ]
    else:
        cmd = [
            "paplay",
            f"--device={sink}",
            "--format=s16le",
            f"--rate={sample_rate}",
            "--channels=1",
            "--raw",
        ]
    from app.request_io import TurnCancelled

    playback = playback or Playback(threading.Event())
    process = playback.start(cmd)
    started = time.monotonic()
    try:
        if on_start:
            on_start()
        data = audio.tobytes()
        deadline = started + audio.size / sample_rate + 5
        while True:
            if playback.event.is_set():
                raise TurnCancelled("Playback cancelled")
            try:
                _, error = process.communicate(data, timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                data = None
                if time.monotonic() >= deadline:
                    raise RuntimeError("Playback timed out")
        if playback.event.is_set():
            raise TurnCancelled("Playback cancelled")
        if process.returncode != 0:
            raise RuntimeError(
                f"Playback failed: {error.decode(errors='replace')[:300]}"
            )
        return {
            "kind": "device_completed",
            "submitted_monotonic": started,
            "completed_monotonic": time.monotonic(),
            "frames": audio.size,
        }
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        playback.clear(process)
