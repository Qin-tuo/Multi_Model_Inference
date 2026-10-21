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

"""通过 parecord/arecord 采集有界 PCM 块，并管理设备路由、输入批次和关闭顺序。"""

import time
import subprocess
import threading
import queue
from typing import Optional
from rich.console import Console
from app.speech.audio import chunk_rms
from app.speech.constants import CHANNELS, SAMPLE_RATE
from app.speech.devices import (
    PulseAudioAEC,
    SpeakerSelector,
    _pa_sink_label,
    find_pa_source,
)
from app.speech.types import AudioChunk


class MicRecorder:
    """拥有采集子进程和读取线程；用输入批次与失效事件隔离暂停、切换和失败前的音频。"""

    def __init__(self, console: Console, chunk_ms: int = 32):
        """按块时长计算 PCM 字节数，创建最多 64 块的输入队列及采集/路由状态锁。"""
        self.console = console
        self.chunk_ms = chunk_ms
        self.chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
        self.chunk_bytes = self.chunk_samples * CHANNELS * 2
        self.audio_q: queue.Queue[AudioChunk] = queue.Queue(maxsize=64)
        self.last_error = None
        self._reader_thread = None
        self._input_lock = threading.Lock()
        self.input_epoch = 0
        self.input_invalidated = threading.Event()
        self._sequence = 0
        self.suppressed_reason = None
        self.last_chunk_monotonic = None
        self.listening = threading.Event()
        self.listening.set()
        self.alive = True
        self._proc: Optional[subprocess.Popen] = None
        self._route_lock = threading.RLock()
        self._hw = ""
        self.pa_source: Optional[str] = None
        self.pa_sink: Optional[str] = None
        self.speaker_selector: Optional[SpeakerSelector] = None
        self.aec: Optional[PulseAudioAEC] = None

    def start(
        self,
        hw: str,
        mic_hint: str,
        speaker_hint: Optional[str] = None,
        echo_cancellation: bool = True,
    ) -> bool:
        """重新初始化录音、扬声器选择和可选 AEC，并确认启动后实际收到音频。

        AEC 不可用时可尝试直接音频路由；设备忙会明确失败，不终止桌面音频服务。
        """
        self.stop()
        self.alive = True
        self.last_error = None
        self.suppressed_reason = None
        self.flush()
        self.listening.set()

        self._hw = hw
        self.pa_source = find_pa_source(mic_hint)
        self.speaker_selector = SpeakerSelector(
            preferred_hint=speaker_hint,
            fallback_hint=mic_hint,
        )
        self.pa_sink = self.speaker_selector.get_sink()

        if self.pa_source:
            if echo_cancellation and self.pa_sink:
                self.aec = PulseAudioAEC(self.pa_source)
                if self.aec.start(self.pa_sink):
                    self.console.print(
                        f"  AEC: [green]WebRTC active[/green] "
                        f"(Microphone → {_pa_sink_label(self.pa_sink)})"
                    )
                    self.pa_sink = self.aec.sink_name
                else:
                    self.console.print(
                        f"  [yellow]AEC unavailable: {self.aec.last_error}; "
                        "using direct audio[/yellow]"
                    )
            self.console.print(f"  PA source: {self.pa_source.split('.')[-2]}")
        else:
            self.console.print(
                "  [yellow]PA source not found, using ALSA direct[/yellow]"
            )
            # A busy ALSA device fails explicitly; never stop the desktop audio service.

        if not self._start_capture():
            if self.aec and self.aec.active:
                self.console.print(
                    "  [yellow]AEC capture failed; using direct audio[/yellow]"
                )
                self.aec.stop()
                self.pa_sink = self.speaker_selector.get_sink()
                if not self._start_capture():
                    return False
            else:
                return False

        if not self._check_capture():
            self.last_error = "Microphone produced no startup audio"
            self.stop()
            return False
        return True

    def _record_command(self) -> list[str]:
        """按当前 PulseAudio/AEC 路由或 ALSA 设备生成单声道 PCM16 录音命令。"""
        if self.pa_source:
            source = (
                self.aec.source_name if self.aec and self.aec.active else self.pa_source
            )
            return [
                "parecord",
                "-d",
                source,
                "--format=s16le",
                f"--rate={SAMPLE_RATE}",
                f"--channels={CHANNELS}",
                "--raw",
            ]
        plughw = self._hw.replace("hw:", "plughw:")
        return [
            "arecord",
            "-D",
            plughw,
            "-f",
            "S16_LE",
            "-r",
            str(SAMPLE_RATE),
            "-c",
            str(CHANNELS),
            "-t",
            "raw",
        ]

    def _start_capture(self) -> bool:
        """有限次启动采集子进程，成功后创建读取线程，失败则回收管道并记录原因。"""
        rec_cmd = self._record_command()
        for attempt in range(3):
            proc = subprocess.Popen(
                rec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            time.sleep(0.5)
            if proc.poll() is None:
                self._proc = proc
                self._reader_thread = threading.Thread(
                    target=self._reader,
                    args=(proc,),
                    name="microphone-reader",
                    daemon=True,
                )
                self._reader_thread.start()
                return True
            err = proc.stderr.read().decode(errors="replace").strip()
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
            self.last_error = (
                err or f"Microphone process exited with status {proc.returncode}"
            )
            self.console.print(f"  [red]Mic attempt {attempt + 1} failed: {err}[/red]")
            time.sleep(1)
        self._proc = None
        return False

    def _check_capture(self):
        """消费少量启动音频确认采集有数据，并显示当前能量水平；安静不等于采集失败。"""
        time.sleep(0.5)
        test_chunks = []
        for _ in range(10):
            try:
                test_chunks.append(self.audio_q.get(timeout=0.5))
            except queue.Empty:
                break
        if test_chunks:
            rms = chunk_rms(b"".join(chunk.data for chunk in test_chunks))
            if rms > 0.003:
                self.console.print("  Mic: [green]✓ live[/green]")
            else:
                self.console.print("  Mic: [yellow]quiet, waiting for speech[/yellow]")
        else:
            running = self._proc is not None and self._proc.poll() is None
            self.console.print(
                f"  [yellow]Mic: no startup audio; capture running: {running}[/yellow]"
            )
        return bool(test_chunks)

    def _stop_capture(self):
        """只终止本实例录音进程，等待读取线程结束后关闭管道；未退出则报告释放失败。"""
        proc = self._proc
        self._proc = None
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            if self._reader_thread:
                self._reader_thread.join(timeout=2)
                if self._reader_thread.is_alive():
                    raise RuntimeError("Microphone reader has not released")
                self._reader_thread = None
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()

    def _rebuild_aec_route(self) -> bool:
        """在路由锁内暂停旧输入、重建 AEC 并重启录音，再恢复用户原有监听状态。"""
        with self._route_lock:
            if not self.speaker_selector:
                return False
            physical_sink = self.speaker_selector.get_sink()
            was_listening = self.listening.is_set()
            self.pause()
            self._stop_capture()

            aec_ok = False
            if self.aec and self.pa_source and physical_sink:
                aec_ok = self.aec.start(physical_sink)
            self.pa_sink = self.aec.sink_name if aec_ok else physical_sink

            capture_ok = self._start_capture()
            if not capture_ok and aec_ok:
                self.aec.stop()
                self.pa_sink = physical_sink
                capture_ok = self._start_capture()

            if was_listening:
                self.resume()
            else:
                self.flush()
            return capture_ok

    def get_pa_sink(self) -> Optional[str]:
        """返回当前实际播放路由，可能是 AEC 虚拟 sink 而非物理扬声器。"""
        with self._route_lock:
            return self.pa_sink

    def _add_aec_state(self, state: dict) -> dict:
        """把 AEC 是否启用及其错误信息补入设备状态。"""
        state["aec_enabled"] = bool(self.aec and self.aec.active)
        if self.aec and self.aec.last_error and not self.aec.active:
            state["aec_error"] = self.aec.last_error
        return state

    def speaker_state(self) -> dict:
        """刷新扬声器状态，必要时随输出设备变化重建 AEC，并暴露采集停止原因。"""
        if not self.speaker_selector:
            return {"speakers": [], "selected": None, "aec_enabled": False}
        previous_sink = self.speaker_selector.get_sink()
        state = self.speaker_selector.state()
        selected_sink = self.speaker_selector.get_sink()
        if (
            self.alive
            and selected_sink
            and previous_sink != selected_sink
            and self.aec
            and self.aec.sink_master != selected_sink
        ):
            self._rebuild_aec_route()
        if not self.alive:
            state["error"] = (
                self.last_error or "Microphone input stopped; restart the application"
            )
        return self._add_aec_state(state)

    def select_speaker(self, sink_id: str) -> dict:
        """校验录音仍可用后切换物理扬声器，重建相关采集/AEC 路由。"""
        if not self.alive:
            raise RuntimeError(
                "Microphone input stopped; restart the application after checking the device"
            )
        if not self.speaker_selector:
            return {
                "speakers": [],
                "selected": None,
                "aec_enabled": False,
                "error": "Speaker routing is not initialized",
            }
        state = self.speaker_selector.select(sink_id)
        if state.get("error"):
            return self._add_aec_state(state)
        if not self._rebuild_aec_route():
            state["error"] = "Could not restart microphone after speaker switch"
        return self._add_aec_state(state)

    def set_speaker_volume(self, volume: int) -> dict:
        """委托扬声器选择器设置物理输出音量，并附加 AEC 状态。"""
        if not self.speaker_selector:
            return {
                "speakers": [],
                "selected": None,
                "aec_enabled": False,
                "error": "Speaker routing is not initialized",
            }
        return self._add_aec_state(self.speaker_selector.set_volume(volume))

    def _reader(self, proc: subprocess.Popen):
        """读取完整 PCM 块并附加序号、批次和接收时间，非阻塞放入有界队列。

        部分帧、队列溢出或采集结束会使当前话语失效，避免消费不连续的旧音频。
        """
        while self.alive and proc.poll() is None:
            epoch = self.input_epoch
            raw = proc.stdout.read(self.chunk_bytes)
            if raw:
                self.last_chunk_monotonic = time.monotonic()
            if not raw:
                if proc.poll() is not None and proc is self._proc:
                    err = proc.stderr.read().decode(errors="replace").strip()
                    if err:
                        self.console.print(f"\n  [red]audio capture died: {err}[/red]")
                break
            with self._input_lock:
                if (
                    proc is self._proc
                    and self.listening.is_set()
                    and epoch == self.input_epoch
                ):
                    if len(raw) != self.chunk_bytes:
                        self.last_error = "Microphone returned partial PCM frame"
                        self.alive = False
                        self._invalidate_input_locked()
                        break
                    self._sequence += 1
                    try:
                        self.audio_q.put_nowait(
                            AudioChunk(
                                raw,
                                epoch,
                                time.monotonic(),
                                self._sequence,
                                self.input_invalidated,
                            )
                        )
                    except queue.Full:
                        self.last_error = (
                            "Microphone input overflow; utterance invalidated"
                        )
                        self.alive = False
                        self._invalidate_input_locked()
                        break
        if proc is self._proc and self.alive:
            self.last_error = self.last_error or "Microphone capture ended"
            self.alive = False
            with self._input_lock:
                self._invalidate_input_locked()

    def _invalidate_input_locked(self):
        """在输入锁内触发旧批次失效事件，切换批次、重置序号并清空队列。"""
        self.input_invalidated.set()
        self.input_invalidated = threading.Event()
        self.input_epoch += 1
        self._sequence = 0
        self.flush()

    def suppress(self, reason):
        """更新处理门控原因并使旧输入失效，同时保留用户的监听/静音偏好。"""
        with self._input_lock:
            if reason != self.suppressed_reason:
                self.suppressed_reason = reason
                self._invalidate_input_locked()

    def fail_input(self, reason):
        """记录输入故障、停止有效采集状态并使所有当前音频证据失效。"""
        with self._input_lock:
            self.last_error = reason
            self.alive = False
            self._invalidate_input_locked()

    def input_state(self):
        """返回监听、批次、门控原因和队列占用，供控制器与界面观察。"""
        return {
            "alive": self.alive,
            "listening": self.listening.is_set(),
            "input_epoch": self.input_epoch,
            "suppressed_reason": self.suppressed_reason,
            "error": self.last_error,
            "queued_chunks": self.audio_q.qsize(),
            "queue_capacity": self.audio_q.maxsize,
        }

    def flush(self):
        """非阻塞清空排队音频，不改变用户的监听开关。"""
        while not self.audio_q.empty():
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break

    def pause(self):
        """关闭监听并废弃当前批次，旧的半句话不会在恢复后续接。"""
        with self._input_lock:
            self.listening.clear()
            self._invalidate_input_locked()

    def resume(self):
        """仅在采集仍存活时开启监听，并先创建新的输入批次。"""
        with self._input_lock:
            if not self.alive:
                raise ValueError(
                    "Microphone capture stopped; restart the application after checking the device"
                )
            self._invalidate_input_locked()
            self.listening.set()

    def stop(self):
        """停止有效输入，依次回收录音进程/线程和本实例 AEC，再清空队列。"""
        self.alive = False
        with self._input_lock:
            self._invalidate_input_locked()
        with self._route_lock:
            try:
                self._stop_capture()
            finally:
                if self.aec:
                    self.aec.stop()
                self.flush()
