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

"""PulseAudio 设备发现、扬声器选择和 WebRTC 回声消除路由。

只管理本应用创建的路由与模块，不通过停止系统音频服务争抢设备。
"""

import subprocess
import threading
import uuid
from typing import Optional
from app.speech.constants import AEC_SINK_NAME, AEC_SOURCE_NAME


_ALSA_ERR_T = None
_alsa_handler = None
try:
    import ctypes

    _ALSA_ERR_T = ctypes.CFUNCTYPE(
        None,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
    )
    # 此回调仅抑制 ALSA 库的噪声日志；实际采集错误仍由设备接口返回。
    _alsa_handler = _ALSA_ERR_T(lambda *_: None)
    ctypes.cdll.LoadLibrary("libasound.so.2").snd_lib_error_set_handler(_alsa_handler)
except Exception:
    pass


def _pa_match(needle: str, haystack: str) -> bool:
    """将空格与下划线规范化后进行不区分大小写的设备名称片段匹配。"""
    n = needle.lower().replace(" ", "_")
    h = haystack.lower().replace(" ", "_")
    return n in h


def find_pa_source(name_hint: str) -> Optional[str]:
    """查找匹配名称的 PulseAudio 输入源，排除扬声器 monitor 回录源。"""
    try:
        r = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in r.stdout.splitlines():
            parts = line.split("\t")
            if (
                len(parts) >= 2
                and _pa_match(name_hint, parts[1])
                and "monitor" not in parts[1].lower()
            ):
                return parts[1]
    except Exception:
        pass
    return None


def find_pa_sink(name_hint: str) -> Optional[str]:
    """从可用物理输出列表中查找匹配名称的 sink，未找到时返回 None。"""
    for sink in list_pa_sinks():
        if _pa_match(name_hint, sink["id"]):
            return sink["id"]
    return None


def _pa_sink_label(sink_name: str) -> str:
    """将已知设备名称转成简短显示标签，未知设备保留原名。"""
    normalized = sink_name.lower()
    if "anker_powerconf" in normalized:
        return "Anker PowerConf"
    if "platform-sound" in normalized:
        return "Jetson Audio"
    return sink_name


def list_pa_sinks() -> list[dict[str, str]]:
    """查询可供选择的输出设备，排除本应用创建的 AEC 虚拟输出。"""
    try:
        r = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        sinks = []
        for line in r.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and not parts[1].startswith("assistant_aec_"):
                sinks.append({"id": parts[1], "label": _pa_sink_label(parts[1])})
        return sinks
    except Exception:
        return []


def get_default_pa_sink() -> Optional[str]:
    """读取 PulseAudio 当前默认输出设备；查询失败或结果为空时返回 None。"""
    try:
        r = subprocess.run(
            ["pactl", "get-default-sink"], capture_output=True, text=True, timeout=5
        )
        sink = r.stdout.strip()
        return sink or None
    except Exception:
        return None


def get_pa_sink_volume(sink_name: str) -> Optional[int]:
    """读取输出设备各声道百分比音量，返回平均值，查询失败返回 None。"""
    try:
        r = subprocess.run(
            ["pactl", "get-sink-volume", sink_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            return None
        percentages = []
        for token in r.stdout.replace("/", " ").replace(",", " ").split():
            if token.endswith("%") and token[:-1].isdigit():
                percentages.append(int(token[:-1]))
        if percentages:
            return round(sum(percentages) / len(percentages))
    except Exception:
        pass
    return None


class SpeakerSelector:
    """以锁保护当前扬声器选择，并在设备变化时刷新可用输出。"""

    def __init__(self, preferred_hint: Optional[str], fallback_hint: Optional[str]):
        """保存首选和备用设备提示，初始化选择状态并立即查询可用扬声器。"""
        self._lock = threading.Lock()
        self._preferred_hint = preferred_hint
        self._fallback_hint = fallback_hint
        self._sink: Optional[str] = None
        self._speakers: list[dict[str, str]] = []
        self.refresh()

    @staticmethod
    def _matching_sink(
        speakers: list[dict[str, str]], hint: Optional[str]
    ) -> Optional[str]:
        """在设备 ID 或显示标签中匹配给定提示，返回首个匹配的设备 ID。"""
        if not hint:
            return None
        for speaker in speakers:
            if _pa_match(hint, speaker["id"]) or _pa_match(hint, speaker["label"]):
                return speaker["id"]
        return None

    def refresh(self) -> dict:
        """刷新设备列表并保留仍有效的选择，再按首选/备用/系统默认规则选取输出。

        配置了明确首选时不会悄悄改用不匹配的默认设备。
        """
        speakers = list_pa_sinks()
        available = {speaker["id"] for speaker in speakers}
        default_sink = get_default_pa_sink()
        with self._lock:
            selected = self._sink if self._sink in available else None
            selected = selected or self._matching_sink(speakers, self._preferred_hint)
            if not self._preferred_hint:
                selected = selected or self._matching_sink(
                    speakers, self._fallback_hint
                )
                selected = selected or (
                    default_sink if default_sink in available else None
                )
            self._speakers = speakers
            self._sink = selected
            return self._state_unlocked()

    def _state_unlocked(self) -> dict:
        """在调用方持锁时组装设备列表、选择结果和可用音量快照。"""
        state = {
            "speakers": [dict(speaker) for speaker in self._speakers],
            "selected": self._sink,
        }
        if self._sink:
            volume = get_pa_sink_volume(self._sink)
            if volume is not None:
                state["volume"] = volume
        return state

    def state(self) -> dict:
        """重新查询当前输出设备并返回最新路由状态。"""
        return self.refresh()

    def get_sink(self) -> Optional[str]:
        """在锁内读取当前选中的物理输出 ID。"""
        with self._lock:
            return self._sink

    def select(self, sink_id: str) -> dict:
        """校验用户选择的设备仍存在，更新首选路由并返回状态；设备消失时返回错误。"""
        speakers = list_pa_sinks()
        available = {speaker["id"] for speaker in speakers}
        if sink_id not in available:
            state = self.refresh()
            state["error"] = "Selected speaker is no longer available"
            return state

        with self._lock:
            self._speakers = speakers
            self._sink = sink_id
            self._preferred_hint = sink_id
            return self._state_unlocked()

    def set_volume(self, volume: int) -> dict:
        """将音量限制在 0–100%，设置当前选中输出并返回刷新后的设备状态。"""
        volume = max(0, min(100, int(volume)))
        sink_id = self.get_sink()
        if not sink_id:
            state = self.refresh()
            state["error"] = "No speaker output is selected"
            return state

        try:
            result = subprocess.run(
                ["pactl", "set-sink-volume", sink_id, f"{volume}%"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                state = self.refresh()
                state["error"] = result.stderr.strip() or "Could not set speaker volume"
                return state
        except Exception as exc:
            state = self.refresh()
            state["error"] = str(exc)
            return state

        return self.refresh()


class PulseAudioAEC:
    """拥有单个麦克风/扬声器组合的回声消除模块及虚拟设备名称。"""

    def __init__(
        self,
        source_master: str,
        source_name: Optional[str] = None,
        sink_name: Optional[str] = None,
    ):
        """保存物理输入源并生成实例独有的虚拟输入/输出名称，此时不加载模块。"""
        self.source_master = source_master
        suffix = uuid.uuid4().hex[:8]
        self.source_name = source_name or f"{AEC_SOURCE_NAME}_{suffix}"
        self.sink_name = sink_name or f"{AEC_SINK_NAME}_{suffix}"
        self.sink_master: Optional[str] = None
        self.module_id: Optional[int] = None
        self.last_error: Optional[str] = None

    @property
    def active(self) -> bool:
        """根据是否持有 PulseAudio 模块 ID 判断 AEC 路由是否已创建。"""
        return self.module_id is not None

    def start(self, sink_master: str) -> bool:
        """先释放旧路由，再为指定物理扬声器加载 WebRTC AEC 模块，记录成功 ID 或失败原因。"""
        self.stop()
        self.last_error = None
        args = [
            "pactl",
            "load-module",
            "module-echo-cancel",
            "aec_method=webrtc",
            f"source_master={self.source_master}",
            f"sink_master={sink_master}",
            f"source_name={self.source_name}",
            f"sink_name={self.sink_name}",
            "source_properties=device.description=Assistant_AEC_Microphone",
            "sink_properties=device.description=Assistant_AEC_Speaker",
        ]
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                self.last_error = result.stderr.strip() or "Could not load WebRTC AEC"
                return False
            self.module_id = int(result.stdout.strip())
            self.sink_master = sink_master
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.module_id = None
            self.sink_master = None
            return False

    def stop(self):
        """卸载本实例拥有的 AEC 模块；卸载失败时保留所有权并抛出错误。"""
        if self.module_id is None:
            return
        result = subprocess.run(
            ["pactl", "unload-module", str(self.module_id)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "AEC module did not unload")
        self.module_id = None
        self.sink_master = None
