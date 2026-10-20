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

"""PulseAudio device selection and echo-cancellation routing.

Extracted from the upstream-derived audio pipeline; behavior is preserved.
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
    _alsa_handler = _ALSA_ERR_T(lambda *_: None)
    ctypes.cdll.LoadLibrary("libasound.so.2").snd_lib_error_set_handler(_alsa_handler)
except Exception:
    pass


def _pa_match(needle: str, haystack: str) -> bool:
    """Match a name hint against a PulseAudio device name, ignoring space/underscore differences."""
    n = needle.lower().replace(" ", "_")
    h = haystack.lower().replace(" ", "_")
    return n in h


def find_pa_source(name_hint: str) -> Optional[str]:
    """Find a PulseAudio input source matching name_hint."""
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
    """Find a PulseAudio output sink matching name_hint."""
    for sink in list_pa_sinks():
        if _pa_match(name_hint, sink["id"]):
            return sink["id"]
    return None


def _pa_sink_label(sink_name: str) -> str:
    """Return a concise, user-facing label for a PulseAudio sink."""
    normalized = sink_name.lower()
    if "anker_powerconf" in normalized:
        return "Anker PowerConf"
    if "platform-sound" in normalized:
        return "Jetson Audio"
    return sink_name


def list_pa_sinks() -> list[dict[str, str]]:
    """List available PulseAudio output sinks for the web speaker selector."""
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
    """Return PulseAudio's current default sink, if available."""
    try:
        r = subprocess.run(
            ["pactl", "get-default-sink"], capture_output=True, text=True, timeout=5
        )
        sink = r.stdout.strip()
        return sink or None
    except Exception:
        return None


def get_pa_sink_volume(sink_name: str) -> Optional[int]:
    """Return the average PulseAudio sink volume as a percentage."""
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
    """Thread-safe, live-selectable PulseAudio output routing."""

    def __init__(self, preferred_hint: Optional[str], fallback_hint: Optional[str]):
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
        if not hint:
            return None
        for speaker in speakers:
            if _pa_match(hint, speaker["id"]) or _pa_match(hint, speaker["label"]):
                return speaker["id"]
        return None

    def refresh(self) -> dict:
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
        return self.refresh()

    def get_sink(self) -> Optional[str]:
        with self._lock:
            return self._sink

    def select(self, sink_id: str) -> dict:
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
        """Set the selected physical sink volume and return refreshed state."""
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
    """Managed WebRTC echo-cancellation route for one mic and selected speaker."""

    def __init__(
        self,
        source_master: str,
        source_name: Optional[str] = None,
        sink_name: Optional[str] = None,
    ):
        self.source_master = source_master
        suffix = uuid.uuid4().hex[:8]
        self.source_name = source_name or f"{AEC_SOURCE_NAME}_{suffix}"
        self.sink_name = sink_name or f"{AEC_SINK_NAME}_{suffix}"
        self.sink_master: Optional[str] = None
        self.module_id: Optional[int] = None
        self.last_error: Optional[str] = None

    @property
    def active(self) -> bool:
        return self.module_id is not None

    def start(self, sink_master: str) -> bool:
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
