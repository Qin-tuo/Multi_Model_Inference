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

"""Pipeline — shared audio I/O, VAD, TTS streaming, and mic recording.

Extracts the common infrastructure used by both run_voice_chat.py and
run_vision_chat.py so each entry point only contains its unique logic.
"""

import sys
import time
import wave
import subprocess
import threading
import queue
import json
import hashlib
import re
import uuid
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Optional, Iterator, Union

import numpy as np
from pathlib import Path
from rich.console import Console

from app.audio import wav_pcm
from app.config import VADConfig

# Suppress noisy ALSA error messages (underrun warnings etc.)
# The callback reference must be kept alive to avoid segfault from GC.
_ALSA_ERR_T = None
_alsa_handler = None
try:
    import ctypes
    _ALSA_ERR_T = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                    ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    _alsa_handler = _ALSA_ERR_T(lambda *_: None)
    ctypes.cdll.LoadLibrary('libasound.so.2').snd_lib_error_set_handler(_alsa_handler)
except Exception:
    pass


# ── Audio constants (fixed by hardware, not user-tunable) ────────

SAMPLE_RATE = 16000
SILERO_CHUNK_SAMPLES = 512  # Silero VAD requires exactly 512 samples (32ms) at 16kHz
CHANNELS = 1

TTS_BREAKS = frozenset('.,;:!?\n')
AEC_SOURCE_NAME = "reachy_aec_source"
AEC_SINK_NAME = "reachy_aec_sink"


# ── Audio helpers ─────────────────────────────────────────────────

def chunk_rms(raw: bytes) -> float:
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(pcm ** 2)))


def save_wav(chunks: list[bytes], path: str):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(chunks))


def warmup_stt(stt_obj) -> float:
    """Run a dummy transcription to warm up CUDA. Returns elapsed seconds."""
    path = "/tmp/_warmup.wav"
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(np.zeros(SAMPLE_RATE // 2, dtype=np.int16).tobytes())
    t0 = time.perf_counter()
    stt_obj.transcribe(path, sample_rate=SAMPLE_RATE)
    Path(path).unlink(missing_ok=True)
    return time.perf_counter() - t0


def _pa_match(needle: str, haystack: str) -> bool:
    """Match a name hint against a PulseAudio device name, ignoring space/underscore differences."""
    n = needle.lower().replace(" ", "_")
    h = haystack.lower().replace(" ", "_")
    return n in h


def find_pa_source(name_hint: str) -> Optional[str]:
    """Find a PulseAudio input source matching name_hint."""
    try:
        r = subprocess.run(["pactl", "list", "short", "sources"],
                           capture_output=True, text=True, timeout=5)
        for line in r.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and _pa_match(name_hint, parts[1]) and "monitor" not in parts[1].lower():
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
    if "pollen_robotics_reachy_mini_audio" in normalized:
        return "Reachy Mini"
    if "anker_powerconf" in normalized:
        return "Anker PowerConf"
    if "platform-sound" in normalized:
        return "Jetson Audio"
    return sink_name


def list_pa_sinks() -> list[dict[str, str]]:
    """List available PulseAudio output sinks for the web speaker selector."""
    try:
        r = subprocess.run(["pactl", "list", "short", "sinks"],
                           capture_output=True, text=True, timeout=5)
        sinks = []
        for line in r.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and not parts[1].startswith("reachy_aec_"):
                sinks.append({"id": parts[1], "label": _pa_sink_label(parts[1])})
        return sinks
    except Exception:
        return []


def get_default_pa_sink() -> Optional[str]:
    """Return PulseAudio's current default sink, if available."""
    try:
        r = subprocess.run(["pactl", "get-default-sink"],
                           capture_output=True, text=True, timeout=5)
        sink = r.stdout.strip()
        return sink or None
    except Exception:
        return None


def get_pa_sink_volume(sink_name: str) -> Optional[int]:
    """Return the average PulseAudio sink volume as a percentage."""
    try:
        r = subprocess.run(
            ["pactl", "get-sink-volume", sink_name],
            capture_output=True, text=True, timeout=5,
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
    def _matching_sink(speakers: list[dict[str, str]], hint: Optional[str]) -> Optional[str]:
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
                selected = selected or self._matching_sink(speakers, self._fallback_hint)
                selected = selected or (default_sink if default_sink in available else None)
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
                capture_output=True, text=True, timeout=5,
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
            "pactl", "load-module", "module-echo-cancel",
            "aec_method=webrtc",
            f"source_master={self.source_master}",
            f"sink_master={sink_master}",
            f"source_name={self.source_name}",
            f"sink_name={self.sink_name}",
            "source_properties=device.description=Reachy_AEC_Microphone",
            "sink_properties=device.description=Reachy_AEC_Speaker",
        ]
        try:
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=10,
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
        if self.module_id is None: return
        result = subprocess.run(["pactl", "unload-module", str(self.module_id)],
                                capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "AEC module did not unload")
        self.module_id = None
        self.sink_master = None


class Playback:
    """Own only this turn's player; cancellation and process launch share a lock."""
    def __init__(self, event):
        self.event = event
        self._lock = threading.Lock()
        self._process = None

    def start(self, cmd):
        from app.request_io import TurnCancelled
        with self._lock:
            if self.event.is_set(): raise TurnCancelled('Playback cancelled')
            self._process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            return self._process

    def cancel(self):
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                try: process.terminate()
                except ProcessLookupError: pass

    def clear(self, process):
        with self._lock:
            if self._process is process: self._process = None


def play_audio(audio: np.ndarray, sample_rate: int, sink: Optional[str] = None, playback=None, on_start=None):
    """Return a device-process receipt; physical audibility is a separate check."""
    if not sink: raise ValueError("Explicit speaker device is required")
    if audio.ndim != 1 or audio.dtype != np.int16 or sample_rate != 16000:
        raise ValueError("Playback requires mono PCM16 at 16kHz")
    if sink.startswith(("hw:", "plughw:")):
        cmd = ["aplay", "-D", sink, "-f", "S16_LE", "-r", str(sample_rate), "-c", "1", "-t", "raw", "-q"]
    else:
        cmd = ["paplay", f"--device={sink}", "--format=s16le", f"--rate={sample_rate}", "--channels=1", "--raw"]
    from app.request_io import TurnCancelled
    playback = playback or Playback(threading.Event())
    process = playback.start(cmd)
    started = time.monotonic()
    try:
        if on_start:on_start()
        data = audio.tobytes()
        deadline = started + audio.size/sample_rate + 5
        while True:
            if playback.event.is_set(): raise TurnCancelled('Playback cancelled')
            try:
                _, error = process.communicate(data, timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                data = None
                if time.monotonic() >= deadline: raise RuntimeError('Playback timed out')
        if playback.event.is_set(): raise TurnCancelled('Playback cancelled')
        if process.returncode != 0: raise RuntimeError(f"Playback failed: {error.decode(errors='replace')[:300]}")
        return {"kind": "device_completed", "submitted_monotonic": started,
                "completed_monotonic": time.monotonic(), "frames": audio.size}
    finally:
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=0.5)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        playback.clear(process)


# ── Silero VAD ────────────────────────────────────────────────────

class SileroVAD:
    """Stateful streaming wrapper for faster-whisper's Silero VAD model."""

    def __init__(self):
        import onnxruntime as ort
        entry = json.loads((Path(__file__).resolve().parents[1] / "models/speech.json").read_text())["vad"]
        model_path = Path(entry["directory"]) / entry["filename"]
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != entry["files"][entry["filename"]]["sha256"]:
            raise RuntimeError("Silero VAD model differs from lock")

        session_options = ort.SessionOptions()
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = 1
        session_options.enable_cpu_mem_arena = False
        session_options.log_severity_level = 4
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.reset()

    def __call__(self, raw_audio: bytes) -> float:
        """Return speech probability for raw int16 PCM audio at 16 kHz."""
        pcm = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        if pcm.size != SILERO_CHUNK_SAMPLES:
            raise ValueError(
                f"Silero VAD expects {SILERO_CHUNK_SAMPLES} samples, got {pcm.size}"
            )

        frame = pcm.reshape(1, -1)
        model_input = np.concatenate((self._context, frame), axis=1)
        output, self._h, self._c = self._session.run(
            ["speech_probs", "hn", "cn"],
            {"input": model_input, "h": self._h, "c": self._c},
        )
        self._context = model_input[:, -64:]
        return float(output.reshape(-1)[0])

    def reset(self):
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)


def load_silero(console: Optional[Console] = None) -> SileroVAD:
    """Load the pinned Silero asset extracted from the upstream distribution."""
    t0 = time.perf_counter()
    vad = SileroVAD()
    dt = time.perf_counter() - t0
    if console:
        console.print(f"  ✓ Silero VAD (ONNX, loaded in {dt:.1f}s)")
    return vad


# ── Speech segment ────────────────────────────────────────────────

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
    end_reason: str = 'silence'


@dataclass(frozen=True)
class SpeechOnset:
    utterance_id: str
    started_monotonic: float
    confirmed_monotonic: float
    input_epoch: int = 0
    input_invalidated: threading.Event | None = None


# ── Mic recorder ──────────────────────────────────────────────────

@dataclass(frozen=True)
class AudioChunk:
    data: bytes
    input_epoch: int
    received_monotonic: float
    sequence: int = 0
    input_invalidated: threading.Event | None = None


class MicRecorder:
    """Manages mic recording via parecord/arecord with a background reader thread."""

    def __init__(self, console: Console, chunk_ms: int = 32):
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
        """Start recording. Returns True on success."""
        self.stop()
        self.alive = True
        self.last_error = None
        self.suppressed_reason=None
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
                        f"(Reachy mic → {_pa_sink_label(self.pa_sink)})"
                    )
                    self.pa_sink = self.aec.sink_name
                else:
                    self.console.print(
                        f"  [yellow]AEC unavailable: {self.aec.last_error}; "
                        "using direct audio[/yellow]"
                    )
            self.console.print(f"  PA source: {self.pa_source.split('.')[-2]}")
        else:
            self.console.print("  [yellow]PA source not found, using ALSA direct[/yellow]")
            # A busy ALSA device fails explicitly; never stop the desktop audio service.

        if not self._start_capture():
            if self.aec and self.aec.active:
                self.console.print("  [yellow]AEC capture failed; using direct audio[/yellow]")
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
        if self.pa_source:
            source = self.aec.source_name if self.aec and self.aec.active else self.pa_source
            return [
                "parecord", "-d", source, "--format=s16le",
                f"--rate={SAMPLE_RATE}", f"--channels={CHANNELS}", "--raw",
            ]
        plughw = self._hw.replace("hw:", "plughw:")
        return [
            "arecord", "-D", plughw, "-f", "S16_LE", "-r", str(SAMPLE_RATE),
            "-c", str(CHANNELS), "-t", "raw",
        ]

    def _start_capture(self) -> bool:
        rec_cmd = self._record_command()
        for attempt in range(3):
            proc = subprocess.Popen(rec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(0.5)
            if proc.poll() is None:
                self._proc = proc
                self._reader_thread = threading.Thread(target=self._reader, args=(proc,), name="microphone-reader", daemon=True)
                self._reader_thread.start()
                return True
            err = proc.stderr.read().decode(errors="replace").strip()
            if proc.stdout:proc.stdout.close()
            if proc.stderr:proc.stderr.close()
            self.last_error=err or f'Microphone process exited with status {proc.returncode}'
            self.console.print(f"  [red]Mic attempt {attempt+1} failed: {err}[/red]")
            time.sleep(1)
        self._proc = None
        return False

    def _check_capture(self):
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
            self.console.print(f"  [yellow]Mic: no startup audio; capture running: {running}[/yellow]")
        return bool(test_chunks)

    def _stop_capture(self):
        proc = self._proc
        self._proc = None
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=2)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            if self._reader_thread:
                self._reader_thread.join(timeout=2)
                if self._reader_thread.is_alive(): raise RuntimeError("Microphone reader has not released")
                self._reader_thread = None
            if proc.stdout: proc.stdout.close()
            if proc.stderr: proc.stderr.close()

    def _rebuild_aec_route(self) -> bool:
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
        with self._route_lock:
            return self.pa_sink

    def _add_aec_state(self, state: dict) -> dict:
        state["aec_enabled"] = bool(self.aec and self.aec.active)
        if self.aec and self.aec.last_error and not self.aec.active:
            state["aec_error"] = self.aec.last_error
        return state

    def speaker_state(self) -> dict:
        if not self.speaker_selector:
            return {"speakers": [], "selected": None, "aec_enabled": False}
        previous_sink = self.speaker_selector.get_sink()
        state = self.speaker_selector.state()
        selected_sink = self.speaker_selector.get_sink()
        if (
            self.alive and selected_sink
            and previous_sink != selected_sink
            and self.aec
            and self.aec.sink_master != selected_sink
        ):
            self._rebuild_aec_route()
        if not self.alive:state['error']=self.last_error or 'Microphone input stopped; restart the application'
        return self._add_aec_state(state)

    def select_speaker(self, sink_id: str) -> dict:
        if not self.alive:raise RuntimeError('Microphone input stopped; restart the application after checking the device')
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
        if not self.speaker_selector:
            return {
                "speakers": [],
                "selected": None,
                "aec_enabled": False,
                "error": "Speaker routing is not initialized",
            }
        return self._add_aec_state(self.speaker_selector.set_volume(volume))

    def _reader(self, proc: subprocess.Popen):
        while self.alive and proc.poll() is None:
            epoch = self.input_epoch
            raw = proc.stdout.read(self.chunk_bytes)
            if raw:self.last_chunk_monotonic=time.monotonic()
            if not raw:
                if proc.poll() is not None and proc is self._proc:
                    err = proc.stderr.read().decode(errors="replace").strip()
                    if err:
                        self.console.print(f"\n  [red]audio capture died: {err}[/red]")
                break
            with self._input_lock:
                if proc is self._proc and self.listening.is_set() and epoch == self.input_epoch:
                    if len(raw) != self.chunk_bytes:
                        self.last_error = "Microphone returned partial PCM frame"
                        self.alive = False
                        self._invalidate_input_locked()
                        break
                    self._sequence+=1
                    try: self.audio_q.put_nowait(AudioChunk(raw, epoch, time.monotonic(),self._sequence,self.input_invalidated))
                    except queue.Full:
                        self.last_error = "Microphone input overflow; utterance invalidated"
                        self.alive = False
                        self._invalidate_input_locked()
                        break
        if proc is self._proc and self.alive:
            self.last_error = self.last_error or "Microphone capture ended"
            self.alive = False
            with self._input_lock:self._invalidate_input_locked()

    def _invalidate_input_locked(self):
        self.input_invalidated.set()
        self.input_invalidated=threading.Event()
        self.input_epoch+=1
        self._sequence=0
        self.flush()

    def suppress(self,reason):
        """Processing gate leaves the user's listening/mute preference intact."""
        with self._input_lock:
            if reason!=self.suppressed_reason:
                self.suppressed_reason=reason
                self._invalidate_input_locked()

    def fail_input(self,reason):
        with self._input_lock:
            self.last_error=reason;self.alive=False
            self._invalidate_input_locked()

    def input_state(self):
        return {'alive':self.alive,'listening':self.listening.is_set(),
                'input_epoch':self.input_epoch,'suppressed_reason':self.suppressed_reason,
                'error':self.last_error,'queued_chunks':self.audio_q.qsize(),'queue_capacity':self.audio_q.maxsize}

    def flush(self):
        while not self.audio_q.empty():
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break

    def pause(self):
        """Discard this capture epoch; old partial input cannot resume later."""
        with self._input_lock:
            self.listening.clear()
            self._invalidate_input_locked()

    def resume(self):
        with self._input_lock:
            if not self.alive:
                raise ValueError(
                    "Microphone capture stopped; restart the application after checking the device"
                )
            self._invalidate_input_locked()
            self.listening.set()

    def stop(self):
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


# ── VAD loop ──────────────────────────────────────────────────────


class SpeechSegmenter:
    """Upstream Silero/RMS/lookback/end-silence rules, independent of the source."""

    def __init__(self, config, silero):
        self.config, self.silero = config, silero
        self.lookback = deque(maxlen=int(config.lookback_ms / config.chunk_ms))
        self.speech_raw = []
        self.speaking = False
        self.silence = 0
        self.started = 0.0
        self.utterance_id = None
        self.first_speech = 0.0
        self.voiced_frames = 0
        self.confirmed = False
        self.onset = None
        self.discarded = None
        self.draining = False
        silero.reset()

    def feed(self, raw, received_time):
        self.onset = self.discarded = None
        cfg = self.config
        if len(raw) != SILERO_CHUNK_SAMPLES * 2:
            raise ValueError("VAD input must be one complete 512-sample PCM16 frame")
        rms = chunk_rms(raw)
        is_speech = (
            rms >= min(0.002, cfg.min_utterance_rms)
            and self.silero(raw) > cfg.silero_threshold
        )
        if self.draining:
            self.silence = 0 if is_speech else self.silence + 1
            if self.silence >= max(1, int(cfg.silence_duration_ms / cfg.chunk_ms)):
                self.draining = False
                self.silence = 0
                self.silero.reset()
            return None
        if is_speech:
            self.silence = 0
            if not self.speaking:
                self.speaking = True
                self.utterance_id = uuid.uuid4().hex
                self.first_speech = received_time - cfg.chunk_ms / 1000
                self.voiced_frames = 0
                self.confirmed = False
                self.started = (
                    received_time - (len(self.lookback) + 1) * cfg.chunk_ms / 1000
                )
                self.speech_raw = list(self.lookback)
            self.speech_raw.append(raw)
            self.voiced_frames += 1
            if (
                not self.confirmed
                and self.voiced_frames * cfg.chunk_ms >= cfg.confirm_speech_ms
            ):
                self.confirmed = True
                self.onset = SpeechOnset(
                    self.utterance_id, self.first_speech, received_time
                )
        elif self.speaking:
            self.speech_raw.append(raw)
            self.silence += 1
        else:
            self.lookback.append(raw)
            return None
        limited = len(self.speech_raw) >= int(cfg.max_speech_secs * 1000 / cfg.chunk_ms)
        ended = self.silence >= max(1, int(cfg.silence_duration_ms / cfg.chunk_ms))
        if not limited and not ended:
            return None
        identity = self.utterance_id
        first_speech = self.first_speech
        confirmed = self.confirmed
        captured = self.speech_raw
        self.speech_raw = []
        self.lookback.clear()
        self.speaking = False
        self.silence = 0
        self.utterance_id = None
        self.voiced_frames = 0
        self.confirmed = False
        self.silero.reset()
        pcm = b"".join(captured)
        duration, rms = len(pcm) / 32000, chunk_rms(pcm)
        if limited and not ended:
            self.draining = True
            self.discarded = (identity, "utterance_too_long")
        if duration < cfg.min_utterance_secs or rms < cfg.min_utterance_rms:
            self.discarded = (identity, "utterance_rejected")
            return None
        if not confirmed and not limited:
            self.onset = SpeechOnset(identity, first_speech, received_time)
        return SpeechSegment(
            np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768,
            captured,
            duration,
            rms,
            self.started,
            received_time,
            utterance_id=identity,
            end_reason="max_duration" if limited and not ended else "silence",
        )


def segment_pcm(pcm: bytes, vad_cfg, silero):
    """File replay using the same segmenter; timestamps are file offsets, not device time."""
    segmenter = SpeechSegmenter(vad_cfg, silero)
    frame_bytes = SILERO_CHUNK_SAMPLES * 2
    # Pad only for the finite file boundary; this is never a measured acoustic endpoint.
    padded = pcm + b"\0" * ((-len(pcm)) % frame_bytes)
    padded += b"\0" * (
        frame_bytes * (int(vad_cfg.silence_duration_ms / vad_cfg.chunk_ms) + 1)
    )
    for offset in range(0, len(padded), frame_bytes):
        result = segmenter.feed(
            padded[offset : offset + frame_bytes], (offset + frame_bytes) / 32000
        )
        if result is not None:
            if result.end_reason == "max_duration":
                raise ValueError(
                    "Speech utterance exceeds VAD duration budget; use shorter utterances"
                )
            yield result


def vad_loop(
    mic,
    console,
    vad_cfg=None,
    silero=None,
    *,
    on_start=None,
    on_discard=None,
    capture_gate=None,
    max_age_seconds=1.0,
):
    """Continuous bounded input; slow inference runs in the conversation worker."""
    if silero is None:
        raise RuntimeError("Prepared Silero VAD is required")
    segmenter = SpeechSegmenter(vad_cfg or VADConfig(), silero)
    epoch = mic.input_epoch
    notified = None
    previous_sequence = 0
    last_received = time.monotonic()
    needs_quiet = False
    quiet_frames = 0

    def discard(reason):
        nonlocal notified
        if notified and on_discard:
            on_discard(notified, reason)
        notified = None

    try:
        while mic.alive:
            gate_reason = capture_gate() if capture_gate else None
            if gate_reason:
                needs_quiet = True
            if capture_gate:
                mic.suppress(
                    gate_reason or ("awaiting_silence" if needs_quiet else None)
                )
            if epoch != mic.input_epoch:
                discard("input_epoch_changed")
                epoch = mic.input_epoch
                previous_sequence = 0
                segmenter = SpeechSegmenter(vad_cfg or VADConfig(), silero)
            try:
                chunk = mic.audio_q.get(timeout=0.1)
            except queue.Empty:
                if (
                    time.monotonic() - max(last_received, mic.last_chunk_monotonic or 0)
                    > max_age_seconds
                ):
                    mic.fail_input(
                        "Microphone capture stalled; current input discarded"
                    )
                continue
            if chunk.input_epoch != mic.input_epoch or not mic.listening.is_set():
                continue
            if time.monotonic() - chunk.received_monotonic > max_age_seconds or (
                previous_sequence and chunk.sequence != previous_sequence + 1
            ):
                mic.fail_input(
                    "Microphone frames stale or discontinuous; current input discarded"
                )
                break
            previous_sequence = chunk.sequence
            last_received = chunk.received_monotonic
            if needs_quiet:
                cfg = vad_cfg or VADConfig()
                speech = (
                    chunk_rms(chunk.data) >= min(0.002, cfg.min_utterance_rms)
                    and silero(chunk.data) > cfg.silero_threshold
                )
                quiet_frames = 0 if speech else quiet_frames + 1
                if not gate_reason and quiet_frames >= max(
                    1, int(cfg.silence_duration_ms / cfg.chunk_ms)
                ):
                    needs_quiet = False
                    quiet_frames = 0
                    mic.suppress(None)
                continue
            result = segmenter.feed(chunk.data, chunk.received_monotonic)
            if segmenter.onset and on_start:
                onset = segmenter.onset
                decision = on_start(
                    SpeechOnset(
                        onset.utterance_id,
                        onset.started_monotonic,
                        onset.confirmed_monotonic,
                        epoch,
                        chunk.input_invalidated,
                    )
                )
                if decision.get("accepted"):
                    notified = onset.utterance_id
            if segmenter.discarded:
                discard(segmenter.discarded[1])
            if (
                result is not None
                and result.end_reason != "max_duration"
                and epoch == mic.input_epoch
                and mic.listening.is_set()
            ):
                result.input_epoch = epoch
                result.input_invalidated = chunk.input_invalidated
                if on_start is None or notified == result.utterance_id:
                    yield result
                notified = None
    finally:
        discard("input_stopped")
    if mic.last_error:
        raise RuntimeError(mic.last_error)


class TextChunker:
    """Chinese punctuation and bounded characters; no English word-count gate."""
    def __init__(self, first_chars=8, max_chars=40):
        self.first_chars, self.max_chars = first_chars, max_chars
        self.buffer = ""
        self.started = False

    def feed(self, text):
        chunks = []
        for char in text:
            self.buffer += char
            limit = self.max_chars if self.started else self.first_chars
            if char in "。！？!?；;\n" or len(self.buffer) >= limit:
                chunk = self.buffer.strip(); self.buffer = ""
                if chunk and any(c.isalnum() for c in chunk):
                    chunks.append(chunk); self.started = True
        return chunks

    def finish(self):
        text, self.buffer = self.buffer.strip(), ""
        return [text] if text and any(c.isalnum() for c in text) else []


class SpeechOutput:
    """One bounded synthesis worker; file and device receipts are distinct."""
    def __init__(self, tts, audio_config, turn_id, output_sink=None, *, cancel_event=None, defer_commit=False, on_output_start=None, trace=None):
        self.tts, self.config, self.turn_id = tts, audio_config, turn_id
        self.output_sink = output_sink or (lambda: audio_config.output_device)
        self.queue = queue.Queue(maxsize=2)
        self.error = None
        self.bytes_written = 0
        self.chunks = []
        self.receipt = None
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.defer_commit = defer_commit
        self.on_output_start = on_output_start
        self.trace = trace
        self._output_started = False
        self._temporary = None
        self._path = None
        self._abort = threading.Event()
        self._playback = Playback(self._abort)
        self._thread = threading.Thread(target=self._run, name="speech-output", daemon=True)
        self._thread.start()

    def submit(self, text):
        while self._thread.is_alive():
            self._check()
            if self.error: raise RuntimeError(self.error)
            try: self.queue.put(text, timeout=0.1); return
            except queue.Full: continue
        raise RuntimeError(self.error or "Speech worker stopped")

    def _check(self):
        from app.request_io import TurnCancelled
        if self._abort.is_set() or self.cancel_event.is_set(): raise TurnCancelled('Speech output cancelled')

    def cancel(self):
        self._abort.set()
        self.tts.cancel(self._abort)
        self._playback.cancel()

    def idle(self):
        return not self._thread.is_alive()

    def publish(self):
        """Caller serializes this commit with turn cancellation."""
        self._check()
        if self._thread.is_alive(): raise RuntimeError('Speech output is still running')
        if self.error: raise RuntimeError(self.error)
        if self._temporary:
            path = self._path
            retained = sorted((p for p in path.parent.glob('reply-*.wav') if p != path and re.fullmatch(r'reply-[0-9a-f]{32}\.wav', p.name)), key=lambda p:(p.stat().st_mtime_ns,p.name), reverse=True)
            for old in retained[self.config.retained_files - 1:]: old.unlink()
            self._begin_output()
            self._temporary.replace(path)
            self._temporary = None
            self.receipt = {'kind':'file_written', 'path':str(path), 'bytes':self.bytes_written}
        return self.receipt

    def _begin_output(self):
        self._check()
        if not self._output_started:
            if self.on_output_start:self.on_output_start()
            self._output_started=True

    def _run(self):
        writer = None; temporary = None; path = None
        try:
            if self.config.output_mode == "file":
                folder = Path(self.config.output_directory).expanduser()
                folder.mkdir(parents=True, exist_ok=True, mode=0o700)
                path = folder / f"reply-{self.turn_id}.wav"
                temporary = path.with_suffix(".partial")
                writer = wave.open(str(temporary), "wb")
                writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(SAMPLE_RATE)
            while True:
                self._check()
                try: text = self.queue.get(timeout=0.1)
                except queue.Empty: continue
                if text is None: break
                self._check()
                with self.trace.span('tts') if self.trace else nullcontext():
                    result = self.tts.synthesize(text, cancel_event=self._abort)
                self._check()
                audio = result.get("audio")
                if audio is None: raise RuntimeError(result.get("error", "TTS returned no audio"))
                if audio.dtype != np.int16 or audio.ndim != 1 or result["sample_rate"] != SAMPLE_RATE:
                    raise ValueError("Unexpected synthesized PCM format")
                raw = audio.tobytes()
                if self.bytes_written + len(raw) > self.config.max_output_seconds * 32000:
                    raise ValueError("Turn audio exceeds duration budget")
                if writer:
                    writer.writeframesraw(raw)
                    receipt = {"kind": "file_chunk_written", "bytes": len(raw)}
                else:
                    receipt = play_audio(audio, SAMPLE_RATE, self.output_sink(), self._playback, on_start=self._begin_output)
                self.bytes_written += len(raw)
                self.chunks.append({"text": text, "bytes": len(raw), "elapsed_seconds": result.get("elapsed_seconds"),
                                    "first_pcm_seconds": result.get("first_pcm_seconds"), "receipt": receipt})
            self._check()
            if not self.bytes_written: raise RuntimeError("No speech output produced")
            if writer:
                writer.close(); writer = None
                self._temporary, self._path = temporary, path
                temporary = None  # Publication belongs to the turn's commit.
            else:
                self.receipt = {"kind": "device_completed", "bytes": self.bytes_written}
        except Exception as exc:
            self.error = str(exc)
        finally:
            if writer: writer.close()
            if temporary and temporary.exists(): temporary.unlink()

    def close(self, success=True):
        if not success: self.cancel()
        elif self._thread.is_alive(): self.submit(None)
        self._thread.join(timeout=self.tts.timeout + 5)
        if self._thread.is_alive(): raise RuntimeError("Speech worker has not released")
        if not success and self._temporary:
            self._temporary.unlink(missing_ok=True)
            self._temporary = None
        if self.error and success: raise RuntimeError(self.error)
        if success and not self.defer_commit: return self.publish()
        return self.receipt
