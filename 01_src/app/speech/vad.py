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

"""Silero state, utterance segmentation and freshness checks.

Extracted from the upstream-derived audio pipeline; behavior is preserved.
"""

import time
import queue
import hashlib
import uuid
from collections import deque
import numpy as np
from pathlib import Path
from project_paths import MODEL_DIR, load_artifact_manifest
from app.config import VADConfig
from app.audio import chunk_rms
from app.speech.constants import SILERO_CHUNK_SAMPLES
from app.speech.types import SpeechOnset, SpeechSegment


class SileroVAD:
    """Stateful streaming wrapper for faster-whisper's Silero VAD model."""

    def __init__(self):
        import onnxruntime as ort

        entry = load_artifact_manifest(MODEL_DIR / "speech.json")["vad"]
        model_path = Path(entry["directory"]) / entry["filename"]
        if (
            hashlib.sha256(model_path.read_bytes()).hexdigest()
            != entry["files"][entry["filename"]]["sha256"]
        ):
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
