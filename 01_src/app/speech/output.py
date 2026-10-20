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

"""Bounded per-turn TTS output and atomic publication.

Extracted from the upstream-derived audio pipeline; behavior is preserved.
"""

import wave
import threading
import queue
import re
from contextlib import nullcontext
import numpy as np
from pathlib import Path
from app.speech.constants import SAMPLE_RATE
from app.speech.playback import Playback, play_audio


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
                chunk = self.buffer.strip()
                self.buffer = ""
                if chunk and any(c.isalnum() for c in chunk):
                    chunks.append(chunk)
                    self.started = True
        return chunks

    def finish(self):
        text, self.buffer = self.buffer.strip(), ""
        return [text] if text and any(c.isalnum() for c in text) else []


class SpeechOutput:
    """One turn's bounded synthesis worker, with explicit output ownership.

    中文导读：队列最多容纳两段文字，满时向生成端施加背压。文件先写
    .partial，由 TurnController 在取消检查与提交锁内发布；已提交文件
    与尚未交付的 PCM 分开处理，避免取消后留下可访问的半份回复。
    """

    def __init__(
        self,
        tts,
        audio_config,
        turn_id,
        output_sink=None,
        *,
        cancel_event=None,
        defer_commit=False,
        on_output_start=None,
        trace=None,
    ):
        self.tts, self.config, self.turn_id = tts, audio_config, turn_id
        self.output_sink = output_sink or (lambda: audio_config.output_device)
        self.queue = queue.Queue(maxsize=2)
        self.error = None
        self.bytes_written = 0
        self.chunks = []
        self.receipt = None
        self.cancel_event = (
            cancel_event if cancel_event is not None else threading.Event()
        )
        self.defer_commit = defer_commit
        self.on_output_start = on_output_start
        self.trace = trace
        self._output_started = False
        self._temporary = None
        self._path = None
        self._abort = threading.Event()
        self._playback = Playback(self._abort)
        self._thread = threading.Thread(
            target=self._run, name="speech-output", daemon=True
        )
        self._thread.start()

    def submit(self, text):
        while self._thread.is_alive():
            self._check()
            if self.error:
                raise RuntimeError(self.error)
            try:
                self.queue.put(text, timeout=0.1)
                return
            except queue.Full:
                continue
        raise RuntimeError(self.error or "Speech worker stopped")

    def _check(self):
        from app.request_io import TurnCancelled

        if self._abort.is_set() or self.cancel_event.is_set():
            raise TurnCancelled("Speech output cancelled")

    def cancel(self):
        self._abort.set()
        self.tts.cancel(self._abort)
        self._playback.cancel()

    def idle(self):
        return not self._thread.is_alive()

    def publish(self):
        """Caller serializes this commit with turn cancellation."""
        self._check()
        if self._thread.is_alive():
            raise RuntimeError("Speech output is still running")
        if self.error:
            raise RuntimeError(self.error)
        if self._temporary:
            path = self._path
            retained = sorted(
                (
                    p
                    for p in path.parent.glob("reply-*.wav")
                    if p != path and re.fullmatch(r"reply-[0-9a-f]{32}\.wav", p.name)
                ),
                key=lambda p: (p.stat().st_mtime_ns, p.name),
                reverse=True,
            )
            for old in retained[self.config.retained_files - 1 :]:
                old.unlink()
            self._begin_output()
            self._temporary.replace(path)
            self._temporary = None
            self.receipt = {
                "kind": "file_written",
                "path": str(path),
                "bytes": self.bytes_written,
            }
        return self.receipt

    def _begin_output(self):
        self._check()
        if not self._output_started:
            if self.on_output_start:
                self.on_output_start()
            self._output_started = True

    def _run(self):
        writer = None
        temporary = None
        path = None
        try:
            if self.config.output_mode == "file":
                folder = Path(self.config.output_directory).expanduser()
                folder.mkdir(parents=True, exist_ok=True, mode=0o700)
                path = folder / f"reply-{self.turn_id}.wav"
                temporary = path.with_suffix(".partial")
                writer = wave.open(str(temporary), "wb")
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(SAMPLE_RATE)
            while True:
                self._check()
                try:
                    text = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if text is None:
                    break
                self._check()
                with self.trace.span("tts") if self.trace else nullcontext():
                    result = self.tts.synthesize(text, cancel_event=self._abort)
                self._check()
                audio = result.get("audio")
                if audio is None:
                    raise RuntimeError(result.get("error", "TTS returned no audio"))
                if (
                    audio.dtype != np.int16
                    or audio.ndim != 1
                    or result["sample_rate"] != SAMPLE_RATE
                ):
                    raise ValueError("Unexpected synthesized PCM format")
                raw = audio.tobytes()
                if (
                    self.bytes_written + len(raw)
                    > self.config.max_output_seconds * 32000
                ):
                    raise ValueError("Turn audio exceeds duration budget")
                if writer:
                    writer.writeframesraw(raw)
                    receipt = {"kind": "file_chunk_written", "bytes": len(raw)}
                else:
                    receipt = play_audio(
                        audio,
                        SAMPLE_RATE,
                        self.output_sink(),
                        self._playback,
                        on_start=self._begin_output,
                    )
                self.bytes_written += len(raw)
                self.chunks.append(
                    {
                        "text": text,
                        "bytes": len(raw),
                        "elapsed_seconds": result.get("elapsed_seconds"),
                        "first_pcm_seconds": result.get("first_pcm_seconds"),
                        "receipt": receipt,
                    }
                )
            self._check()
            if not self.bytes_written:
                raise RuntimeError("No speech output produced")
            if writer:
                writer.close()
                writer = None
                self._temporary, self._path = temporary, path
                temporary = None  # Publication belongs to the turn's commit.
            else:
                self.receipt = {"kind": "device_completed", "bytes": self.bytes_written}
        except Exception as exc:
            self.error = str(exc)
        finally:
            if writer:
                writer.close()
            if temporary and temporary.exists():
                temporary.unlink()

    def close(self, success=True):
        if not success:
            self.cancel()
        elif self._thread.is_alive():
            self.submit(None)
        self._thread.join(timeout=self.tts.timeout + 5)
        if self._thread.is_alive():
            raise RuntimeError("Speech worker has not released")
        if not success and self._temporary:
            self._temporary.unlink(missing_ok=True)
            self._temporary = None
        if self.error and success:
            raise RuntimeError(self.error)
        if success and not self.defer_commit:
            return self.publish()
        return self.receipt
