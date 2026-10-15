# SPDX-License-Identifier: Apache-2.0
"""One bounded request worker around the upstream VLM streaming sequence.

Owns text, file/live audio, VLM and bounded TTS output. History, tools and
full cancellation are integrated in R3 into this same owner.
"""
import queue
import threading
import time
import uuid
import re
from pathlib import Path
from app.audio import wav_pcm
from app.pipeline import segment_pcm, SpeechOutput, TextChunker


class TurnController:
    def __init__(self, config, llm, broadcaster, camera=None, stt=None, tts=None, vad=None):
        self.config, self.llm, self.broadcaster, self.camera = config, llm, broadcaster, camera
        self.stt, self.tts, self.vad = stt, tts, vad
        self.output_sink = None
        self._queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._active = None
        self._last = None

    def start(self):
        if self._thread and self._thread.is_alive(): return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="conversation", daemon=True)
        self._thread.start()

    def submit(self, text, use_image=False):
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("Text must contain 1–4000 characters")
        return self._admit({"input_kind": "text", "text": text.strip()}, use_image)

    def submit_audio(self, data, use_image=False):
        if self.stt is None or self.vad is None: raise ValueError("Speech input is not enabled")
        pcm = wav_pcm(data, self.config.stt.max_seconds)
        return self._admit({"input_kind": "audio_file", "pcm": pcm}, use_image)

    def submit_segment(self, segment):
        if self.stt is None: raise ValueError("Speech input is not enabled")
        return self._admit({"input_kind": "microphone", "segment": segment}, self.camera is not None)

    def _admit(self, data, use_image):
        if type(use_image) is not bool: raise ValueError("use_image must be boolean")
        with self._lock:
            if self._stop.is_set() or not self._thread or not self._thread.is_alive():
                raise RuntimeError("Conversation worker is not ready")
            if self._active is not None: raise RuntimeError("The current request is still running")
            if use_image and (self.camera is None or not self.camera.health_check()):
                raise ValueError("No fresh camera image is available")
            request = {"turn_id": uuid.uuid4().hex, "use_image": use_image, **data}
            self._active = request
            self._queue.put_nowait(request)
            return {"accepted": True, "turn_id": request["turn_id"]}

    def audio_file(self, turn_id):
        if not re.fullmatch(r"[0-9a-f]{32}", turn_id): return None
        folder = Path(self.config.audio.output_directory).expanduser().resolve()
        path = folder / f"reply-{turn_id}.wav"
        return path if path.is_file() and not path.is_symlink() else None

    def state(self):
        with self._lock:
            return {"ready": bool(self._thread and self._thread.is_alive() and not self._stop.is_set()),
                    "active_turn": self._active["turn_id"] if self._active else None,
                    "last_result": self._last, "pending": self._queue.qsize()}

    def _run(self):
        while not self._stop.is_set():
            try: request = self._queue.get(timeout=0.1)
            except queue.Empty: continue
            result = {"turn_id": request["turn_id"], "text": "", "metadata": {}, "mode": "camera_text" if request["use_image"] else "text"}
            def send(kind, **values):
                self.broadcaster.send({"type": kind, "turn_id": request["turn_id"], **values})
            start = time.monotonic()
            first = None
            output = None
            try:
                if self._stop.is_set(): raise InterruptedError("Application stopping")
                prompt = request.get("text")
                if request["input_kind"] != "text":
                    send("status", stage="transcribing")
                    segments = ([request["segment"]] if request["input_kind"] == "microphone" else
                                list(segment_pcm(request["pcm"], self.config.vad, self.vad)))
                    if not segments: raise ValueError("No speech detected in audio input")
                    result["asr"] = []
                    for segment in segments:
                        if self._stop.is_set(): raise InterruptedError("Application stopping")
                        transcript = self.stt.transcribe(segment.audio)
                        if transcript.get("error"): raise RuntimeError(transcript["error"])
                        result["asr"].append(transcript)
                    prompt = "".join(item["text"] for item in result["asr"])
                    if not prompt.strip() or len(prompt) > 4000: raise ValueError("Invalid combined ASR transcript")
                    result["input_kind"] = request["input_kind"]
                    result["vad_segments"] = [{"duration": x.duration, "start": x.start_time, "end": x.end_time} for x in segments]
                    result["vad_time_kind"] = "file_offset_with_end_padding" if request["input_kind"] == "audio_file" else "host_receive_monotonic"
                if self._stop.is_set(): raise InterruptedError("Application stopping")
                result["prompt"] = prompt
                images = None
                if request["use_image"]:
                    frame = self.camera.snapshot()
                    if frame is None: raise ValueError("Camera image expired before processing")
                    result["frame"] = frame.metadata()
                    images = [self.camera._base64_jpeg(frame.jpeg)]
                send("transcript", text=prompt)
                send("status", stage="thinking")
                if self.tts is not None:
                    output = SpeechOutput(self.tts, self.config.audio, request["turn_id"], self.output_sink)
                chunker = TextChunker(self.config.tts.first_chunk_chars, self.config.tts.max_chunk_chars)
                stream = self.llm.generate_stream(prompt, images_b64=images,
                    system_prompt=self.config.vision.system_prompt if images else None)
                try:
                    for text, metadata in stream:
                        if self._stop.is_set(): raise InterruptedError("Application stopping")
                        result["metadata"].update(metadata)
                        if text:
                            first = first or time.monotonic()
                            result["text"] += text
                            send("token", text=text)
                            if output:
                                for chunk in chunker.feed(text): output.submit(chunk)
                finally:
                    stream.close()
                if not result["metadata"].get("done") or not result["text"].strip():
                    raise RuntimeError("Model did not produce a complete text answer")
                if output:
                    send("status", stage="synthesizing" if self.config.audio.output_mode == "file" else "speaking")
                    for chunk in chunker.finish(): output.submit(chunk)
                    receipt = output.close(success=True)
                    result["speech"] = {"receipt": receipt, "chunks": output.chunks, "duration": output.bytes_written/32000}
                    if receipt["kind"] == "file_written": result["audio_url"] = f"/api/audio/{request['turn_id']}"
                    output = None
                result["status"] = "complete"
            except Exception as exc:
                result.update(status="failed", error=str(exc))
                send("error", error=str(exc))
            finally:
                if output:
                    try: output.close(success=False)
                    except Exception as exc:
                        result.update(status="failed", cleanup_error=str(exc))
                result.update(elapsed_seconds=time.monotonic() - start,
                              first_content_seconds=first - start if first else None)
                # The HTTP stream has closed before the next request is admitted.
                # Provider-release and cross-modality cancellation are R3 checks.
                with self._lock:
                    self._last = result
                    self._active = None
                self._queue.task_done()
                send("done", text=result["text"], status=result["status"],
                     tokens=result["metadata"].get("eval_count"),
                     ttft=result["first_content_seconds"], total_time=result["elapsed_seconds"],
                     audio_url=result.get("audio_url"), output_kind=result.get("speech", {}).get("receipt", {}).get("kind"))
                send("status", stage="idle")

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.config.llm.timeout + 5)
            if self._thread.is_alive():
                raise RuntimeError("Conversation worker has not released")
            self._thread = None
