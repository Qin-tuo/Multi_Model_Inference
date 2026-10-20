# SPDX-License-Identifier: Apache-2.0
"""Per-turn stages executed synchronously by the controller's single worker.

The controller keeps admission, cancellation, session state and resource locks.
This object owns only one turn's transient input/output and cleanup progress.
"""

import json
import time
from app.speech.vad import segment_pcm
from app.speech.output import SpeechOutput, TextChunker
from app.request_io import TurnCancelled


def _notice_output_callback(owner, event, notice):
    # Capture only notification state, not the execution/output object. This
    # avoids a per-turn callback cycle while retaining the shared commit lock.
    started = False

    def begin():
        nonlocal started
        with owner._lock:
            if event.is_set() or owner._stop.is_set():
                raise TurnCancelled("Turn cancelled")
            if not started:
                owner.actions.begin_delivery(
                    notice, owner.config.audio.output_mode if owner.tts else "text"
                )
                started = True

    return begin


class TurnExecution:
    def __init__(self, owner, request):
        self.owner, self.request = owner, request
        self.result = {
            "turn_id": self.request["turn_id"],
            "text": "",
            "metadata": {},
            "mode": "camera_text" if self.request["use_image"] else "text",
        }
        self.result["request_id"] = self.request["request_id"]
        if self.request.get("notice"):
            self.result["notification_id"] = self.request["notice"]["id"]
        if self.request.get("gesture"):
            self.result["gesture"] = self.request["gesture"]
        self.start = time.monotonic()
        self.trace = self.request["trace"]
        self.trace.mark("dequeued")
        self.first = None
        self.output = None
        self.used = {}
        self.event = self.request["cancel_event"]
        self.registered = False
        self.notice = self.request.get("notice")
        self.begin_notice_output = _notice_output_callback(
            owner, self.event, self.notice
        )

    def check(self):
        if self.event.is_set() or self.owner._stop.is_set():
            raise TurnCancelled("Turn cancelled")

    def send(self, kind, **values):
        discard = kind in {"token", "transcript"} or (
            kind == "status"
            and values.get("stage") not in {"idle", "releasing", "release_unconfirmed"}
        )
        self.owner.broadcaster.send(
            {"type": kind, "turn_id": self.request["turn_id"], **values},
            cancel_event=self.request["cancel_event"] if discard else None,
        )

    def run(self):
        try:
            self.prepare_input()
            self.select_image()
            self.generate()
            self.commit()
        except Exception as exc:
            if self.event.is_set() or isinstance(exc, TurnCancelled):
                self.result.update(
                    status="cancelled",
                    cancel_reason=self.request.get("cancel_reason", "shutdown"),
                )
            else:
                self.result.update(status="failed", error=str(exc))
                self.send("error", error=str(exc))
        finally:
            self.cleanup()
            self.finish()

    def prepare_input(self):
        """Register durable requests and prepare text or ASR input."""
        self.check()
        if self.request["input_kind"] == "microphone":
            segment = self.request["segment"]
            self.result["voice_input"] = {
                "utterance_id": segment.utterance_id,
                "input_epoch": segment.input_epoch,
                "end_reason": segment.end_reason,
                "samples": len(segment.audio),
                "wait_before_admission_seconds": max(
                    0, self.request["admitted_monotonic"] - segment.end_time
                ),
            }
        self.prompt = self.request.get("text")
        self.fixed = self.prompt if self.notice else None
        if (
            self.owner.actions is not None
            and (not self.request.get("gesture"))
            and (not self.notice)
        ):
            from app.actions.schema import receipts_text

            registration = self.owner.actions.begin_request(
                self.request["request_id"], self.request["fingerprint"]
            )
            if registration["replay"]:
                self.result.update(
                    replayed=True,
                    actions=registration["receipts"],
                    original_request_status=registration["status"],
                )
                self.result["mode"] = "receipt_replay"
                self.fixed = "此前请求的已提交回执：" + receipts_text(
                    registration["receipts"]
                )
                self.prompt = self.prompt or "[重复语音请求，沿用已有回执]"
            else:
                self.registered = True
        if self.request["input_kind"] != "text" and self.fixed is None:
            self.send("status", stage="transcribing")
            if self.request["input_kind"] == "microphone":
                segments = [self.request["segment"]]
            else:
                with self.trace.span("vad"):
                    segments = list(
                        segment_pcm(
                            self.request["pcm"], self.owner.config.vad, self.owner.vad
                        )
                    )
            if not segments:
                raise ValueError("No speech detected in audio input")
            self.result["asr"] = []
            for segment in segments:
                self.check()
                self.used["asr"] = self.owner.stt
                with self.trace.span("asr"):
                    transcript = self.owner.stt.transcribe(
                        segment.audio, cancel_event=self.event
                    )
                self.check()
                if transcript.get("error"):
                    raise RuntimeError(transcript["error"])
                self.result["asr"].append(transcript)
            self.prompt = "".join((item["text"] for item in self.result["asr"]))
            if not self.prompt.strip() or len(self.prompt) > 4000:
                raise ValueError("Invalid combined ASR transcript")
            self.result["input_kind"] = self.request["input_kind"]
            self.result["vad_segments"] = [
                {"duration": x.duration, "start": x.start_time, "end": x.end_time}
                for x in segments
            ]
            self.result["vad_time_kind"] = (
                "file_offset_with_end_padding"
                if self.request["input_kind"] == "audio_file"
                else "host_receive_monotonic"
            )
        self.check()
        self.result["prompt"] = self.prompt

    def select_image(self):
        """Pair the image and auxiliary observations from the selected frame."""
        self.images = None
        self.vision_prompt = self.owner.config.vision.system_prompt
        if self.request["use_image"] and self.fixed is None:
            frame = None
            if self.owner.perception:
                observed, scene = self.owner.perception.preview()
                if observed is not None and scene["ready"]:
                    frame = observed
                    providers = scene.get("providers", {})
                    observations = {
                        "person_candidates": len(scene["persons"])
                        if providers.get("body")
                        else None,
                        "prominent_face_detected": scene["face"]["face_detected"]
                        if providers.get("face")
                        else None,
                        "hand_candidates": len(scene["hands"])
                        if providers.get("hands")
                        else None,
                    }
                    self.result["perception_observation"] = {
                        "frame": scene["frame"],
                        "values": observations,
                    }
                    self.vision_prompt += (
                        "\n本帧小模型的辅助观测（可能漏检，请结合图像判断）："
                        + json.dumps(observations, ensure_ascii=False)
                    )
            if frame is None:
                frame = self.owner.camera.snapshot()
            if frame is None:
                raise ValueError("Camera image expired before processing")
            self.result["frame"] = frame.metadata()
            self.images = [self.owner.camera._base64_jpeg(frame.jpeg)]

    def generate(self):
        """Consume model output and bounded speech chunks on the controller worker."""
        if not self.notice:
            self.send("transcript", text=self.prompt)
        self.send("status", stage="thinking")
        if self.owner.tts is not None:
            with self.owner._lock:
                self.check()
                self.output = SpeechOutput(
                    self.owner.tts,
                    self.owner.config.audio,
                    self.request["turn_id"],
                    self.owner.output_sink,
                    cancel_event=self.event,
                    defer_commit=True,
                    on_output_start=self.begin_notice_output if self.notice else None,
                    trace=self.trace,
                )
                self.request["output"] = self.output
            self.used["tts"] = self.owner.tts
        chunker = TextChunker(
            self.owner.config.tts.first_chunk_chars,
            self.owner.config.tts.max_chunk_chars,
        )
        self.check()
        self.remember = (
            self.owner.config.session.enabled
            and (not self.request.get("gesture"))
            and (not self.notice)
            and (not self.result.get("replayed"))
        )
        with self.owner._lock:
            self.history = self.owner._context_messages() if self.remember else []
            self.history_sizes = (
                [2 + len(item.get("tool_messages", [])) for item in self.owner._history]
                if self.remember
                else []
            )
        if self.fixed is not None:
            stream = iter(
                [
                    (
                        self.fixed,
                        {
                            "done": True,
                            "source": "notification"
                            if self.notice
                            else "durable_receipts",
                        },
                    )
                ]
            )
        else:
            self.used["llm"] = self.owner.llm
            stream = self.owner._model_stream(
                self.request,
                self.result,
                self.prompt,
                self.images,
                self.vision_prompt if self.images else None,
                self.history,
                self.history_sizes,
                self.send,
                self.registered,
            )
        try:
            for text, metadata in stream:
                self.check()
                self.result["metadata"].update(metadata)
                if text:
                    if self.notice and self.output is None:
                        self.begin_notice_output()
                    if len(self.result["text"]) + len(text) > 16384:
                        raise ValueError("Model text exceeds turn response budget")
                    self.first = self.first or time.monotonic()
                    self.trace.mark("first_content")
                    self.result["text"] += text
                    self.send("token", text=text)
                    if self.output:
                        for chunk in chunker.feed(text):
                            self.output.submit(chunk)
        finally:
            if hasattr(stream, "close"):
                stream.close()
        if not self.result["metadata"].get("done") or not self.result["text"].strip():
            raise RuntimeError("Model did not produce a complete text answer")
        if self.output:
            self.send(
                "status",
                stage="synthesizing"
                if self.owner.config.audio.output_mode == "file"
                else "speaking",
            )
            for chunk in chunker.finish():
                self.output.submit(chunk)
            self.output.close(success=True)

    def commit(self):
        """Serialize publication, cancellation and session updates under the owner lock."""
        with self.owner._lock:
            self.check()
            if self.output:
                receipt = self.output.publish()
                self.trace.mark("output_published")
                self.result["speech"] = {
                    "receipt": receipt,
                    "chunks": self.output.chunks,
                    "duration": self.output.bytes_written / 32000,
                }
                if receipt["kind"] == "file_written":
                    self.result["audio_url"] = f"/api/audio/{self.request['turn_id']}"
            if self.notice:
                self.owner.actions.finish_delivery(
                    self.notice,
                    self.result.get("speech", {}).get("receipt")
                    or {"kind": "text_available", "turn_id": self.request["turn_id"]},
                )
            self.request["phase"] = "releasing"
            self.result["status"] = "complete"
            if self.remember:
                for _ in range(self.result["metadata"].get("history_pairs_dropped", 0)):
                    if self.owner._history:
                        self.owner._history.popleft()
                delivered = self.result["text"]
                if self.result["metadata"].get("finish_reason") == "length":
                    delivered += "\n[本次回答达到输出上限，文字被截断。]"
                self.owner._history.append(
                    {
                        "user": self.prompt,
                        "assistant": delivered,
                        "visual": bool(self.images),
                        "tool_messages": self.result.get("tool_messages", []),
                    }
                )
                self.owner.broadcaster.send(
                    {"type": "session_state", "session": self.owner.session_state()}
                )
            self.result["session_id"] = self.owner._session_id
        self.output = None

    def cleanup(self):
        """Close uncommitted output and finalize durable request/delivery receipts."""
        if self.output:
            try:
                self.output.close(success=False)
            except Exception as exc:
                self.result["cleanup_error"] = str(exc)
        if self.registered:
            try:
                self.owner.actions.finish_request(
                    self.request["request_id"],
                    "cancelled" if self.event.is_set() else self.result["status"],
                )
                self.result["actions_committed"] = self.owner.actions.receipts(
                    self.request["request_id"]
                )
            except Exception as exc:
                self.owner._action_error = str(exc)
                self.result["action_finalize_error"] = str(exc)
        if self.notice and self.result.get("status") != "complete":
            try:
                self.owner.actions.finish_delivery(
                    self.notice,
                    cancelled=self.event.is_set(),
                    error=self.result.get("error", "turn cancelled"),
                )
            except Exception as exc:
                self.owner._action_error = str(exc)
                self.result["notification_error"] = str(exc)

    def finish(self):
        """Report the turn and release admission only after every used backend settles."""
        self.result.update(
            elapsed_seconds=time.monotonic() - self.start,
            first_content_seconds=self.first - self.start if self.first else None,
        )
        self.trace.mark("worker_done")
        with self.owner._lock:
            if self.event.is_set() and self.result["status"] != "complete":
                self.result["status"] = "cancelled"
                self.result["discarded_text_chars"] = len(self.result["text"])
                self.result["text"] = ""
            self.owner._last = self.result
            self.request["phase"] = "releasing"
        self.send("status", stage="releasing")
        self.send(
            "done",
            text=self.result["text"],
            status=self.result["status"],
            tokens=self.result["metadata"].get("eval_count"),
            ttft=self.result["first_content_seconds"],
            total_time=self.result["elapsed_seconds"],
            actions_committed=self.result.get("actions_committed", []),
            audio_url=self.result.get("audio_url"),
            output_kind=self.result.get("speech", {}).get("receipt", {}).get("kind"),
        )
        with self.trace.span("release"):
            released = self.owner._wait_for_release(self.request, self.used)
        self.trace.finish(self.result["status"] if released else "release_unconfirmed")
        with self.owner._lock:
            self.result["resources_released"] = released
            self.result["release"] = self.request["release"]
            self.owner._last = self.result
            if released:
                self.owner._active = None
            self.send("status", stage="idle" if released else "release_unconfirmed")
        self.owner._queue.task_done()
