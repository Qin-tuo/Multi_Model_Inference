# SPDX-License-Identifier: Apache-2.0
"""由控制器单工作线程执行的单轮阶段对象。

控制器保留轮次身份、取消锁和会话；本对象只拥有本轮临时输入、输出及清理进度。
"""

import json
import time
from app.speech.vad import segment_pcm
from app.speech.output import SpeechOutput, TextChunker
from app.inference.request_io import TurnCancelled


def _notice_output_callback(owner, event, notice):
    # 闭包只捕获通知所需状态，避免执行对象与输出对象互相引用；
    # 实际交付仍在控制器的同一提交锁内完成。
    """为提醒构造只捕获必要状态的输出开始回调，避免回调反向持有整个执行/输出对象形成循环引用。"""
    started = False

    def begin():
        """在控制器提交锁内检查取消，并且最多一次将通知标记为开始交付。"""
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
    """依次执行输入准备、选图、生成、提交、清理和释放确认的一轮执行上下文。"""

    def __init__(self, owner, request):
        """绑定请求和控制器，初始化本轮结果、阶段计时、已使用后端和通知回调。"""
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
        """检查本轮取消及应用停止，禁止后续阶段继续消费或提交。"""
        if self.event.is_set() or self.owner._stop.is_set():
            raise TurnCancelled("Turn cancelled")

    def send(self, kind, **values):
        """为广播消息附加轮次身份，给可丢弃内容绑定取消事件，保留结束/释放状态通知。"""
        discard = kind in {"token", "transcript"} or (
            kind == "status"
            and values.get("stage") not in {"idle", "releasing", "release_unconfirmed"}
        )
        self.owner.broadcaster.send(
            {"type": kind, "turn_id": self.request["turn_id"], **values},
            cancel_event=self.request["cancel_event"] if discard else None,
        )

    def run(self):
        """按阶段执行一轮，将异常区分为取消或失败，并无条件进入清理与释放确认。"""
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
        """登记持久请求并准备文本或 ASR 输入，重复请求直接使用已有提交回执。

        文件语音先经 VAD 分段，实时语段复用采集结果；记录输入时间口径，避免把文件偏移当作设备时间。
        """
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
        """优先取得感知结果对应的同一帧图像与辅助观察，必要时取相机新鲜快照。

        过期图像会被拒绝，辅助观测明确可能漏检；本轮保留帧元数据，不把旧图像沿用到历史轮次。
        """
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
        """消费模型或固定回执文本，发送增量并将有界文本段交给独立语音输出线程。

        工具回执重放和提醒无需再次推理；普通轮次可使用短历史，结束前等待合成完成但暂不发布文件。
        """
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
        """在控制器锁内串行完成取消检查、文件发布、通知回执和短会话更新。

        只有成功提交后才设置完成状态与音频 URL，随后进入资源释放阶段。
        """
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
        """关闭未提交输出，并终结持久请求或未完成通知的状态。

        已提交工具回执保留；台账收尾失败会暴露恢复错误，阻止后续请求把不确定状态当作成功。
        """
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
        """发布轮次结果并等待所有已使用后端及输出对象释放，确认后清除活动轮次。

        未成功提交的取消文本会丢弃；done 消息与 resources_released 是不同完成边界。
        """
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
