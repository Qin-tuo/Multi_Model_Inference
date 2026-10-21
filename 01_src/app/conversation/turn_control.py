# SPDX-License-Identifier: Apache-2.0
"""当前轮次的唯一接纳、取消和资源释放协调者。

文字、文件语音、实时语句及提醒共享一个有界工作线程；短会话和工具提交状态也由此管理。
"""

import queue
import threading
import time
import uuid
import re
import json
import hashlib
from copy import deepcopy
from collections import deque
from pathlib import Path
from app.speech.audio import wav_pcm
from app.speech.types import SpeechOnset
from app.inference.request_io import TurnCancelled
from app.monitoring.diagnostics import TurnTrace
from app.conversation.turn_execution import TurnExecution


class TurnController:
    """当前轮次的唯一所有者：接纳、生成、工具、输出提交和取消。

    阅读顺序：_admit → _run → _model_stream → _wait_for_release。
    _lock 保护轮次身份与提交边界；后端 HTTP 请求在锁外执行。取消先阻止
    新输出和未开始的工具，再等待后端释放；已完成的持久操作保留。
    """

    def __init__(
        self,
        config,
        llm,
        broadcaster,
        camera=None,
        stt=None,
        tts=None,
        vad=None,
        perception=None,
        actions=None,
    ):
        """绑定模型与应用资源，创建单元素请求队列、轮次锁及有界历史/手势状态。"""
        self.config, self.llm, self.broadcaster, self.camera = (
            config,
            llm,
            broadcaster,
            camera,
        )
        self.stt, self.tts, self.vad = stt, tts, vad
        self.output_sink = None
        self.perception = perception
        self._queue = queue.Queue(maxsize=1)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._active = None
        self._last = None
        self._trace = None
        self._gesture_seen = deque(maxlen=128)
        self._last_greeting = float("-inf")
        self._last_gesture = None
        self._history = deque(maxlen=config.session.max_turns)
        self._session_id = uuid.uuid4().hex
        self.actions = actions
        self._action_error = None
        self._next_notice_check = 0
        self._voice = None
        self._last_voice = None

    def start(self):
        """在上一轮释放已确认后启动唯一对话线程，并重建短会话身份和实时语音状态。"""
        if self._thread and self._thread.is_alive():
            return
        with self._lock:
            if self._active is not None:
                raise RuntimeError("Previous turn release is unconfirmed")
            self._history.clear()
            self._last = None
            self._trace = None
            self._session_id = uuid.uuid4().hex
            self._voice = self._last_voice = None
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="conversation", daemon=True
        )
        self._thread.start()

    def submit(self, text, use_image=False, request_id=None):
        """校验用户文本长度和非空性，再交给统一接纳逻辑生成轮次。"""
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("Text must contain 1–4000 characters")
        return self._admit(
            {"input_kind": "text", "text": text.strip()}, use_image, request_id
        )

    def submit_audio(self, data, use_image=False, request_id=None):
        """校验语音功能和 WAV 格式/时长，将提取的 PCM 作为文件语音请求接纳。"""
        if self.stt is None or self.vad is None:
            raise ValueError("Speech input is not enabled")
        pcm = wav_pcm(data, self.config.stt.max_seconds)
        return self._admit(
            {"input_kind": "audio_file", "pcm": pcm}, use_image, request_id
        )

    def submit_segment(self, segment):
        """把实时完整语段交给已预留的话语身份，核对批次、失效事件、结束原因和帧龄。

        语段可在旧轮次释放期间暂存，但不会绕过单轮接纳限制。
        """
        if self.stt is None:
            raise ValueError("Speech input is not enabled")
        with self._lock:
            self._expire_voice_locked()
            voice = self._voice
            if voice is None or segment.utterance_id != voice["turn_id"]:
                return {"accepted": False, "reason": "utterance_not_reserved"}
            if (
                segment.input_epoch != voice["input_epoch"]
                or segment.input_invalidated is not voice["invalidated"]
                or segment.end_reason != "silence"
                or not 0 < len(segment.audio) <= self.config.stt.max_seconds * 16000
                or not 0
                <= time.monotonic() - segment.end_time
                <= self.config.audio.input_stale_seconds
            ):
                self._discard_voice_locked("invalid_or_expired_segment")
                return {"accepted": False, "reason": "invalid_or_expired_segment"}
            if voice["segment"] is not None:
                return {"accepted": False, "reason": "segment_already_received"}
            voice["segment"] = segment
            self._notify_voice_locked()
            return {
                "accepted": True,
                "turn_id": voice["turn_id"],
                "waiting_for_release": self._active is not None,
            }

    def _voice_state_locked(self):
        """在调用方持锁时生成待处理话语摘要及最近处理结果，不复制大音频数组。"""
        voice = self._voice
        return {
            "pending": None
            if voice is None
            else {
                "turn_id": voice["turn_id"],
                "input_epoch": voice["input_epoch"],
                "phase": "capturing"
                if voice["segment"] is None
                else "waiting_for_release",
                "expires_monotonic": voice["expires_monotonic"],
                "pcm_samples": len(voice["segment"].audio)
                if voice["segment"] is not None
                else 0,
            },
            "last": deepcopy(self._last_voice),
        }

    def _notify_voice_locked(self):
        """广播当前实时话语的捕获/等待/丢弃状态，调用方需持轮次锁。"""
        self.broadcaster.send({"type": "voice_state", **self._voice_state_locked()})

    def _discard_voice_locked(self, reason):
        """放弃已预留话语并记录原因、发送状态更新，返回是否确实有话语被丢弃。"""
        if self._voice is None:
            return False
        self._last_voice = {
            "turn_id": self._voice["turn_id"],
            "outcome": "discarded",
            "reason": reason,
        }
        self._voice = None
        self._notify_voice_locked()
        return True

    def _expire_voice_locked(self):
        """检查待处理话语的输入失效事件及等待期限，及时清除已不可用的预约。"""
        if self._voice is not None:
            if self._voice["invalidated"].is_set():
                self._discard_voice_locked("input_epoch_changed")
            elif time.monotonic() >= self._voice["expires_monotonic"]:
                self._discard_voice_locked("pending_voice_expired")

    def discard_voice(self, utterance_id, reason="input_discarded"):
        """只丢弃身份与指定 utterance_id 相同的预约，避免迟到通知影响新话语。"""
        with self._lock:
            if self._voice is not None and self._voice["turn_id"] == utterance_id:
                return self._discard_voice_locked(reason)
            return False

    def begin_voice(self, onset: SpeechOnset):
        """确认一个新鲜语音起点并预留唯一话语；需要打断时只绑定当时观察到的活动轮次。

        旧输入、较早话语和未启用的打断被拒绝；锁外执行取消，防止把延迟取消施加给后续请求。
        """
        target = None
        with self._lock:
            self._expire_voice_locked()
            now = time.monotonic()
            if (
                not isinstance(onset, SpeechOnset)
                or not re.fullmatch(r"[0-9a-f]{32}", onset.utterance_id)
                or onset.input_invalidated is None
                or onset.input_invalidated.is_set()
                or not onset.started_monotonic <= onset.confirmed_monotonic <= now
                or now - onset.confirmed_monotonic
                > self.config.audio.input_stale_seconds
            ):
                return {"accepted": False, "reason": "stale_voice_source"}
            if (
                self.stt is None
                or self._stop.is_set()
                or not self._thread
                or not self._thread.is_alive()
                or self._action_error
            ):
                return {"accepted": False, "reason": "not_ready"}
            active = self._active
            if active is not None:
                if not self.config.audio.voice_interruptions:
                    return {"accepted": False, "reason": "voice_interruption_disabled"}
                if onset.started_monotonic < active["admitted_monotonic"]:
                    return {"accepted": False, "reason": "voice_precedes_active_turn"}
                target = active["turn_id"]
            if self._voice is not None:
                if self._voice["turn_id"] == onset.utterance_id:
                    return {"accepted": True, "turn_id": onset.utterance_id}
                if onset.started_monotonic <= self._voice["started_monotonic"]:
                    return {"accepted": False, "reason": "older_utterance"}
                self._discard_voice_locked("superseded_by_new_voice")
            self._voice = {
                "turn_id": onset.utterance_id,
                "input_epoch": onset.input_epoch,
                "invalidated": onset.input_invalidated,
                "started_monotonic": onset.started_monotonic,
                "confirmed_monotonic": onset.confirmed_monotonic,
                "expires_monotonic": now + self.config.audio.pending_voice_seconds,
                "segment": None,
            }
            self._last_voice = {
                "turn_id": onset.utterance_id,
                "outcome": "capturing",
                "interrupted_turn": target,
            }
            self._notify_voice_locked()
        cancelled = self.cancel(target, reason="speech") if target else None
        return {
            "accepted": True,
            "turn_id": onset.utterance_id,
            "interrupted_turn": target,
            "cancellation": cancelled,
        }

    def _dispatch_voice(self):
        """在旧轮次释放且完整语段已到达后，将仍有效的预约转为正式麦克风请求。"""
        with self._lock:
            self._expire_voice_locked()
            voice = self._voice
            if (
                voice is None
                or voice["segment"] is None
                or self._active is not None
                or self._stop.is_set()
            ):
                return
            try:
                self._admit(
                    {"input_kind": "microphone", "segment": voice["segment"]},
                    self.camera is not None,
                    _voice_id=voice["turn_id"],
                )
            except (ValueError, RuntimeError) as exc:
                self._discard_voice_locked(str(exc))

    def conversation_busy(self):
        """在锁内检查是否仍有活动轮次，包括取消及释放确认阶段。"""
        with self._lock:
            return self._active is not None

    def _admit(self, data, use_image, request_id=None, *, _voice_id=None):
        """校验请求身份、输入摘要和资源状态后，原子分配轮次及取消事件并放入有界队列。

        同 ID 同输入可返回已有接纳回执；当前轮次或语音预约未释放时拒绝新请求，图像请求必须有新鲜图像或可重放回执。
        """
        if type(use_image) is not bool:
            raise ValueError("use_image must be boolean")
        if request_id is not None and (
            not isinstance(request_id, str)
            or not re.fullmatch(r"[0-9a-f]{32}", request_id)
        ):
            raise ValueError("request_id must contain 32 hexadecimal characters")
        identity = {
            "kind": data["input_kind"],
            "text": data.get("text"),
            "use_image": use_image,
        }
        raw = data.get("pcm") or (
            data["segment"].audio.tobytes() if data.get("segment") is not None else None
        )
        if raw:
            identity["audio_sha256"] = hashlib.sha256(raw).hexdigest()
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        with self._lock:
            self._expire_voice_locked()
            if self._stop.is_set() or not self._thread or not self._thread.is_alive():
                raise RuntimeError("Conversation worker is not ready")
            if self._action_error:
                raise RuntimeError(
                    "Action ledger requires recovery: " + self._action_error
                )
            if self._voice is not None and self._voice["turn_id"] != _voice_id:
                raise RuntimeError(
                    "A voice utterance is being captured or waiting for release"
                )
            if _voice_id and (
                self._voice is None or self._voice["turn_id"] != _voice_id
            ):
                raise RuntimeError("Voice reservation is no longer valid")
            if _voice_id and data["segment"].input_invalidated.is_set():
                raise RuntimeError("Voice input changed before admission")
            if self._active is not None:
                if request_id and self._active["request_id"] == request_id:
                    if self._active["fingerprint"] != digest:
                        raise ValueError("Request ID was reused for different input")
                    return {
                        "accepted": True,
                        "turn_id": self._active["turn_id"],
                        "request_id": request_id,
                        "replayed": True,
                    }
                raise RuntimeError("The current request is still running")
            if use_image and (self.camera is None or not self.camera.health_check()):
                if not (
                    request_id
                    and self.actions is not None
                    and self.actions.knows_request(request_id, digest)
                ):
                    raise ValueError("No fresh camera image is available")
            request = {
                "turn_id": _voice_id or uuid.uuid4().hex,
                "use_image": use_image,
                **data,
                "cancel_event": threading.Event(),
                "phase": "running",
                "output": None,
                "admitted_monotonic": time.monotonic(),
            }
            request.update(
                request_id=request_id or request["turn_id"], fingerprint=digest
            )
            self._trace = TurnTrace(request["turn_id"], request["admitted_monotonic"])
            request["trace"] = self._trace
            self._active = request
            if _voice_id:
                self._voice = None
                self._last_voice = {"turn_id": _voice_id, "outcome": "admitted"}
                self._notify_voice_locked()
            self.broadcaster.send(
                {"type": "status", "stage": "running", "turn_id": request["turn_id"]}
            )
            self._queue.put_nowait(request)
            return {
                "accepted": True,
                "turn_id": request["turn_id"],
                "request_id": request["request_id"],
            }

    def cancel(self, turn_id=None, reason="user"):
        """标记目标轮次取消并阻止后续输出/工具开始，随后中断匹配的模型请求和本轮播放器。

        返回仅表示取消已接纳，资源释放另行确认；已结束轮次及其他轮次不会被误取消。
        """
        with self._lock:
            discarded = False
            if self._voice is not None and (
                turn_id is None or turn_id == self._voice["turn_id"]
            ):
                discarded = self._discard_voice_locked(reason)
            request = self._active
            if request is None or (
                turn_id is not None and turn_id != request["turn_id"]
            ):
                if discarded:
                    return {
                        "accepted": True,
                        "turn_id": turn_id,
                        "pending_voice_discarded": True,
                    }
                return {"accepted": False, "reason": "turn_not_active"}
            if request["phase"] in {"releasing", "release_unconfirmed"}:
                return {
                    "accepted": False,
                    "reason": "turn_finalized",
                    "turn_id": request["turn_id"],
                }
            event = request["cancel_event"]
            event.set()
            request.update(phase="cancelling", cancel_reason=reason)
            request["trace"].mark("cancel_requested")
            output = request["output"]
            self.broadcaster.send(
                {"type": "cancelled", "turn_id": request["turn_id"], "reason": reason}
            )
        # Adapter event identity prevents a delayed abort from affecting a new turn.
        for adapter in (self.llm, self.stt):
            if adapter is not None:
                adapter.cancel(event)
        if output:
            output.cancel()
        return {
            "accepted": True,
            "turn_id": request["turn_id"],
            "resources_released": False,
        }

    def handle_gesture(self, event):
        """对感知事件去重并校验来源/时效，将挥手接入问候轮次、张掌接入统一取消。

        问候受忙闲、静音和冷却限制；停止绑定事件确认时的轮次，不能误取消之后到来的新请求。
        """
        now = time.monotonic()
        with self._lock:
            decision = {"type": "gesture", "event": event, "accepted": False}
            if event["event_id"] in self._gesture_seen:
                decision["reason"] = "duplicate"
            else:
                self._gesture_seen.append(event["event_id"])
                if not self.config.gestures.enabled:
                    decision["reason"] = "disabled"
                elif (
                    self.camera is None
                    or event["source_epoch"] != self.camera.source_epoch
                    or not event["evidence_end_monotonic"]
                    <= event["confirmed_monotonic"]
                    <= now
                    <= event["expires_monotonic"]
                ):
                    decision["reason"] = "expired_or_wrong_source"
                elif event["kind"] == "stop":
                    active = self._voice or self._active
                    if active is None:
                        decision["reason"] = "idle"
                    elif event["evidence_end_monotonic"] < active.get(
                        "admitted_monotonic", active.get("confirmed_monotonic")
                    ):
                        decision["reason"] = "frame_precedes_turn"
                    else:
                        decision["target_turn_id"] = active["turn_id"]
                elif event["kind"] == "wave":
                    muted = (
                        self.config.audio.input_mode == "device"
                        and not self.broadcaster.ptt_active
                    )
                    if not self.config.gestures.greetings_enabled or muted:
                        decision["reason"] = "greeting_muted"
                    elif self._active is not None or self._voice is not None:
                        decision["reason"] = "busy"
                    elif (
                        now - self._last_greeting
                        < self.config.gestures.greeting_cooldown_seconds
                    ):
                        decision["reason"] = "cooldown"
                    else:
                        try:
                            admitted = self._admit(
                                {
                                    "input_kind": "text",
                                    "text": "请用一句简短的中文问候向你挥手的人，不猜测对方身份。",
                                    "gesture": event,
                                },
                                False,
                            )
                            self._last_greeting = now
                            decision.update(admitted, reason="greeting_started")
                        except RuntimeError as exc:
                            decision.update(reason="not_ready", error=str(exc))
                else:
                    decision["reason"] = "unknown_gesture"
        if "target_turn_id" in decision:
            # Bind to the observed turn, so a later request cannot receive this stop.
            cancelled = self.cancel(
                decision["target_turn_id"], reason="stationary_palm"
            )
            decision.update(cancelled)
            decision["reason"] = (
                "turn_cancelled" if cancelled["accepted"] else cancelled["reason"]
            )
        with self._lock:
            self._last_gesture = decision
        self.broadcaster.send(decision)
        return decision

    def audio_file(self, turn_id):
        """将合法轮次 ID 映射到已发布 WAV，仅允许真实普通文件，拒绝任意路径与符号链接。"""
        if not re.fullmatch(r"[0-9a-f]{32}", turn_id):
            return None
        folder = Path(self.config.audio.output_directory).expanduser().resolve()
        path = folder / f"reply-{turn_id}.wav"
        return path if path.is_file() and not path.is_symlink() else None

    def session_state(self):
        """返回短会话身份和保留轮数，明确历史仅在进程内保存且不保留图片。"""
        with self._lock:
            return {
                "id": self._session_id,
                "enabled": self.config.session.enabled,
                "retained_turns": len(self._history),
                "max_turns": self.config.session.max_turns,
                "persistence": "process_memory",
                "images_retained": False,
            }

    def clear_history(self, session_id):
        """仅在会话身份匹配且没有活动轮次/语音预约时清空历史，并生成新的会话 ID。"""
        with self._lock:
            if session_id != self._session_id:
                raise RuntimeError("Session changed; refresh before clearing")
            if self._active is not None or self._voice is not None:
                raise RuntimeError(
                    "Wait for the current turn, voice input and resource release before clearing"
                )
            self._history.clear()
            self._last = None
            self._session_id = uuid.uuid4().hex
            state = self.session_state()
            self.broadcaster.send({"type": "session_cleared", "session": state})
            return {"cleared": True, "session": state}

    def _context_messages(self):
        """将已完成历史重建为配对文本和闭合工具消息；图像轮只保留文字及图像已丢弃的说明。"""
        messages = []
        for item in self._history:
            text = item["user"]
            if item["visual"]:
                text = "[该轮附有图片；历史仅保留文字，原图已丢弃。] " + text
            messages.append({"role": "user", "content": text})
            messages.extend(item.get("tool_messages", []))
            messages.append({"role": "assistant", "content": item["assistant"]})
        return messages

    def state(self):
        """在锁内汇总就绪、活动阶段、释放观测、会话及最近结果，释放未确认时不宣称可接纳。"""
        with self._lock:
            self._expire_voice_locked()
            return {
                "ready": bool(
                    self._thread
                    and self._thread.is_alive()
                    and not self._stop.is_set()
                    and not self._action_error
                    and not (
                        self._active and self._active["phase"] == "release_unconfirmed"
                    )
                ),
                "active_turn": self._active["turn_id"] if self._active else None,
                "phase": self._active["phase"] if self._active else "idle",
                "release": deepcopy(self._active.get("release"))
                if self._active
                else None,
                "last_gesture": deepcopy(self._last_gesture),
                "session": self.session_state(),
                "action_error": self._action_error,
                "voice": self._voice_state_locked(),
                "last_result": deepcopy(self._last),
                "pending": self._queue.qsize(),
                "turn_trace": self._trace.snapshot() if self._trace else None,
                "event_cursor": getattr(self.broadcaster, "cursor", 0),
            }

    def _run(self):
        """按队列顺序交由 TurnExecution 执行一轮，空闲时尝试派发实时语段和到期提醒。"""
        while not self._stop.is_set() or not self._queue.empty():
            self._dispatch_voice()
            try:
                request = self._queue.get(timeout=0.1)
            except queue.Empty:
                self._dispatch_notice()
                continue
            TurnExecution(self, request).run()

    def _model_stream(
        self,
        request,
        result,
        prompt,
        images,
        system_prompt,
        history,
        history_sizes,
        send,
        registered,
    ):
        """在同一轮内组织模型与有限次串行工具交互，验证参数并收集已提交回执。

        工具开始边界与取消共用轮次锁，已完成操作保留；模型流未完整结束时不执行不完整工具调用。
        """
        if not registered:
            if self.actions is None and not request.get("gesture"):
                system_prompt = (
                    system_prompt or self.config.llm.system_prompt
                ) + "\n当前未启用持久笔记或提醒工具，不能声称已保存笔记或已设置提醒。"
            yield from self._traced_model(
                request,
                prompt,
                images_b64=images,
                system_prompt=system_prompt,
                cancel_event=request["cancel_event"],
                few_shot=history,
                trim_history=True,
                history_group_sizes=history_sizes,
            )
            return
        from app.actions.schema import tools, instruction, parse

        system_prompt = (system_prompt or self.config.llm.system_prompt) + instruction(
            self.config.actions.timezone
        )
        chain = []
        count = 0
        dropped = 0
        result["model_rounds"] = []

        def check():
            """在下一次模型或工具操作前检查轮次取消与应用停止信号。"""
            if request["cancel_event"].is_set() or self._stop.is_set():
                raise TurnCancelled("Turn cancelled before next tool/model operation")

        while True:
            check()
            parts = []
            meta = {}
            send("status", stage="thinking")
            stream = self._traced_model(
                request,
                prompt,
                images_b64=images,
                system_prompt=system_prompt,
                cancel_event=request["cancel_event"],
                few_shot=history,
                trim_history=True,
                history_group_sizes=history_sizes,
                tools=tools() if count < self.config.actions.max_calls_per_turn else [],
                tool_messages=chain,
            )
            try:
                for content, entry in stream:
                    check()
                    meta.update(entry)
                    if content:
                        parts.append(content)
            finally:
                stream.close()
            if not meta.get("done"):
                raise RuntimeError("Incomplete model tool round")
            removed = meta.get("history_pairs_dropped", 0)
            dropped += removed
            history = history[sum(history_sizes[:removed]) :]
            history_sizes = history_sizes[removed:]
            result["model_rounds"].append(meta)
            calls = meta.get("tool_calls", [])
            if not calls:
                result["tool_messages"] = chain
                yield ("".join(parts), {**meta, "history_pairs_dropped": dropped})
                return
            if count + len(calls) > self.config.actions.max_calls_per_turn:
                raise RuntimeError("Tool budget exceeded; remaining calls not executed")
            chain.append(
                {
                    "role": "assistant",
                    "content": meta.get("assistant_content", ""),
                    "tool_calls": calls,
                }
            )
            for call in calls:
                name = call["function"]["name"]
                position = count
                count += 1
                try:
                    arguments = parse(name, call["function"]["arguments"])
                    with self._lock:
                        check()
                        request["tool_started"] = {
                            "name": name,
                            "position": position,
                            "started_monotonic": time.monotonic(),
                        }
                    send("status", stage="executing_tool")
                    with request["trace"].span("tool"):
                        value = self.actions.execute(
                            request["request_id"], position, name, arguments
                        )
                    receipt = {
                        "name": name,
                        "position": position,
                        "committed": True,
                        "result": value,
                    }
                except ValueError as exc:
                    value = {"status": "not_executed", "error": str(exc)[:500]}
                    receipt = {
                        "name": name,
                        "position": position,
                        "committed": False,
                        "result": value,
                    }
                result.setdefault("actions", []).append(receipt)
                send("action_receipt", receipt=receipt)
                check()
                chain.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(value, ensure_ascii=False),
                    }
                )

    def _traced_model(self, request, *args, **kwargs):
        # Host iterator lifetime includes consumer backpressure; not GPU decode time.
        """为模型迭代器整个消费周期计时，包括消费端背压，不能将其当作纯 GPU 解码耗时。"""
        with request["trace"].span("llm"):
            yield from self.llm.generate_stream(*args, **kwargs)

    def _dispatch_notice(self):
        """仅在应用空闲且允许通知时领取到期提醒并接入普通输出轮次，接纳失败则释放领取。"""
        if (
            self.actions is None
            or self._stop.is_set()
            or self._action_error
            or time.monotonic() < self._next_notice_check
        ):
            return
        self._next_notice_check = time.monotonic() + 0.5
        with self._lock:
            if self._active is not None or self._voice is not None:
                return
        if self.config.audio.input_mode == "device" and not self.broadcaster.ptt_active:
            return
        try:
            notice = self.actions.claim_due()
            if not notice:
                return
            try:
                self._admit(
                    {
                        "input_kind": "text",
                        "text": "提醒：" + notice["text"],
                        "notice": notice,
                    },
                    False,
                )
            except RuntimeError:
                self.actions.finish_delivery(notice, release=True)
        except Exception as exc:
            self._action_error = str(exc)
            self.broadcaster.send(
                {
                    "type": "error",
                    "error": "Reminder dispatch requires recovery: " + str(exc),
                }
            )

    def _wait_for_release(self, request, adapters):
        """持续观测本地输出和已使用后端是否空闲，全部确认后才开放下一轮接纳。

        观察超过阈值仍未确认时暴露 release_unconfirmed；停止请求可结束等待，但不会伪造已释放状态。
        """
        started = time.monotonic()
        warned = False
        while True:
            observed = {}
            for name, adapter in adapters.items():
                try:
                    observed[name] = {"idle": bool(adapter.idle())}
                except Exception as exc:
                    observed[name] = {"idle": False, "error": str(exc)}
            output = request["output"]
            if output:
                observed["output"] = {"idle": output.idle()}
            released = all(value["idle"] for value in observed.values())
            with self._lock:
                self._expire_voice_locked()
                request["release"] = {
                    "observations": observed,
                    "seconds": time.monotonic() - started,
                    "confirmed": released,
                }
                if not released and time.monotonic() - started >= 5:
                    request["phase"] = "release_unconfirmed"
            if released:
                return True
            if not warned and time.monotonic() - started >= 5:
                warned = True
                self.broadcaster.send(
                    {
                        "type": "status",
                        "turn_id": request["turn_id"],
                        "stage": "release_unconfirmed",
                    }
                )
            if self._stop.wait(0.25):
                return False

    def close(self):
        """停止接纳并取消当前轮次，等待对话线程退出；线程或后端释放未确认时明确报错。"""
        self._stop.set()
        self.cancel(reason="shutdown")
        if self._thread:
            self._thread.join(timeout=self.config.llm.timeout + 5)
            if self._thread.is_alive():
                raise RuntimeError("Conversation worker has not released")
            self._thread = None
        if self._active is not None:
            raise RuntimeError(
                "Backend release is unconfirmed; inspect conversation state"
            )
