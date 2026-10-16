# SPDX-License-Identifier: Apache-2.0
"""One bounded request worker around the upstream VLM streaming sequence.

Owns text, file/live audio, VLM, session context, local actions, bounded TTS
output and whole-turn cancellation.
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
from app.audio import wav_pcm
from app.pipeline import segment_pcm, SpeechOutput, TextChunker, SpeechOnset
from app.request_io import TurnCancelled


class TurnController:
    def __init__(self, config, llm, broadcaster, camera=None, stt=None, tts=None, vad=None, perception=None, actions=None):
        self.config, self.llm, self.broadcaster, self.camera = config, llm, broadcaster, camera
        self.stt, self.tts, self.vad = stt, tts, vad
        self.output_sink = None
        self.perception = perception
        self._queue = queue.Queue(maxsize=1)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._active = None
        self._last = None
        self._gesture_seen = deque(maxlen=128)
        self._last_greeting = float('-inf')
        self._last_gesture = None
        self._history = deque(maxlen=config.session.max_turns)
        self._session_id = uuid.uuid4().hex
        self.actions = actions
        self._action_error = None
        self._next_notice_check = 0
        self._voice = None
        self._last_voice = None

    def start(self):
        if self._thread and self._thread.is_alive(): return
        with self._lock:
            if self._active is not None: raise RuntimeError('Previous turn release is unconfirmed')
            self._history.clear()
            self._last = None
            self._session_id = uuid.uuid4().hex
            self._voice = self._last_voice = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="conversation", daemon=True)
        self._thread.start()

    def submit(self, text, use_image=False, request_id=None):
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("Text must contain 1–4000 characters")
        return self._admit(
            {"input_kind": "text", "text": text.strip()}, use_image, request_id
        )

    def submit_audio(self, data, use_image=False, request_id=None):
        if self.stt is None or self.vad is None:
            raise ValueError("Speech input is not enabled")
        pcm = wav_pcm(data, self.config.stt.max_seconds)
        return self._admit(
            {"input_kind": "audio_file", "pcm": pcm}, use_image, request_id
        )

    def submit_segment(self, segment):
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
        self.broadcaster.send({"type": "voice_state", **self._voice_state_locked()})

    def _discard_voice_locked(self, reason):
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
        if self._voice is not None:
            if self._voice["invalidated"].is_set():
                self._discard_voice_locked("input_epoch_changed")
            elif time.monotonic() >= self._voice["expires_monotonic"]:
                self._discard_voice_locked("pending_voice_expired")

    def discard_voice(self, utterance_id, reason="input_discarded"):
        with self._lock:
            if self._voice is not None and self._voice["turn_id"] == utterance_id:
                return self._discard_voice_locked(reason)
            return False

    def begin_voice(self, onset: SpeechOnset):
        """Reserve one utterance; cancel only the turn observed at confirmation."""
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
        with self._lock:return self._active is not None

    def _admit(self, data, use_image, request_id=None, *, _voice_id=None):
        if type(use_image) is not bool: raise ValueError("use_image must be boolean")
        if request_id is not None and (not isinstance(request_id,str) or not re.fullmatch(r'[0-9a-f]{32}',request_id)):
            raise ValueError('request_id must contain 32 hexadecimal characters')
        identity={'kind':data['input_kind'],'text':data.get('text'),'use_image':use_image}
        raw=data.get('pcm') or (data['segment'].audio.tobytes() if data.get('segment') is not None else None)
        if raw:identity['audio_sha256']=hashlib.sha256(raw).hexdigest()
        digest=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        with self._lock:
            self._expire_voice_locked()
            if self._stop.is_set() or not self._thread or not self._thread.is_alive():
                raise RuntimeError("Conversation worker is not ready")
            if self._action_error:raise RuntimeError('Action ledger requires recovery: '+self._action_error)
            if self._voice is not None and self._voice['turn_id']!=_voice_id:
                raise RuntimeError('A voice utterance is being captured or waiting for release')
            if _voice_id and (self._voice is None or self._voice['turn_id']!=_voice_id):
                raise RuntimeError('Voice reservation is no longer valid')
            if _voice_id and data['segment'].input_invalidated.is_set():
                raise RuntimeError('Voice input changed before admission')
            if self._active is not None:
                if request_id and self._active['request_id']==request_id:
                    if self._active['fingerprint']!=digest:raise ValueError('Request ID was reused for different input')
                    return {'accepted':True,'turn_id':self._active['turn_id'],'request_id':request_id,'replayed':True}
                raise RuntimeError("The current request is still running")
            if use_image and (self.camera is None or not self.camera.health_check()):
                if not (request_id and self.actions is not None and self.actions.knows_request(request_id,digest)):
                    raise ValueError("No fresh camera image is available")
            request = {"turn_id": _voice_id or uuid.uuid4().hex, "use_image": use_image, **data,
                       "cancel_event": threading.Event(), "phase": "running", "output": None,
                       "admitted_monotonic":time.monotonic()}
            request.update(request_id=request_id or request['turn_id'],fingerprint=digest)
            self._active = request
            if _voice_id:
                self._voice=None
                self._last_voice={'turn_id':_voice_id,'outcome':'admitted'}
                self._notify_voice_locked()
            self.broadcaster.send({"type":"status", "stage":"running", "turn_id":request["turn_id"]})
            self._queue.put_nowait(request)
            return {"accepted": True, "turn_id": request["turn_id"], 'request_id':request['request_id']}

    def cancel(self, turn_id=None, reason="user"):
        with self._lock:
            discarded=False
            if self._voice is not None and (turn_id is None or turn_id==self._voice['turn_id']):
                discarded=self._discard_voice_locked(reason)
            request = self._active
            if request is None or (turn_id is not None and turn_id != request["turn_id"]):
                if discarded:return {'accepted':True,'turn_id':turn_id,'pending_voice_discarded':True}
                return {"accepted": False, "reason": "turn_not_active"}
            if request["phase"] in {"releasing", "release_unconfirmed"}:
                return {"accepted": False, "reason": "turn_finalized", "turn_id": request["turn_id"]}
            event = request["cancel_event"]
            event.set()
            request.update(phase="cancelling", cancel_reason=reason)
            output = request["output"]
            self.broadcaster.send({"type":"cancelled", "turn_id":request["turn_id"], "reason":reason})
        # Adapter event identity prevents a delayed abort from affecting a new turn.
        for adapter in (self.llm, self.stt):
            if adapter is not None: adapter.cancel(event)
        if output: output.cancel()
        return {"accepted":True, "turn_id":request["turn_id"], "resources_released":False}

    def handle_gesture(self, event):
        """Route fresh perception events through the same admission/cancel owner."""
        now = time.monotonic()
        with self._lock:
            decision = {'type':'gesture', 'event':event, 'accepted':False}
            if event['event_id'] in self._gesture_seen:
                decision['reason'] = 'duplicate'
            else:
                self._gesture_seen.append(event['event_id'])
                if not self.config.gestures.enabled:
                    decision['reason'] = 'disabled'
                elif (self.camera is None or event['source_epoch']!=self.camera.source_epoch or
                      not event['evidence_end_monotonic'] <= event['confirmed_monotonic'] <= now <= event['expires_monotonic']):
                    decision['reason'] = 'expired_or_wrong_source'
                elif event['kind']=='stop':
                    active = self._voice or self._active
                    if active is None:
                        decision['reason'] = 'idle'
                    elif event['evidence_end_monotonic'] < active.get('admitted_monotonic',active.get('confirmed_monotonic')):
                        decision['reason'] = 'frame_precedes_turn'
                    else:
                        decision['target_turn_id'] = active['turn_id']
                elif event['kind']=='wave':
                    muted = self.config.audio.input_mode=='device' and not self.broadcaster.ptt_active
                    if not self.config.gestures.greetings_enabled or muted:
                        decision['reason'] = 'greeting_muted'
                    elif self._active is not None or self._voice is not None:
                        decision['reason'] = 'busy'
                    elif now-self._last_greeting < self.config.gestures.greeting_cooldown_seconds:
                        decision['reason'] = 'cooldown'
                    else:
                        try:
                            admitted = self._admit({'input_kind':'text', 'text':'请用一句简短的中文问候向你挥手的人，不猜测对方身份。',
                                                    'gesture':event}, False)
                            self._last_greeting = now
                            decision.update(admitted,reason='greeting_started')
                        except RuntimeError as exc:decision.update(reason='not_ready',error=str(exc))
                else:
                    decision['reason'] = 'unknown_gesture'
        if 'target_turn_id' in decision:
            # Bind to the observed turn, so a later request cannot receive this stop.
            cancelled = self.cancel(decision['target_turn_id'], reason='stationary_palm')
            decision.update(cancelled)
            decision['reason'] = 'turn_cancelled' if cancelled['accepted'] else cancelled['reason']
        with self._lock:self._last_gesture = decision
        self.broadcaster.send(decision)
        return decision

    def audio_file(self, turn_id):
        if not re.fullmatch(r"[0-9a-f]{32}", turn_id): return None
        folder = Path(self.config.audio.output_directory).expanduser().resolve()
        path = folder / f"reply-{turn_id}.wav"
        return path if path.is_file() and not path.is_symlink() else None

    def session_state(self):
        with self._lock:
            return {'id':self._session_id, 'enabled':self.config.session.enabled,
                    'retained_turns':len(self._history), 'max_turns':self.config.session.max_turns,
                    'persistence':'process_memory', 'images_retained':False}

    def clear_history(self, session_id):
        with self._lock:
            if session_id != self._session_id: raise RuntimeError('Session changed; refresh before clearing')
            if self._active is not None or self._voice is not None: raise RuntimeError('Wait for the current turn, voice input and resource release before clearing')
            self._history.clear(); self._last = None; self._session_id = uuid.uuid4().hex
            state = self.session_state()
            self.broadcaster.send({'type':'session_cleared', 'session':state})
            return {'cleared':True, 'session':state}

    def _context_messages(self):
        messages = []
        for item in self._history:
            text = item['user']
            if item['visual']:text = '[该轮附有图片；历史仅保留文字，原图已丢弃。] '+text
            messages.append({'role':'user','content':text})
            messages.extend(item.get('tool_messages',[]))
            messages.append({'role':'assistant','content':item['assistant']})
        return messages

    def state(self):
        with self._lock:
            self._expire_voice_locked()
            return {"ready": bool(self._thread and self._thread.is_alive() and not self._stop.is_set()
                                  and not self._action_error and not (self._active and self._active["phase"] == "release_unconfirmed")),
                    "active_turn": self._active["turn_id"] if self._active else None,
                    "phase": self._active["phase"] if self._active else "idle",
                    "release": deepcopy(self._active.get("release")) if self._active else None,
                    "last_gesture": deepcopy(self._last_gesture),
                    "session": self.session_state(),
                    "action_error": self._action_error,
                    'voice':self._voice_state_locked(),
                    "last_result": deepcopy(self._last), "pending": self._queue.qsize(),
                    "event_cursor": getattr(self.broadcaster,'cursor',0)}

    def _run(self):
        while not self._stop.is_set() or not self._queue.empty():
            self._dispatch_voice()
            try: request = self._queue.get(timeout=0.1)
            except queue.Empty:
                self._dispatch_notice()
                continue
            result = {"turn_id": request["turn_id"], "text": "", "metadata": {}, "mode": "camera_text" if request["use_image"] else "text"}
            result['request_id'] = request['request_id']
            if request.get('notice'):result['notification_id'] = request['notice']['id']
            if request.get('gesture'):result['gesture'] = request['gesture']
            def send(kind, **values):
                discard = kind in {"token", "transcript"} or (kind == "status" and values.get("stage") not in {"idle", "releasing", "release_unconfirmed"})
                self.broadcaster.send({"type":kind, "turn_id":request["turn_id"], **values},
                                      cancel_event=request["cancel_event"] if discard else None)
            start = time.monotonic()
            first = None
            output = None
            used = {}
            event = request["cancel_event"]
            registered = False
            notice = request.get('notice')
            delivery_started = False
            def begin_notice_output():
                nonlocal delivery_started
                with self._lock:
                    check()
                    if not delivery_started:
                        self.actions.begin_delivery(notice,self.config.audio.output_mode if self.tts else 'text')
                        delivery_started = True
            def check():
                if event.is_set() or self._stop.is_set(): raise TurnCancelled("Turn cancelled")
            try:
                check()
                if request['input_kind']=='microphone':
                    segment=request['segment']
                    result['voice_input']={'utterance_id':segment.utterance_id,'input_epoch':segment.input_epoch,
                        'end_reason':segment.end_reason,'samples':len(segment.audio),
                        'wait_before_admission_seconds':max(0,request['admitted_monotonic']-segment.end_time)}
                prompt = request.get("text")
                fixed = prompt if notice else None
                if self.actions is not None and not request.get('gesture') and not notice:
                    from app.actions.schema import receipts_text
                    registration = self.actions.begin_request(request['request_id'],request['fingerprint'])
                    if registration['replay']:
                        result.update(replayed=True,actions=registration['receipts'],original_request_status=registration['status'])
                        result['mode'] = 'receipt_replay'
                        fixed = '此前请求的已提交回执：'+receipts_text(registration['receipts'])
                        prompt = prompt or '[重复语音请求，沿用已有回执]'
                    else:registered = True
                if request["input_kind"] != "text" and fixed is None:
                    send("status", stage="transcribing")
                    segments = ([request["segment"]] if request["input_kind"] == "microphone" else
                                list(segment_pcm(request["pcm"], self.config.vad, self.vad)))
                    if not segments: raise ValueError("No speech detected in audio input")
                    result["asr"] = []
                    for segment in segments:
                        check()
                        used["asr"] = self.stt
                        transcript = self.stt.transcribe(segment.audio, cancel_event=event)
                        check()
                        if transcript.get("error"): raise RuntimeError(transcript["error"])
                        result["asr"].append(transcript)
                    prompt = "".join(item["text"] for item in result["asr"])
                    if not prompt.strip() or len(prompt) > 4000: raise ValueError("Invalid combined ASR transcript")
                    result["input_kind"] = request["input_kind"]
                    result["vad_segments"] = [{"duration": x.duration, "start": x.start_time, "end": x.end_time} for x in segments]
                    result["vad_time_kind"] = "file_offset_with_end_padding" if request["input_kind"] == "audio_file" else "host_receive_monotonic"
                check()
                result["prompt"] = prompt
                images = None
                vision_prompt = self.config.vision.system_prompt
                if request["use_image"] and fixed is None:
                    frame = None
                    if self.perception:
                        observed, scene = self.perception.preview()
                        if observed is not None and scene['ready']:
                            frame = observed
                            providers = scene.get('providers', {})
                            observations = {'person_candidates':len(scene['persons']) if providers.get('body') else None,
                                            'prominent_face_detected':scene['face']['face_detected'] if providers.get('face') else None,
                                            'hand_candidates':len(scene['hands']) if providers.get('hands') else None}
                            result['perception_observation'] = {'frame':scene['frame'],'values':observations}
                            vision_prompt += '\n本帧小模型的辅助观测（可能漏检，请结合图像判断）：'+json.dumps(observations,ensure_ascii=False)
                    if frame is None: frame = self.camera.snapshot()
                    if frame is None: raise ValueError("Camera image expired before processing")
                    result["frame"] = frame.metadata()
                    images = [self.camera._base64_jpeg(frame.jpeg)]
                if not notice:send("transcript", text=prompt)
                send("status", stage="thinking")
                if self.tts is not None:
                    with self._lock:
                        check()
                        output = SpeechOutput(self.tts, self.config.audio, request["turn_id"], self.output_sink,
                                              cancel_event=event, defer_commit=True,
                                              on_output_start=begin_notice_output if notice else None)
                        request["output"] = output
                    used["tts"] = self.tts
                chunker = TextChunker(self.config.tts.first_chunk_chars, self.config.tts.max_chunk_chars)
                check()
                remember = self.config.session.enabled and not request.get('gesture') and not notice and not result.get('replayed')
                with self._lock:
                    history = self._context_messages() if remember else []
                    history_sizes = [2+len(item.get('tool_messages',[])) for item in self._history] if remember else []
                if fixed is not None:
                    stream = iter([(fixed,{'done':True,'source':'notification' if notice else 'durable_receipts'})])
                else:
                    used['llm'] = self.llm
                    stream = self._model_stream(request,result,prompt,images,vision_prompt if images else None,history,history_sizes,send,registered)
                try:
                    for text, metadata in stream:
                        check()
                        result["metadata"].update(metadata)
                        if text:
                            if notice and output is None:begin_notice_output()
                            if len(result['text'])+len(text)>16384:raise ValueError('Model text exceeds turn response budget')
                            first = first or time.monotonic()
                            result["text"] += text
                            send("token", text=text)
                            if output:
                                for chunk in chunker.feed(text): output.submit(chunk)
                finally:
                    if hasattr(stream,'close'):stream.close()
                if not result["metadata"].get("done") or not result["text"].strip():
                    raise RuntimeError("Model did not produce a complete text answer")
                if output:
                    send("status", stage="synthesizing" if self.config.audio.output_mode == "file" else "speaking")
                    for chunk in chunker.finish(): output.submit(chunk)
                    output.close(success=True)
                # Commit and cancellation are serialized. A cancelled staged WAV
                # is never published; a completed earlier turn stays intact.
                with self._lock:
                    check()
                    if output:
                        receipt = output.publish()
                        result["speech"] = {"receipt":receipt, "chunks":output.chunks, "duration":output.bytes_written/32000}
                        if receipt["kind"] == "file_written": result["audio_url"] = f"/api/audio/{request['turn_id']}"
                    if notice:
                        self.actions.finish_delivery(notice,result.get('speech',{}).get('receipt') or {'kind':'text_available','turn_id':request['turn_id']})
                    request["phase"] = "releasing"
                    result["status"] = "complete"
                    if remember:
                        for _ in range(result['metadata'].get('history_pairs_dropped',0)):
                            if self._history:self._history.popleft()
                        delivered = result['text']
                        if result['metadata'].get('finish_reason')=='length':delivered += '\n[本次回答达到输出上限，文字被截断。]'
                        self._history.append({'user':prompt,'assistant':delivered,'visual':bool(images),'tool_messages':result.get('tool_messages',[])})
                        self.broadcaster.send({'type':'session_state','session':self.session_state()})
                    result['session_id'] = self._session_id
                output = None
            except Exception as exc:
                if event.is_set() or isinstance(exc, TurnCancelled):
                    result.update(status="cancelled", cancel_reason=request.get("cancel_reason", "shutdown"))
                else:
                    result.update(status="failed", error=str(exc))
                    send("error", error=str(exc))
            finally:
                if output:
                    try: output.close(success=False)
                    except Exception as exc:
                        result["cleanup_error"] = str(exc)
                if registered:
                    try:
                        self.actions.finish_request(request['request_id'],'cancelled' if event.is_set() else result['status'])
                        result['actions_committed'] = self.actions.receipts(request['request_id'])
                    except Exception as exc:
                        self._action_error = str(exc);result['action_finalize_error'] = str(exc)
                if notice and result.get('status')!='complete':
                    try:self.actions.finish_delivery(notice,cancelled=event.is_set(),error=result.get('error','turn cancelled'))
                    except Exception as exc:self._action_error=str(exc);result['notification_error']=str(exc)
                result.update(elapsed_seconds=time.monotonic() - start,
                              first_content_seconds=first - start if first else None)
                with self._lock:
                    if event.is_set() and result["status"] != "complete":
                        result["status"] = "cancelled"
                        result["discarded_text_chars"] = len(result["text"])
                        result["text"] = ""
                    self._last = result
                    request["phase"] = "releasing"
                send("status", stage="releasing")
                send("done", text=result["text"], status=result["status"],
                     tokens=result["metadata"].get("eval_count"),
                     ttft=result["first_content_seconds"], total_time=result["elapsed_seconds"],
                     actions_committed=result.get('actions_committed',[]),
                     audio_url=result.get("audio_url"), output_kind=result.get("speech", {}).get("receipt", {}).get("kind"))
                released = self._wait_for_release(request, used)
                with self._lock:
                    result["resources_released"] = released
                    result["release"] = request["release"]
                    self._last = result
                    if released: self._active = None
                    send("status", stage="idle" if released else "release_unconfirmed")
                self._queue.task_done()

    def _model_stream(self,request,result,prompt,images,system_prompt,history,history_sizes,send,registered):
        """At most two serial tool executions, still owned by the same turn."""
        if not registered:
            if self.actions is None and not request.get('gesture'):
                system_prompt=(system_prompt or self.config.llm.system_prompt)+'\n当前未启用持久笔记或提醒工具，不能声称已保存笔记或已设置提醒。'
            yield from self.llm.generate_stream(prompt,images_b64=images,system_prompt=system_prompt,
                cancel_event=request['cancel_event'],few_shot=history,trim_history=True,history_group_sizes=history_sizes)
            return
        from app.actions.schema import tools,instruction,parse
        system_prompt=(system_prompt or self.config.llm.system_prompt)+instruction(self.config.actions.timezone)
        chain=[];count=0;dropped=0;result['model_rounds']=[]
        def check():
            if request['cancel_event'].is_set() or self._stop.is_set():raise TurnCancelled('Turn cancelled before next tool/model operation')
        while True:
            check();parts=[];meta={}
            send('status',stage='thinking')
            stream=self.llm.generate_stream(prompt,images_b64=images,system_prompt=system_prompt,
                cancel_event=request['cancel_event'],few_shot=history,trim_history=True,
                history_group_sizes=history_sizes,
                tools=tools() if count<self.config.actions.max_calls_per_turn else [],tool_messages=chain)
            try:
                for content,entry in stream:
                    check();meta.update(entry)
                    if content:parts.append(content)
            finally:stream.close()
            if not meta.get('done'):raise RuntimeError('Incomplete model tool round')
            removed=meta.get('history_pairs_dropped',0);dropped+=removed
            history=history[sum(history_sizes[:removed]):];history_sizes=history_sizes[removed:]
            result['model_rounds'].append(meta)
            calls=meta.get('tool_calls',[])
            if not calls:
                result['tool_messages'] = chain
                yield(''.join(parts),{**meta,'history_pairs_dropped':dropped})
                return
            if count+len(calls)>self.config.actions.max_calls_per_turn:raise RuntimeError('Tool budget exceeded; remaining calls not executed')
            chain.append({'role':'assistant','content':meta.get('assistant_content',''),'tool_calls':calls})
            for call in calls:
                name=call['function']['name'];position=count;count+=1
                try:
                    arguments=parse(name,call['function']['arguments'])
                    with self._lock:
                        check()
                        request['tool_started']={'name':name,'position':position,'started_monotonic':time.monotonic()}
                    send('status',stage='executing_tool')
                    value=self.actions.execute(request['request_id'],position,name,arguments)
                    receipt={'name':name,'position':position,'committed':True,'result':value}
                except ValueError as exc:
                    value={'status':'not_executed','error':str(exc)[:500]}
                    receipt={'name':name,'position':position,'committed':False,'result':value}
                result.setdefault('actions',[]).append(receipt)
                send('action_receipt',receipt=receipt)
                check()
                chain.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps(value,ensure_ascii=False)})

    def _dispatch_notice(self):
        if self.actions is None or self._stop.is_set() or self._action_error or time.monotonic()<self._next_notice_check:return
        self._next_notice_check=time.monotonic()+.5
        with self._lock:
            if self._active is not None or self._voice is not None:return
        if self.config.audio.input_mode=='device' and not self.broadcaster.ptt_active:return
        try:
            notice=self.actions.claim_due()
            if not notice:return
            try:self._admit({'input_kind':'text','text':'提醒：'+notice['text'],'notice':notice},False)
            except RuntimeError:self.actions.finish_delivery(notice,release=True)
        except Exception as exc:
            self._action_error=str(exc)
            self.broadcaster.send({'type':'error','error':'Reminder dispatch requires recovery: '+str(exc)})

    def _wait_for_release(self, request, adapters):
        """Keep admission closed until local output and native workers are idle."""
        started = time.monotonic()
        warned = False
        while True:
            observed = {}
            for name, adapter in adapters.items():
                try: observed[name] = {"idle": bool(adapter.idle())}
                except Exception as exc: observed[name] = {"idle":False, "error":str(exc)}
            output = request["output"]
            if output: observed["output"] = {"idle":output.idle()}
            released = all(value["idle"] for value in observed.values())
            with self._lock:
                self._expire_voice_locked()
                request["release"] = {"observations":observed, "seconds":time.monotonic()-started, "confirmed":released}
                if not released and time.monotonic()-started >= 5:
                    request["phase"] = "release_unconfirmed"
            if released: return True
            if not warned and time.monotonic()-started >= 5:
                warned = True
                self.broadcaster.send({"type":"status", "turn_id":request["turn_id"], "stage":"release_unconfirmed"})
            if self._stop.wait(0.25): return False

    def close(self):
        self._stop.set()
        self.cancel(reason="shutdown")
        if self._thread:
            self._thread.join(timeout=self.config.llm.timeout + 5)
            if self._thread.is_alive():
                raise RuntimeError("Conversation worker has not released")
            self._thread = None
        if self._active is not None:
            raise RuntimeError("Backend release is unconfirmed; inspect conversation state")
