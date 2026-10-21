# SPDX-License-Identifier: Apache-2.0
"""Bench-only pressure driver using the application's real conversation owner."""

from copy import deepcopy
import threading
import time
from types import SimpleNamespace

from app.actions.store import ActionStore
from app.llm import LLM
from app.speech.vad import SileroVAD
from app.stt import STT
from app.tts import create_tts
from app.turn_control import TurnController


class ConversationLoad:
    """Resident clients/control for all backend conditions; optional backlogged input."""

    def __init__(self, cfg, camera, perception, events, phase, workload):
        self.cfg, self.events, self.phase, self.workload = cfg, events, phase, workload
        self.stop = threading.Event()
        self.thread = None
        self.completed = 0
        self.errors = []
        self.busy = False
        self.control = self.actions = None
        self.models = []
        self.llm = LLM(
            **{k: v for k, v in vars(cfg.llm).items() if k != "system_prompt_no_rag"}
        )
        self.stt = STT(**vars(cfg.stt))
        self.tts = create_tts(
            **{
                k: v
                for k, v in vars(cfg.tts).items()
                if k not in {"first_chunk_chars", "max_chunk_chars"}
            }
        )
        try:
            for model in (self.llm, self.stt, self.tts):
                self.models.append(model)
                if not model.load():
                    raise RuntimeError(model.last_error)
            self.llm.require_tools()
            self.actions = ActionStore(cfg.actions)
            self.control = TurnController(
                cfg,
                self.llm,
                SimpleNamespace(send=lambda *a, **k: None, ptt_active=False),
                camera,
                self.stt,
                self.tts,
                SileroVAD(),
                perception,
                self.actions,
            )
            self.control.start()
        except BaseException:
            self.close()
            raise

    def start(self, mode):
        if mode not in {"audio", "vision"} or self.thread is not None:
            raise ValueError("Invalid pressure mode or duplicate start")
        self.thread = threading.Thread(
            target=self._loop, args=(mode,), name="conversation-pressure", daemon=True
        )
        self.thread.start()

    def _loop(self, mode):
        try:
            from project_paths import ROOT

            wav = (ROOT / self.workload["audio_input"]["path"]).read_bytes()
            index = 0
            while not self.stop.is_set():
                index += 1
                self.control.clear_history(self.control.session_state()["id"])
                tick = time.monotonic()
                self.busy = True
                submitted = (
                    self.control.submit_audio(wav, use_image=False)
                    if mode == "audio"
                    else self.control.submit(
                        self.workload["vision_prompt"], use_image=True
                    )
                )
                deadline = (
                    time.monotonic()
                    + self.cfg.llm.timeout
                    + self.cfg.stt.timeout
                    + self.cfg.tts.timeout
                    + 30
                )
                cancel_sent = False
                while True:
                    state = self.control.state()
                    result = state.get("last_result") or {}
                    if (
                        state["phase"] == "idle"
                        and result.get("turn_id") == submitted["turn_id"]
                    ):
                        break
                    if self.stop.is_set() and not cancel_sent:
                        self.control.cancel(
                            reason="benchmark_window_end", turn_id=submitted["turn_id"]
                        )
                        cancel_sent = True
                    if time.monotonic() > deadline:
                        raise TimeoutError("Background conversation did not release")
                    time.sleep(0.025)
                ended = time.monotonic()
                self.busy = False
                complete = result["status"] == "complete"
                if complete:
                    if not result.get("audio_url") or not result.get("speech", {}).get(
                        "receipt"
                    ):
                        raise RuntimeError("Conversation did not publish TTS output")
                    if mode == "audio" and not result.get("asr"):
                        raise RuntimeError("Speech input did not execute ASR")
                    if mode == "vision" and not result.get("frame"):
                        raise RuntimeError(
                            "Visual conversation did not use a camera frame"
                        )
                    self.completed += 1
                elif not (self.stop.is_set() and result["status"] == "cancelled"):
                    raise RuntimeError("Conversation failed: " + repr(result))
                self.events.write(
                    kind="background_turn",
                    phase=self.phase(),
                    mode=mode,
                    index=index,
                    started_monotonic=tick,
                    completed_monotonic=ended,
                    window_end_cancel=self.stop.is_set(),
                    result=deepcopy(result),
                    trace=state.get("turn_trace"),
                )
        except Exception as exc:
            self.errors.append(repr(exc))
        finally:
            self.busy = False

    def close(self):
        self.stop.set()
        if self.control:
            self.control.cancel(reason="benchmark_window_end")
        if self.thread:
            self.thread.join(35)
            if self.thread.is_alive():
                raise RuntimeError("Pressure worker still owns a request")
            self.thread = None
        errors = []
        for close in (
            ([self.control.close] if self.control else [])
            + ([self.actions.close] if self.actions else [])
            + [model.unload for model in reversed(self.models)]
        ):
            try:
                close()
            except Exception as exc:
                errors.append(exc)
        self.control = self.actions = None
        self.models.clear()
        if errors:
            raise ExceptionGroup("Conversation benchmark cleanup failed", errors)


PROMPT = "用中文分点介绍如何分析程序的性能，先测量再定位瓶颈，回答尽量具体。"


class BackgroundLLM:
    def __init__(self, cfg, events, phase):
        from app.llm import LLM

        self.model = LLM(
            **{k: v for k, v in vars(cfg.llm).items() if k != "system_prompt_no_rag"}
        )
        self.events, self.phase = events, phase
        self.cancel = threading.Event()
        self.thread = None
        self.completed = 0
        self.errors = []
        self.busy = False

    def start(self):
        if not self.model.load():
            raise RuntimeError(self.model.last_error)
        self.thread = threading.Thread(
            target=self._loop, name="baseline-4b", daemon=True
        )
        self.thread.start()

    def _loop(self):
        index = 0
        while not self.cancel.is_set():
            index += 1
            start = time.monotonic()
            first = None
            chars = 0
            metadata = {}
            self.busy = True
            text = PROMPT + f"这是第{index}个请求。"
            error = None
            try:
                for chunk, meta in self.model.generate_stream(
                    text, cancel_event=self.cancel
                ):
                    metadata.update(meta)
                    if chunk:
                        first = first or time.monotonic()
                        chars += len(chunk)
                if not metadata.get("done") and not self.cancel.is_set():
                    raise RuntimeError("Background generation did not complete")
                if metadata.get("done"):
                    self.completed += 1
            except Exception as exc:
                error = repr(exc)
                if not self.cancel.is_set():
                    self.errors.append(error)
            finally:
                self.busy = False
                self.events.write(
                    kind="background_llm",
                    phase=self.phase(),
                    request=index,
                    prompt=text,
                    started_monotonic=start,
                    completed_monotonic=time.monotonic(),
                    first_content_seconds=first - start if first else None,
                    chars=chars,
                    metadata=metadata,
                    window_end_cancel=self.cancel.is_set(),
                    error=error,
                )
            if self.errors:
                return
            deadline = time.monotonic() + 30
            while not self.cancel.is_set() and not self.model.idle():
                if time.monotonic() > deadline:
                    self.errors.append("Background native slot did not release")
                    return
                self.cancel.wait(0.02)

    def close(self):
        self.cancel.set()
        self.model.cancel(self.cancel)
        if self.thread:
            self.thread.join(10)
            if self.thread.is_alive():
                raise RuntimeError("Background model client did not release")
        deadline = time.monotonic() + 30
        while not self.model.idle():
            if time.monotonic() > deadline:
                raise RuntimeError("Background native model did not release")
            time.sleep(0.02)
        self.model.unload()

