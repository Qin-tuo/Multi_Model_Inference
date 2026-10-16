# SPDX-License-Identifier: Apache-2.0
"""Voice reservation, cancellation and source lifetime; no acoustic qualification."""

from dataclasses import replace
from types import SimpleNamespace
import threading
import time
import uuid
import subprocess
import sys

import numpy as np
import pytest
from rich.console import Console

from app.config import Config
from app.pipeline import (
    AudioChunk,
    MicRecorder,
    SpeechOnset,
    SpeechSegment,
    SpeechSegmenter,
    vad_loop,
)
from app.turn_control import TurnController


def until(test, seconds=3):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if test():
            return
        time.sleep(0.005)
    assert test()


class Model:
    def __init__(self):
        self.entered = threading.Event()
        self.cancelled = threading.Event()
        self.released = threading.Event()
        self.calls = 0

    def generate_stream(self, *a, **kw):
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            self.cancelled.wait(2)
        yield "答复", {"done": True}

    def cancel(self, event):
        self.cancelled.set()

    def idle(self):
        return self.released.is_set()


class ASR:
    def __init__(self):
        self.calls = 0

    def transcribe(self, *a, **kw):
        self.calls += 1
        return {"text": "新指令"}

    def idle(self):
        return True

    def cancel(self, event):
        pass


def setup_control():
    model = Model()
    asr = ASR()
    events = []
    control = TurnController(
        Config.load(),
        model,
        SimpleNamespace(send=lambda m, **kw: events.append(m)),
        stt=asr,
    )
    control.start()
    return control, model, asr, events


def onset(event=None):
    now = time.monotonic()
    return SpeechOnset(uuid.uuid4().hex, now, now, 1, event or threading.Event())


def segment(start):
    now = time.monotonic()
    return SpeechSegment(
        np.ones(16000, dtype=np.float32) * 0.1,
        [],
        1,
        0.1,
        now - 1,
        now,
        input_epoch=start.input_epoch,
        utterance_id=start.utterance_id,
        input_invalidated=start.input_invalidated,
    )


def test_speech_onset_cancels_before_endpoint_and_asr_waits_for_native_release():
    control, model, asr, events = setup_control()
    try:
        old = control.submit("old")["turn_id"]
        assert model.entered.wait(1)
        start = onset()
        decision = control.begin_voice(start)
        assert decision["accepted"] and decision["interrupted_turn"] == old
        assert model.cancelled.is_set() and asr.calls == 0
        assert control.submit_segment(segment(start))["accepted"]
        until(lambda: control.state()["last_result"] is not None)
        assert control.state()["last_result"]["status"] == "cancelled"
        with pytest.raises(RuntimeError):
            control.submit("cannot jump voice reservation")
        assert (
            asr.calls == 0
            and control.state()["voice"]["pending"]["turn_id"] == start.utterance_id
        )
        # A stale cancel bound to the old turn cannot drop the later utterance.
        control.cancel(old)
        assert control.state()["voice"]["pending"]["turn_id"] == start.utterance_id
        model.released.set()
        until(
            lambda: control.state()["last_result"]["turn_id"] == start.utterance_id
            and control.state()["active_turn"] is None
        )
        assert asr.calls == 1 and control.state()["last_result"]["status"] == "complete"
        assert not any(e["type"] == "token" and e.get("turn_id") == old for e in events)
    finally:
        model.released.set()
        model.cancelled.set()
        control.close()


@pytest.mark.parametrize("reason", ["cancel", "source_change", "expiry"])
def test_pending_voice_can_be_discarded_without_model_or_tool_work(reason):
    control, model, asr, _ = setup_control()
    try:
        start = onset()
        assert control.begin_voice(start)["accepted"]
        if reason == "cancel":
            assert control.cancel(start.utterance_id)["accepted"]
        elif reason == "source_change":
            start.input_invalidated.set()
        else:
            # Advance the reservation's deadline without waiting 30 real seconds.
            with control._lock:
                control._voice["expires_monotonic"] = time.monotonic() - 1
        assert not control.submit_segment(segment(start))["accepted"]
        assert control.state()["voice"]["pending"] is None
        assert model.calls == asr.calls == 0
    finally:
        model.released.set()
        model.cancelled.set()
        control.close()


def test_latest_pending_utterance_replaces_only_unstarted_voice():
    control, model, asr, _ = setup_control()
    try:
        first = onset()
        second = onset()
        assert (
            control.begin_voice(first)["accepted"]
            and control.begin_voice(second)["accepted"]
        )
        assert not control.submit_segment(segment(first))["accepted"]
        assert control.state()["voice"]["pending"]["turn_id"] == second.utterance_id
        assert not control.cancel(first.utterance_id)["accepted"]
        model.released.set()
        model.cancelled.set()
        assert control.submit_segment(segment(second))["accepted"]
        until(
            lambda: control.state()["last_result"] is not None
            and control.state()["active_turn"] is None
        )
        assert (
            control.state()["last_result"]["turn_id"] == second.utterance_id
            and asr.calls == 1
        )
    finally:
        model.released.set()
        model.cancelled.set()
        control.close()


def test_old_speech_cannot_cancel_a_later_turn_and_palm_can_clear_pending_voice():
    control, model, asr, _ = setup_control()
    try:
        old_voice = onset()
        control.submit("newer turn")
        assert model.entered.wait(1)
        assert (
            not control.begin_voice(old_voice)["accepted"]
            and not model.cancelled.is_set()
        )
        fresh = onset()
        assert control.begin_voice(fresh)["accepted"]
        control.camera = SimpleNamespace(source_epoch="camera")
        now = time.monotonic()
        event = {
            "event_id": "stop",
            "source_epoch": "camera",
            "kind": "stop",
            "evidence_end_monotonic": now,
            "confirmed_monotonic": now,
            "expires_monotonic": now + 2,
        }
        assert control.handle_gesture(event)["accepted"]
        assert control.state()["voice"]["pending"] is None and asr.calls == 0
    finally:
        model.released.set()
        model.cancelled.set()
        control.close()


class VAD:
    def reset(self):
        pass

    def __call__(self, raw):
        return float(np.frombuffer(raw, dtype=np.int16).any())


def test_segmenter_confirms_once_and_does_not_execute_truncated_run_on_speech():
    cfg = replace(Config.load().vad, max_speech_secs=1, lookback_ms=0)
    engine = SpeechSegmenter(cfg, VAD())
    raw = np.full(512, 1000, dtype=np.int16).tobytes()
    quiet = b"\0" * 1024
    onsets = []
    segments = []
    t = 0.0
    for _ in range(100):
        t += 0.032
        value = engine.feed(raw, t)
        if engine.onset:
            onsets.append(engine.onset)
        if value:
            segments.append(value)
    assert (
        len(onsets) == len(segments) == 1 and segments[0].end_reason == "max_duration"
    )
    assert len(segments[0].audio) <= 16000 and engine.draining
    for _ in range(20):
        t += 0.032
        engine.feed(quiet, t)
    assert not engine.draining
    good = []
    for frame in [raw] * 8 + [quiet] * 20:
        t += 0.032
        value = engine.feed(frame, t)
        if value:
            good.append(value)
    assert (
        len(good) == 1
        and good[0].end_reason == "silence"
        and good[0].utterance_id != segments[0].utterance_id
    )


def test_short_valid_utterance_confirms_at_endpoint():
    engine = SpeechSegmenter(Config.load().vad, VAD())
    raw = np.full(512, 1000, dtype=np.int16).tobytes()
    for n in range(3):
        assert engine.feed(raw, (n + 1) * 0.032) is None and engine.onset is None
    result = None
    for n in range(3, 30):
        value = engine.feed(b"\0" * 1024, (n + 1) * 0.032)
        if value:
            result = value
            assert engine.onset.utterance_id == value.utterance_id
    assert result is not None and result.end_reason == "silence"


def test_half_duplex_requires_quiet_before_rearming():
    mic = MicRecorder(Console())
    busy = threading.Event()
    busy.set()
    starts = []
    segments = []
    errors = []

    def listen():
        try:
            for value in vad_loop(
                mic,
                Console(),
                Config.load().vad,
                VAD(),
                on_start=lambda s: starts.append(s) or {"accepted": True},
                capture_gate=lambda: "half_duplex" if busy.is_set() else None,
            ):
                segments.append(value)
        except Exception as exc:
            errors.append(exc)

    listener = threading.Thread(target=listen)
    listener.start()

    def feed(speech, count):
        for _ in range(count):
            with mic._input_lock:
                mic._sequence += 1
                raw = np.full(512, 1000 if speech else 0, dtype=np.int16).tobytes()
                mic.audio_q.put_nowait(
                    AudioChunk(
                        raw,
                        mic.input_epoch,
                        time.monotonic(),
                        mic._sequence,
                        mic.input_invalidated,
                    )
                )
            time.sleep(0.005)

    try:
        feed(True, 20)
        busy.clear()
        feed(True, 20)
        assert not starts and mic.suppressed_reason == "awaiting_silence"
        feed(False, 30)
        until(lambda: mic.suppressed_reason is None)
        feed(True, 12)
        feed(False, 20)
        until(lambda: len(segments) == 1)
        assert len(starts) == 1 and segments[0].utterance_id == starts[0].utterance_id
    finally:
        mic.stop()
        listener.join(2)
    assert not listener.is_alive() and not errors


def test_missing_pcm_is_a_visible_terminal_input_fault():
    mic = MicRecorder(Console())
    with pytest.raises(RuntimeError, match="stalled"):
        list(vad_loop(mic, Console(), Config.load().vad, VAD(), max_age_seconds=0.03))
    assert not mic.alive and mic.audio_q.empty()
    with pytest.raises(ValueError):
        mic.resume()
    mic.stop()


def test_failed_capture_attempts_close_owned_process_pipes(monkeypatch):
    actual = subprocess.Popen
    children = []

    def exited(*args, **kwargs):
        child = actual(
            [
                sys.executable,
                "-c",
                'import sys;sys.stderr.write("capture unavailable");sys.exit(1)',
            ],
            **kwargs,
        )
        child.wait(2)
        children.append(child)
        return child

    monkeypatch.setattr("app.pipeline.subprocess.Popen", exited)
    mic = MicRecorder(Console())
    try:
        assert not mic._start_capture()
        assert len(children) == 3 and all(
            c.stdout.closed and c.stderr.closed and c.poll() == 1 for c in children
        )
        assert mic.last_error == "capture unavailable"
    finally:
        mic.stop()
