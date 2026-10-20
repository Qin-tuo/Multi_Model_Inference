# SPDX-License-Identifier: Apache-2.0
"""Notification callbacks must be cancellable, idempotent and lifetime-bounded."""

import gc
import threading
from types import SimpleNamespace
import weakref

import pytest

from app.request_io import TurnCancelled
from app.turn_execution import TurnExecution


def execution():
    calls = []
    owner = SimpleNamespace(
        _lock=threading.RLock(),
        _stop=threading.Event(),
        actions=SimpleNamespace(begin_delivery=lambda notice, mode: calls.append((notice, mode))),
        config=SimpleNamespace(audio=SimpleNamespace(output_mode="file")),
        tts=object(),
    )
    event = threading.Event()
    request = {
        "turn_id": "turn",
        "request_id": "request",
        "use_image": False,
        "trace": SimpleNamespace(mark=lambda name: None),
        "cancel_event": event,
        "notice": {"id": "notice"},
    }
    return TurnExecution(owner, request), calls, event


def test_notification_output_begins_once_across_multiple_chunks():
    turn, calls, _ = execution()
    turn.begin_notice_output()
    turn.begin_notice_output()
    assert calls == [({"id": "notice"}, "file")]


def test_cancelled_notification_cannot_start_delivery():
    turn, calls, event = execution()
    event.set()
    with pytest.raises(TurnCancelled):
        turn.begin_notice_output()
    assert calls == []


def test_output_callback_does_not_retain_the_whole_turn():
    turn, calls, _ = execution()
    reference = weakref.ref(turn)
    callback = turn.begin_notice_output
    del turn
    gc.collect()
    assert reference() is None
    callback()
    assert len(calls) == 1
