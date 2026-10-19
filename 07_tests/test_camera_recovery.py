# SPDX-License-Identifier: Apache-2.0
"""Capture owner must retry, invalidate old work and stop without racing read."""

import asyncio
import threading
import time

import numpy as np

from app.camera import Camera
from app.web import Broadcaster
from app.web_limits import ClientBuffer


def until(test, seconds=2):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if test():
            return
        time.sleep(0.01)
    assert test()


def test_reconnect_uses_same_owner_and_invalidates_pending_frame(monkeypatch):
    online = threading.Event()
    online.set()
    owners = []
    captures = []

    class Capture:
        def __init__(self, *args):
            self.released = False
            self.reading = False
            captures.append(self)
            owners.append(threading.get_ident())

        def isOpened(self):
            return online.is_set() and not self.released

        def set(self, *args):
            return True

        def get(self, *args):
            return 32

        def read(self):
            self.reading = True
            try:
                return (
                    (True, np.zeros((32, 32, 3), dtype=np.uint8))
                    if online.is_set()
                    else (False, None)
                )
            finally:
                self.reading = False

        def release(self):
            assert not self.reading
            self.released = True
            owners.append(threading.get_ident())

    monkeypatch.setattr("app.camera.cv2.VideoCapture", Capture)
    camera = Camera(capture_fps=100, reconnect_seconds=0.1)
    loop = asyncio.new_event_loop()
    broadcaster = Broadcaster()
    broadcaster.set_loop(loop)
    viewer = ClientBuffer(16, 10000)
    broadcaster.register(viewer)
    try:
        assert camera.start()
        before = camera.snapshot()
        owner = camera._thread
        broadcaster.send(
            {"type": "frame", "frame": before.metadata()},
            cancel_event=before.invalidated,
        )
        online.clear()
        until(lambda: camera.status()["state"] == "retrying")
        assert (
            before.invalidated.is_set()
            and camera.snapshot() is None
            and camera.buffer_bytes == 0
        )
        assert camera.source_epoch != before.source_epoch
        loop.run_until_complete(asyncio.sleep(0))
        assert not viewer.items
        online.set()
        until(camera.health_check)
        after = camera.snapshot()
        assert (
            after.source_epoch != before.source_epoch and not after.invalidated.is_set()
        )
        assert camera._thread is owner and camera.status()["reconnects"] == 1
        assert camera.last_error is None
    finally:
        camera.close()
        broadcaster.set_loop(None)
        loop.close()
    assert len(set(owners)) == 1 and all(c.released for c in captures)
    assert after.invalidated.is_set() and not camera.status()["capture_worker_alive"]


def test_stop_during_read_never_publishes_late_frame_or_releases_concurrently(
    monkeypatch,
):
    entered = threading.Event()
    unblock = threading.Event()
    capture = None

    class Capture:
        def __init__(self, *args):
            nonlocal capture
            capture = self
            self.first = True
            self.reading = False
            self.released = False

        def isOpened(self):
            return not self.released

        def set(self, *args):
            return True

        def get(self, *args):
            return 32

        def read(self):
            self.reading = True
            try:
                if not self.first:
                    entered.set()
                    unblock.wait(2)
                self.first = False
                return True, np.zeros((32, 32, 3), dtype=np.uint8)
            finally:
                self.reading = False

        def release(self):
            assert not self.reading
            self.released = True

    monkeypatch.setattr("app.camera.cv2.VideoCapture", Capture)
    camera = Camera(capture_fps=20)
    assert camera.start()
    before = camera.snapshot()
    assert entered.wait(1)
    errors = []

    def close():
        try:
            camera.close()
        except Exception as exc:
            errors.append(exc)

    closer = threading.Thread(target=close)
    closer.start()
    try:
        until(camera._stop.is_set)
        assert (
            camera.snapshot() is None
            and before.invalidated.is_set()
            and not capture.released
        )
    finally:
        unblock.set()
        closer.join(3)
    assert not closer.is_alive() and not errors and capture.released
    assert camera.buffer_count == 0 and camera.snapshot() is None


def test_speech_window_cannot_return_old_source_after_concurrent_reconnect(monkeypatch):
    camera = Camera()
    now = time.monotonic()
    jpeg = b"old compressed frame"
    camera._ring.append((now, jpeg))

    def recovered():
        camera._invalidate("streaming")
        return True

    monkeypatch.setattr(camera, "health_check", recovered)
    assert camera.get_speech_frames(now - 0.1, now + 0.1, max_frames=1) == []
