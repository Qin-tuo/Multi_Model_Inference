"""Measurement wrappers must preserve call results, ownership and failures."""

import pytest

from benchmarks.baseline_support import CallTimes, NativeProxy, distribution, matching


def test_native_proxy_preserves_configuration_and_face_result():
    class Native:
        def setInputSize(self, value):
            self.size = value

        def detect(self, value):
            return self.size, value

    class Detector:
        def __init__(self):
            self._detector = Native()

    model = Detector()
    native = model._detector
    times = CallTimes()
    proxy = NativeProxy(native, times, {"setInputSize": "face.native_config", "detect": "face.native_detect"})
    proxy.setInputSize((640, 480))
    assert proxy.detect("pixels") == ((640, 480), "pixels")
    assert model._detector is native
    assert [c["label"] for c in times.calls] == [
        "face.native_config",
        "face.native_detect",
    ]


def test_no_samples_and_invalid_numeric_outputs_are_not_valid_measurements():
    assert distribution([])["p95"] is None
    assert not matching([float("nan")], [float("nan")])
    assert not matching({"hands": []}, {"hands": [1]})
    with pytest.raises(ValueError):
        distribution([float("inf")])


def test_replay_skips_missed_arrivals_instead_of_slowing_with_consumer(monkeypatch):
    import threading
    from benchmarks.replay import ReplayCapture
    from benchmarks import replay

    now = [10.0]
    monkeypatch.setattr(replay.time, "monotonic", lambda: now[0])

    class Recorder:
        def __init__(self):
            self.rows = []

        def write(self, **row):
            self.rows.append(row)

    events = Recorder()
    replay = ReplayCapture(
        [("a", "image-a"), ("b", "image-b")], threading.Event(), events, lambda: "test"
    )
    assert replay.read() == (True, "image-a")
    now[0] += 2.15  # Consumer is delayed; producer's 10Hz clock keeps advancing.
    assert replay.read() == (True, "image-b")
    assert replay.last == 21 and replay.missed == 20 and replay.delivered == 2
    assert events.rows[-1]["scheduled_monotonic"] == 12.1
