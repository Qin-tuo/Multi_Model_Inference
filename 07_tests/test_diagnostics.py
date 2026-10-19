# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import Config
from app.diagnostics import BoardDiagnostics, TurnTrace
from app.web import Broadcaster, create_app
from app.web_limits import ClientBuffer, encode


def test_trace_uses_board_clock_retains_overlap_and_marks_once():
    now = [10.0]
    trace = TurnTrace("turn", 10.0, clock=lambda: now[0])
    trace.mark("dequeued")
    with trace.span("llm"):
        now[0] = 11.0
        trace.mark("first_content")
        with trace.span("tts"):
            now[0] = 12.0
            trace.mark("first_content")
            active = trace.snapshot()
            assert len(active["spans"]) == 2
    trace.finish("complete")
    now[0] = 20.0
    result = trace.snapshot()
    assert result["elapsed_ms"] == 2000
    assert result["marks_ms"]["first_content"] == 1000
    assert result["stages"]["llm"]["duration_ms"] == 2000
    assert result["stages"]["tts"]["duration_ms"] == 1000


def test_trace_interruption_and_bounded_retention():
    trace = TurnTrace("turn", 0, clock=lambda: 1.0)
    with pytest.raises(RuntimeError):
        with trace.span("asr"):
            raise RuntimeError("cancelled")
    assert trace.snapshot()["spans"][0]["outcome"] == "interrupted"
    for _ in range(80):
        with trace.span("tts"):
            pass
    result = trace.snapshot()
    assert len(result["spans"]) == 64
    assert result["omitted_spans"] == 17
    assert result["stages"]["tts"]["count"] == 80
    result["stages"]["tts"]["count"] = 999
    assert trace.snapshot()["stages"]["tts"]["count"] == 80


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    monkeypatch.setattr("app.diagnostics.psutil.virtual_memory", lambda: SimpleNamespace(total=16*2**30, available=6*2**30))
    monkeypatch.setattr("app.diagnostics.psutil.swap_memory", lambda: SimpleNamespace(used=2**30))
    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    write("class/thermal/thermal_zone1/type", "gpu-thermal")
    write("class/thermal/thermal_zone1/temp", "42000")
    write("class/hwmon/hwmon3/in1_label", "VDD_IN")
    write("class/hwmon/hwmon3/in1_input", "12000")
    write("class/hwmon/hwmon3/curr1_input", "1500")
    return BoardDiagnostics(Config.load(), sysfs=tmp_path)


def state(epoch="a", captured=100, processed=50, received=9.9):
    return {"camera": {"ready": True, "source_epoch": epoch, "frame_id": captured},
            "perception": {"ready": True, "processed": processed, "elapsed_ms": 120,
                "frame": {"received_monotonic": received, "source_epoch": epoch, "frame_id": captured},
                "stages": {"body_ms": 60}, "body_stages_ms": {"postprocess_ms": 42}}}


STATS = SimpleNamespace(cpu_percent=30, gpu_percent=None)


def test_observed_rates_differ_from_targets_and_epoch_resets(monitor):
    first = monitor.sample(state(), STATS, now=10)
    assert first["vision"]["capture_fps"] is None
    second = monitor.sample(state(captured=114, processed=56, received=11.8), STATS, now=12)
    assert second["vision"]["capture_fps"] == 7
    assert second["vision"]["perception_fps"] == 3
    assert second["vision"]["capture_target_fps"] == 10
    assert second["vision"]["frame_age_ms"] == pytest.approx(200)
    reset = monitor.sample(state(epoch="b", captured=1, processed=57, received=13.9), STATS, now=14)
    assert reset["vision"]["capture_fps"] is None
    assert reset["vision"]["perception_fps"] is None


def test_stale_results_and_missing_sensors_never_become_zero_latency(monitor):
    result = monitor.sample(state(received=5), STATS, now=10)
    assert not result["vision"]["ready"]
    assert result["vision"]["processing_ms"] is None
    assert result["vision"]["frame_age_ms"] is None
    assert result["vision"]["stages_ms"] == {}
    resources = result["resources"]
    assert resources["gpu_percent"] is None
    assert resources["gpu_mhz"] is None
    assert resources["temperatures_c"]["gpu-thermal"] == 42
    assert resources["module_input_w"] == 18
    assert resources["ram_available_gib"] == 6
    assert resources["ram_used_gib"] == 10
    snapshot = monitor.snapshot()
    snapshot["vision"]["ready"] = True
    assert not monitor.snapshot()["vision"]["ready"]


def test_diagnostics_messages_coalesce_without_evicting_answer():
    buffer = ClientBuffer(2, 4096)
    buffer.put(encode({"type": "token", "text": "answer"}, 4096))
    for seq in range(100):
        buffer.put(encode({"type": "diagnostics", "sequence": seq}, 4096))
    assert not buffer.failed
    assert len(buffer.items) == 2
    assert buffer.items[0].kind == "token"
    assert b'"sequence":99' in buffer.items[1].wire


def test_debug_routes_share_index_and_only_serve_allowed_assets():
    with TestClient(create_app(Broadcaster(), state=lambda: {"diagnostics": {"ready": True, "source": "board"}})) as client:
        assert client.get("/debug").content == client.get("/").content
        assert client.get("/api/debug").json() == {"ready": True, "source": "board"}
        assert "body" in client.get("/api/debug/baseline").json()["modules"]
        assert client.get("/static/debug.js").status_code == 200
        assert client.get("/static/debug.css").status_code == 200
        assert client.get("/static/unknown.txt").status_code == 404
