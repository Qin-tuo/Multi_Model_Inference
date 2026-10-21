# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest

from benchmarks import observe_live_service as live


def state(count=10, elapsed=20, *, ready=True, phase="idle", epoch="camera", session="app"):
    return {
        "phase": phase,
        "camera": {"source_epoch": epoch, "frame_id": count * 2,
                   "reconnects": 0, "rgbd_rejected_groups": 0},
        "perception": {
            "ready": ready, "model_set": "B2-standard-2d", "processed": count,
            "failures": 0, "skipped_capture_frames": count,
            "frame": {"source_epoch": epoch, "frame_id": count * 2,
                      "received_monotonic": count - .2},
            "completed_monotonic": count - .1,
            "elapsed_ms": elapsed,
            "stages": {"detection_segmentation_ms": elapsed},
            "task_counts": {"pose_people_processed": 1},
            "persons": [{"private_label": "must not be saved"}],
        },
        "diagnostics": {"session": session, "sequence": count,
                        "sampled_monotonic": count, "resources": {"module_input_w": 15},
                        "turn": {"last": {}}},
        "last_result": {"text": "private conversation"},
    }


def row(count=10, elapsed=20, **kwargs):
    return live.observe(state(count, elapsed, **kwargs), count, 1)


def test_deduplicate_latest_frame_and_exclude_anchor_from_percentiles():
    observation = live.Observation(row(10), "resident-idle")
    assert observation.add(row(10, 999)) is None
    assert observation.add(row(11, 20)) is not None
    assert observation.add(row(11, 999)) is None
    assert observation.add(row(12, 40)) is not None
    report = observation.report()
    assert report["frame_coverage_fraction"] == 1
    assert report["status"] == "OBSERVED_COUNTER_WINDOW_COMPLETE"
    metric = report["frame_metrics_ms"]["detection_segmentation"]
    assert (metric["count"], metric["p50"], metric["p95"]) == (2, 20, 40)
    assert report["task_counts_total_observed"] == {"pose_people_processed": 2}
    assert report["observed_frames_with_person"] == 2
    assert "private" not in json.dumps(observation.samples)


def test_missed_frames_are_reported_not_silently_treated_as_complete():
    observation = live.Observation(row(10), "resident-idle")
    observation.add(row(11))
    observation.add(row(13))
    report = observation.report()
    assert report["counter_changes"]["processed"] == 3
    assert report["unique_observed_frames"] == 2
    assert report["unobserved_completed_frames"] == 1
    assert report["frame_coverage_fraction"] == pytest.approx(2 / 3)
    assert "frame_coverage_incomplete" in report["issues"]
    assert report["status"] == "OBSERVATION_WITH_LIMITATIONS"


def test_restart_and_missing_counter_do_not_produce_a_valid_baseline():
    observation = live.Observation(row(10), "resident-idle")
    changed = row(1, epoch="new camera", session="new app")
    changed["processed"] = None
    observation.add(changed)
    observation.add(row(2, epoch="new camera", session="new app"))
    report = observation.report()
    assert report["counter_changes"]["processed"] is None
    assert report["frame_coverage_fraction"] is None
    assert {"epoch_changed", "session_changed",
            "processed_counter_invalid_or_reset"} <= set(report["issues"])


def test_unavailable_results_and_background_activity_stay_visible():
    observation = live.Observation(row(10), "resident-idle")
    observation.add(row(11, ready=False, phase="generating"))
    observation.add(row(12))
    observation.http_errors.append({"error": "timeout"})
    report = observation.report()
    assert report["unavailable_result_samples"] == 1
    assert report["latest_result_age_available_samples_ms"]["count"] == 1
    assert report["frame_metrics_ms"]["processing"]["count"] == 1
    assert {"non_idle_phase_observed", "http_observation_errors",
            "frame_coverage_incomplete"} <= set(report["issues"])


def test_cli_observes_get_only_and_writes_readable_report(tmp_path, monkeypatch):
    now = [10.0]
    monkeypatch.setattr(live, "time", SimpleNamespace(
        monotonic=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    ))
    requests = []

    class Connection:
        def __init__(self, host, port, timeout):
            assert host == "127.0.0.1"
        def request(self, method, path):
            requests.append((method, path))
            assert (method, path) == ("GET", "/api/state")
        def getresponse(self):
            current = state(len(requests) + 9)
            current["perception"]["frame"]["received_monotonic"] = now[0] - .2
            current["perception"]["completed_monotonic"] = now[0] - .1
            body = json.dumps(current).encode()
            return SimpleNamespace(status=200, read=lambda limit: body)
        def close(self):
            pass

    monkeypatch.setattr(live.http.client, "HTTPConnection", Connection)
    output = tmp_path / "live"
    monkeypatch.setattr("sys.argv", [
        "observe_live_service.py", "--seconds", "1", "--hz", "2",
        "--note", "one person, fixed scene", "--output", str(output),
    ])
    live.main()
    report = json.loads((output / "report.json").read_text())
    assert report["counter_changes"]["processed"] == 2
    assert report["frame_coverage_fraction"] == 1
    assert len((output / "frames.jsonl").read_text().splitlines()) == 2
    assert "主机接收" in (output / "report.md").read_text()
    assert "private conversation" not in (output / "environment.json").read_text()


def test_wrong_model_set_or_busy_anchor_is_rejected():
    with pytest.raises(ValueError, match="expected model"):
        live.validate_first(row(), "another-model-set", "resident-idle")
    with pytest.raises(ValueError, match="conversation"):
        live.validate_first(row(phase="generating"), "B2-standard-2d", "resident-idle")
