# SPDX-License-Identifier: Apache-2.0
"""Reject wrong-frame and wrong-clock joins before they reach visual tasks."""

from app.realsense_source import correspondence


def streams():
    return {
        name: {
            "frame_number": 10,
            "timestamp_ms": 1000,
            "domain": "timestamp_domain.global_time",
        }
        for name in ("left_ir", "right_ir", "color")
    }


def test_local_sensor_clock_and_stale_rgb_cannot_be_joined():
    values = streams()
    assert correspondence(values, 33)["rgb_time_matched"]
    values["color"]["timestamp_ms"] += 100
    result = correspondence(values, 33)
    assert result["stereo_valid"] and not result["rgb_time_matched"]
    values = streams()
    for value in values.values():
        value["domain"] = "timestamp_domain.hardware_clock"
    result = correspondence(values, 33)
    assert result["stereo_valid"] and not result["rgb_time_matched"]
    assert result["rgb_skew_ms"] is None


def test_wrong_stereo_counter_missing_view_and_clock_change_are_rejected():
    for key, value in (
        ("frame_number", 11),
        ("domain", "timestamp_domain.hardware_clock"),
        ("timestamp_ms", 1001),
    ):
        values = streams()
        values["right_ir"][key] = value
        assert not correspondence(values, 33)["stereo_valid"]
    values = streams()
    del values["right_ir"]
    assert not correspondence(values, 33)["stereo_valid"]
