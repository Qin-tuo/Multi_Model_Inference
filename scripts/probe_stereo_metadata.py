#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read advertised stereo/RGB profiles and calibration without starting streams."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def intrinsics(profile):
    value = profile.as_video_stream_profile().get_intrinsics()
    return {
        "width": value.width,
        "height": value.height,
        "fx": value.fx,
        "fy": value.fy,
        "ppx": value.ppx,
        "ppy": value.ppy,
        "distortion_model": str(value.model),
        "coefficients": list(value.coeffs),
    }


def inspect(width, height, fps, serial=None):
    import pyrealsense2 as rs

    context = rs.context({"dds": {"enabled": False}})
    devices = list(context.query_devices())
    if serial is not None:
        devices = [
            d for d in devices if d.get_info(rs.camera_info.serial_number) == serial
        ]
    if len(devices) != 1:
        raise RuntimeError("Select exactly one connected camera using --serial")
    device = devices[0]
    profiles = [
        p
        for s in device.query_sensors()
        for p in s.get_stream_profiles()
        if p.is_video_stream_profile()
    ]
    requests = {
        "left_ir": (rs.stream.infrared, 1, rs.format.y8),
        "right_ir": (rs.stream.infrared, 2, rs.format.y8),
        "color": (rs.stream.color, 0, rs.format.rgb8),
        "sdk_depth_reference": (rs.stream.depth, 0, rs.format.z16),
    }
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Advertised profiles and calibration only; no streaming, synchronization or neural-depth quality qualification",
        "device": device.get_info(rs.camera_info.name),
        "firmware": device.get_info(rs.camera_info.firmware_version),
        "serial_sha256": hashlib.sha256(
            device.get_info(rs.camera_info.serial_number).encode()
        ).hexdigest(),
        "usb_type": device.get_info(rs.camera_info.usb_type_descriptor),
        "sdk_version": rs.__version__,
        "requested": {"width": width, "height": height, "fps": fps},
        "profiles": {},
        "extrinsics": {},
        "depth_units_m": device.first_depth_sensor().get_depth_scale(),
    }
    selected = {}
    for role, (stream, index, format_) in requests.items():
        matches = [
            p
            for p in profiles
            if p.stream_type() == stream
            and p.stream_index() == index
            and p.format() == format_
            and p.fps() == fps
            and p.as_video_stream_profile().width() == width
            and p.as_video_stream_profile().height() == height
        ]
        if not matches:
            result["profiles"][role] = {"available": False}
            continue
        profile = matches[0]
        selected[role] = profile
        result["profiles"][role] = {
            "available": True,
            "format": str(profile.format()),
            "fps": profile.fps(),
            "stream_index": profile.stream_index(),
            "intrinsics": intrinsics(profile),
        }
    for target in ("right_ir", "color", "sdk_depth_reference"):
        if "left_ir" in selected and target in selected:
            value = selected["left_ir"].get_extrinsics_to(selected[target])
            result["extrinsics"]["left_ir_to_" + target] = {
                "rotation": list(value.rotation),
                "rotation_layout": "SDK column-major 3x3",
                "translation_m": list(value.translation),
            }
    if "left_ir_to_right_ir" in result["extrinsics"]:
        translation = result["extrinsics"]["left_ir_to_right_ir"]["translation_m"]
        result["stereo_baseline_m"] = math.sqrt(sum(x * x for x in translation))
    result["all_requested_profiles_advertised"] = all(
        p["available"] for p in result["profiles"].values()
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--serial")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.width, args.height, args.fps, args.serial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "device",
                    "usb_type",
                    "sdk_version",
                    "all_requested_profiles_advertised",
                    "stereo_baseline_m",
                )
                if k in result
            }
        )
    )
    raise SystemExit(not result["all_requested_profiles_advertised"])


if __name__ == "__main__":
    main()
