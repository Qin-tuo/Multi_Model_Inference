#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify the application's bounded Camera with native D435i depth and SDK align."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01_src"))
from app.camera import Camera
from deploy.operations import exclusive_build
from deploy import runtime, services


def probe(output):
    camera = Camera(backend="realsense", capture_fps=10)
    report = {
        "created_at": runtime.stamp(),
        "status": "RUNNING",
        "scope": "Application RGBD capture, SDK alignment, provenance and lifecycle; not absolute distance accuracy or pipeline performance",
        "script_sha256": runtime.sha256(Path(__file__)),
        "frames": [],
    }
    try:
        if not camera.start():
            raise RuntimeError(camera.last_error)
        deadline = time.monotonic() + 15
        previous = None
        held = None
        while len(report["frames"]) < 30 and time.monotonic() < deadline:
            frame = camera.snapshot()
            if frame is None or frame.frame_id == previous:
                time.sleep(0.01)
                continue
            previous = frame.frame_id
            if frame.depth_m is None or frame.depth_valid is None:
                raise ValueError("Missing aligned native depth")
            if (
                frame.depth_m.shape != frame.bgr.shape[:2]
                or frame.depth_m.dtype != np.float32
            ):
                raise ValueError("RGB/depth dimensions or type mismatch")
            if not np.isnan(frame.depth_m[~frame.depth_valid]).all():
                raise ValueError("Invalid depth was not preserved as unknown")
            if not frame.sensor["correspondence"]["rgb_depth_time_matched"]:
                raise ValueError("RGB/depth timestamp mismatch")
            if frame.depth_m.flags.writeable or frame.bgr.flags.writeable:
                raise ValueError("Published image storage is mutable")
            valid_values = frame.depth_m[frame.depth_valid]
            report["frames"].append(
                {
                    "frame": frame.metadata(),
                    "valid_depth_pixels": int(frame.depth_valid.sum()),
                    "depth_m_p05_p50_p95": np.quantile(
                        valid_values, [0.05, 0.5, 0.95]
                    ).tolist()
                    if valid_values.size
                    else [],
                }
            )
            if held is None:
                held = frame
                np.savez(
                    output / "first-frame.npz",
                    bgr=frame.bgr,
                    depth_m=frame.depth_m,
                    valid=frame.depth_valid,
                )
        if len(report["frames"]) != 30:
            raise RuntimeError("Did not observe 30 unique fresh application frames")
        report["before_close"] = camera.status()
        camera.close()
        report["after_close"] = camera.status()
        report["held_frame_invalidated"] = held.invalidated.is_set()
        if (
            camera.snapshot() is not None
            or not held.invalidated.is_set()
            or camera.status()["capture_worker_alive"]
        ):
            raise RuntimeError(
                "Camera close did not invalidate and release the RGBD source"
            )
        report["status"] = "APP_RGBD_CAPTURE_PASS_LIMITED_WINDOW"
    except Exception as exc:
        report.update(status="FAILED", error=repr(exc))
        raise
    finally:
        camera.close()
        runtime.write_json(output / "report.json", report)
    print(report["status"], len(report["frames"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = runtime.config()
    if services.profile_path(cfg).exists():
        cfg = runtime.config(services.profile_path(cfg))
    with exclusive_build(cfg, output):
        probe(output)


if __name__ == "__main__":
    main()
