#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Optional raw-IR diagnostics under the project's service/build lease.

Use the main application Python to orchestrate service restoration. The worker
uses the isolated pinned SDK environment. SDK depth is explicitly reference
data for these IR experiments. The B1 application uses native SDK RGBD;
scripts/probe_rgbd_camera.py validates its production capture path.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_src"))


def capture(output, count):
    import numpy as np
    from app.realsense_source import RealSenseSource

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "scope": "Short real stereo capture and correspondence, not B1 performance or neural-depth quality acceptance",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(
            (ROOT / "01_src/app/realsense_source.py").read_bytes()
        ).hexdigest(),
        "warmup_frames": 30,
        "requested_groups": count,
        "saved": [],
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    rows = []
    try:
        with RealSenseSource(with_ir=True) as source:
            report["calibration"] = source.calibration
            report["max_rgb_skew_ms"] = source.max_rgb_skew_ms
            for _ in range(report["warmup_frames"]):
                source.read()
            start = time.monotonic()
            with (output / "frames.jsonl").open("x") as log:
                for index in range(count):
                    frame = source.read()
                    row = frame.metadata()
                    rows.append(row)
                    if index in {0, count // 2, count - 1}:
                        path = output / f"group-{frame.group_id}.npz"
                        np.savez(path, **frame.arrays)
                        report["saved"].append(
                            {
                                "file": path.name,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "frame": row,
                            }
                        )
                    log.write(json.dumps(row) + "\n")
            report["elapsed_seconds"] = time.monotonic() - start
        report["groups"] = len(rows)
        report["valid_stereo_groups"] = sum(
            r["correspondence"]["stereo_valid"] for r in rows
        )
        report["rgb_time_matched_groups"] = sum(
            r["correspondence"]["rgb_time_matched"] for r in rows
        )
        report["frame_counter_gaps"] = {}
        report["repeated_frames"] = {}
        for name in ("left_ir", "right_ir", "color", "sdk_depth"):
            counters = [
                r["streams"][name]["frame_number"] for r in rows if name in r["streams"]
            ]
            report["frame_counter_gaps"][name] = sum(
                max(0, b - a - 1) for a, b in zip(counters, counters[1:])
            )
            report["repeated_frames"][name] = sum(
                b == a for a, b in zip(counters, counters[1:])
            )
        report["status"] = (
            "CAPTURE_PASS_LIMITED_WINDOW"
            if report["valid_stereo_groups"] == count
            and report["rgb_time_matched_groups"] == count
            else "CAPTURE_CORRESPONDENCE_INCOMPLETE"
        )
    except Exception as exc:
        report.update(status="CAPTURE_FAILED", error=repr(exc))
        raise
    finally:
        save()
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "groups",
                    "valid_stereo_groups",
                    "rgb_time_matched_groups",
                )
            }
        )
    )
    return report["status"] == "CAPTURE_PASS_LIMITED_WINDOW"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=60)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 30 <= args.groups <= 120:
        parser.error("Use 30 to 120 frame groups for this bounded capture probe")
    output = args.output.resolve()
    if args.worker:
        return int(not capture(output, args.groups))
    from build_vision_b1 import exclusive_build
    from deploy import runtime, services

    output.mkdir(parents=True, exist_ok=False)
    cfg = runtime.config()
    if services.profile_path(cfg).exists():
        cfg = runtime.config(services.profile_path(cfg))
    with exclusive_build(cfg, output):
        result = subprocess.run(
            [
                str(ROOT / "05_runtime/builds/vision-tools/bin/python"),
                str(Path(__file__).resolve()),
                "--worker",
                "--output",
                str(output),
                "--groups",
                str(args.groups),
            ],
            timeout=60,
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
