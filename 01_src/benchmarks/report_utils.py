#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize frozen V1/V2 evidence; never rerun inference or omit failures."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_src"))
from benchmarks.baseline_support import distribution  # noqa: E402
from deploy import runtime  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def artifacts(folder):
    return [
        {
            "file": str(p.relative_to(folder)),
            "bytes": p.stat().st_size,
            "sha256": runtime.sha256(p),
        }
        for p in sorted(folder.rglob("*"))
        if p.is_file() and "source" not in p.relative_to(folder).parts
    ]


def resource_summary(path, phase):
    rows = [r for r in csv.DictReader(path.open()) if phase(r["phase"])]
    if not rows:
        return {
            "sample_count": 0,
            "error": "No resource observations in requested phase",
        }
    result = {
        "sample_count": len(rows),
        "minimum_mem_available_gib": min(float(r["mem_available"]) for r in rows)
        / 2**30,
        "maximum_swap_used_gib": max(float(r["swap_used"]) for r in rows) / 2**30,
        "host_cpu_percent": distribution([float(r["cpu_percent"]) for r in rows]),
        "gpu_load_snapshot_percent": distribution(
            [float(r["gpu_load_percent"]) for r in rows if r["gpu_load_percent"]]
        ),
    }
    for name in rows[0]:
        if name.endswith(("_pss", "_rss", "_fds", "_threads")):
            values = [float(r[name]) for r in rows]
            result[name] = {
                "first": values[0],
                "last": values[-1],
                "sampled_max": max(values),
            }
        if name.endswith("_cpu_seconds") and len(rows) > 1:
            result[name.replace("_seconds", "_percent_one_core")] = (
                (float(rows[-1][name]) - float(rows[0][name]))
                / (float(rows[-1]["monotonic"]) - float(rows[0]["monotonic"]))
                * 100
            )
    return result

