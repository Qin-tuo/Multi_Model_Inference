#!/usr/bin/env python3
"""Render the vision component report with system matplotlib; no inference."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ["YOLOv8n\nTensorRT", "YuNet\nCPU", "Palm + hands\nCPU"]
    modules = [
        next(
            f
            for f in report["modules"][name]["fixtures"]
            if f["input"] == "public-camera-shape.png"
        )["plain_ms"]
        for name in ("body", "face", "hands")
    ]
    conditions = ["replay-only", "replay-resident", "replay-4b", "live-resident"]
    streams = [report["streams"][name] for name in conditions]
    condition_labels = ["Vision\nonly", "Models\nidle", "+ 4B\nload", "Live\ncamera"]
    for ax, data, names, title in [
        (axes[0], modules, labels, "Module call latency · fixed 640×480"),
        (
            axes[1],
            [s["latest_age_ms"] for s in streams],
            condition_labels,
            "Latest-result age · independent samples",
        ),
    ]:
        x = np.arange(len(names))
        for offset, metric, color in [
            (-0.18, "p50", "#285c86"),
            (0.18, "p95", "#89b6d6"),
        ]:
            bars = ax.bar(
                x + offset,
                [d[metric] for d in data],
                0.36,
                label=metric.upper(),
                color=color,
            )
            ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
        ax.set_xticks(x, names)
        ax.set_ylabel("milliseconds")
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, fontsize=9)
    bars = axes[2].bar(
        condition_labels,
        [s["resources"]["minimum_mem_available_gib"] for s in streams],
        color="#4f8c75",
    )
    axes[2].bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    axes[2].set_title("Available unified memory · sampled minimum", fontsize=11)
    axes[2].set_ylabel("GiB")
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.13)
    fig.suptitle(
        "B0-SW · Vision component measurements", fontsize=17, fontweight="bold"
    )
    fig.text(
        0.5,
        0.035,
        "20 paired calls per module/input; 60 s per stream condition. First three conditions use fixed replay.\n"
        "Live camera is observational. One ordered run with dynamic clocks; full application B0 is still pending.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.9))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
