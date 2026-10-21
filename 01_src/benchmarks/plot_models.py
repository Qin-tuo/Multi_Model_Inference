#!/usr/bin/env python3
"""Render component latency measurements; use system matplotlib."""

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
    data = json.loads(args.summary.read_text())["roles"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    speech = [
        ("s1", "vad_seconds", "File VAD"),
        ("s1", "asr_seconds", "ASR"),
        ("t1", "first_pcm_seconds", "TTS first\nPCM block"),
        ("t1", "call_seconds", "TTS full\nPCM"),
    ]
    x = np.arange(len(speech))
    for shift, stat, color in [(-0.18, "p50", "#285c86"), (0.18, "p95", "#89b6d6")]:
        values = [
            data[role]["measured_after_warmup"]["metrics"][metric][stat] * 1000
            for role, metric, _ in speech
        ]
        bars = axes[0].bar(x + shift, values, 0.36, label=stat.upper(), color=color)
        axes[0].bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
    axes[0].set_xticks(x, [label for _, _, label in speech])
    axes[0].set_ylabel("milliseconds")
    axes[0].set_title("Speech interface latency · warm calls", fontsize=11)
    axes[0].legend(frameon=False, fontsize=9)
    roles = ["l1-text", "l1-image"]
    x = np.arange(2)
    for shift, phase, label, color in [
        (-0.18, "first_input_call", "First call for each input", "#e2ad59"),
        (0.18, "measured_after_warmup", "After five warmups", "#285c86"),
    ]:
        values = [
            data[role][phase]["metrics"]["first_content_seconds"]["p50"]
            for role in roles
        ]
        bars = axes[1].bar(x + shift, values, 0.36, label=label, color=color)
        axes[1].bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    axes[1].set_xticks(x, ["Text", "Image"])
    axes[1].set_ylabel("seconds")
    axes[1].set_title("Client first content · P50", fontsize=11)
    axes[1].legend(frameon=False, fontsize=8)
    values = [
        data[role]["measured_after_warmup"]["native_decode_tokens_per_second"]["p50"]
        for role in roles
    ]
    bars = axes[2].bar(["Text", "Image"], values, color="#4f8c75")
    axes[2].bar_label(bars, fmt="%.2f", fontsize=10, padding=3)
    axes[2].set_ylabel("tokens / second")
    axes[2].set_title("Native decode rate · P50", fontsize=11)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.2)
    fig.suptitle("Model interface measurements", fontsize=17, fontweight="bold")
    fig.text(
        0.5,
        0.035,
        "20 fixed cases per role; cache and model settings unchanged. File audio, no physical playback.\n"
        "First input calls may share cached prefixes. These are component timings; complete-application measurements are separate.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.9))
    fig.savefig(args.output, dpi=160)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
