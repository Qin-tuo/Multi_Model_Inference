#!/usr/bin/env python3
"""E1 output latency and an actual host-call timeline; no kernel-time claims."""

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
    data = json.loads(args.summary.read_text())
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]}
    )
    classes = ["text", "audio", "audio_image", "actions"]
    x = np.arange(4)
    for shift, stat, color in [(-0.18, "p50", "#285c86"), (0.18, "p95", "#89b6d6")]:
        values = [
            data["classes"][k]["metrics"]["audio_publication_seconds"][stat]
            for k in classes
        ]
        bars = axes[0].bar(x + shift, values, 0.36, label=stat.upper(), color=color)
        axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[0].set_xticks(x, ["Text", "WAV voice", "WAV + image", "Create note"])
    axes[0].set_ylabel("seconds from client submit")
    axes[0].set_title("Complete app → published reply WAV", fontsize=12)
    axes[0].legend(frameon=False)
    cancel = data["classes"]["cancellation"]["metrics"]["cancel_to_idle_seconds"]
    bars = axes[1].bar(
        ["P50", "P95", "Max"],
        [cancel[k] * 1000 for k in ("p50", "p95", "max")],
        color="#4f8c75",
    )
    axes[1].bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    axes[1].set_ylabel("milliseconds")
    axes[1].set_title("Cancel send → observed native release / idle", fontsize=10)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    fig.suptitle(
        "B0-SW · Complete application measurements", fontsize=17, fontweight="bold"
    )
    fig.text(
        0.5,
        0.035,
        "20 events per class; fixed public-scene replay, native models, tools enabled.\n"
        "WAV publication and logical cancellation are not audible response / speaker silence. One initial run; no performance tuning.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.9))
    fig.savefig(args.output, dpi=160)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)

    # The extra read tools in this actual image question are deliberately kept.
    case = next(t for t in data["trace_examples"] if t["case"] == "audio_image-03")
    stages = [
        s
        for s in case["stages"]
        if s["kind"] not in {"image_base64", "audio_published"}
    ]
    names = {
        "file_vad": "File VAD",
        "asr_call": "ASR adapter",
        "llm_call": "LLM adapter",
        "tts_call": "TTS adapter",
        "tool_completed": "Local tool",
    }
    colors = {
        "file_vad": "#8c8b70",
        "asr_call": "#4f8c75",
        "llm_call": "#285c86",
        "tts_call": "#89b6d6",
        "tool_completed": "#cf914c",
    }
    fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * len(stages) + 1.6)))
    labels = []
    for index, stage in enumerate(stages):
        label = names[stage["kind"]]
        if stage.get("round"):
            label += f" round {stage['round']}"
        if stage.get("name"):
            label += ": " + stage["name"]
        labels.append(label)
        ax.barh(
            index,
            stage["elapsed_seconds"],
            left=stage["start_seconds"],
            height=0.65,
            color=colors[stage["kind"]],
        )
        if stage["elapsed_seconds"] < 0.05:
            ax.plot(
                stage["start_seconds"],
                index,
                "|",
                color=colors[stage["kind"]],
                markersize=15,
            )
    ax.set_yticks(np.arange(len(stages)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("seconds from client submission")
    ax.axvline(
        case["audio_publication_seconds"],
        color="#b0473a",
        linestyle="--",
        label="WAV published",
    )
    ax.set_title(
        f"Actual trace: {case['case']} · image question with extra read-tool calls",
        fontsize=13,
    )
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.02,
        "Host API spans include waits and I/O; this is not a CUDA kernel timeline. All rounds and unrequested reads are retained.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = args.output.with_name(args.output.stem + "-trace" + args.output.suffix)
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
