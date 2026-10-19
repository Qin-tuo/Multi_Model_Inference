#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze public-image and synthetic-PCM fixtures; reuse only verified files.

No downloads, model selection or inference tuning. Generated audio is explicitly
synthetic and its prompt is not an independently transcribed reference.
"""

import argparse
import json
from pathlib import Path
import sys
import wave

import cv2
import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_src"))
from project_paths import RUNTIME_DIR
from deploy import runtime, services  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=RUNTIME_DIR / "datasets/reachy-b0-sw-v1"
    )
    args = parser.parse_args()
    out = args.output.resolve()
    manifest = out / "manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        for entry in saved["files"]:
            if runtime.sha256(out / entry["filename"]) != entry["sha256"]:
                raise RuntimeError("Frozen input changed: " + entry["filename"])
        print(json.dumps({"status": "VERIFIED_REUSE", "manifest": str(manifest)}))
        return
    out.mkdir(parents=True, exist_ok=False)
    cfg = runtime.config(services.profile_path(runtime.config()))
    ready = services.state(services.ORDER, cfg)
    if (
        ready["status"] != "READY"
        or ready["services"]["app"]["health"]["phase"] != "idle"
    ):
        raise RuntimeError("Wait for the managed stack to be ready and idle")
    corpus_path = ROOT / "benchmarks/corpora/reachy-b0-speech-v1.json"
    corpus = json.loads(corpus_path.read_text())
    public = RUNTIME_DIR / "datasets/perception-reference/woman_hands.jpg"
    expected = "70cbeb38e198c9862202e0979c21a99b40ca980d3e7b250176c85b1636a40f12"
    if runtime.sha256(public) != expected:
        raise RuntimeError("Public reference image is missing or changed")
    source = cv2.imread(str(public))
    if source is None or source.shape != (960, 640, 3):
        raise RuntimeError("Unexpected public image geometry")
    canvas = np.zeros((480, 640, 3), np.uint8)
    canvas[:, 160:480] = cv2.resize(source, (320, 480), interpolation=cv2.INTER_AREA)
    report = {
        "created_at": runtime.stamp(),
        "corpus": str(corpus_path),
        "corpus_sha256": runtime.sha256(corpus_path),
        "preparation_script_sha256": runtime.sha256(__file__),
        "public_image_url": "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg",
        "public_image_sha256": expected,
        "files": [],
        "speech_source": {
            "kind": "synthetic",
            "independent_transcription": False,
            "model": cfg.tts.model,
            "voice": cfg.tts.voice,
            "speed": cfg.tts.speed,
            "sample_rate": 16000,
            "channels": 1,
            "sample_width": 2,
            "runtime_artifact_lock_sha256": runtime.sha256(
                ROOT / "01_src/deploy/runtime-artifacts.lock.json"
            ),
        },
    }
    for name, image, transform in [
        (
            "public-original.png",
            source,
            "JPEG decode to BGR uint8, lossless PNG encode; original 640x960",
        ),
        (
            "public-camera-shape.png",
            canvas,
            "INTER_AREA resize to 320x480; black pad 160 pixels left/right; 640x480",
        ),
        (
            "blank-camera-shape.png",
            np.zeros_like(canvas),
            "Synthetic black uint8 640x480 negative workload",
        ),
    ]:
        if not cv2.imwrite(str(out / name), image):
            raise RuntimeError("Could not save image fixture")
        report["files"].append(
            {
                "filename": name,
                "kind": "image",
                "transform": transform,
                "shape": list(image.shape),
                "sha256": runtime.sha256(out / name),
            }
        )
    with httpx.Client(timeout=90, trust_env=False) as client:
        for case in corpus["cases"]:
            print("prepare " + case["id"], flush=True)
            response = client.post(
                cfg.tts.base_url + "/audio/speech",
                json={
                    "model": cfg.tts.model,
                    "voice": cfg.tts.voice,
                    "speed": cfg.tts.speed,
                    "input": case["text"],
                    "response_format": "pcm",
                },
            )
            response.raise_for_status()
            data = response.content
            if not data or len(data) % 2 or len(data) > cfg.stt.max_seconds * 32000:
                raise RuntimeError("Invalid or overlong synthetic PCM input")
            path = out / (case["id"] + ".wav")
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(data)
            report["files"].append(
                {
                    "filename": path.name,
                    "kind": "audio",
                    "id": case["id"],
                    "reference_prompt": case["text"],
                    "duration_seconds": len(data) / 32000,
                    "sha256": runtime.sha256(path),
                }
            )
    runtime.write_json(manifest, report)
    print(
        json.dumps(
            {
                "status": "INPUTS_FROZEN",
                "manifest": str(manifest),
                "count": len(report["files"]),
            }
        )
    )


if __name__ == "__main__":
    main()
