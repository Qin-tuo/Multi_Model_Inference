#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze B1 public inputs and runtime contracts after model/observer smoke tests.

Module ROI arguments come from the selected TensorRT models. Pipeline replay
uses explicit synthetic aligned depth; live RGBD will be reported separately.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_src"))
from app.config import Config
from app.vision_models import ROLES, VisionModels
from benchmarks.baseline_support import jsonable
from benchmarks.vision_b1_support import PROTOCOL, outputs_match
from benchmarks.vision_b1_timing import TensorRTTrace
from build_vision_b1 import exclusive_build
from deploy import runtime, services


def letterbox(image):
    height, width = image.shape[:2]
    ratio = min(640 / width, 480 / height)
    w, h = round(width * ratio), round(height * ratio)
    x, y = (640 - w) // 2, (480 - h) // 2
    result = np.zeros((480, 640, 3), np.uint8)
    result[y : y + h, x : x + w] = cv2.resize(
        image, (w, h), interpolation=cv2.INTER_LINEAR
    )
    return result


def prepare(output, cfg):
    inputs, images = {}, {}
    public = ROOT / "05_runtime/datasets/b0-sw-v1/public-original.png"
    bus = (
        ROOT
        / "05_runtime/validation/b1-segmentation-20261019/parity-opencv411/reference/bus.npz"
    )
    with np.load(bus, allow_pickle=False) as data:
        bus_image = data["source"]
    originals = {
        "public-person": (cv2.imread(str(public)), public),
        "bus": (bus_image, bus),
    }
    for name, (image, source) in originals.items():
        for suffix, factor in (("", 1.0), ("-dimmed", 0.7)):
            key = name + suffix
            photo = letterbox(
                np.rint(image.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)
            )
            images[key] = photo
            path = output / (key + ".npy")
            np.save(path, photo, allow_pickle=False)
            inputs[key] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": runtime.sha256(path),
                "source": str(source.relative_to(ROOT)),
                "source_sha256": runtime.sha256(source),
                "brightness": factor,
                "transform": "BGR 640x480 aspect-preserving INTER_LINEAR resize, centered black padding",
            }
    images["blank"] = np.zeros((480, 640, 3), np.uint8)
    path = output / "blank.npy"
    np.save(path, images["blank"], allow_pickle=False)
    inputs["blank"] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": runtime.sha256(path),
        "source": "synthetic black negative",
    }
    models = VisionModels()
    cases, smoke = {role: [] for role in ROLES}, []
    try:
        models.load()
        for name, image in images.items():
            if name in ("public-person", "bus", "blank"):
                for role in ("detection_segmentation", "face"):
                    cases[role].append({"input": name})
            detections = models.run("detection_segmentation", image)
            people = [r for r in detections if r["class_id"] == 0]
            if name in ("public-person", "bus"):
                if not people:
                    raise ValueError("No person in positive fixture: " + name)
                person = max(
                    people,
                    key=lambda r: (r["box_xyxy"][2] - r["box_xyxy"][0])
                    * (r["box_xyxy"][3] - r["box_xyxy"][1]),
                )
                for role in ("body_hand_pose", "person_reid"):
                    cases[role].append(
                        {"input": name, "arguments": [person["box_xyxy"]]}
                    )
            if name.startswith("public-person"):
                faces = models.run("face", image)
                if not faces:
                    raise ValueError("No face in positive fixture: " + name)
                face = faces[0]
                head = models.run("head_pose", image, face["box_xyxy"])
                points = models.run("gaze_landmarks", image, face["box_xyxy"])
                for role in ("head_pose", "gaze_landmarks"):
                    cases[role].append({"input": name, "arguments": [face["box_xyxy"]]})
                for role in ("face_identity", "expression"):
                    cases[role].append(
                        {"input": name, "arguments": [face["landmarks"]]}
                    )
                cases["gaze"].append(
                    {
                        "input": name,
                        "arguments": [points["landmarks"], head["angles_degrees"]],
                    }
                )
            if name.startswith("bus"):
                objects = [r for r in detections if r["class_id"] == 5]
                if not objects:
                    raise ValueError("No bus in object fixture")
                cases["object_embedding"].append(
                    {"input": name, "arguments": [objects[0]["box_xyxy"]]}
                )
        for role, entries in cases.items():
            if not entries:
                raise RuntimeError("No benchmark fixture for " + role)
            with TensorRTTrace(models.models[role]) as trace:
                for case in entries:
                    args = (images[case["input"]], *case.get("arguments", []))
                    plain = models.models[role].infer(*args)
                    traced, timings = trace.call(*args)
                    if not outputs_match(plain, traced) or not all(
                        np.isfinite(v) and v >= 0 for v in timings.values()
                    ):
                        raise RuntimeError("Timing observer parity failed: " + role)
                    smoke.append(
                        {
                            "role": role,
                            "input": case["input"],
                            "passed": True,
                            "timings": timings,
                        }
                    )
                    runtime.write_json(output / "observer-smoke.json", smoke)
    finally:
        models.close()
    config_path = output / "effective-config.json"
    runtime.write_json(config_path, cfg.effective())
    paths = [
        p
        for folder in ("01_src/app", "01_src/deploy")
        for p in (ROOT / folder).rglob("*.py")
    ]
    paths += [
        ROOT / p
        for p in (
            "run_web_vision_chat.py",
            "01_src/project_paths.py",
            "requirements.lock",
            "01_src/deploy/vision/requirements-runtime.lock",
            "03_models/candidates/vision-b1.json",
            "03_models/vision-b1-runtime.json",
            "03_models/manifest.json",
            "03_models/perception.json",
            "03_models/speech.json",
            "02_config/settings.orin-nx-b1.yaml",
        )
    ]
    paths.append(config_path)
    protocol = {
        "schema_version": 1,
        "created_at": runtime.stamp(),
        "status": "INPUTS_AND_RUNTIME_FROZEN_MEASUREMENTS_PENDING",
        "runtime_source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "effective_config": str(config_path.relative_to(ROOT)),
        "locked_files": {
            str(p.relative_to(ROOT)): runtime.sha256(p) for p in sorted(set(paths))
        },
        "inputs": inputs,
        "module_cases": jsonable(cases),
        "measurement": {
            "warmup_per_case": 5,
            "paired_samples_per_case": 20,
            "percentile_method": "nearest-rank",
            "module_processes": "One selected engine per isolated worker; 4B/audio/app services stopped",
            "instrumentation": "Paired plain/instrumented calls in alternating order; host preprocess/native/post and CUDA-stream H2D/execution/D2H spans; output parity required",
            "resources": "1s host/process samples and tegrastats, process RSS high-water; stream spans are not isolated kernel times; telemetry includes harness validation between calls",
            "pipeline_replay": "10 arrivals/s, latest-frame worker with unchanged 5 FPS cap; uniform synthetic aligned 1.5m depth for task activation, not depth accuracy",
            "live_scope": "D435i native RGBD reported separately; configured FPS is not measured throughput",
            "benchmark_source": "Exact harness source hashes/copies and environment recorded for every measurement; runtime locks stay unchanged",
        },
        "preservation": "B0 remains immutable; B1 model/config changes are not same-model optimization improvements",
    }
    runtime.write_json(output / "protocol.json", protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "05_runtime/baselines/b1-vision-v1"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if PROTOCOL.exists():
        raise FileExistsError(
            "B1 protocol already exists; do not overwrite a frozen baseline"
        )
    if not output.is_relative_to(ROOT / "05_runtime"):
        raise ValueError("Inputs must stay in project runtime storage")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        text=True,
    )
    if dirty:
        raise RuntimeError("Commit reviewed source changes before freezing B1")
    output.mkdir(parents=True, exist_ok=False)
    cfg = Config.load(ROOT / "02_config/settings.orin-nx-b1.yaml")
    prior = runtime.config()
    prior = runtime.config(services.profile_path(prior))
    with exclusive_build(prior, output):
        protocol = prepare(output, cfg)
    runtime.write_json(PROTOCOL, protocol)
    print("B1 inputs and runtime contracts frozen:", PROTOCOL, flush=True)


if __name__ == "__main__":
    main()
