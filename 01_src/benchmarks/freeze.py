#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze selected visual inputs and runtime contracts after observer smoke tests.

Module ROI arguments come from the selected TensorRT models. Pipeline replay
uses explicit synthetic aligned depth; live RGBD will be reported separately.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01_src"))
from app.config import Config
from app.vision_models import ROLES, VisionModels
from benchmarks.baseline_support import jsonable
from benchmarks.vision_protocol import PROTOCOL, outputs_match
from benchmarks.vision_timing import TensorRTTrace
from deploy.operations import exclusive_build
from deploy import runtime, services
from deploy.validation_inputs import input_path


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


def input_images(output, parent_path=None):
    inputs, images = {}, {}
    if parent_path is not None:
        parent = json.loads(parent_path.read_text())
        if set(parent["inputs"]) != {
            "public-person",
            "public-person-dimmed",
            "bus",
            "bus-dimmed",
            "blank",
        }:
            raise ValueError("Expected the five established B1 input names")
        for name, recorded in parent["inputs"].items():
            source = (ROOT / recorded["path"]).resolve()
            if (
                not source.is_relative_to(ROOT / "05_runtime")
                or runtime.sha256(source) != recorded["sha256"]
            ):
                raise ValueError("Parent input missing or changed: " + name)
            image = np.load(source, allow_pickle=False)
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError("Parent input is not uint8 BGR 640x480")
            path = output / (name + ".npy")
            np.save(path, image, allow_pickle=False)
            inputs[name] = {
                **recorded,
                "path": str(path.relative_to(ROOT)),
                "sha256": runtime.sha256(path),
            }
            images[name] = image
        return inputs, images
    public = ROOT / "05_runtime/datasets/b0-sw-v1/public-original.png"
    bus = (
        ROOT
        / "05_runtime/validation/b1-segmentation-20261019/parity-opencv411/reference/bus.npz"
    )
    bus = input_path(bus, ROOT)
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
    return inputs, images


def prepare(output, cfg, inputs_from=None, config_path=None):
    inputs, images = input_images(output, inputs_from)
    models = VisionModels(cfg.perception.model_manifest)
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
    effective_path = output / "effective-config.json"
    runtime.write_json(effective_path, cfg.effective())
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
            "03_models/manifest.json",
            "03_models/runtime-components.json",
            "03_models/speech.json",
            "02_config/settings.yaml",
        )
    ]
    selected_manifest = ROOT / cfg.perception.model_manifest
    selected_sources = json.loads(selected_manifest.read_text())["source_manifest"]
    paths += [
        selected_manifest,
        ROOT / selected_sources,
        Path(config_path or ROOT / "02_config/settings.orin-nx.yaml"),
    ]
    paths.append(effective_path)
    protocol = {
        "schema_version": 1,
        "created_at": runtime.stamp(),
        "status": "INPUTS_AND_RUNTIME_FROZEN_MEASUREMENTS_PENDING",
        "runtime_source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "effective_config": str(effective_path.relative_to(ROOT)),
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
        "preservation": "B0/B1 inputs and measurements remain immutable; model/config revisions are not same-model optimization improvements",
    }
    if inputs_from is not None:
        protocol["input_parent"] = {
            "path": str(inputs_from.relative_to(ROOT)),
            "sha256": runtime.sha256(inputs_from),
            "scope": "Reuse verified pixel bytes only; runtime lock and observer parity are generated for this revision.",
        }
    runtime.write_json(output / "protocol.json", protocol)
    return protocol


def uncommitted_sources():
    """Ignore prose drafts, while rejecting edited, renamed or untracked code."""
    paths = [
        "01_src",
        "02_config",
        "03_models",
        "04_web",
        "scripts",
        "run.sh",
        "main.py",
        "run_web_vision_chat.py",
        "requirements.lock",
    ]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "-z", "HEAD", "--", *paths],
        cwd=ROOT,
        text=True,
    )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", *paths],
        cwd=ROOT,
        text=True,
    )
    return sorted(
        {
            path
            for path in (changed + untracked).split("\0")
            if path and Path(path).suffix.lower() not in (".md", ".rst")
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "05_runtime/baselines/current-vision"
    )
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "02_config/settings.orin-nx.yaml"
    )
    parser.add_argument(
        "--inputs-from",
        type=Path,
        help="Reuse an existing immutable input protocol, including in a restored runtime bundle",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    protocol_path = args.protocol.resolve()
    inputs_from = args.inputs_from.resolve() if args.inputs_from else None
    if protocol_path.exists():
        raise FileExistsError(
            "Protocol already exists; do not overwrite a frozen baseline"
        )
    if not protocol_path.is_relative_to(ROOT / "benchmarks/manifests") or (
        inputs_from is not None and not inputs_from.is_relative_to(ROOT)
    ):
        raise ValueError("Keep protocols in the project benchmark manifest directory")
    if not output.is_relative_to(ROOT / "05_runtime"):
        raise ValueError("Inputs must stay in project runtime storage")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        text=True,
    )
    # Unrelated drafts must not be stashed, deleted or silently committed just
    # to measure a reviewed runtime. All executable/config/model sources must
    # still match the committed revision.
    relevant_dirty = uncommitted_sources()
    if relevant_dirty:
        raise RuntimeError(
            "Commit runtime/harness/config/model changes before freezing: "
            + repr(relevant_dirty)
        )
    output.mkdir(parents=True, exist_ok=False)
    config_path = args.config.resolve()
    cfg = Config.load(config_path)
    prior = runtime.config()
    prior = runtime.config(services.profile_path(prior))
    with exclusive_build(prior, output):
        protocol = prepare(output, cfg, inputs_from, config_path)
    protocol["unrelated_worktree_changes"] = dirty.splitlines()
    runtime.write_json(protocol_path, protocol)
    print("Visual inputs and runtime contracts frozen:", protocol_path, flush=True)


if __name__ == "__main__":
    main()
