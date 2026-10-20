#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage and package selected B1 runtime assets with the existing bundle format.

Prepared environments/frameworks and verified assets are copied, not linked
back to the working runtime. User state, experiments, build/reference tool
environments and deferred model candidates are excluded. B0 rollback assets
and frozen B1 inputs are included. This is not a Jetson OS image.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_src"))
from deploy import runtime, services
from import_runtime import relocate_environments
from runtime_bundle import inventory, pack, verify


DIRECTORIES = (
    "environments/edge-assistant",
    "environments/speech",
    "builds/llama/bin",
    "frameworks/llama.cpp",
    "frameworks/openvoicestream",
    "frameworks/tensorrt-samples-16ed0636",
    "datasets/b0-sw-v1",
    "baselines/b0",
    "baselines/b1-vision-v1",
    "baselines/b1-vision-v2",
)


def selection():
    configs = [
        runtime.config(ROOT / "02_config/settings.orin-nx-b1.yaml"),
        runtime.config(ROOT / "02_config/settings.orin-nx-headless.yaml"),
    ]
    selected = {}
    for cfg in configs:
        selected.update(runtime.artifact_files(cfg, services.ORDER))
    # Runtime helpers already appear in artifact_files; preserve their notices.
    manifest = json.loads((ROOT / "03_models/candidates/vision-b1.json").read_text())
    for role in json.loads((ROOT / "03_models/vision-b1-runtime.json").read_text())[
        "models"
    ]:
        entry = manifest["models"][role]
        card = ROOT / entry["path"]
        card = card.parent / "MODEL_CARD.md"
        selected[str(card)] = entry["model_card_sha256"]
        references = entry.get("prepost_source", {})
        for source in [
            *references.get("files", {}).values(),
            *references.get("license_sources", []),
        ]:
            selected[str(ROOT / source["path"])] = source["sha256"]
        for notice in card.parent.glob("*LICENSE*"):
            if notice.is_file():
                selected[str(notice)] = runtime.sha256(notice)
    # OpenCV Zoo is imported by model subdirectory, not as a complete checkout.
    for relative in (
        "frameworks/b1/mmpose/LICENSE",
        "frameworks/opencv-zoo/palm_detection_mediapipe-233e619d/LICENSE",
        "frameworks/opencv-zoo/handpose_estimation_mediapipe-4b2a0b44/LICENSE",
    ):
        path = ROOT / "05_runtime" / relative
        selected[str(path)] = runtime.sha256(path)
    return configs, selected


def stage_assets(destination):
    source = ROOT / "05_runtime"
    if destination.exists():
        raise FileExistsError("Use a new stage directory")
    if not destination.resolve().is_relative_to(source / "exports"):
        raise ValueError("Stage must be inside project 05_runtime/exports")
    configs, selected = selection()
    for cfg in configs:
        runtime.verify(cfg, services.ORDER)
    for path, expected in selected.items():
        if runtime.sha256(Path(path)) != expected:
            raise ValueError("Selected artifact differs from lock: " + path)
    needed = sum(
        p.stat().st_size
        for name in DIRECTORIES
        for p in (source / name).rglob("*")
        if p.is_file() and not p.is_symlink()
    )
    needed += sum(Path(p).stat().st_size for p in selected)
    if shutil.disk_usage(source).free < needed * 2 + 2 * 1024**3:
        raise RuntimeError("Insufficient space for stage and archive")
    destination.mkdir(parents=True)
    for name in DIRECTORIES:
        print("Copying", name, flush=True)
        shutil.copytree(
            source / name,
            destination / name,
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
    for name in selected:
        path = Path(name)
        relative = path.relative_to(source)
        target = destination / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            target.symlink_to(path.readlink())
        else:
            shutil.copy2(path, target)
    changes = relocate_environments(destination, source)
    metadata = {
        "created_at": runtime.stamp(),
        "scope": __doc__,
        "source_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "profiles": [
            "02_config/settings.orin-nx-b1.yaml",
            "02_config/settings.orin-nx-headless.yaml",
        ],
        "runtime_directories": list(DIRECTORIES),
        "selected_assets": {
            str(Path(p).relative_to(source)): sha for p, sha in selected.items()
        },
        "relocated_environment_files": changes,
        "host_requirements": {
            "architecture": "aarch64",
            "python": "3.12",
            "l4t": Path("/etc/nv_tegra_release").read_text(),
            "tensorrt": "10.16.2.10",
            "cuda": "13.2",
            "note": "System CUDA/TensorRT, OS libraries and speech system-site packages remain host-provided.",
        },
    }
    runtime.write_json(destination / "bundle-selection.json", metadata)
    manifest = inventory(destination)
    verify(destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination, archive = args.stage.resolve(), args.output.resolve()
    if archive.exists():
        raise FileExistsError("Archive already exists")
    if archive.is_relative_to(destination):
        raise ValueError("Archive must be outside its own stage directory")
    stage_assets(destination)
    result = pack(destination, archive)
    runtime.write_json(archive.with_suffix(".json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
