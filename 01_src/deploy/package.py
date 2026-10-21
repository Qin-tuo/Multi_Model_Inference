#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage and package the current runtime assets with the existing bundle format.

Prepared environments/frameworks and verified assets are copied, not linked
back to the working runtime. User state, experiments, build/reference tool
environments and deferred model candidates are excluded. Frozen B2 and current replay inputs are included. This is not a Jetson OS image.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01_src"))
from deploy import runtime, services
from deploy.environments import relocate_environments
from deploy.bundle import inventory, pack, verify


DIRECTORIES = (
    "environments/edge-assistant",
    "environments/speech",
    "builds/llama/bin",
    "frameworks/llama.cpp",
    "frameworks/openvoicestream",
    "frameworks/tensorrt-samples-16ed0636",
    "datasets/b0-sw-v1",
    "baselines/b2-vision-v1",
    "baselines/current-vision",
)


def selection():
    """One current profile plus its models, helpers, notices and replay inputs."""
    from project_paths import load_artifact_manifest
    from deploy.validation_inputs import input_path

    cfg = runtime.config(ROOT / "02_config/settings.orin-nx.yaml")
    selected = runtime.artifact_files(cfg, services.ORDER)
    current = json.loads((ROOT / cfg.perception.model_manifest).read_text())
    sources = json.loads((ROOT / current["source_manifest"]).read_text())["models"]

    def add(path, expected=None):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_relative_to(ROOT / "05_runtime"):
            raise ValueError("Runtime selection escaped project storage: " + str(path))
        selected[str(path)] = expected or runtime.sha256(path)

    for role in current["models"]:
        entry = sources[role]
        add(entry["path"], entry["sha256"])
        if entry.get("model_card_sha256"):
            add(entry.get("model_card_path", str(Path(entry["path"]).parent / "MODEL_CARD.md")), entry["model_card_sha256"])
        for group in ("prepost_source", "postprocess_source"):
            for row in entry.get(group, {}).get("license_sources", []):
                add(row["path"], row["sha256"])
        for key, row in entry.get("native_sources", {}).items():
            if Path(key).name.upper().startswith(("LICENSE", "COPYING", "NOTICE", "README")):
                add(row["path"], row["sha256"])
        for notice in (ROOT / entry["path"]).parent.glob("*LICENSE*"):
            if notice.is_file():
                add(notice)
    vad = load_artifact_manifest(ROOT / "03_models/speech.json")["vad"]
    for name, row in vad["files"].items():
        add(ROOT / vad["directory"] / name, row["sha256"])
    # The separately labeled palm segment is retained as replay evidence. It is
    # not regenerated or filtered using current model predictions.
    fixture_path = ROOT / "benchmarks/manifests/b2-held-palm-v1.json"
    fixture = json.loads(fixture_path.read_text())
    recording = input_path(ROOT / fixture["recording"], ROOT)
    add(recording, fixture["recording_sha256"])
    first, last = fixture["frame_interval_inclusive"]
    rows = json.loads(recording.read_text())["preview"]["frames"]
    for row in rows:
        if first <= row["frame"]["frame_id"] <= last:
            add(input_path(ROOT / row["path"], ROOT), row["sha256"])
    return [cfg], selected


def bundled_path(path):
    """Retained replay evidence must survive the bundle's validation exclusion."""
    relative = Path(path).relative_to(ROOT / "05_runtime")
    if relative.is_relative_to("validation"):
        return Path("datasets/validation-inputs") / relative.relative_to("validation")
    return relative


def stage_assets(destination):
    source = ROOT / "05_runtime"
    if destination.exists():
        raise FileExistsError("Use a new stage directory")
    if not destination.resolve().is_relative_to((source / "exports").resolve()):
        raise ValueError("Stage must be inside project 05_runtime/exports")
    for name in DIRECTORIES:
        if not (source / name).is_dir():
            raise FileNotFoundError("Freeze the current inputs before packaging: " + name)
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--",
         "01_src", "02_config", "03_models", "scripts", "benchmarks/manifests",
         "benchmarks/corpora", "main.py", "run.sh", "run_web_vision_chat.py"],
        cwd=ROOT, text=True,
    )
    if changed:
        raise RuntimeError("Commit the current runtime/tool sources before packaging")
    from benchmarks.vision_protocol import validate_locks
    from benchmarks.model_protocol import validate as validate_model_inputs

    protocol_path = ROOT / "benchmarks/manifests/vision-measurements.json"
    validate_locks(protocol_path)
    validate_model_inputs()
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
        relative = bundled_path(path)
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
            "02_config/settings.orin-nx.yaml",
        ],
        "runtime_directories": list(DIRECTORIES),
        "verification_protocols": {
            name: runtime.sha256(ROOT / "benchmarks/manifests" / name)
            for name in ("vision-measurements.json", "model-measurements.json")
        },
        "selected_assets": {
            str(bundled_path(p)): sha for p, sha in selected.items()
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
    if archive.with_name(archive.name + ".partial").exists():
        raise FileExistsError(
            "Inspect the interrupted partial archive before packaging"
        )
    if archive.is_relative_to(destination):
        raise ValueError("Archive must be outside its own stage directory")
    stage_assets(destination)
    result = pack(destination, archive)
    runtime.write_json(archive.with_suffix(".json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
