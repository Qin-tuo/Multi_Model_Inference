#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Copy the selected local runtime into this project; never alter the source.

This prepares artifacts only. Stop services before transferring mutable state
and switching their registered unit paths. Host Jetson libraries remain host-owned.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def relocate_environments(destination, source):
    """Repair venv entry scripts; Python executables still use host Python 3.12."""
    changed = []
    for name in ("reachy-headless", "speech"):
        env = destination / "environments" / name
        aliases = [str(source / "environments" / name)]
        if name == "speech":
            aliases += [str(ROOT / ".venv-speech")]
        for file in [env / "pyvenv.cfg", *list((env / "bin").iterdir())]:
            if file.is_symlink() or not file.is_file():
                continue
            try:
                text = file.read_text()
            except UnicodeDecodeError:
                continue
            updated = text
            for old in aliases:
                updated = updated.replace(old, str(env))
            if updated != text:
                file.write_text(updated)
                changed.append(str(file.relative_to(destination)))
        # Future editable installs must not retain an external source root.
        site = env / "lib/python3.12/site-packages"
        for file in [*site.glob("*.pth"), *site.glob("__editable__*.py")]:
            text = file.read_text()
            updated = text.replace(str(source), str(destination))
            if updated != text:
                file.write_text(updated)
                changed.append(str(file.relative_to(destination)))
    return changed


def copy_runtime(source, destination):
    if destination.exists():
        raise ValueError(
            "Destination already exists; do not overwrite a prepared runtime"
        )
    if not (source / "environments/reachy-headless/bin/python").is_file():
        raise ValueError("Source is not a prepared AI_Stack")
    directories = [
        "environments/reachy-headless",
        "environments/speech",
        "builds/llama/bin",
        "frameworks/llama.cpp",
        "frameworks/openvoicestream",
        "frameworks/opencv-zoo",
        "frameworks/tensorrt-samples-16ed0636",
        "models/matcha-host",
        "models/face",
        "models/hands",
        "models/vad",
        "datasets/reachy-b0-sw-v1",
        "datasets/reachy-b0-e1-v1",
        "datasets/perception-reference",
    ]
    model_manifest = json.loads((ROOT / "03_models/manifest.json").read_text())[
        "models"
    ]
    files = [
        "models/" + row["name"]
        for role in ("vlm", "asr")
        for row in model_manifest[role]["files"]
    ]
    body = json.loads((ROOT / "03_models/perception.json").read_text())["body"]
    files += [
        "models/yolo/" + Path(body[key]).name for key in ("model_path", "engine_path")
    ]
    needed = sum(
        p.stat().st_size
        for name in directories
        for p in (source / name).rglob("*")
        if p.is_file() and not p.is_symlink()
    )
    needed += sum((source / name).stat().st_size for name in files)
    if shutil.disk_usage(destination.parent).free < needed + 2 * 1024**3:
        raise RuntimeError("Insufficient space for the runtime copy")
    destination.mkdir()
    for name in directories:
        print("Copying " + name, flush=True)
        shutil.copytree(
            source / name,
            destination / name,
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
    for name in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    frozen = destination / "baselines/b0"
    frozen.mkdir(parents=True)
    prior = source / "baselines/b0/effective-config.json"
    shutil.copy2(
        prior
        if prior.is_file()
        else source / "runtime-data/reachy-headless/systemd/settings.json",
        frozen / "effective-config.json",
    )
    changes = relocate_environments(destination, source)
    for name in ("frameworks/llama.cpp", "frameworks/openvoicestream"):
        print(
            name,
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=destination / name, text=True
            ).strip(),
        )
    # An accidental external symlink would undermine an isolated-runtime test.
    external = []
    for current, dirs, names in os.walk(destination, followlinks=False):
        for name in dirs + names:
            path = Path(current) / name
            if path.is_symlink() and path.resolve().is_relative_to(source):
                external.append(str(path.relative_to(destination)))
    if external:
        raise RuntimeError("Links still point to source: " + repr(external))
    report = {
        "source": str(source),
        "destination": str(destination),
        "directories": directories,
        "model_files": files,
        "relocated_environment_files": changes,
        "mutable_state_copied": False,
        "copied_bytes_estimate": needed,
    }
    (destination / "import-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    report = copy_runtime(args.source.expanduser().resolve(), ROOT / "05_runtime")
    print(json.dumps({"prepared": True, "bytes": report["copied_bytes_estimate"]}))


if __name__ == "__main__":
    main()
