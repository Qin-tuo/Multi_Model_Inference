"""Move project-owned dependencies into a visible home directory with aliases.

Active users of these directories must be quiesced during the short rename
window. System package installations are indexed, never relocated.
"""
from __future__ import annotations

import argparse
import configparser
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

PROJECT = Path(__file__).resolve().parents[1]


def relocate(source: Path, target: Path):
    if source.is_symlink():
        if source.resolve() != target.resolve():
            raise RuntimeError(f"Existing alias has a different destination: {source}")
        return
    if not source.exists():
        return
    if target.exists():
        raise RuntimeError(f"Refusing to merge or replace an existing directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
    try:
        source.symlink_to(target, target_is_directory=target.is_dir())
    except BaseException:
        target.rename(source)
        raise


def link(source: Path, target: Path):
    if not target.exists():
        return
    if source.is_symlink() and source.resolve() == target.resolve():
        return
    if source.exists() or source.is_symlink():
        raise RuntimeError(f"Refusing to replace {source}")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.symlink_to(target, target_is_directory=target.is_dir())


def framework_inventory(root: Path):
    entries = {}
    for path in sorted((root / "frameworks").iterdir()):
        target = path.resolve(strict=True)
        entry = {"path": str(path), "resolved_path": str(target),
                 "is_link": path.is_symlink()}
        if (target / ".git").exists():
            entry["kind"] = "git_checkout"
            entry["commit"] = subprocess.check_output(
                ["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        elif root / "environments" in target.parents:
            entry["kind"] = "installed_python_package"
        else:
            entry["kind"] = "installed_dependency"
        entries[path.name] = entry
    return entries


def reference_inventory():
    """Index the preserved study references without initializing submodules."""
    modules = configparser.ConfigParser(interpolation=None)
    modules.read(PROJECT / ".gitmodules")
    entries = []
    for section in modules.sections():
        relative = modules[section]["path"]
        path = PROJECT / relative
        entries.append({"path": str(path), "url": modules[section]["url"],
                        "checkout_available": (path / ".git").exists(),
                        "role": "preserved_reference_not_runtime_dependency"})
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.home() / "AI_Stack")
    args = parser.parse_args()
    root = args.root.expanduser().absolute()
    cache = Path(os.environ.get("ASSISTANT_CACHE_DIR", Path.home() / ".cache/multi-model-assistant"))
    data = Path(os.environ.get("ASSISTANT_DATA_DIR", Path.home() / ".local/share/multi-model-assistant"))
    root.mkdir(parents=True, exist_ok=True)
    if root == cache or cache in root.parents or root == data or data in root.parents:
        raise ValueError("Destination cannot be inside the source cache or runtime data")
    relocate(cache, root / "cache")
    for old, new in {"upstreams": "frameworks", "build": "builds", "models": "models",
                     "sdk": "sdk", "downloads": "downloads", "fixtures": "fixtures",
                     "evidence": "reports/preparation", "prototypes": "archives/prototypes", "workspaces": "workspaces"}.items():
        relocate(root / "cache" / old, root / new)
    for logfile in (root / "cache").glob("*.log"):
        relocate(logfile, root / "logs" / logfile.name)
    for report in (root / "cache").glob("*-report.json"):
        relocate(report, root / "reports/install" / report.name)
    relocate(data, root / "runtime-data")
    for old, name in {".venv": "assistant", ".venv-speech": "speech", ".venv-vision": "vision"}.items():
        relocate(PROJECT / old, root / "environments" / name)
    # Keep this target present before the active ORT build installs into it.
    (root / "installed").mkdir(exist_ok=True)
    if (root / "cache/install").exists():
        if not (root / "cache/install").is_symlink():
            (root / "installed").rmdir()
            relocate(root / "cache/install", root / "installed")
    else:
        link(root / "cache/install", root / "installed")
    for name, target in {"cuda": Path("/usr/local/cuda-13.2"), "ros2-jazzy": Path("/opt/ros/jazzy"),
                         "tensorrt-tools": Path("/usr/src/tensorrt"),
                         "tensorrt-python": Path("/usr/lib/python3.12/dist-packages/tensorrt"),
                         "portaudio": Path("/usr/lib/aarch64-linux-gnu/libportaudio.so.2"),
                         "cudnn": root / "sdk/cudnn", "tensorrt-sdk": root / "sdk/tensorrt"}.items():
        link(root / "system" / name, target)
    installed_packages = {}
    for name in ("assistant", "speech", "vision", "ort-build", "memory-eval"):
        python = root / "environments" / name / "bin/python"
        if not python.exists(): continue
        command = "import importlib.metadata as m,json; print(json.dumps(sorted([{'name':d.metadata['Name'],'version':d.version,'location':str(d.locate_file(''))} for d in m.distributions()],key=lambda d:d['name'].lower())))"
        installed_packages[name] = json.loads(subprocess.check_output([str(python), "-c", command], text=True))
    sites = {name: root / "environments" / name / "lib/python3.12/site-packages" for name in installed_packages}
    for name, environment, package in [("mem0", "assistant", "mem0"), ("apscheduler", "assistant", "apscheduler"),
        ("qdrant-client", "assistant", "qdrant_client"), ("sentence-transformers", "assistant", "sentence_transformers"),
        ("transformers", "assistant", "transformers"), ("ultralytics", "vision", "ultralytics"),
        ("voxedge", "speech", "voxedge"), ("piper", "speech", "piper"),
        ("pytorch-cpu", "assistant", "torch"), ("onnxruntime-cpu", "assistant", "onnxruntime"),
        ("onnx", "vision", "onnx"), ("sqlalchemy", "assistant", "sqlalchemy")]:
        link(root / "frameworks" / name, sites[environment] / package)
    link(root / "frameworks/pyaudio", sites["assistant"] / "pyaudio")
    link(root / "frameworks/jionlp", sites["assistant"] / "jionlp")
    link(root / "frameworks/jiojio", sites["assistant"] / "jiojio")
    link(root / "frameworks/onnxruntime", root / "builds/onnxruntime-1.28.0/onnxruntime")
    link(root / "frameworks/voice-engine", root / "frameworks/openvoicestream/third_party/jetson-voice-engine")
    link(root / "frameworks/huggingface-hub-cpp", root / "workspaces/vision/build/yolo_hfhub_vendor/_deps/yolo_hfhub-src")
    link(root / "frameworks/realsense-ros", Path("/opt/ros/jazzy/share/realsense2_camera"))
    link(root / "references", PROJECT / "references")
    for executable in ("python3", "git", "cmake", "ninja", "g++", "nvcc", "gh"):
        binary = shutil.which(executable)
        if binary:
            link(root / "system/tools" / executable, Path(binary))
    (root / "projects").mkdir(exist_ok=True)
    link(root / "projects/Multi_Model_Inference", PROJECT)
    inventory = {"created_at": datetime.now(timezone.utc).isoformat(), "root": str(root),
                 "project": str(PROJECT), "framework_lock": str(PROJECT / "deploy/upstreams.lock.json"),
                 "model_lock": str(PROJECT / "models/manifest.json"), "environments": installed_packages,
                 "frameworks": framework_inventory(root), "references": reference_inventory(),
                 "compatibility": {str(cache): str(root / "cache"), str(data): str(root / "runtime-data")},
                 "system_packages_relocated": False}
    query = subprocess.run(["dpkg-query", "-W", "-f=${Package}\t${Version}\n",
                            "nvidia-l4t-core", "libnvinfer10", "libcudnn9-cuda-13",
                            "ros-jazzy-ros-base", "ros-jazzy-realsense2-camera"],
                           text=True, capture_output=True)
    inventory["system_package_versions"] = [line.split("\t") for line in query.stdout.splitlines()]
    packages = subprocess.check_output(["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${db:Status-Abbrev}\n"], text=True)
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports/system-packages.tsv").write_text(packages)
    (root / "dependencies.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    (root / "README.md").write_text((PROJECT / "deploy/dependencies-layout.md").read_text())
    print(json.dumps({"root": str(root), "inventory": str(root / "dependencies.json"),
                      "environment_package_counts": {key: len(value) for key, value in installed_packages.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
