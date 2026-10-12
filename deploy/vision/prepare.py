#!/usr/bin/env python3
"""Prepare a pinned Jetson yolo_ros/ORT build; heavy builds require explicit flags."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE / "source-lock.json").read_text())


def run(args: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def fetch(source: Path) -> None:
    pin = LOCK["upstream"]["commit"]
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", LOCK["upstream"]["url"], str(source)], check=True)
    if run(["git", "rev-parse", "HEAD"], source) != pin:
        if run(["git", "status", "--porcelain"], source):
            raise RuntimeError("Cached yolo_ros checkout has local changes; refusing to replace it")
        subprocess.run(["git", "fetch", "--depth", "1", "origin", pin], cwd=source, check=True)
        subprocess.run(["git", "checkout", "--detach", pin], cwd=source, check=True)


def sdk_links(prefix: Path, header_dir: Path, library_dir: Path) -> None:
    prefix.mkdir(parents=True, exist_ok=True)
    for name, target in (("include", header_dir), ("lib", library_dir), ("lib64", library_dir)):
        link = prefix / name
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise RuntimeError(f"Refusing to replace existing SDK path: {link}")
        link.symlink_to(target, target_is_directory=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path.home() / ".cache/multi-model-assistant"
    parser.add_argument("--source", type=Path, default=cache / "upstreams/yolo_ros")
    parser.add_argument("--cache", type=Path, default=cache)
    parser.add_argument("--fetch", action="store_true", help="Acquire the pinned yolo_ros source")
    parser.add_argument("--stage-sdk", action="store_true", help="Create user-cache SDK symlinks; no system package changes")
    parser.add_argument("--build-ort", action="store_true", help="Start the heavy CUDA/TensorRT ONNX Runtime source build")
    parser.add_argument("--build-ros", action="store_true", help="Build yolo_ros after ROS/ORT prerequisites are present")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    cache = args.cache.expanduser().resolve()
    if args.fetch:
        fetch(source)
    if not (source / ".git").exists():
        parser.error("Pinned yolo_ros source is absent; run with --fetch first")
    revision = run(["git", "rev-parse", "HEAD"], source)
    if revision != LOCK["upstream"]["commit"]:
        parser.error(f"yolo_ros revision differs from source-lock.json: {revision}")
    header_dir = Path("/usr/include/aarch64-linux-gnu")
    library_dir = Path("/usr/lib/aarch64-linux-gnu")
    required = [header_dir / "cudnn.h", header_dir / "NvInfer.h", library_dir / "libcudnn.so", library_dir / "libnvinfer.so"]
    missing_sdk = [str(p) for p in required if not p.exists()]
    sdk = cache / "sdk"
    if args.stage_sdk or args.build_ort:
        if missing_sdk:
            parser.error("Missing development files: " + ", ".join(missing_sdk))
        sdk_links(sdk / "cudnn", header_dir, library_dir)
        sdk_links(sdk / "tensorrt", header_dir, library_dir)
    version = LOCK["onnxruntime"]["version"]
    ort_prefix = cache / "install" / ("onnxruntime-" + version + "-cuda13-sm87")
    ort_work = cache / "build" / ("onnxruntime-" + version)
    ort_source = ort_work / "onnxruntime"
    env = os.environ.copy()
    build_env = {
        "CUDA_HOME": "/usr/local/cuda-13.2",
        "CUDNN_HOME": str(sdk / "cudnn"),
        "TENSORRT_HOME": str(sdk / "tensorrt"),
        "ORT_BUILD_DIR": str(ort_work),
        "ORT_SOURCE_DIR": str(ort_source),
        "ORT_PARALLEL": str(LOCK["onnxruntime"]["parallel_build_jobs"]),
        "ORT_CMAKE_EXTRA_DEFINES": "onnxruntime_USE_FLASH_ATTENTION=OFF onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF",
    }
    env.update(build_env)
    ort_cmd = ["bash", str(source / "yolo_onnxruntime_vendor/scripts/build_ort_from_source.sh"), version, str(ort_prefix), "--ep", "cuda", "--cuda-arch", LOCK["onnxruntime"]["cuda_architecture"]]
    ros_ws = cache / "workspaces/vision"
    ros_setup = Path("/opt/ros/jazzy/setup.bash")
    colcon_cmd = ["colcon", "build", "--symlink-install", "--executor", "sequential", "--cmake-args", "-DONNXRUNTIME_ROOT=" + str(ort_prefix), "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF"]
    if args.build_ort:
        if platform.machine() != "aarch64":
            parser.error("This build recipe targets aarch64 Orin only")
        ort_source.parent.mkdir(parents=True, exist_ok=True)
        if not (ort_source / ".git").exists():
            subprocess.run(["git", "clone", "--depth", "1", "--branch", LOCK["onnxruntime"]["tag"], "--recurse-submodules", "https://github.com/microsoft/onnxruntime.git", str(ort_source)], check=True)
        ort_revision = run(["git", "rev-parse", "HEAD"], ort_source)
        if ort_revision != LOCK["onnxruntime"]["tag_commit"]:
            parser.error(f"ORT source revision differs from source-lock.json: {ort_revision}")
        subprocess.run(ort_cmd, env=env, check=True)
    if args.build_ros:
        if not ros_setup.exists() or not (ort_prefix / "lib/libonnxruntime.so").exists():
            parser.error("ROS 2 Jazzy setup and completed ONNX Runtime prefix are required")
        (ros_ws / "src").mkdir(parents=True, exist_ok=True)
        link = ros_ws / "src/yolo_ros"
        if not link.exists():
            link.symlink_to(source, target_is_directory=True)
        elif link.resolve() != source:
            parser.error("Existing vision workspace points at a different source")
        subprocess.run(["bash", "-c", "source " + shlex.quote(str(ros_setup)) + " && " + shlex.join(colcon_cmd)], cwd=ros_ws, check=True)
    report = {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "dependencies_pending",
        "source_commit": revision,
        "source": str(source),
        "architecture": platform.machine(),
        "ros_jazzy_setup_exists": ros_setup.exists(),
        "colcon_on_current_path": shutil.which("colcon"),
        "missing_sdk_files": missing_sdk,
        "sdk_staged": all((sdk / name / "include").exists() for name in ("cudnn", "tensorrt")),
        "video_nodes": sorted(str(p) for p in Path("/dev").glob("video*")),
        "onnxruntime_expected": LOCK["onnxruntime"],
        "onnxruntime_prefix": str(ort_prefix),
        "onnxruntime_library_exists": (ort_prefix / "lib/libonnxruntime.so").exists(),
        "build_environment": build_env,
        "ort_build_command": shlex.join(ort_cmd),
        "ros_workspace": str(ros_ws),
        "ros_build_command": "source " + shlex.quote(str(ros_setup)) + " && " + shlex.join(colcon_cmd),
        "runtime_tested": False,
        "gpu_provider_verified": False,
        "model_artifacts_locked": False,
        "next_steps": [
            "Install ROS 2 Jazzy and package.xml dependencies with rosdep on Noble ARM64",
            "Build pinned ORT v1.28.0 with CUDA/TensorRT for sm87 using two compile jobs",
            "Export and SHA256-lock YOLOv8n/YOLOv8n-pose ONNX models with static shape/batch and model license",
            "Build yolo_ros and set provider=tensorrt explicitly; validate actual per-node placement without unreported CPU fallback",
            "Run file/image then D435i tests and add runtime admission plus real inference-start/finish callbacks",
        ],
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
