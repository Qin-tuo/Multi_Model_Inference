#!/usr/bin/env python3
"""Prepare a pinned Jetson yolo_ros/ORT build; heavy builds require explicit flags."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
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
ROOT = HERE.parents[1]


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


def apply_patches(source, section="patches"):
    for item in LOCK.get(section, []):
        patch = ROOT / item["path"]
        if hashlib.sha256(patch.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("Vision patch differs from its lock")
        reverse = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=source, capture_output=True)
        if reverse.returncode == 0:
            continue
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)


def claim_build(cache):
    # Older in-progress builds predate this lock; inspect their real processes
    # as well, so a missing/stale job report never causes a duplicate build.
    for proc in Path("/proc").glob("[0-9]*"):
        if int(proc.name) == os.getpid(): continue
        try: argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError: continue
        if b"--build-ort" in argv and any(arg.endswith(b"deploy/vision/prepare.py") for arg in argv):
            raise RuntimeError(f"ORT build is already live at PID {proc.name}; resume observing that process")
        if argv and (argv[0].endswith(b"/llama-server") or any(arg.endswith(b"deploy/speech/serve.py") for arg in argv)):
            raise RuntimeError(f"A model service is still live at PID {proc.name}; stop model services before heavy compilation")
    owner = open(cache / "ort-build.owner.lock", "a+b")
    try: fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        owner.close()
        raise RuntimeError("Another process owns the ORT build lock") from None
    return owner


def stage_compile_sdk(source, destination):
    """Copy the completed core ABI for C++ compilation while GPU DSOs build."""
    library = source / "build/Linux/Release/libonnxruntime.so"
    if not library.exists():
        raise RuntimeError("ORT core library is not built yet")
    if run(["git", "rev-parse", "HEAD"], source) != LOCK["onnxruntime"]["tag_commit"]:
        raise RuntimeError("Staged ORT source does not match its pin")
    for folder in ("lib", "include"):
        (destination / folder).mkdir(parents=True, exist_ok=True)
    for item in library.parent.glob("libonnxruntime*.so*"):
        target = destination / "lib" / item.name
        if target.is_symlink(): target.unlink()
        shutil.copy2(item, target, follow_symlinks=True)
    headers = ("onnxruntime_c_api.h", "onnxruntime_cxx_api.h", "onnxruntime_cxx_inline.h",
               "onnxruntime_error_code.h", "onnxruntime_ep_c_api.h",
               "onnxruntime_float16.h", "onnxruntime_session_options_config_keys.h",
               "onnxruntime_run_options_config_keys.h", "provider_options.h",
               "cpu_provider_factory.h", "cuda_provider_factory.h", "tensorrt_provider_factory.h")
    for name in headers:
        item = next((source / "include").rglob(name), None)
        if item: shutil.copy2(item, destination / "include" / name)
        elif name in headers[:5]: raise RuntimeError("Required ORT header missing: " + name)
    for name in ("LICENSE", "ThirdPartyNotices.txt"):
        shutil.copy2(source / name, destination / name)
    (destination / "compile-only.json").write_text(json.dumps({
        "scope": "C++ compilation only; GPU runtime is not declared ready",
        "source_commit": LOCK["onnxruntime"]["tag_commit"],
        "core_sha256": hashlib.sha256(library.read_bytes()).hexdigest()}, indent=2) + "\n")


def hfhub_source(cache):
    item = LOCK["hfhub_vendor"]
    target = cache / "upstreams/huggingface-hub-cpp"
    previous = cache / "workspaces/vision/build/yolo_hfhub_vendor/_deps/yolo_hfhub-src"
    if not target.exists():
        if previous.exists() and run(["git", "rev-parse", "HEAD"], previous) == item["commit"]:
            target.symlink_to(previous.resolve(), target_is_directory=True)
        else:
            target.mkdir(parents=True)
            subprocess.run(["git", "init", str(target)], check=True)
            subprocess.run(["git", "remote", "add", "origin", item["repository"]], cwd=target, check=True)
            subprocess.run(["git", "fetch", "--depth", "1", "origin", item["commit"]], cwd=target, check=True)
            subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=target, check=True)
    if run(["git", "rev-parse", "HEAD"], target) != item["commit"]:
        raise RuntimeError("Hugging Face C++ source differs from its lock")
    return target.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path.home() / ".cache/multi-model-assistant"
    parser.add_argument("--source", type=Path, default=cache / "upstreams/yolo_ros")
    parser.add_argument("--cache", type=Path, default=cache)
    parser.add_argument("--fetch", action="store_true", help="Acquire the pinned yolo_ros source")
    parser.add_argument("--stage-sdk", action="store_true", help="Create user-cache SDK symlinks; no system package changes")
    parser.add_argument("--build-ort", action="store_true", help="Start the heavy CUDA/TensorRT ONNX Runtime source build")
    parser.add_argument("--resume-ort", action="store_true", help="With --build-ort, compile existing configuration without --update")
    parser.add_argument("--build-ros", action="store_true", help="Build yolo_ros after ROS/ORT prerequisites are present")
    parser.add_argument("--build-ros-prereqs", action="store_true", help="Build ROS messages/vendor while the GPU ORT build continues")
    parser.add_argument("--stage-ort-core", action="store_true", help="Stage a separate compile-only ORT SDK; does not make GPU services ready")
    parser.add_argument("--build-venv", type=Path, default=Path.home() / "AI_Stack/environments/ort-build",
                        help="Isolated Python environment for future ORT build invocations")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.resume_ort and not args.build_ort: parser.error("--resume-ort requires --build-ort")
    # Keep the stable logical path used by existing CMake caches. The visible
    # AI_Stack directory is reached through compatibility aliases after relocation.
    source = args.source.expanduser().absolute()
    cache = args.cache.expanduser().absolute()
    if args.fetch:
        fetch(source)
    if not (source / ".git").exists():
        parser.error("Pinned yolo_ros source is absent; run with --fetch first")
    revision = run(["git", "rev-parse", "HEAD"], source)
    if revision != LOCK["upstream"]["commit"]:
        parser.error(f"yolo_ros revision differs from source-lock.json: {revision}")
    apply_patches(source)
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
    # The pinned upstream packaging helper predates ORT 1.28's split C API.
    # Add the two transitive public headers once its normal GPU install exists.
    if (ort_prefix / ".ort-build-info").is_file():
        for header in ("onnxruntime_error_code.h", "onnxruntime_ep_c_api.h"):
            source_header = ort_source / "include/onnxruntime/core/session" / header
            if not source_header.is_file(): raise RuntimeError("Missing pinned ORT public header: " + header)
            shutil.copy2(source_header, ort_prefix / "include" / header)
    compile_prefix = cache / "install/onnxruntime-compile-sdk"
    if args.stage_ort_core:
        stage_compile_sdk(ort_source, compile_prefix)
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
    if args.build_ort:
        sys.path.insert(0, str(ROOT))
        from deploy.vision.cccl_compat import prepare_overlay
        overlay = prepare_overlay(cache)
        if overlay:
            env["NVCC_PREPEND_FLAGS"] = "-I" + shlex.quote(str(overlay))
            build_env["NVCC_PREPEND_FLAGS"] = env["NVCC_PREPEND_FLAGS"]
        if args.resume_ort:
            env["ORT_SKIP_UPDATE"] = "1"
            build_env["ORT_SKIP_UPDATE"] = "1"
    ort_cmd = ["bash", str(source / "yolo_onnxruntime_vendor/scripts/build_ort_from_source.sh"), version, str(ort_prefix), "--ep", "cuda", "--cuda-arch", LOCK["onnxruntime"]["cuda_architecture"]]
    ros_ws = cache / "workspaces/vision"
    ros_setup = Path("/opt/ros/jazzy/setup.bash")
    selected_prefix = compile_prefix if args.stage_ort_core else ort_prefix
    colcon_cmd = ["colcon", "build", "--symlink-install", "--executor", "sequential", "--cmake-args", "-DONNXRUNTIME_ROOT=" + str(selected_prefix), "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF"]
    if args.build_ort:
        build_owner = claim_build(cache)
        build_venv = args.build_venv.expanduser().absolute()
        if not (build_venv / "bin/python").exists():
            subprocess.run([sys.executable, "-m", "venv", str(build_venv)], check=True)
        env.update(VIRTUAL_ENV=str(build_venv), PATH=str(build_venv / "bin") + os.pathsep + env.get("PATH", ""),
                   PYTHONNOUSERSITE="1")
        if platform.machine() != "aarch64":
            parser.error("This build recipe targets aarch64 Orin only")
        ort_source.parent.mkdir(parents=True, exist_ok=True)
        if not (ort_source / ".git").exists():
            subprocess.run(["git", "clone", "--depth", "1", "--branch", LOCK["onnxruntime"]["tag"], "--recurse-submodules", "https://github.com/microsoft/onnxruntime.git", str(ort_source)], check=True)
        ort_revision = run(["git", "rev-parse", "HEAD"], ort_source)
        if ort_revision != LOCK["onnxruntime"]["tag_commit"]:
            parser.error(f"ORT source revision differs from source-lock.json: {ort_revision}")
        apply_patches(ort_source, "ort_patches")
        try:
            subprocess.run(ort_cmd, env=env, check=True)
        finally:
            build_owner.close()
    if args.build_ros or args.build_ros_prereqs:
        if not ros_setup.exists() or (args.build_ros and not (selected_prefix / "lib/libonnxruntime.so").exists()):
            parser.error("ROS 2 Jazzy setup and completed ONNX Runtime prefix are required")
        (ros_ws / "src").mkdir(parents=True, exist_ok=True)
        link = ros_ws / "src/yolo_ros"
        if not link.exists():
            link.symlink_to(source, target_is_directory=True)
        elif link.resolve() != source.resolve():
            parser.error("Existing vision workspace points at a different source")
        if args.build_ros_prereqs:
            colcon_cmd[2:2] = ["--packages-select", "yolo_msgs", "yolo_hfhub_vendor"]
        colcon_cmd += ["-DFETCHCONTENT_SOURCE_DIR_YOLO_HFHUB=" + str(hfhub_source(cache))]
        ros_env = os.environ.copy()
        ros_env.update(MAKEFLAGS="-j2", CMAKE_BUILD_PARALLEL_LEVEL="2")
        subprocess.run(["bash", "-c", "source " + shlex.quote(str(ros_setup)) + " && " + shlex.join(colcon_cmd)], cwd=ros_ws, env=ros_env, check=True)
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
        "compile_sdk_prefix": str(selected_prefix),
        "compile_only_staging": args.stage_ort_core,
        "onnxruntime_library_exists": (ort_prefix / "lib/libonnxruntime.so").exists(),
        "build_environment": build_env,
        "ort_build_command": shlex.join(ort_cmd),
        "ros_workspace": str(ros_ws),
        "ros_build_command": "source " + shlex.quote(str(ros_setup)) + " && " + shlex.join(colcon_cmd),
        "runtime_tested": False,
        "gpu_provider_verified": False,
        "model_artifacts_locked": all("onnx" in json.loads((ROOT / "models/manifest.json").read_text())["models"][task]
                                      for task in ("detection", "pose")),
        "next_steps": [
            "Complete the pinned ORT GPU build and its full provider installation",
            "Reconfigure the compiled ROS workspace against the final GPU prefix instead of the compile-only SDK",
            "Run detection/pose on D435i and inspect actual ORT node placement; API provider selection alone is insufficient",
            "Add bounded visual context and measure capture-to-consumption age under concurrent assistant workloads",
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
