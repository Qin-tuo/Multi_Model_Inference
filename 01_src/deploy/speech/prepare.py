#!/usr/bin/env python3
"""Inspect the pinned OVS speech subset without loading models or containers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE / "source-lock.json").read_text())


def run(args: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(
        args, cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def package_version(package: str) -> str | None:
    try:
        return run(["dpkg-query", "-W", "-f=${Version}", package])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def acquire(source: Path) -> None:
    pin = LOCK["upstream"]["commit"]
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", LOCK["upstream"]["url"], str(source)],
            check=True,
        )
    if run(["git", "rev-parse", "HEAD"], source) != pin:
        if run(["git", "status", "--porcelain"], source):
            raise RuntimeError(
                "Cached OVS checkout has local changes; refusing to replace it"
            )
        subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", pin], cwd=source, check=True
        )
        subprocess.run(["git", "checkout", "--detach", pin], cwd=source, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / ".cache/multi-model-assistant/upstreams/openvoicestream",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Acquire the pinned source; downloads no model/image",
    )
    parser.add_argument(
        "--engine-source",
        action="store_true",
        help="Initialize only the pinned Jetson engine source submodule",
    )
    parser.add_argument(
        "--output", type=Path, help="Write the inspection JSON to this path"
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    if args.fetch:
        acquire(source)
    if not (source / ".git").exists():
        parser.error("Pinned OVS source is absent; run with --fetch first")
    revision = run(["git", "rev-parse", "HEAD"], source)
    if revision != LOCK["upstream"]["commit"]:
        parser.error(f"OVS revision differs from source-lock.json: {revision}")
    if args.engine_source:
        subprocess.run(
            [
                "git",
                "submodule",
                "update",
                "--init",
                "--depth",
                "1",
                "--",
                "third_party/jetson-voice-engine",
            ],
            cwd=source,
            check=True,
        )
    release_path = source / "deploy/artifacts/v091-release-lock.json"
    release = json.loads(release_path.read_text())
    profile_path = (
        source / "configs/profiles" / (LOCK["selection"]["profile"] + ".json")
    )
    profile = json.loads(profile_path.read_text())
    for model, expected in LOCK["models"].items():
        actual = release["model_artifacts"][model]
        if any(actual.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Upstream artifact metadata drifted for {model}")
    compose = source / LOCK["selection"]["compose"]
    services = []
    in_services = False
    for line in compose.read_text().splitlines():
        if line == "services:":
            in_services = True
        elif line and not line.startswith((" ", "#")):
            in_services = False
        elif in_services and re.fullmatch(r"  [a-zA-Z0-9_-]+:", line):
            services.append(line.strip().rstrip(":"))
    if services != ["speech"]:
        raise RuntimeError(f"Expected standalone speech compose, found {services}")
    cuda_path = Path("/usr/local/cuda-13.2/version.json")
    cuda = None
    if cuda_path.exists():
        cuda = json.loads(cuda_path.read_text()).get("cuda", {}).get("version")
    trt = package_version("libnvinfer10")
    observed = {
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "cuda": cuda,
        "tensorrt_package": trt,
        "l4t_package": package_version("nvidia-l4t-core"),
        "docker": shutil.which("docker"),
        "old_python_trt_mount_exists": Path(
            "/usr/lib/python3.10/dist-packages/tensorrt"
        ).exists(),
        "video_nodes": sorted(str(p) for p in Path("/dev").glob("video*")),
        "alsa_cards": Path("/proc/asound/cards").read_text()
        if Path("/proc/asound/cards").exists()
        else None,
    }
    differences = []
    if cuda is None or not cuda.startswith(
        LOCK["qualified_upstream_target"]["cuda"] + "."
    ):
        differences.append(
            "Published engines target CUDA 12.6; observed toolkit is different or absent"
        )
    if trt is None or not trt.startswith(
        LOCK["qualified_upstream_target"]["tensorrt"] + "."
    ):
        differences.append(
            "Published engines/plugins target TensorRT 10.3; observed package is different or absent"
        )
    if not observed["old_python_trt_mount_exists"]:
        differences.append(
            "Standalone upstream compose requires an unavailable Python 3.10 TensorRT host mount"
        )
    engine = source / "third_party/jetson-voice-engine"
    engine_revision = (
        run(["git", "rev-parse", "HEAD"], engine)
        if (engine / ".git").exists()
        else None
    )
    if (
        engine_revision
        and engine_revision != LOCK["upstream"]["engine_submodule_commit"]
    ):
        raise RuntimeError(f"Engine submodule revision drifted: {engine_revision}")
    report = {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "needs_host_port"
        if differences
        else "source_verified_runtime_unverified",
        "runtime_tested": False,
        "models_loaded": False,
        "source": str(source),
        "source_commit": revision,
        "engine_source_commit": engine_revision,
        "upstream_release_lock_sha256": hashlib.sha256(
            release_path.read_bytes()
        ).hexdigest(),
        "upstream_qualified_target": release["target"],
        "observed": observed,
        "selected_services": services,
        "selected_profile": profile["name"],
        "selected_backends": {
            "asr": profile["asr_backend"],
            "tts": profile["tts_backend"],
        },
        "selected_models": LOCK["models"],
        "tts_format": {
            "sample_rate": int(profile["env"]["OVS_TTS_SAMPLE_RATE"]),
            "channels": int(profile["env"]["OVS_TTS_CHANNELS"]),
            "pcm_bits": 16,
        },
        "compatibility_gaps": differences,
        "source_dependency_drift": {
            "release_lock_voxedge": release["dependencies"]["voxedge"]["version"],
            "dockerfile_default_voxedge": re.search(
                r"ARG VOXEDGE_VERSION=(\S+)",
                (
                    source / "deploy/docker/Dockerfile.jetson.edgellm-v091-runtime"
                ).read_text(),
            ).group(1),
        },
        "next_steps": [
            "Port the pinned engine-overlay target manifests and build preflight to the observed host tuple; retain source/model/patch provenance",
            "Build ASR worker/plugin/encoder/thinker and Matcha split estimator/Vocos engines against the local CUDA/TensorRT",
            "Create a local speech-only runtime/profile/lock with actual Python ABI and library paths; do not overwrite the upstream release lock",
            "Verify /readyz, /v1/models, /v1/capabilities, file ASR/TTS, then real audio once validated devices are connected",
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
