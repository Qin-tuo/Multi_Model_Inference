# SPDX-License-Identifier: Apache-2.0
"""Repository locations and translation of paths in immutable historical records.

编号用于外层目录，app/deploy/benchmarks 仍是普通 Python 包。
旧基线清单保持原字节，通过 repo_path 读取迁移后的同一份工件。
"""

from pathlib import Path, PurePosixPath
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "01_src"
CONFIG_DIR = ROOT / "02_config"
MODEL_DIR = ROOT / "03_models"
WEB_DIR = ROOT / "04_web"
BENCHMARK_DATA_DIR = ROOT / "benchmarks"
RUNTIME_DIR = ROOT / "05_runtime"
LEGACY_STACK_DIR = Path("/home/nvidia/AI_Stack")


def artifact_path(value):
    """Translate the original artifact root without rewriting frozen manifests."""
    path = Path(value).expanduser()
    for previous in (LEGACY_STACK_DIR, ROOT / "11_runtime"):
        if path.is_relative_to(previous):
            return RUNTIME_DIR / path.relative_to(previous)
    return path


def relocate_artifact_values(value):
    if isinstance(value, dict):
        return {key: relocate_artifact_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate_artifact_values(item) for item in value]
    if isinstance(value, str):
        for previous in (LEGACY_STACK_DIR, ROOT / "11_runtime"):
            if value == str(previous) or value.startswith(str(previous) + "/"):
                return str(artifact_path(value))
    return value


def load_artifact_manifest(path):
    """Keep recorded bytes/hashes intact; return project-local paths for execution."""
    return relocate_artifact_values(json.loads(Path(path).read_text()))


_LEGACY_PREFIXES = {
    "app": "01_src/app",
    "deploy": "01_src/deploy",
    "config": "02_config",
    "models": "03_models",
    "static": "04_web",
    "scripts": "scripts",
    "tests": "tests",
    "docs": "docs",
    "patches": "patches",
    "licenses": "licenses",
    "05_scripts": "scripts",
    "06_benchmarks": "benchmarks",
    "07_tests": "tests",
    "08_docs": "docs",
    "09_patches": "patches",
    "10_licenses": "licenses",
    "11_runtime": "05_runtime",
}


def repo_path(relative):
    """Resolve a current or historical repository-relative path, never host paths."""
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected a repository-relative path")
    if not path.parts:
        return ROOT
    first, *tail = path.parts
    if first == "benchmarks":
        prefix = (
            "01_src/benchmarks" if tail and tail[0].endswith(".py") else "benchmarks"
        )
    else:
        prefix = _LEGACY_PREFIXES.get(first, first)
    return ROOT.joinpath(prefix, *tail)
