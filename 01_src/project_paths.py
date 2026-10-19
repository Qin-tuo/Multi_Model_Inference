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

# Only compatibility readers know the former names. Frozen evidence is never
# rewritten just to change product branding; current paths use neutral names.
LEGACY_RUNTIME_PATHS = {
    "environments/reachy-headless": "environments/edge-assistant",
    "runtime-data/reachy-headless": "runtime-data/edge-assistant",
    "runtime-data/reachy-actions": "runtime-data/actions",
    "runtime-data/reachy-audio": "runtime-data/audio",
    "datasets/reachy-b0-sw-v1": "datasets/b0-sw-v1",
    "datasets/reachy-b0-e1-v1": "datasets/b0-e1-v1",
}


def runtime_relative_path(value):
    """Map exact legacy directory prefixes, not arbitrary user-provided text."""
    path = Path(value)
    for old, new in LEGACY_RUNTIME_PATHS.items():
        if path.is_relative_to(old):
            return Path(new) / path.relative_to(old)
    return path


def artifact_path(value):
    """Translate the original artifact root without rewriting frozen manifests."""
    path = Path(value).expanduser()
    for previous in (LEGACY_STACK_DIR, ROOT / "11_runtime", RUNTIME_DIR):
        if path.is_relative_to(previous):
            return RUNTIME_DIR / runtime_relative_path(path.relative_to(previous))
    return path


def relocate_artifact_values(value):
    if isinstance(value, dict):
        return {key: relocate_artifact_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate_artifact_values(item) for item in value]
    if isinstance(value, str):
        for previous in (LEGACY_STACK_DIR, ROOT / "11_runtime", RUNTIME_DIR):
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
    resolved = PurePosixPath(prefix, *tail)
    if resolved.parts[0] == "benchmarks" and resolved.name.startswith("reachy-"):
        filename = resolved.name.removeprefix("reachy-")
        resolved = resolved.with_name(filename)
    aliases = {
        "01_src/deploy/reachy-import.json": "01_src/deploy/upstream-import.json",
        "01_src/deploy/reachy-upstream.lock.json": "01_src/deploy/upstream.lock.json",
        "docs/reachy-migration.md": "docs/migration-history.md",
    }
    return ROOT / aliases.get(str(resolved), str(resolved))
