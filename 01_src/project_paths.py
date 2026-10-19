# SPDX-License-Identifier: Apache-2.0
"""Repository locations and translation of paths in immutable historical records.

编号用于外层目录，app/deploy/benchmarks 仍是普通 Python 包。
旧基线清单保持原字节，通过 repo_path 读取迁移后的同一份工件。
"""

from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "01_src"
CONFIG_DIR = ROOT / "02_config"
MODEL_DIR = ROOT / "03_models"
WEB_DIR = ROOT / "04_web"
BENCHMARK_DATA_DIR = ROOT / "06_benchmarks"

_LEGACY_PREFIXES = {
    "app": "01_src/app",
    "deploy": "01_src/deploy",
    "config": "02_config",
    "models": "03_models",
    "static": "04_web",
    "scripts": "05_scripts",
    "tests": "07_tests",
    "docs": "08_docs",
    "patches": "09_patches",
    "licenses": "10_licenses",
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
            "01_src/benchmarks" if tail and tail[0].endswith(".py") else "06_benchmarks"
        )
    else:
        prefix = _LEGACY_PREFIXES.get(first, first)
    return ROOT.joinpath(prefix, *tail)
