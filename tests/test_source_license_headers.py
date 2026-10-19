# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from project_paths import repo_path


ROOT = Path(__file__).resolve().parents[1]
COPYRIGHT = (
    "SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA "
    "CORPORATION & AFFILIATES. All rights reserved."
)
LICENSE = "SPDX-License-Identifier: Apache-2.0"
SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".css",
    ".h",
    ".hpp",
    ".html",
    ".js",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}


def repository_files():
    # Preserve attribution on retained imported source; local additions must
    # not falsely claim NVIDIA authorship.
    manifest = json.loads((ROOT / "01_src/deploy/upstream-import.json").read_text())
    return [
        repo_path(name)
        for name in manifest["source_sha256"]
        if repo_path(name).is_file()
    ]


def test_nvidia_source_files_have_spdx_headers():
    missing = []
    for path in repository_files():
        relative = path.relative_to(ROOT)
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if "packaging" in relative.parts:
            continue
        first_lines = "\n".join(path.read_text(encoding="utf-8").splitlines()[:20])
        if COPYRIGHT not in first_lines or LICENSE not in first_lines:
            missing.append(str(relative))
    assert missing == [], f"source files missing NVIDIA SPDX headers: {missing}"
