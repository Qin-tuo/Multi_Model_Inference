# SPDX-License-Identifier: Apache-2.0
"""Locate retained reference inputs after disposable validation output is removed."""

from pathlib import Path


def input_path(path, root):
    root, path = Path(root), Path(path)
    if not path.is_absolute():
        path = root / path
    if path.exists():
        return path
    try:
        relative = path.relative_to(root / "05_runtime/validation")
    except ValueError:
        return path
    return root / "05_runtime/datasets/validation-inputs" / relative


def is_local_input(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return any(
        path.is_relative_to((root / directory).resolve())
        for directory in (
            "05_runtime/validation",
            "05_runtime/datasets/validation-inputs",
        )
    )
