# SPDX-License-Identifier: Apache-2.0
"""Pinned upstream model factories for build/reference processes, not the app."""

import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def checked_sources(recipe):
    rows = list(recipe["prepost_source"]["files"].values()) + recipe["source_files"]
    for row in rows:
        path = ROOT / row["path"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != row["sha256"]:
            raise ValueError("Source artifact differs from its lock: " + str(path))


def torch_model(role, recipe):
    import torch

    checked_sources(recipe)
    weights = ROOT / recipe["source_files"][0]["path"]
    if role == "object_embedding":
        folder = ROOT / "05_runtime/frameworks/b1/clip"
        sys.path.insert(0, str(folder))
        import clip

        model, _ = clip.load(str(weights), device="cpu", jit=False)
        # Export only the learned image encoder/projection, not the text branch.
        return model.visual.float().eval()
    raise ValueError("No reviewed PyTorch factory for " + role)
