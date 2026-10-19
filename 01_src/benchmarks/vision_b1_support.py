# SPDX-License-Identifier: Apache-2.0
"""Frozen B1 inputs/contracts and array-aware observer parity helpers."""

import json
import math
import numpy as np
from app.config import Config
from deploy import runtime
from project_paths import ROOT

PROTOCOL = ROOT / "benchmarks/manifests/b1-vision-measurements-v1.json"


def validate():
    protocol = json.loads(PROTOCOL.read_text())
    for path, expected in protocol["locked_files"].items():
        if runtime.sha256(ROOT / path) != expected:
            raise ValueError("B1 frozen file changed: " + path)
    for row in protocol["inputs"].values():
        if runtime.sha256(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("B1 input changed: " + row["path"])
    return protocol, Config.load(ROOT / protocol["effective_config"])


def arguments(protocol, case):
    image = np.load(
        ROOT / protocol["inputs"][case["input"]]["path"], allow_pickle=False
    )
    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise ValueError("Frozen B1 image is not uint8 BGR 640x480")
    image.flags.writeable = False
    return (image, *case.get("arguments", []))


def outputs_match(left, right):
    """No conversion of masks into millions of boxed Python booleans."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and bool(np.allclose(left, right, rtol=1e-5, atol=1e-5))
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            outputs_match(left[k], right[k]) for k in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            outputs_match(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-5)
    return left == right


def plain_call(adapter, args):
    import time

    start = time.perf_counter()
    result = adapter.infer(*args)
    return result, (time.perf_counter() - start) * 1000
