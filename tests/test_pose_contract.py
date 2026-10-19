# SPDX-License-Identifier: Apache-2.0
"""Regression against a tensor hash produced by the original MMPose pipeline."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from app.wholebody_pose import prepare_pose


def test_fractional_person_roi_matches_original_mmpose_preprocessing():
    reference = json.loads(
        (Path(__file__).parent / "fixtures/pose-affine-reference.json").read_text()
    )
    y, x = np.indices(reference["image_hw"])
    image = np.stack(
        [(a * x + b * y) % 256 for a, b in reference["bgr_formula"]], axis=-1
    ).astype(np.uint8)
    assert hashlib.sha256(image.tobytes()).hexdigest() == reference["image_sha256"]
    values, matrix = prepare_pose(image, reference["box"])
    assert values["input"].shape == (1, 3, 384, 288)
    assert values["input"].dtype == np.float32
    assert (
        hashlib.sha256(values["input"].tobytes()).hexdigest()
        == reference["normalized_tensor_sha256"]
    )
    assert matrix.shape == (2, 3) and np.isfinite(matrix).all()


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [2, 0, 1, 1], [0, 0, float("nan"), 1]])
def test_invalid_pose_roi_is_rejected_before_native_inference(box):
    with pytest.raises(ValueError, match="pose person box"):
        prepare_pose(np.zeros((16, 16, 3), np.uint8), box)
