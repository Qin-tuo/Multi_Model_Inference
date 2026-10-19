# SPDX-License-Identifier: Apache-2.0
"""Input rejection must occur before any work reaches a model's GPU buffers."""

import numpy as np
import pytest

from app.tensorrt_runner import validate_inputs


def test_stereo_inputs_require_both_views_and_correct_dtype():
    shape = (1, 3, 4, 6)
    specs = {name: (shape, np.dtype("float32")) for name in ("left", "right")}
    left = np.zeros(shape, np.float32)
    with pytest.raises(ValueError, match="names"):
        validate_inputs({"left": left}, specs)
    with pytest.raises(ValueError, match="shape/dtype"):
        validate_inputs({"left": left, "right": left.astype(np.float64)}, specs)
    with pytest.raises(ValueError, match="shape/dtype"):
        validate_inputs({"left": left, "right": left[:, :, :1]}, specs)


def test_noncontiguous_roi_is_preserved_and_nan_is_rejected():
    crop = np.arange(120, dtype=np.float32).reshape(1, 3, 4, 10)[..., ::2]
    spec = {"image": (crop.shape, crop.dtype)}
    actual = validate_inputs({"image": crop}, spec)["image"]
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, crop)
    crop[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        validate_inputs({"image": crop}, spec)
