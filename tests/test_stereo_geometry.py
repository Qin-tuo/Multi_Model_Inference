# SPDX-License-Identifier: Apache-2.0
"""Calibration changes must not silently change the physical distance."""

import numpy as np
import pytest

from app.stereo_geometry import disparity_to_depth


def test_principal_offset_and_resize_preserve_metric_distance():
    disparity = np.array([[12.0, 22.0]], np.float32)
    depth, valid = disparity_to_depth(
        disparity, focal_x_px=400, baseline_m=0.05, principal_offset_px=2
    )
    np.testing.assert_allclose(depth, [[2.0, 1.0]])
    assert valid.all()
    resized, _ = disparity_to_depth(
        disparity * 0.5, focal_x_px=200, baseline_m=0.05, principal_offset_px=1
    )
    np.testing.assert_array_equal(resized, depth)


def test_invalid_disparity_and_padding_do_not_become_zero_distance():
    disparity = np.array([[0.0, -1.0, np.nan, np.inf, 10.0, 10.0]], np.float32)
    mask = np.array([[True, True, True, True, False, True]])
    depth, valid = disparity_to_depth(
        disparity, focal_x_px=400, baseline_m=0.05, valid_mask=mask
    )
    assert valid.tolist() == [[False, False, False, False, False, True]]
    assert np.isnan(depth[~valid]).all()
    assert depth[0, -1] == 2.0


@pytest.mark.parametrize("baseline", [0, -0.05, float("nan")])
def test_invalid_baseline_fails_explicitly(baseline):
    with pytest.raises(ValueError, match="Baseline"):
        disparity_to_depth(
            np.ones((2, 3), np.float32), focal_x_px=400, baseline_m=baseline
        )
