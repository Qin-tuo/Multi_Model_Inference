# SPDX-License-Identifier: Apache-2.0
"""Bounded masks and fractional box edges are part of the deployment contract."""

import numpy as np
import pytest

from app.segmentation import native_masks, prepare_segmentation


def test_mask_budget_is_checked_before_expanding_to_source_resolution():
    with pytest.raises(ValueError, match="pixel budget"):
        native_masks(
            np.ones((2, 4, 4), np.float32),
            np.ones((2, 2), np.float32),
            np.zeros((2, 4), np.float32),
            (11, 13),
            140,
        )


def test_mask_contains_only_pixel_centers_inside_fractional_box():
    masks = native_masks(
        np.ones((2, 4, 4), np.float32),
        np.ones((1, 2), np.float32),
        np.array([[2.2, 3.5, 8.8, 9.2]], np.float32),
        (11, 13),
        200,
    )
    expected = np.zeros((11, 13), bool)
    expected[4:10, 3:9] = True
    np.testing.assert_array_equal(masks[0], expected)
    assert not masks.flags.writeable


def test_unsupported_source_geometry_fails_before_model_or_mask_allocation():
    with pytest.raises(ValueError, match="budget"):
        prepare_segmentation(np.zeros((1080, 1921, 3), np.uint8))
