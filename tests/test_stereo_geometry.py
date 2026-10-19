# SPDX-License-Identifier: Apache-2.0
"""Calibration changes must not silently change the physical distance."""

import numpy as np
import pytest

from benchmarks.stereo_geometry import (
    disparity_to_depth,
    project_depth_points,
    register_depth,
    rectified_contract,
)


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


def camera(width=3, height=2, fx=2):
    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": 2,
        "ppx": 0,
        "ppy": 0,
        "distortion_model": "distortion.none",
        "coefficients": [0] * 5,
    }


def transform(rotation=None, translation=(0, 0, 0)):
    rotation = np.eye(3) if rotation is None else np.asarray(rotation)
    return {
        "rotation": rotation.ravel(order="F").tolist(),
        "rotation_layout": "SDK column-major 3x3",
        "translation_m": list(translation),
    }


def test_identity_registration_preserves_holes_and_optical_z():
    depth = np.array([[1, np.nan, 3], [4, 5, 0]], np.float32)
    result, valid = register_depth(depth, camera(), camera(), transform())
    assert valid.tolist() == [[True, False, True], [True, True, False]]
    np.testing.assert_array_equal(result[valid], depth[valid])
    assert np.isnan(result[~valid]).all()


def test_nearest_surface_wins_when_points_share_a_rgb_pixel():
    depth = np.array([[2, 1, 3], [np.nan, np.nan, np.nan]], np.float32)
    result, valid = register_depth(depth, camera(), camera(fx=0.1), transform())
    assert valid.sum() == 1 and result[0, 0] == 1


def test_column_major_rotation_and_translation_are_not_transposed():
    angle = np.deg2rad(15)
    r = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    depth = np.full((2, 3), np.nan, np.float32)
    depth[0, 0] = 2
    pixels, z, source = project_depth_points(
        depth, camera(), camera(), transform(r, (0.1, 0, 0))
    )
    expected = r @ np.array([0, 0, 2]) + [0.1, 0, 0]
    np.testing.assert_allclose(pixels[0], expected[:2] / expected[2] * 2)
    np.testing.assert_allclose(z, [expected[2]])
    assert source.tolist() == [[0, 0]]


def test_unrectified_pair_and_unhandled_distortion_fail_explicitly():
    calibration = {
        "profiles": {"left_ir": camera(), "right_ir": camera()},
        "extrinsics": {"left_ir_to_right_ir": transform(translation=(-0.05, 0, 0))},
    }
    assert rectified_contract(calibration, (2, 3))["baseline_m"] == 0.05
    calibration["extrinsics"]["left_ir_to_right_ir"] = transform(
        translation=(0.05, 0, 0)
    )
    with pytest.raises(ValueError, match="Rectified"):
        rectified_contract(calibration, (2, 3))
    distorted = camera()
    distorted["coefficients"][0] = 0.1
    with pytest.raises(ValueError, match="distortion"):
        register_depth(np.ones((2, 3), np.float32), distorted, camera(), transform())
