# SPDX-License-Identifier: Apache-2.0
"""Attribute adapters preserve units and reject invalid/over-budget geometry."""

import numpy as np
import pytest

from app.appearance import prepare_appearance, normalized_features
from app.head_gaze import decode_gaze, prepare_gaze, landmark_pixels
from app.vision_roi import crop_roi


def test_clipped_roi_and_landmarks_use_the_actual_source_crop():
    image = np.zeros((10, 12, 3), np.uint8)
    crop, roi = crop_roi(image, [-2, 2.5, 8.9, 9.9])
    assert roi == [0, 2, 8, 9] and crop.shape == (7, 8, 3)
    points = landmark_pixels(np.full((1, 70), 0.5, np.float32), roi)
    assert points.shape == (35, 2) and points[0].tolist() == [4, 5]


def test_extremely_thin_clip_crop_is_rejected_before_large_resize():
    with pytest.raises(ValueError, match="aspect-ratio budget"):
        prepare_appearance(
            np.zeros((480, 640, 3), np.uint8), [1, 0, 2, 480], "object_embedding"
        )


@pytest.mark.parametrize("value", [0, np.nan, np.inf])
def test_invalid_features_are_not_normalized_into_valid_association_data(value):
    with pytest.raises(ValueError, match="features"):
        normalized_features(np.full(512, value, np.float32))


def test_gaze_angles_and_roll_compensation_follow_explicit_demo_axes():
    forward = decode_gaze([0, 0, -2], 0)
    assert forward["gaze_yaw_deg"] == 0 and forward["gaze_pitch_deg"] == 0
    result = decode_gaze([1, 1, -np.sqrt(2)], 90)
    np.testing.assert_allclose(
        result["gaze_vector"], [0.5, -0.5, -np.sqrt(0.5)], atol=1e-6
    )
    assert abs(result["gaze_pitch_deg"] + 30) < 1e-4
    with pytest.raises(ValueError, match="gaze vector"):
        decode_gaze([0, 0, 0], 0)


def test_eye_crops_outside_the_image_are_unavailable_not_silently_clipped():
    points = np.zeros((35, 2), np.int32)
    points[1] = [4, 0]
    points[2] = [10, 10]
    points[3] = [14, 10]
    with pytest.raises(ValueError, match="Eye ROI"):
        prepare_gaze(np.zeros((20, 20, 3), np.uint8), points, [0, 0, 0])
