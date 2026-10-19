# SPDX-License-Identifier: Apache-2.0
"""Metric-depth conversion for a calibrated, rectified stereo pair.

Intrinsics/disparity must describe the same resized/cropped pixel coordinates.
The principal-point offset matters when the rectified views have different cx.
This module does not turn a pose model's relative z into a range measurement.
"""

import math

import numpy as np


def disparity_to_depth(
    disparity, *, focal_x_px, baseline_m, principal_offset_px=0.0, valid_mask=None
):
    if (
        not isinstance(disparity, np.ndarray)
        or disparity.ndim != 2
        or not disparity.size
    ):
        raise ValueError("Expected a nonempty two-dimensional disparity map")
    if not np.issubdtype(disparity.dtype, np.floating):
        raise ValueError("Expected floating-point disparity in pixels")
    if not math.isfinite(focal_x_px) or focal_x_px <= 0:
        raise ValueError("Focal length must be positive pixels")
    if not math.isfinite(baseline_m) or baseline_m <= 0:
        raise ValueError("Baseline must be positive meters")
    if not math.isfinite(principal_offset_px):
        raise ValueError("Principal-point offset must be finite pixels")
    denominator = disparity.astype(np.float32) - principal_offset_px
    valid = np.isfinite(denominator) & (denominator > 0)
    if valid_mask is not None:
        if valid_mask.shape != disparity.shape or valid_mask.dtype != np.bool_:
            raise ValueError("Validity mask shape/type differs from disparity")
        valid &= valid_mask
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        np.divide(focal_x_px * baseline_m, denominator, out=depth, where=valid)
    valid &= np.isfinite(depth) & (depth > 0)
    depth[~valid] = np.nan
    return depth, valid
