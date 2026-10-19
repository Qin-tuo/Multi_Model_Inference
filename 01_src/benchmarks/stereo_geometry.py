# SPDX-License-Identifier: Apache-2.0
"""Geometry for the deferred neural-stereo experiment, not B1 SDK alignment.

Intrinsics/disparity must describe the same resized/cropped pixel coordinates.
The principal-point offset matters when the rectified views have different cx.
This module does not turn a pose model's relative z into a range measurement.
"""

import math

import numpy as np


def pinhole_intrinsics(value):
    """This D435i contract has zero distortion; fail instead of ignoring it."""
    if value["distortion_model"] not in {
        "distortion.none",
        "distortion.brown_conrady",
        "distortion.inverse_brown_conrady",
    } or not np.allclose(value["coefficients"], 0, atol=1e-12, rtol=0):
        raise ValueError("A calibrated distortion-aware mapping is required")
    numbers = [value[k] for k in ("fx", "fy", "ppx", "ppy")]
    if not np.isfinite(numbers).all() or min(numbers[:2]) <= 0:
        raise ValueError("Invalid camera intrinsics")
    if any(type(value[k]) is not int or value[k] <= 0 for k in ("width", "height")):
        raise ValueError("Invalid calibrated image size")
    return numbers


def rigid_transform(extrinsics):
    if extrinsics["rotation_layout"] != "SDK column-major 3x3":
        raise ValueError("Unknown extrinsics rotation layout")
    rotation = np.array(extrinsics["rotation"], np.float64).reshape(3, 3, order="F")
    translation = np.array(extrinsics["translation_m"], np.float64)
    if (
        translation.shape != (3,)
        or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
    ):
        raise ValueError("Invalid camera rigid transform")
    return rotation, translation


def rectified_contract(calibration, shape):
    left, right = (calibration["profiles"][name] for name in ("left_ir", "right_ir"))
    lf, rf = pinhole_intrinsics(left), pinhole_intrinsics(right)
    rotation, translation = rigid_transform(
        calibration["extrinsics"]["left_ir_to_right_ir"]
    )
    if (
        shape != (left["height"], left["width"])
        or shape != (right["height"], right["width"])
        or not np.allclose(rotation, np.eye(3), atol=1e-5)
        or not np.allclose(translation[1:], 0, atol=1e-6)
        or translation[0] >= 0
        or not np.allclose([lf[0], lf[1], lf[3]], [rf[0], rf[1], rf[3]], atol=1e-3)
    ):
        raise ValueError("Rectified left/right geometry or calibrated size mismatch")
    return {
        "focal_x_px": lf[0],
        "baseline_m": -translation[0],
        "principal_offset_px": lf[2] - rf[2],
    }


def project_depth_points(depth, source, target, extrinsics):
    """Project valid pixel centers; depth values are optical-axis Z in meters."""
    fx, fy, cx, cy = pinhole_intrinsics(source)
    tx, ty, tcx, tcy = pinhole_intrinsics(target)
    if depth.shape != (source["height"], source["width"]):
        raise ValueError("Depth size differs from its calibration")
    rotation, translation = rigid_transform(extrinsics)
    ys, xs = np.nonzero(np.isfinite(depth) & (depth > 0))
    z = depth[ys, xs]
    points = np.column_stack(((xs - cx) / fx * z, (ys - cy) / fy * z, z))
    transformed = points @ rotation.T + translation
    front = np.isfinite(transformed).all(axis=1) & (transformed[:, 2] > 0)
    transformed, xs, ys = transformed[front], xs[front], ys[front]
    pixels = transformed[:, :2] / transformed[:, 2:3] * [tx, ty] + [tcx, tcy]
    return pixels, transformed[:, 2], np.column_stack((xs, ys))


def register_depth(depth, source, target, extrinsics):
    """Nearest-center projection with a Z buffer; holes remain explicitly invalid.

    This is a sparse pixel-center registration, not SDK footprint filling or
    occlusion-confidence estimation. Do not interpolate holes into evidence.
    """
    pixels, z, _ = project_depth_points(depth, source, target, extrinsics)
    width, height = target["width"], target["height"]
    inside = (
        (pixels[:, 0] >= -0.5)
        & (pixels[:, 0] < width - 0.5)
        & (pixels[:, 1] >= -0.5)
        & (pixels[:, 1] < height - 0.5)
    )
    xy = np.floor(pixels[inside] + 0.5).astype(np.int64)
    result = np.full(width * height, np.inf, np.float32)
    np.minimum.at(result, xy[:, 1] * width + xy[:, 0], z[inside])
    result = result.reshape(height, width)
    valid = np.isfinite(result)
    result[~valid] = np.nan
    return result, valid


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
