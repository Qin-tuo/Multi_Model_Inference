# SPDX-License-Identifier: Apache-2.0
"""Explicit source-pixel crops shared by the small visual model adapters."""

import numpy as np


def crop_roi(image, box):
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or not image.size
    ):
        raise ValueError("Expected a nonempty uint8 BGR source image")
    values = np.asarray(box, np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Expected a finite xyxy source-pixel box")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in values)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) > 1920 * 1080:
        raise ValueError("Empty or over-budget visual ROI")
    return image[y1:y2, x1:x2], [x1, y1, x2, y2]
