# SPDX-License-Identifier: Apache-2.0
"""为视觉小模型提供统一的原图像素坐标裁剪与输入边界校验。"""

import numpy as np


def crop_roi(image, box):
    """从 uint8 BGR 原图按 xyxy 框裁剪，返回图像视图和实际整数边界。

    先将坐标截断为整数并限制到图像范围；空区域、非有限坐标或超预算区域会报错。
    """
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
