# SPDX-License-Identifier: Apache-2.0
"""Pinned MMPose crop geometry shared by runtime and reference validation."""

import cv2
from functools import lru_cache
import json
import numpy as np
from app.model_artifacts import load_source
from project_paths import ROOT


@lru_cache(maxsize=2)
def geometry_helper(path=None, sha256=None):
    if path is None:
        from app.config import Config

        runtime = json.loads(
            (ROOT / Config.load().perception.model_manifest).read_text()
        )
        entry = json.loads((ROOT / runtime["source_manifest"]).read_text())["models"][
            "body_hand_pose"
        ]
        source = entry["prepost_source"]["files"]["bbox/transforms.py"]
        path, sha256 = source["path"], source["sha256"]
    return load_source(str(ROOT / path), sha256, "assistant_pose_geometry")


def prepare_pose(image, box, geometry=None, input_size=(288, 384)):
    box = np.asarray(box, np.float32)
    if box.shape != (4,) or not np.isfinite(box).all() or min(box[2:] - box[:2]) <= 0:
        raise ValueError("Invalid pose person box")
    geometry = geometry or geometry_helper()
    center, scale = geometry.bbox_xyxy2cs(box, padding=1.25)
    # Match TopdownAffine._fix_aspect_ratio's array arithmetic and precision.
    # A mathematically equivalent hand-written matrix changed interpolation
    # rounding for fractional ROIs, so use the pinned upstream warp function.
    width, height = np.hsplit(scale[None], [1])
    aspect = input_size[0] / input_size[1]
    scale = np.where(
        width > height * aspect,
        np.hstack([width, width / aspect]),
        np.hstack([height * aspect, height]),
    )[0]
    matrix = geometry.get_warp_matrix(center, scale, 0, output_size=input_size)
    crop = cv2.warpAffine(image, matrix, input_size, flags=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    values = (rgb - np.array([123.675, 116.28, 103.53], np.float32)) / np.array(
        [58.395, 57.12, 57.375], np.float32
    )
    return {"input": np.ascontiguousarray(values.transpose(2, 0, 1)[None])}, matrix
