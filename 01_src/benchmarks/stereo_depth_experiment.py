# SPDX-License-Identifier: Apache-2.0
"""Deferred TensorRT stereo experiment; B1 uses native D435i depth.

Model normalization follows the pinned NVIDIA exporter. Grayscale IR is
replicated to three channels and must pass separate sensor-domain validation.
"""

import time

import cv2
import numpy as np

from benchmarks.stereo_geometry import (
    disparity_to_depth,
    rectified_contract,
    register_depth,
)
from app.tensorrt_runner import TensorRTModel

MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


def prepare_pair(left, right, target_hw=(480, 736)):
    if left.shape != right.shape or left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("Stereo views need matching uint8 dimensions")
    if left.ndim != 2 and (left.ndim != 3 or left.shape[2] != 3):
        raise ValueError("Expected grayscale IR or RGB images")
    height, width = left.shape[:2]
    if min(height, width) <= 0:
        raise ValueError("Empty stereo pair")
    th, tw = target_hw
    scale = min(1.0, tw / width, th / height)
    rw, rh = round(width * scale), round(height * scale)
    tensors = {}
    for name, source in (("left_image", left), ("right_image", right)):
        image = cv2.resize(source, (rw, rh), interpolation=cv2.INTER_AREA)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        image = cv2.copyMakeBorder(image, 0, th - rh, 0, tw - rw, cv2.BORDER_REPLICATE)
        tensors[name] = np.ascontiguousarray(
            ((image.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
        )
    return tensors, {
        "original_hw": [height, width],
        "unpad_hw": [rh, rw],
        "scale_xy": [rw / width, rh / height],
        "padding": "bottom/right replicate",
    }


class StereoDepth:
    def __init__(self, build):
        if build["status"] != "BUILT_NOT_ACCEPTED":
            raise ValueError("A verified successful engine build is required")
        shapes = {name: [1, 3, 480, 736] for name in ("left_image", "right_image")}
        if build["input_shapes"] != shapes:
            raise ValueError("Depth engine does not match the reviewed contract")
        self.model = TensorRTModel(build["engine_path"], build["engine_sha256"], shapes)
        self.provider = self.model.provider

    def infer(self, left, right, calibration, *, register_rgb=False):
        tick = time.monotonic()
        contract = rectified_contract(calibration, left.shape[:2])
        inputs, transform = prepare_pair(left, right)
        prepared = time.monotonic()
        value = self.model.infer(inputs)["disparity"]
        inferred = time.monotonic()
        if value.shape != (1, 1, 480, 736):
            raise ValueError("Unexpected disparity output dimensions")
        rh, rw = transform["unpad_hw"]
        height, width = transform["original_hw"]
        disparity = (
            cv2.resize(
                value[0, 0, :rh, :rw], (width, height), interpolation=cv2.INTER_LINEAR
            )
            / transform["scale_xy"][0]
        )
        # A left-image match must land inside the actual, unpadded right view.
        right_x = np.arange(width)[None, :] - disparity
        support = (right_x >= 0) & (right_x <= width - 1)
        depth, valid = disparity_to_depth(disparity, **contract, valid_mask=support)
        result = {
            "disparity_px": disparity,
            "depth_m": depth,
            "valid": valid,
            "transform": transform,
            "calibration_contract": contract,
            "provider": self.provider,
            "validity_scope": "Finite positive disparity and in-view correspondence; no learned confidence or occlusion guarantee",
        }
        if register_rgb:
            result["rgb_depth_m"], result["rgb_valid"] = register_depth(
                depth,
                calibration["profiles"]["left_ir"],
                calibration["profiles"]["color"],
                calibration["extrinsics"]["left_ir_to_color"],
            )
        result["stages_ms"] = {
            "preprocess": (prepared - tick) * 1000,
            "synchronous_inference_and_copies": (inferred - prepared) * 1000,
            "postprocess": (time.monotonic() - inferred) * 1000,
        }
        return result

    def close(self):
        self.model.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
