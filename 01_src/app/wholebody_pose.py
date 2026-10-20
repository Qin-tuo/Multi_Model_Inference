# SPDX-License-Identifier: Apache-2.0
"""RTMW3D TensorRT adapter using the pinned MMPose SimCC decoder.

COCO-WholeBody has 133 joints: 17 body, 6 foot, 68 face and 21 per hand.
Raw SimCC peak scores are not probabilities. Learned relative z is kept
separate from the camera's metric depth and is never presented as range.
"""

import cv2
from functools import lru_cache
import json
import numpy as np

from app.body_detector import load_source
from app.tensorrt_runner import TensorRTModel
from project_paths import ROOT


@lru_cache(maxsize=2)
def geometry_helper(path=None, sha256=None):
    if path is None:
        entry = json.loads((ROOT / "03_models/candidates/vision-b1.json").read_text())[
            "models"
        ]["body_hand_pose"]
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


class WholeBodyPose:
    def __init__(self, build, entry):
        geometry = entry["prepost_source"]["files"]["bbox/transforms.py"]
        self.geometry = geometry_helper(geometry["path"], geometry["sha256"])
        source = entry["decoder_source"]
        self.decoder = load_source(
            str(ROOT / source["path"]), source["sha256"], "assistant_simcc3d"
        )
        self.model = TensorRTModel(
            build["engine_path"], build["engine_sha256"], {"input": [1, 3, 384, 288]}
        )
        self.provider = self.model.provider

    def infer(self, image, box, joint_threshold=0.2):
        inputs, matrix = prepare_pose(image, box, self.geometry)
        raw = self.model.infer(inputs)
        bins, scores = self.decoder.get_simcc_maximum(
            raw["output"], raw["1554"], raw["1556"]
        )
        points = bins[0] / 2
        scores = scores[0]
        points[:, :2] = cv2.transform(
            points[None, :, :2], cv2.invertAffineTransform(matrix)
        )[0]
        points[:, 2] = (points[:, 2] / 144 - 1) * 2.1744869
        valid = scores >= joint_threshold
        hands = []
        for side, start in (("left", 91), ("right", 112)):
            keypoints = points[start : start + 21]
            confidence = scores[start : start + 21]
            hands.append(
                {
                    "type": "hand",
                    "side": side,
                    "landmarks": keypoints.tolist(),
                    "landmark_scores": confidence.tolist(),
                    "score": float(confidence.min()),
                    "score_kind": "simcc_peak",
                    "landmark_order": "COCO-WholeBody hand21 (wrist, thumb, index, middle, ring, pinky)",
                    "z_kind": "rtmw3d_relative_root_training_scale",
                    "box_xyxy": np.r_[
                        keypoints[:, :2].min(0), keypoints[:, :2].max(0)
                    ].tolist(),
                }
            )
        return {
            "keypoints": points.tolist(),
            "scores": scores.tolist(),
            "valid": valid.tolist(),
            "layout": "COCO-WholeBody133",
            "z_kind": "rtmw3d_relative_root_training_scale",
            "score_kind": "simcc_peak",
            "hands": hands,
            "affine_source_to_input": matrix.tolist(),
        }

    def close(self):
        self.model.close()
