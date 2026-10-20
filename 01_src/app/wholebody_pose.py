# SPDX-License-Identifier: Apache-2.0
"""RTMW3D TensorRT adapter using the pinned MMPose SimCC decoder.

COCO-WholeBody has 133 joints: 17 body, 6 foot, 68 face and 21 per hand.
Raw SimCC peak scores are not probabilities. Learned relative z is kept
separate from the camera's metric depth and is never presented as range.
"""

import cv2
import numpy as np
from app.pose_geometry import geometry_helper, prepare_pose

from app.model_artifacts import load_source
from app.tensorrt_runner import TensorRTModel
from project_paths import ROOT


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
