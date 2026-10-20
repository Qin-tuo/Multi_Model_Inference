# SPDX-License-Identifier: Apache-2.0
"""Official RTMPose whole-body xy output; metric depth belongs to the camera."""

import cv2
import numpy as np

from app.body_detector import load_source
from app.tensorrt_runner import TensorRTModel
from app.wholebody_pose import geometry_helper, prepare_pose
from project_paths import ROOT


class WholeBodyPose2D:
    def __init__(self, build, entry):
        geometry = entry["prepost_source"]["files"]["bbox/transforms.py"]
        self.geometry = geometry_helper(geometry["path"], geometry["sha256"])
        decoder = entry["decoder_source"]
        self.decoder = load_source(
            str(ROOT / decoder["path"]), decoder["sha256"], "assistant_simcc2d"
        )
        self.input_size = tuple(entry["input_size_wh"])
        if self.input_size not in ((192, 256), (288, 384)):
            raise ValueError("Unsupported standard whole-body input profile")
        width, height = self.input_size
        self.model = TensorRTModel(
            build["engine_path"],
            build["engine_sha256"],
            {"input": [1, 3, height, width]},
        )
        self.provider = self.model.provider

    def infer(self, image, box, joint_threshold=0.2):
        inputs, matrix = prepare_pose(image, box, self.geometry, self.input_size)
        raw = self.model.infer(inputs)
        if raw["simcc_x"].shape != (1, 133, self.input_size[0] * 2) or raw[
            "simcc_y"
        ].shape != (1, 133, self.input_size[1] * 2):
            raise ValueError("Unexpected RTMPose whole-body output")
        bins, scores = self.decoder.get_simcc_maximum(raw["simcc_x"], raw["simcc_y"])
        points = cv2.transform(
            (bins / 2).astype(np.float32), cv2.invertAffineTransform(matrix)
        )[0]
        scores = scores[0]
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
                    "coordinate_space": "source_rgb_pixels",
                    "coordinate_dimensions": 2,
                    "z_kind": "not_predicted",
                    "box_xyxy": np.r_[keypoints.min(0), keypoints.max(0)].tolist(),
                }
            )
        return {
            "keypoints": points.tolist(),
            "scores": scores.tolist(),
            "valid": (scores >= joint_threshold).tolist(),
            "layout": "COCO-WholeBody133",
            "coordinate_space": "source_rgb_pixels",
            "coordinate_dimensions": 2,
            "z_kind": "not_predicted; metric target range is from aligned D435i depth",
            "score_kind": "simcc_peak",
            "hands": hands,
            "affine_source_to_input": matrix.tolist(),
        }

    def close(self):
        self.model.close()
