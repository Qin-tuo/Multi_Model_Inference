# SPDX-License-Identifier: Apache-2.0
"""RTMW 二维全身姿态适配器：输出原图人体和双手坐标，米制距离由相机深度提供。"""

import cv2
import numpy as np

from app.model_artifacts import load_source
from app.tensorrt_runner import TensorRTModel
from app.pose_geometry import geometry_helper, prepare_pose
from project_paths import ROOT


class WholeBodyPose2D:
    """组合几何裁剪、SimCC 解码和 TensorRT 引擎的全身姿态适配器。"""

    def __init__(self, build, entry):
        """校验并加载几何与 SimCC 辅助代码，按清单中的宽、高创建固定形状姿态引擎。"""
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
        """将人体框裁剪后送入引擎，解码 133 个 SimCC 关键点并反变换到原图。

        返回关键点、峰值分数、阈值有效性和两只手各 21 点；分数不是概率，结果不包含模型预测的 z。
        """
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
        """释放姿态引擎持有的执行资源。"""
        self.model.close()
