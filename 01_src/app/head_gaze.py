# SPDX-License-Identifier: Apache-2.0
"""TensorRT adapters for the official OMZ head/landmark/gaze cascade.

Eye-box sizing, roll alignment/compensation and gaze angle formulas follow
Open Model Zoo 2c9b5c563ab3a202d085a42e2bbfc48f77961f4b (Apache-2.0).
The gaze angles use the demo convention, not a calibrated world-space ray.
"""

import cv2
import numpy as np

from app.tensorrt_runner import TensorRTModel
from app.vision_roi import crop_roi


def image_blob(image):
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or not image.size
    ):
        raise ValueError("Expected a nonempty uint8 BGR attribute ROI")
    image = cv2.resize(image, (60, 60), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)


def prepare_face_attribute(image, box):
    crop, roi = crop_roi(image, box)
    return {"data": image_blob(crop)}, roi


def landmark_pixels(values, roi):
    if values.size != 70 or not np.isfinite(values).all():
        raise ValueError("Expected 35 finite face landmarks")
    x1, y1, x2, y2 = roi
    points = values.reshape(35, 2) * np.array(
        [x2 - x1, y2 - y1], np.float32
    ) + np.array([x1, y1], np.float32)
    # Match OMZ simplePostprocess: integer source-image landmarks.
    return points.astype(np.int32)


def prepare_gaze(image, landmarks, angles):
    points = np.asarray(landmarks)
    angles = np.asarray(angles, np.float32)
    if (
        points.shape != (35, 2)
        or not np.isfinite(points).all()
        or angles.shape != (3,)
        or not np.isfinite(angles).all()
    ):
        raise ValueError(
            "Gaze requires same-face 35 landmarks and yaw/pitch/roll degrees"
        )
    roll = float(angles[2])
    inputs = {"head_pose_angles": np.array([[angles[0], angles[1], 0]], np.float32)}
    boxes = {}
    height, width = image.shape[:2]
    for name, pair in (
        ("left_eye_image", points[:2]),
        ("right_eye_image", points[2:4]),
    ):
        side = int(1.8 * np.linalg.norm(pair[0] - pair[1]))
        center = ((pair[0] + pair[1]) / 2).astype(np.int32)
        x, y = int(center[0]) - side // 2, int(center[1]) - side // 2
        if side < 4 or x < 0 or y < 0 or x + side > width or y + side > height:
            raise ValueError("Eye ROI is too small or falls outside the source image")
        crop = image[y : y + side, x : x + side]
        rotation = cv2.getRotationMatrix2D((side // 2, side // 2), roll, 1)
        rotated = cv2.warpAffine(
            crop,
            rotation,
            (side, side),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        inputs[name] = image_blob(rotated)
        boxes[name] = [x, y, x + side, y + side]
    return inputs, {
        "eye_boxes_xyxy": boxes,
        "roll_alignment_deg": roll,
        "eye_openness_estimated": False,
    }


def decode_gaze(vector, roll):
    vector = np.asarray(vector, np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if (
        vector.shape != (3,)
        or not np.isfinite(vector).all()
        or not np.isfinite(norm)
        or norm <= 1e-8
        or not np.isfinite(roll)
    ):
        raise ValueError("Invalid gaze vector or roll")
    x, y, z = vector / norm
    c, s = np.cos(np.deg2rad(roll)), np.sin(np.deg2rad(roll))
    vector = np.array([x * c + y * s, -x * s + y * c, z], np.float32)
    # OMZ demo: yaw = 90 + atan2(z,x), pitch = asin(y/||v||).
    yaw = (90 + np.degrees(np.arctan2(vector[2], vector[0])) + 180) % 360 - 180
    pitch = np.degrees(np.arcsin(np.clip(vector[1] / np.linalg.norm(vector), -1, 1)))
    return {
        "gaze_vector": vector.tolist(),
        "gaze_yaw_deg": float(yaw),
        "gaze_pitch_deg": float(pitch),
        "angle_convention": "OMZ demo degrees, yaw wrapped to [-180,180); zero yaw follows model -Z",
        "coordinate_frame": "OMZ model frame; not calibrated camera/world coordinates",
    }


class HeadPose:
    def __init__(self, build):
        self.model = TensorRTModel(
            build["engine_path"], build["engine_sha256"], {"data": [1, 3, 60, 60]}
        )
        self.provider = self.model.provider

    def infer(self, image, box):
        inputs, roi = prepare_face_attribute(image, box)
        out = self.model.infer(inputs)
        values = [float(out[name].item()) for name in ("fc_y", "fc_p", "fc_r")]
        return {
            "angles_degrees": values,
            "angle_order": ["yaw", "pitch", "roll"],
            "roi_xyxy": roi,
            "within_supported_range": all(
                abs(v) <= limit for v, limit in zip(values, (90, 70, 70))
            ),
        }

    def close(self):
        self.model.close()


class GazeLandmarks:
    def __init__(self, build):
        self.model = TensorRTModel(
            build["engine_path"], build["engine_sha256"], {"data": [1, 3, 60, 60]}
        )
        self.provider = self.model.provider
        self.output_name = next(iter(self.model.output_specs))

    def infer(self, image, box):
        inputs, roi = prepare_face_attribute(image, box)
        return {
            "landmarks": landmark_pixels(
                self.model.infer(inputs)[self.output_name], roi
            ),
            "roi_xyxy": roi,
            "order": "OMZ35",
        }

    def close(self):
        self.model.close()


class Gaze:
    def __init__(self, build):
        self.model = TensorRTModel(
            build["engine_path"], build["engine_sha256"], build["input_shapes"]
        )
        self.provider = self.model.provider
        self.output_name = next(iter(self.model.output_specs))

    def infer(self, image, landmarks, angles):
        inputs, transform = prepare_gaze(image, landmarks, angles)
        return {
            **decode_gaze(self.model.infer(inputs)[self.output_name], angles[2]),
            "transform": transform,
        }

    def close(self):
        self.model.close()
