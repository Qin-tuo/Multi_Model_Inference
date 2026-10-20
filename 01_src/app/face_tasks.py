# SPDX-License-Identifier: Apache-2.0
"""TensorRT face features/expression with the upstream alignment contracts.

SFace alignment is a direct vectorized translation of OpenCV 4.11.0
face_recognize.cpp (31b0eeea0b44b370fd0712312df4214d4ae1b158, Apache-2.0).
Expression alignment reuses the pinned OpenCV Zoo FaceAlignment class.
CPU operations here are geometric/image transforms, never neural inference.
"""

import cv2
import numpy as np

from app.model_artifacts import load_source
from app.tensorrt_runner import TensorRTModel
from project_paths import ROOT

EXPRESSION_CLASSES = (
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
)
FACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    np.float32,
)


def checked_landmarks(landmarks):
    points = np.asarray(landmarks, np.float32)
    if (
        points.shape != (5, 2)
        or not np.isfinite(points).all()
        or np.linalg.norm(points[0] - points[1]) < 4
    ):
        raise ValueError("Five finite, non-degenerate face landmarks are required")
    return points


def sface_transform(landmarks):
    points = checked_landmarks(landmarks)
    source_mean = points.sum(axis=0, dtype=np.float32) / np.float32(5)
    target_mean = np.array([56.0262, 71.9008], np.float32)
    source = points - source_mean
    target = FACE_TEMPLATE - target_mean
    # Preserve the C++ contract: float products accumulated into double sums.
    covariance = (target[:, :, None] * source[:, None, :]).sum(
        axis=0, dtype=np.float64
    ) / 5
    singular, u, vt = cv2.SVDecomp(covariance)
    if singular[1, 0] <= singular[0, 0] * 1e-8:
        raise ValueError("Degenerate face alignment covariance")
    signs = np.array([1.0, -1.0 if np.linalg.det(covariance) < 0 else 1.0])
    rotation = (u * signs[None]) @ vt
    variance = (source * source).sum(dtype=np.float64) / 5
    scale = float(singular[:, 0] @ signs) / variance
    translation = target_mean - scale * (rotation @ source_mean)
    return np.column_stack((rotation * scale, translation))


def aligned_bgr(image, matrix):
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or not image.size
    ):
        raise ValueError("Face preprocessing expects a uint8 BGR image")
    return cv2.warpAffine(image, matrix, (112, 112), flags=cv2.INTER_LINEAR)


def face_tensor(crop, *, expression=False):
    if crop.shape != (112, 112, 3) or crop.dtype != np.uint8:
        raise ValueError("Expected an aligned uint8 112x112 BGR face")
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    if expression:
        rgb = (rgb / 255.0 - 0.5) / 0.5
    return {"data": np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])}


def normalize_embedding(value, dimensions=512):
    vector = np.asarray(value, np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if (
        vector.shape != (dimensions,)
        or not np.isfinite(vector).all()
        or not np.isfinite(norm)
        or norm <= 1e-8
    ):
        raise ValueError("Invalid face embedding")
    result = vector / norm
    result.flags.writeable = False
    return result


def expression_scores(logits):
    values = np.asarray(logits, np.float32).reshape(-1)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("Invalid expression logits")
    weights = np.exp(values - values.max())
    return dict(zip(EXPRESSION_CLASSES, (weights / weights.sum()).tolist()))


def expression_alignment(entry):
    source = entry["prepost_source"]["files"]["facial_fer_model.py"]
    module = load_source(
        str(ROOT / source["path"]), source["sha256"], "assistant_expression_source"
    )
    return module.FaceAlignment(reflective=False)


class ArcFaceIdentity:
    """Original MobileNetV1 ArcFace checkpoint, with explicit RGB normalization."""

    def __init__(self, build):
        self.model = TensorRTModel(
            build["engine_path"], build["engine_sha256"], {"images": [1, 3, 112, 112]}
        )
        self.provider = self.model.provider

    def infer(self, image, landmarks):
        matrix = sface_transform(landmarks)
        crop = aligned_bgr(image, matrix)
        # Both upstream FER and ArcFace use (RGB/255 - .5)/.5. Alignment
        # remains the existing, independently checked five-point frontend.
        values = face_tensor(crop, expression=True)["data"]
        vector = normalize_embedding(
            self.model.infer({"images": values})["embedding"], dimensions=512
        )
        return {
            "embedding": vector,
            "alignment_source_to_crop": matrix.tolist(),
            "space": "ArcFaceMobileNetV1_512_L2",
        }

    def close(self):
        self.model.close()


class FaceExpression:
    def __init__(self, build, entry):
        self.aligner = expression_alignment(entry)
        self.model = TensorRTModel(
            build["engine_path"], build["engine_sha256"], {"data": [1, 3, 112, 112]}
        )
        self.provider = self.model.provider

    def infer(self, image, landmarks):
        points = checked_landmarks(landmarks)
        matrix, _ = self.aligner.get_similarity_transform_for_cv2(
            points, self.aligner._std_points
        )
        crop = aligned_bgr(image, matrix)
        scores = expression_scores(
            self.model.infer(face_tensor(crop, expression=True))["label"]
        )
        return {
            "scores": scores,
            "label": max(scores, key=scores.get),
            "alignment_source_to_crop": matrix.tolist(),
            "score_kind": "softmax model scores, not calibrated emotional-state probabilities",
        }

    def close(self):
        self.model.close()
