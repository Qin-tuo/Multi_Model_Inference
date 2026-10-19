# SPDX-License-Identifier: Apache-2.0
"""YuNet TensorRT inference with the OpenCV 4.11.0 decoding contract.

Decoder translated from OpenCV modules/objdetect/src/face_detect.cpp,
commit 31b0eeea0b44b370fd0712312df4214d4ae1b158 (Apache-2.0).
The original model is MIT licensed. OpenCV is used only for image transforms
and NMS here; neural-network execution always uses TensorRT.
"""

import cv2
import numpy as np

from app.tensorrt_runner import TensorRTModel


def prepare_face(image):
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or not image.size
    ):
        raise ValueError("YuNet expects a nonempty uint8 BGR image")
    height, width = image.shape[:2]
    scale = min(1.0, 640 / width, 640 / height)
    rw, rh = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, 0, 640 - rh, 0, 640 - rw, cv2.BORDER_CONSTANT, value=0
    )
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
    return {"input": tensor}, {
        "source_hw": [height, width],
        "scale_xy": [rw / width, rh / height],
        "unpad_hw": [rh, rw],
    }


def decode_yunet(outputs, threshold=0.5, nms_threshold=0.3, top_k=5000):
    expected = {
        f"{kind}_{stride}"
        for kind in ("cls", "obj", "bbox", "kps")
        for stride in (8, 16, 32)
    }
    if set(outputs) != expected:
        raise ValueError("YuNet output names differ from the reviewed contract")
    rows = []
    for stride in (8, 16, 32):
        cols = 640 // stride
        for kind, channels in (("cls", 1), ("obj", 1), ("bbox", 4), ("kps", 10)):
            if outputs[f"{kind}_{stride}"].shape != (1, cols * cols, channels):
                raise ValueError(
                    "YuNet output dimensions differ from the reviewed contract"
                )
        score = np.sqrt(
            np.clip(outputs[f"cls_{stride}"][0, :, 0], 0, 1)
            * np.clip(outputs[f"obj_{stride}"][0, :, 0], 0, 1)
        )
        indexes = np.flatnonzero(score >= threshold)
        if not indexes.size:
            continue
        grid = np.column_stack((indexes % cols, indexes // cols)).astype(np.float32)
        box = outputs[f"bbox_{stride}"][0, indexes]
        size = np.exp(box[:, 2:]) * stride
        origin = (grid + box[:, :2]) * stride - size / 2
        landmarks = (
            outputs[f"kps_{stride}"][0, indexes].reshape(-1, 5, 2) + grid[:, None]
        ) * stride
        rows.append(
            np.column_stack((origin, size, landmarks.reshape(-1, 10), score[indexes]))
        )
    if not rows:
        return np.empty((0, 15), np.float32)
    faces = np.concatenate(rows).astype(np.float32)
    if not np.isfinite(faces).all():
        raise ValueError("Non-finite decoded YuNet result")
    if len(faces) > 1:
        # Match OpenCV's C++ Rect2i conversion, including truncation toward zero.
        keep = cv2.dnn.NMSBoxes(
            faces[:, :4].astype(np.int32).tolist(),
            faces[:, 14].tolist(),
            threshold,
            nms_threshold,
            top_k=top_k,
        )
        faces = faces[np.asarray(keep, dtype=np.int64).reshape(-1)]
    return faces


class TensorRTFace:
    def __init__(self, build, threshold=0.5, nms_threshold=0.3, max_faces=4):
        if (
            not 0 < threshold <= 1
            or not 0 < nms_threshold <= 1
            or not 1 <= max_faces <= 10
        ):
            raise ValueError("Invalid face thresholds or capacity")
        if build["input_shapes"] != {"input": [1, 3, 640, 640]}:
            raise ValueError("Unexpected YuNet engine shape")
        self.model = TensorRTModel(
            build["engine_path"], build["engine_sha256"], build["input_shapes"]
        )
        self.threshold, self.nms_threshold, self.max_faces = (
            threshold,
            nms_threshold,
            max_faces,
        )
        self.provider = "YuNet " + self.model.provider

    def infer(self, image):
        inputs, transform = prepare_face(image)
        faces = decode_yunet(
            self.model.infer(inputs), self.threshold, self.nms_threshold
        )
        sx, sy = transform["scale_xy"]
        faces[:, :14:2] /= sx
        faces[:, 1:14:2] /= sy
        height, width = transform["source_hw"]
        result = []
        for row in sorted(faces, key=lambda f: -float(f[14])):
            x, y, w, h = row[:4]
            if (
                w <= 0
                or h <= 0
                or min(x + w, width) <= max(x, 0)
                or min(y + h, height) <= max(y, 0)
            ):
                continue
            result.append(
                {
                    "box_xyxy": [float(x), float(y), float(x + w), float(y + h)],
                    "score": float(row[14]),
                    "landmarks": row[4:14].reshape(5, 2).tolist(),
                    "landmark_order": "right_eye,left_eye,nose_tip,right_mouth_corner,left_mouth_corner",
                    "coordinate_space": "source_rgb_pixels",
                }
            )
            if len(result) == self.max_faces:
                break
        return result

    def close(self):
        self.model.close()
