# SPDX-License-Identifier: AGPL-3.0-only
"""YOLOv8n-seg TensorRT adapter with bounded native-resolution masks.

LetterBox, box scaling and mask decoding follow Ultralytics 8.4.175,
commit 69c95d1add13a232382e53b3aa5094ec0c7738db (AGPL-3.0).
OpenCV/NumPy implement those image/geometry operations; the neural network
runs only through the shared TensorRT runner. Masks remain internal arrays.
"""

import math

import cv2
import numpy as np

from app.tensorrt_runner import TensorRTModel


def prepare_segmentation(image):
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or not image.size
    ):
        raise ValueError("Segmentation expects a nonempty uint8 BGR image")
    height, width = image.shape[:2]
    if height * width > 1920 * 1080:
        raise ValueError("Source image exceeds the reviewed segmentation budget")
    gain = min(640 / height, 640 / width)
    rw, rh = round(width * gain), round(height * gain)
    left, top = round((640 - rw) / 2 - 0.1), round((640 - rh) / 2 - 0.1)
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized,
        top,
        640 - rh - top,
        left,
        640 - rw - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = (
        np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    )
    return {"images": tensor}, {
        "source_hw": [height, width],
        "resized_hw": [rh, rw],
        "scale_xy": [rw / width, rh / height],
        "pad_xy": [left, top],
    }


def native_masks(prototypes, coefficients, boxes, shape, pixel_budget):
    height, width = shape
    count, channels = coefficients.shape
    if count * height * width > pixel_budget:
        raise ValueError("Instance masks exceed the configured pixel budget")
    _, mh, mw = prototypes.shape
    logits = (coefficients @ prototypes.reshape(channels, -1)).reshape(count, mh, mw)
    # Upstream selects 1/2/4x prototype upsampling so odd input padding can
    # be removed at whole-pixel boundaries before resizing to the source.
    gain = min(4 * mh / height, 4 * mw / width)
    pad = (4 * mh - round(height * gain)) | (4 * mw - round(width * gain))
    factor = 4 // math.gcd(4, pad | pad // 2)
    result = np.empty((count, height, width), bool)
    x = np.arange(width)[None, :]
    y = np.arange(height)[:, None]
    for index, mask in enumerate(logits):
        if factor > 1:
            mask = cv2.resize(
                mask, (mw * factor, mh * factor), interpolation=cv2.INTER_LINEAR
            )
        ih, iw = mask.shape
        gain = min(ih / height, iw / width)
        left = round((iw - round(width * gain)) / 2 - 0.1)
        top = round((ih - round(height * gain)) / 2 - 0.1)
        crop = mask[top : top + round(height * gain), left : left + round(width * gain)]
        upsampled = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
        x1, y1, x2, y2 = boxes[index]
        result[index] = (upsampled > 0) & (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
    result.flags.writeable = False
    return result


def decode_segmentation(
    outputs,
    transform,
    *,
    threshold=0.25,
    iou=0.45,
    max_detections=10,
    mask_pixel_budget=1920 * 1080 * 10,
):
    if (
        set(outputs) != {"output0", "output1"}
        or outputs["output0"].shape != (1, 116, 8400)
        or outputs["output1"].shape != (1, 32, 160, 160)
    ):
        raise ValueError("Segmentation outputs differ from the reviewed contract")
    rows = outputs["output0"][0].T
    classes = rows[:, 4:84].argmax(axis=1)
    scores = rows[np.arange(len(rows)), 4 + classes]
    selected = scores > threshold
    rows, classes, scores = rows[selected], classes[selected], scores[selected]
    height, width = transform["source_hw"]
    if not len(rows):
        return np.empty((0, 6), np.float32), np.empty((0, height, width), bool)
    xywh = rows[:, :4].copy()
    xywh[:, :2] -= xywh[:, 2:] / 2
    keep = np.asarray(
        cv2.dnn.NMSBoxesBatched(
            xywh.tolist(), scores.tolist(), classes.tolist(), threshold, iou
        ),
        np.int64,
    ).reshape(-1)[:max_detections]
    boxes = xywh[keep].copy()
    boxes[:, 2:] += boxes[:, :2]
    sx, sy = transform["scale_xy"]
    left, top = transform["pad_xy"]
    boxes[:, 0::2] = np.clip((boxes[:, 0::2] - left) / sx, 0, width)
    boxes[:, 1::2] = np.clip((boxes[:, 1::2] - top) / sy, 0, height)
    masks = native_masks(
        outputs["output1"][0],
        rows[keep, 84:],
        boxes,
        (height, width),
        mask_pixel_budget,
    )
    nonempty = masks.any(axis=(1, 2))
    detections = np.column_stack((boxes, scores[keep], classes[keep])).astype(
        np.float32
    )
    return detections[nonempty], masks[nonempty]


class DetectionSegmentation:
    def __init__(self, build, entry, threshold=0.25, iou=0.45, max_detections=10):
        if not 0 < threshold < 1 or not 0 < iou < 1 or not 1 <= max_detections <= 10:
            raise ValueError("Invalid detection thresholds or capacity")
        classes = entry["class_names"]
        if (
            not isinstance(classes, list)
            or len(classes) != 80
            or any(not isinstance(name, str) or not name for name in classes)
        ):
            raise ValueError("Expected the pinned 80-class COCO label mapping")
        self.model = TensorRTModel(
            build["engine_path"], build["engine_sha256"], {"images": [1, 3, 640, 640]}
        )
        self.classes = classes
        self.threshold, self.iou, self.max_detections = threshold, iou, max_detections
        self.provider = self.model.provider

    def infer(self, image):
        inputs, transform = prepare_segmentation(image)
        detections, masks = decode_segmentation(
            self.model.infer(inputs),
            transform,
            threshold=self.threshold,
            iou=self.iou,
            max_detections=self.max_detections,
        )
        result = []
        for row, mask in zip(detections, masks):
            mask.flags.writeable = False
            result.append(
                {
                    "box_xyxy": row[:4].tolist(),
                    "score": float(row[4]),
                    "class_id": int(row[5]),
                    "class_name": self.classes[int(row[5])],
                    "mask": mask,
                    "coordinate_space": "source_rgb_pixels",
                }
            )
        return result

    def close(self):
        self.model.close()
