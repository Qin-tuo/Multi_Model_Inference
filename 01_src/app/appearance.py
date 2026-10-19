# SPDX-License-Identifier: Apache-2.0
"""TensorRT appearance features with upstream Pillow/Torchvision transforms.

OSNet follows Torchreid's evaluation transform. CLIP follows OpenAI CLIP's
image preprocessing. PyTorch/text encoders are not runtime dependencies.
"""

import cv2
import numpy as np
from PIL import Image

from app.tensorrt_runner import TensorRTModel
from app.vision_roi import crop_roi


def prepare_appearance(image, box, role):
    crop, roi = crop_roi(image, box)
    photo = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if role == "person_reid":
        photo = photo.resize((128, 256), Image.Resampling.BILINEAR)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        transform = {"resize_hw": [256, 128], "center_crop": None}
    elif role == "object_embedding":
        width, height = photo.size
        if width <= height:
            shape = (224, int(224 * height / width))
        else:
            shape = (int(224 * width / height), 224)
        if shape[0] * shape[1] > 224 * 224 * 16:
            raise ValueError("CLIP resize exceeds the reviewed aspect-ratio budget")
        photo = photo.resize(shape, Image.Resampling.BICUBIC)
        left, top = int(round((shape[0] - 224) / 2)), int(round((shape[1] - 224) / 2))
        photo = photo.crop((left, top, left + 224, top + 224))
        mean, std = (
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        transform = {
            "resize_hw": [shape[1], shape[0]],
            "center_crop_xyxy": [left, top, left + 224, top + 224],
        }
    else:
        raise ValueError("Unknown appearance feature space")
    values = np.asarray(photo).astype(np.float32) / 255.0
    values = (values - np.array(mean, np.float32)) / np.array(std, np.float32)
    return {"images": np.ascontiguousarray(values.transpose(2, 0, 1)[None])}, {
        "roi_xyxy": roi,
        **transform,
    }


def normalized_features(values):
    values = np.asarray(values, np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if (
        values.shape != (512,)
        or not np.isfinite(values).all()
        or not np.isfinite(norm)
        or norm <= 1e-8
    ):
        raise ValueError("Invalid 512D appearance features")
    result = values / norm
    result.flags.writeable = False
    return result


class AppearanceFeatures:
    def __init__(self, build, role):
        if role not in ("person_reid", "object_embedding"):
            raise ValueError("Unknown appearance role")
        self.role = role
        shape = [1, 3, 256, 128] if role == "person_reid" else [1, 3, 224, 224]
        self.model = TensorRTModel(
            build["engine_path"], build["engine_sha256"], {"images": shape}
        )
        self.space = role + ":" + build["onnx_sha256"]
        self.provider = self.model.provider

    def infer(self, image, box):
        inputs, transform = prepare_appearance(image, box, self.role)
        vector = normalized_features(self.model.infer(inputs)["embedding"])
        return {
            "embedding": vector,
            "embedding_space": self.space,
            "transform": transform,
        }

    def close(self):
        self.model.close()
