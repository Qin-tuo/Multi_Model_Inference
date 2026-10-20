# SPDX-License-Identifier: Apache-2.0
"""Pinned reference images for retained face/head/gaze validation."""

import json
import cv2
import numpy as np
from project_paths import ROOT
from app.face_trt import TensorRTFace
from deploy.validation_inputs import input_path


def fixtures(face_root):
    face_root = input_path(face_root, ROOT)
    build = json.loads((face_root / "face/build.json").read_text())
    build["engine_path"] = str(input_path(ROOT / build["engine_path"], ROOT))
    passed = json.loads((face_root / "parity/face/report.json").read_text())
    if (
        passed["status"] != "PARITY_PASS_LIMITED_FIXTURES"
        or passed["engine_sha256"] != build["engine_sha256"]
    ):
        raise ValueError("Face detector reference gate is not verified")
    detector = TensorRTFace(build)
    result = []
    try:
        for name, relative, factor in (
            ("public-person", "05_runtime/datasets/b0-sw-v1/public-original.png", 1.0),
            (
                "public-person-dimmed",
                "05_runtime/datasets/b0-sw-v1/public-original.png",
                0.7,
            ),
            (
                "upstream-crowd-face",
                "05_runtime/datasets/b1-public/expression-selfie.jpg",
                1.0,
            ),
        ):
            path = ROOT / relative
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            image = (
                np.rint(image.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)
            )
            faces = detector.infer(image)
            if not faces:
                raise ValueError("No face in positive fixture " + name)
            face = max(
                faces,
                key=lambda f: (f["box_xyxy"][2] - f["box_xyxy"][0])
                * (f["box_xyxy"][3] - f["box_xyxy"][1]),
            )
            result.append(
                (
                    name,
                    image,
                    face,
                    {
                        "source": relative,
                        "source_sha256": runtime.sha256(path),
                        "brightness": factor,
                        "detector_engine_sha256": build["engine_sha256"],
                        "face": face,
                    },
                )
            )
    finally:
        detector.close()
    return result
