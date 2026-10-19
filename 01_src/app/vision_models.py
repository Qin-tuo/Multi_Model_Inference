# SPDX-License-Identifier: Apache-2.0
"""One owner for the selected in-process visual models and bounded call metrics."""

from collections import deque
from importlib.metadata import version
import json
import hashlib
import threading
import time

import numpy as np
import psutil

from project_paths import ROOT
from app.appearance import AppearanceFeatures
from app.face_tasks import FaceIdentity, FaceExpression
from app.face_trt import TensorRTFace
from app.head_gaze import HeadPose, GazeLandmarks, Gaze
from app.segmentation import DetectionSegmentation
from app.wholebody_pose import WholeBodyPose

ROLES = (
    "detection_segmentation",
    "face",
    "body_hand_pose",
    "person_reid",
    "head_pose",
    "gaze_landmarks",
    "gaze",
    "face_identity",
    "expression",
    "object_embedding",
)


def contract_hash(entry):
    keys = (
        "sha256",
        "preprocessing",
        "output_semantics",
        "class_names",
        "decoder_source",
        "prepost_source",
    )
    return hashlib.sha256(
        json.dumps(
            {k: entry.get(k) for k in keys}, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


class VisionModels:
    def __init__(
        self,
        manifest="03_models/vision-b1-runtime.json",
        threshold=0.25,
        iou=0.45,
        max_detections=10,
    ):
        self.manifest = json.loads((ROOT / manifest).read_text())
        self.entries = json.loads(
            (ROOT / self.manifest["source_manifest"]).read_text()
        )["models"]
        self.models = {}
        self.lock = threading.Lock()
        self.stats = {}
        self.samples = {role: deque(maxlen=120) for role in ROLES}
        self.detection_options = {
            "threshold": threshold,
            "iou": iou,
            "max_detections": max_detections,
        }

    def load(self):
        if self.models:
            raise RuntimeError("Visual models are already loaded")
        for name, expected in self.manifest["runtime_versions"].items():
            if version(name) != expected:
                raise RuntimeError(
                    "Runtime dependency differs from B1 contract: " + name
                )
        try:
            for role in ROLES:
                spec = dict(self.manifest["models"][role])
                spec["engine_path"] = str(ROOT / spec["engine_path"])
                if spec["onnx_sha256"] != self.entries[role]["sha256"]:
                    raise ValueError("B1 model/source lock mismatch: " + role)
                if spec["candidate_contract_sha256"] != contract_hash(
                    self.entries[role]
                ):
                    raise ValueError("B1 model semantic contract changed: " + role)
                available = psutil.virtual_memory().available
                if available < 2 * spec["engine_bytes"] + 256 * 1024**2:
                    raise RuntimeError("Insufficient available RAM to load " + role)
                tick = time.monotonic()
                if role == "detection_segmentation":
                    model = DetectionSegmentation(
                        spec, self.entries[role], **self.detection_options
                    )
                elif role == "face":
                    model = TensorRTFace(spec)
                elif role == "body_hand_pose":
                    model = WholeBodyPose(spec, self.entries[role])
                elif role in ("person_reid", "object_embedding"):
                    model = AppearanceFeatures(spec, role)
                elif role == "face_identity":
                    model = FaceIdentity(spec)
                elif role == "expression":
                    model = FaceExpression(spec, self.entries[role])
                else:
                    model = {
                        "head_pose": HeadPose,
                        "gaze_landmarks": GazeLandmarks,
                        "gaze": Gaze,
                    }[role](spec)
                self.models[role] = model
                self.stats[role] = {
                    "provider": model.provider,
                    "calls": 0,
                    "errors": 0,
                    "last_error": None,
                    "last_ms": None,
                    "load_ms": (time.monotonic() - tick) * 1000,
                    "io_bytes_each_host_device": model.model.io_bytes,
                    "available_ram_after_load_bytes": psutil.virtual_memory().available,
                }
        except BaseException:
            self.close()
            raise

    def run(self, role, *args, **kwargs):
        tick = time.monotonic()
        try:
            result = self.models[role].infer(*args, **kwargs)
        except Exception as exc:
            with self.lock:
                self.stats[role]["errors"] += 1
                self.stats[role]["last_error"] = str(exc)
            raise
        elapsed = (time.monotonic() - tick) * 1000
        with self.lock:
            self.stats[role]["calls"] += 1
            self.stats[role]["last_ms"] = elapsed
            self.stats[role]["last_error"] = None
            self.samples[role].append(elapsed)
        return result

    def snapshot(self):
        with self.lock:
            return {
                role: {
                    **row,
                    "p50_ms": float(np.median(self.samples[role]))
                    if self.samples[role]
                    else None,
                    "p95_ms": float(np.quantile(self.samples[role], 0.95))
                    if self.samples[role]
                    else None,
                    "timing_scope": "model adapter wall time including preprocessing/inference/postprocessing",
                }
                for role, row in self.stats.items()
            }

    def close(self):
        errors = []
        for role, model in reversed(list(self.models.items())):
            try:
                model.close()
            except Exception as exc:
                errors.append(exc)
            else:
                self.models.pop(role)
        if errors:
            raise ExceptionGroup("Visual model release failed", errors)
