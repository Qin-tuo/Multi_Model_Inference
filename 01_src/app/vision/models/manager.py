# SPDX-License-Identifier: Apache-2.0
"""统一持有当前视觉引擎，检查模型契约并采集有界的逐模型调用统计。"""

from collections import deque
from importlib.metadata import version
import json
import hashlib
import threading
import time

import numpy as np
import psutil

from project_paths import ROOT
from app.vision.models.appearance import AppearanceFeatures
from app.vision.models.face_tasks import ArcFaceIdentity, FaceExpression
from app.vision.models.face_trt import TensorRTFace
from app.vision.models.head_gaze import HeadPose, GazeLandmarks, Gaze
from app.vision.models.segmentation import DetectionSegmentation
from app.vision.models.wholebody_pose2d import WholeBodyPose2D

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
    """对影响输入输出解释的清单字段生成稳定摘要，防止权重与适配器契约错配。"""
    keys = (
        "sha256",
        "preprocessing",
        "output_semantics",
        "class_names",
        "decoder_source",
        "prepost_source",
    )
    values = {k: entry.get(k) for k in keys}
    # Adapters also lock shape, dimensions and matching policy so a model
    # replacement cannot silently reuse another model's semantics.
    if "adapter" in entry:
        for key in (
            "adapter",
            "input_size_wh",
            "input_shapes",
            "output_names",
            "embedding_space_suffix",
            "gallery_policy",
        ):
            values[key] = entry.get(key)
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class VisionModels:
    """视觉引擎集合的资源所有者，维护清单契约和有界调用统计。"""

    def __init__(
        self,
        manifest=None,
        threshold=0.25,
        iou=0.45,
        max_detections=10,
    ):
        """读取引擎与源模型清单，初始化模型容器、统计锁和检测参数，不立即加载 GPU 资源。"""
        if manifest is None:
            from app.config import Config

            manifest = Config.load().perception.model_manifest
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

    def load(self, roles=None):
        """校验运行依赖版本、模型摘要和语义契约后逐个加载所选适配器。

        加载前检查可用内存；部分加载失败时统一关闭成功创建的模型，禁止重复加载覆盖所有权。
        """
        if self.models:
            raise RuntimeError("Visual models are already loaded")
        roles = tuple(ROLES if roles is None else roles)
        if not roles or len(set(roles)) != len(roles) or set(roles) - set(ROLES):
            raise ValueError("Unknown, duplicate or empty visual model selection")
        for name, expected in self.manifest["runtime_versions"].items():
            if version(name) != expected:
                raise RuntimeError(
                    "Runtime dependency differs from visual runtime contract: " + name
                )
        with self.lock:
            self.stats.clear()
            for sample in self.samples.values():
                sample.clear()
        try:
            for role in roles:
                spec = dict(self.manifest["models"][role])
                spec["engine_path"] = str(ROOT / spec["engine_path"])
                if spec["onnx_sha256"] != self.entries[role]["sha256"]:
                    raise ValueError("Visual model/source lock mismatch: " + role)
                if spec["candidate_contract_sha256"] != contract_hash(
                    self.entries[role]
                ):
                    raise ValueError("Visual model semantic contract changed: " + role)
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
                    if self.entries[role].get("adapter") != "rtmpose2d":
                        raise ValueError("Unsupported current whole-body adapter")
                    model = WholeBodyPose2D(spec, self.entries[role])
                elif role in ("person_reid", "object_embedding"):
                    model = AppearanceFeatures(spec, role)
                elif role == "face_identity":
                    if self.entries[role].get("adapter") != "arcface_mobilenet_v1":
                        raise ValueError("Unsupported current face identity adapter")
                    model = ArcFaceIdentity(spec)
                elif role == "expression":
                    model = FaceExpression(spec, self.entries[role])
                else:
                    model = {
                        "head_pose": HeadPose,
                        "gaze_landmarks": GazeLandmarks,
                        "gaze": Gaze,
                    }[role](spec)
                self.models[role] = model
                stats = {
                    "provider": model.provider,
                    "model_name": self.entries[role].get("name", role),
                    "calls": 0,
                    "errors": 0,
                    "last_error": None,
                    "last_ms": None,
                    "load_ms": (time.monotonic() - tick) * 1000,
                    "io_bytes_each_host_device": model.model.io_bytes,
                    "available_ram_after_load_bytes": psutil.virtual_memory().available,
                }
                with self.lock:
                    self.stats[role] = stats
        except BaseException:
            self.close()
            raise

    def run(self, role, *args, **kwargs):
        """调用指定模型并记录成功耗时或异常信息，保留最近 120 次成功耗时样本。"""
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
        """在统计锁内生成各模型调用数、最近耗时和 P50/P95 快照。"""
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
        """按加载的逆序尝试关闭全部模型，仅移除已释放实例，汇总仍未释放的异常。"""
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
