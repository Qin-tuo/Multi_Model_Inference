# SPDX-License-Identifier: Apache-2.0
"""为图执行提供经过校验的引擎元数据，执行上下文由原生运行时拥有。"""

import hashlib
import math
from pathlib import Path

import numpy as np


class EngineDescription:
    """只描述引擎契约，不创建 CUDA 上下文、设备缓冲区或额外引擎实例。

    模型适配器在图模式下只使用 prepare/decode；同步 infer 接口明确拒绝执行。
    """

    def __init__(self, build, tensorrt_version):
        """校验工件与静态张量描述，保留 TensorRT 实际的输入输出顺序。"""
        import tensorrt as trt

        if trt.__version__ != tensorrt_version:
            raise RuntimeError("TensorRT version differs from the pinned graph runtime")
        self.tensorrt_version = tensorrt_version
        self.engine_path = Path(build["engine_path"])
        self.engine_sha256 = build["engine_sha256"]
        with self.engine_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != self.engine_sha256:
            raise ValueError("Artifact checksum mismatch: " + str(self.engine_path))
        self.input_specs = self._specs(build["io_tensors"]["inputs"])
        self.output_specs = self._specs(build["io_tensors"]["outputs"])
        if {name: list(shape) for name, (shape, _) in self.input_specs.items()} != build["input_shapes"]:
            raise ValueError("Input tensor metadata differs from model contract")
        if set(self.input_specs) & set(self.output_specs):
            raise ValueError("Input and output tensor names overlap")
        self.io_bytes = sum(
            math.prod(shape) * dtype.itemsize
            for shape, dtype in (*self.input_specs.values(), *self.output_specs.values())
        )
        if self.io_bytes > 64 * 1024**2:
            raise ValueError("Engine I/O exceeds the configured buffer budget")
        self.provider = "TensorRT " + tensorrt_version + " / Holoscan"
        self.graph_managed = True

    @staticmethod
    def _specs(rows):
        """解析非空、无重名且固定形状的浮点张量；当前模型组均为 FP32 I/O。"""
        specs = {}
        for row in rows:
            name, shape = row["name"], tuple(row["shape"])
            dtype = np.dtype(row["dtype"])
            if (
                not isinstance(name, str)
                or not name
                or name in specs
                or not shape
                or any(type(dim) is not int or dim <= 0 for dim in shape)
                or dtype != np.dtype("float32")
            ):
                raise ValueError("Unsupported or duplicate graph tensor descriptor")
            specs[name] = shape, dtype
        if not specs:
            raise ValueError("Graph tensor descriptor cannot be empty")
        return specs

    def infer(self, inputs):
        """防止图模式意外退回同步模型调用；推理由连接的 InferenceOp 执行。"""
        raise RuntimeError("Graph-managed adapter requires prepare -> InferenceOp -> decode")

    def close(self):
        """描述对象无设备资源；图的所有者负责停止算子并释放执行资源。"""
