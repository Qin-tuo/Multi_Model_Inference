# SPDX-License-Identifier: Apache-2.0
"""封装固定形状的 TensorRT 命名张量推理。

每个实例拥有独立执行上下文、CUDA 流和可复用缓冲区；调用由锁串行化，模型预后处理交给上层适配器。
"""

import hashlib
import importlib.util
import math
from pathlib import Path
import threading

import numpy as np

from project_paths import ROOT, MODEL_DIR, load_artifact_manifest

_LOGGER = None
_LOGGER_LOCK = threading.Lock()


def runtime_logger(trt):
    """按进程复用 TensorRT 日志器，确保其生命周期长于任何引擎或执行上下文。"""
    global _LOGGER
    with _LOGGER_LOCK:
        if _LOGGER is None:
            _LOGGER = trt.Logger(trt.Logger.WARNING)
        return _LOGGER


def validate_inputs(values, specs):
    """逐项校验输入名称、形状、dtype 和有限值，返回连续内存布局的 NumPy 张量。

    这里不隐式改变类型或形状，避免将错误输入送进引擎。
    """
    if set(values) != set(specs):
        raise ValueError("Input tensor names differ from the engine contract")
    result = {}
    for name, (shape, dtype) in specs.items():
        value = values[name]
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != dtype
        ):
            raise ValueError("Input tensor shape/dtype mismatch: " + name)
        if not np.isfinite(value).all():
            raise ValueError("Non-finite input tensor: " + name)
        result[name] = np.ascontiguousarray(value)
    return result


def checked_bytes(path, expected):
    """读取工件字节并检查 SHA256；只有内容与清单一致才返回给反序列化器。"""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Artifact checksum mismatch: " + str(path))
    return data


class TensorRTModel:
    """管理单个 TensorRT 引擎的资源与同步推理接口。"""

    def __init__(
        self, engine_path, engine_sha256, input_shapes, max_io_bytes=64 * 1024**2
    ):
        """校验 TensorRT 版本和引擎摘要，创建上下文并绑定固定形状的设备缓冲区。

        解析全部输入输出形状，拒绝未确定形状、非设备张量及超过字节预算的分配；初始化失败时回收已创建资源。
        """
        import tensorrt as trt

        self._lock = threading.RLock()
        self._closing = False
        self.runtime = self.engine = self.context = self.stream = None
        self.inputs, self.outputs = [], []
        self.input_specs, self.output_specs = {}, {}
        lock = load_artifact_manifest(MODEL_DIR / "runtime-components.json")["tensorrt"]
        if trt.__version__ != lock["tensorrt_version"]:
            raise ValueError("TensorRT version differs from the pinned host runtime")
        helper = ROOT / lock["runtime_helper"]
        checked_bytes(helper, lock["runtime_helper_sha256"])
        spec = importlib.util.spec_from_file_location("assistant_trt_buffers", helper)
        self.common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.common)
        self.logger = runtime_logger(trt)
        try:
            self._select_device()
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(
                checked_bytes(engine_path, engine_sha256)
            )
            if self.engine is None:
                raise RuntimeError("Cannot deserialize TensorRT engine")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Cannot create TensorRT execution context")
            # 先确定所有 I/O 的名称和形状，再分配缓冲区，避免输出尺寸不明时盲目申请内存。
            names = [
                self.engine.get_tensor_name(i)
                for i in range(self.engine.num_io_tensors)
            ]
            inputs = [
                n
                for n in names
                if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
            ]
            if set(input_shapes) != set(inputs):
                raise ValueError("Declared input names differ from the engine")
            for name in inputs:
                shape = tuple(input_shapes[name])
                if not self.context.set_input_shape(name, shape):
                    raise ValueError("Shape is outside the engine profile: " + name)
            unresolved = self.context.infer_shapes()
            if unresolved:
                raise ValueError("Unspecified shape tensors: " + repr(unresolved))
            io_bytes = 0
            for name in names:
                shape = tuple(self.context.get_tensor_shape(name))
                if not shape or any(d <= 0 for d in shape):
                    raise ValueError(
                        "Data-dependent or invalid output allocation: " + name
                    )
                if self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                    raise ValueError("This adapter requires device I/O: " + name)
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                io_bytes += math.prod(shape) * dtype.itemsize
                specs = self.input_specs if name in inputs else self.output_specs
                specs[name] = (shape, dtype)
            if io_bytes > max_io_bytes:
                raise ValueError("Engine I/O exceeds the configured buffer budget")
            # 缓冲区在实例内复用；io_bytes 只统计显式张量，不等于引擎全部显存占用。
            self.inputs, self.outputs, bindings = self.common.allocate_buffers(
                self.engine, context=self.context
            )
            self.stream = self.common.CudaStreamContext()
            for name, address in zip(names, bindings):
                if not self.context.set_tensor_address(name, address):
                    raise RuntimeError("Tensor address binding failed: " + name)
            self.io_bytes = io_bytes
            self.provider = "TensorRT " + trt.__version__
        except BaseException:
            self.close()
            raise

    def _select_device(self):
        """将调用线程切换到 CUDA 设备 0，保证线程中的分配、推理和释放使用同一设备。"""
        self.common.cuda_call(self.common.cudart.cudaSetDevice(0))

    def infer(self, values):
        """串行执行输入校验、主机到设备复制、TensorRT 推理和设备到主机复制。

        返回按输出名称组织的主机数组；辅助执行器完成同步和结果复制后才返回，下一次调用不会覆盖已发布结果。
        """
        with self._lock:
            if self.context is None or self._closing:
                raise RuntimeError("TensorRT model is closed or closing")
            arrays = validate_inputs(values, self.input_specs)
            self._select_device()
            for name, buffer in zip(self.input_specs, self.inputs):
                buffer.host = arrays[name]

            def execute():
                """在本实例的 CUDA 流上提交 execute_async_v3，提交失败立即抛出异常。"""
                if not self.context.execute_async_v3(
                    stream_handle=int(self.stream.stream)
                ):
                    raise RuntimeError("TensorRT execution failed")

            # 辅助执行器等待 H2D、计算和 D2H 完成，再返回独立主机副本；
            # 下一轮复用缓冲区不会覆盖上一轮已经发布的数组。
            outputs = self.common._do_inference_base(
                self.inputs, self.outputs, self.stream.stream, execute
            )
            result = {
                name: value.reshape(shape)
                for (name, (shape, _)), value in zip(self.output_specs.items(), outputs)
            }
            if any(not np.isfinite(value).all() for value in result.values()):
                raise RuntimeError("Non-finite TensorRT output")
            return result

    def close(self):
        """按流同步、缓冲区、CUDA 流、上下文和引擎的顺序释放资源。

        释放未确认时保留所有权并禁止继续推理，允许后续再次调用 close 重试。
        """
        with self._lock:
            self._closing = True
            if self.stream is not None:
                self._select_device()
                self.stream.synchronize()
            if self.inputs or self.outputs:
                self.common.free_buffers(self.inputs, self.outputs)
                # 上游 RAII 辅助器可能抑制释放异常，因此还要检查实际指针。
                # 未确认释放前保留对象并禁止推理，后续 close 才能继续重试。
                if any(
                    b._device_mem.device_ptr is not None
                    or b._host_mem._host_ptr is not None
                    for b in self.inputs + self.outputs
                ):
                    raise RuntimeError("TensorRT I/O buffer release is unconfirmed")
                self.inputs, self.outputs = [], []
            if self.stream is not None:
                self.stream.free()
                if self.stream._stream is not None:
                    raise RuntimeError("TensorRT CUDA stream release is unconfirmed")
                self.stream = None
            self.context = self.engine = self.runtime = None

    def __enter__(self):
        """返回当前资源所有者，支持 with 语句使用。"""
        return self

    def __exit__(self, *_):
        """退出 with 作用域时释放模型资源；不吞掉作用域内的异常。"""
        self.close()
