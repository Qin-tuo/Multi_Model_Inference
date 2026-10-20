# SPDX-License-Identifier: Apache-2.0
"""Named-tensor adapter over the project's pinned NVIDIA TensorRT sample runtime.

One instance owns one execution context and reusable buffers at fixed shapes.
Calls are serialized; preprocessing, decoding and task scheduling belong to the
model/application modules. There is no fallback to another inference backend.
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
    """TensorRT shares one logger; retain it while any model/context can exist."""
    global _LOGGER
    with _LOGGER_LOCK:
        if _LOGGER is None:
            _LOGGER = trt.Logger(trt.Logger.WARNING)
        return _LOGGER


def validate_inputs(values, specs):
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
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Artifact checksum mismatch: " + str(path))
    return data


class TensorRTModel:
    def __init__(
        self, engine_path, engine_sha256, input_shapes, max_io_bytes=64 * 1024**2
    ):
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
        self.common.cuda_call(self.common.cudart.cudaSetDevice(0))

    def infer(self, values):
        with self._lock:
            if self.context is None or self._closing:
                raise RuntimeError("TensorRT model is closed or closing")
            arrays = validate_inputs(values, self.input_specs)
            self._select_device()
            for name, buffer in zip(self.input_specs, self.inputs):
                buffer.host = arrays[name]

            def execute():
                if not self.context.execute_async_v3(
                    stream_handle=int(self.stream.stream)
                ):
                    raise RuntimeError("TensorRT execution failed")

            # NVIDIA's helper waits for copies and computation, then returns
            # host-owned copies. A later call cannot overwrite a published result.
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
        with self._lock:
            self._closing = True
            if self.stream is not None:
                self._select_device()
                self.stream.synchronize()
            if self.inputs or self.outputs:
                self.common.free_buffers(self.inputs, self.outputs)
                # The pinned NVIDIA RAII helpers intentionally suppress release
                # errors. Keep ownership and prohibit inference until every
                # explicit release is confirmed, so a later close can retry.
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
        return self

    def __exit__(self, *_):
        self.close()
