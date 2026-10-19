# SPDX-License-Identifier: AGPL-3.0-only
"""YOLOv8 body adapter using pinned Ultralytics OpenCV preprocessing/NMS.

Derived from Ultralytics' YOLOv8-OpenCV-ONNX-Python example. Local changes:
accept BGR arrays, filter person class, expose original-pixel xyxy boxes,
replace the runtime with the existing TensorRT engine, remove UI/global assets.
TensorRT allocation/copies use NVIDIA's pinned common_runtime sample.
"""

import hashlib
import importlib.util
from pathlib import Path
import threading
import time
import cv2
import numpy as np


def load_source(path, expected_sha, name):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError(f"Source hash mismatch: {path.name}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_file(path, expected):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Model artifact hash mismatch: {path.name}")
    return path


class BodyDetector:
    def __init__(
        self, entry, threshold=0.5, iou=0.45, max_detections=10, reference=False
    ):
        self.entry, self.threshold, self.iou, self.max_detections = (
            entry,
            threshold,
            iou,
            max_detections,
        )
        self._lock = threading.Lock()
        self.reference = reference
        self.session = self.context = self.engine = self.runtime = self.stream = None
        self.inputs, self.outputs = [], []
        model = checked_file(entry["model_path"], entry["model_sha256"])
        if reference:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(model), options, providers=["CPUExecutionProvider"]
            )
            self.provider = "CPUExecutionProvider (explicit reference only)"
            return
        import tensorrt as trt

        if trt.__version__ != entry["tensorrt_version"]:
            raise ValueError("TensorRT runtime differs from engine lock")
        self.common = load_source(
            entry["runtime_helper"], entry["runtime_helper_sha256"], "reachy_trt_common"
        )
        self.common.cuda_call(self.common.cudart.cudaSetDevice(0))
        self.logger = trt.Logger(trt.Logger.WARNING)
        try:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(
                checked_file(entry["engine_path"], entry["engine_sha256"]).read_bytes()
            )
            if self.engine is None:
                raise RuntimeError("TensorRT engine could not deserialize")
            if self.engine.num_io_tensors != 2:
                raise ValueError("Expected one input and one output")
            names = [
                self.engine.get_tensor_name(i)
                for i in range(self.engine.num_io_tensors)
            ]
            ins = [
                n
                for n in names
                if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
            ]
            outs = [
                n
                for n in names
                if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
            ]
            if len(ins) != 1 or len(outs) != 1:
                raise ValueError("Unexpected engine I/O")
            self.input_name, self.output_name = ins[0], outs[0]
            for name, shape in [
                (self.input_name, entry["input_shape"]),
                (self.output_name, entry["output_shape"]),
            ]:
                if (
                    tuple(self.engine.get_tensor_shape(name)) != tuple(shape)
                    or self.engine.get_tensor_dtype(name) != trt.float32
                ):
                    raise ValueError("Engine tensor shape/dtype differs from lock")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("TensorRT execution context unavailable")
            self.inputs, self.outputs, self.bindings = self.common.allocate_buffers(
                self.engine, context=self.context
            )
            self.stream = self.common.CudaStreamContext()
            for name, address in zip(names, self.bindings):
                if not self.context.set_tensor_address(name, address):
                    raise RuntimeError("Failed binding engine tensor")
            self.provider = "TensorRT " + trt.__version__
        except Exception:
            self.close()
            raise

    @staticmethod
    def preprocess(image):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Body detector requires BGR uint8")
        height, width = image.shape[:2]
        length = max(height, width)
        if not length:
            raise ValueError("Empty image")
        square = np.zeros((length, length, 3), np.uint8)
        square[:height, :width] = image
        blob = cv2.dnn.blobFromImage(
            square, scalefactor=1 / 255, size=(640, 640), swapRB=True
        )
        return blob, length / 640

    def predict_tensor(self, blob):
        if blob.shape != tuple(self.entry["input_shape"]) or blob.dtype != np.float32:
            raise ValueError("Unexpected body input tensor")
        with self._lock:
            if self.reference:
                if self.session is None:
                    raise RuntimeError("Reference model is closed")
                return self.session.run(
                    None, {self.session.get_inputs()[0].name: blob}
                )[0]
            if self.context is None:
                raise RuntimeError("TensorRT model is closed")
            self.common.cuda_call(self.common.cudart.cudaSetDevice(0))
            self.inputs[0].host = blob

            def execute():
                if not self.context.execute_async_v3(
                    stream_handle=int(self.stream.stream)
                ):
                    raise RuntimeError("TensorRT execution failed")

            output = self.common._do_inference_base(
                self.inputs, self.outputs, self.stream.stream, execute
            )[0]
            return output.reshape(self.entry["output_shape"])

    def postprocess(self, output, scale):
        if (
            output.shape != tuple(self.entry["output_shape"])
            or not np.isfinite(output).all()
        ):
            raise ValueError("Invalid body output tensor")
        rows = cv2.transpose(output[0])
        boxes, scores = [], []
        for row in rows:
            # Only person class enters body NMS; unrelated objects cannot suppress it.
            if (
                int(np.argmax(row[4:])) != self.entry["class_id"]
                or row[4] < self.threshold
            ):
                continue
            if row[2] <= 0 or row[3] <= 0:
                continue
            boxes.append(
                [
                    float(row[0] - 0.5 * row[2]),
                    float(row[1] - 0.5 * row[3]),
                    float(row[2]),
                    float(row[3]),
                ]
            )
            scores.append(float(row[4]))
        keep = np.asarray(
            cv2.dnn.NMSBoxes(boxes, scores, self.threshold, self.iou)
        ).reshape(-1)
        results = []
        for index in keep[: self.max_detections]:
            x, y, w, h = boxes[int(index)]
            results.append(
                {
                    "type": "person",
                    "box_xyxy": [
                        x * scale,
                        y * scale,
                        (x + w) * scale,
                        (y + h) * scale,
                    ],
                    "score": scores[int(index)],
                }
            )
        return results

    def infer(self, image):
        """BGR → 模型输入 → 同步原生调用 → 原图坐标检测框。

        last_timings 仅供观测；native_sync 包含数据传输、执行与同步，
        不能当成纯 CUDA kernel 耗时。
        """
        start = time.monotonic()
        tensor, scale = self.preprocess(image)
        prepared = time.monotonic()
        output = self.predict_tensor(tensor)
        predicted = time.monotonic()
        result = self.postprocess(output, scale)
        self.last_timings = {
            "preprocess_ms": (prepared - start) * 1000,
            "native_sync_ms": (predicted - prepared) * 1000,
            "postprocess_ms": (time.monotonic() - predicted) * 1000,
        }
        return result

    def close(self):
        with self._lock:
            if self.reference:
                self.session = None
                return
            if not hasattr(self, "common"):
                return
            self.common.cuda_call(self.common.cudart.cudaSetDevice(0))
            self.common.free_buffers(self.inputs, self.outputs)
            if self.stream:
                self.stream.free()
            # The pinned NVIDIA sample suppresses free errors. Surface any
            # still-owned allocation instead of claiming successful release.
            if any(
                x._device_mem._device_ptr is not None
                or x._host_mem._host_ptr is not None
                for x in self.inputs + self.outputs
            ):
                raise RuntimeError("CUDA buffers did not release")
            if self.stream and self.stream._stream is not None:
                raise RuntimeError("CUDA stream did not release")
            self.inputs, self.outputs = [], []
            self.context = self.engine = self.runtime = self.stream = None
