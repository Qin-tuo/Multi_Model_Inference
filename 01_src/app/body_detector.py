# SPDX-License-Identifier: AGPL-3.0-only
"""YOLOv8 body adapter using pinned Ultralytics OpenCV preprocessing/NMS.

Derived from Ultralytics' YOLOv8-OpenCV-ONNX-Python example. Local changes:
accept BGR arrays, filter person class, expose original-pixel xyxy boxes,
replace the runtime with the existing TensorRT engine, remove UI/global assets.
TensorRT allocation/copies use NVIDIA's pinned common_runtime sample.
"""

# 阅读顺序：先看 infer()，再看 preprocess()、predict_tensor()、postprocess()。
# __init__() 和 close() 负责模型加载与资源管理。
# 数据流：原图 BGR → RGB 浮点张量 → TensorRT → 候选框 → 原图中的人体框。

import hashlib
import importlib.util
from pathlib import Path
import threading
import time
import cv2
import numpy as np


def load_source(path, expected_sha, name):
    # 核对辅助代码的 SHA256，确保运行的是清单锁定的版本。
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError(f"Source hash mismatch: {path.name}")
    # 从指定文件动态导入模块，使用其中的 CUDA 缓冲区和推理辅助函数。
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_file(path, expected):
    # 校验模型或引擎文件内容；文件名相同并不代表权重内容相同。
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Model artifact hash mismatch: {path.name}")
    return path


class BodyDetector:
    def __init__(
        self, entry, threshold=0.5, iou=0.45, max_detections=10, reference=False
    ):
        # entry：模型路径、哈希、版本和输入/输出形状等清单信息。
        # threshold：人体分数阈值；iou：NMS 判断检测框重叠的阈值。
        # max_detections：最多返回的人体框数量。
        # reference=True：显式使用 ONNX Runtime CPU 参考实现，供输出对照。
        self.entry, self.threshold, self.iou, self.max_detections = (
            entry,
            threshold,
            iou,
            max_detections,
        )
        # predict_tensor() 与 close() 共用锁，串行访问上下文和共享缓冲区。
        self._lock = threading.Lock()
        self.reference = reference
        self.session = self.context = self.engine = self.runtime = self.stream = None
        self.inputs, self.outputs = [], []
        # 先确认原始 ONNX 的身份，用于参考推理与工件追溯。
        model = checked_file(entry["model_path"], entry["model_sha256"])
        if reference:
            import onnxruntime as ort

            # 固定 CPU 参考实现的线程数，使对照配置可重复。
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(model), options, providers=["CPUExecutionProvider"]
            )
            self.provider = "CPUExecutionProvider (explicit reference only)"
            return
        import tensorrt as trt

        # 本项目严格匹配清单中的 TensorRT 版本及辅助代码版本。
        if trt.__version__ != entry["tensorrt_version"]:
            raise ValueError("TensorRT runtime differs from engine lock")
        self.common = load_source(
            entry["runtime_helper"], entry["runtime_helper_sha256"], "assistant_trt_common"
        )
        self.common.cuda_call(self.common.cudart.cudaSetDevice(0))
        self.logger = trt.Logger(trt.Logger.WARNING)
        try:
            # Runtime 将已经编译的引擎反序列化为 Engine，此处不重新构建网络。
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(
                checked_file(entry["engine_path"], entry["engine_sha256"]).read_bytes()
            )
            if self.engine is None:
                raise RuntimeError("TensorRT engine could not deserialize")
            if self.engine.num_io_tensors != 2:
                raise ValueError("Expected one input and one output")
            # 当前适配器要求恰好一个输入张量、一个输出张量。
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
            # 检查 I/O 的固定形状及 FP32 类型，避免用错误布局解释数据。
            # 内部计算精度由引擎构建配置决定，与这里的 I/O 类型分开理解。
            for name, shape in [
                (self.input_name, entry["input_shape"]),
                (self.output_name, entry["output_shape"]),
            ]:
                if (
                    tuple(self.engine.get_tensor_shape(name)) != tuple(shape)
                    or self.engine.get_tensor_dtype(name) != trt.float32
                ):
                    raise ValueError("Engine tensor shape/dtype differs from lock")
            # Engine 保存网络，ExecutionContext 保存执行时的运行状态。
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("TensorRT execution context unavailable")
            # 初始化时分配主机/GPU 缓冲区和 CUDA stream，后续每帧重复使用。
            self.inputs, self.outputs, self.bindings = self.common.allocate_buffers(
                self.engine, context=self.context
            )
            self.stream = self.common.CudaStreamContext()
            # 将张量名称绑定到 GPU 缓冲区地址，供 execute_async_v3() 使用。
            for name, address in zip(names, self.bindings):
                if not self.context.set_tensor_address(name, address):
                    raise RuntimeError("Failed binding engine tensor")
            self.provider = "TensorRT " + trt.__version__
        except Exception:
            # 初始化中途失败时，清理已经分配的资源。
            self.close()
            raise

    @staticmethod
    def preprocess(image):
        # 相机/OpenCV 输入为 H×W×3 的 BGR 图像，通道类型是 uint8（0～255）。
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Body detector requires BGR uint8")
        height, width = image.shape[:2]
        length = max(height, width)
        if not length:
            raise ValueError("Empty image")
        # 原图放在正方形左上角，右侧或下侧补黑色，避免直接拉伸导致形状失真。
        # 后处理按此补边方式还原坐标，调整补边方式时需同步检查坐标转换。
        square = np.zeros((length, length, 3), np.uint8)
        square[:height, :width] = image
        # 缩放到 640×640，BGR→RGB，并乘 1/255 将数值归一化到 0～1。
        # blobFromImage 同时把 HWC 转为 NCHW，当前张量形状为 (1, 3, 640, 640)。
        blob = cv2.dnn.blobFromImage(
            square, scalefactor=1 / 255, size=(640, 640), swapRB=True
        )
        # scale 是模型坐标还原到原图坐标的倍数；左上角没有补边偏移。
        return blob, length / 640

    def predict_tensor(self, blob):
        # 输入必须符合固定引擎的形状与数据类型。
        if blob.shape != tuple(self.entry["input_shape"]) or blob.dtype != np.float32:
            raise ValueError("Unexpected body input tensor")
        with self._lock:
            if self.reference:
                # CPU 对照路径也返回与默认路径相同布局的候选框张量。
                if self.session is None:
                    raise RuntimeError("Reference model is closed")
                return self.session.run(
                    None, {self.session.get_inputs()[0].name: blob}
                )[0]
            if self.context is None:
                raise RuntimeError("TensorRT model is closed")
            # CUDA 当前设备与线程有关，因此在实际调用线程中再次选择设备。
            self.common.cuda_call(self.common.cudart.cudaSetDevice(0))
            # 将本帧张量交给输入主机缓冲区接口，实际上传由辅助函数完成。
            self.inputs[0].host = blob

            def execute():
                # 将引擎执行异步提交到指定 CUDA stream。
                if not self.context.execute_async_v3(
                    stream_handle=int(self.stream.stream)
                ):
                    raise RuntimeError("TensorRT execution failed")

            # 辅助函数执行 CPU→GPU 拷贝、引擎执行、GPU→CPU 拷贝及 stream 同步。
            # 最后返回 CPU 输出的副本；本方法返回时已等待计算与传输完成。
            output = self.common._do_inference_base(
                self.inputs, self.outputs, self.stream.stream, execute
            )[0]
            return output.reshape(self.entry["output_shape"])

    def postprocess(self, output, scale):
        # 确认输出形状正确且没有 NaN/Inf，避免异常数值进入筛选与 NMS。
        if (
            output.shape != tuple(self.entry["output_shape"])
            or not np.isfinite(output).all()
        ):
            raise ValueError("Invalid body output tensor")
        # 当前 YOLOv8n 输出为 (1, 84, 8400)：1 张图、84 个数值、8400 个候选框。
        # 每个框含 [中心 x, 中心 y, 宽, 高] 及 80 个 COCO 类别分数。
        # 转置后每行对应一个候选框，下面的筛选和坐标计算都在 CPU 上执行。
        rows = cv2.transpose(output[0])
        boxes, scores = [], []
        for row in rows:
            # Only person class enters body NMS; unrelated objects cannot suppress it.
            # 本项目 class_id=0（person），row[4] 是人体分数。
            # 同时要求最高分类为人、人体分数达到阈值；优化时应保留两个条件。
            if (
                int(np.argmax(row[4:])) != self.entry["class_id"]
                or row[4] < self.threshold
            ):
                continue
            # 排除宽或高不为正的无效框。
            if row[2] <= 0 or row[3] <= 0:
                continue
            # NMSBoxes 需要 [左上 x, 左上 y, 宽, 高]，先由中心点坐标转换。
            boxes.append(
                [
                    float(row[0] - 0.5 * row[2]),
                    float(row[1] - 0.5 * row[3]),
                    float(row[2]),
                    float(row[3]),
                ]
            )
            scores.append(float(row[4]))
        # NMS（非极大值抑制）保留高分框，抑制与其重叠过大的重复框。
        # 返回的是候选框索引，之后最多取 max_detections 个。
        keep = np.asarray(
            cv2.dnn.NMSBoxes(boxes, scores, self.threshold, self.iou)
        ).reshape(-1)
        results = []
        for index in keep[: self.max_detections]:
            x, y, w, h = boxes[int(index)]
            # 乘 scale 还原到原图像素，输出 [左上 x, 左上 y, 右下 x, 右下 y]。
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
        # 无人体时返回 []，表示一次有效的空检测结果。
        return results

    def infer(self, image):
        """BGR → 模型输入 → 同步原生调用 → 原图坐标检测框。

        last_timings 仅供观测；native_sync 包含数据传输、执行与同步，
        不能当成纯 CUDA kernel 耗时。
        """
        # 用单调时钟分三段计时，避免系统校时影响耗时计算。
        start = time.monotonic()
        # 1. CPU 预处理：原图→模型张量，同时记录坐标还原比例。
        tensor, scale = self.preprocess(image)
        prepared = time.monotonic()
        # 2. 同步原生调用：默认为 TensorRT，包含传输、执行、同步和输出复制。
        output = self.predict_tensor(tensor)
        predicted = time.monotonic()
        # 3. CPU 后处理：原始候选框→最终人体框。
        result = self.postprocess(output, scale)
        # 秒乘 1000 得到毫秒，供调试面板与性能观察使用。
        self.last_timings = {
            "preprocess_ms": (prepared - start) * 1000,
            "native_sync_ms": (predicted - prepared) * 1000,
            "postprocess_ms": (time.monotonic() - predicted) * 1000,
        }
        return result

    def close(self):
        # 与 predict_tensor() 互斥，避免推理期间释放其缓冲区。
        with self._lock:
            if self.reference:
                # CPU 参考路径释放 session 引用即可。
                self.session = None
                return
            if not hasattr(self, "common"):
                return
            self.common.cuda_call(self.common.cudart.cudaSetDevice(0))
            # 先释放主机/GPU 缓冲区，再释放 CUDA stream。
            self.common.free_buffers(self.inputs, self.outputs)
            if self.stream:
                self.stream.free()
            # The pinned NVIDIA sample suppresses free errors. Surface any
            # still-owned allocation instead of claiming successful release.
            # 检查底层句柄是否确实清空，暴露辅助库未成功释放的资源。
            if any(
                x._device_mem._device_ptr is not None
                or x._host_mem._host_ptr is not None
                for x in self.inputs + self.outputs
            ):
                raise RuntimeError("CUDA buffers did not release")
            if self.stream and self.stream._stream is not None:
                raise RuntimeError("CUDA stream did not release")
            # 清空 Python 引用，释放上下文、引擎等对象持有的资源。
            self.inputs, self.outputs = [], []
            self.context = self.engine = self.runtime = self.stream = None
