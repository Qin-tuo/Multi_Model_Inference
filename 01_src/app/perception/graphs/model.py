# SPDX-License-Identifier: Apache-2.0
"""将纯模型适配器连接到 Holoscan 原生 InferenceOp。

每条模型链保留配对的请求上下文，内部队列按框架默认容量施加背压。
此处不决定交互目标或任务优先级，也不实现独立执行器。
"""

from dataclasses import dataclass
import math
import time

from app.contracts import ModelCall, TaskOutcome
from app.inference.holoscan_runtime import require_holoscan_runtime
from app.inference.tensorrt_runner import validate_inputs


@dataclass(frozen=True)
class PreparedContext:
    """伴随一次输入传给解码节点的原始请求、几何变换和计时信息。"""

    call: ModelCall
    transform: object
    submitted_to_backend: float
    timings_ms: dict
    input_owner: object


@dataclass(frozen=True)
class ModelNodes:
    """向图组装层返回明确的入口、原生推理节点和解码节点。"""

    prepare: object
    inference: object
    decode: object
    output_pool_bytes: int


def connect_model(fragment, role, adapter, on_complete, source_epoch):
    """创建一个模型的准备/原生执行/解码链，外部只需连接 prepare 的 request 端口。

    source_epoch 为设备层提供的当前来源查询；on_complete 消费统一完成事件。
    图的所有者负责停止和销毁算子；适配器只持有无设备资源的引擎描述。
    初始串行基线由图组装层选择单工作线程调度器，不能用每节点线程固定模拟限流。
    """
    require_holoscan_runtime()
    import cupy as cp
    import numpy as np
    from holoscan.core import Operator
    from holoscan.operators import InferenceOp
    from holoscan.resources import BlockMemoryPool

    engine = adapter.model
    if not getattr(engine, "graph_managed", False):
        raise ValueError("Native model graph requires an EngineDescription")
    output_bytes = max(math.prod(shape) * dtype.itemsize for shape, dtype in engine.output_specs.values())
    block_size = max(256, ((output_bytes + 255) // 256) * 256)
    # 为框架双缓冲及当前生产/消费保留四组块，容量显式固定，先不做内存调优。
    block_count = 4 * len(engine.output_specs)
    if block_size * block_count > 64 * 1024**2:
        raise ValueError("Native output pool exceeds the per-model budget")

    def finish(request, status, timings, *, payload=None, error=None):
        """回报业务可处理的结果；观察时间仍取请求源帧，不更新成当前帧。"""
        completed = time.monotonic()
        if status == "ok" and request.is_expired(completed, source_epoch()):
            status, payload, error = "expired", None, "expired_before_publication"
        on_complete(TaskOutcome(
            request=request,
            status=status,
            completed_monotonic=completed,
            payload=payload,
            error=error,
            timings_ms=timings,
        ))

    class Prepare(Operator):
        """处理一项 ROI 请求，并将张量与原始请求上下文成对发送。"""

        def setup(self, spec):
            """两个输出均可接收时才执行，避免丢弃单边上下文造成错配。"""
            spec.input("request", size=1)
            spec.output("tensors")
            spec.output("context")

        def compute(self, op_input, op_output, context):
            """验证时效，调用原预处理并上传输入；普通输入错误作为不可用结果。"""
            call = op_input.receive("request")
            if not isinstance(call, ModelCall) or call.request.role != role:
                raise ValueError("Model graph received a mismatched request")
            request = call.request
            started = time.monotonic()
            timings = {"queue_ms": (started - request.submitted_monotonic) * 1000}
            if request.is_expired(started, source_epoch()):
                finish(request, "expired", timings, error="expired_before_prepare")
                return
            try:
                inputs, transform = adapter.prepare(request.frame.bgr, *call.args, **call.kwargs)
                inputs = validate_inputs(inputs, engine.input_specs)
            except ValueError as error:
                timings["prepare_ms"] = (time.monotonic() - started) * 1000
                finish(request, "unavailable", timings, error=str(error))
                return
            prepared = time.monotonic()
            timings["prepare_ms"] = (prepared - started) * 1000
            if request.is_expired(prepared, source_epoch()):
                finish(request, "expired", timings, error="expired_after_prepare")
                return
            # 初始对照保留显式上传完成边界，后续异步重叠由用户另行实验。
            device_inputs = {name: cp.asarray(inputs[name]) for name in engine.input_specs}
            cp.cuda.get_current_stream().synchronize()
            submitted = time.monotonic()
            timings["upload_wall_ms"] = (submitted - prepared) * 1000
            if request.is_expired(submitted, source_epoch()):
                finish(request, "expired", timings, error="expired_after_upload")
                return
            paired = PreparedContext(call, transform, submitted, timings, device_inputs)
            op_output.emit(device_inputs, "tensors")
            op_output.emit(paired, "context")

    class Decode(Operator):
        """消费同一模型 FIFO 顺序中的上下文和输出，在发布前再次核对期限。"""

        def setup(self, spec):
            """等待成对结果；内部端口不使用独立丢帧策略。"""
            spec.input("tensors", size=1)
            spec.input("context", size=1)

        def compute(self, op_input, op_output, context):
            """校验输出、执行原解码并发送完成事件；迟到结果不进入业务状态。"""
            raw = op_input.receive("tensors")
            paired = op_input.receive("context")
            if not isinstance(paired, PreparedContext):
                raise ValueError("Model output lost its request context")
            request = paired.call.request
            received = time.monotonic()
            timings = dict(paired.timings_ms)
            # 该区间包括原生节点排队、执行和输出复制，不能标为纯 GPU 时间。
            timings["backend_wall_ms"] = (received - paired.submitted_to_backend) * 1000
            if request.is_expired(received, source_epoch()):
                finish(request, "expired", timings, error="expired_after_inference")
                return
            outputs = {}
            for name, (shape, dtype) in engine.output_specs.items():
                value = np.asarray(raw[name])
                if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                    raise RuntimeError("Invalid native inference output: " + role + "/" + name)
                outputs[name] = value
            try:
                payload = adapter.decode(outputs, paired.transform)
            except ValueError as error:
                timings["decode_ms"] = (time.monotonic() - received) * 1000
                finish(request, "unavailable", timings, error=str(error))
                return
            finished = time.monotonic()
            timings["decode_ms"] = (finished - received) * 1000
            timings["task_wall_ms"] = (finished - request.submitted_monotonic) * 1000
            if request.is_expired(finished, source_epoch()):
                finish(request, "expired", timings, error="expired_after_decode")
                return
            finish(request, "ok", timings, payload=payload)

    prepare = Prepare(fragment, name=role + "_prepare")
    inference = InferenceOp(
        fragment,
        name=role + "_infer",
        backend="trt",
        allocator=BlockMemoryPool(
            fragment,
            name=role + "_allocator",
            storage_type=0,
            block_size=block_size,
            num_blocks=block_count,
        ),
        model_path_map={role: str(engine.engine_path)},
        pre_processor_map={role: list(engine.input_specs)},
        inference_map={role: list(engine.output_specs)},
        is_engine_path=True,
        input_on_cuda=True,
        output_on_cuda=False,
        transmit_on_cuda=False,
        parallel_inference=False,
        enable_cuda_graphs=False,
    )
    decode = Decode(fragment, name=role + "_decode")
    fragment.add_flow(prepare, inference, {("tensors", "receivers")})
    fragment.add_flow(prepare, decode, {("context", "context")})
    fragment.add_flow(inference, decode, {("transmitter", "tensors")})
    return ModelNodes(prepare, inference, decode, block_size * block_count)
