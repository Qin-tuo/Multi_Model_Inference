# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""设备、感知策略与执行图共享的数据契约，不依赖 Web 或推理框架。

帧数据由采集者持有，消费者只读共享；缓存结果始终保留真实的观测身份和时间。
"""

from dataclasses import dataclass, field, replace
from enum import IntEnum
import math
import threading
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrameKey:
    """使用相机重连代次和帧号唯一标识观测，避免重连后的帧号重用。"""

    source_epoch: str
    frame_id: int


@dataclass(frozen=True)
class FrameSnapshot:
    """一帧只读共享的 RGBD 数据；失效事件和原始主机接收时间随帧传递。"""

    frame_id: int
    source_epoch: str
    received_monotonic: float
    bgr: np.ndarray
    jpeg: bytes
    time_kind: str = "host_receive_monotonic"
    invalidated: threading.Event | None = None
    depth_m: np.ndarray | None = None
    depth_valid: np.ndarray | None = None
    sensor: dict | None = None

    @property
    def key(self) -> FrameKey:
        """返回与大数组解耦的帧身份，供依赖匹配和去重使用。"""
        return FrameKey(self.source_epoch, self.frame_id)

    def metadata(self):
        """保持现有公开帧元数据格式，不将图像数组带入 Web 或日志。"""
        result = {
            "frame_id": self.frame_id,
            "source_epoch": self.source_epoch,
            "received_monotonic": self.received_monotonic,
            "time_kind": self.time_kind,
            "width": self.bgr.shape[1],
            "height": self.bgr.shape[0],
            "stride": self.bgr.strides[0],
            "encoding": "bgr8",
        }
        if self.sensor is not None:
            result["sensor"] = self.sensor
        return result

    def is_fresh(self, now: float, source_epoch: str, max_age: float) -> bool:
        """按主机单调时钟判断观测是否仍可使用，并拒绝重连失效和未来时间。"""
        age = now - self.received_monotonic
        return (
            math.isfinite(age)
            and math.isfinite(max_age)
            and max_age > 0
            and self.source_epoch == source_epoch
            and not (self.invalidated and self.invalidated.is_set())
            and 0 <= age <= max_age
        )


class TaskPriority(IntEnum):
    """表达业务准入顺序；数值不代表 CUDA 抢占或操作系统实时优先级。"""

    INTERACTION = 0
    DISCOVERY = 1
    ATTRIBUTE = 2
    BACKGROUND = 3


@dataclass(frozen=True)
class TaskRequest:
    """一次有期限的模型请求；目标 ID 必须包含跟踪代次，不能跨人复用。"""

    role: str
    frame: FrameSnapshot
    target_id: str | None
    submitted_monotonic: float
    deadline_monotonic: float
    priority: TaskPriority

    def __post_init__(self):
        """拒绝无角色、时钟异常或在观测之前提交的请求。"""
        times = (self.frame.received_monotonic, self.submitted_monotonic, self.deadline_monotonic)
        if (
            not self.role
            or not all(math.isfinite(value) for value in times)
            or self.submitted_monotonic < self.frame.received_monotonic
            or self.deadline_monotonic < self.submitted_monotonic
            or not isinstance(self.priority, TaskPriority)
            or self.target_id == ""
        ):
            raise ValueError("Invalid perception task identity or timing")

    @property
    def key(self) -> tuple[str, FrameKey, str | None]:
        """返回模型、源帧及目标共同构成的请求身份。"""
        return self.role, self.frame.key, self.target_id

    def is_expired(self, now: float, source_epoch: str) -> bool:
        """计算前后均可检查期限；过期不表示在途 CUDA 工作已停止。"""
        return (
            not math.isfinite(now)
            or now < self.submitted_monotonic
            or now > self.deadline_monotonic
            or self.frame.source_epoch != source_epoch
            or bool(self.frame.invalidated and self.frame.invalidated.is_set())
        )


@dataclass(frozen=True)
class ModelObservation:
    """记录实际推理结果及有效期，结果载荷由模型适配器定义并只读传递。"""

    role: str
    frame_key: FrameKey
    target_id: str | None
    observed_monotonic: float
    completed_monotonic: float
    valid_until_monotonic: float
    payload: Any
    coordinate_space: str
    cached: bool = False

    def __post_init__(self):
        """验证时间顺序；过期完成的结果允许记录，但不能被当成有效观测。"""
        times = (self.observed_monotonic, self.completed_monotonic, self.valid_until_monotonic)
        if (
            not self.role
            or not self.coordinate_space
            or not all(math.isfinite(value) for value in times)
            or self.completed_monotonic < self.observed_monotonic
            or self.valid_until_monotonic < self.observed_monotonic
            or self.target_id == ""
        ):
            raise ValueError("Invalid model observation identity or timing")

    def is_valid(self, now: float, source_epoch: str, target_id: str | None) -> bool:
        """仅对相同来源和目标返回有效性；缓存不会延长原有效期。"""
        return (
            math.isfinite(now)
            and self.frame_key.source_epoch == source_epoch
            and self.target_id == target_id
            and self.completed_monotonic <= now <= self.valid_until_monotonic
        )

    def as_cached(self):
        """标记缓存复用，保留原帧号、观测时刻、完成时刻与有效期。"""
        return replace(self, cached=True)

    def require_same_subject(self, other: "ModelObservation") -> None:
        """为视线等依赖拒绝跨帧或跨人组合；各类坐标转换由对应适配器核对。"""
        if (
            self.frame_key != other.frame_key
            or self.target_id != other.target_id
            or self.observed_monotonic != other.observed_monotonic
        ):
            raise ValueError("Dependent observations refer to different frames or subjects")


@dataclass(frozen=True)
class ModelCall:
    """提供模型适配所需的目标参数；原图统一取自请求帧，避免传入另一帧图像。"""

    request: TaskRequest
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TaskOutcome:
    """模型任务的完成事件；时延字段明确区分主机墙钟与单独测量的 GPU 时间。"""

    request: TaskRequest
    status: str
    completed_monotonic: float
    payload: Any = None
    error: str | None = None
    timings_ms: dict = field(default_factory=dict)

    def __post_init__(self):
        """拒绝未知状态和倒置完成时间，保持调度统计与业务结果含义明确。"""
        if (
            self.status not in {"ok", "expired", "unavailable", "cancelled"}
            or not math.isfinite(self.completed_monotonic)
            or self.completed_monotonic < self.request.submitted_monotonic
        ):
            raise ValueError("Invalid model task outcome")
