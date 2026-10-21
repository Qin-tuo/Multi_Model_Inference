# SPDX-License-Identifier: Apache-2.0
"""使用 RealSense SDK 采集 RGB 与原生深度，保留传感器时间戳及计数信息。

SDK 帧组不等于同时曝光；时间匹配只比较可信时钟域，返回数组与 SDK 内存池脱离。
"""

from dataclasses import dataclass
import hashlib
import math
import time
import uuid

import numpy as np


def intrinsics(profile):
    """把 SDK 视频配置的内参、畸变参数和实际图像尺寸转成普通字典。"""
    value = profile.as_video_stream_profile().get_intrinsics()
    return {
        "width": value.width,
        "height": value.height,
        "fx": value.fx,
        "fy": value.fy,
        "ppx": value.ppx,
        "ppy": value.ppy,
        "distortion_model": str(value.model),
        "coefficients": list(value.coeffs),
    }


def correspondence(streams, max_rgb_skew_ms):
    """检查红外双目计数/时间一致性，以及 RGB 与深度在全局时钟域内的时间接近程度。

    缺帧、时钟域不兼容或超出偏差预算时将相应匹配标记置为 False，不假定帧组已经同步曝光。
    """
    left, right, color = (streams.get(n) for n in ("left_ir", "right_ir", "color"))
    stereo = bool(
        left
        and right
        and left["domain"] == right["domain"]
        and left["frame_number"] == right["frame_number"]
        and abs(left["timestamp_ms"] - right["timestamp_ms"]) <= 0.1
    )
    # RGB and stereo use different sensors. Compare only SDK global time,
    # never presume equal local-clock labels mean the same clock origin.
    global_time = bool(
        stereo
        and color
        and left["domain"] == color["domain"] == "timestamp_domain.global_time"
    )
    skew = abs(color["timestamp_ms"] - left["timestamp_ms"]) if global_time else None
    depth = streams.get("sdk_depth")
    depth_global = bool(
        depth
        and color
        and depth["domain"] == color["domain"] == "timestamp_domain.global_time"
    )
    depth_skew = (
        abs(color["timestamp_ms"] - depth["timestamp_ms"]) if depth_global else None
    )
    return {
        "stereo_valid": stereo,
        "rgb_time_matched": bool(global_time and skew <= max_rgb_skew_ms),
        "rgb_skew_ms": skew,
        "rgb_depth_time_matched": bool(depth_global and depth_skew <= max_rgb_skew_ms),
        "rgb_depth_skew_ms": depth_skew,
        "rgb_match_kind": "SDK global-time proximity; not exposure synchronization",
    }


@dataclass(frozen=True)
class SensorFrame:
    """保存一个 SDK 帧组的独立数组、流元数据和时间对应关系。"""

    source_epoch: str
    group_id: int
    received_monotonic: float
    arrays: dict
    streams: dict
    correspondence: dict

    def metadata(self):
        """返回帧组身份与各流对应关系，明确主机接收时刻不是传感器曝光时刻。"""
        return {
            "source_epoch": self.source_epoch,
            "group_id": self.group_id,
            "received_monotonic": self.received_monotonic,
            "host_time_kind": "receive_monotonic; not sensor capture time",
            "streams": self.streams,
            "correspondence": self.correspondence,
        }


class RealSenseSource:
    """由单一调用方拥有 open/read/close，负责设备采集和标定数据。"""

    def __init__(self, width=640, height=480, fps=15, serial=None, with_ir=False):
        """限定已支持的 640×480@15 采集配置，设置可选设备序列号与红外输出开关。"""
        if (width, height, fps) != (640, 480, 15):
            raise ValueError("Only the verified 640x480@15 capture contract is enabled")
        self.width, self.height, self.fps = width, height, fps
        self.serial, self.with_ir = serial, with_ir
        self.pipeline = None
        self.align = None
        self.calibration = None
        self.source_epoch = None
        self.group_id = 0
        self.max_rgb_skew_ms = 1000 / fps / 2

    def open(self):
        """选择唯一设备、启动所需流并检查实际模式，记录内外参和深度单位。

        创建对齐到彩色图的 SDK 对象；任一步失败都会关闭已启动的管线。
        """
        import pyrealsense2 as rs

        if self.pipeline is not None:
            raise RuntimeError("Stereo capture is already open")
        if rs.__version__ != "2.58.4":
            raise RuntimeError("Stereo SDK version differs from the verified contract")
        context = rs.context({"dds": {"enabled": False}})
        devices = list(context.query_devices())
        if self.serial is not None:
            devices = [
                d
                for d in devices
                if d.get_info(rs.camera_info.serial_number) == self.serial
            ]
        if len(devices) != 1:
            raise RuntimeError("Select exactly one connected camera")
        device = devices[0]
        serial = device.get_info(rs.camera_info.serial_number)
        config = rs.config()
        config.enable_device(serial)
        requests = {
            "color": (rs.stream.color, 0, rs.format.rgb8),
            "sdk_depth": (rs.stream.depth, 0, rs.format.z16),
        }
        if self.with_ir:
            requests.update(
                left_ir=(rs.stream.infrared, 1, rs.format.y8),
                right_ir=(rs.stream.infrared, 2, rs.format.y8),
            )
        for stream, index, fmt in requests.values():
            config.enable_stream(stream, index, self.width, self.height, fmt, self.fps)
        pipeline = rs.pipeline(context)
        active = pipeline.start(config)
        self.pipeline = pipeline
        try:
            profiles = {
                name: active.get_stream(stream, index)
                for name, (stream, index, _) in requests.items()
            }
            for name, profile in profiles.items():
                video = profile.as_video_stream_profile()
                if (video.width(), video.height(), profile.fps(), profile.format()) != (
                    self.width,
                    self.height,
                    self.fps,
                    requests[name][2],
                ):
                    raise RuntimeError(
                        "Active camera profile differs from request: " + name
                    )
            calibration = {
                "device": device.get_info(rs.camera_info.name),
                "serial_sha256": hashlib.sha256(serial.encode()).hexdigest(),
                "sdk_version": rs.__version__,
                "firmware": device.get_info(rs.camera_info.firmware_version),
                "usb_type": device.get_info(rs.camera_info.usb_type_descriptor),
                "profiles": {
                    name: intrinsics(profile) for name, profile in profiles.items()
                },
                "extrinsics": {},
                "depth_units_m": device.first_depth_sensor().get_depth_scale(),
                "depth_provider": "D435i native depth via RealSense SDK; no neural inference",
                "alignment": "librealsense rs.align(stream.color)",
            }
            for origin in ("sdk_depth", "left_ir"):
                if origin not in profiles:
                    continue
                for name in sorted(profiles.keys() - {origin}):
                    value = profiles[origin].get_extrinsics_to(profiles[name])
                    calibration["extrinsics"][origin + "_to_" + name] = {
                        "rotation": list(value.rotation),
                        "rotation_layout": "SDK column-major 3x3",
                        "translation_m": list(value.translation),
                    }
            if self.with_ir:
                translation = calibration["extrinsics"]["left_ir_to_right_ir"][
                    "translation_m"
                ]
                calibration["stereo_baseline_m"] = math.sqrt(
                    sum(v * v for v in translation)
                )
            self.calibration = calibration
            self.align = rs.align(rs.stream.color)
            self.source_epoch = uuid.uuid4().hex
            self.group_id = 0
        except BaseException:
            self.close()
            raise

    def read(self, timeout_ms=1000):
        """等待 SDK 帧组、将深度对齐到 RGB，复制各流数组并标记只读后返回。

        同时记录帧号、传感器时间域和主机接收时间，供上层拒绝失配或重复帧。
        """
        if self.pipeline is None:
            raise RuntimeError("Stereo capture is closed")
        frames = self.pipeline.wait_for_frames(timeout_ms)
        received = time.monotonic()
        values = {
            "color": frames.get_color_frame(),
            "sdk_depth": frames.get_depth_frame(),
        }
        if self.with_ir:
            values.update(
                left_ir=frames.get_infrared_frame(1),
                right_ir=frames.get_infrared_frame(2),
            )
        if values["color"] and values["sdk_depth"]:
            aligned = self.align.process(frames)
            values["aligned_depth"] = aligned.get_depth_frame()
        arrays, streams = {}, {}
        for name, frame in values.items():
            if not frame:
                continue
            streams[name] = {
                "frame_number": frame.get_frame_number(),
                "timestamp_ms": frame.get_timestamp(),
                "domain": str(frame.get_frame_timestamp_domain()),
                "format": str(frame.get_profile().format()),
            }
            # Detach from SDK pools before publishing to another consumer.
            arrays[name] = np.asanyarray(frame.get_data()).copy()
            arrays[name].flags.writeable = False
        self.group_id += 1
        return SensorFrame(
            self.source_epoch,
            self.group_id,
            received,
            arrays,
            streams,
            correspondence(streams, self.max_rgb_skew_ms),
        )

    def close(self):
        """停止 SDK 管线并清空对齐对象；即使停止抛错也不保留已关闭句柄。"""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            finally:
                self.pipeline = None
                self.align = None

    def __enter__(self):
        """进入 with 作用域时打开相机并返回采集对象。"""
        self.open()
        return self

    def __exit__(self, *_):
        """退出 with 作用域时停止相机采集。"""
        self.close()
