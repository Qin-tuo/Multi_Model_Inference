"""ROS 2/message_filters input adapter for the bounded RGB-D cache."""
import os
import json
from pathlib import Path
import threading
import time
import hashlib
import math

from app.vision_frames import VisionFrameStore
from app.perception import GestureRules, Person


class RosVisionBridge:
    def __init__(self, *, domain_id=None, frame_bytes_limit=128 * 1024 * 1024,
                 retention_seconds=2, stale_seconds=.5, perception=False):
        # Reserve 16 MiB for the bounded DDS/synchronizer queues and one
        # in-progress copy/packed image. Admission is fixed at 640x480 RGB-D.
        self.transport_reserve = 16 * 1024 * 1024
        if frame_bytes_limit <= self.transport_reserve: raise ValueError("Vision budget is smaller than its transport reserve")
        self.frames = VisionFrameStore(max_bytes=frame_bytes_limit - self.transport_reserve,
            retention_seconds=retention_seconds, stale_seconds=stale_seconds)
        self.domain_id = int(os.environ.get("ASSISTANT_ROS_DOMAIN_ID", "42")) if domain_id is None else domain_id
        if not isinstance(self.domain_id, int) or not 0 <= self.domain_id <= 232:
            raise ValueError("ROS domain must be 0..232")
        self.clock_offset = None
        self.failure = None
        self.thread = self.context = self.node = self.executor = None
        self.sync = self.subscribers = None
        self._source_times = {}
        self._arrival_times = {}
        self.pair_deltas_ms = []
        self.perception = GestureRules() if perception else None
        self.pose_subscription = None
        self.pose_offset = None
        self.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()

    @classmethod
    def configured(cls):
        settings = json.loads((Path(__file__).resolve().parents[2] / "configs/runtime.yaml").read_text())["vision"]
        return cls(frame_bytes_limit=settings["frame_bytes_limit"], retention_seconds=settings["frame_window_seconds"],
                   perception=True)

    def start(self):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from sensor_msgs.msg import Image
        from message_filters import TimeSynchronizer, Subscriber
        if self.perception:
            from yolo_msgs.msg import DetectionArray

        if self.thread is not None: raise RuntimeError("Vision bridge is already started")
        os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
        os.environ.pop("ROS_LOCALHOST_ONLY", None)
        self.context = Context()
        rclpy.init(context=self.context, domain_id=self.domain_id)
        self.node = Node("assistant_vision_bridge", context=self.context)
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.subscribers = [Subscriber(self.node, Image, topic, qos_profile=qos) for topic in (
            "/camera/camera/color/image_raw", "/camera/camera/aligned_depth_to_color/image_raw")]
        bridge = self

        class ResettableSynchronizer(TimeSynchronizer):
            def add(self, message, queue, index=None):
                if (message.width, message.height) != (640, 480) or len(message.data) > 2 * 1024 * 1024:
                    with bridge.frames.lock: bridge.frames.counts["unsupported_input_shape"] += 1
                    return
                now = time.monotonic_ns()
                source = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
                offset = bridge.node.get_clock().now().nanoseconds - now
                previous = bridge._source_times.get(index)
                last_arrival = bridge._arrival_times.get(index)
                reset = ((previous is not None and previous - source > bridge.frames.stale_ns)
                         or (last_arrival is not None and now - last_arrival > bridge.frames.stale_ns)
                         or (bridge.clock_offset is not None and abs(offset - bridge.clock_offset) > 100_000_000))
                if reset:
                    # Clear in place: upstream callbacks retain these queue
                    # objects. A source-clock reset must not strand new frames
                    # behind the prior epoch's newer timestamp keys.
                    with self.lock:
                        for cached in self.queues: cached.clear()
                    bridge.frames.reset("ros_input_epoch")
                    bridge._source_times.clear(); bridge._arrival_times.clear()
                bridge._source_times[index] = source
                bridge._arrival_times[index] = now
                bridge.clock_offset = offset
                super().add(message, queue, index)

        # The qualified RealSense aligned stream uses identical timestamps.
        # A 34 ms approximate window can incorrectly pair adjacent 30 Hz frames.
        self.sync = ResettableSynchronizer(self.subscribers, queue_size=3)
        self.sync.registerCallback(self._receive)
        if self.perception:
            self.pose_subscription = self.node.create_subscription(DetectionArray,
                '/pose/d435/detections_3d', self._receive_pose, qos)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)

        def spin():
            from rclpy.executors import ExternalShutdownException
            try: self.executor.spin()
            except ExternalShutdownException: pass
            except Exception as error:
                self.failure = f'{type(error).__name__}: {str(error)[:200]}'
                self.frames.reset("ros_executor_failed")
                if self.perception: self.perception.invalidate("ros_executor_failed")
        self.thread = threading.Thread(target=spin, name="assistant-ros-vision", daemon=True)
        self.thread.start()
        return self

    def _receive_pose(self, message, info):
        now = time.monotonic_ns()
        offset = self.node.get_clock().now().nanoseconds - now
        if self.pose_offset is not None and abs(offset-self.pose_offset)>100_000_000:
            self.perception.invalidate('ros_clock_changed')
        self.pose_offset = offset
        stamp = message.header.stamp.sec*1_000_000_000 + message.header.stamp.nanosec
        # This installed Jazzy rclpy exposes timestamps/sequence numbers but
        # omits publisher_gid from callback metadata. Bind to the single
        # discovered endpoint and reject ambiguous multi-publisher input.
        endpoints = self.node.get_publishers_info_by_topic('/pose/d435/detections_3d')
        if len(endpoints) != 1:
            self.perception.invalidate('ambiguous_pose_publisher')
            return
        publisher = hashlib.sha256(self.boot_id.encode()+bytes(endpoints[0].endpoint_gid)).hexdigest()
        if len(message.detections)>100:
            self.perception.invalidate('oversized_detection_array')
            return
        people = []
        for detection in message.detections:
            if detection.class_id != 0 or len(detection.keypoints.data)>17: continue
            z = detection.bbox3d.center.position.z
            # Only optical-frame Z denotes range along the camera's view axis.
            distance = z if ('optical' in detection.bbox3d.frame_id and math.isfinite(z) and 0<z<=20) else None
            people.append(Person(str(detection.id), detection.score, detection.bbox.center.position.x,
                detection.bbox.center.position.y, detection.bbox.size.x, detection.bbox.size.y,
                {p.id:(p.point.x,p.point.y,p.score) for p in detection.keypoints.data}, distance))
        self.perception.update(people, source_key=publisher, source_stamp_ns=stamp,
            capture_ns=stamp-offset, arrival_ns=now)

    def _receive(self, color, depth):
        now = time.monotonic_ns()
        color_stamp = color.header.stamp.sec * 1_000_000_000 + color.header.stamp.nanosec
        depth_stamp = depth.header.stamp.sec * 1_000_000_000 + depth.header.stamp.nanosec
        valid = (color.encoding in {"rgb8", "bgr8"} and depth.encoding == "16UC1" and not depth.is_bigendian
                 and color.header.frame_id == depth.header.frame_id and color.step == 640 * 3 and depth.step == 640 * 2)
        if not valid:
            with self.frames.lock: self.frames.counts["unaligned_or_unsupported_format"] += 1
            return
        if self.clock_offset is None: return
        accepted = self.frames.add(source_time_ns=color_stamp, capture_time_ns=color_stamp - self.clock_offset,
            arrival_time_ns=now, width=color.width, height=color.height, rgb_step=color.step,
            rgb_encoding=color.encoding, rgb=color.data, depth_step=depth.step, depth=depth.data,
            optical_frame=color.header.frame_id)
        if accepted:
            self.pair_deltas_ms.append(abs(color_stamp - depth_stamp) / 1e6)
            self.pair_deltas_ms = self.pair_deltas_ms[-1000:]

    def status(self):
        return {**self.frames.status(), "ros_domain": self.domain_id, "ros_failure": self.failure,
                "perception": self.perception.snapshot() if self.perception else None,
                "transport_reserve_bytes": self.transport_reserve,
                "synchronizer_queue_depth": 3, "dds_queue_depth": 2,
                "rgbd_pair_delta_max_ms": max(self.pair_deltas_ms) if self.pair_deltas_ms else None}

    def close(self):
        if self.perception: self.perception.invalidate('bridge_closed')
        if self.executor: self.executor.shutdown(timeout_sec=3)
        if self.context: self.context.try_shutdown()
        if self.thread:
            self.thread.join(3)
            if self.thread.is_alive(): raise RuntimeError("ROS vision thread did not stop")
        if self.node: self.node.destroy_node()
        self.thread = self.node = self.context = self.executor = None
        self.frames.reset("bridge_closed")
