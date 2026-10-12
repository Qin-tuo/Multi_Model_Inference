"""Observe a fresh ROS sample as a readiness check, without saving images."""
import argparse
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=["camera", "vision", "pose"])
    parser.add_argument("--seconds", type=float, default=3)
    args = parser.parse_args()
    if args.component == "camera":
        from sensor_msgs.msg import Image
        topic, message_type = "/camera/camera/color/image_raw", Image
    else:
        from yolo_msgs.msg import DetectionArray
        topic, message_type = f"/{args.component}/d435/detections_3d", DetectionArray
    rclpy.init()
    node = Node("assistant_readiness_probe")
    result = None
    def receive(message):
        nonlocal result
        timestamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        age = (node.get_clock().now().nanoseconds - timestamp) / 1e6
        if timestamp <= 0 or not 0 <= age <= 2000: return
        if args.component == "camera" and (message.width, message.height, message.encoding) != (640, 480, "rgb8"):
            return
        result = {"ready": True, "topic": topic, "capture_age_ms": age,
                  "scope": "fresh ROS message; not model accuracy or GPU-placement verification"}
    subscription = node.create_subscription(message_type, topic, receive, qos_profile_sensor_data)
    try:
        deadline = time.monotonic() + args.seconds
        while result is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
        print(json.dumps(result or {"ready": False, "topic": topic, "reason": "no_fresh_sample"}))
        return 0 if result else 1
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__": raise SystemExit(main())
