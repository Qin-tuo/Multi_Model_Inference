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
    parser.add_argument("--discovery-seconds", type=float, default=5)
    args = parser.parse_args()
    if not 0<args.seconds<=30 or not 0<=args.discovery_seconds<=30:
        parser.error('Observation must be positive and bounded; discovery must be0..30seconds')
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
        started=time.monotonic()
        discovery_deadline=started+args.discovery_seconds
        while result is None and subscription.get_publisher_count()==0 and time.monotonic()<discovery_deadline:
            rclpy.spin_once(node,timeout_sec=.1)
        discovered=time.monotonic()
        matched=subscription.get_publisher_count()
        deadline = time.monotonic() + args.seconds
        while result is None and matched>0 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
        observation=result or {"ready": False, "topic": topic,
            "reason": "no_fresh_sample" if matched>0 else "no_matched_publisher"}
        observation.update(matched_publishers=matched,discovery_wait_ms=(discovered-started)*1000,
            observation_wait_ms=(time.monotonic()-discovered)*1000,
            freshness_limit_ms=2000)
        print(json.dumps(observation))
        return 0 if result else 1
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__": raise SystemExit(main())
