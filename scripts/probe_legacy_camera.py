#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read-only ROS probe of the legacy RGB/depth service after camera handover.

Run with system Python after sourcing ROS Jazzy. Saves metadata, no images.
This is a migration verification tool, not a dependency of the new application.
"""

import argparse
import json
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rclpy.init()
    node = Node("reachy_recovery_restore_probe")
    samples = {"rgb": [], "depth": []}
    started = time.time()

    def receive(kind, message):
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        if (
            stamp >= started
            and len(samples[kind]) < 3
            and (not samples[kind] or samples[kind][-1]["stamp"] != stamp)
        ):
            samples[kind].append(
                {
                    "stamp": stamp,
                    "age_ms": (time.time() - stamp) * 1000,
                    "width": message.width,
                    "height": message.height,
                    "encoding": message.encoding,
                }
            )

    for kind, topic in [
        ("rgb", "/camera/camera/color/image_raw"),
        ("depth", "/camera/camera/aligned_depth_to_color/image_raw"),
    ]:
        node.create_subscription(
            Image, topic, lambda msg, k=kind: receive(k, msg), qos_profile_sensor_data
        )
    try:
        end = time.monotonic() + 40
        while time.monotonic() < end and not all(len(v) >= 3 for v in samples.values()):
            rclpy.spin_once(node, timeout_sec=0.2)
        result = {
            "fresh": all(
                len(v) >= 3 and 0 <= v[-1]["age_ms"] < 2000 for v in samples.values()
            ),
            "samples": samples,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if not result["fresh"]:
            raise RuntimeError("Legacy RGB/depth restoration not verified")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
