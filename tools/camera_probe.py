#!/usr/bin/env python3
"""Measure real ROS RGB-D arrival rates without recording camera images."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import statistics
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class Probe(Node):
    def __init__(self, color_topic, depth_topic):
        super().__init__("assistant_camera_probe")
        self.samples = {"color": [], "depth": []}
        self.formats = {}
        self.invalid_payloads = {"color": 0, "depth": 0}
        self.subscribers = [self.create_subscription(Image, topic, self.callback(name), qos_profile_sensor_data)
                            for name, topic in (("color", color_topic), ("depth", depth_topic))]

    def callback(self, name):
        def receive(message):
            now = time.monotonic_ns()
            source = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
            self.samples[name].append((now, source))
            self.formats[name] = {"width": message.width, "height": message.height,
                                  "step": message.step, "encoding": message.encoding}
            if len(message.data) != message.step * message.height:
                self.invalid_payloads[name] += 1
        return receive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--color-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/camera/depth/image_rect_raw")
    parser.add_argument("--output", default="benchmarks/results/camera-smoke.json")
    args = parser.parse_args()
    if not 3 <= args.seconds <= 300: parser.error("seconds must be 3..300")
    rclpy.init()
    node = Probe(args.color_topic, args.depth_topic)
    try:
        until = time.monotonic() + args.seconds
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=0.1)
        streams = {}
        for name, samples in node.samples.items():
            span = (samples[-1][0] - samples[0][0])/1e9 if len(samples) > 1 else 0
            hz = (len(samples)-1)/span if span else 0
            gaps = [(b[0]-a[0])/1e6 for a,b in zip(samples, samples[1:])]
            formats = node.formats.get(name, {})
            streams[name] = {"count": len(samples), "arrival_hz": hz, "format": formats,
                             "median_gap_ms": statistics.median(gaps) if gaps else None,
                             "max_gap_ms": max(gaps) if gaps else None,
                             "timestamps_nondecreasing": all(b[1] >= a[1] for a,b in zip(samples,samples[1:])),
                             "invalid_payloads": node.invalid_payloads[name]}
        passed = all(s["arrival_hz"] >= 27 and s["format"].get("width") == 640 and
                     s["format"].get("height") == 480 and s["timestamps_nondecreasing"] and
                     s["invalid_payloads"] == 0 for s in streams.values())
        report = {"schema_version": 1, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "status": "PASS" if passed else "FAIL", "input_mode": "live_camera", "seconds": args.seconds,
                  "scope": "RGB-D acquisition only; no detector/pose inference, synchronization quality or JT-03 acceptance",
                  "topics": {"color": args.color_topic, "depth": args.depth_topic}, "streams": streams}
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if passed else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
