#!/usr/bin/env python3
"""Measure real ROS RGB-D arrival rates without recording camera images."""
from __future__ import annotations

import argparse
from bisect import bisect_left
import datetime as dt
import json
from pathlib import Path
import statistics
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class Probe(Node):
    def __init__(self, color_topic, depth_topic, *, aligned=False):
        super().__init__("assistant_camera_probe")
        self.samples = {"color": [], "depth": []}
        self.formats = {}
        self.invalid_payloads = {"color": 0, "depth": 0}
        self.intrinsics = {}
        self.subscribers = [self.create_subscription(Image, topic, self.callback(name), qos_profile_sensor_data)
                            for name, topic in (("color", color_topic), ("depth", depth_topic))]
        if aligned:
            for name, topic in (("color", "/camera/camera/color/camera_info"),
                                ("depth", "/camera/camera/aligned_depth_to_color/camera_info")):
                self.subscribers.append(self.create_subscription(CameraInfo, topic,
                    lambda message, key=name: self.intrinsics.update({key: {
                        "width": message.width, "height": message.height, "frame_id": message.header.frame_id,
                        "positive_focal_lengths": bool(message.k[0] > 0 and message.k[4] > 0)}}), qos_profile_sensor_data))

    def callback(self, name):
        def receive(message):
            now = time.monotonic_ns()
            source = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
            self.samples[name].append((now, source))
            self.formats[name] = {"width": message.width, "height": message.height,
                                  "step": message.step, "encoding": message.encoding, "frame_id": message.header.frame_id}
            if len(message.data) != message.step * message.height:
                self.invalid_payloads[name] += 1
        return receive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--color-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/camera/depth/image_rect_raw")
    parser.add_argument("--output", default="benchmarks/results/camera-smoke.json")
    parser.add_argument("--require-aligned", action="store_true", help="Also require color-aligned depth and valid intrinsics")
    args = parser.parse_args()
    if not 3 <= args.seconds <= 300: parser.error("seconds must be 3..300")
    rclpy.init()
    node = Probe(args.color_topic, args.depth_topic, aligned=args.require_aligned)
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
                             "timestamps_nonzero": all(item[1] > 0 for item in samples),
                             "invalid_payloads": node.invalid_payloads[name]}
        passed = all(s["arrival_hz"] >= 27 and s["format"].get("width") == 640 and
                     s["format"].get("height") == 480 and s["timestamps_nondecreasing"] and
                     s["timestamps_nonzero"] and s["invalid_payloads"] == 0 for s in streams.values())
        alignment = None
        if args.require_aligned:
            depth_times = sorted(item[1] for item in node.samples["depth"])
            deltas = []
            for _, source_time in node.samples["color"]:
                index = bisect_left(depth_times, source_time)
                candidates = depth_times[max(0, index - 1):index + 1]
                if candidates: deltas.append(min(abs(source_time - candidate) for candidate in candidates) / 1e6)
            p95 = sorted(deltas)[int((len(deltas) - 1) * .95)] if deltas else None
            frames = [node.formats.get(name, {}).get("frame_id") for name in ("color", "depth")]
            aligned = (bool(frames[0]) and frames[0] == frames[1] and len(node.intrinsics) == 2
                and all(value["width"] == 640 and value["height"] == 480 and value["positive_focal_lengths"]
                        and value["frame_id"] == frames[0] for value in node.intrinsics.values())
                and p95 is not None and p95 <= 34)
            alignment = {"status": "PASS" if aligned else "FAIL", "nearest_source_timestamp_delta_p95_ms": p95,
                         "intrinsics": node.intrinsics, "scope": "coordinate-frame/format/intrinsics/timestamp contract; no physical depth accuracy ground truth"}
            passed = passed and aligned
        report = {"schema_version": 1, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "status": "PASS" if passed else "FAIL", "input_mode": "live_camera", "seconds": args.seconds,
                  "scope": "RGB-D acquisition only; no detector/pose inference, synchronization quality or JT-03 acceptance",
                  "topics": {"color": args.color_topic, "depth": args.depth_topic}, "streams": streams}
        if alignment is not None: report["alignment"] = alignment
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
