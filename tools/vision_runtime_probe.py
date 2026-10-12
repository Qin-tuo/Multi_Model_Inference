"""Sample continuous ROS vision result age, including intervals with no result."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.metrics import sampled_age_summary

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from yolo_msgs.msg import DetectionArray


class Probe(Node):
    def __init__(self):
        super().__init__("assistant_vision_runtime_probe")
        self.rows = {name: {"arrivals": [], "latest": None, "sampled_ages": [], "missing": 0,
                           "invalid_clock": 0, "labels": Counter(), "tracked": 0, "valid_depth": 0,
                           "keypoint_messages": 0} for name in ("camera", "detection", "pose")}
        self.started_ns = time.monotonic_ns()
        self.measuring = False
        self.subscribers = []
        for name, topic, kind in (("camera", "/camera/camera/color/image_raw", Image),
                                  ("detection", "/vision/d435/detections_3d", DetectionArray),
                                  ("pose", "/pose/d435/detections_3d", DetectionArray)):
            self.subscribers.append(self.create_subscription(kind, topic, self.callback(name), qos_profile_sensor_data))
        self.timer = self.create_timer(.05, self.sample)

    def callback(self, name):
        def receive(message):
            row = self.rows[name]
            now = time.monotonic_ns()
            source = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
            age_ns = self.get_clock().now().nanoseconds - source
            if source <= 0 or not 0 <= age_ns <= 5_000_000_000:
                row["invalid_clock"] += 1
                return
            # Map the source age into this consumer's monotonic clock domain.
            row["latest"] = now - age_ns
            if not self.measuring: return
            row["arrivals"].append(now)
            if name != "camera":
                for detection in message.detections:
                    row["labels"][detection.class_name] += 1
                    row["tracked"] += bool(detection.id)
                    row["valid_depth"] += detection.bbox3d.center.position.z > 0
                    row["keypoint_messages"] += bool(detection.keypoints.data)
        return receive

    def sample(self):
        if not self.measuring: return
        now = time.monotonic_ns()
        for row in self.rows.values():
            if row["latest"] is None: row["missing"] += 1
            else: row["sampled_ages"].append((now - row["latest"]) / 1e6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--startup-timeout", type=float, default=15)
    parser.add_argument("--scenario", choices=["standalone", "file_speech"], default="standalone")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/vision-runtime-smoke.json"))
    args = parser.parse_args()
    if not 5 <= args.seconds <= 300: parser.error("seconds must be 5..300")
    rclpy.init()
    node = Probe()
    try:
        setup_started = time.monotonic()
        while not all(row["latest"] is not None and time.monotonic_ns() - row["latest"] < 500_000_000 for row in node.rows.values()):
            if time.monotonic() - setup_started > args.startup_timeout:
                raise RuntimeError("The three ROS streams did not become ready before the sampling window")
            rclpy.spin_once(node, timeout_sec=.05)
        startup_seconds = time.monotonic() - setup_started
        node.started_ns = time.monotonic_ns()
        node.measuring = True
        end = time.monotonic() + args.seconds
        while time.monotonic() < end: rclpy.spin_once(node, timeout_sec=.05)
        elapsed = (time.monotonic_ns() - node.started_ns) / 1e9
        rows = {}
        for name, row in node.rows.items():
            rows[name] = {"received": len(row["arrivals"]), "effective_hz": len(row["arrivals"]) / elapsed,
                **sampled_age_summary(row["sampled_ages"], row["missing"]),
                "invalid_source_clock_messages": row["invalid_clock"], "class_observations": dict(row["labels"]),
                "tracked_observations": row["tracked"], "positive_depth_observations": row["valid_depth"],
                "observations_with_keypoints": row["keypoint_messages"]}
        checks = {"camera_rate": rows["camera"]["effective_hz"] >= 27,
                  "vision_rates": all(rows[name]["effective_hz"] >= 10 for name in ("detection", "pose")),
                  "vision_freshness": all(rows[name]["latest_age_p95_ms"] is not None
                      and rows[name]["latest_age_p95_ms"] <= 200 and rows[name]["no_result_fraction"] <= .05
                      and rows[name]["invalid_source_clock_messages"] == 0 for name in ("detection", "pose"))}
        report = {"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "PASS" if all(checks.values()) else "FAIL", "input_mode": "live_camera", "seconds": elapsed,
            "subscription_startup_seconds": startup_seconds,
            "scenario": args.scenario, "measurement_start_monotonic_ns": node.started_ns,
            "measurement_end_monotonic_ns": time.monotonic_ns(),
            "scope": "camera/detector/pose transport timing in the declared workload; no ground-truth quality or final JT-03 acceptance",
            "age_measurement": "periodic latest-result age every50ms; missing periods explicitly included", "checks": checks, "streams": rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "PASS" else 1
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__": raise SystemExit(main())
