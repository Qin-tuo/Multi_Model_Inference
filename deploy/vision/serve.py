"""Launch the pinned RealSense/yolo_ros services using their native ROS entry points."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.project import CACHE, DATA, sha256, write_json


def ros_environment():
    env = os.environ.copy()
    domain = int(env.get("ASSISTANT_ROS_DOMAIN_ID", "42"))
    if not 0 <= domain <= 232: raise ValueError("ROS domain must be 0..232")
    env["ROS_DOMAIN_ID"] = str(domain)
    env["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    env.pop("ROS_LOCALHOST_ONLY", None)
    return env


def parameters(task):
    model = json.loads((ROOT / "models/manifest.json").read_text())["models"][task]
    path = CACHE / "models/yolo" / model["onnx"]["name"]
    if not path.is_file() or sha256(path) != model["onnx"]["sha256"]:
        raise RuntimeError("YOLO ONNX artifact does not match its manifest")
    namespace = "vision" if task == "detection" else "pose"
    camera = "d435"
    values = {
        "plugins": ["detection", "tracking", "detection3d"], "cameras": [camera],
        "d435.rgb_topic": "/camera/camera/color/image_raw",
        "d435.depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
        "d435.depth_info_topic": "/camera/camera/aligned_depth_to_color/camera_info",
        "d435.image_reliability": 2, "d435.depth_reliability": 2,
        "detection.plugin": "yolo_ros/DetectionPlugin", "detection.cameras": [camera],
        "detection.model_path": str(path), "detection.model_repo": "", "detection.model_filename": "",
        "detection.force_download": False, "detection.model_type": "detect" if task == "detection" else "pose",
        "detection.provider": "tensorrt", "detection.device": "tensorrt:0",
        "detection.trt_fp16_enable": False, "detection.cuda_graph_enable": False,
        "detection.trt_engine_cache_enable": True,
        "detection.trt_engine_cache_path": str(CACHE / "models/yolo/ort-trt-fp32" / task),
        "detection.threshold": 0.5, "detection.iou": 0.45, "detection.max_det": 100,
        "detection.n_threads": 2, "detection.max_fps": 15, "detection.max_batch_size": 1,
        "detection.img_width": 640, "detection.img_height": 640,
        "tracking.plugin": "yolo_ros/TrackingPlugin", "tracking.cameras": [camera],
        "tracking.tracker_type": "bytetrack", "tracking.track_high_thresh": 0.25,
        "tracking.track_low_thresh": 0.1, "tracking.new_track_thresh": 0.25,
        "tracking.track_buffer": 30, "tracking.match_thresh": 0.8, "tracking.fuse_score": True,
        "detection3d.plugin": "yolo_ros/Detect3DPlugin", "detection3d.cameras": [camera],
        "detection3d.target_frame": "camera_color_optical_frame",
        "detection3d.depth_image_units_divisor": 1000,
    }
    return namespace, {f"/{namespace}/yolo_node": {"ros__parameters": values}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=["camera", "detection", "pose"])
    parser.add_argument("--profile", action="store_true", help="Opt-in ORT raw provider trace, finalized on clean shutdown")
    parser.add_argument("--render-only", action="store_true", help="Write parameters without starting services")
    args = parser.parse_args()
    env = ros_environment()
    setups = [Path("/opt/ros/jazzy/setup.bash")]
    if args.component == "camera":
        command = ["ros2", "launch", "realsense2_camera", "rs_launch.py",
            "camera_namespace:=camera", "camera_name:=camera", "enable_color:=true", "enable_depth:=true",
            "enable_infra1:=false", "enable_infra2:=false", "enable_gyro:=false", "enable_accel:=false",
            "rgb_camera.color_profile:=640x480x30", "depth_module.depth_profile:=640x480x30",
            "align_depth.enable:=true", "enable_sync:=true", "pointcloud.enable:=false"]
    else:
        model_lease = None
        if not args.render_only:
            from deploy.resource_guard import hold_model_lease
            model_lease = hold_model_lease()
        namespace, params = parameters(args.component)
        path = DATA / "config" / f"ros-{args.component}.json"
        write_json(path, params)
        if args.render_only:
            print(json.dumps({"params_file": str(path), "ros_domain": env["ROS_DOMAIN_ID"], "scope": "configuration only"}))
            return 0
        prefix = CACHE / "install/onnxruntime-1.28.0-cuda13-sm87"
        required = [prefix / ".ort-build-info", *[prefix / "lib" / filename for filename in (
            "libonnxruntime.so", "libonnxruntime_providers_cuda.so", "libonnxruntime_providers_tensorrt.so")]]
        if not all(item.is_file() for item in required):
            raise RuntimeError("Full GPU ORT installation is not ready; a staged compile-only SDK cannot start this service")
        env["LD_LIBRARY_PATH"] = str(prefix / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        setups.append(CACHE / "workspaces/vision/install/setup.bash")
        if args.profile:
            profile = DATA / "traces" / f"ort-{args.component}"
            profile.parent.mkdir(parents=True, exist_ok=True)
            env["ASSISTANT_ORT_PROFILE_PREFIX"] = str(profile)
        command = ["ros2", "run", "yolo_ros", "yolo_node", "--ros-args", "-r", f"__ns:=/{namespace}",
                   "-r", "__node:=yolo_node", "--params-file", str(path)]
    if args.render_only:
        print(json.dumps({"command": command, "ros_domain": env["ROS_DOMAIN_ID"], "scope": "configuration only"}))
        return 0
    if not all(path.is_file() for path in setups):
        raise RuntimeError("Required ROS installation/setup file is missing")
    script = " && ".join("source " + shlex.quote(str(path)) for path in setups) + " && exec " + shlex.join(command)
    os.execve("/bin/bash", ["bash", "-c", script], env)


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
