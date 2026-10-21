#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export the pinned official YOLOv8n-seg weights with Ultralytics, on CPU.

Run the controller with the application Python and pass an export environment
with --python. That environment is a build tool, not a production dependency.
No auto-installation, FP16, NMS graph insertion or graph simplification occurs.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "01_src"))
WEIGHTS = ROOT / "05_runtime/models/b1/detection_segmentation/yolov8n-seg.pt"
WEIGHT_SHA = "a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c"
VERSIONS = {
    "torch": "2.9.1+cpu",
    "ultralytics": "8.4.175",
    "onnx": "1.23.2",
    "numpy": "1.26.4",
    "opencv-python-headless": "4.11.0.86",
}


def canonicalize(raw):
    """Remove only wall-clock metadata so repeat exports can retain stable hashes."""
    import onnx

    model = onnx.load(raw)
    graph_before = model.graph.SerializeToString()
    metadata = sorted(
        (item.key, item.value) for item in model.metadata_props if item.key != "date"
    )
    del model.metadata_props[:]
    for key, value in metadata:
        model.metadata_props.add(key=key, value=value)
    target = raw.with_name(raw.stem + "-deploy.onnx")
    if target.exists():
        raise FileExistsError(target)
    onnx.save(model, target)
    if onnx.load(target).graph.SerializeToString() != graph_before:
        raise ValueError("Canonicalization changed the inference graph")
    return target


def worker(output, weights):
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "05_runtime/builds/ultralytics-config")
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS:
        raise RuntimeError(
            "Export tool versions differ from the reviewed environment: "
            + repr(versions)
        )
    if (
        weights.name != "yolov8n-seg.pt"
        or hashlib.sha256(weights.read_bytes()).hexdigest() != WEIGHT_SHA
    ):
        raise ValueError("Official segmentation weight checksum mismatch")
    if weights.with_suffix(".onnx").exists():
        raise FileExistsError("Refusing to overwrite an existing exported ONNX")
    import torch
    import ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    kwargs = {
        "format": "onnx",
        "imgsz": [640, 640],
        "batch": 1,
        "device": "cpu",
        "half": False,
        "dynamic": False,
        "simplify": False,
        "opset": 17,
        "nms": False,
    }
    raw = Path(YOLO(str(weights)).export(**kwargs))
    exported = canonicalize(raw)
    record = {
        "status": "EXPORTED_NOT_ACCEPTED",
        "weights_sha256": WEIGHT_SHA,
        "versions": versions,
        "export_arguments": kwargs,
        "onnx_path": str(exported.relative_to(ROOT)),
        "onnx_sha256": hashlib.sha256(exported.read_bytes()).hexdigest(),
        "onnx_bytes": exported.stat().st_size,
        "raw_export_path": str(raw.relative_to(ROOT)),
        "raw_export_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "canonicalization": "Only remove date metadata and sort metadata keys; graph bytes verified unchanged",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_source_sha256": {},
    }
    package = Path(ultralytics.__file__).parent
    for name in (
        "engine/exporter.py",
        "utils/ops.py",
        "utils/nms.py",
        "data/augment.py",
        "models/yolo/segment/predict.py",
    ):
        path = package / name
        record["package_source_sha256"][name] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (output / "export.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / "export-environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        default=WEIGHTS,
        help="Pinned weights; copy to a new directory for an independent repeat export",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=ROOT / "05_runtime/builds/vision-export/bin/python",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        worker(output, args.weights.resolve())
        return 0
    if not args.python.is_file():
        parser.error("Provide the pinned CPU export environment using --python")
    from deploy.operations import exclusive_build
    from deploy import runtime, services

    output.mkdir(parents=True, exist_ok=False)
    cfg = runtime.config()
    if services.profile_path(cfg).exists():
        cfg = runtime.config(services.profile_path(cfg))
    with exclusive_build(cfg, output):
        with (output / "export.log").open("w") as log:
            process = subprocess.run(
                [
                    str(args.python),
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--weights",
                    str(args.weights.resolve()),
                    "--output",
                    str(output),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=240,
            )
        runtime.write_json(
            output / "worker.json",
            {"returncode": process.returncode, "python": str(args.python)},
        )
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
