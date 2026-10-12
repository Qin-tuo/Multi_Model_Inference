"""Compare actual TensorRT and CPU ONNX detections on an upstream image."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.project import CACHE, sha256, stamp, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-engine", action="store_true", help="Execute an existing engine and record its hash")
    args = parser.parse_args()
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ASSETS
    from ultralytics.utils.metrics import box_iou
    from ultralytics.utils.nms import non_max_suppression

    torch.set_num_threads(2)
    model = json.loads((ROOT / "models/manifest.json").read_text())["models"]["detection"]
    onnx = CACHE / "models/yolo" / model["onnx"]["name"]
    if sha256(onnx) != model["onnx"]["sha256"]:
        raise RuntimeError("ONNX hash does not match the locked model")
    fixture = CACHE / "fixtures/vision"
    fixture.mkdir(parents=True, exist_ok=True)
    source = ASSETS / "bus.jpg"
    image = cv2.imread(str(source))
    image = LetterBox(new_shape=(640, 640), auto=False)(image=image)
    tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    binary = fixture / "bus-input.bin"
    tensor.tofile(binary)
    engine = CACHE / "models/yolo/yolov8n-fp16.engine"
    trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    output = fixture / "trt-output.json"
    output.unlink(missing_ok=True)
    command = [trtexec]
    if args.reuse_engine:
        if not engine.is_file():
            raise RuntimeError("No existing engine to execute")
        command += [f"--loadEngine={engine}"]
    else:
        command += [f"--onnx={onnx}", f"--saveEngine={engine}", "--fp16", "--memPoolSize=workspace:512"]
    command += [f"--loadInputs=images:{binary}", "--warmUp=0", "--duration=0", "--iterations=1",
                f"--exportOutput={output}"]
    log = CACHE / "yolo-engine-smoke.log"
    with log.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=900)

    tensors = json.loads(output.read_text())
    if len(tensors) != 1:
        raise RuntimeError("Expected exactly one detection output")
    actual = np.array(tensors[0]["values"], dtype=np.float32).reshape(1, 84, 8400)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx), options, providers=["CPUExecutionProvider"])
    expected = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    detections = [non_max_suppression(torch.from_numpy(raw.copy()), conf_thres=0.5,
                  iou_thres=0.45, nc=80, max_time_img=2)[0] for raw in (expected, actual)]
    reference, gpu = detections
    matches = []
    remaining = set(range(len(gpu)))
    for box in reference:
        candidates = [index for index in remaining if gpu[index, 5] == box[5]]
        if not candidates:
            continue
        overlap = box_iou(box[None, :4], gpu[candidates, :4])[0]
        best = int(overlap.argmax())
        index = candidates[best]
        remaining.remove(index)
        matches.append({"class_id": int(box[5]), "iou": float(overlap[best]),
                        "confidence_difference": abs(float(box[4] - gpu[index, 4]))})
    passed = (len(reference) > 0 and len(matches) == len(reference) == len(gpu)
              and all(row["iou"] >= 0.98 and row["confidence_difference"] <= 0.03 for row in matches)
              and bool(np.isfinite(actual).all()))
    report = {"created_at": stamp(), "status": "PASS" if passed else "FAIL",
              "scope": "one public upstream image; TensorRT engine output vs CPU ONNX using Ultralytics NMS; not ROS, camera quality, or performance acceptance",
              "input": {"source": "ultralytics/assets/bus.jpg", "sha256": sha256(source),
                        "tensor_sha256": sha256(binary), "shape": list(tensor.shape)},
              "onnx_sha256": sha256(onnx), "engine_sha256": sha256(engine),
              "engine_reused": args.reuse_engine, "executor": "NVIDIA TensorRT trtexec",
              "trtexec_passed": "PASSED TensorRT.trtexec" in log.read_text(),
              "nms": {"confidence": 0.5, "iou": 0.45},
              "reference_count": len(reference), "tensorrt_count": len(gpu),
              "class_counts": dict(Counter(int(x[5]) for x in gpu)), "matches": matches}
    write_json(ROOT / "benchmarks/results/yolo-engine-smoke.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed and report["trtexec_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
