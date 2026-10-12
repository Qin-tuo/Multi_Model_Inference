"""Export the locked YOLOv8 checkpoints and check raw CPU numerical parity."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.project import CACHE, download_file, sha256, stamp, write_json


def main():
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO, settings
    torch.set_num_threads(2)
    settings.update({"sync": False, "runs_dir": str(CACHE / "runs"), "datasets_dir": str(CACHE / "datasets")})
    locks = json.loads((ROOT / "deploy/vision/weights.lock.json").read_text())
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    reports = []
    for item in locks["files"]:
        checkpoint = CACHE / "models/yolo" / item["name"]
        download_file(item["url"], checkpoint, item["sha256"], item["size"])
        model = YOLO(str(checkpoint))
        output = Path(model.export(format="onnx", imgsz=640, opset=17, simplify=False, dynamic=False,
                                   batch=1, nms=False, half=False, device="cpu"))
        onnx.checker.check_model(str(output))
        rng = np.random.default_rng(20261012)
        image = rng.random((1,3,640,640),dtype=np.float32)
        model.model.eval()
        with torch.no_grad():
            expected = model.model(torch.from_numpy(image))
            if isinstance(expected,(tuple,list)): expected=expected[0]
            expected = expected.cpu().numpy()
        options = ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        session = ort.InferenceSession(str(output),options,providers=["CPUExecutionProvider"])
        actual = session.run(None,{session.get_inputs()[0].name:image})[0]
        error = float(np.max(np.abs(expected-actual)))
        passed = expected.shape==actual.shape and bool(np.allclose(expected,actual,rtol=1e-4,atol=0.01))
        task = "pose" if "pose" in item["name"] else "detection"
        row = {"task":task,"status":"PASS" if passed else "FAIL","source_sha256":item["sha256"],
               "onnx_sha256":sha256(output),"shape":list(actual.shape),"max_absolute_error":error,
               "input_shape":[1,3,640,640],"opset":17,"dynamic":False,"nms_in_graph":False,
               "scope":"synthetic input CPU Torch/ONNX export regression; not GPU or visual quality acceptance"}
        reports.append(row)
        manifest["models"][task] = {"model":item["name"].removesuffix(".pt"),"backend":"onnxruntime_tensorrt_pending",
                                    "checkpoint":item,"onnx":{"name":output.name,"sha256":row["onnx_sha256"],"size":output.stat().st_size},
                                    "input_shape":[1,3,640,640],"precision":"FP32 ONNX; TensorRT FP16 to validate",
                                    "exporter":"ultralytics8.4.175 / torch2.9.1+cpu / opset17",
                                    "license":"AGPL-3.0","status":"cpu_export_verified" if passed else "export_regression_failed"}
        print(json.dumps(row,ensure_ascii=False),flush=True)
    write_json(ROOT / "models/manifest.json",manifest)
    write_json(ROOT / "benchmarks/results/yolo-export.json",{"created_at":stamp(),"status":"PASS" if all(x["status"]=="PASS" for x in reports) else "FAIL","models":reports})
    return 0 if all(x["status"]=="PASS" for x in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
