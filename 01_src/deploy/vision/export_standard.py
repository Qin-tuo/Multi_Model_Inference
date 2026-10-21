#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export the three pinned standard replacements and capture native references."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "01_src"))
from deploy import runtime, services
from deploy.operations import exclusive_build

MANIFEST = ROOT / "03_models/vision-models.json"
ROLES = ("body_hand_pose", "face_identity", "person_reid")


def native_cases(role, model, extras, folder):
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from torchvision import transforms

    protocol = json.loads(
        (ROOT / "benchmarks/manifests/b2-vision-measurements-v1.json").read_text()
    )
    aligner = None
    if role == "face_identity":
        aligner = cv2.FaceRecognizerSF.create(
            str(
                ROOT
                / "05_runtime/models/b1/face_identity/face_recognition_sface_2021dec.onnx"
            ),
            "",
        )
    rows = []
    for index, case in enumerate(protocol["module_cases"][role]):
        item = protocol["inputs"][case["input"]]
        path = ROOT / item["path"]
        if runtime.sha256(path) != item["sha256"]:
            raise ValueError("Frozen source image changed")
        image = np.load(path, allow_pickle=False)
        args = case["arguments"]
        extra = {}
        if role == "body_hand_pose":
            inputs, center, scale = extras["prepare"](image, args[0])
            extra.update(center=center, scale=scale)
        elif role == "person_reid":
            from app.vision_roi import crop_roi

            crop, roi = crop_roi(image, args[0])
            # The network and evaluation transform are the upstream reference;
            # the integer/clamped ROI is the application's explicit boundary.
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        (256, 128), interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            inputs = transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))[
                None
            ]
            extra["roi"] = roi
        else:
            landmarks = np.array(args[0], np.float32)
            face = np.r_[
                0, 0, image.shape[1], image.shape[0], landmarks.ravel(), 1
            ].astype(np.float32)
            crop = aligner.alignCrop(image, face)
            photo = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            photo = extras["image_utils"].resize_image(photo, (112, 112), True)
            values = extras["image_utils"].preprocess_input(np.array(photo, np.float32))
            inputs = torch.from_numpy(
                np.ascontiguousarray(values.transpose(2, 0, 1)[None])
            )
            extra["aligned_bgr"] = crop
        with torch.inference_mode():
            result = model(inputs)
        outputs = tuple(result) if isinstance(result, (tuple, list)) else (result,)
        arrays = [value.detach().numpy() for value in outputs]
        if role == "body_hand_pose":
            points, scores = extras["codec"].decode(*arrays)
            extra["decoded_xy"] = (
                points / np.array(extras["input_size"]) * scale + center - 0.5 * scale
            )
            # RTMW's official codec additionally returns visibility. The
            # application uses the original peak score for its existing gate.
            if isinstance(scores, tuple):
                extra["decoded_scores"], extra["decoded_visibility"] = scores
            else:
                extra["decoded_scores"] = scores
        target = folder / f"case-{index}.npz"
        np.savez(
            target,
            source=image,
            input=inputs.numpy(),
            **extra,
            **{f"output_{i}": a for i, a in enumerate(arrays)},
        )
        rows.append(
            {
                **case,
                "fixture": str(target.relative_to(ROOT)),
                "sha256": runtime.sha256(target),
            }
        )
    return rows


def worker(role, output):
    import importlib.metadata
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from deploy.vision.standard_reference import load_reference

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    entry = json.loads(MANIFEST.read_text())["models"][role]
    model, extras, loaded = load_reference(role, entry)
    target = ROOT / entry["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError("Keep the previous export; use a new revision")
    input_name, shape = next(iter(entry["input_shapes"].items()))
    dummy = torch.zeros(shape, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy,),
        str(target),
        input_names=[input_name],
        output_names=entry["output_names"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    exported = onnx.load(target)
    onnx.checker.check_model(exported)
    cases = native_cases(role, model, extras, output)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(
        str(target), options, providers=["CPUExecutionProvider"]
    )
    comparisons = []
    for case in cases:
        with np.load(ROOT / case["fixture"], allow_pickle=False) as fixture:
            actual = session.run(entry["output_names"], {input_name: fixture["input"]})
            for index, value in enumerate(actual):
                expected = fixture[f"output_{index}"]
                passed = bool(np.allclose(value, expected, atol=1e-3, rtol=1e-3))
                comparisons.append(
                    {
                        "input": case["input"],
                        "output": entry["output_names"][index],
                        "max_abs_error": float(np.abs(value - expected).max()),
                        "passed": passed,
                    }
                )
                if not passed:
                    raise ValueError("ONNX differs from the native reference")

    def tensor(value):
        return {
            "name": value.name,
            "shape": [
                d.dim_value if d.HasField("dim_value") else d.dim_param
                for d in value.type.tensor_type.shape.dim
            ],
            "dtype": onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type),
        }

    report = {
        "status": "EXPORTED_NATIVE_ONNX_FIXED_INPUT_PARITY_PASS",
        "role": role,
        "native_load": loaded,
        "path": str(target.relative_to(ROOT)),
        "sha256": runtime.sha256(target),
        "bytes": target.stat().st_size,
        "observed_onnx_contract": {
            "inputs": [tensor(x) for x in exported.graph.input],
            "outputs": [tensor(x) for x in exported.graph.output],
        },
        "cases": cases,
        "comparisons": comparisons,
        "versions": {
            k: importlib.metadata.version(k)
            for k in ("torch", "onnx", "onnxruntime", "numpy", "torchvision")
        },
        "script_sha256": runtime.sha256(Path(__file__)),
        "scope": "Unmodified upstream pretrained network; no TensorRT or field-accuracy claim yet",
    }
    runtime.write_json(output / "export.json", report)
    print(role, report["status"], report["bytes"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=ROLES, nargs="+", default=list(ROLES))
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.role[0], args.output)
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = runtime.config()
    if services.profile_path(cfg).exists():
        cfg = runtime.config(services.profile_path(cfg))
    python = ROOT / "05_runtime/builds/vision-export/bin/python"
    with exclusive_build(cfg, output):
        for role in args.role:
            folder = output / role
            folder.mkdir()
            with (folder / "worker.log").open("w") as log:
                done = subprocess.run(
                    [
                        str(python),
                        __file__,
                        "--worker",
                        "--role",
                        role,
                        "--output",
                        str(folder),
                    ],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=360,
                )
            runtime.write_json(folder / "worker.json", {"returncode": done.returncode})
            if done.returncode:
                raise RuntimeError(f"Export/reference failed: {folder / 'worker.log'}")
            report = json.loads((folder / "export.json").read_text())
            candidate_path = output / "model-candidate.json"
            manifest = json.loads((candidate_path if candidate_path.exists() else MANIFEST).read_text())
            entry = manifest["models"][role]
            for key in ("path", "sha256", "bytes", "observed_onnx_contract"):
                entry[key] = report[key]
            entry["input_contract_status"] = report["status"]
            entry["export_report"] = str((folder / "export.json").relative_to(ROOT))
            entry.pop("validation_summary", None)
            manifest["status"] = "EXPORTED_REFERENCE_AND_INTEGRATION_PENDING"
            runtime.write_json(candidate_path, manifest)
            print(role, report["status"], flush=True)


if __name__ == "__main__":
    main()
