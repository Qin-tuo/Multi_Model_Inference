#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export pinned remaining visual models using their native model definitions.

PyTorch and OpenVINO run only in separate build/reference environments. Every
production neural graph is subsequently built and executed with TensorRT.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_src"))
MANIFEST = ROOT / "03_models/candidates/vision-b1.json"


def worker(role, output):
    import onnx
    from deploy.vision.export_models import checked_sources, torch_model

    recipe = json.loads(MANIFEST.read_text())["export_sources"][role]
    checked_sources(recipe)
    target = ROOT / recipe["directory"] / recipe["onnx_name"]
    if target.exists():
        raise FileExistsError(
            "Refusing to replace an exported candidate: " + str(target)
        )
    if recipe["kind"] == "pytorch":
        import torch

        versions = {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "onnx",
                "pillow",
                "opencv-python-headless",
            )
        }
        expected = {
            "torch": "2.9.1+cpu",
            "torchvision": "0.24.1",
            "numpy": "1.26.4",
            "onnx": "1.23.2",
            "pillow": "12.3.0",
            "opencv-python-headless": "4.11.0.86",
        }
        if versions != expected:
            raise RuntimeError("PyTorch export environment differs from its lock")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        model = torch_model(role, recipe)
        inputs = tuple(
            torch.zeros(shape, dtype=torch.float32)
            for shape in recipe["input_shapes"].values()
        )
        with torch.no_grad():
            torch.onnx.export(
                model,
                inputs,
                str(target),
                input_names=list(recipe["input_shapes"]),
                output_names=recipe["output_names"],
                opset_version=17,
                dynamo=False,
                do_constant_folding=True,
            )
    else:
        import openvino2onnx
        from deploy.vision.onnx_compat import normalize_omz

        versions = {
            name: importlib.metadata.version(name)
            for name in (
                "openvino",
                "openvino2onnx",
                "numpy",
                "onnx",
                "onnx-graphsurgeon",
            )
        }
        expected = {
            "openvino": "2026.4.1",
            "openvino2onnx": "1.1.0",
            "numpy": "2.1.3",
            "onnx": "1.17.0",
            "onnx-graphsurgeon": "0.5.8",
        }
        if versions != expected:
            raise RuntimeError("OpenVINO conversion environment differs from its lock")
        graph = openvino2onnx.convert_graph(
            str(ROOT / recipe["source_xml"]),
            target_opset=17,
            strict=True,
            print_passes=False,
        )
        graph.save(str(target))
        target, changes = normalize_omz(target, ROOT / recipe["source_xml"])
    model = onnx.load(target)
    onnx.checker.check_model(model)
    if any(node.domain not in ("", "ai.onnx") for node in model.graph.node):
        raise ValueError("Custom operator domain remains after conversion")
    initializers = {v.name for v in model.graph.initializer}

    def tensor(value):
        return {
            "name": value.name,
            "dtype": onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type),
            "shape": [
                d.dim_value or d.dim_param for d in value.type.tensor_type.shape.dim
            ],
        }

    import hashlib

    record = {
        "status": "EXPORTED_NOT_ACCEPTED",
        "role": role,
        "path": str(target.relative_to(ROOT)),
        "bytes": target.stat().st_size,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "versions": versions,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compatibility_changes": changes if recipe["kind"] == "openvino_ir" else [],
        "observed_onnx_contract": {
            "inputs": [
                tensor(v) for v in model.graph.input if v.name not in initializers
            ],
            "outputs": [tensor(v) for v in model.graph.output],
            "nodes": len(model.graph.node),
            "opsets": {v.domain: v.version for v in model.opset_import},
            "producer": model.producer_name,
            "producer_version": model.producer_version,
        },
    }
    (output / "export.json").write_text(json.dumps(record, indent=2) + "\n")
    print(role, record["status"], record["bytes"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role",
        nargs="+",
        choices=(
            "head_pose",
            "gaze",
            "gaze_landmarks",
            "person_reid",
            "object_embedding",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        worker(args.role[0], output)
        return 0
    from deploy.operations import exclusive_build
    from deploy import runtime, services

    output.mkdir(parents=True, exist_ok=False)
    cfg = runtime.config()
    if services.profile_path(cfg).exists():
        cfg = runtime.config(services.profile_path(cfg))
    failed = []
    with exclusive_build(cfg, output):
        for role in args.role:
            folder = output / role
            folder.mkdir()
            interpreter = (
                ROOT
                / "05_runtime/builds"
                / (
                    "openvino-tools"
                    if role in ("head_pose", "gaze", "gaze_landmarks")
                    else "vision-export"
                )
                / "bin/python"
            )
            with (folder / "export.log").open("w") as log:
                result = subprocess.run(
                    [
                        str(interpreter),
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--role",
                        role,
                        "--output",
                        str(folder),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                )
            runtime.write_json(
                folder / "worker.json",
                {"returncode": result.returncode, "python": str(interpreter)},
            )
            if result.returncode:
                failed.append(role)
                print(role, "EXPORT_FAILED", flush=True)
                continue
            record = json.loads((folder / "export.json").read_text())
            manifest = json.loads(MANIFEST.read_text())
            recipe = manifest["export_sources"][role]
            manifest["models"][role] = {
                **{
                    k: recipe[k]
                    for k in (
                        "name",
                        "repository",
                        "revision",
                        "license",
                        "model_card_url",
                        "model_card_sha256",
                        "prepost_source",
                    )
                },
                "artifact_kind": "local_export",
                "filename": Path(record["path"]).name,
                "path": record["path"],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "observed_onnx_contract": record["observed_onnx_contract"],
                "export_record": str((folder / "export.json").relative_to(ROOT)),
                "input_contract_status": "SOURCE_EXPORTED_TRT_AND_REFERENCE_PENDING",
                "preprocessing": {"status": "adapter_contract_pending"},
                "reference": "Native OpenVINO FP32 CPU"
                if recipe["kind"] == "openvino_ir"
                else "Pinned upstream PyTorch FP32 CPU",
            }
            runtime.write_json(MANIFEST, manifest)
            print(role, record["status"], record["bytes"], flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
