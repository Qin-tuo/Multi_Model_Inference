#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Native model parity for appearance features and the head/eye/gaze cascade.

All neural inference in the application is TensorRT. OpenVINO and PyTorch
are isolated references here. Fixed public fixtures do not establish field
accuracy, human identity error rates or universal object re-identification.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_src"))
MANIFEST = ROOT / "03_models/candidates/vision-b1.json"
CRITERIA = {
    "raw_atol": 0.001,
    "raw_rtol": 0.001,
    "input_atol": 1e-6,
    "embedding_min_cosine": 0.9999,
    "head_max_error_deg": 0.01,
    "landmark_max_error_px": 1,
    "cascade_max_angle_deg": 1.0,
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def source_fixtures(output):
    from probe_face_tasks_b1 import fixtures
    from app.vision_roi import crop_roi

    rows = []
    for name, image, face, provenance in fixtures(
        ROOT / "05_runtime/validation/b1-face-20261019"
    ):
        crop, roi = crop_roi(image, face["box_xyxy"])
        key = "face-" + name
        np.savez(output / (key + ".npz"), image=image, crop=crop, roi=np.array(roi))
        rows.append({"name": key, "group": "face", "provenance": provenance})
    base = (
        ROOT
        / "05_runtime/validation/b1-segmentation-20261019/parity-opencv411/reference"
    )
    for filename in ("public-person", "bus"):
        with np.load(base / (filename + ".npz"), allow_pickle=False) as data:
            image = data["source"]
            boxes = data["boxes"]
        selected = np.flatnonzero(boxes[:, 5] == 0)[:2]
        for index in selected:
            crop, roi = crop_roi(image, boxes[index, :4])
            key = f"person-{filename}-{index}"
            np.savez(output / (key + ".npz"), image=image, crop=crop, roi=np.array(roi))
            rows.append(
                {
                    "name": key,
                    "group": "person",
                    "provenance": {
                        "source": str(base / (filename + ".npz")),
                        "box_index": int(index),
                    },
                }
            )
        if filename == "bus":
            index = int(np.flatnonzero(boxes[:, 5] == 5)[0])
            for suffix, factor in (("normal", 1.0), ("dimmed", 0.7)):
                photo = (
                    np.rint(image.astype(np.float32) * factor)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                crop, roi = crop_roi(photo, boxes[index, :4])
                key = "object-bus-" + suffix
                np.savez(
                    output / (key + ".npz"), image=photo, crop=crop, roi=np.array(roi)
                )
                rows.append(
                    {
                        "name": key,
                        "group": "object",
                        "provenance": {
                            "source": str(base / "bus.npz"),
                            "box_index": index,
                            "brightness": factor,
                        },
                    }
                )
    for row in rows:
        path = output / (row["name"] + ".npz")
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(output / "fixtures.json", rows)
    return rows


def native_worker(kind, output):
    rows = json.loads((output / "fixtures/fixtures.json").read_text())
    manifest = json.loads(MANIFEST.read_text())
    if kind == "appearance":
        import torch
        from PIL import Image
        from torchvision import transforms
        from deploy.vision.export_models import torch_model

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        for role in ("person_reid", "object_embedding"):
            model = torch_model(role, manifest["export_sources"][role])
            if role == "person_reid":
                transform = transforms.Compose(
                    [
                        transforms.Resize((256, 128)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                        ),
                    ]
                )
            else:
                from clip.clip import _transform

                transform = _transform(224)
            for row in rows:
                if row["group"] != ("person" if role == "person_reid" else "object"):
                    continue
                with np.load(
                    output / "fixtures" / (row["name"] + ".npz"), allow_pickle=False
                ) as f:
                    crop = f["crop"]
                image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                tensor = transform(image)[None]
                with torch.no_grad():
                    value = model(tensor).cpu().numpy()
                np.savez(
                    output / (role + "-" + row["name"] + ".npz"),
                    input=tensor.numpy(),
                    embedding=value,
                )
            del model
    else:
        import openvino as ov
        from app.head_gaze import (
            prepare_face_attribute,
            landmark_pixels,
            prepare_gaze,
            decode_gaze,
        )

        core = ov.Core()
        models = {}
        for role in ("head_pose", "gaze_landmarks", "gaze"):
            recipe = manifest["export_sources"][role]
            models[role] = core.compile_model(
                str(ROOT / recipe["source_xml"]),
                "CPU",
                {
                    "INFERENCE_PRECISION_HINT": "f32",
                    "INFERENCE_NUM_THREADS": 2,
                    "NUM_STREAMS": 1,
                },
            )
        native_cascade = []
        for row in rows:
            if row["group"] != "face":
                continue
            with np.load(
                output / "fixtures" / (row["name"] + ".npz"), allow_pickle=False
            ) as f:
                image = f["image"]
                box = f["roi"]
            inputs, roi = prepare_face_attribute(image, box)
            head = models["head_pose"](inputs)
            values = {
                name: head[models["head_pose"].output(name)]
                for name in ("fc_y", "fc_p", "fc_r")
            }
            angles = np.array(
                [values[name].item() for name in ("fc_y", "fc_p", "fc_r")], np.float32
            )
            np.savez(
                output / ("head_pose-" + row["name"] + ".npz"),
                input=inputs["data"],
                **values,
            )
            raw_landmarks = next(iter(models["gaze_landmarks"](inputs).values()))
            landmarks = landmark_pixels(raw_landmarks, roi)
            np.savez(
                output / ("gaze_landmarks-" + row["name"] + ".npz"),
                input=inputs["data"],
                landmarks=raw_landmarks,
            )
            gaze_inputs, transform = prepare_gaze(image, landmarks, angles)
            gaze = next(iter(models["gaze"](gaze_inputs).values()))
            np.savez(
                output / ("gaze-" + row["name"] + ".npz"), **gaze_inputs, gaze=gaze
            )
            native_cascade.append(
                {
                    "name": row["name"],
                    "angles": angles.tolist(),
                    "landmarks": landmarks.tolist(),
                    "gaze": decode_gaze(gaze, angles[2]),
                    "transform": transform,
                }
            )
        write(output / "native-cascade.json", native_cascade)


def compare(output, build_root, attribute_root):
    from app.appearance import (
        AppearanceFeatures,
        prepare_appearance,
        normalized_features,
    )
    from app.head_gaze import HeadPose, GazeLandmarks, Gaze
    from deploy import runtime

    manifest = json.loads(MANIFEST.read_text())
    rows = json.loads((output / "fixtures/fixtures.json").read_text())
    report = {
        "status": "RUNNING",
        "scope": __doc__,
        "criteria": CRITERIA,
        "script_sha256": runtime.sha256(Path(__file__)),
        "models": {},
        "cascade": [],
        "fixtures": rows,
    }

    def save():
        write(output / "report.json", report)

    save()
    for row in rows:
        if (
            runtime.sha256(output / "fixtures" / (row["name"] + ".npz"))
            != row["sha256"]
        ):
            raise ValueError("Fixture changed")
    for role in ("person_reid", "object_embedding"):
        build = json.loads((build_root / role / "build.json").read_text())
        if build["onnx_sha256"] != manifest["models"][role]["sha256"]:
            raise ValueError("Wrong model build")
        model = AppearanceFeatures(build, role)
        cases = []
        try:
            for row in rows:
                if row["group"] != ("person" if role == "person_reid" else "object"):
                    continue
                with np.load(
                    output / "fixtures" / (row["name"] + ".npz"), allow_pickle=False
                ) as f:
                    image = f["image"]
                    box = f["roi"]
                with np.load(
                    output / (role + "-" + row["name"] + ".npz"), allow_pickle=False
                ) as f:
                    expected_input = f["input"]
                    expected = f["embedding"]
                inputs, transform = prepare_appearance(image, box, role)
                actual = model.model.infer(inputs)["embedding"]
                api = model.infer(image, box)
                cosine = float(api["embedding"] @ normalized_features(expected))
                passed = (
                    np.allclose(
                        inputs["images"],
                        expected_input,
                        atol=CRITERIA["input_atol"],
                        rtol=0,
                    )
                    and np.allclose(actual, expected, atol=0.001, rtol=0.001)
                    and cosine >= CRITERIA["embedding_min_cosine"]
                )
                cases.append(
                    {
                        "name": row["name"],
                        "passed": bool(passed),
                        "input_max_abs_error": float(
                            np.abs(inputs["images"] - expected_input).max()
                        ),
                        "raw_max_abs_error": float(np.abs(actual - expected).max()),
                        "native_embedding_cosine": cosine,
                        "transform": transform,
                    }
                )
                np.savez(
                    output / (role + "-" + row["name"] + "-trt.npz"),
                    raw=actual,
                    normalized=api["embedding"],
                )
                print(role, row["name"], bool(passed), flush=True)
        finally:
            model.close()
        report["models"][role] = {
            "engine_sha256": build["engine_sha256"],
            "onnx_sha256": build["onnx_sha256"],
            "cases": cases,
            "passed": all(c["passed"] for c in cases),
        }
        save()
    models = {}
    builds = {}
    classes = {"head_pose": HeadPose, "gaze_landmarks": GazeLandmarks, "gaze": Gaze}
    try:
        for role, klass in classes.items():
            base = build_root if role == "gaze_landmarks" else attribute_root
            build = json.loads((base / role / "build.json").read_text())
            if build["onnx_sha256"] != manifest["models"][role]["sha256"]:
                raise ValueError("Wrong attribute model build")
            builds[role] = build
            models[role] = klass(build)
            report["models"][role] = {
                "engine_sha256": build["engine_sha256"],
                "onnx_sha256": build["onnx_sha256"],
                "cases": [],
            }
        # Exercise a model replacement while the dependent models remain live.
        # The process logger must outlive the first-created execution context.
        import gc

        models["head_pose"].close()
        del models["head_pose"]
        gc.collect()
        models["head_pose"] = HeadPose(builds["head_pose"])
        report["lifecycle"] = {
            "head_reloaded_with_other_models_resident": True,
            "shared_logger": len({id(m.model.logger) for m in models.values()}) == 1,
        }
        native = {
            r["name"]: r
            for r in json.loads((output / "native-cascade.json").read_text())
        }
        for row in rows:
            if row["group"] != "face":
                continue
            with np.load(
                output / "fixtures" / (row["name"] + ".npz"), allow_pickle=False
            ) as f:
                image = f["image"]
                box = f["roi"]
            for role, model in models.items():
                with np.load(
                    output / (role + "-" + row["name"] + ".npz"), allow_pickle=False
                ) as f:
                    if role == "gaze":
                        inputs = {
                            name: f[name]
                            for name in (
                                "left_eye_image",
                                "right_eye_image",
                                "head_pose_angles",
                            )
                        }
                        expected = {model.output_name: f["gaze"]}
                    elif role == "gaze_landmarks":
                        inputs = {"data": f["input"]}
                        expected = {model.output_name: f["landmarks"]}
                    else:
                        inputs = {"data": f["input"]}
                        expected = {name: f[name] for name in ("fc_y", "fc_p", "fc_r")}
                actual = model.model.infer(inputs)
                passed = all(
                    np.allclose(actual[k], v, atol=0.001, rtol=0.001)
                    for k, v in expected.items()
                )
                report["models"][role]["cases"].append(
                    {
                        "name": row["name"],
                        "passed": bool(passed),
                        "max_abs_error": max(
                            float(np.abs(actual[k] - v).max())
                            for k, v in expected.items()
                        ),
                    }
                )
                np.savez(output / (role + "-" + row["name"] + "-trt.npz"), **actual)
            head = models["head_pose"].infer(image, box)
            points = models["gaze_landmarks"].infer(image, box)
            gaze = models["gaze"].infer(
                image, points["landmarks"], head["angles_degrees"]
            )
            wanted = native[row["name"]]
            a = np.asarray(gaze["gaze_vector"], np.float64)
            b = np.asarray(wanted["gaze"]["gaze_vector"], np.float64)
            angle = float(
                np.degrees(
                    np.arccos(
                        np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)
                    )
                )
            )
            head_error = float(
                np.abs(np.asarray(head["angles_degrees"]) - wanted["angles"]).max()
            )
            point_error = int(np.abs(points["landmarks"] - wanted["landmarks"]).max())
            passed = (
                head_error <= CRITERIA["head_max_error_deg"]
                and point_error <= CRITERIA["landmark_max_error_px"]
                and angle <= CRITERIA["cascade_max_angle_deg"]
            )
            report["cascade"].append(
                {
                    "name": row["name"],
                    "passed": bool(passed),
                    "head_max_error_deg": head_error,
                    "landmark_max_error_px": point_error,
                    "gaze_angular_error_deg": angle,
                    "head": head,
                    "gaze": gaze,
                }
            )
            print("head/gaze cascade", row["name"], bool(passed), angle, flush=True)
            save()
        for role in classes:
            report["models"][role]["passed"] = all(
                c["passed"] for c in report["models"][role]["cases"]
            )
    finally:
        for model in models.values():
            model.close()
        save()
    report["status"] = (
        "PARITY_PASS_LIMITED_FIXTURES"
        if all(m["passed"] for m in report["models"].values())
        and all(c["passed"] for c in report["cascade"])
        and report["lifecycle"]["shared_logger"]
        else "PARITY_FAILED"
    )
    save()
    return report["status"] == "PARITY_PASS_LIMITED_FIXTURES"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--attribute-root", type=Path, required=True)
    parser.add_argument("--worker", choices=("appearance", "attributes"))
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        native_worker(args.worker, output)
        return 0
    from build_vision_b1 import exclusive_build
    from deploy import runtime, services

    output.mkdir(parents=True, exist_ok=False)
    (output / "fixtures").mkdir()
    cfg = runtime.config()
    if services.profile_path(cfg).exists():
        cfg = runtime.config(services.profile_path(cfg))
    with exclusive_build(cfg, output):
        source_fixtures(output / "fixtures")
        for mode, environment in (
            ("appearance", "vision-export"),
            ("attributes", "openvino-tools"),
        ):
            with (output / (mode + "-reference.log")).open("w") as log:
                result = subprocess.run(
                    [
                        str(ROOT / "05_runtime/builds" / environment / "bin/python"),
                        str(Path(__file__).resolve()),
                        "--worker",
                        mode,
                        "--output",
                        str(output),
                        "--build-root",
                        str(args.build_root),
                        "--attribute-root",
                        str(args.attribute_root),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=240,
                )
            if result.returncode:
                raise RuntimeError("Reference failed: " + mode)
        passed = compare(
            output, args.build_root.resolve(), args.attribute_root.resolve()
        )
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
