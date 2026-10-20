#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check complete face crops/features/expression against independent references.

Uses public image fixtures only. Numerical agreement and a small matching smoke
test do not establish identity error rates or real-world expression accuracy.
"""

import argparse
import gc
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_src"))
from app.face_trt import TensorRTFace
from app.face_tasks import (
    FaceIdentity,
    FaceExpression,
    aligned_bgr,
    sface_transform,
    face_tensor,
    expression_scores,
    normalize_embedding,
)
from app.body_detector import load_source
from build_vision_b1 import exclusive_build
from deploy import runtime, services
from validation_inputs import input_path
from prepare_vision_b1 import MANIFEST, prepare

CRITERIA = {
    "raw_atol": 1e-3,
    "raw_rtol": 1e-3,
    "crop_mean_absolute_error": 0.05,
    "native_embedding_min_cosine": 0.999,
    "native_expression_max_score_error": 1e-3,
    "example_cosine_threshold": 0.363,
}


def fixtures(face_root):
    face_root = input_path(face_root, ROOT)
    build = json.loads((face_root / "face/build.json").read_text())
    build["engine_path"] = str(input_path(ROOT / build["engine_path"], ROOT))
    passed = json.loads((face_root / "parity/face/report.json").read_text())
    if (
        passed["status"] != "PARITY_PASS_LIMITED_FIXTURES"
        or passed["engine_sha256"] != build["engine_sha256"]
    ):
        raise ValueError("Face detector reference gate is not verified")
    detector = TensorRTFace(build)
    result = []
    try:
        for name, relative, factor in (
            ("public-person", "05_runtime/datasets/b0-sw-v1/public-original.png", 1.0),
            (
                "public-person-dimmed",
                "05_runtime/datasets/b0-sw-v1/public-original.png",
                0.7,
            ),
            (
                "upstream-crowd-face",
                "05_runtime/datasets/b1-public/expression-selfie.jpg",
                1.0,
            ),
        ):
            path = ROOT / relative
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            image = (
                np.rint(image.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)
            )
            faces = detector.infer(image)
            if not faces:
                raise ValueError("No face in positive fixture " + name)
            face = max(
                faces,
                key=lambda f: (f["box_xyxy"][2] - f["box_xyxy"][0])
                * (f["box_xyxy"][3] - f["box_xyxy"][1]),
            )
            result.append(
                (
                    name,
                    image,
                    face,
                    {
                        "source": relative,
                        "source_sha256": runtime.sha256(path),
                        "brightness": factor,
                        "detector_engine_sha256": build["engine_sha256"],
                        "face": face,
                    },
                )
            )
    finally:
        detector.close()
    return result


def probe(build_root, output, face_root):
    import onnxruntime as ort

    entries = json.loads(MANIFEST.read_text())["models"]
    cases = fixtures(face_root)
    report = {
        "created_at": runtime.stamp(),
        "status": "RUNNING",
        "scope": __doc__,
        "criteria": CRITERIA,
        "script_sha256": runtime.sha256(Path(__file__)),
        "adapter_sha256": runtime.sha256(ROOT / "01_src/app/face_tasks.py"),
        "models": {},
    }

    def save():
        runtime.write_json(output / "report.json", report)

    save()
    try:
        for role in ("face_identity", "expression"):
            entry = entries[role]
            prepare(role, entry, check_only=True)
            build = json.loads((build_root / role / "build.json").read_text())
            if (
                build["status"] != "BUILT_NOT_ACCEPTED"
                or build["onnx_sha256"] != entry["sha256"]
            ):
                raise ValueError("No matching build for " + role)
            folder = output / role
            folder.mkdir()
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            session = ort.InferenceSession(
                str(ROOT / entry["path"]), options, providers=["CPUExecutionProvider"]
            )
            if [x.name for x in session.get_inputs()] != ["data"]:
                raise ValueError("Unexpected runtime data inputs")
            if role == "face_identity":
                native = cv2.FaceRecognizerSF.create(str(ROOT / entry["path"]), "")
                model = FaceIdentity(build)
            else:
                source = entry["prepost_source"]["files"]["facial_fer_model.py"]
                module = load_source(
                    str(ROOT / source["path"]), source["sha256"], "reference_fer"
                )
                native = module.FacialExpressionRecog(str(ROOT / entry["path"]))
                model = FaceExpression(build, entry)
            record = {
                "engine_sha256": build["engine_sha256"],
                "onnx_sha256": entry["sha256"],
                "cases": [],
            }
            report["models"][role] = record
            embeddings = []
            try:
                for name, image, face, provenance in cases:
                    points = np.asarray(face["landmarks"], np.float32)
                    x1, y1, x2, y2 = face["box_xyxy"]
                    row = np.r_[
                        np.array([x1, y1, x2 - x1, y2 - y1], np.float32), points.ravel()
                    ]
                    if role == "face_identity":
                        crop = aligned_bgr(image, sface_transform(points))
                        native_crop = native.alignCrop(image, row)
                        inputs = face_tensor(crop)
                        native_raw = native.feature(native_crop)
                        crop_error = float(
                            np.abs(crop.astype(np.float32) - native_crop).mean()
                        )
                    else:
                        matrix, _ = model.aligner.get_similarity_transform_for_cv2(
                            points, model.aligner._std_points
                        )
                        crop = aligned_bgr(image, matrix)
                        inputs = face_tensor(crop, expression=True)
                        native_input = native._preprocess(image, row)
                        crop_error = float(np.abs(inputs["data"] - native_input).mean())
                        native._model.setInput(native_input, "data")
                        native_raw = native._model.forward(["label"])[0]
                    expected = session.run(None, inputs)[0]
                    actual = model.model.infer(inputs)[
                        "fc1" if role == "face_identity" else "label"
                    ]
                    api = model.infer(image, points)
                    raw_pass = bool(
                        np.allclose(
                            expected,
                            actual,
                            atol=CRITERIA["raw_atol"],
                            rtol=CRITERIA["raw_rtol"],
                        )
                    )
                    case = {
                        "name": name,
                        "provenance": provenance,
                        "raw_max_abs_error": float(np.abs(expected - actual).max()),
                        "crop_or_input_mean_abs_error": crop_error,
                        "raw_pass": raw_pass,
                    }
                    if role == "face_identity":
                        embeddings.append(api["embedding"])
                        cosine = float(
                            api["embedding"] @ normalize_embedding(native_raw)
                        )
                        case.update(
                            native_embedding_cosine=cosine,
                            passed=raw_pass
                            and crop_error <= CRITERIA["crop_mean_absolute_error"]
                            and cosine >= CRITERIA["native_embedding_min_cosine"],
                        )
                    else:
                        expected_scores = expression_scores(native_raw)
                        error = max(
                            abs(api["scores"][key] - value)
                            for key, value in expected_scores.items()
                        )
                        label = max(expected_scores, key=expected_scores.get)
                        case.update(
                            native_max_score_error=error,
                            label=api["label"],
                            scores=api["scores"],
                            passed=raw_pass
                            and crop_error <= 1e-6
                            and error <= CRITERIA["native_expression_max_score_error"]
                            and label == api["label"],
                        )
                    np.savez(
                        folder / (name + ".npz"),
                        input=inputs["data"],
                        onnx=expected,
                        tensorrt=actual,
                        native=native_raw,
                        crop=crop,
                    )
                    record["cases"].append(case)
                    save()
                    print(role, name, "passed", case["passed"], flush=True)
                if role == "face_identity":
                    vectors = np.stack(embeddings)
                    similarities = vectors @ vectors.T
                    record["matching_smoke"] = {
                        "cosines": similarities.tolist(),
                        "same_face_brightness_pass": bool(similarities[0, 1] >= 0.363),
                        "different_fixture_rejected": bool(similarities[0, 2] < 0.363),
                        "scope": "Two public identities; no false-accept/false-reject rate claim",
                    }
                record["status"] = (
                    "PARITY_PASS_LIMITED_FIXTURES"
                    if all(c["passed"] for c in record["cases"])
                    else "PARITY_FAILED"
                )
            finally:
                model.close()
                del native, session
                gc.collect()
        report["status"] = (
            "PARITY_PASS_LIMITED_FIXTURES"
            if all(
                r["status"] == "PARITY_PASS_LIMITED_FIXTURES"
                for r in report["models"].values()
            )
            else "PARITY_FAILED"
        )
    except Exception as exc:
        report.update(status="ERROR", error=repr(exc))
        raise
    finally:
        save()
    return report["status"] == "PARITY_PASS_LIMITED_FIXTURES"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--face-build-root",
        type=Path,
        default=ROOT / "05_runtime/validation/b1-face-20261019",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = runtime.config()
    if services.profile_path(cfg).exists():
        cfg = runtime.config(services.profile_path(cfg))
    with exclusive_build(cfg, output):
        passed = probe(
            args.build_root.resolve(), output, args.face_build_root.resolve()
        )
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
