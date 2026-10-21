# SPDX-License-Identifier: Apache-2.0
"""Compare the retained expression adapter with pinned OpenCV Zoo and ONNX.

Public fixtures establish numerical agreement only, not field expression accuracy.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01_src"))
from app.face_tasks import FaceExpression, aligned_bgr, expression_scores, face_tensor
from app.model_artifacts import load_source
from deploy import runtime, services
from deploy.operations import exclusive_build
from deploy.validation_inputs import input_path
from deploy.vision.artifacts import prepare
from deploy.vision.reference_inputs import fixtures


def probe(manifest_path, output, face_root):
    import onnxruntime as ort

    selected = json.loads(manifest_path.read_text())
    entry = json.loads((ROOT / selected["source_manifest"]).read_text())["models"]["expression"]
    prepare("expression", entry, check_only=True)
    spec = dict(selected["models"]["expression"])
    if spec["onnx_sha256"] != entry["sha256"]:
        raise ValueError("Expression engine/source mismatch")
    spec["engine_path"] = str(ROOT / spec["engine_path"])
    cases = fixtures(input_path(face_root, ROOT))
    source = entry["prepost_source"]["files"]["facial_fer_model.py"]
    module = load_source(str(ROOT / source["path"]), source["sha256"], "reference_fer")
    native = module.FacialExpressionRecog(str(ROOT / entry["path"]))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(ROOT / entry["path"]), options, providers=["CPUExecutionProvider"]
    )
    report = {
        "status": "RUNNING",
        "scope": __doc__,
        "onnx_sha256": entry["sha256"],
        "engine_sha256": spec["engine_sha256"],
        "criteria": {"raw_atol": 1e-3, "raw_rtol": 1e-3, "input_mae": 1e-6, "score_max_error": 1e-3},
        "cases": [],
    }
    model = None
    try:
        if [item.name for item in session.get_inputs()] != ["data"]:
            raise ValueError("Unexpected expression input contract")
        model = FaceExpression(spec, entry)
        for name, image, face, provenance in cases:
            points = np.asarray(face["landmarks"], np.float32)
            x1, y1, x2, y2 = face["box_xyxy"]
            row = np.r_[np.array([x1, y1, x2 - x1, y2 - y1], np.float32), points.ravel()]
            matrix, _ = model.aligner.get_similarity_transform_for_cv2(points, model.aligner._std_points)
            crop = aligned_bgr(image, matrix)
            inputs = face_tensor(crop, expression=True)
            native_input = native._preprocess(image, row)
            input_error = float(np.abs(inputs["data"] - native_input).mean())
            native._model.setInput(native_input, "data")
            native_raw = native._model.forward(["label"])[0]
            expected = session.run(None, inputs)[0]
            actual = model.model.infer(inputs)["label"]
            api = model.infer(image, points)
            expected_scores = expression_scores(native_raw)
            error = max(abs(api["scores"][key] - score) for key, score in expected_scores.items())
            raw_pass = bool(np.allclose(expected, actual, atol=1e-3, rtol=1e-3))
            passed = (
                raw_pass and input_error <= 1e-6 and error <= 1e-3
                and max(expected_scores, key=expected_scores.get) == api["label"]
            )
            report["cases"].append({
                "name": name, "provenance": provenance, "passed": passed,
                "raw_max_abs_error": float(np.abs(expected - actual).max()),
                "input_mean_abs_error": input_error, "native_max_score_error": error,
                "label": api["label"], "scores": api["scores"],
            })
            np.savez(output / (name + ".npz"), input=inputs["data"], onnx=expected,
                     tensorrt=actual, native=native_raw, crop=crop)
        report["status"] = "PARITY_PASS_LIMITED_FIXTURES" if all(row["passed"] for row in report["cases"]) else "PARITY_FAILED"
    except BaseException as exc:
        report.update(status="ERROR", error=repr(exc))
        raise
    finally:
        if model is not None:
            model.close()
        runtime.write_json(output / "report.json", report)
    return report["status"] == "PARITY_PASS_LIMITED_FIXTURES"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "03_models/vision-runtime.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--face-build-root", type=Path, default=ROOT / "05_runtime/validation/b1-face-20261019")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = runtime.config()
    if services.profile_path(cfg).exists():
        cfg = runtime.config(services.profile_path(cfg))
    with exclusive_build(cfg, output):
        passed = probe(args.manifest, output, args.face_build_root)
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
