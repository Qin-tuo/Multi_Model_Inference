# SPDX-License-Identifier: Apache-2.0
"""Tool consolidation must preserve artifact identity and activation boundaries."""

from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from benchmarks import model_protocol
from deploy import runtime
from deploy.vision import build


@pytest.mark.parametrize("engine_matches", [True, False])
def test_reference_images_use_current_engine_and_preserve_old_gate(tmp_path, monkeypatch, engine_matches):
    import cv2
    import numpy as np
    from deploy.vision import reference_inputs as reference

    monkeypatch.setattr(reference, "ROOT", tmp_path)
    gate = tmp_path / "05_runtime/datasets/validation-inputs/old-face"
    (gate / "face").mkdir(parents=True)
    (gate / "parity/face").mkdir(parents=True)
    report = gate / "face/build.json"
    report.write_text(json.dumps({"engine_path": "/removed/build.engine", "engine_sha256": "accepted"}))
    original = report.read_bytes()
    (gate / "parity/face/report.json").write_text(json.dumps({
        "status": "PARITY_PASS_LIMITED_FIXTURES", "engine_sha256": "accepted",
    }))
    selected = tmp_path / "03_models/vision-runtime.json"
    selected.parent.mkdir()
    selected.write_text(json.dumps({"models": {"face": {
        "engine_path": "05_runtime/models/current.engine",
        "engine_sha256": "accepted" if engine_matches else "changed",
    }}}))
    for relative in ("b0-sw-v1/public-original.png", "b1-public/expression-selfie.jpg"):
        image = tmp_path / "05_runtime/datasets" / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image), np.full((16, 16, 3), 128, np.uint8))
    calls = []

    class Detector:
        def __init__(self, spec):
            calls.append(spec["engine_path"])

        def infer(self, image):
            return [{"box_xyxy": [1, 2, 10, 12], "score": 0.9}]

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(reference, "TensorRTFace", Detector)
    old_location = tmp_path / "05_runtime/validation/old-face"
    if not engine_matches:
        with pytest.raises(ValueError, match="reference gate"):
            reference.fixtures(old_location)
        assert calls == []
    else:
        rows = reference.fixtures(old_location)
        assert len(rows) == 3
        assert calls == [str(tmp_path / "05_runtime/models/current.engine"), "closed"]
        for _, _, _, provenance in rows:
            assert provenance["source_sha256"] == runtime.sha256(tmp_path / provenance["source"])
    assert report.read_bytes() == original


def test_engine_build_outputs_candidate_without_changing_selected_runtime(tmp_path, monkeypatch):
    source = tmp_path / "models.json"
    source.write_text(json.dumps({"models": {"face": {"sha256": "onnx-hash"}}}))
    current = tmp_path / "03_models/vision-runtime.json"
    current.parent.mkdir()
    current.write_text(json.dumps({"source_manifest": "models.json", "models": {
        "face": {"engine_sha256": "original"}, "expression": {"engine_sha256": "retained"},
    }}))
    previous = current.read_bytes()
    output = tmp_path / "candidate"
    cfg = SimpleNamespace(perception=SimpleNamespace(model_manifest=str(current)))
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "MANIFEST", source)
    monkeypatch.setattr(build.runtime, "config", lambda *args: cfg)
    monkeypatch.setattr(build.services, "profile_path", lambda _: tmp_path / "not-installed")
    monkeypatch.setattr(build, "exclusive_build", lambda *args: nullcontext())

    def compile_engine(role, entry, folder, *args):
        engine = folder / "test.engine"
        engine.write_bytes(b"compiled")
        return {"status": "BUILT_NOT_ACCEPTED", "engine_path": str(engine),
                "engine_sha256": runtime.sha256(engine), "engine_bytes": 8,
                "onnx_sha256": entry["sha256"], "input_shapes": {"input": [1, 3, 640, 640]},
                "precision": "FP32; TF32 disabled"}

    monkeypatch.setattr(build, "build", compile_engine)
    monkeypatch.setattr(sys, "argv", ["build", "--output", str(output), "--role", "face"])
    with pytest.raises(SystemExit) as done:
        build.main()
    assert done.value.code == 0
    assert current.read_bytes() == previous
    candidate = json.loads((output / "runtime-candidate.json").read_text())
    assert candidate["models"]["expression"]["engine_sha256"] == "retained"
    assert candidate["models"]["face"]["engine_sha256"] != "original"
    assert candidate["status"] == "BUILT_REFERENCE_AND_INTEGRATION_CHECKS_PENDING"


@pytest.mark.parametrize("modified", ["input", "corpus", "runtime"])
def test_model_measurement_rejects_changed_evidence_before_running_models(tmp_path, monkeypatch, modified):
    monkeypatch.setattr(model_protocol, "ROOT", tmp_path)
    selected = object()
    monkeypatch.setattr(model_protocol, "vision_validate", lambda _: ({}, selected))
    files = {}
    for name in ("input", "corpus", "runtime"):
        files[name] = tmp_path / name
        files[name].write_bytes(("frozen " + name).encode())
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps({"files": [
        {"filename": "input", "sha256": runtime.sha256(files["input"])}
    ]}))

    def record(path):
        return {"path": path.name, "sha256": runtime.sha256(path)}

    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "runtime_protocol": record(files["runtime"]),
        "input_manifest": record(manifest), "input_directory": ".",
        "corpora": {"speech": record(files["corpus"])},
    }))
    assert model_protocol.validate(protocol)[1] is selected
    files[modified].write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        model_protocol.validate(protocol)
