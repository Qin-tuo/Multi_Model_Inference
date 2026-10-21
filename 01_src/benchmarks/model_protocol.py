# SPDX-License-Identifier: Apache-2.0
"""Reuse frozen speech/text fixtures with the selected current runtime revision."""

import argparse
import json
from pathlib import Path

from project_paths import ROOT
from deploy import runtime
from benchmarks.vision_protocol import PROTOCOL as VISION_PROTOCOL, validate as vision_validate

PROTOCOL = ROOT / "benchmarks/manifests/model-measurements.json"
CORPORA = {
    "llm": "benchmarks/corpora/b0-llm-v1.json",
    "speech": "benchmarks/corpora/b0-speech-v1.json",
}


def validate(path=PROTOCOL):
    protocol = json.loads(Path(path).read_text())
    locked = protocol["runtime_protocol"]
    source = ROOT / locked["path"]
    if runtime.sha256(source) != locked["sha256"]:
        raise ValueError("Model benchmark runtime protocol changed")
    _, cfg = vision_validate(source)
    record = protocol["input_manifest"]
    manifest_path = ROOT / record["path"]
    if runtime.sha256(manifest_path) != record["sha256"]:
        raise ValueError("Model input manifest changed")
    fixtures = json.loads(manifest_path.read_text())
    directory = ROOT / protocol["input_directory"]
    for item in fixtures["files"]:
        if runtime.sha256(directory / item["filename"]) != item["sha256"]:
            raise ValueError("Model input changed: " + item["filename"])
    for item in protocol["corpora"].values():
        if runtime.sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Model corpus changed: " + item["path"])
    return {**protocol, "input_directory": str(directory)}, cfg, fixtures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-protocol", type=Path, default=VISION_PROTOCOL)
    parser.add_argument("--inputs", type=Path, default=ROOT / "05_runtime/datasets/b0-sw-v1")
    parser.add_argument("--output", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    vision_validate(args.runtime_protocol)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("A frozen protocol cannot be overwritten")

    def record(path):
        path = path.absolute()
        return {"path": str(path.relative_to(ROOT)), "sha256": runtime.sha256(path)}

    value = {
        "schema_version": 1,
        "scope": "Same frozen speech/text/image fixtures; current runtime parameters. Historical results retain their original runtime identity.",
        "runtime_protocol": record(args.runtime_protocol),
        "input_manifest": record(args.inputs / "manifest.json"),
        "input_directory": str(args.inputs.absolute().relative_to(ROOT)),
        "corpora": {name: record(ROOT / path) for name, path in CORPORA.items()},
    }
    # Publish only after all referenced bytes pass validation.
    temporary = output.with_name(output.name + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    try:
        validate(temporary)
        import os

        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(output)
