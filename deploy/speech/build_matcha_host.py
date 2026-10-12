#!/usr/bin/env python3
"""Rebuild the pinned OVS Matcha ONNX assets for the actual host TensorRT.

Old JP6.2 engines remain in a separate input cache and are never copied to the
host runtime directory. The upstream split/export implementation is reused.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.project import CACHE, download_file, sha256, write_json

ENGINE_SOURCE = CACHE / "upstreams/openvoicestream/third_party/jetson-voice-engine"
INPUT = CACHE / "models/matcha-upstream-jp62"
OUTPUT = CACHE / "models/matcha-host"
PAYLOAD = "matcha-icefall-zh-en-jp62-trt103-sm87-edgellm-v091-20260804-r1.tar"
PAYLOAD_SHA = "3d4856abe3f71b07988cef816d49c81933ee96191819ef37af6c1c01193faf71"
REVISION = "737422aea690cd7604eae893a2168d0840a094dc"


def prepare_inputs():
    pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ENGINE_SOURCE, text=True).strip()
    if pin != "e1d8d507300e329642937b8461ce03bbd2bd3a16":
        raise RuntimeError("Jetson voice engine source differs from pinned revision")
    archive = CACHE / "downloads" / PAYLOAD
    download_file(f"https://huggingface.co/harvestsu/matcha-icefall-zh-en-jetson-artifacts/resolve/{REVISION}/{PAYLOAD}",
                  archive, PAYLOAD_SHA, 376135680)
    if not (INPUT / "model-steps-3.onnx").exists():
        INPUT.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            tar.extractall(INPUT, filter="data")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for source in INPUT.iterdir():
        if source.name in {"engines", "onnx"}: continue
        target = OUTPUT / source.name
        if source.is_dir(): shutil.copytree(source, target, dirs_exist_ok=True)
        else: shutil.copy2(source, target)
    (OUTPUT / "onnx").mkdir(exist_ok=True)
    (OUTPUT / "engines").mkdir(exist_ok=True)
    exporter = ENGINE_SOURCE / "models/matcha/split_matcha_trt.py"
    split_manifest = OUTPUT / "split-regression.json"
    if not split_manifest.exists():
        subprocess.run([str(ROOT / ".venv/bin/python"), str(exporter), "--input", str(OUTPUT / "model-steps-3.onnx"),
                        "--output-dir", str(OUTPUT / "onnx"), "--verify"], check=True)
        write_json(split_manifest, {"status": "PASS", "source_sha256": sha256(OUTPUT / "model-steps-3.onnx"),
                                    "exporter_sha256": sha256(exporter),
                                    "scope": "upstream deterministic CPU split-graph regression at noise_scale=0"})
    return pin


def build_engine(name, onnx, precision, shapes, target_version):
    engine = OUTPUT / "engines" / name
    stamp = engine.with_suffix(engine.suffix + ".host.json")
    identity = {"onnx_sha256": sha256(onnx), "tensorrt": target_version, "precision": precision, "shapes": shapes}
    if engine.exists() and stamp.exists():
        saved = json.loads(stamp.read_text())
        if saved.get("identity") == identity and saved.get("engine_sha256") == sha256(engine):
            print("Verified host engine:", name, flush=True)
            return saved
        raise RuntimeError(f"Existing host engine identity differs: {name}; use a new output/version")
    temp = engine.with_suffix(engine.suffix + ".building")
    command = ["trtexec", "--onnx="+str(onnx), "--saveEngine="+str(temp), "--"+precision,
               "--memPoolSize=workspace:1024", "--skipInference"]
    command.extend("--"+key+"="+value for key,value in shapes.items())
    log = engine.with_suffix(".build.log")
    print("Building:", name, flush=True)
    with log.open("w") as out:
        subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=True)
    if not temp.exists() or temp.stat().st_size == 0: raise RuntimeError("Builder produced no engine")
    temp.replace(engine)
    result = {"identity": identity, "engine_sha256": sha256(engine), "bytes": engine.stat().st_size,
              "command": command, "built_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "status": "built_not_speech_validated"}
    write_json(stamp, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    pin = prepare_inputs()
    if args.prepare_only: return 0
    import tensorrt as trt
    shapes = {"minShapes": "z:1x80x72,mu:1x80x72,mask:1x1x72",
              "optShapes": "z:1x80x256,mu:1x80x256,mask:1x1x256",
              "maxShapes": "z:1x80x600,mu:1x80x600,mask:1x1x600"}
    built = {}
    for step in range(3):
        name = f"matcha_estimator_step{step}_bf16.engine"
        built[name] = build_engine(name, OUTPUT / f"onnx/matcha_estimator_step{step}_trt.onnx", "bf16", shapes, trt.__version__)
    built["vocos_fp16.engine"] = build_engine("vocos_fp16.engine", OUTPUT / "vocos-16khz-univ.onnx", "fp16",
                                            {"minShapes": "mels:1x80x72", "optShapes": "mels:1x80x256", "maxShapes": "mels:1x80x600"}, trt.__version__)
    report = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "built_not_speech_validated",
              "input_revision": REVISION, "input_payload_sha256": PAYLOAD_SHA, "export_source_commit": pin,
              "host_l4t": Path('/etc/nv_tegra_release').read_text().splitlines()[0],
              "tensorrt": trt.__version__, "engines": built}
    write_json(OUTPUT / "host-manifest.json", report)
    write_json(ROOT / "benchmarks/results/matcha-build.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
