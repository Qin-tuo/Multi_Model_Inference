"""Prepare/verify a local BGE ONNX artifact using upstream SentenceTransformers."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.project import CACHE, download_file, sha256, stamp, write_json


def paths():
    return CACHE / "models/bge-small-zh-v1.5", CACHE / "models/bge-small-zh-v1.5-onnx"


def prepare():
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    entry = manifest["models"]["embedding"]
    source, output = paths()
    for item in entry["files"]:
        url = f"https://huggingface.co/{entry['repository']}/resolve/{entry['revision']}/{item['name']}"
        download_file(url, source / item["name"], item["sha256"], item["size"])
    # The checkpoint and tokenizer are already cached. Conversion must not fetch models.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import onnxruntime as ort
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    if not (output / "onnx/model.onnx").exists():
        model = SentenceTransformer(str(source), backend="onnx", device="cpu", local_files_only=True,
                                    model_kwargs={"export": True, "provider": "CPUExecutionProvider", "session_options": opts})
        model.save_pretrained(str(output))
        del model
        gc.collect()
    reference = SentenceTransformer(str(source), device="cpu", local_files_only=True)
    texts = ["我喜欢简短的中文回答。", "下次见面提醒我带笔记本。", "这个项目运行在Jetson上。", "桌子上有一个蓝色杯子。"]
    expected = reference.encode(texts, normalize_embeddings=True)
    del reference
    gc.collect()
    model = SentenceTransformer(str(output), backend="onnx", device="cpu", local_files_only=True,
                                model_kwargs={"export": False, "provider": "CPUExecutionProvider",
                                              "file_name": "onnx/model.onnx", "session_options": opts})
    actual = model.encode(texts, normalize_embeddings=True)
    import numpy as np
    max_error = float(np.max(np.abs(expected-actual)))
    min_cosine = float(np.min(np.sum(expected*actual, axis=1)))
    artifact = output / "onnx/model.onnx"
    report = {"created_at": stamp(), "status": "PASS" if max_error < 0.002 and min_cosine > 0.999 else "FAIL",
              "scope": "four-sentence CPU reference/ONNX conversion regression; not final retrieval quality",
              "dimensions": int(actual.shape[1]), "max_absolute_error": max_error, "min_cosine": min_cosine,
              "reference_revision": entry["revision"], "onnx_sha256": sha256(artifact),
              "execution_provider": "CPUExecutionProvider", "threads": 2}
    write_json(ROOT / "benchmarks/results/bge-conversion.json", report)
    entry["onnx_artifact"] = {"name": "onnx/model.onnx", "sha256": report["onnx_sha256"], "size": artifact.stat().st_size}
    entry["status"] = "conversion_verified" if report["status"] == "PASS" else "conversion_failed"
    write_json(ROOT / "models/manifest.json", manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS": raise RuntimeError("BGE conversion regression failed")


def smoke():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MEM0_TELEMETRY"] = "false"
    from app.memory import Mem0Adapter
    _, model = paths()
    if not (model / "onnx/model.onnx").exists(): raise RuntimeError("Prepare BGE ONNX first")
    # Synthetic, independent facts; never use the owner's real database in a smoke test.
    examples = [
        ("style", "我喜欢简短的中文回答。", "我希望你用什么风格回复？"),
        ("drink", "我平时最爱喝无糖绿茶。", "我最喜欢喝什么饮料？"),
        ("city", "我现在居住在杭州。", "我住在哪个城市？"),
        ("pet", "我养的猫叫橘子。", "我家猫咪的名字是什么？"),
        ("sport", "我每周六去游泳。", "周六我通常做什么运动？"),
        ("book", "我最近在读三体。", "我最近看的是哪本书？"),
        ("food", "我不吃香菜。", "做饭时哪种配菜不要给我放？"),
        ("project", "我正在开发一个离线语音助手。", "我现在在开发什么项目？"),
        ("meeting", "我周一上午九点有团队例会。", "团队例会是在什么时候？"),
        ("place", "我的备用钥匙放在书桌左边抽屉。", "我的备用钥匙收在哪里？"),
    ]
    cases, latencies = [], []
    with tempfile.TemporaryDirectory(prefix="assistant-bge-smoke-") as temp:
        with Mem0Adapter(temp, bge_path=model, embedding_backend="onnx") as memory:
            for i, (key, content, _) in enumerate(examples):
                memory.remember("smoke-owner", "fact", key, content, source_id=f"synthetic-{i}", request_id=f"save-{i}")
        # Reopen both SDK stores and the CPU model to verify genuine persistence.
        with Mem0Adapter(temp, bge_path=model, embedding_backend="onnx") as memory:
            for key, _, query in examples:
                started = time.perf_counter()
                result = memory.search("smoke-owner", query, top_k=3)
                elapsed_ms = (time.perf_counter()-started)*1000
                latencies.append(elapsed_ms)
                results = result["items"]
                found = [x.get("key") for x in results]
                cases.append({"key": key, "query": query, "returned_keys": found, "top1_correct": bool(found and found[0] == key), "latency_ms": elapsed_ms})
            guest = memory.search("smoke-guest", "我喜欢什么？", top_k=3)
            isolated = not guest["items"]
    report = {"created_at": stamp(), "status": "PASS" if all(c["top1_correct"] for c in cases) and isolated else "FAIL",
              "scope": "10 synthetic semantic recall cases after restart; not JT-10/11 acceptance",
              "backend": "real_mem0_qdrant_bge_onnx_cpu", "cases": cases,
              "owner_isolation": isolated, "median_query_ms": statistics.median(latencies),
              "max_query_ms": max(latencies), "record_count": len(examples)}
    write_json(ROOT / "benchmarks/results/memory-semantic-smoke.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS": raise RuntimeError("Semantic smoke has failing cases; inspect report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "smoke"])
    args = parser.parse_args()
    {"prepare": prepare, "smoke": smoke}[args.action]()


if __name__ == "__main__":
    main()
