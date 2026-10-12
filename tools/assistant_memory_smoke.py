"""Verify save/correct/forget across separate real Pipecat processes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MEM0_TELEMETRY"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from app.memory import Mem0Adapter
from tools.project import CACHE, stamp, write_json


def main():
    results = []
    model = CACHE / "models/bge-small-zh-v1.5-onnx"
    with tempfile.TemporaryDirectory(prefix="assistant-real-memory-") as temporary:
        data = Path(temporary) / "memory"
        def ask(label):
            output = Path(temporary) / (label + ".json")
            run = subprocess.run([sys.executable, "-m", "app.pipecat_bot", "--text-smoke",
                                  "--prompt", "我最喜欢喝什么饮料？只根据你记住的内容回答；不知道就说不知道。",
                                  "--memory-dir", str(data), "--embedding-path", str(model),
                                  "--output", str(output)], cwd=ROOT, text=True, capture_output=True, timeout=120)
            if run.returncode:
                raise RuntimeError(run.stderr[-5000:])
            result = json.loads(output.read_text())
            result["case"] = label
            results.append(result)
            return result
        with Mem0Adapter(data, bge_path=model, embedding_backend="onnx") as memory:
            saved = memory.remember("owner", "preference", "drink", "我最喜欢喝无糖绿茶。",
                                    source_id="synthetic-save", request_id="save")
        first = ask("remember_after_reopen")
        with Mem0Adapter(data, bge_path=model, embedding_backend="onnx") as memory:
            memory.correct("owner", saved["memory_id"], "我现在最喜欢喝无糖乌龙茶。", expected_version=1,
                           source_id="synthetic-correct", request_id="correct")
        corrected = ask("correct_after_reopen")
        with Mem0Adapter(data, bge_path=model, embedding_backend="onnx") as memory:
            memory.forget("owner", saved["memory_id"], expected_version=2, request_id="forget")
        forgotten = ask("forget_after_reopen")
    passed = ("绿茶" in first["response"] and "乌龙茶" in corrected["response"] and
              "绿茶" not in corrected["response"] and
              forgotten["memory_evidence"]["injected_count"] == 0 and
              "绿茶" not in forgotten["response"] and "乌龙茶" not in forgotten["response"] and
              any(word in forgotten["response"] for word in ["不知道", "没有", "不清楚"]))
    report = {"created_at": stamp(), "status": "PASS" if passed else "FAIL", "cases": results,
              "scope": "three synthetic text-only cases, real local Mem0/BGE/llama/Pipecat; not full JT acceptance",
              "separate_application_process_per_question": True}
    write_json(ROOT / "benchmarks/results/assistant-memory-smoke.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
