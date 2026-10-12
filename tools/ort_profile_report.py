"""Verify real ORT provider events and full-model TensorRT tensor boundaries."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.project import sha256, stamp, write_json


def shapes(value):
    if not isinstance(value, list): return []
    return [tuple(shape) for item in value if isinstance(item, dict) for shape in item.values() if isinstance(shape, list)]


def inspect_profile(path, task):
    records = json.loads(path.read_text())
    expected_output = (1, 84 if task == "detection" else 56, 8400)
    providers, operations = Counter(), Counter()
    complete_graph_runs = 0
    cpu_weighted = Counter()
    for record in records:
        args = record.get("args", {})
        provider = args.get("provider")
        if record.get("cat") != "Node" or not provider: continue
        operation = args.get("op_name", "unknown")
        providers[provider] += 1
        operations[provider + ":" + operation] += 1
        if provider == "CPUExecutionProvider" and operation in {"Conv", "MatMul", "Gemm", "Attention", "GRU", "LSTM"}:
            cpu_weighted[operation] += 1
        if "tensorrt" in provider.lower() and (1, 3, 640, 640) in shapes(args.get("input_type_shape")):
            if expected_output in shapes(args.get("output_type_shape")): complete_graph_runs += 1
    passed = complete_graph_runs >= 5 and not cpu_weighted
    return {"status": "PASS" if passed else "FAIL", "task": task, "trace_sha256": sha256(path),
            "providers": dict(providers), "operations": dict(operations),
            "full_model_tensorrt_executions": complete_graph_runs, "weighted_cpu_operations": dict(cpu_weighted),
            "scope": "actual ORT per-node trace with model-shaped TensorRT boundaries; no quality/performance acceptance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--task", choices=["detection", "pose"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"created_at": stamp(), **inspect_profile(args.trace, args.task)}
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
