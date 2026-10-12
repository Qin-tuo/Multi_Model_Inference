"""Replay a frozen synthetic request through native llama.cpp tool parsing.

Only model planning is requested: returned tool calls are never executed here.
This is a narrow request-format comparison, not the full B0-B3/JT evaluation.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.adapters.llama import LlamaConfig, apply_required_tool_prefix
from app.tools import CreateReminderArgs
from tools.project import request_json, sha256, stamp, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/required-tool-prefill.json")
    args = parser.parse_args()
    if not 1 <= args.trials <= 10: parser.error("trials must be 1..10")
    request = json.loads(args.request.read_text())
    if request.get("tool_choice") != "required": raise ValueError("The frozen request must require a tool")
    config = LlamaConfig.from_manifest(ROOT / "models/manifest.json")
    results = []
    for trial in range(args.trials):
        for enabled in (False, True):
            body = apply_required_tool_prefix(request, config.tool_call_prefix if enabled else None)
            started = time.monotonic()
            response = request_json(config.base_url + "/chat/completions", body)
            choice = response["choices"][0]
            message = choice["message"]
            calls = message.get("tool_calls", [])
            arguments = None
            if len(calls) == 1:
                try:
                    arguments = CreateReminderArgs.model_validate(json.loads(calls[0]["function"]["arguments"])).model_dump()
                except (ValueError, KeyError): pass
            valid = bool(len(calls) == 1 and calls[0]["function"]["name"] == "create_reminder" and arguments
                and arguments.get("delay_seconds") == 10 and arguments.get("text", "").strip("。！! ") == "喝水")
            row = {"trial": trial, "prefill": enabled, "valid_call": valid,
                "seconds": time.monotonic() - started, "usage": response["usage"],
                "finish_reason": choice["finish_reason"], "message": message}
            results.append(row)
            print(json.dumps({k: row[k] for k in ("trial", "prefill", "valid_call", "seconds", "finish_reason")}), flush=True)
    checks = {"prefilled_calls_have_correct_arguments": all(r["valid_call"] for r in results if r["prefill"]),
              "prefilled_calls_have_no_prose_preamble": all(not r["message"].get("content") for r in results if r["prefill"])}
    report = {"created_at": stamp(), "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "scope": f"{args.trials} alternating paired trials of one frozen synthetic reminder request; native tool parsing/arguments, no tool execution and no broad quality or performance acceptance",
        "request_sha256": sha256(args.request), "trials": args.trials, "model": config.model, "cases": results}
    write_json(args.output, report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
