"""Exercise the public console in separate processes and native cancellation."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.adapters.llama import LlamaConfig
from app.memory import Mem0Adapter
from app.records import Records
from app.reminders import ReminderService
from app.session import AssistantSession
from tools.project import CACHE, stamp, write_json


async def cancel_probe(data):
    with Mem0Adapter(data / "memory", bge_path=CACHE / "models/bge-small-zh-v1.5-onnx", embedding_backend="onnx") as memory:
        with ReminderService(data / "reminders.sqlite") as reminders, Records(data / "records.sqlite") as records:
            config = replace(LlamaConfig.from_manifest(ROOT / "models/manifest.json"), max_tokens=256)
            session = await AssistantSession(config, memory, reminders, records, user_id="cancellation-test").start()
            emitted = []
            async def emit(text): emitted.append(text)
            task = asyncio.create_task(session.ask("请从一开始逐个列出一百个整数，每个整数都解释含义。", emit=emit))
            observed_running = False
            cancelled = False
            try:
                async with httpx.AsyncClient(timeout=2) as client:
                    deadline = time.monotonic() + 15
                    while time.monotonic() < deadline and not task.done():
                        response = await client.get("http://127.0.0.1:8088/slots")
                        response.raise_for_status()
                        if any(slot.get("is_processing") for slot in response.json()):
                            observed_running = True
                            break
                        await asyncio.sleep(0.05)
                if not observed_running: raise RuntimeError("Never observed the actual model request running")
                started = time.monotonic()
                await session.cancel()
                try: await task
                except asyncio.CancelledError: cancelled = True
                cancel_elapsed = time.monotonic() - started
                old_outputs = list(emitted)
                old_history = records.recent("cancellation-test")
                resumed = await session.ask("一加一等于几？只回答结果。", emit=emit)
                return {"observed_backend_running": observed_running, "request_cancelled": cancelled,
                        "provider_slot_idle_confirmed": session.backend_available,
                        "cancel_to_idle_seconds": cancel_elapsed, "cancelled_output_count": len(old_outputs),
                        "cancelled_completed_history_count": len(old_history), "next_response": resumed["response"]}
            finally:
                if not task.done():
                    await session.cancel()
                    try: await task
                    except asyncio.CancelledError: pass
                await session.close()


def main():
    cases = []
    with tempfile.TemporaryDirectory(prefix="assistant-public-console-") as temporary:
        data = Path(temporary)
        for name, user, prompt in [
            ("save_process", "owner", "请记住，我喜欢桂花乌龙茶。"),
            ("recall_new_process", "owner", "我喜欢喝什么茶？"),
            ("guest_new_process", "guest", "我喜欢喝什么茶？"),
        ]:
            source = data / (name + ".txt")
            source.write_text(prompt + "\n")
            result = subprocess.run([sys.executable, str(ROOT / "tools/project.py"), "interact",
                "--user", user, "--data-dir", str(data), "--input-file", str(source)],
                cwd=ROOT, text=True, capture_output=True, timeout=90)
            answers = [line.removeprefix("助手：") for line in result.stdout.splitlines() if line.startswith("助手：")]
            cases.append({"name": name, "exit_code": result.returncode, "answers": answers,
                          "error_tail": result.stderr[-1000:] if result.returncode else ""})
        cancellation = asyncio.run(cancel_probe(data))
    checks = {"public_commands_succeeded": all(case["exit_code"] == 0 and case["answers"] for case in cases),
              "new_process_recalled": "桂花" in "".join(cases[1]["answers"]),
              "guest_did_not_receive_owner_fact": "桂花" not in "".join(cases[2]["answers"]),
              "native_cancellation_observed": cancellation["request_cancelled"] and cancellation["provider_slot_idle_confirmed"],
              "cancelled_text_not_delivered_or_recorded": cancellation["cancelled_output_count"] == cancellation["cancelled_completed_history_count"] == 0,
              "next_turn_works": any(word in cancellation["next_response"] for word in ("2", "二", "两"))}
    report = {"created_at": stamp(), "status": "PASS" if all(checks.values()) else "FAIL",
              "scope": "three separate console processes, owner/guest profiles, native logical text interruption and observed llama slot idle; not physical playback/AEC or GPU memory-release acceptance",
              "checks": checks, "cases": cases, "cancellation": cancellation}
    write_json(ROOT / "benchmarks/results/application-smoke.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
