"""Real persistent Pipecat conversation using isolated, synthetic user data."""
import asyncio
import json
import re
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.adapters.llama import LlamaConfig
from app.memory import Mem0Adapter
from app.records import Records
from app.reminders import ReminderService
from app.session import AssistantSession
from tools.project import CACHE, stamp, write_json


async def main():
    reports = []
    with tempfile.TemporaryDirectory(prefix="assistant-session-") as temporary:
        data = Path(temporary)
        for phase, prompts in [
            ("continuous", ["请记住，我喜欢茉莉花茶。", "我喜欢喝什么茶？",
                            "创建笔记，标题是采购，内容是周末买绿茶和杯子。"]),
            ("reopened", ["我喜欢喝什么茶？", "请查询我的笔记。", "请忘记我喜欢喝什么茶的记忆。", "我喜欢喝什么茶？"]),
        ]:
            with Mem0Adapter(data / "memory", bge_path=CACHE / "models/bge-small-zh-v1.5-onnx", embedding_backend="onnx") as memory:
                with ReminderService(data / "reminders.sqlite") as reminders, Records(data / "records.sqlite") as records:
                    session = await AssistantSession(LlamaConfig.from_manifest(ROOT / "models/manifest.json"), memory, reminders, records).start()
                    async def emit(text):
                        print(json.dumps({"phase": phase, "delivered": text}, ensure_ascii=False), flush=True)
                    try:
                        for prompt in prompts:
                            result = await session.ask(prompt, emit=emit)
                            result.update(phase=phase, prompt=prompt)
                            reports.append(result)
                    finally:
                        await session.close()
                    saved = memory.list_memories("owner")
                    notes = records.list_notes("owner")
                    history = records.recent("owner")
    checks = {
        "remember_executed": any(c["tool"] == "remember" and c["outcome"]["ok"] for c in reports[0]["calls"]),
        "same_session_recall": "茉莉" in reports[1]["response"],
        "note_saved": bool(notes) and "绿茶" in notes[0]["body"],
        "reopened_recall": "茉莉" in reports[3]["response"],
        "note_tool_queried": any(c["tool"] == "list_notes" and c["outcome"]["ok"] for c in reports[4]["calls"]),
        "forget_executed": any(c["tool"] == "forget" and c["outcome"]["ok"] for c in reports[5]["calls"]),
        "forget_persisted": not saved,
        "forgotten_fact_absent_from_final_answer": "茉莉" not in reports[6]["response"],
        "does_not_invent_preference_from_shopping_note": bool(re.search(r"不知道|不清楚|没有.*(?:记忆|记录|信息)|不确定|尚未", reports[6]["response"])),
        "forgotten_fact_absent_from_retained_history": "茉莉" not in json.dumps(history, ensure_ascii=False),
        "every_request_measured": all(r["prompt_tokens"] for r in reports),
        "tool_continuations_measured": all(len(r["prompt_tokens"]) >= 2 for r in reports if r["calls"]),
    }
    report = {"created_at": stamp(), "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "scope": "seven synthetic text requests in two native Pipecat workers with real local models; SDK stores closed/reopened between phases; not full quality, process restart or acoustic acceptance",
              "cases": reports, "remaining_memory_count": len(saved), "note_count": len(notes), "retained_history_count": len(history)}
    write_json(ROOT / "benchmarks/results/session-smoke.json", report)
    print(json.dumps(checks, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
