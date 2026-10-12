"""Run real model-selected Pipecat tools against isolated SDK stores."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.adapters.llama import LlamaConfig
from app.memory import Mem0Adapter
from app.reminders import ReminderService
from app.tools import ToolTurn
from app.pipecat_bot import input_messages,run_smoke
from tools.project import CACHE,stamp,write_json


async def main():
    results=[]
    with tempfile.TemporaryDirectory(prefix="assistant-tools-") as temporary:
        with Mem0Adapter(Path(temporary)/"memory",bge_path=CACHE/"models/bge-small-zh-v1.5-onnx",embedding_backend="onnx") as memory:
            with ReminderService(Path(temporary)/"reminders.sqlite") as reminders:
                for text,expected in [("现在几点？请查询真实时间。","get_time"),
                                      ("请记住，我喜欢简短的中文回答。","remember"),
                                      ("五分钟后提醒我休息。","create_reminder")]:
                    turn=ToolTurn("owner",text,memory,reminders)
                    messages=input_messages(text)
                    result=await run_smoke(LlamaConfig.from_manifest(ROOT/"models/manifest.json"),
                                           messages,tool_turn=turn,timeout_seconds=90)
                    success=any(c["tool"]==expected and c["outcome"]["ok"] for c in turn.calls)
                    result.update({"prompt":text,"expected_tool":expected,"actual_calls":turn.calls,"status":"PASS" if success else "FAIL"})
                    results.append(result)
                    print(json.dumps({"prompt":text,"response":result["response"],"calls":turn.calls},ensure_ascii=False),flush=True)
                memories=memory.list_memories("owner")
                tasks=reminders.list_reminders("owner")
    report={"created_at":stamp(),"status":"PASS" if all(r["status"]=="PASS" for r in results) and memories and tasks else "FAIL",
            "scope":"three synthetic real LLM tool calls with actual SDK state; not complete tool/cancellation acceptance",
            "cases":results,"saved_memories":memories,"scheduled_reminders":tasks}
    write_json(ROOT/"benchmarks/results/tool-smoke.json",report)
    return 0 if report["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(asyncio.run(main()))
