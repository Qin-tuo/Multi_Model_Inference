"""Evaluate frozen typed tool requests with real llama.cpp/BGE and SDK stores.

This development corpus grades declared slots and actual durable mutations.
It is not a held-out human-language set, ASR test, or audible reminder acceptance.
"""
import argparse
import asyncio
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from zoneinfo import ZoneInfo

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.adapters.llama import LlamaConfig
from app.memory import Mem0Adapter
from app.records import Records
from app.reminders import ReminderService
from app.session import AssistantSession
from app.tools import RememberArgs, CorrectArgs, ForgetArgs, CreateNoteArgs, CreateReminderArgs, CancelReminderArgs
from tools.project import CACHE, sha256, stamp, write_json

WRITE_MODELS = {"remember": RememberArgs, "correct": CorrectArgs, "forget": ForgetArgs,
    "create_note": CreateNoteArgs, "create_reminder": CreateReminderArgs, "cancel_reminder": CancelReminderArgs}
LOCAL = ZoneInfo("Asia/Shanghai")


def normalize(value):
    return re.sub(r"[\s，。！？：；、,.!?:;\"'“”‘’]", "", value).lower()


def has_terms(value, terms):
    return all(any(normalize(term) in normalize(value) for term in (item if isinstance(item, list) else [item])) for item in terms)


def state(memory, reminders, records, user):
    return {
        "memories": sorted([{k:r[k] for k in ("memory_id", "content", "key", "version")} for r in memory.list_memories(user)], key=lambda r:r["memory_id"]),
        "reminders": sorted([{k:r[k] for k in ("id", "text", "due_utc", "status")} for r in reminders.list_reminders(user, include_inactive=True)], key=lambda r:r["id"]),
        "notes": sorted([{k:r[k] for k in ("id", "title", "body")} for r in records.list_notes(user, limit=100)], key=lambda r:r["id"]),
    }


def seed(case, memory, reminders, records, user):
    ids = {"memories": [], "reminders": [], "notes": []}
    for index, content in enumerate(case.get("setup", {}).get("memories", [])):
        result = memory.remember(user, "fact", f"seed_fact_{index}", content,
            source_id=f"{case['id']}-seed-{index}", source_excerpt=content, request_id=f"memory-seed-{index}")
        ids["memories"].append(result["memory_id"])
    for index, text in enumerate(case.get("setup", {}).get("reminders", [])):
        result = reminders.create(user, text, datetime.now(timezone.utc) + timedelta(days=1, hours=index), request_id=f"reminder-seed-{index}")
        ids["reminders"].append(result["reminder_id"])
    for index, note in enumerate(case.get("setup", {}).get("notes", [])):
        result = records.create_note(user, **note, source_id=case["id"] + "-seed", request_id=f"note-seed-{index}")
        ids["notes"].append(result["note_id"])
    return ids


def judge(case, before, after, ids, result, anchor, future_date, error):
    expected = case["expected"]
    kind = expected["kind"]
    calls = result.get("calls", [])
    successful = [c for c in calls if c["outcome"].get("ok")]
    successful_writes = [c for c in successful if c["tool"] in WRITE_MODELS]
    checks = {"request_completed": error is None and bool(result.get("response", "").strip())}
    wanted = {"reminder":"create_reminder", "remember":"remember", "correct":"correct",
              "forget":"forget", "note":"create_note", "cancel_reminder":"cancel_reminder"}.get(kind)
    if wanted:
        checks["expected_operation_succeeded"] = any(c["tool"] == wanted for c in successful)
        checks["no_other_write_operations"] = all(c["tool"] == wanted for c in successful_writes)
    if kind == "reminder":
        rows = after["reminders"]
        checks["one_reminder_only"] = len(rows) == len(before["reminders"]) + 1
        checks["reminder_content"] = len(rows) == 1 and normalize(rows[0]["text"]) == normalize(expected["content"])
        wanted_due = None
        if anchor:
            if "seconds" in expected: wanted_due = anchor + timedelta(seconds=expected["seconds"])
            else:
                day = {"future": future_date, "today": anchor.astimezone(LOCAL).date(),
                       "tomorrow": anchor.astimezone(LOCAL).date()+timedelta(days=1),
                       "day_after": anchor.astimezone(LOCAL).date()+timedelta(days=2)}[expected["day"]]
                wanted_due = datetime.fromisoformat(day.isoformat() + "T" + expected["clock"]).replace(tzinfo=LOCAL)
        checks["scheduled_time_correct"] = bool(wanted_due and len(rows)==1 and abs((datetime.fromisoformat(rows[0]["due_utc"])-wanted_due).total_seconds()) <= 1)
        checks["other_stores_unchanged"] = after["memories"] == before["memories"] and after["notes"] == before["notes"]
    elif kind == "remember":
        checks["one_fact_with_required_content"] = len(after["memories"]) == 1 and has_terms(after["memories"][0]["content"], expected["terms"])
        checks["other_stores_unchanged"] = after["reminders"] == before["reminders"] and after["notes"] == before["notes"]
    elif kind in {"correct", "forget"}:
        target = ids["memories"][expected["target"]]
        old = {r["memory_id"]:r for r in before["memories"]}
        new = {r["memory_id"]:r for r in after["memories"]}
        if kind == "correct":
            checks["correct_target_and_version"] = target in new and new[target]["version"] == old[target]["version"] + 1 and has_terms(new[target]["content"], expected["terms"])
        else: checks["target_forgotten"] = target not in new
        checks["other_facts_unchanged"] = {k:v for k,v in new.items() if k!=target} == {k:v for k,v in old.items() if k!=target}
        checks["other_stores_unchanged"] = after["reminders"] == before["reminders"] and after["notes"] == before["notes"]
    elif kind == "note":
        checks["one_note_with_required_content"] = len(after["notes"]) == 1 and normalize(after["notes"][0]["title"]) == normalize(expected["title"]) and has_terms(after["notes"][0]["body"], expected["terms"])
        checks["other_stores_unchanged"] = after["reminders"] == before["reminders"] and after["memories"] == before["memories"]
    elif kind == "cancel_reminder":
        target = ids["reminders"][expected["target"]]
        old, new = [{r["id"]:r for r in group["reminders"]} for group in (before, after)]
        checks["only_target_cancelled"] = target in new and new[target]["status"] == "cancelled" and {k:v for k,v in new.items() if k!=target} == {k:v for k,v in old.items() if k!=target}
        checks["other_stores_unchanged"] = after["notes"] == before["notes"] and after["memories"] == before["memories"]
    else:
        checks["no_durable_tool_mutations"] = before == after and not successful_writes
        if kind == "read":
            checks["actual_read_operation"] = any(c["tool"] == expected["tool"] for c in successful)
            checks["answer_has_expected_data"] = has_terms(result.get("response", ""), expected.get("terms", []))
            if "星期" in case["prompt"] and anchor:
                weekday = "一二三四五六日"[anchor.astimezone(LOCAL).weekday()]
                checks["weekday_correct"] = any(word in result.get("response", "") for word in ("星期"+weekday, "周"+weekday, "星期天" if weekday=="日" else "星期"+weekday))
        elif kind == "clarify":
            checks["asks_for_missing_information"] = bool(re.search(r"[？?]|哪|具体|请.*(?:提供|告诉|说明)|需要.*(?:时间|内容|对象)", result.get("response", "")))
        elif kind == "no_write":
            checks["no_false_write_confirmation"] = not bool(re.match(r"^(?:(?:好的|好|没问题|收到|明白了)[，,:：。\s]*)?(?:我)?(?:已(?:经)?(?:记住|保存|删除|清除|取消|设置|创建|更正|更新)|提醒已设置)", result.get("response", "").strip()))
    return checks


async def evaluate(args):
    logger.remove(); logger.add(sys.stderr, level="WARNING")
    corpus = json.loads(args.corpus.read_text())
    selected = [c for c in corpus["cases"] if not args.ids or c["id"] in args.ids.split(",")]
    if not selected: raise ValueError("No selected cases")
    if args.limit: selected = selected[:args.limit]
    report = {"created_at":stamp(), "status":"RUNNING", "dataset_id":corpus["dataset_id"],
        "dataset_sha256":sha256(args.corpus), "scope":"frozen synthetic development requests; real llama.cpp/BGE/Mem0/APScheduler/SQLAlchemy; deterministic operation/slot/store criteria, not held-out language or physical reminder acceptance",
        "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "source_state":"working tree; checkpoint commit containing this report", "total_selected":len(selected), "cases":[]}
    write_json(args.output, report)
    config = LlamaConfig.from_manifest(ROOT / "models/manifest.json")
    with tempfile.TemporaryDirectory(prefix="assistant-tool-quality-") as temporary, ExitStack() as stack:
        root = Path(temporary)
        memory = stack.enter_context(Mem0Adapter(root / "memory", bge_path=CACHE / "models/bge-small-zh-v1.5-onnx", embedding_backend="onnx"))
        reminders = stack.enter_context(ReminderService(root / "reminders.sqlite"))
        records = stack.enter_context(Records(root / "records.sqlite"))
        for case in selected:
            user = "evaluation-" + case["id"]
            ids = seed(case, memory, reminders, records, user)
            before = state(memory, reminders, records, user)
            future_date = datetime.now(LOCAL).date() + timedelta(days=14)
            prompt = case["prompt"].format(future_date=f"{future_date.year}年{future_date.month}月{future_date.day}日", future_iso=future_date.isoformat())
            session = AssistantSession(config, memory, reminders, records, user_id=user)
            result, error, anchor, replay = {}, None, None, None
            started = time.monotonic()
            try:
                await session.start()
                async def emit(_text): pass  # Explicit typed-output observer, no speaker claim.
                result = await session.ask(prompt, emit=emit, timeout=45)
                anchor = session.turn.accepted_at
            except Exception as exc:
                error = {"type":type(exc).__name__, "message":str(exc)}
                anchor = session.turn.accepted_at
                result = {"calls":session.turn.calls, "response":"", "repair_attempted":session.repair_attempted}
            after = state(memory, reminders, records, user)
            checks = judge(case, before, after, ids, result, anchor, future_date, error)
            if case.get("replay"):
                call = next((c for c in result.get("calls",[]) if c["tool"] in WRITE_MODELS and c["outcome"].get("ok")), None)
                if call:
                    replay = await session.turn.execute(call["tool"], WRITE_MODELS[call["tool"]], call["arguments"])
                    checks["same_turn_replay_has_no_extra_mutation"] = replay == call["outcome"] and state(memory, reminders, records, user) == after
                else: checks["same_turn_replay_has_no_extra_mutation"] = False
            try: await session.close()
            except Exception as exc:
                checks["session_closed"] = False
                error = error or {"type":type(exc).__name__, "message":str(exc)}
            row = {"id":case["id"], "category":case["category"], "prompt":prompt, "status":"PASS" if all(checks.values()) else "FAIL",
                "checks":checks, "expected":case["expected"], "before":before, "after":after,
                "response":result.get("response"), "calls":result.get("calls",[]),
                "repair_attempted":result.get("repair_attempted"), "seconds":time.monotonic()-started, "error":error}
            report["cases"].append(row)
            report["completed"] = len(report["cases"])
            report["passed"] = sum(r["status"]=="PASS" for r in report["cases"])
            report["updated_at"] = stamp()
            write_json(args.output, report)
            print(json.dumps({"id":row["id"], "status":row["status"], "failed_checks":[k for k,v in checks.items() if not v], "seconds":round(row["seconds"],2)},ensure_ascii=False),flush=True)
    report["pass_rate"] = report["passed"] / len(selected)
    report["unexpected_mutations"] = sum(r["expected"]["kind"] in {"no_write","clarify","read"} and r["before"] != r["after"] for r in report["cases"])
    report["replay_failures"] = sum(r["checks"].get("same_turn_replay_has_no_extra_mutation") is False for r in report["cases"])
    report["status"] = "PASS" if report["pass_rate"] >= .95 and not report["unexpected_mutations"] and not report["replay_failures"] else "FAIL"
    report["full_corpus_run"] = len(selected) == len(corpus["cases"])
    report["categories"] = {name:{"total":sum(r["category"]==name for r in report["cases"]),
        "passed":sum(r["category"]==name and r["status"]=="PASS" for r in report["cases"])} for name in sorted(set(r["category"] for r in report["cases"]))}
    write_json(args.output, report)
    print(json.dumps({"status":report["status"], "passed":report["passed"], "total":len(selected)},ensure_ascii=False),flush=True)
    return 0 if report["status"]=="PASS" else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus",type=Path,default=ROOT/"benchmarks/corpora/tool-quality-v1.json")
    parser.add_argument("--output",type=Path,default=ROOT/"benchmarks/results/tool-quality-development.json")
    parser.add_argument("--ids")
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    return asyncio.run(evaluate(args))


if __name__=="__main__": raise SystemExit(main())
