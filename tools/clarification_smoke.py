"""Actual persistent dialogue for missing reminder time, withdrawal and stop."""
import asyncio
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
import tempfile
import uuid
from zoneinfo import ZoneInfo

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.adapters.llama import LlamaConfig
from app.memory import Mem0Adapter
from app.records import Records
from app.reminders import ReminderService
from app.session import AssistantSession
from app.voice_assistant import VoiceAssistant
from tools.project import CACHE, DATA, stamp, write_json
from tools.voice_assistant_smoke import synthesize


async def main(voice_mode=False):
    logger.remove(); logger.add(sys.stderr, level="WARNING")
    results=[]
    async def emit(_text): pass
    with tempfile.TemporaryDirectory(prefix="assistant-clarification-") as temporary:
        data=Path(temporary)
        with Mem0Adapter(data/"memory",bge_path=CACHE/"models/bge-small-zh-v1.5-onnx",embedding_backend="onnx") as memory, Records(data/"records.sqlite") as records, ReminderService(data/"reminders.sqlite") as reminders:
            config=LlamaConfig.from_manifest(ROOT/"models/manifest.json")
            voice=None
            if voice_mode:
                artifacts=DATA/"fixtures"/("voice-clarification-"+uuid.uuid4().hex)
                artifacts.mkdir(parents=True)
                voice=await VoiceAssistant(config,memory,reminders,records,output_path=artifacts/"answers.wav").start()
                session=voice.session
            else:
                session=await AssistantSession(config,memory,reminders,records).start()
            sequence=0
            async def ask(text):
                nonlocal sequence
                if not voice:return await session.ask(text,emit=emit)
                path=artifacts/f"input-{sequence}.wav";sequence+=1
                await synthesize(text,path)
                return await voice.run_file(path)
            try:
                first=await ask("明天提醒我交报告。");results.append(first)
                empty_before_reply=not reminders.list_reminders("owner")
                second=await ask("上午八点");results.append(second)
                rows=reminders.list_reminders("owner")
                local=session.turn.accepted_at.astimezone(ZoneInfo("Asia/Shanghai"))
                expected=(local+timedelta(days=1)).replace(hour=8,minute=0,second=0,microsecond=0)
                date_correct=len(rows)==1 and datetime.fromisoformat(rows[0]["due_utc"])==expected and rows[0]["text"]=="交报告"
                results.append(await ask("设置一个喝水提醒。"))
                results.append(await ask("算了，不用设置了。"))
                withdrawn_has_no_effect=len(reminders.list_reminders("owner"))==1
                results.append(await ask("设置一个散步提醒。"))
                await session.cancel()
                results.append(await ask("十分钟后"))
                checks={"missing_time_did_not_create":empty_before_reply,
                    "clarification_did_not_invoke_model":first["model_request_skipped"] and not first["model_usage"],
                    "followup_preserved_tomorrow_and_action":date_correct,
                    "withdrawal_did_not_create":withdrawn_has_no_effect,
                    "explicit_stop_cleared_pending_authorization":len(reminders.list_reminders("owner"))==1,
                    "all_turns_have_visible_response":all(r["response"].strip() for r in results)}
                if voice: checks["all_voice_outputs_flushed_as_fixture"] = all(r["speech_receipt"]["channel"]=="fixture" and r["speech_receipt"]["bytes_written"]>0 for r in results)
            finally:
                if voice:await voice.close()
                else:await session.close()
    report={"created_at":stamp(),"status":"PASS" if all(checks.values()) else "FAIL",
        "scope":("synthetic speech-file dialogue with real Silero/ASR/TTS/WAV output" if voice_mode else "synthetic typed dialogue") + "; actual Pipecat, llama.cpp, BGE and SDK stores; deterministic clarification uses native response frames and no model; no physical audio acceptance",
        "checks":checks,"cases":results}
    write_json(ROOT/("benchmarks/results/voice-clarification-smoke.json" if voice_mode else "benchmarks/results/clarification-smoke.json"),report)
    print(json.dumps({"status":report["status"],"checks":checks},ensure_ascii=False))
    return 0 if report["status"]=="PASS" else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice",action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args().voice)))
