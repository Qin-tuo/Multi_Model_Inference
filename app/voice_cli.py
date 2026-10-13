"""Persistent local voice assistant, with a paced WAV mode for device-free checks."""
import argparse
import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
import sys

from loguru import logger
from app.adapters.llama import LlamaConfig
from app.adapters.vision import RosVisionBridge
from app.memory_runtime import open_memory
from app.records import Records
from app.reminders import ReminderService
from app.voice_assistant import VoiceAssistant
from app.proactive import VisualFeedback
from tools.project import ROOT, CACHE, DATA


async def run(args):
    settings = json.loads((ROOT / "configs/runtime.yaml").read_text())
    with ExitStack() as stack:
        bridge = None
        if not args.no_vision:
            bridge = RosVisionBridge.configured().start()
            stack.callback(bridge.close)
        memory = stack.enter_context(open_memory(args.data_dir/'memory',settings=settings))
        options = {key: settings["reminders"][key] for key in
                   ("notification_ttl_seconds", "max_delivery_attempts", "retry_delay_seconds")}
        reminders = stack.enter_context(ReminderService(args.data_dir / "reminders.sqlite", **options))
        feedback = VisualFeedback(reminders, bridge.perception, args.user) if bridge else None
        if feedback:
            stack.callback(feedback.close)
            if args.visual_feedback is not None: feedback.set_muted(args.visual_feedback == 'off')
        records = stack.enter_context(Records(args.data_dir / "records.sqlite",
            retention_days=settings["memory"]["transcript_retention_days"]))
        voice = VoiceAssistant(LlamaConfig.from_manifest(ROOT / "models/manifest.json"), memory, reminders, records,
            user_id=args.user, session_id=args.session_id, vision_frames=bridge.frames if bridge else None,
            output_path=args.output_wav, input_device=args.input_device, output_device=args.output_device,
            context_tokens=settings["llama"]["context_tokens"])
        if feedback: voice.notifications.before_claim = feedback.tick
        try:
            await voice.start()
            if args.input_wav:
                for path in args.input_wav:
                    result = await voice.run_file(path)
                    print("用户：" + result["transcript"], flush=True)
                    print("助手：" + result["response"], flush=True)
                print("音频已保存：" + str(args.output_wav), flush=True)
            else:
                print(f"语音助手已就绪，当前档案：{args.user}。按 Ctrl+C 退出。", flush=True)
                await asyncio.Event().wait()
        finally:
            await voice.close()
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps({"output_channel": "fixture" if args.input_wav else "audio",
                    "turns": voice.results, "errors": voice.errors}, ensure_ascii=False, indent=2) + "\n")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="owner")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--session-id")
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument('--visual-feedback', choices=['on','off'], help='Persist gesture greeting preference')
    parser.add_argument("--input-wav", type=Path, nargs="+", help="One spoken utterance per file, PCM16/16kHz/mono, at most 30s")
    parser.add_argument("--output-wav", type=Path, help="New WAV file for answers and reminders; never overwrites")
    parser.add_argument("--report", type=Path, help="Optional detailed local report, including transcripts")
    parser.add_argument("--input-device", type=int)
    parser.add_argument("--output-device", type=int)
    args = parser.parse_args(argv)
    if bool(args.input_wav) != bool(args.output_wav): parser.error("--input-wav and --output-wav must be used together")
    if args.input_wav and (args.input_device is not None or args.output_device is not None):
        parser.error("Select file input or audio devices")
    if not args.input_wav and (args.input_device is None or args.output_device is None):
        parser.error("Local audio requires --input-device and --output-device")
    if args.output_wav and args.output_wav.exists(): parser.error("Output WAV already exists; choose a new file")
    logger.remove(); logger.add(sys.stderr, level="WARNING")
    try: return asyncio.run(run(args))
    except KeyboardInterrupt: return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print("语音助手未完成：" + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__": raise SystemExit(main())
