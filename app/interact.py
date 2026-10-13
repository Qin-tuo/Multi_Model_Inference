"""Local text console for the Pipecat assistant; microphone transport is separate."""
from __future__ import annotations

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
from app.session import AssistantSession
from app.notifications import NotificationPump, TextNotificationSink
from app.proactive import VisualFeedback
from tools.project import CACHE, DATA, ROOT


async def console(args):
    config = LlamaConfig.from_manifest(ROOT / "models/manifest.json")
    runtime = json.loads((ROOT / "configs/runtime.yaml").read_text())
    with ExitStack() as stack:
        vision = None
        if not args.no_vision:
            vision = RosVisionBridge.configured().start()
            stack.callback(vision.close)
        memory = stack.enter_context(open_memory(args.data_dir/'memory',settings=runtime))
        reminder_options = {key: runtime["reminders"][key] for key in
            ("notification_ttl_seconds", "max_delivery_attempts", "retry_delay_seconds") if key in runtime["reminders"]}
        reminders = stack.enter_context(ReminderService(args.data_dir / "reminders.sqlite", **reminder_options))
        feedback = VisualFeedback(reminders, vision.perception, args.user) if vision else None
        if feedback:
            stack.callback(feedback.close)
            if args.visual_feedback is not None: feedback.set_muted(args.visual_feedback == 'off')
        records = stack.enter_context(Records(args.data_dir / "records.sqlite",
            retention_days=runtime["memory"]["transcript_retention_days"]))
        session = AssistantSession(config, memory, reminders, records, user_id=args.user,
            session_id=args.session_id, context_tokens=runtime["llama"]["context_tokens"],
            vision_frames=vision.frames if vision else None,memory_top_k=runtime['memory']['top_k'],
            memory_context_tokens=runtime['memory']['context_token_limit'])
        await session.start()
        pending = None
        errors = []

        async def emit_notice(text):
            print("通知：" + text, flush=True)

        occupied = lambda: session.busy.locked()
        notifications = NotificationPump(reminders, args.user,
            TextNotificationSink(reminders, emit_notice, occupied=occupied), occupied=occupied)
        if feedback: notifications.before_claim = feedback.tick
        notification_task = asyncio.create_task(notifications.run())

        async def emit(text):
            print("助手：" + text, flush=True)

        async def answer(text, *, visual=None):
            try:
                await session.ask(text, emit=emit, visual=visual)
            except asyncio.CancelledError:
                print("已停止本轮回答。已提交的操作可通过查询确认。", flush=True)
            except Exception as error:
                errors.append(type(error).__name__)
                print("本轮未完成：" + str(error), flush=True)

        async def command(line):
            nonlocal pending, session
            line = line.strip()
            if not line: return True
            if line == "/quit": return False
            if line == "/stop":
                await session.cancel()
                if pending: await pending
                return True
            if pending and not pending.done() and line not in {
                    '/scene', '/mute-visual on', '/mute-visual off', '/mute-reminders on', '/mute-reminders off'}:
                print("正在回答；可输入 /stop 停止。", flush=True)
                return True
            if line == "/help":
                print("直接输入中文对话。命令：/look [问题] 看当前画面，/scene 人物与手势状态，/new 新会话，/stop 停止，/memories 记忆，/notes 笔记，/reminders 提醒，/history 已输出历史，/mute-reminders on或off 通知静音，/mute-visual on或off 手势静音，/quit 退出。", flush=True)
            elif line == '/scene':
                print(json.dumps(feedback.status(),ensure_ascii=False,indent=2) if feedback else '视觉未启用。', flush=True)
            elif line in {'/mute-visual on','/mute-visual off'}:
                if feedback:
                    feedback.set_muted(line.endswith(' on'))
                    print('手势主动回应已静音。' if feedback.muted else '手势主动回应已开启。', flush=True)
                else: print('视觉未启用。', flush=True)
            elif line == "/look" or line.startswith("/look "):
                pending = asyncio.create_task(answer(line[5:].strip() or "你现在看到了什么？", visual=True))
            elif line in {"/mute-reminders on", "/mute-reminders off"}:
                muted = line.endswith(" on")
                reminders.set_muted(args.user, muted)
                print("提醒静音已开启。" if muted else "提醒静音已关闭。", flush=True)
            elif line == "/new":
                await session.close()
                session = await AssistantSession(config, memory, reminders, records, user_id=args.user,
                    context_tokens=runtime["llama"]["context_tokens"], vision_frames=vision.frames if vision else None,
                    memory_top_k=runtime['memory']['top_k'],memory_context_tokens=runtime['memory']['context_token_limit']).start()
                print("已开启新会话。", flush=True)
            elif line in {"/memories", "/notes", "/reminders", "/history"}:
                if line == "/memories": rows = memory.list_memories(args.user, limit=20)
                elif line == "/notes": rows = records.list_notes(args.user, limit=20)
                elif line == "/reminders": rows = reminders.list_reminders(args.user)
                else: rows = records.recent(args.user, limit=20)
                print(json.dumps(rows, ensure_ascii=False, indent=2) if rows else "暂无记录。", flush=True)
            elif line.startswith("/"):
                print("未知命令，输入 /help 查看。", flush=True)
            else:
                pending = asyncio.create_task(answer(line))
            return True

        print(f"本地助手已就绪，档案：{args.user}，会话：{session.session_id}。输入 /help 查看命令。", flush=True)
        try:
            if args.input_file:
                if args.input_file.stat().st_size > 65536: raise ValueError("Input file exceeds 64 KiB")
                for line in args.input_file.read_text().splitlines():
                    if not await command(line): break
                    if pending: await pending
            else:
                reader = asyncio.StreamReader(limit=8192)
                transport, _ = await asyncio.get_running_loop().connect_read_pipe(
                    lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
                try:
                    while raw := await reader.readline():
                        if not await command(raw.decode("utf-8")): break
                finally:
                    transport.close()
        finally:
            await notifications.stop()
            await notification_task
            if pending and not pending.done():
                await session.cancel()
                await pending
            await session.close()
        return 1 if errors else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="owner", help="Manually selected isolated profile")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--session-id", help="Resume delivered text from this owner's session")
    parser.add_argument("--input-file", type=Path, help="Bounded local text script, executed sequentially")
    parser.add_argument("--no-vision", action="store_true", help="Text-only operation without a ROS image subscription")
    parser.add_argument('--visual-feedback', choices=['on','off'], help='Persist gesture greeting preference; omit to retain it')
    args = parser.parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    try:
        return asyncio.run(console(args))
    except (RuntimeError, ValueError, OSError) as error:
        print("启动或运行失败：" + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
