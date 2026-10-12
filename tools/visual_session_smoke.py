"""Real VLM/BGE/Pipecat image turns, labelled RGB fixtures and a live D435i turn.

Public evidence contains synthetic answers and live-camera metadata only.
No raw image is saved. The isolated temporary transcript store is deleted.
"""
import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import time

import httpx
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.adapters.llama import LlamaConfig
from app.adapters.vision import RosVisionBridge
from app.memory import Mem0Adapter
from app.records import Records
from app.reminders import ReminderService
from app.session import AssistantSession
from app.vision_frames import VisionFrameStore
from tools.project import CACHE, stamp, write_json


async def main():
    logger.remove(); logger.add(sys.stderr, level="WARNING")
    results, checks = [], {}
    with tempfile.TemporaryDirectory(prefix="assistant-visual-") as temporary, ExitStack() as stack:
        data = Path(temporary)
        memory = stack.enter_context(Mem0Adapter(data / "memory", bge_path=CACHE / "models/bge-small-zh-v1.5-onnx", embedding_backend="onnx"))
        reminders = stack.enter_context(ReminderService(data / "reminders.sqlite"))
        records = stack.enter_context(Records(data / "records.sqlite"))
        config = LlamaConfig.from_manifest(ROOT / "models/manifest.json")
        frames = VisionFrameStore()
        color = bytes([255, 0, 0])
        stop = asyncio.Event()

        async def feed():
            while not stop.is_set():
                now = time.monotonic_ns()
                frames.add(source_time_ns=now, capture_time_ns=now, arrival_time_ns=now,
                    width=640, height=480, rgb_step=1920, rgb_encoding="rgb8", rgb=color * (640 * 480),
                    depth_step=1280, depth=bytes(640 * 480 * 2), optical_frame="SYNTHETIC_RGB_FIXTURE")
                await asyncio.sleep(.03)

        feeder = asyncio.create_task(feed())
        session = await AssistantSession(config, memory, reminders, records, vision_frames=frames).start()
        emitted = []
        async def emit(text): emitted.append(text)
        try:
            remembered = await session.ask("请记住，我喜欢茉莉花茶。", emit=emit)
            checks["real_memory_tool_write"] = any(c["tool"] == "remember" and c["outcome"]["ok"] for c in remembered["calls"])
            for expected, rgb in [("红", (255, 0, 0)), ("绿", (0, 220, 0)), ("蓝", (0, 0, 255))]:
                color = bytes(rgb)
                # Allow a new frame after switching the synthetic source.
                await asyncio.sleep(.12)
                result = await session.ask("图片里主要是什么颜色？只说颜色。", emit=emit)
                result.update(input_kind="synthetic_rgb_fixture", expected_color=expected)
                results.append(result)
                print(json.dumps({"fixture": expected, "answer": result["response"], "tokens": result["prompt_tokens"]}, ensure_ascii=False), flush=True)
            recall = await session.ask("我喜欢喝什么茶？", emit=emit)
            checks["memory_recall_after_image_turns"] = "茉莉" in recall["response"]
            checks["text_turn_has_no_image"] = recall["image"] is None and "image_url" not in str(session.bot.context.get_messages())
            checks["known_color_answers"] = all(r["expected_color"] in r["response"] for r in results)
            checks["actual_usage_matches_native_multimodal_count"] = all(
                r["model_usage"] and r["prompt_tokens"] == [x["prompt_tokens"] for x in r["model_usage"]] for r in results)
            checks["images_released_after_answers"] = frames.status()["leased_frames"] == 0
            before = len(emitted)
            task = asyncio.create_task(session.ask("观察画面，先回答主要颜色，再从一到一百逐个数数。", emit=emit, visual=True))
            observed = False
            try:
                async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                    until = time.monotonic() + 15
                    while not task.done() and time.monotonic() < until:
                        response = await client.get("http://127.0.0.1:8088/slots")
                        response.raise_for_status()
                        if any(slot.get("is_processing") for slot in response.json()):
                            observed = True
                            break
                        await asyncio.sleep(.02)
                if not observed: raise RuntimeError("The real visual request was never observed running")
                started = time.monotonic()
                await session.cancel()
                cancelled = False
                try: await task
                except asyncio.CancelledError: cancelled = True
                cancellation = {"backend_was_processing": observed, "cancelled": cancelled,
                    "cancel_to_idle_seconds": time.monotonic() - started,
                    "pipeline_interruption_acknowledged": session.interruption_received.is_set(),
                    "backend_idle": session.backend_available, "emitted_count": len(emitted) - before,
                    "leased_frames": frames.status()["leased_frames"]}
            finally:
                if not task.done():
                    await session.cancel()
                    try: await task
                    except asyncio.CancelledError: pass
            checks["real_visual_cancellation"] = cancelled and session.backend_available and cancellation["emitted_count"] == 0 and cancellation["leased_frames"] == 0
            resumed = await session.ask("图片里主要是什么颜色？只说颜色。", emit=emit)
            checks["visual_turn_after_cancel"] = "蓝" in resumed["response"]
            stop.set(); await feeder
            await asyncio.sleep(.6)
            refusal = None
            try: await session.ask("当前画面是什么？", emit=emit)
            except RuntimeError as error: refusal = str(error)
            checks["stale_source_refused"] = refusal is not None and any(x in refusal for x in ("STALE", "NO_FRESH_FRAME"))
            checks["stale_source_never_admitted"] = not session.budget.measurements
            checks["no_raw_image_in_history"] = "data:image" not in json.dumps(records.recent("owner"))
        finally:
            stop.set(); await feeder
            await session.close()

        bridge = RosVisionBridge.configured().start()
        stack.callback(bridge.close)
        live = await AssistantSession(config, memory, reminders, records, vision_frames=bridge.frames).start()
        try:
            deadline = time.monotonic() + 15
            while bridge.status()["state"] != "READY":
                if time.monotonic() > deadline: raise RuntimeError("D435i image subscription did not become ready")
                await asyncio.sleep(.05)
            # This is a connectivity/ownership check, not a ground-truth scene test.
            observed = await live.ask("请描述当前画面里能看清的内容，不确定就说明。", emit=emit)
            live_result = {"input_kind": "live_d435i", "image": observed["image"],
                "response_characters": len(observed["response"]), "elapsed_seconds": observed["elapsed_seconds"],
                "prompt_tokens": observed["prompt_tokens"], "model_usage": observed["model_usage"],
                "bridge_status": bridge.status(), "raw_image_saved": False, "scene_description_in_public_report": False,
                "transcript_storage": "isolated temporary database deleted after the probe"}
            checks["live_camera_model_response"] = bool(observed["response"].strip()) and observed["image"] is not None
            checks["live_usage_matches_native_count"] = bool(observed["model_usage"]) and observed["prompt_tokens"] == [x["prompt_tokens"] for x in observed["model_usage"]]
            checks["live_image_released"] = bridge.status()["leased_frames"] == 0
        finally: await live.close()
        # The temporary record store is deleted at the end of this probe.
    report = {"created_at": stamp(), "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "scope": "real Pipecat/llama/BGE/Mem0 image and text turns; synthetic color sanity plus live RGB-D connectivity; not held-out scene quality, acoustic cancellation or final JT acceptance",
        "synthetic_cases": results, "cancellation": cancellation, "stale_refusal": refusal, "live_camera": live_result}
    write_json(ROOT / "benchmarks/results/visual-session-smoke.json", report)
    print(json.dumps({"status": report["status"], "checks": checks}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(asyncio.run(main()))
