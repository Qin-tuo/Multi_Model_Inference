#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise real VAD/ASR/LLM/TTS voice interruption with a paced PCM subprocess.

The capture/discovery source is explicitly substituted. Production MicRecorder,
VAD loop, turn controller, native models, SQL actions and Web routes are used.
No microphone, loudspeaker, AEC or acoustic latency qualification is claimed.
"""

import argparse
import asyncio
import json
from pathlib import Path
import socket
import sys
import threading
import time
import uuid
import wave


def emit(args):
    with wave.open(str(args.input)) as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (
            16000,
            1,
            2,
        )
        pcm = audio.readframes(audio.getnframes())
    pcm += b"\0" * ((-len(pcm)) % 1024)
    frames = [pcm[n : n + 1024] for n in range(0, len(pcm), 1024)]
    nonce = None
    position = len(frames)
    stalled = False
    tick = time.monotonic()
    while True:
        if args.control.exists():
            value = json.loads(args.control.read_text())
            if value["nonce"] != nonce:
                nonce = value["nonce"]
                stalled = value["mode"] == "stall"
                position = 0 if value["mode"] == "speech" else len(frames)
        if not stalled:
            sys.stdout.buffer.write(
                frames[position] if position < len(frames) else b"\0" * 1024
            )
            sys.stdout.buffer.flush()
            position += 1
        tick += 0.032
        time.sleep(max(0, tick - time.monotonic()))


async def probe(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import httpx
    import psutil
    import uvicorn
    from websockets.asyncio.client import connect
    from app.config import Config
    from app import pipeline
    import run_web_vision_chat as entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    folder = args.output.parent
    cfg = Config.load()
    cfg.audio.input_mode = "device"
    cfg.audio.output_mode = "file"
    cfg.audio.input_device = "__pcm_replay__"
    cfg.audio.output_directory = str(folder / "audio")
    cfg.actions.database = str(folder / "actions.sqlite")
    cfg.validate()
    if Path(cfg.actions.database).exists():
        raise RuntimeError("Use a fresh isolated directory for this probe")
    report = {
        "scope": __doc__,
        "status": "RUNNING",
        "cases": [],
        "effective_config": cfg.effective(),
    }
    microphone = None

    class ReplayMic(pipeline.MicRecorder):
        def __init__(self, *a, **kw):
            nonlocal microphone
            super().__init__(*a, **kw)
            microphone = self

        def _record_command(self):
            return [
                sys.executable,
                str(Path(__file__).resolve()),
                "--emit",
                "--input",
                str(args.input),
                "--control",
                str(args.control),
            ]

    original_mic, original_find = entry.MicRecorder, entry.find_alsa_device
    original_source, original_selector = (
        pipeline.find_pa_source,
        pipeline.SpeakerSelector,
    )
    entry.MicRecorder = ReplayMic
    entry.find_alsa_device = lambda *a: (99, 0, "Explicit PCM replay substitute")
    pipeline.find_pa_source = lambda *a: None

    class NoSpeaker:
        def __init__(self, *a, **k):
            pass

        def get_sink(self):
            return None

        def state(self):
            return {"speakers": [], "selected": None}

    pipeline.SpeakerSelector = NoSpeaker
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(32)
    base = f"http://127.0.0.1:{listener.getsockname()[1]}"
    app = entry.build_app(cfg, no_camera=True)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            log_level="warning",
            ws_max_size=cfg.web.ws_message_bytes,
            ws_max_queue=4,
        )
    )
    running = asyncio.create_task(server.serve(sockets=[listener]))
    events = []
    driver = None
    browser_session = None

    def trigger(mode):
        temporary = args.control.with_suffix(".tmp")
        temporary.write_text(json.dumps({"nonce": uuid.uuid4().hex, "mode": mode}))
        temporary.replace(args.control)

    async def wait(predicate, seconds=40):
        async with asyncio.timeout(seconds):
            while True:
                value = await predicate()
                if value:
                    return value
                await asyncio.sleep(0.02)

    try:

        async def started():
            if running.done():
                running.result()
                raise RuntimeError("App exited before ready")
            return server.started

        await wait(started)
        source_pid = microphone._proc.pid
        report["source_pid"] = source_pid
        async with httpx.AsyncClient(
            base_url=base, timeout=5, trust_env=False
        ) as client:
            if args.webdriver:
                driver = httpx.AsyncClient(
                    base_url=args.webdriver, timeout=30, trust_env=False
                )

                async def call(method, path, payload=None):
                    value = (
                        (await driver.request(method, path, json=payload))
                        .raise_for_status()
                        .json()["value"]
                    )
                    if isinstance(value, dict) and value.get("error"):
                        raise RuntimeError(value)
                    return value

                browser_session = (
                    await call(
                        "POST",
                        "/session",
                        {
                            "capabilities": {
                                "alwaysMatch": {
                                    "browserName": "firefox",
                                    "moz:firefoxOptions": {"args": ["-headless"]},
                                }
                            }
                        },
                    )
                )["sessionId"]

                async def js(script):
                    return await call(
                        "POST",
                        f"/session/{browser_session}/execute/sync",
                        {"script": script, "args": []},
                    )

                await call("POST", f"/session/{browser_session}/url", {"url": base})

                async def connected():
                    return await js(
                        "return document.querySelector('#connStatus').textContent.includes('Connected') && !document.querySelector('#sendText').disabled"
                    )

                await wait(connected)
                await js(
                    "const original=WebSocket.prototype.send;WebSocket.prototype.send=function(raw){const m=JSON.parse(raw);if(m.type==='cancel')window.cancelTarget=m.turn_id;return original.call(this,raw);}"
                )

            async def state():
                return (await client.get("/api/state")).raise_for_status().json()

            async def idle():
                s = await state()
                return (
                    s
                    if s["active_turn"] is None and s["voice"]["pending"] is None
                    else None
                )

            async def ask(text):
                return (
                    (await client.post("/api/chat", json={"text": text}))
                    .raise_for_status()
                    .json()["turn_id"]
                )

            async def final(key):
                async def match():
                    s = await state()
                    r = s.get("last_result")
                    return (
                        r
                        if r and r["turn_id"] == key and r.get("resources_released")
                        else None
                    )

                return await wait(match)

            async with connect(
                base.replace("http://", "ws://") + "/ws", max_size=2**21
            ) as ws:

                async def receive():
                    async for raw in ws:
                        value = json.loads(raw)
                        if value["type"] in {
                            "voice_state",
                            "status",
                            "done",
                            "cancelled",
                            "transcript",
                            "error",
                        }:
                            events.append(value)
                        if len(events) > 2000:
                            raise RuntimeError("Probe event budget exceeded")

                receiver = asyncio.create_task(receive())
                try:
                    for name in [
                        "complete_new_voice",
                        "cancel_pending_voice",
                        "mute_pending_voice",
                        "stalled_capture",
                    ]:
                        await wait(idle)
                        old = await ask(
                            "请详细解释十种常见数据结构，每项说明具体用途和优缺点。"
                        )

                        async def busy():
                            slots = (
                                (await client.get(cfg.llm.base_url + "/slots"))
                                .raise_for_status()
                                .json()
                            )
                            return any(s["is_processing"] for s in slots)

                        await wait(busy, 10)
                        trigger("speech")

                        async def pending():
                            s = await state()
                            v = s["voice"]["pending"]
                            return (
                                v
                                if v
                                and s["voice"]["last"].get("interrupted_turn") == old
                                else None
                            )

                        voice = await wait(pending, 8)
                        new = voice["turn_id"]
                        assert voice["phase"] == "capturing"
                        case = {
                            "name": name,
                            "old_turn": old,
                            "voice_turn": new,
                            "confirmed_before_endpoint": True,
                        }
                        report["cases"].append(case)
                        if name == "cancel_pending_voice":
                            if browser_session:

                                async def voice_visible():
                                    return await js(
                                        "return document.querySelector('#voiceInfo').textContent.includes('正在听取') && !document.querySelector('#cancelTurn').disabled"
                                    )

                                await wait(voice_visible, 2)
                                await js(
                                    "document.querySelector('#cancelTurn').click()"
                                )
                                assert await js("return window.cancelTarget") == new
                                case["browser_cancel_target"] = new
                            else:
                                await ws.send(
                                    json.dumps({"type": "cancel", "turn_id": new})
                                )
                        elif name == "mute_pending_voice":
                            await ws.send(json.dumps({"type": "ptt", "active": False}))
                        elif name == "stalled_capture":
                            trigger("stall")
                        old_result = await final(old)
                        assert (
                            old_result["status"] == "cancelled"
                            and not old_result["text"]
                        )
                        assert (
                            await client.get("/api/audio/" + old)
                        ).status_code == 404
                        case["old_result"] = old_result
                        if name == "complete_new_voice":
                            result = await final(new)
                            assert (
                                result["status"] == "complete"
                                and result["input_kind"] == "microphone"
                            )
                            assert (
                                "语音连接成功" in result["text"]
                                and len(result["asr"]) == 1
                            )
                            assert (
                                await client.get(result["audio_url"])
                            ).status_code == 200
                            case["new_result"] = result
                        else:
                            await asyncio.sleep(4)
                            s = await wait(idle)
                            assert s["last_result"]["turn_id"] == old
                            assert not any(
                                e.get("turn_id") == new
                                and e["type"] in {"transcript", "done"}
                                for e in events
                            )
                            case["voice_state"] = s["voice"]
                            case["live_audio"] = s["live_audio"]
                            if name == "mute_pending_voice":
                                assert (
                                    microphone.alive
                                    and not microphone.listening.is_set()
                                )
                                await ws.send(
                                    json.dumps({"type": "ptt", "active": True})
                                )

                                async def unmuted():
                                    return microphone.listening.is_set()

                                await wait(unmuted)
                            if name == "stalled_capture":
                                assert (
                                    not microphone.alive
                                    and "stalled" in microphone.last_error
                                )
                                assert (
                                    microphone._proc is None
                                    and not psutil.pid_exists(source_pid)
                                )
                        case["status"] = "PASSED"
                        print(
                            json.dumps(
                                {"case": name, "status": "PASSED"}, ensure_ascii=False
                            ),
                            flush=True,
                        )
                    next_key = await ask("请只回答：输入故障后文字仍然可用。")
                    report["after_input_failure"] = await final(next_key)
                    assert report["after_input_failure"]["status"] == "complete"
                    report["events"] = events
                    if browser_session:
                        report["browser"] = await js(
                            "return {phase:document.querySelector('#pipeStatus').textContent,voice:document.querySelector('#voiceInfo').textContent,audio:document.querySelector('#liveAudioInfo').textContent,send_enabled:!document.querySelector('#sendText').disabled}"
                        )
                        assert (
                            report["browser"]["send_enabled"]
                            and "已停止" in report["browser"]["audio"]
                        )
                    report["status"] = "PASSED_WITH_DECLARED_SCOPE"
                finally:
                    receiver.cancel()
                    await asyncio.gather(receiver, return_exceptions=True)
    except BaseException as exc:
        report.update(status="FAILED", error=repr(exc))
        raise
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(running, 30)
            report["shutdown"] = {
                "source_released": not microphone or microphone._proc is None,
                "owned_threads": [
                    t.name
                    for t in threading.enumerate()
                    if t.name
                    in {
                        "audio-vad",
                        "microphone-reader",
                        "conversation",
                        "speech-output",
                        "model-deadline",
                    }
                ],
            }
        finally:
            if browser_session:
                try:
                    await driver.delete(f"/session/{browser_session}")
                finally:
                    await driver.aclose()
            listener.close()
            entry.MicRecorder, entry.find_alsa_device = original_mic, original_find
            pipeline.find_pa_source, pipeline.SpeakerSelector = (
                original_source,
                original_selector,
            )
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--webdriver")
    args = parser.parse_args()
    if args.emit:
        emit(args)
    elif args.output is None:
        parser.error("--output is required for the application probe")
    else:
        asyncio.run(probe(args))
