#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise real TCP/Uvicorn transport with synthetic producers and stalled reads.

Runs its own ephemeral loopback server. No camera, speech or model inference;
these are boundedness/fault regressions, not model performance measurements.
"""

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import socket
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_src"))
import httpx
import psutil
import uvicorn
from websockets.asyncio.client import connect

from app.config import WebConfig
from app.web import Broadcaster, create_app


async def probe(output):
    cfg = WebConfig(send_timeout=0.2, read_timeout=0.2, body_timeout=0.5)
    broadcaster = Broadcaster(cfg)
    hold = threading.Event()
    hold.set()
    started = threading.Event()
    cancelled = []

    def state():
        started.set()
        hold.wait(10)
        return {"ready": True, "web": broadcaster.snapshot()}

    def cancel(**kwargs):
        cancelled.append(kwargs["turn_id"])
        return {"accepted": True}

    app = create_app(
        broadcaster, state=state, cancel=cancel, submit=lambda *a: {"accepted": True}
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(32)
    port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            ws_max_size=cfg.ws_message_bytes,
            ws_max_queue=4,
            limit_concurrency=16,
            timeout_graceful_shutdown=5,
        )
    )
    running = asyncio.create_task(server.serve(sockets=[listener]))
    report = {"scope": __doc__, "config": asdict(cfg), "status": "RUNNING"}
    process = psutil.Process()

    def resources():
        return {
            "fds": process.num_fds(),
            "threads": process.num_threads(),
            "pss_bytes": process.memory_full_info().pss,
            "clients": broadcaster.client_count,
        }

    async def initialized(ws):
        while json.loads(await ws.recv())["type"] != "capabilities":
            pass

    async def wait_for(predicate, seconds=3):
        async with asyncio.timeout(seconds):
            while not predicate():
                await asyncio.sleep(0.01)

    async def slow_body():
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            b"POST /api/chat HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{"
        )
        await writer.drain()
        return reader, writer

    try:
        await wait_for(lambda: server.started)
        async with httpx.AsyncClient(base_url=base, timeout=5, trust_env=False) as http:
            # Warm up library allocations and both native read worker slots.
            await asyncio.gather(http.get("/api/state"), http.get("/api/state"))
            async with connect(base.replace("http", "ws") + "/ws") as ws:
                await initialized(ws)
            await wait_for(lambda: broadcaster.client_count == 0)
            report["before_cycles"] = resources()
            for _ in range(20):
                async with connect(base.replace("http", "ws") + "/ws") as ws:
                    await initialized(ws)
                await wait_for(lambda: broadcaster.client_count == 0)
            report["after_cycles"] = resources()
            report["cycles"] = 20
            assert report["after_cycles"]["fds"] == report["before_cycles"]["fds"]
            assert (
                report["after_cycles"]["threads"] == report["before_cycles"]["threads"]
            )
            async with connect(base.replace("http", "ws") + "/ws") as ws:
                await initialized(ws)
                hold.clear()
                started.clear()
                replies = await asyncio.gather(
                    http.get("/api/state"), http.get("/api/state")
                )
                assert [r.status_code for r in replies] == [503, 503]
                for _ in range(20):
                    assert (await http.get("/api/state")).status_code == 503
                assert app.state.calls.snapshot()["in_flight"] == 2
                t = time.monotonic()
                await ws.send(json.dumps({"type": "cancel", "turn_id": "a" * 32}))
                await wait_for(lambda: len(cancelled) == 1)
                report["stalled_reads"] = {
                    "workers": app.state.calls.snapshot(),
                    "cancel_observed_seconds": time.monotonic() - t,
                    "cancel_transport": "existing WebSocket",
                    "threads": process.num_threads(),
                }
                hold.set()
                await wait_for(lambda: app.state.calls.snapshot()["in_flight"] == 0)
                assert (await http.get("/api/state")).status_code == 200
                bodies = await asyncio.gather(*(slow_body() for _ in range(4)))
                await asyncio.sleep(0.03)
                assert (
                    await http.post("/api/chat", json={"text": "extra"})
                ).status_code == 503
                assert (
                    await http.post("/api/cancel", json={"turn_id": "b" * 32})
                ).status_code == 202
                codes = []
                for reader, writer in bodies:
                    data = await asyncio.wait_for(reader.read(), 2)
                    codes.append(int(data.split(b" ")[1]))
                    writer.close()
                    await writer.wait_closed()
                assert codes == [408] * 4
                large = await http.post("/api/chat", content=b"x" * 33000)

                async def chunks():
                    yield b"x" * 20000
                    yield b"x" * 20000

                chunked = await http.post("/api/chat", content=chunks())
                assert large.status_code == chunked.status_code == 413
                report["body_pressure"] = {
                    "slow_readers": 4,
                    "slow_codes": codes,
                    "extra_data_code": 503,
                    "cancel_code": 202,
                    "declared_oversize": large.status_code,
                    "chunked_oversize": chunked.status_code,
                }
            await wait_for(lambda: broadcaster.client_count == 0)
            slow = await connect(
                base.replace("http", "ws") + "/ws",
                max_queue=1,
                compression=None,
                close_timeout=0.2,
            )
            try:
                await initialized(slow)
                for n in range(120):
                    broadcaster.send({"type": "frame", "n": n, "data": "x" * 262144})
                    await asyncio.sleep(0.02)
                    if broadcaster.client_count == 0:
                        break
                await wait_for(lambda: broadcaster.client_count == 0)
                report["slow_viewer"] = {
                    "produced_frames": n + 1,
                    "released": True,
                    "transport": broadcaster.snapshot(),
                }
            finally:
                slow.transport.abort()
                await slow.wait_closed()
            report["final"] = resources()
            report["status"] = "PASSED_WITH_DECLARED_SCOPE"
    except Exception as exc:
        report.update(status="FAILED", error=repr(exc))
        raise
    finally:
        hold.set()
        server.should_exit = True
        await asyncio.wait_for(running, 10)
        listener.close()
        report["after_shutdown"] = resources()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(probe(parser.parse_args().output))
