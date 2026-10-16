# SPDX-License-Identifier: Apache-2.0
"""Real loop/ASGI regressions for transport ownership, overload and cancellation."""

import asyncio
from dataclasses import replace
import json
import threading
import time

import httpx
import pytest
from starlette.websockets import WebSocket

from app.config import WebConfig
from app.web import Broadcaster, create_app
from app.web_limits import BoundedCalls, BusyRead, ClientBuffer, encode


def test_paused_loop_has_one_callback_and_latest_bounded_frame():
    loop = asyncio.new_event_loop()
    b = Broadcaster()
    b.set_loop(loop)
    q = ClientBuffer(16, 4 * 1024 * 1024)
    b.register(q)
    try:
        for n in range(10000):
            b.send({"type": "frame", "n": n, "data": "x" * 1024})
        assert len(loop._ready) == 1  # diagnostic of actual pending callbacks
        assert (
            b.snapshot()["bridge_events"] == 1 and b.snapshot()["bridge_bytes"] < 1200
        )
        loop.run_until_complete(asyncio.sleep(0))
        assert len(q.items) == 1 and json.loads(q.items[0].wire)["n"] == 9999
        assert not b.snapshot()["drain_scheduled"]
    finally:
        b.set_loop(None)
        loop.close()


def test_cancel_expiry_immutable_encoding_and_old_loop_epoch():
    loop = asyncio.new_event_loop()
    b = Broadcaster()
    b.set_loop(loop)
    q = ClientBuffer(16, 1024 * 1024)
    b.register(q)
    event = threading.Event()
    try:
        msg = {"type": "token", "text": "owned"}
        b.send(msg)
        msg["text"] = "mutated"
        b.send({"type": "token", "text": "cancelled"}, cancel_event=event)
        event.set()
        b.send(
            {"type": "frame", "frame": {"received_monotonic": time.monotonic() - 10}}
        )
        loop.run_until_complete(asyncio.sleep(0))
        assert [json.loads(m.wire)["text"] for m in q.items] == ["owned"]
        b.send({"type": "token", "text": "old epoch"})
        b.set_loop(loop)
        fresh = ClientBuffer(16, 1024 * 1024)
        b.register(fresh)
        b.send({"type": "token", "text": "new epoch"})
        loop.run_until_complete(asyncio.sleep(0))
        assert [json.loads(m.wire)["text"] for m in fresh.items] == ["new epoch"]
    finally:
        b.set_loop(None)
        loop.close()


def test_reliable_overflow_and_invalid_wire_are_explicit():
    loop = asyncio.new_event_loop()
    b = Broadcaster()
    b.set_loop(loop)
    q = ClientBuffer(16, 1024 * 1024)
    b.register(q)
    try:
        for n in range(1000):
            b.send({"type": "token", "text": str(n)})
        assert len(loop._ready) == 1 and b.snapshot()["bridge_events"] == 1
        loop.run_until_complete(asyncio.sleep(0))
        assert q.failed and len(q.items) == 1 and q.items[0].kind == "overflow"
        assert not b.send(["invalid"])
        assert not b.send({"type": []})
        assert not b.send({"type": "frame", "data": "x" * (b.config.message_bytes + 1)})
        assert b.snapshot()["rejected_messages"] == 3
    finally:
        b.set_loop(None)
        loop.close()


def test_snapshot_covers_delayed_events_but_preserves_active_tokens():
    q = ClientBuffer(16, 10000)

    def item(kind, turn, seq):
        return replace(encode({"type": kind, "turn_id": turn}, 1000), sequence=seq)

    q.put(item("token", "active", 3))
    q.put(item("status", "old", 4))
    q.prime([encode({"type": "capabilities"}, 1000)], 5, "active")
    q.put(item("status", "old", 5))
    q.put(item("action_receipt", "active", 5))
    q.put(item("done", "active", 6))
    assert [m.kind for m in q.items] == [
        "capabilities",
        "token",
        "action_receipt",
        "done",
    ]


def test_timed_out_native_reads_keep_slots_until_actual_release():
    async def check():
        release = threading.Event()
        pool = BoundedCalls(2)
        try:
            results = await asyncio.gather(
                *(pool.run(lambda: release.wait(2), 0.03) for _ in range(2)),
                return_exceptions=True,
            )
            assert all(isinstance(r, TimeoutError) for r in results)
            for _ in range(20):
                with pytest.raises(BusyRead):
                    await pool.run(lambda: None, 0.01)
            assert pool.snapshot()["in_flight"] == 2
            assert (
                len([t for t in threading.enumerate() if t.name.startswith("web-read")])
                == 2
            )
        finally:
            release.set()
            await pool.close()
        assert pool.snapshot()["in_flight"] == 0

    asyncio.run(check())


def test_control_is_singleflight_and_does_not_occupy_event_loop():
    async def check():
        release = threading.Event()
        pool = BoundedCalls(2)
        try:
            pool.submit(lambda: release.wait(2), control=True)
            with pytest.raises(BusyRead):
                pool.submit(lambda: None, control=True)
            assert await pool.run(lambda: 42, 0.5) == 42
        finally:
            release.set()
            await pool.close()

    asyncio.run(check())


def scope(kind="websocket", path="/ws"):
    return {
        "type": kind,
        "asgi": {"version": "3.0"},
        "scheme": "ws" if kind == "websocket" else "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 8093),
        "root_path": "",
        "method": "POST",
        "http_version": "1.1",
        "subprotocols": [],
    }


def test_concurrent_delayed_handshakes_reserve_four_and_cancel_releases():
    async def check():
        b = Broadcaster()
        app = create_app(b)
        gates = [asyncio.Event() for _ in range(6)]
        started = [asyncio.Event() for _ in range(6)]
        responses = [[] for _ in range(6)]

        async def connection(i):
            async def recv():
                return {"type": "websocket.connect"}

            async def send(message):
                responses[i].append(message)
                started[i].set()
                if message["type"] == "websocket.accept":
                    await gates[i].wait()

            await app(scope(), recv, send)

        async with app.router.lifespan_context(app):
            tasks = [asyncio.create_task(connection(i)) for i in range(6)]
            await asyncio.wait_for(asyncio.gather(*(s.wait() for s in started)), 1)
            assert b.client_count == 4
            assert sum(rows[0]["type"] == "websocket.close" for rows in responses) == 2
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            assert b.client_count == 0

    asyncio.run(check())


def test_slow_sender_times_out_and_unregisters():
    async def check():
        b = Broadcaster(WebConfig(send_timeout=0.03))
        app = create_app(b)
        incoming = asyncio.Queue()
        incoming.put_nowait({"type": "websocket.connect"})
        messages = []

        async def send(message):
            messages.append(message)
            if message["type"] == "websocket.send":
                await asyncio.Event().wait()

        async with app.router.lifespan_context(app):
            await asyncio.wait_for(app(scope(), incoming.get, send), 1)
            assert b.client_count == 0
            assert messages[-1] == {
                "type": "websocket.close",
                "code": 1013,
                "reason": "",
            }

    asyncio.run(check())


def test_cancel_received_while_state_initialization_is_stalled():
    async def check():
        release = threading.Event()
        started = threading.Event()
        cancelled = asyncio.Event()

        def state():
            started.set()
            release.wait(2)
            return {"ready": True}

        def cancel(**kwargs):
            cancelled.set()
            return {"accepted": True}

        b = Broadcaster()
        app = create_app(b, state=state, cancel=cancel)
        incoming = asyncio.Queue()
        incoming.put_nowait({"type": "websocket.connect"})

        async def send(message):
            pass

        async with app.router.lifespan_context(app):
            task = asyncio.create_task(app(scope(), incoming.get, send))
            try:
                for _ in range(100):
                    if started.is_set():
                        break
                    await asyncio.sleep(0.005)
                assert started.is_set()
                incoming.put_nowait(
                    {
                        "type": "websocket.receive",
                        "text": json.dumps({"type": "cancel", "turn_id": "a" * 32}),
                    }
                )
                await asyncio.wait_for(cancelled.wait(), 0.2)
            finally:
                release.set()
                incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})
                await asyncio.wait_for(task, 1)

    asyncio.run(check())


def test_body_limits_before_json_and_reserved_control_capacity():
    async def check():
        b = Broadcaster(WebConfig(body_readers=1, body_timeout=0.05))
        app = create_app(
            b,
            cancel=lambda **kw: {"accepted": True},
            submit=lambda *a: {"accepted": True},
        )
        held = asyncio.Event()
        outputs = []

        async def slow_receive():
            held.set()
            await asyncio.Event().wait()

        async def send(message):
            outputs.append(message)

        async with app.router.lifespan_context(app):
            slow = asyncio.create_task(
                app(scope("http", "/api/chat"), slow_receive, send)
            )
            await held.wait()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                assert (
                    await client.post("/api/chat", content=b"x" * 33000)
                ).status_code == 413
                assert (
                    await client.post("/api/chat", json={"text": "busy"})
                ).status_code == 503
                assert (
                    await client.post("/api/cancel", json={"turn_id": "a" * 32})
                ).status_code == 202
            await slow
            assert outputs[0]["status"] == 408
            chunks = iter([b"x" * 20000, b"x" * 20000])

            async def chunked():
                for chunk in chunks:
                    yield chunk

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                assert (
                    await client.post("/api/chat", content=chunked())
                ).status_code == 413

    asyncio.run(check())
