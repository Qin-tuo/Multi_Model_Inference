# SPDX-License-Identifier: Apache-2.0
"""Small transport bounds around Starlette/asyncio, not a conversation runtime."""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import threading
import time
from fastapi import HTTPException
from starlette.responses import JSONResponse


VOLATILE = frozenset(
    {
        "frame",
        "stats",
        "perception",
        "scene",
        "gesture_state",
        "actions_state",
        "camera_state",
    }
)


@dataclass(frozen=True)
class Message:
    wire: bytes
    kind: str
    cancel_event: object = None
    sequence: int = 0
    expires: float | None = None
    turn_id: str | None = None

    @property
    def stale(self):
        return (self.cancel_event is not None and self.cancel_event.is_set()) or (
            self.expires is not None and time.monotonic() > self.expires
        )


def encode(message, limit, cancel_event=None):
    if (
        not isinstance(message, dict)
        or not isinstance(message.get("type"), str)
        or not 1 <= len(message["type"]) <= 64
    ):
        raise ValueError("Invalid Web message type")
    data = bytearray()
    for chunk in json.JSONEncoder(
        ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).iterencode(message):
        if len(chunk) > limit - len(data):
            raise ValueError("Message exceeds Web byte budget")
        raw = chunk.encode("utf-8")
        if len(data) + len(raw) > limit:
            raise ValueError("Message exceeds Web byte budget")
        data.extend(raw)
    return Message(
        bytes(data),
        message.get("type", ""),
        cancel_event,
        turn_id=message.get("turn_id"),
    )


def overflow(reason):
    return encode({"type": "overflow", "error": reason}, 4096)


class ClientBuffer:
    """Loop-owned event/byte budget; volatile values replace their older value."""

    def __init__(self, events, byte_limit):
        self.event_limit, self.byte_limit = events, byte_limit
        self.items = deque()
        self.bytes = 0
        self.closed = False
        self.failed = False
        self.ready = asyncio.Event()
        self.initialized = asyncio.Event()
        self.dropped = 0
        self.cursor = 0
        self.streaming_turn = None

    def covered(self, item):
        return 0 < item.sequence <= self.cursor and not (
            self.streaming_turn
            and item.turn_id == self.streaming_turn
            and item.kind in {"token", "transcript", "action_receipt", "error"}
        )

    def _remove(self, item):
        self.items.remove(item)
        self.bytes -= len(item.wire)

    def put(self, item):
        if self.closed or self.failed or item.stale or self.covered(item):
            return
        for previous in list(self.items):
            if previous.stale or (item.kind in VOLATILE and previous.kind == item.kind):
                self._remove(previous)
        while (
            len(self.items) >= self.event_limit
            or self.bytes + len(item.wire) > self.byte_limit
        ):
            previous = next((m for m in self.items if m.kind in VOLATILE), None)
            if previous is not None:
                self._remove(previous)
                self.dropped += 1
                continue
            if item.kind in VOLATILE:
                self.dropped += 1
                return
            self.fail("Viewer fell behind; reconnect to refresh authoritative state")
            return
        self.items.append(item)
        self.bytes += len(item.wire)
        self.ready.set()

    def fail(self, reason):
        if self.closed or self.failed:
            return
        self.failed = True
        self.items.clear()
        self.bytes = 0
        item = overflow(reason)
        self.items.append(item)
        self.bytes = len(item.wire)
        self.ready.set()

    def prime(self, messages, cursor, streaming_turn=None):
        self.cursor = cursor
        self.streaming_turn = streaming_turn
        if not self.failed and not self.closed:
            # A state snapshot covers lifecycle and completed answers, but not
            # an unfinished answer's text or committed action receipts.
            pending = [item for item in self.items if not self.covered(item)]
            self.items.clear()
            self.bytes = 0
            for item in [*messages, *pending]:
                self.put(item)
        self.initialized.set()

    async def get(self):
        await self.initialized.wait()
        while True:
            await self.ready.wait()
            if self.items:
                item = self.items.popleft()
                self.bytes -= len(item.wire)
                if not self.items:
                    self.ready.clear()
                if not item.stale:
                    return item
            elif self.closed:
                raise EOFError("Client buffer closed")
            else:
                self.ready.clear()

    def close(self):
        self.closed = True
        self.items.clear()
        self.bytes = 0
        self.ready.set()
        self.initialized.set()


class BusyRead(RuntimeError):
    pass


class BoundedCalls:
    """Keep native futures owned until work ends, even if its HTTP waiter leaves."""

    def __init__(self, workers):
        self.limit = workers
        self.pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="web-read"
        )
        self.pending = set()
        self.lock = threading.Lock()
        self.closed = False
        self.control = None
        self.rejected = 0
        self.completed = 0

    def submit(self, fn, *args, control=False):
        with self.lock:
            if (
                self.closed
                or len(self.pending) >= self.limit
                or (control and self.control is not None)
            ):
                self.rejected += 1
                raise BusyRead("Web read/control workers are occupied")
            future = self.pool.submit(fn, *args)
            self.pending.add(future)
            if control:
                self.control = future

        def finished(f):
            f.exception() if not f.cancelled() else None
            with self.lock:
                self.pending.discard(f)
                self.completed += 1
                if self.control is f:
                    self.control = None

        future.add_done_callback(finished)
        return future

    async def run(self, fn, timeout, wait_for_slot=False):
        deadline = time.monotonic() + timeout
        while True:
            try:
                future = self.submit(fn)
                break
            except BusyRead:
                if not wait_for_slot or time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(0.025)
        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(
            lambda f: f.exception() if not f.cancelled() else None
        )
        return await asyncio.wait_for(
            asyncio.shield(wrapped), max(0.001, deadline - time.monotonic())
        )

    def snapshot(self):
        with self.lock:
            return {
                "in_flight": len(self.pending),
                "limit": self.limit,
                "rejected": self.rejected,
                "completed": self.completed,
            }

    async def close(self):
        with self.lock:
            self.closed = True
            pending = list(self.pending)
        self.pool.shutdown(wait=False, cancel_futures=True)
        if pending:
            owned = asyncio.gather(
                *(asyncio.wrap_future(f) for f in pending), return_exceptions=True
            )
            try:
                await asyncio.wait_for(asyncio.shield(owned), 5)
            except asyncio.TimeoutError:
                raise RuntimeError(
                    "Web read workers have not released; no cleanup success claimed"
                ) from None


class InputLimits:
    """Bound request bytes before FastAPI JSON parsing, plus slow-body lifetime."""

    def __init__(self, app, config, audio_limit):
        self.app, self.config, self.audio_limit = app, config, audio_limit
        self.active = {"data": 0, "control": 0}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        path = scope["path"]
        control = path in {"/api/cancel", "/api/session/clear"}
        group = "control" if control else "data"
        capacity = 2 if control else self.config.body_readers
        limit = (
            self.audio_limit
            if path == "/api/audio"
            else (4096 if control else self.config.json_body_bytes)
        )
        headers = dict(scope.get("headers", []))

        async def reject(status, detail):
            await JSONResponse(
                {"detail": detail}, status_code=status, headers={"Connection": "close"}
            )(scope, receive, send)

        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await reject(400, "Invalid Content-Length")
        if length < 0:
            return await reject(400, "Invalid Content-Length")
        if length > limit:
            return await reject(413, "Request body exceeds byte budget")
        if self.active[group] >= capacity:
            return await reject(503, "Request body readers are occupied")
        self.active[group] += 1
        seen = 0
        deadline = time.monotonic() + self.config.body_timeout

        async def bounded_receive():
            nonlocal seen
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HTTPException(
                    408, "Request body timed out", headers={"Connection": "close"}
                )
            try:
                message = await asyncio.wait_for(receive(), remaining)
            except asyncio.TimeoutError:
                raise HTTPException(
                    408, "Request body timed out", headers={"Connection": "close"}
                ) from None
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    raise HTTPException(
                        413,
                        "Request body exceeds byte budget",
                        headers={"Connection": "close"},
                    )
            return message

        try:
            await self.app(scope, bounded_receive, send)
        finally:
            self.active[group] -= 1
