# SPDX-License-Identifier: Apache-2.0
"""Web 传输和同步调用的资源边界：消息编码、客户端队列、工作槽位与请求体时限。"""

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
        "diagnostics",
        "perception",
        "scene",
        "gesture_state",
        "actions_state",
        "camera_state",
        "voice_state",
        "audio_state",
    }
)


@dataclass(frozen=True)
class Message:
    """已编码消息及其取消、过期、事件序号和轮次归属信息。"""

    wire: bytes
    kind: str
    cancel_event: object = None
    sequence: int = 0
    expires: float | None = None
    turn_id: str | None = None

    @property
    def stale(self):
        """检查消息绑定的取消事件或绝对过期时刻，决定是否应从队列丢弃。"""
        return (self.cancel_event is not None and self.cancel_event.is_set()) or (
            self.expires is not None and time.monotonic() > self.expires
        )


def encode(message, limit, cancel_event=None):
    """按 UTF-8 实际字节数分块编码合法消息，拒绝 NaN 和超预算载荷，再附加轮次及取消身份。"""
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
    """生成小型可靠溢出通知，告知客户端应重新获取状态。"""
    return encode({"type": "overflow", "error": reason}, 4096)


class ClientBuffer:
    """由事件循环拥有的客户端缓冲，易失状态可覆盖旧值，可靠事件溢出需要重同步。"""

    def __init__(self, events, byte_limit):
        """初始化条数/字节预算、事件等待器、快照游标和未完成流的身份。"""
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
        """判断消息是否已被初始快照覆盖；未完成轮次的文本和工具回执仍需保留。"""
        return 0 < item.sequence <= self.cursor and not (
            self.streaming_turn
            and item.turn_id == self.streaming_turn
            and item.kind in {"token", "transcript", "action_receipt", "error"}
        )

    def _remove(self, item):
        """移除指定队列项并同步扣减实际编码字节数。"""
        self.items.remove(item)
        self.bytes -= len(item.wire)

    def put(self, item):
        """接纳未失效、未被快照覆盖的消息，合并同类易失状态并优先淘汰易失项。

        可靠事件仍无法放入时转入失败状态，而不假装完整保留对话。
        """
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
        """清空普通待发内容，仅保留一次溢出通知并唤醒发送协程。"""
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
        """把初始快照放在尚未被其覆盖的增量前面，再标记连接初始化完成。"""
        self.cursor = cursor
        self.streaming_turn = streaming_turn
        if not self.failed and not self.closed:
            # 初始快照覆盖生命周期和已完成答案；在途文本及已提交工具回执
            # 仍需接上增量，不能因游标较旧就全部丢弃。
            pending = [item for item in self.items if not self.covered(item)]
            self.items.clear()
            self.bytes = 0
            for item in [*messages, *pending]:
                self.put(item)
        self.initialized.set()

    async def get(self):
        """等待初始化和可发送内容，跳过过期消息；队列关闭后以 EOFError 结束消费者。"""
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
        """清空客户端缓存并唤醒所有等待器，使挂起发送者能够结束。"""
        self.closed = True
        self.items.clear()
        self.bytes = 0
        self.ready.set()
        self.initialized.set()


class BusyRead(RuntimeError):
    """表示有界读取/控制工作槽位已满，调用方应返回忙状态。"""

    pass


class BoundedCalls:
    """持续持有同步任务的 Future，直到任务真正结束，HTTP 等待超时不代表线程已经释放。"""

    def __init__(self, workers):
        """创建固定大小线程池、在途 Future 集合和控制操作单飞状态。"""
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
        """在容量允许时提交同步调用，控制类操作额外限制为单个在途任务。"""
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
            """任务实际结束后读取异常并释放其槽位，避免 HTTP 超时提前丢失资源所有权。"""
            f.exception() if not f.cancelled() else None
            with self.lock:
                self.pending.discard(f)
                self.completed += 1
                if self.control is f:
                    self.control = None

        future.add_done_callback(finished)
        return future

    async def run(self, fn, timeout, wait_for_slot=False):
        """在总截止时间内申请槽位并异步等待同步任务，可选短暂等待空位。

        shield 保留仍在执行的底层任务，HTTP 等待超时后仍由 Future 完成回调负责释放槽位。
        """
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
            # HTTP 等待者离开后仍取走后台异常，避免遗失 Future 的完成结果。
            lambda f: f.exception() if not f.cancelled() else None
        )
        return await asyncio.wait_for(
            asyncio.shield(wrapped), max(0.001, deadline - time.monotonic())
        )

    def snapshot(self):
        """在锁内返回在途数、容量及完成/拒绝计数。"""
        with self.lock:
            return {
                "in_flight": len(self.pending),
                "limit": self.limit,
                "rejected": self.rejected,
                "completed": self.completed,
            }

    async def close(self):
        """禁止新任务并取消未启动工作，等待仍拥有的 Future 完成；超时则明确报告线程未释放。"""
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
    """在 FastAPI 解析前限制 POST 请求体大小、读取数量和总读取时间。"""

    def __init__(self, app, config, audio_limit):
        """绑定下游 ASGI 应用，分别初始化普通请求与取消/清会话控制请求的读取计数。"""
        self.app, self.config, self.audio_limit = app, config, audio_limit
        self.active = {"data": 0, "control": 0}

    async def __call__(self, scope, receive, send):
        """按 POST 路径设置读取槽位、字节上限和总时限，再用受限 receive 包装下游处理。

        先检查 Content-Length，随后仍检查实际分块字节，控制请求使用独立容量以免被大上传完全占满。
        """
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
            """发送带关闭连接标记的 JSON 错误，拒绝无效、过大或超出并发预算的请求。"""
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
            """在同一总截止时间内读取下一块请求体，累加真实字节并在超时或超限时中断。"""
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
