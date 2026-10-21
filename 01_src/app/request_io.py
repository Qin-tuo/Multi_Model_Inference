# SPDX-License-Identifier: Apache-2.0
"""本机 HTTP 请求的单次所有权、取消与总时限。

通过连接 trace 获取套接字，取消时中断阻塞读取；HTTPX 仍拥有连接关闭职责，后端是否空闲由另一路观测确认。
"""

from contextlib import contextmanager
import socket
import threading
import math
import time
import json
import httpx


class TurnCancelled(RuntimeError):
    """标识主动轮次取消，与网络失败和超时区分。"""

    pass


class Operation:
    """拥有一次 HTTP 操作的取消事件、连接引用和总时限计时器。"""

    def __init__(self, event, timeout=None):
        """校验总超时并记录单调时钟截止点，初始化连接与结束状态。"""
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Operation timeout must be finite and positive")
        self.event = event
        self._lock = threading.Lock()
        self._socket = None
        self.deadline = time.monotonic() + timeout if timeout is not None else None
        self._timer = None
        self._finished = False

    def start(self):
        """有截止时间时启动守护计时器，到期中断该操作的套接字读取。"""
        if self.deadline is not None:
            timer = threading.Timer(
                max(0, self.deadline - time.monotonic()), self.abort
            )
            timer.name = "model-deadline"
            timer.daemon = True
            timer.start()
            self._timer = timer

    def check(self):
        """先检查主动取消，再检查总时限，分别抛出取消或 HTTP 超时异常。"""
        if self.event.is_set():
            raise TurnCancelled("Turn cancelled")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise httpx.TimeoutException(
                "Model operation exceeded its total time budget"
            )

    def trace(self, name, info):
        """在 TCP 建连完成时登记套接字；若已经取消或超时，立即中断该连接。"""
        if name.endswith("connect_tcp.complete"):
            stream = info.get("return_value")
            with self._lock:
                self._socket = stream.get_extra_info("socket") if stream else None
            if self.event.is_set() or (
                self.deadline is not None and time.monotonic() >= self.deadline
            ):
                self.abort()

    @property
    def extensions(self):
        """提供 HTTPX trace 扩展参数，使请求连接归属于本次操作。"""
        return {"trace": self.trace}

    def abort(self):
        """关闭本次连接的读写方向以唤醒阻塞读取，不接管 HTTPX 的最终 close。"""
        with self._lock:
            sock = self._socket if not self._finished else None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # The HTTPX owner may already have closed it.

    def finish(self):
        """撤销连接引用并取消、等待计时器结束，防止旧定时器影响后续请求。"""
        with self._lock:
            self._finished = True
            self._socket = None
        if self._timer is not None:
            self._timer.cancel()
            self._timer.join()


class RequestIO:
    """限制适配器同一时刻只有一个在途 HTTP 操作。"""

    def __init__(self):
        """初始化单请求互斥锁和当前操作引用。"""
        self._lock = threading.Lock()
        self._active = None

    @contextmanager
    def operation(self, event=None, *, timeout=None):
        """在上下文中登记唯一操作，执行前后检查取消/超时，退出时清理计时器并释放槽位。"""
        op = Operation(event if event is not None else threading.Event(), timeout)
        with self._lock:
            if self._active is not None:
                raise RuntimeError("Overlapping request on single-flight adapter")
            self._active = op
        try:
            op.start()
            op.check()
            yield op
            op.check()
        except httpx.HTTPError:
            op.check()
            raise
        finally:
            op.finish()
            with self._lock:
                if self._active is op:
                    self._active = None

    def cancel(self, event):
        """仅中断取消事件对象与当前操作一致的连接，避免旧轮次取消影响新请求。"""
        with self._lock:
            op = self._active
            if op is None or op.event is not event:
                return
        op.abort()

    def idle(self):
        """在锁内判断适配器是否已经没有在途 HTTP 操作。"""
        with self._lock:
            return self._active is None


def control_json(url, timeout, max_bytes=65536):
    """以独立连接读取小型控制接口 JSON，同时限制总时间和响应字节数。

    既检查声明长度也检查实际累计数据，避免缓慢分块或未声明长度绕过上限。
    """
    with (
        RequestIO().operation(timeout=timeout) as operation,
        httpx.Client(
            timeout=timeout,
            trust_env=False,
        ) as client,
    ):
        data = bytearray()
        with client.stream(
            "GET",
            url,
            extensions=operation.extensions,
            headers={"Accept-Encoding": "identity"},
        ) as response:
            response.raise_for_status()
            if int(response.headers.get("content-length", "0")) > max_bytes:
                raise ValueError("Control response exceeds byte budget")
            for chunk in response.iter_bytes(8192):
                operation.check()
                if len(data) + len(chunk) > max_bytes:
                    raise ValueError("Control response exceeds byte budget")
                data.extend(chunk)
        return json.loads(data)


def llama_slots_idle(base_url, timeout=0.5):
    """校验 llama.cpp 原生槽位响应结构，仅在全部槽位未处理请求时返回 True。"""
    slots = control_json(base_url.rstrip("/") + "/slots", timeout)
    if (
        not isinstance(slots, list)
        or not slots
        or any(
            not isinstance(x, dict) or type(x.get("is_processing")) is not bool
            for x in slots
        )
    ):
        raise ValueError("Invalid native slot observation")
    return all(not slot["is_processing"] for slot in slots)
