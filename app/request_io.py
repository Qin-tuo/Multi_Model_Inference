# SPDX-License-Identifier: Apache-2.0
"""Cancellation and a total deadline for one owned local HTTP operation.

The trace exposes the connected socket before response headers arrive. Shutting
that socket down interrupts a blocked read without closing unrelated requests.
HTTPX retains ownership of closing the connection itself. A bounded per-operation
timer interrupts slow trickles as well as silent reads; it is joined before the
adapter accepts another operation. Backend release is observed separately.
"""

from contextlib import contextmanager
import socket
import threading
import math
import time
import httpx


class TurnCancelled(RuntimeError):
    pass


class Operation:
    def __init__(self, event, timeout=None):
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Operation timeout must be finite and positive")
        self.event = event
        self._lock = threading.Lock()
        self._socket = None
        self.deadline = time.monotonic() + timeout if timeout is not None else None
        self._timer = None
        self._finished = False

    def start(self):
        if self.deadline is not None:
            timer = threading.Timer(
                max(0, self.deadline - time.monotonic()), self.abort
            )
            timer.name = "model-deadline"
            timer.daemon = True
            timer.start()
            self._timer = timer

    def check(self):
        if self.event.is_set():
            raise TurnCancelled("Turn cancelled")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise httpx.TimeoutException(
                "Model operation exceeded its total time budget"
            )

    def trace(self, name, info):
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
        return {"trace": self.trace}

    def abort(self):
        with self._lock:
            sock = self._socket if not self._finished else None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # The HTTPX owner may already have closed it.

    def finish(self):
        with self._lock:
            self._finished = True
            self._socket = None
        if self._timer is not None:
            self._timer.cancel()
            self._timer.join()


class RequestIO:
    def __init__(self):
        self._lock = threading.Lock()
        self._active = None

    @contextmanager
    def operation(self, event=None, *, timeout=None):
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
        with self._lock:
            op = self._active
            if op is None or op.event is not event:
                return
        op.abort()

    def idle(self):
        with self._lock:
            return self._active is None


def llama_slots_idle(base_url, timeout=0.5):
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        response = client.get(base_url.rstrip("/") + "/slots")
        response.raise_for_status()
        slots = response.json()
    if (
        not isinstance(slots, list)
        or not slots
        or any(type(x.get("is_processing")) is not bool for x in slots)
    ):
        raise ValueError("Invalid native slot observation")
    return all(not slot["is_processing"] for slot in slots)
