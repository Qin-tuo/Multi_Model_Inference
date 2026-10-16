# SPDX-License-Identifier: Apache-2.0
"""Cancellation of one owned local HTTP operation using HTTPX/httpcore tracing.

The trace exposes the connected socket before response headers arrive. Shutting
that socket down interrupts a blocked read without closing unrelated requests.
HTTPX retains ownership of closing the connection itself.
"""
from contextlib import contextmanager
import socket
import threading
import httpx


class TurnCancelled(RuntimeError):
    pass


class Operation:
    def __init__(self, event):
        self.event = event
        self._lock = threading.Lock()
        self._socket = None

    def check(self):
        if self.event.is_set(): raise TurnCancelled('Turn cancelled')

    def trace(self, name, info):
        if name.endswith('connect_tcp.complete'):
            stream = info.get('return_value')
            with self._lock:
                self._socket = stream.get_extra_info('socket') if stream else None
            if self.event.is_set(): self.abort()

    @property
    def extensions(self):
        return {'trace':self.trace}

    def abort(self):
        with self._lock: sock = self._socket
        if sock is not None:
            try: sock.shutdown(socket.SHUT_RDWR)
            except OSError: pass  # The HTTPX owner may already have closed it.


class RequestIO:
    def __init__(self):
        self._lock = threading.Lock()
        self._active = None

    @contextmanager
    def operation(self, event=None):
        op = Operation(event if event is not None else threading.Event())
        with self._lock:
            if self._active is not None: raise RuntimeError('Overlapping request on single-flight adapter')
            self._active = op
        try:
            op.check()
            yield op
            op.check()
        except httpx.HTTPError as exc:
            if op.event.is_set(): raise TurnCancelled('Turn cancelled') from exc
            raise
        finally:
            with self._lock:
                if self._active is op: self._active = None

    def cancel(self, event):
        with self._lock:
            op = self._active
            if op is None or op.event is not event: return
        op.abort()

    def idle(self):
        with self._lock: return self._active is None


def llama_slots_idle(base_url, timeout=.5):
    with httpx.Client(timeout=timeout,trust_env=False) as client:
        response=client.get(base_url.rstrip('/')+'/slots');response.raise_for_status();slots=response.json()
    if not isinstance(slots,list) or not slots or any(type(x.get('is_processing')) is not bool for x in slots):
        raise ValueError('Invalid native slot observation')
    return all(not slot['is_processing'] for slot in slots)
