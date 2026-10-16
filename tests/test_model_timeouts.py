# SPDX-License-Identifier: Apache-2.0
"""Slow real sockets must not turn a per-read timeout into an unlimited call."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

import httpx
import numpy as np
import pytest

from app.llm import LLM
from app.request_io import RequestIO, TurnCancelled
from app.stt import STT
from app.tts import LocalTTS


@contextmanager
def provider(kind, mode):
    stop = threading.Event()
    entered = threading.Event()
    healthy = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            if self.path.endswith("/input_tokens"):
                body = b'{"input_tokens":10}'
            elif kind == "llm":
                body = b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
            elif kind == "asr":
                body = b'{"text":"ok"}'
            else:
                body = b"\0\0" * 256
            fault = not self.path.endswith("/input_tokens") and not healthy.is_set()
            if fault:
                entered.set()
                if mode == "before_headers":
                    stop.wait(2)
            try:
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if fault and mode == "trickle":
                    for byte in body:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        if stop.wait(0.03):
                            break
                else:
                    self.wfile.write(body)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", entered, healthy
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("kind", ["llm", "asr", "tts"])
@pytest.mark.parametrize("mode", ["before_headers", "trickle"])
def test_adapter_timeout_releases_local_operation_and_next_call_recovers(kind, mode):
    with provider(kind, mode) as (url, entered, healthy):
        if kind == "llm":
            adapter = LLM(model="assistant-vlm", base_url=url, timeout=0.15)
            adapter._loaded = True

            def call():
                return list(adapter.generate_stream("hello"))
        elif kind == "asr":
            adapter = STT(base_url=url + "/v1", timeout=0.15)
            adapter._client = httpx.Client(timeout=0.15, trust_env=False)

            def call():
                return adapter.transcribe(np.ones(160, dtype=np.int16))
        else:
            adapter = LocalTTS(base_url=url + "/v1", timeout=0.15)
            adapter._client = httpx.Client(timeout=0.15, trust_env=False)

            def call():
                return adapter.synthesize("hello")

        try:
            started = time.monotonic()
            if kind == "llm":
                with pytest.raises(httpx.TimeoutException):
                    call()
            else:
                result = call()
                assert result.get("error") and not result.get("text")
                if kind == "tts":
                    assert result["audio"] is None
            assert entered.is_set() and time.monotonic() - started < 0.8
            assert adapter.requests.idle()
            assert not any(t.name == "model-deadline" for t in threading.enumerate())
            healthy.set()
            result = call()
            if kind == "llm":
                assert result[-1][1]["done"]
            elif kind == "asr":
                assert result["text"] == "ok"
            else:
                assert result["audio"].size == 256
            assert adapter.last_error is None
        finally:
            adapter.unload()


def test_finished_deadline_cannot_abort_a_subsequent_operation():
    io = RequestIO()
    with provider("asr", "trickle") as (url, entered, healthy):
        healthy.set()
        with httpx.Client(timeout=0.1, trust_env=False) as client:
            with io.operation(timeout=0.15) as first:
                assert (
                    client.post(url, extensions=first.extensions).json()["text"] == "ok"
                )
            healthy.clear()
            with io.operation(timeout=1) as second:
                started = time.monotonic()
                assert (
                    client.post(url, extensions=second.extensions).json()["text"]
                    == "ok"
                )
                assert time.monotonic() - started > 0.3
    assert io.idle() and not any(
        t.name == "model-deadline" for t in threading.enumerate()
    )


def test_cancellation_wins_when_deadline_and_stop_are_both_set():
    event = threading.Event()
    io = RequestIO()
    with pytest.raises(TurnCancelled):
        with io.operation(event, timeout=0.01):
            event.set()
            time.sleep(0.02)
    assert io.idle()
