# SPDX-License-Identifier: Apache-2.0
"""Control responses must be bounded and describe the expected backend."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from app.config import Config
from app.llm import LLM
from app.request_io import control_json, llama_slots_idle
from app.stt import STT
from app.tts import LocalTTS
from app.turn_control import TurnController


@contextmanager
def controls():
    mode = {"kind": "healthy"}
    stop = threading.Event()
    entered = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            kind = mode["kind"]
            entered.set()
            if self.path == "/slots":
                value = [{"is_processing": False}]
            elif self.path == "/v1/models":
                value = {
                    "data": [
                        {"id": "wrong" if kind == "wrong_model" else "assistant-vlm"}
                    ]
                }
            elif self.path == "/props":
                value = {
                    "chat_template_caps": {
                        "supports_tools": True,
                        "supports_tool_calls": True,
                    }
                }
            else:
                value = {
                    role: {
                        "ready": True,
                        "model_id": model,
                        "backend": backend,
                        "audio": {
                            "sample_rate": 16000,
                            "channels": 1,
                            "sample_format": "pcm_s16le",
                            "response_formats": ["pcm"],
                        },
                        "concurrency": {"active": 0},
                        "voices": {"items": [{"payload": "0"}]},
                    }
                    for role, model, backend in [
                        ("asr", "qwen3-asr", "llama_qwen3_asr"),
                        ("tts", "matcha-icefall-zh-en", "matcha_trt"),
                    ]
                }
            body = json.dumps(value).encode()
            if kind.startswith("oversize"):
                body = b" " * 70000 + body
            if kind == "headers":
                stop.wait(2)
            try:
                self.send_response(200)
                if kind != "oversize_unannounced":
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if kind == "trickle":
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
        yield f"http://127.0.0.1:{server.server_port}", mode, entered
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "mode", ["headers", "trickle", "oversize_declared", "oversize_unannounced"]
)
def test_control_get_is_bounded_before_parsing_and_recovers(mode):
    with controls() as (url, setting, entered):
        # Establish the listener/client before measuring the injected fault.
        # Otherwise cold client initialization can consume the entire 150 ms
        # budget before this test ever reaches its delayed-header handler.
        assert control_json(url + "/slots", 2) == [{"is_processing": False}]
        entered.clear()
        setting["kind"] = mode
        started = time.monotonic()
        with pytest.raises((httpx.TimeoutException, ValueError)):
            control_json(url + "/slots", 0.15)
        assert entered.is_set() and time.monotonic() - started < 0.8
        assert not any(t.name == "model-deadline" for t in threading.enumerate())
        setting["kind"] = "healthy"
        assert llama_slots_idle(url)


def test_http_200_wrong_model_fails_health_and_recovery_clears_error():
    with controls() as (url, setting, _):
        model = LLM(model="assistant-vlm", base_url=url, timeout=0.15)
        assert model.load() and model.health_check()
        setting["kind"] = "wrong_model"
        assert not model.health_check()
        assert "Expected model" in model.last_error and model.model == "assistant-vlm"
        setting["kind"] = "healthy"
        assert model.health_check() and model.last_error is None
        model.require_tools()


@pytest.mark.parametrize("kind", ["llm", "asr", "tts"])
def test_startup_and_health_use_bounded_control_get(kind):
    with controls() as (url, setting, _):
        if kind == "llm":
            adapter = LLM(model="assistant-vlm", base_url=url, timeout=0.15)
        elif kind == "asr":
            adapter = STT(base_url=url + "/v1", native_base_url=url, timeout=0.15)
        else:
            adapter = LocalTTS(base_url=url + "/v1", timeout=0.15)
        try:
            setting["kind"] = "trickle"
            started = time.monotonic()
            assert not adapter.load() and time.monotonic() - started < 0.8
            setting["kind"] = "healthy"
            assert adapter.load() and adapter.health_check()
            setting["kind"] = "trickle"
            started = time.monotonic()
            assert not adapter.health_check() and time.monotonic() - started < 0.8
            setting["kind"] = "healthy"
            assert adapter.health_check() and adapter.idle()
        finally:
            adapter.unload()


def test_bad_release_probe_cannot_hold_worker_forever_or_fake_release():
    with controls() as (url, setting, entered):

        class Model:
            def generate_stream(self, *a, **kw):
                yield "ok", {"done": True}

            def idle(self):
                return llama_slots_idle(url, 0.15)

            def cancel(self, event):
                pass

        setting["kind"] = "trickle"
        control = TurnController(
            Config.load(), Model(), SimpleNamespace(send=lambda *a, **kw: None)
        )
        control.start()
        try:
            key = control.submit("test")["turn_id"]
            assert entered.wait(1)
            end = time.monotonic() + 1
            while not control.state()["release"] and time.monotonic() < end:
                time.sleep(0.01)
            value = control.state()
            assert value["active_turn"] == key and not value["release"]["confirmed"]
            assert "error" in value["release"]["observations"]["llm"]
            with pytest.raises(RuntimeError):
                control.submit("too early")
            setting["kind"] = "healthy"
            end = time.monotonic() + 2
            while control.state()["active_turn"] and time.monotonic() < end:
                time.sleep(0.01)
            assert control.state()["active_turn"] is None
            assert control.state()["last_result"]["resources_released"]
        finally:
            setting["kind"] = "healthy"
            control.close()
