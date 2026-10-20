# SPDX-License-Identifier: Apache-2.0
import queue
import json
import threading
import time
from fastapi.testclient import TestClient
from app.config import Config
from app.web import Broadcaster
from app.web_limits import ClientBuffer, encode
import app.application as entry


class FakeLLM:
    def __init__(self, **kwargs): self.model = kwargs["model"]; self.last_error = None
    def load(self): return True
    def unload(self): pass
    def idle(self): return True
    def cancel(self, event): pass
    def generate_stream(self, prompt, **kwargs):
        yield "测试回答", {}
        yield "", {"done": True, "eval_count": 2}


def test_headless_web_turn_and_shutdown(monkeypatch):
    monkeypatch.setattr(entry, "LLM", FakeLLM)
    app = entry.build_app(Config.load(), no_camera=True, no_audio=True, no_actions=True)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        state = client.get("/api/state").json()
        assert state["ready"] and not state["camera_enabled"] and not state["audio_enabled"]
        assert client.post("/api/chat", json={"text":"看图", "use_image":True}).status_code == 400
        assert client.post("/api/chat", json={"text":"x", "use_image":"yes"}).status_code == 422
        with client.websocket_connect("/ws") as ws:
            response = client.post("/api/chat", json={"text":"你好"})
            assert response.status_code == 202
            messages = []
            while True:
                event = ws.receive_json(); messages.append(event)
                if event["type"] == "done": break
            assert messages[-1]["status"] == "complete" and messages[-1]["text"] == "测试回答"
            assert messages[-1]["turn_id"] == response.json()["turn_id"]
            # Complete the close handshake before TestClient cancels its task
            # scope on __exit__; otherwise its cancellation can race with the
            # ASGI handler's finally block and obscure the cleanup assertion.
            ws.close()
            deadline = time.monotonic() + 2
            while client.get("/api/state").json()["web"]["clients"] and time.monotonic() < deadline:
                time.sleep(0.01)
            assert client.get("/api/state").json()["web"]["clients"] == 0
    assert not any(t.name in {"conversation", "stats-broadcast"} for t in threading.enumerate())


def test_one_active_request_rejects_extra_work(monkeypatch):
    started = threading.Event(); release = threading.Event()
    class SlowLLM(FakeLLM):
        def generate_stream(self, *args, **kwargs):
            started.set(); release.wait(2)
            yield from super().generate_stream(*args, **kwargs)
    monkeypatch.setattr(entry, "LLM", SlowLLM)
    with TestClient(entry.build_app(Config.load(), no_camera=True, no_audio=True, no_actions=True)) as client:
        try:
            assert client.post("/api/chat", json={"text":"first"}).status_code == 202
            assert started.wait(1)
            assert client.post("/api/chat", json={"text":"second"}).status_code == 409
            assert client.get("/api/state").json()["pending"] <= 1
        finally: release.set()


def test_frame_pressure_does_not_silently_discard_response():
    q = ClientBuffer(1,1024)
    q.put(encode({"type":"token", "text":"important"},1024))
    q.put(encode({"type":"frame", "data":"large"},1024))
    assert json.loads(q.items[0].wire)["text"] == "important"
    q.put(encode({"type":"done"},1024))
    assert q.failed and q.items[0].kind == "overflow"


def test_browser_from_another_origin_cannot_read_camera_stream():
    from app.web import create_app
    from starlette.websockets import WebSocketDisconnect
    import pytest
    with TestClient(create_app(Broadcaster())) as client:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect("/ws", headers={"origin": "https://unrelated.example"}): pass
        assert exc.value.code == 1008
