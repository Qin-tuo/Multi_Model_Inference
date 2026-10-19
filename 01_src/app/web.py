# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Web UI — lightweight FastAPI server with WebSocket broadcasting.

Provides a Broadcaster class for thread-safe message delivery from the
pipeline thread to all connected browser clients, and a FastAPI app
that serves the single-page frontend and a WebSocket endpoint.
"""

import asyncio
import json
import threading
import math
from collections import deque
from dataclasses import replace
from app.config import WebConfig
from app.web_limits import (
    ClientBuffer,
    BoundedCalls,
    BusyRead,
    InputLimits,
    VOLATILE,
    encode,
    overflow,
)
from pathlib import Path
from project_paths import WEB_DIR
from typing import Callable, Optional
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from pydantic import BaseModel, Field, StrictBool
from fastapi.responses import FileResponse, HTMLResponse

STATIC_DIR = WEB_DIR


class Broadcaster:
    """Thread-safe fan-out from the synchronous pipeline to async WebSocket clients.

    A byte/event-bounded staging buffer owns one scheduled drain. Each viewer
    has a bounded loop-owned buffer; reliable overflow forces resynchronization.

    Also holds shared push-to-talk (PTT) state: set = unmuted / listening
    by default, cleared = muted. Any client can toggle via WebSocket.
    """

    def __init__(self, config=None, frame_ttl=1.0):
        self.config = config or WebConfig()
        self.frame_ttl = frame_ttl
        self._clients = {}
        self._pending = deque()
        self._pending_bytes = 0
        self._scheduled = False
        self._epoch = 0
        self._sequence = 0
        self._overflow = False
        self._coalesced = self._dropped = self._overflows = self._rejected = 0
        self._peak_events = self._peak_bytes = 0
        self._speaker_cached = {
            "type": "speaker_state",
            "speakers": [],
            "selected": None,
        }
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ptt = threading.Event()
        self._ptt.clear()
        self._input_setter = None
        self._speaker_getter: Optional[Callable[[], dict]] = None
        self._speaker_setter: Optional[Callable[[str], dict]] = None
        self._speaker_volume_setter: Optional[Callable[[int], dict]] = None

    def set_loop(self, loop):
        with self._lock:
            self._epoch += 1
            self._loop = loop
            self._pending.clear()
            self._pending_bytes = 0
            self._scheduled = False
            self._overflow = False
            for client in self._clients:
                client.close()
            self._clients.clear()

    def register(self, client):
        # Reservation precedes the awaited WebSocket handshake.
        with self._lock:
            if len(self._clients) >= self.config.max_clients:
                return False
            self._clients[client] = self._sequence
            return True

    def unregister(self, client):
        with self._lock:
            self._clients.pop(client, None)
            if not self._clients:
                self._pending.clear()
                self._pending_bytes = 0
                self._overflow = False
        client.close()

    @property
    def cursor(self):
        with self._lock:
            return self._sequence

    def snapshot(self):
        with self._lock:
            return {
                "clients": len(self._clients),
                "client_limit": self.config.max_clients,
                "bridge_events": len(self._pending),
                "bridge_bytes": self._pending_bytes,
                "drain_scheduled": self._scheduled,
                "bridge_peak_events": self._peak_events,
                "bridge_peak_bytes": self._peak_bytes,
                "coalesced": self._coalesced,
                "dropped": self._dropped,
                "overflows": self._overflows,
                "rejected_messages": self._rejected,
            }

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    @property
    def ptt_active(self) -> bool:
        return self._ptt.is_set()

    def set_ptt(self, active: bool):
        if self._input_setter:
            try:
                self._input_setter(active)
            except ValueError:
                self._ptt.clear()
                self.send({"type": "ptt_state", "active": False})
                raise
        if active:
            self._ptt.set()
        else:
            self._ptt.clear()
        self.send({"type": "ptt_state", "active": active})

    def configure_speakers(
        self,
        getter: Callable[[], dict],
        setter: Callable[[str], dict],
        volume_setter: Optional[Callable[[int], dict]] = None,
    ):
        """Attach live speaker state, selection, and volume callbacks."""
        self._speaker_getter = getter
        self._speaker_setter = setter
        self._speaker_volume_setter = volume_setter

    def get_speaker_state(self) -> dict:
        if not self._speaker_getter:
            return {"type": "speaker_state", "speakers": [], "selected": None}
        try:
            self._speaker_cached = {"type": "speaker_state", **self._speaker_getter()}
            return dict(self._speaker_cached)
        except Exception as exc:
            return {
                "type": "speaker_state",
                "speakers": [],
                "selected": None,
                "error": str(exc),
            }

    def select_speaker(self, sink_id: str):
        if not self._speaker_setter:
            state = {
                "speakers": [],
                "selected": None,
                "error": "Speaker routing is not initialized",
            }
        else:
            try:
                state = self._speaker_setter(sink_id)
            except Exception as exc:
                state = {**self.get_speaker_state(), "error": str(exc)}
                state.pop("type", None)
        self._speaker_cached = {"type": "speaker_state", **state}
        self.send(self._speaker_cached)

    def set_speaker_volume(self, volume: int):
        if not self._speaker_volume_setter:
            state = {
                **self.get_speaker_state(),
                "error": "Speaker volume control is not initialized",
            }
            state.pop("type", None)
        else:
            try:
                state = self._speaker_volume_setter(volume)
            except Exception as exc:
                state = {**self.get_speaker_state(), "error": str(exc)}
                state.pop("type", None)
        self._speaker_cached = {"type": "speaker_state", **state}
        self.send(self._speaker_cached)

    def cached_speaker_state(self):
        return dict(self._speaker_cached)

    def _alarm(self, reason):
        self._pending.clear()
        self._pending_bytes = 0
        self._overflow = True
        self._overflows += 1
        self._sequence += 1
        message = replace(overflow(reason), sequence=self._sequence)
        self._pending.append(message)
        self._pending_bytes = len(message.wire)

    def send(self, msg, *, cancel_event=None):
        with self._lock:
            loop, epoch = self._loop, self._epoch
            if loop is None or not self._clients:
                return False
        try:
            message = encode(msg, self.config.message_bytes, cancel_event)
            frame = msg.get("frame")
            if (
                message.kind in VOLATILE
                and isinstance(frame, dict)
                and isinstance(frame.get("received_monotonic"), (int, float))
            ):
                message = replace(
                    message, expires=frame["received_monotonic"] + self.frame_ttl
                )
        except (TypeError, ValueError):
            with self._lock:
                if self._epoch != epoch:
                    return False
                self._rejected += 1
                if (
                    isinstance(msg, dict)
                    and isinstance(msg.get("type"), str)
                    and msg["type"] in VOLATILE
                ):
                    return False
                self._alarm(
                    "Output exceeds its transport contract; reconnect to inspect state"
                )
                self._schedule(loop, epoch)
            return False
        with self._lock:
            if self._epoch != epoch or not self._clients or message.stale:
                return False
            if self._overflow:
                self._dropped += 1
                return False
            self._sequence += 1
            message = replace(message, sequence=self._sequence)
            for old in list(self._pending):
                if old.stale or (message.kind in VOLATILE and old.kind == message.kind):
                    self._pending.remove(old)
                    self._pending_bytes -= len(old.wire)
                    self._coalesced += 1
            while (
                len(self._pending) >= self.config.bridge_events
                or self._pending_bytes + len(message.wire) > self.config.bridge_bytes
            ):
                old = next(
                    (item for item in self._pending if item.kind in VOLATILE), None
                )
                if old is not None:
                    self._pending.remove(old)
                    self._pending_bytes -= len(old.wire)
                    self._dropped += 1
                elif message.kind in VOLATILE:
                    self._dropped += 1
                    return False
                else:
                    self._alarm(
                        "Output bridge overflow; reconnect to inspect authoritative state"
                    )
                    self._schedule(loop, epoch)
                    return False
            self._pending.append(message)
            self._pending_bytes += len(message.wire)
            self._peak_events = max(self._peak_events, len(self._pending))
            self._peak_bytes = max(self._peak_bytes, self._pending_bytes)
            self._schedule(loop, epoch)
            return True

    def _schedule(self, loop, epoch):
        if self._scheduled:
            return
        self._scheduled = True
        try:
            loop.call_soon_threadsafe(self._drain, epoch)
        except RuntimeError:
            self._scheduled = False
            self._pending.clear()
            self._pending_bytes = 0

    def _drain(self, epoch):
        with self._lock:
            if self._epoch != epoch:
                return
            messages = []
            for _ in range(min(16, len(self._pending))):
                item = self._pending.popleft()
                self._pending_bytes -= len(item.wire)
                messages.append(item)
            clients = list(self._clients.items())
            if self._pending:
                self._loop.call_soon(self._drain, epoch)
            else:
                self._scheduled = False
                self._overflow = False
        for message in messages:
            if message.stale:
                continue
            for client, cursor in clients:
                if message.sequence > cursor:
                    if message.kind == "overflow":
                        client.fail(
                            "Output bridge overflow; reconnect to inspect state"
                        )
                    else:
                        client.put(message)


class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    use_image: StrictBool = False
    request_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


class CancelRequest(BaseModel):
    turn_id: str = Field(pattern=r"^[0-9a-f]{32}$")


class ClearSessionRequest(BaseModel):
    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")


class VisionIdentityRequest(BaseModel):
    target_id: str = Field(min_length=1, max_length=160)
    label: str = Field(min_length=1, max_length=64)


class VisionTargetRequest(BaseModel):
    track_id: str | None = Field(default=None, max_length=160)


class VisionRemoveIdentityRequest(BaseModel):
    label: str = Field(min_length=1, max_length=64)


def create_app(
    broadcaster: Broadcaster,
    *,
    submit=None,
    state=None,
    lifespan=None,
    submit_audio=None,
    audio_file=None,
    cancel=None,
    clear_history=None,
    vision=None,
    max_upload_bytes=484096,
) -> FastAPI:
    config = broadcaster.config

    @asynccontextmanager
    async def application_lifespan(app):
        broadcaster.set_loop(asyncio.get_running_loop())
        app.state.calls = BoundedCalls(config.read_workers)
        try:
            if lifespan:
                async with lifespan(app):
                    try:
                        yield
                    finally:
                        await app.state.calls.close()
            else:
                try:
                    yield
                finally:
                    await app.state.calls.close()
        finally:
            broadcaster.set_loop(None)
            await app.state.calls.close()

    app = FastAPI(title="Orin NX Companion", lifespan=application_lifespan)
    app.add_middleware(InputLimits, config=config, audio_limit=max_upload_bytes)

    async def read_state(*, wait_for_slot=False):
        def read():
            return state() if state else {"ready": False}

        try:
            value = await app.state.calls.run(read, config.read_timeout, wait_for_slot)
        except (BusyRead, asyncio.TimeoutError) as exc:
            raise HTTPException(503, "State reader occupied or timed out") from exc
        return {**value, "web_reads": app.state.calls.snapshot()}

    @app.get("/api/state")
    async def _state():
        return await read_state()

    async def vision_call(name, *args):
        if vision is None or name not in vision:
            raise HTTPException(503, "B1 vision operation unavailable")
        try:
            return await app.state.calls.run(
                lambda: vision[name](*args), config.read_timeout, True
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (BusyRead, asyncio.TimeoutError) as exc:
            raise HTTPException(503, "Vision operation occupied or timed out") from exc

    @app.get("/api/vision/identities")
    async def _vision_identities():
        return await vision_call("identities")

    @app.post("/api/vision/identities")
    async def _vision_enroll(request: VisionIdentityRequest):
        return await vision_call("enroll", request.target_id, request.label)

    @app.delete("/api/vision/identities")
    async def _vision_remove(request: VisionRemoveIdentityRequest):
        return await vision_call("remove", request.label)

    @app.post("/api/vision/target")
    async def _vision_target(request: VisionTargetRequest):
        return await vision_call("target", request.track_id)

    @app.get("/api/debug")
    async def _debug_state():
        value = await read_state()
        return value.get("diagnostics", {"ready": False})

    @app.get("/api/debug/baseline")
    async def _debug_baseline():
        return FileResponse(
            STATIC_DIR.parent / "benchmarks/results/b0-vision-summary.json",
            media_type="application/json",
        )

    @app.get("/static/{asset}")
    async def _debug_asset(asset: str):
        if asset not in {"app.js", "app.css", "debug.js", "debug.css"}:
            raise HTTPException(404, "Asset not found")
        return FileResponse(STATIC_DIR / asset, headers={"Cache-Control": "no-cache"})

    @app.post("/api/chat", status_code=202)
    async def _chat(request: ChatRequest):
        if submit is None:
            raise HTTPException(503, "Conversation input unavailable")
        try:
            return (
                submit(request.text, request.use_image, request_id=request.request_id)
                if request.request_id
                else submit(request.text, request.use_image)
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/audio", status_code=202)
    async def _audio(
        request: Request, use_image: bool = False, request_id: str | None = None
    ):
        if submit_audio is None:
            raise HTTPException(503, "Speech input unavailable")
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > max_upload_bytes:
                raise HTTPException(413, "Audio input exceeds size limit")
        try:
            return (
                submit_audio(bytes(data), use_image, request_id=request_id)
                if request_id
                else submit_audio(bytes(data), use_image)
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/cancel", status_code=202)
    async def _cancel(request: CancelRequest):
        if cancel is None:
            raise HTTPException(503, "Cancellation unavailable")
        return cancel(turn_id=request.turn_id)

    @app.post("/api/session/clear")
    async def _clear_session(request: ClearSessionRequest):
        if clear_history is None:
            raise HTTPException(503, "Session control unavailable")
        try:
            return clear_history(request.session_id)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/audio/{turn_id}")
    async def _download_audio(turn_id: str):
        path = audio_file(turn_id) if audio_file else None
        if path is None:
            raise HTTPException(404, "Audio output is unavailable")
        return FileResponse(
            path,
            media_type="audio/wav",
            filename="reply.wav",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/")
    @app.get("/debug")
    async def _index():
        path = STATIC_DIR / "index.html"
        if path.exists():
            return FileResponse(
                path,
                media_type="text/html",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )
        return HTMLResponse("<h1>static/index.html not found</h1>", status_code=404)

    @app.websocket("/ws")
    async def _ws(ws: WebSocket):
        origin = ws.headers.get("origin")
        if origin:
            parsed = urlparse(origin)
            if parsed.scheme not in {
                "http",
                "https",
            } or parsed.netloc != ws.headers.get("host"):
                await ws.close(code=1008)
                return
        q = ClientBuffer(config.queue_events, config.queue_bytes)
        if not broadcaster.register(q):
            await ws.close(code=1013)
            return
        tasks = []

        def local(message):
            q.put(encode(message, config.message_bytes))

        async def close(code):
            try:
                await asyncio.wait_for(ws.close(code=code), config.send_timeout)
            except (RuntimeError, WebSocketDisconnect, asyncio.TimeoutError):
                pass

        async def initialize():
            try:
                value = await read_state(wait_for_slot=True)
                initial = [
                    {"type": "ptt_state", "active": broadcaster.ptt_active},
                    broadcaster.cached_speaker_state(),
                    {"type": "capabilities", **value},
                ]
                active = value.get("active_turn")
                if (value.get("last_result") or {}).get("turn_id") == active:
                    active = None
                q.prime(
                    [encode(m, config.message_bytes) for m in initial],
                    value.get("event_cursor", 0),
                    active,
                )
            except Exception:
                q.fail("Initial state unavailable; reconnect to retry")
                q.initialized.set()

        async def sender():
            while True:
                message = await q.get()
                if message.stale:
                    continue
                await asyncio.wait_for(
                    ws.send_text(message.wire.decode("utf-8")), config.send_timeout
                )
                if message.kind == "overflow":
                    await close(1013)
                    return

        async def receiver():
            while True:
                packet = await ws.receive()
                if packet["type"] == "websocket.disconnect":
                    return
                data = packet.get("text")
                if data is None or len(data.encode("utf-8")) > config.ws_message_bytes:
                    await close(1009)
                    return
                try:
                    msg = json.loads(data)
                    if not isinstance(msg, dict):
                        raise ValueError("Expected a control object")
                    kind = msg.get("type")
                    if kind == "cancel":
                        request = CancelRequest.model_validate(msg)
                        if cancel is None:
                            raise ValueError("Cancellation unavailable")
                        local(
                            {
                                "type": "cancel_ack",
                                "turn_id": request.turn_id,
                                **cancel(turn_id=request.turn_id),
                            }
                        )
                    elif kind == "ptt":
                        if type(msg.get("active")) is not bool:
                            raise ValueError("PTT requires a boolean")
                        if broadcaster._input_setter is None:
                            raise ValueError("Audio input is not enabled")
                        broadcaster.set_ptt(msg["active"])
                    elif kind == "set_speaker":
                        sink = msg.get("sink")
                        if not isinstance(sink, str) or not 1 <= len(sink) <= 256:
                            raise ValueError("Invalid speaker")
                        app.state.calls.submit(
                            broadcaster.select_speaker, sink, control=True
                        )
                    elif kind == "set_speaker_volume":
                        volume = msg.get("volume")
                        if (
                            type(volume) not in {int, float}
                            or not math.isfinite(volume)
                            or not 0 <= volume <= 100
                        ):
                            raise ValueError("Volume must be between 0 and 100")
                        app.state.calls.submit(
                            broadcaster.set_speaker_volume, int(volume), control=True
                        )
                    else:
                        raise ValueError("Unknown control")
                except BusyRead:
                    local(
                        {
                            **broadcaster.cached_speaker_state(),
                            "error": "Speaker control busy; retry shortly",
                        }
                    )
                except (ValueError, TypeError, RecursionError):
                    local({"type": "error", "error": "Invalid or unavailable control"})

        try:
            await asyncio.wait_for(ws.accept(), config.send_timeout)
            tasks = [
                asyncio.create_task(sender()),
                asyncio.create_task(receiver()),
                asyncio.create_task(initialize()),
            ]
            # Initialization finishing is not a reason to close a healthy stream.
            done, _ = await asyncio.wait(tasks[:2], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (WebSocketDisconnect, EOFError):
            pass
        except asyncio.TimeoutError:
            await close(1013)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            broadcaster.unregister(q)

    return app
