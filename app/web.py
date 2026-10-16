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
from pathlib import Path
from typing import Callable, Optional
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from pydantic import BaseModel, Field, StrictBool
from fastapi.responses import FileResponse, HTMLResponse

STATIC_DIR = Path(__file__).parent.parent / "static"


class Broadcaster:
    """Thread-safe fan-out from the synchronous pipeline to async WebSocket clients.

    Each connected client gets its own asyncio.Queue.  The pipeline calls
    send() from any thread; messages are routed into the event loop via
    call_soon_threadsafe.

    Also holds shared push-to-talk (PTT) state: set = unmuted / listening
    by default, cleared = muted. Any client can toggle via WebSocket.
    """

    def __init__(self):
        self._clients: dict[asyncio.Queue, None] = {}
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ptt = threading.Event()
        self._ptt.clear()
        self._input_setter = None
        self._speaker_getter: Optional[Callable[[], dict]] = None
        self._speaker_setter: Optional[Callable[[str], dict]] = None
        self._speaker_volume_setter: Optional[Callable[[int], dict]] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def register(self, q: asyncio.Queue):
        with self._lock:
            self._clients[q] = None

    def unregister(self, q: asyncio.Queue):
        with self._lock:
            self._clients.pop(q, None)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    @property
    def ptt_active(self) -> bool:
        return self._ptt.is_set()

    def set_ptt(self, active: bool):
        if self._input_setter: self._input_setter(active)
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
            return {"type": "speaker_state", **self._speaker_getter()}
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
        self.send({"type": "speaker_state", **state})

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
        self.send({"type": "speaker_state", **state})

    @staticmethod
    def _enqueue_latest(q: asyncio.Queue, msg: dict):
        """Keep a slow client's queue current without leaking payloads to logs."""
        if msg.get('_cancel_event') is not None and msg['_cancel_event'].is_set(): return
        try:
            q.put_nowait(msg)
            return
        except asyncio.QueueFull:
            pass

        if msg.get("type") in {"frame", "stats", "perception"}:
            return
        # Reliable output cannot be silently discarded. Notify and disconnect
        # a slow client; the last complete result is available at /api/state.
        msg = {"type": "overflow", "error": "Viewer fell behind; reconnect to refresh state"}
        # Camera frames are large and quickly become stale. Drop one queued
        # message and retry instead of letting QueueFull escape from the event
        # loop callback, where asyncio would log the complete base64 payload.
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass

        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    @staticmethod
    def visible(msg):
        event = msg.get('_cancel_event')
        if event is not None and event.is_set(): return None
        return {key:value for key,value in msg.items() if key != '_cancel_event'}

    def send(self, msg: dict, *, cancel_event=None):
        """Enqueue *msg* for every connected client (thread-safe)."""
        loop = self._loop
        if not loop:
            return
        if cancel_event is not None: msg = {**msg, '_cancel_event':cancel_event}
        with self._lock:
            for q in self._clients:
                try:
                    loop.call_soon_threadsafe(self._enqueue_latest, q, msg)
                except Exception:
                    pass


class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    use_image: StrictBool = False


class CancelRequest(BaseModel):
    turn_id: str = Field(pattern=r'^[0-9a-f]{32}$')


def create_app(broadcaster: Broadcaster, *, submit=None, state=None, lifespan=None,
               submit_audio=None, audio_file=None, cancel=None, max_upload_bytes=484096) -> FastAPI:
    @asynccontextmanager
    async def application_lifespan(app):
        broadcaster.set_loop(asyncio.get_running_loop())
        try:
            if lifespan:
                async with lifespan(app): yield
            else:
                yield
        finally:
            broadcaster.set_loop(None)

    app = FastAPI(title="Orin NX Companion", lifespan=application_lifespan)

    @app.get("/api/state")
    async def _state():
        return state() if state else {"ready": False}

    @app.post("/api/chat", status_code=202)
    async def _chat(request: ChatRequest):
        if submit is None: raise HTTPException(503, "Conversation input unavailable")
        try:
            return submit(request.text, request.use_image)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/audio", status_code=202)
    async def _audio(request: Request, use_image: bool = False):
        if submit_audio is None: raise HTTPException(503, "Speech input unavailable")
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > max_upload_bytes: raise HTTPException(413, "Audio input exceeds size limit")
        try: return submit_audio(bytes(data), use_image)
        except ValueError as exc: raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc: raise HTTPException(409, str(exc)) from exc

    @app.post('/api/cancel', status_code=202)
    async def _cancel(request: CancelRequest):
        if cancel is None: raise HTTPException(503, 'Cancellation unavailable')
        return cancel(turn_id=request.turn_id)

    @app.get("/api/audio/{turn_id}")
    async def _download_audio(turn_id: str):
        path = audio_file(turn_id) if audio_file else None
        if path is None: raise HTTPException(404, "Audio output is unavailable")
        return FileResponse(path, media_type="audio/wav", filename="reply.wav",
                            headers={"Cache-Control":"no-store"})

    @app.get("/")
    async def _index():
        path = STATIC_DIR / "index.html"
        if path.exists():
            return FileResponse(
                path, media_type="text/html",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )
        return HTMLResponse("<h1>static/index.html not found</h1>", status_code=404)

    @app.websocket("/ws")
    async def _ws(ws: WebSocket):
        origin = ws.headers.get("origin")
        if origin:
            parsed = urlparse(origin)
            if parsed.scheme not in {"http", "https"} or parsed.netloc != ws.headers.get("host"):
                await ws.close(code=1008)
                return
        if broadcaster.client_count >= 4:
            await ws.close(code=1013)
            return
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=16)
        broadcaster.register(q)

        q.put_nowait({"type": "ptt_state", "active": broadcaster.ptt_active})
        q.put_nowait(broadcaster.get_speaker_state())
        if state:
            q.put_nowait({"type": "capabilities", **state()})

        async def _sender():
            try:
                while True:
                    msg = await q.get()
                    msg = broadcaster.visible(msg)
                    if msg is None: continue
                    await ws.send_text(json.dumps(msg))
                    if msg.get("type") == "overflow":
                        await ws.close(code=1013)
                        return
            except (WebSocketDisconnect, Exception):
                pass

        async def _receiver():
            try:
                while True:
                    data = await ws.receive_text()
                    try:
                        msg = json.loads(data)
                        if msg.get("type") == "ptt":
                            if broadcaster._speaker_getter is None:
                                q.put_nowait({"type": "error", "error": "Audio input is not enabled"})
                            else:
                                broadcaster.set_ptt(bool(msg.get("active", False)))
                        elif msg.get("type") == "set_speaker":
                            sink_id = msg.get("sink")
                            if isinstance(sink_id, str) and sink_id:
                                await asyncio.to_thread(
                                    broadcaster.select_speaker, sink_id,
                                )
                        elif msg.get("type") == "set_speaker_volume":
                            volume = msg.get("volume")
                            if (
                                isinstance(volume, (int, float))
                                and not isinstance(volume, bool)
                            ):
                                await asyncio.to_thread(
                                    broadcaster.set_speaker_volume, int(volume),
                                )
                    except (ValueError, KeyError):
                        pass
            except (WebSocketDisconnect, Exception):
                pass

        send_task = asyncio.ensure_future(_sender())
        recv_task = asyncio.ensure_future(_receiver())
        try:
            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
        except Exception:
            send_task.cancel()
            recv_task.cancel()
        finally:
            send_task.cancel()
            recv_task.cancel()
            await asyncio.gather(send_task, recv_task, return_exceptions=True)
            broadcaster.unregister(q)

    return app
