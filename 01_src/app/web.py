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

"""FastAPI 页面、HTTP 控制接口与 WebSocket 广播。

同步模型/采集线程只提交消息，异步事件循环负责客户端发送；连接、消息和读取工作均有预算。
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
    """将同步生产者消息送入异步客户端缓冲，维护有界桥接队列、序号和共享监听/扬声器状态。"""

    def __init__(self, config=None, frame_ttl=1.0):
        """初始化客户端与桥接队列预算、消息序号及路由回调；监听开关初始关闭。"""
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
        """切换所属事件循环并更新代次，清空旧队列和客户端，阻止旧 drain 回调继续发布。"""
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
        """在 WebSocket 握手前预留客户端名额并记录当前事件游标，达到上限时拒绝连接。"""
        with self._lock:
            if len(self._clients) >= self.config.max_clients:
                return False
            self._clients[client] = self._sequence
            return True

    def unregister(self, client):
        """移除并关闭客户端；最后一个客户端离开后清空待发送桥接数据。"""
        with self._lock:
            self._clients.pop(client, None)
            if not self._clients:
                self._pending.clear()
                self._pending_bytes = 0
                self._overflow = False
        client.close()

    @property
    def cursor(self):
        """在锁内读取最新全局事件序号，供初始快照和增量消息衔接。"""
        with self._lock:
            return self._sequence

    def snapshot(self):
        """返回连接数、桥接队列占用/峰值及合并、丢弃、溢出计数。"""
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
        """在锁内读取当前已预留或已连接的客户端数量。"""
        with self._lock:
            return len(self._clients)

    @property
    def ptt_active(self) -> bool:
        """返回共享监听开关是否启用。"""
        return self._ptt.is_set()

    def set_ptt(self, active: bool):
        """调用采集控制回调更新监听状态并广播；无法恢复输入时保持关闭并返回错误。"""
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
        """绑定扬声器状态、设备切换和音量控制回调，不在这里直接访问音频设备。"""
        self._speaker_getter = getter
        self._speaker_setter = setter
        self._speaker_volume_setter = volume_setter

    def get_speaker_state(self) -> dict:
        """通过回调刷新设备状态并缓存；失败时生成带错误的状态，避免异常传播到广播循环。"""
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
        """执行设备切换回调，缓存并广播结果或当前可见错误。"""
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
        """调用音量控制回调并广播刷新结果，未配置控制能力时明确报告不可用。"""
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
        """返回设备状态缓存副本，避免 Web 读者触发可能重建 AEC 的实时查询。"""
        return dict(self._speaker_cached)

    def _alarm(self, reason):
        """在持锁路径将桥接队列替换为溢出通知，要求客户端重连读取权威状态。"""
        self._pending.clear()
        self._pending_bytes = 0
        self._overflow = True
        self._overflows += 1
        self._sequence += 1
        message = replace(overflow(reason), sequence=self._sequence)
        self._pending.append(message)
        self._pending_bytes = len(message.wire)

    def send(self, msg, *, cancel_event=None):
        """编码并校验消息，在字节/条数预算内合并易失状态、过滤取消或过期内容并安排发送。

        先淘汰可替换状态；可靠消息无法保留时触发重同步告警，而不是悄悄丢失关键回执。
        """
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
        """保证同一代次最多安排一个 drain 回调；事件循环已关闭时清理待发消息。"""
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
        """在事件循环中分批取出桥接消息，并只向游标之前已注册的客户端派发新事件。"""
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


# 对话请求结构；图像开关必须是真正的布尔值。
class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    use_image: StrictBool = False
    request_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


# 取消操作必须指定明确的轮次 ID。
class CancelRequest(BaseModel):
    turn_id: str = Field(pattern=r"^[0-9a-f]{32}$")


# 清空历史需携带当前会话 ID，防止旧页面误清新会话。
class ClearSessionRequest(BaseModel):
    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")


# 身份登记请求只传当前目标 ID 和标签，不直接上传特征。
class VisionIdentityRequest(BaseModel):
    target_id: str = Field(min_length=1, max_length=160)
    label: str = Field(min_length=1, max_length=64)


# 目标选择参数；空 ID 表示取消选择。
class VisionTargetRequest(BaseModel):
    track_id: str | None = Field(default=None, max_length=160)


# 身份删除请求按受长度限制的标签定位。
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
    """把轮次、状态和视觉回调接入 FastAPI，统一配置请求体限制与有界同步调用池。"""
    config = broadcaster.config

    @asynccontextmanager
    async def application_lifespan(app):
        """绑定广播事件循环并创建 Web 工作池，在外部应用资源释放前等待读者退出。"""
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
        """通过有界线程池取得权威状态，槽位忙或超时时返回 503，并附加读取池指标。"""
        def read():
            """调用应用状态提供者；未注入提供者时返回未就绪状态。"""
            return state() if state else {"ready": False}

        try:
            value = await app.state.calls.run(read, config.read_timeout, wait_for_slot)
        except (BusyRead, asyncio.TimeoutError) as exc:
            raise HTTPException(503, "State reader occupied or timed out") from exc
        return {**value, "web_reads": app.state.calls.snapshot()}

    @app.get("/api/state")
    async def _state():
        """提供统一状态 HTTP 接口，复用带预算的状态读取路径。"""
        return await read_state()

    async def vision_call(name, *args):
        """通过有界线程池调用登记、查询或目标选择等视觉操作，并将业务冲突/忙转换为 HTTP 状态。"""
        if vision is None or name not in vision:
            raise HTTPException(503, "Vision operation unavailable")
        try:
            return await app.state.calls.run(
                # 封装同步视觉操作交给有界线程池，避免阻塞事件循环。
                lambda: vision[name](*args), config.read_timeout, True
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (BusyRead, asyncio.TimeoutError) as exc:
            raise HTTPException(503, "Vision operation occupied or timed out") from exc

    @app.get("/api/vision/identities")
    async def _vision_identities():
        """查询当前身份图库的标签摘要。"""
        return await vision_call("identities")

    @app.post("/api/vision/identities")
    async def _vision_enroll(request: VisionIdentityRequest):
        """把指定可见人脸目标的当前特征登记到用户给定标签。"""
        return await vision_call("enroll", request.target_id, request.label)

    @app.delete("/api/vision/identities")
    async def _vision_remove(request: VisionRemoveIdentityRequest):
        """按标签删除身份图库条目。"""
        return await vision_call("remove", request.label)

    @app.post("/api/vision/target")
    async def _vision_target(request: VisionTargetRequest):
        """选择当前可见实体轨迹，或传入空值取消选择。"""
        return await vision_call("target", request.track_id)

    @app.get("/api/debug")
    async def _debug_state():
        """从同一状态快照中返回板端诊断数据，不在 HTTP 请求内重新采样硬件。"""
        value = await read_state()
        return value.get("diagnostics", {"ready": False})

    @app.get("/static/{asset}")
    async def _debug_asset(asset: str):
        """仅提供允许列表中的前端脚本和样式，拒绝任意文件路径访问。"""
        if asset not in {"app.js", "app.css", "debug.js", "debug.css"}:
            raise HTTPException(404, "Asset not found")
        return FileResponse(STATIC_DIR / asset, headers={"Cache-Control": "no-cache"})

    @app.post("/api/chat", status_code=202)
    async def _chat(request: ChatRequest):
        """校验请求后接纳文字/图像问答，返回轮次回执；参数错误与轮次忙分别映射为 400/409。"""
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
        """按字节上限收集上传音频并提交语音请求，不在 HTTP 处理器中同步执行识别。"""
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
        """按明确轮次 ID 发起取消并返回接纳结果，后端资源释放由轮次状态另行确认。"""
        if cancel is None:
            raise HTTPException(503, "Cancellation unavailable")
        return cancel(turn_id=request.turn_id)

    @app.post("/api/session/clear")
    async def _clear_session(request: ClearSessionRequest):
        """使用会话 ID 清空短历史，身份变化或当前忙时返回冲突。"""
        if clear_history is None:
            raise HTTPException(503, "Session control unavailable")
        try:
            return clear_history(request.session_id)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/audio/{turn_id}")
    async def _download_audio(turn_id: str):
        """只返回回调确认可用的已发布 WAV，禁止缓存并对缺失结果返回 404。"""
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
        """提供对话与实时状态共用的页面，文件缺失时返回明确的 404。"""
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
        """校验 WebSocket 来源、预留连接名额，并管理初始化、发送和控制接收协程。

        发送或接收结束后取消其余任务并注销缓冲，避免初始化完成就误关连接或遗留后台任务。
        """
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
            """把仅面向当前客户端的控制回执编码后放入其有界队列。"""
            q.put(encode(message, config.message_bytes))

        async def close(code):
            """在发送时限内尝试关闭当前连接，容忍已断开或关闭超时。"""
            try:
                await asyncio.wait_for(ws.close(code=code), config.send_timeout)
            except (RuntimeError, WebSocketDisconnect, asyncio.TimeoutError):
                pass

        async def initialize():
            """读取权威初始状态并设置消息游标，保留尚未完成轮次的增量内容与工具回执。"""
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
            """持续发送未失效的队列消息，对单次网络发送限时；溢出告警发送后关闭连接等待重同步。"""
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
            """接收受大小限制的 JSON 控制消息，校验取消、监听、设备和音量参数后调用相应控制。

            慢速扬声器操作进入有界工作池，错误只反馈给当前客户端，不累积无限待处理任务。
            """
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
