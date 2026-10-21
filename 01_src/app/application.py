#!/usr/bin/env python3
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


"""Web 应用资源组装和生命周期入口。

在输入与后端可用后启动消费者，统一关闭采集、广播、轮次、感知和模型接口。
"""

import argparse
import asyncio
from contextlib import asynccontextmanager
import logging
import threading
import time
from app.config import Config
from app.camera import Camera
from app.face_state import FaceState
from app.vision_pipeline import VisionPipeline
from app.gestures import GestureRecognizer
from app.llm import LLM
from app.stt import STT
from app.tts import create_tts
from app.speech.vad import SileroVAD, vad_loop
from app.speech.capture import MicRecorder
from app.speech.devices import find_pa_sink
from app.audio import find_alsa_device
from rich.console import Console
from app.turn_control import TurnController
from app.web import Broadcaster, create_app
from app.diagnostics import BoardDiagnostics
from app.broadcast import _frame_broadcast_thread, _stats_broadcast_thread

log = logging.getLogger(__name__)


def build_app(config, *, no_camera=False, no_audio=False, no_actions=False):
    """组装一份应用资源图；实际加载与释放由下面的 lifespan 统一拥有。

    Camera 是唯一采集者，Perception 消费同一来源的最新帧；
    TurnController 持有当前轮次，WebSocket 只负责输入输出与观测。
    """
    broadcaster = Broadcaster(config.web, frame_ttl=config.vision.stale_after)
    diagnostics = BoardDiagnostics(config)
    stop = threading.Event()
    llm = LLM(
        model=config.llm.model,
        base_url=config.llm.base_url,
        backend=config.llm.backend,
        max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature,
        timeout=config.llm.timeout,
        system_prompt=config.llm.system_prompt,
        context_tokens=config.llm.context_tokens,
        chat_template_kwargs=config.llm.chat_template_kwargs,
    )
    cam = (
        None
        if no_camera
        else Camera(
            device=config.vision.camera_device,
            width=config.vision.width,
            height=config.vision.height,
            jpeg_quality=config.vision.jpeg_quality,
            capture_fps=config.vision.capture_fps,
            stale_after=config.vision.stale_after,
            buffer_bytes_limit=config.vision.buffer_bytes_limit,
            reconnect_seconds=config.vision.reconnect_seconds,
            v4l_select_timeout=config.vision.v4l_select_timeout,
            backend=config.vision.backend,
            serial=config.vision.serial,
        )
    )
    tracker = (
        FaceState(
            cam,
            dead_zone=config.face.dead_zone,
            good_frame_zone=config.face.good_frame_zone,
            min_face_size=config.face.min_face_size,
            stable_frames=config.face.stable_frames,
        )
        if cam and config.face.enabled
        else None
    )
    perception = (
        VisionPipeline(cam, config.perception, tracker)
        if cam and config.perception.enabled
        else None
    )
    speech_enabled = config.audio.enabled and not no_audio
    stt = STT(**vars(config.stt)) if speech_enabled else None
    tts = (
        create_tts(
            **{
                k: v
                for k, v in vars(config.tts).items()
                if k not in {"first_chunk_chars", "max_chunk_chars"}
            }
        )
        if speech_enabled
        else None
    )
    mic = None
    live_vad = None
    control = TurnController(
        config, llm, broadcaster, cam, stt=stt, tts=tts, perception=perception
    )
    gestures = (
        GestureRecognizer(
            config.gestures,
            config.perception.hand_threshold,
            config.perception.pose_joint_threshold,
        )
        if perception and config.perception.hands_enabled
        else None
    )
    if gestures:

        def consume_scene(scene):
            """把同帧感知结果交给手势识别器，再由轮次控制器统一接纳问候或执行取消。"""
            for event in gestures.update(scene):
                control.handle_gesture(event)

        perception.on_result = consume_scene
    threads = []

    def action_state():
        """读取动作存储状态；服务关闭或读取失败时返回可见错误，而不阻断整个状态接口。"""
        if control.actions is None:
            return {"enabled": False}
        try:
            return control.actions.snapshot()
        except Exception as exc:
            return {"enabled": True, "error": str(exc)}

    def state():
        """汇总对话、相机、视觉、语音、动作和板端诊断，生成 Web/服务健康检查使用的状态。"""
        camera_state = cam.status() if cam else {"ready": False, "state": "disabled"}
        echo_ready = bool(mic and mic.aec and mic.aec.active)
        return {
            **control.state(),
            "audio_enabled": bool(mic and mic.alive),
            "live_audio": {
                "enabled": bool(mic),
                "input": mic.input_state() if mic else None,
                "interruptions_available": bool(
                    mic
                    and mic.alive
                    and config.audio.voice_interruptions
                    and (config.audio.output_mode != "device" or echo_ready)
                ),
                "aec_active": echo_ready,
            },
            "actions": action_state(),
            "web": broadcaster.snapshot(),
            "diagnostics": diagnostics.snapshot(),
            "speech_ready": bool(speech_enabled and control.vad is not None),
            "audio_input_mode": config.audio.input_mode
            if speech_enabled
            else "disabled",
            "audio_output_mode": config.audio.output_mode
            if speech_enabled
            else "disabled",
            "microphone_error": mic.last_error if mic else None,
            "aec_active": bool(mic and mic.aec and mic.aec.active),
            "camera_enabled": cam is not None,
            "camera_ready": camera_state["ready"],
            "camera": camera_state,
            "camera_error": cam.last_error if cam else None,
            "face": {
                "enabled": bool(tracker),
                **(tracker.snapshot() if tracker else {"face_detected": False}),
            },
            "perception": perception.snapshot()
            if perception
            else {
                "ready": False,
                "error": "Perception disabled",
                "persons": [],
                "hands": [],
            },
            "gestures": {**gestures.snapshot(), "error": perception.callback_error}
            if gestures
            else {"ready": False, "reason": "disabled", "tracks": []},
            "models": {
                "vlm": llm.model,
                "stt": config.stt.model if speech_enabled else "未启用",
                "tts": config.tts.model if speech_enabled else "未启用",
                "vad": "Silero v6" if speech_enabled else "未启用",
            },
            "effective_config": config.effective(),
        }

    def listen():
        """消费实时 VAD 语段并交给轮次控制器，丢弃旧输入批次；退出时关闭所拥有的麦克风。"""
        quiet_until = 0.0

        def gate():
            """根据打断配置、AEC 可用性和对话忙闲决定半双工门控，并保留一段静音恢复时间。"""
            nonlocal quiet_until
            duplex = config.audio.voice_interruptions and (
                config.audio.output_mode != "device" or bool(mic.aec and mic.aec.active)
            )
            if not duplex and control.conversation_busy():
                quiet_until = time.monotonic() + config.vad.silence_duration_ms / 1000
            return "half_duplex" if time.monotonic() < quiet_until else None

        try:
            for segment in vad_loop(
                mic,
                Console(),
                config.vad,
                live_vad,
                on_start=control.begin_voice,
                on_discard=control.discard_voice,
                capture_gate=gate,
                max_age_seconds=config.audio.input_stale_seconds,
            ):
                if stop.is_set():
                    break
                if segment.input_epoch != mic.input_epoch:
                    continue
                try:
                    control.submit_segment(segment)
                except (RuntimeError, ValueError) as exc:
                    broadcaster.send({"type": "error", "error": str(exc)})
        except Exception as exc:
            broadcaster.send({"type": "error", "error": str(exc)})
            log.exception("Audio input failed")
        finally:
            # Input failure is visible and closes only the owned capture. It
            # requires application restart; speaker selection cannot revive an
            # orphan capture after this VAD worker has exited.
            try:
                mic.stop()
            except Exception:
                log.exception("Microphone cleanup failed")

    @asynccontextmanager
    async def lifespan(app):
        """异步管理应用资源的加载、线程启动和有序释放。

        先检查模型/设备，再启动感知和消费者；退出时先停输入与广播，再等待轮次和原生资源释放，未确认释放会汇总报错。
        """
        nonlocal mic, live_vad
        try:
            # 先确认后端与输入可用，再启动消费者；加载失败也进入 finally 回收。
            if not await asyncio.to_thread(llm.load):
                raise RuntimeError(llm.last_error or "Local VLM is not ready")
            if config.actions.enabled and not no_actions:
                from app.actions.store import ActionStore

                await asyncio.to_thread(llm.require_tools)
                control.actions = await asyncio.to_thread(ActionStore, config.actions)
            if cam and not await asyncio.to_thread(cam.start):
                raise RuntimeError(cam.last_error or "Camera is not ready")
            if speech_enabled:
                for model in (stt, tts):
                    if not await asyncio.to_thread(model.load):
                        raise RuntimeError(model.last_error or "Speech model not ready")
                control.vad = await asyncio.to_thread(SileroVAD)
                if config.audio.input_mode == "device":
                    device = await asyncio.to_thread(
                        find_alsa_device, config.audio.input_device
                    )
                    if not device:
                        raise RuntimeError("Configured microphone is not connected")
                    card, number, _ = device
                    if (
                        config.audio.output_mode == "device"
                        and not config.audio.output_device.startswith(
                            ("hw:", "plughw:")
                        )
                    ):
                        if not await asyncio.to_thread(
                            find_pa_sink, config.audio.output_device
                        ):
                            raise RuntimeError("Configured speaker is not connected")
                    mic = MicRecorder(Console(), config.vad.chunk_ms)
                    if not await asyncio.to_thread(
                        mic.start,
                        f"hw:{card},{number}",
                        config.audio.input_device,
                        config.audio.output_device,
                        config.audio.echo_cancellation
                        and config.audio.output_mode == "device"
                        and not config.audio.output_device.startswith(
                            ("hw:", "plughw:")
                        ),
                    ):
                        raise RuntimeError(mic.last_error or "Microphone not ready")
                    live_vad = await asyncio.to_thread(SileroVAD)
                    if (
                        config.audio.output_mode == "device"
                        and not config.audio.output_device.startswith(
                            ("hw:", "plughw:")
                        )
                    ):
                        control.output_sink = mic.get_pa_sink
                    broadcaster.configure_speakers(
                        mic.speaker_state, mic.select_speaker, mic.set_speaker_volume
                    )
                    # The upstream getter may rebuild an AEC route; do this
                    # once during startup, never concurrently in Web readers.
                    await asyncio.to_thread(broadcaster.get_speaker_state)
                    broadcaster._input_setter = (
                        # 监听开关只切换已有采集者，不创建第二条录音链路。
                        lambda active: mic.resume() if active else mic.pause()
                    )
                    broadcaster.set_ptt(True)
            if perception:
                await asyncio.to_thread(perception.load)
                perception.start()
            control.start()
            if mic:
                threads.append(
                    threading.Thread(target=listen, name="audio-vad", daemon=True)
                )
            if cam:
                threads.append(
                    threading.Thread(
                        target=_frame_broadcast_thread,
                        args=(
                            cam,
                            broadcaster,
                            stop,
                            config.web.ui_fps,
                            tracker,
                            perception,
                        ),
                        name="frame-broadcast",
                        daemon=True,
                    )
                )
            threads.append(
                threading.Thread(
                    target=_stats_broadcast_thread,
                    args=(broadcaster, stop, state, diagnostics),
                    name="stats-broadcast",
                    daemon=True,
                )
            )
            for thread in threads:
                thread.start()
            yield
        finally:
            # 先停止麦克风与广播线程，再关闭轮次、感知和相机，最后卸载接口。
            # join 后仍存活的线程属于未释放资源，不能当作成功退出。
            stop.set()
            errors = []
            if mic:
                try:
                    await asyncio.to_thread(mic.stop)
                except Exception as exc:
                    errors.append(exc)
            for thread in threads:
                await asyncio.to_thread(thread.join, 5)
                if thread.is_alive():
                    errors.append(RuntimeError(f"{thread.name} still running"))
            for resource in [control, perception, tracker, cam]:
                if resource:
                    try:
                        await asyncio.to_thread(
                            resource.stop if resource is tracker else resource.close
                        )
                    except Exception as exc:
                        errors.append(exc)
            if control.actions:
                try:
                    await asyncio.to_thread(control.actions.close)
                except Exception as exc:
                    errors.append(exc)
            llm.unload()
            for model in (stt, tts):
                if model:
                    model.unload()
            if errors:
                raise ExceptionGroup("Shutdown resources not fully released", errors)

    return create_app(
        broadcaster,
        submit=control.submit,
        state=state,
        lifespan=lifespan,
        submit_audio=control.submit_audio,
        audio_file=control.audio_file,
        cancel=control.cancel,
        clear_history=control.clear_history,
        vision={
            # 图库在生命周期中加载，回调执行时再取得当前实例。
            "identities": lambda: perception.gallery.list(),
            "enroll": perception.enroll,
            # 删除操作限定在本应用图库内，复用其事务和锁。
            "remove": lambda label: perception.gallery.remove(label),
            "target": perception.select_target,
        }
        if perception
        else None,
        max_upload_bytes=int(config.stt.max_seconds * 32000) + 4096,
    )


def main(default_no_camera=False):
    """解析运行参数、加载并验证配置，组装应用后启动带连接与消息预算的 Uvicorn 服务。"""
    parser = argparse.ArgumentParser(description="Orin NX headless Web Vision Chat")
    parser.add_argument("--config")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--no-audio", action="store_true", help="Disable speech backends and audio I/O"
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Disable local notes/reminders and scheduler",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        default=default_no_camera,
        help="Explicit text-only software mode",
    )
    args = parser.parse_args()
    config = Config.load(args.config)
    if args.host:
        config.web.host = args.host
    if args.port is not None:
        config.web.port = args.port
    config.validate()
    import uvicorn

    uvicorn.run(
        build_app(
            config,
            no_camera=args.no_camera,
            no_audio=args.no_audio,
            no_actions=args.no_tools,
        ),
        host=config.web.host,
        port=config.web.port,
        log_level="info",
        ws_max_size=config.web.ws_message_bytes,
        ws_max_queue=4,
        limit_concurrency=16,
        timeout_graceful_shutdown=10,
    )
