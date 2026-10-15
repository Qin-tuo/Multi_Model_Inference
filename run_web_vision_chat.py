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

"""Derived upstream Web Vision Chat: shared camera, VLM streaming, WebSocket UI.

R1 exposes text and optional live camera input. Audio adapters join this same
application in R2. No robot SDK, daemon, motor or RAG initialization.
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
import logging
import threading

from app.config import Config
from app.camera import Camera
from app.face_detector import FaceDetector
from app.face_tracker import FaceTracker
from app.llm import LLM
from app.monitor import get_system_stats, get_jetson_model
from app.turn_control import TurnController
from app.web import Broadcaster, create_app

log = logging.getLogger(__name__)


def _frame_broadcast_thread(cam, broadcaster, stop, fps, tracker=None):
    last_key = None
    while not stop.wait(1 / fps):
        frame = cam.snapshot()
        face = None
        if tracker:
            observed, face = tracker.preview()
            if observed is not None:
                frame = observed
        if frame is None:
            broadcaster.send({"type": "camera_state", "ready": False, "error": cam.last_error or "No fresh frame"})
            continue
        key = (frame.source_epoch, frame.frame_id)
        if key == last_key: continue
        last_key = key
        if broadcaster.client_count:
            message = {"type": "frame", "data": cam._base64_jpeg(frame.jpeg), "frame": frame.metadata()}
            if tracker:
                face = face or tracker.snapshot()
                matched = face["frame"] and (face["frame"]["source_epoch"], face["frame"]["frame_id"]) == key
                message.update(face_detected=bool(matched and face["face_detected"]),
                               centered=bool(matched and face["centered"]), stable=bool(matched and face["stable"]))
                if matched and face["face_box"]: message["face_box"] = face["face_box"]
            broadcaster.send(message)


def _stats_broadcast_thread(broadcaster, stop, state, interval=2):
    while not stop.is_set():
        try:
            stats = get_system_stats()
            current = state()
            broadcaster.send({"type": "stats", "cpu": round(stats.cpu_percent, 1),
                "ram_used": stats.ram_used_mb / 1024, "ram_total": stats.ram_total_mb / 1024,
                "gpu": stats.gpu_percent, "models": current["models"],
                "platform": get_jetson_model(), "clients": broadcaster.client_count})
            broadcaster.send({"type": "perception", **current["face"]})
        except Exception:
            log.exception("Resource/vision status update failed")
        stop.wait(interval)


def build_app(config, *, no_camera=False):
    broadcaster = Broadcaster()
    stop = threading.Event()
    llm = LLM(model=config.llm.model, base_url=config.llm.base_url,
              backend=config.llm.backend, max_tokens=config.llm.max_tokens,
              temperature=config.llm.temperature, timeout=config.llm.timeout,
              system_prompt=config.llm.system_prompt, context_tokens=config.llm.context_tokens,
              chat_template_kwargs=config.llm.chat_template_kwargs)
    cam = None if no_camera else Camera(device=config.vision.camera_device,
              width=config.vision.width, height=config.vision.height,
              jpeg_quality=config.vision.jpeg_quality, capture_fps=config.vision.capture_fps,
              stale_after=config.vision.stale_after, buffer_bytes_limit=config.vision.buffer_bytes_limit)
    detector = FaceDetector() if cam and config.face.enabled else None
    tracker = FaceTracker(cam, detector, fps=config.face.fps, dead_zone=config.face.dead_zone,
              good_frame_zone=config.face.good_frame_zone, min_face_size=config.face.min_face_size,
              stable_frames=config.face.stable_frames) if detector else None
    control = TurnController(config, llm, broadcaster, cam)
    threads = []

    def state():
        return {**control.state(), "audio_enabled": False, "camera_enabled": cam is not None,
                "camera_ready": bool(cam and cam.health_check()),
                "camera_error": cam.last_error if cam else None,
                "face": {"enabled": bool(tracker), **(tracker.snapshot() if tracker else {"face_detected": False})},
                "models": {"vlm": llm.model, "stt": "未启用", "tts": "未启用", "vad": "未启用"},
                "effective_config": config.effective()}

    @asynccontextmanager
    async def lifespan(app):
        try:
            if not await asyncio.to_thread(llm.load):
                raise RuntimeError(llm.last_error or "Local VLM is not ready")
            if cam and not await asyncio.to_thread(cam.start):
                raise RuntimeError(cam.last_error or "Camera is not ready")
            if detector and not await asyncio.to_thread(detector.load):
                raise RuntimeError(detector.last_error or "Face detector is not ready")
            if tracker: tracker.start()
            control.start()
            if cam:
                threads.append(threading.Thread(target=_frame_broadcast_thread,
                    args=(cam, broadcaster, stop, config.web.ui_fps, tracker), name="frame-broadcast", daemon=True))
            threads.append(threading.Thread(target=_stats_broadcast_thread,
                args=(broadcaster, stop, state), name="stats-broadcast", daemon=True))
            for thread in threads: thread.start()
            yield
        finally:
            stop.set()
            errors = []
            for thread in threads:
                await asyncio.to_thread(thread.join, 5)
                if thread.is_alive(): errors.append(RuntimeError(f"{thread.name} still running"))
            for resource in [control, tracker, cam]:
                if resource:
                    try:
                        await asyncio.to_thread(resource.stop if resource is tracker else resource.close)
                    except Exception as exc: errors.append(exc)
            if detector: detector.unload()
            llm.unload()
            if errors: raise ExceptionGroup("Shutdown resources not fully released", errors)

    return create_app(broadcaster, submit=control.submit, state=state, lifespan=lifespan)


def main():
    parser = argparse.ArgumentParser(description="Orin NX headless Web Vision Chat")
    parser.add_argument("--config")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-camera", action="store_true", help="Explicit text-only software mode")
    args = parser.parse_args()
    config = Config.load(args.config)
    if args.host: config.web.host = args.host
    if args.port is not None: config.web.port = args.port
    config.validate()
    import uvicorn
    uvicorn.run(build_app(config, no_camera=args.no_camera), host=config.web.host,
                port=config.web.port, log_level="info", ws_max_size=65536,
                limit_concurrency=16, timeout_graceful_shutdown=10)


if __name__ == "__main__":
    main()
