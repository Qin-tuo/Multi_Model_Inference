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


"""Publish paired camera/results and bounded board diagnostics to Web viewers."""

import logging
from app.monitor import get_system_stats, get_jetson_model

log = logging.getLogger(__name__)


def _frame_broadcast_thread(cam, broadcaster, stop, fps, tracker=None, perception=None):
    last_key = None
    while not stop.wait(1 / fps):
        frame = cam.snapshot()
        face = None
        scene = None
        if perception:
            observed, scene = perception.preview()
            if observed is not None:
                frame = observed
                face = scene["face"]
        elif tracker:
            observed, face = tracker.preview()
            if observed is not None:
                frame = observed
        if frame is None:
            broadcaster.send(
                {
                    "type": "camera_state",
                    "ready": False,
                    "error": cam.last_error or "No fresh frame",
                }
            )
            continue
        key = (frame.source_epoch, frame.frame_id)
        if key == last_key:
            continue
        last_key = key
        if broadcaster.client_count:
            message = {
                "type": "frame",
                "data": cam._base64_jpeg(frame.jpeg),
                "frame": frame.metadata(),
            }
            if tracker:
                face = face or tracker.snapshot()
                matched = (
                    face["frame"]
                    and (face["frame"]["source_epoch"], face["frame"]["frame_id"])
                    == key
                )
                message.update(
                    face_detected=bool(matched and face["face_detected"]),
                    centered=bool(matched and face["centered"]),
                    stable=bool(matched and face["stable"]),
                )
                if matched and face["face_box"]:
                    message["face_box"] = face["face_box"]
            if (
                scene
                and scene["ready"]
                and scene["frame"]["frame_id"] == frame.frame_id
                and scene["frame"]["source_epoch"] == frame.source_epoch
            ):
                message.update(
                    persons=scene["persons"],
                    hands=scene["hands"],
                    objects=scene.get("objects", []),
                    poses=scene.get("poses", []),
                    faces=scene.get("faces", []),
                    attributes=scene.get("attributes", []),
                    perception_frame=scene["frame"],
                )
            else:
                message.update(persons=[], hands=[])
            broadcaster.send(message, cancel_event=frame.invalidated)


def _stats_broadcast_thread(broadcaster, stop, state, diagnostics, interval=2):
    while not stop.is_set():
        try:
            stats = get_system_stats()
            current = state()
            broadcaster.send(
                {"type": "diagnostics", **diagnostics.sample(current, stats)}
            )
            broadcaster.send(
                {
                    "type": "stats",
                    "cpu": round(stats.cpu_percent, 1),
                    "ram_used": stats.ram_used_mb / 1024,
                    "ram_total": stats.ram_total_mb / 1024,
                    "gpu": stats.gpu_percent,
                    "models": current["models"],
                    "platform": get_jetson_model(),
                    "clients": broadcaster.client_count,
                }
            )
            broadcaster.send({"type": "perception", **current["face"]})
            broadcaster.send({"type": "scene", **current["perception"]})
            broadcaster.send(
                {
                    "type": "gesture_state",
                    **current["gestures"],
                    "last_action": current["last_gesture"],
                }
            )
            broadcaster.send({"type": "actions_state", **current["actions"]})
            broadcaster.send({"type": "audio_state", **current["live_audio"]})
        except Exception:
            log.exception("Resource/vision status update failed")
        stop.wait(interval)
