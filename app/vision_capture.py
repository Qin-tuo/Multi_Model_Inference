# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Software-only snapshot selection from the upstream shared camera buffer."""
from app.camera import Camera


def capture_frames_for_vlm(camera: Camera, tracker=None, n_frames=1, **_legacy_options):
    if n_frames != 1:
        raise ValueError("Current profile admits exactly one current image")
    frame = camera.snapshot()
    if frame is None or not camera.health_check():
        raise RuntimeError("No fresh camera image is available")
    return [camera._base64_jpeg(frame.jpeg)], bool(tracker and tracker.stable)
