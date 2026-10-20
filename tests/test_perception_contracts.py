# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from types import SimpleNamespace
import json
import time
import numpy as np
import pytest
from app.camera import Camera, FrameSnapshot
from app.config import Config
from app.face_state import FaceState
from app.frame_worker import FrameWorker
from app.vision_pipeline import VisionPipeline
from app.vision_models import ROLES


def frame(camera):
    image = np.zeros((100, 200, 3), np.uint8)
    return FrameSnapshot(7, camera.source_epoch, time.monotonic(), image, b"jpeg")


def test_current_models_share_the_original_frame_and_target_coordinates():
    camera = Camera(stale_after=1.5)
    owner = VisionPipeline(camera, Config.load().perception)
    owner.gallery = SimpleNamespace(match=lambda *args: {"status": "unknown"}, close=lambda: None)
    owner.networks.models = {
        role: SimpleNamespace(provider="test", close=lambda: None) for role in ROLES
    }
    image = np.zeros((100, 200, 3), np.uint8)
    seen = []
    vector = np.eye(1, 512, dtype=np.float32)[0]

    def infer(role, received, *arguments, **kwargs):
        seen.append((role, id(received)))
        if role == "detection_segmentation":
            return [
                {"box_xyxy": [0, 0, 200, 100], "score": 0.9, "class_id": 0,
                 "class_name": "person", "mask": np.ones((100, 200), bool)},
                {"box_xyxy": [0, 0, 40, 40], "score": 0.9, "class_id": 5,
                 "class_name": "bus", "mask": np.ones((100, 200), bool)},
            ]
        if role == "face":
            return [{"box_xyxy": [80, 30, 120, 70], "score": 0.9,
                     "landmarks": [[85, 40], [110, 40], [100, 50], [90, 60], [108, 60]]}]
        if role in ("face_identity", "person_reid", "object_embedding"):
            return {"embedding": vector, "embedding_space": "test-space"}
        if role == "body_hand_pose":
            assert arguments[0] == [0, 0, 200, 100]
            return {"hands": []}
        if role == "head_pose":
            return {"angles_degrees": [0, 0, 0]}
        if role == "gaze_landmarks":
            return {"landmarks": np.zeros((35, 2))}
        return {"available": True}

    owner.networks.run = infer
    try:
        for index in (7, 8):
            sample = FrameSnapshot(
                index, camera.source_epoch, time.monotonic(), image, b"jpeg",
                depth_m=np.full((100, 200), 1.5, np.float32),
                depth_valid=np.ones((100, 200), bool),
            )
            result = owner.process_frame(sample)
        assert {role for role, _ in seen} == set(ROLES)
        assert all(identity == id(image) for _, identity in seen)
        assert result["frame"]["frame_id"] == result["face"]["frame"]["frame_id"] == 8
        assert all(pose["frame"]["frame_id"] == 8 for pose in result["poses"])
    finally:
        owner.close()


def test_current_pipeline_rejects_camera_epoch_change_during_inference():
    camera = Camera()
    owner = VisionPipeline(camera, Config.load().perception)
    sample = frame(camera)

    def infer(role, image, *args, **kwargs):
        camera.source_epoch = "reconnected"
        return []

    owner.networks.run = infer
    try:
        with pytest.raises(RuntimeError, match="expired"):
            owner.process_frame(sample)
    finally:
        owner.close()


def test_expired_preview_cannot_keep_boxes_alive():
    camera = Camera(stale_after=0.01)
    sample = frame(camera)
    perception = FrameWorker(camera, Config.load().perception)
    perception._preview = sample
    perception._state = {"ready": True, "frame": sample.metadata()}
    time.sleep(0.02)
    assert perception.preview()[0] is None
    assert not perception.snapshot()["ready"]


@pytest.mark.parametrize("body_provider", ["fake", None])
def test_vlm_image_and_auxiliary_observation_use_same_frame(body_provider):
    from app.turn_control import TurnController

    cfg = Config.load()
    camera = Camera()
    chosen = frame(camera)
    camera._alive = True
    camera._latest = (time.monotonic(), chosen.bgr)
    camera._latest_jpeg = b"newer"
    camera._frame_id = 8
    scene = {
        "ready": True,
        "frame": chosen.metadata(),
        "persons": [{}],
        "hands": [{}, {}],
        "face": {"face_detected": True},
        "providers": {"body": body_provider, "face": "fake", "hands": "fake"},
    }

    class Model:
        def idle(self):
            return True

        def generate_stream(self, prompt, **kwargs):
            assert kwargs["images_b64"] == [camera._base64_jpeg(chosen.jpeg)]
            assert "person_candidates" in kwargs["system_prompt"]
            yield "回答", {}
            yield "", {"done": True}

    sink = SimpleNamespace(send=lambda event, **kwargs: None)
    control = TurnController(
        cfg,
        Model(),
        sink,
        camera,
        perception=SimpleNamespace(preview=lambda: (chosen, scene)),
    )
    try:
        control.start()
        control.submit("画面是什么", True)
        deadline = time.monotonic() + 2
        while control.state()["active_turn"] and time.monotonic() < deadline:
            time.sleep(0.01)
        result = control.state()["last_result"]
        assert result["status"] == "complete"
        assert (
            result["frame"]["frame_id"]
            == result["perception_observation"]["frame"]["frame_id"]
            == 7
        )
        assert result["perception_observation"]["values"]["person_candidates"] == (
            1 if body_provider else None
        )
    finally:
        control.close()
        camera.close()
