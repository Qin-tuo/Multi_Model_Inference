# SPDX-License-Identifier: Apache-2.0
"""Exercise failed-load rollback and per-lifetime state without GPU fixtures."""

from types import SimpleNamespace

import pytest

from app.config import Config
from app.vision_pipeline import VisionPipeline
from app.vision_models import VisionModels
from app.visual_tracks import VisualTracks
import app.vision_models as models


@pytest.fixture(autouse=True)
def fake_memory_admission(monkeypatch):
    monkeypatch.setattr(
        models.psutil, "virtual_memory", lambda: SimpleNamespace(available=16 * 1024**3)
    )


class FakeFace:
    provider = "test-only"

    def __init__(self, spec):
        self.model = SimpleNamespace(io_bytes=64)
        self.closed = False

    def infer(self, value):
        assert not self.closed
        return value

    def close(self):
        self.closed = True


def test_reload_starts_new_metrics_window(monkeypatch):
    owner = VisionModels()
    owner.manifest["runtime_versions"] = {}
    monkeypatch.setattr(models, "ROLES", ("face",))
    monkeypatch.setattr(models, "TensorRTFace", FakeFace)
    try:
        owner.load()
        owner.run("face", "first-lifetime")
        assert owner.snapshot()["face"]["p50_ms"] is not None
        owner.close()
        owner.load()
        fresh = owner.snapshot()["face"]
        assert fresh["calls"] == 0
        assert fresh["p50_ms"] is None and fresh["p95_ms"] is None
    finally:
        owner.close()


def test_partial_load_failure_releases_successfully_loaded_models(monkeypatch):
    owner = VisionModels()
    owner.manifest["runtime_versions"] = {}
    created = []

    def face(spec):
        value = FakeFace(spec)
        created.append(value)
        return value

    def broken(*args):
        raise RuntimeError("injected load failure")

    monkeypatch.setattr(models, "ROLES", ("face", "body_hand_pose"))
    monkeypatch.setattr(models, "TensorRTFace", face)
    monkeypatch.setattr(models, "WholeBodyPose2D", broken)
    with pytest.raises(RuntimeError, match="injected load failure"):
        owner.load()
    assert len(created) == 1 and created[0].closed
    assert owner.models == {}


def test_tracking_reset_cannot_reuse_old_frame_order_or_target_ids():
    tracker = VisualTracks("entity")
    detection = {"box_xyxy": [0, 0, 50, 100], "score": 0.9, "class_id": 0}

    def observe(i):
        return tracker.update(
            [detection],
            SimpleNamespace(
                source_epoch="same-camera", frame_id=i, received_monotonic=float(i)
            ),
        )[0]["track_id"]

    observe(1)
    old = observe(2)
    tracker.reset()
    assert observe(1) is None
    assert observe(2) != old


def test_pipeline_release_attempts_gallery_even_if_model_cleanup_fails():
    cfg = Config.load("02_config/settings.orin-nx.yaml")
    camera = SimpleNamespace(source_epoch="test", stale_after=1.5)
    owner = VisionPipeline(camera, cfg.perception)
    calls = []

    def failure():
        calls.append("networks")
        raise RuntimeError("injected native cleanup failure")

    owner.networks = SimpleNamespace(close=failure)
    owner.gallery = SimpleNamespace(close=lambda: calls.append("gallery"))
    owner._selected = "old-target"
    owner._selected_last = {"old": True}
    with pytest.raises(ExceptionGroup, match="resource release"):
        owner.close()
    assert calls == ["networks", "gallery"]
    assert owner.gallery is None
    assert owner._selected is None and owner._selected_last is None


def test_duplicate_pipeline_load_preserves_current_resource_owner():
    cfg = Config.load("02_config/settings.orin-nx.yaml")
    owner = VisionPipeline(
        SimpleNamespace(source_epoch="test", stale_after=1.5), cfg.perception
    )
    closed = []
    owner.networks = SimpleNamespace(
        models={"face": object()}, close=lambda: closed.append(True)
    )
    with pytest.raises(RuntimeError, match="already loaded"):
        owner.load()
    assert not closed and "face" in owner.networks.models
