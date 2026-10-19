# SPDX-License-Identifier: Apache-2.0
"""Data ownership and bounded-state tests; no claims about model accuracy."""

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.identity_gallery import IdentityGallery
from app.config import Config
from app.perception_b1 import range_from_mask
from app.visual_tracks import VisualTracks
from app.web import Broadcaster, create_app


@pytest.mark.parametrize("field", ["body_enabled", "hands_enabled"])
def test_b1_rejects_unsupported_partial_pipeline_flags(field):
    cfg = Config.load("02_config/settings.orin-nx-b1.yaml")
    setattr(cfg.perception, field, False)
    with pytest.raises(ValueError, match="B1 requires"):
        cfg.validate()


def vector(index=0):
    result = np.zeros(128, np.float32)
    result[index] = 1
    return result


def test_identity_persistence_unknown_and_separate_model_spaces(tmp_path):
    path = tmp_path / "gallery.sqlite"
    gallery = IdentityGallery(path, capacity=2)
    try:
        assert gallery.match("sface-v1", vector())["status"] == "unknown"
        gallery.enroll("sample", "sface-v1", vector())
        assert gallery.match("sface-v1", vector())["label"] == "sample"
        assert gallery.match("sface-v1", vector(1))["status"] == "unknown"
        assert gallery.match("sface-v2", vector())["status"] == "unknown"
        gallery.enroll("similar", "sface-v1", vector())
        ambiguous = gallery.match("sface-v1", vector())
        assert ambiguous["status"] == "ambiguous" and ambiguous["label"] is None
        with pytest.raises(ValueError, match="full"):
            gallery.enroll("third", "sface-v1", vector(2))
        # Re-enrollment replaces a label atomically even at capacity.
        gallery.enroll("similar", "sface-v1", vector(1))
        assert gallery.match("sface-v1", vector())["label"] == "sample"
    finally:
        gallery.close()
    reopened = IdentityGallery(path)
    try:
        assert reopened.match("sface-v1", vector())["label"] == "sample"
        assert len(reopened.list()) == 2
        assert reopened.remove("sample")
        assert not reopened.remove("sample")
        assert reopened.match("sface-v1", vector())["status"] == "unknown"
    finally:
        reopened.close()


def frame(index, stamp=None, epoch="camera"):
    return SimpleNamespace(
        source_epoch=epoch,
        frame_id=index,
        received_monotonic=index * 0.1 if stamp is None else stamp,
    )


def row(x=0, category=0):
    return {"box_xyxy": [x, 0, x + 50, 100], "score": 0.9, "class_id": category}


def test_track_ids_keep_observed_geometry_and_expire_by_time_or_source():
    tracks = VisualTracks("entity", ttl=2)
    assert tracks.update([row()], frame(1))[0]["track_id"] is None
    confirmed = tracks.update([row(1)], frame(2))[0]
    key = confirmed["track_id"]
    assert key and confirmed["box_xyxy"] == row(1)["box_xyxy"]
    assert tracks.update([row(2)], frame(3))[0]["track_id"] == key
    assert tracks.update([], frame(4)) == []  # A prediction is not an observation.
    assert tracks.update([row(3)], frame(5))[0]["track_id"] == key
    with pytest.raises(ValueError, match="Out-of-order"):
        tracks.update([row()], frame(5))
    assert tracks.update([row()], frame(6, stamp=3))[0]["track_id"] is None
    fresh = tracks.update([row()], frame(7, stamp=3.1))[0]["track_id"]
    assert fresh and fresh != key
    assert (
        tracks.update([row()], frame(1, stamp=3.2, epoch="reconnected"))[0]["track_id"]
        is None
    )
    assert (
        tracks.update([row()], frame(2, stamp=3.3, epoch="reconnected"))[0]["track_id"]
        != fresh
    )


def test_tracking_class_separation_and_capacity():
    tracks = VisualTracks("entity", capacity=3)
    tracks.update([row()], frame(1))
    original = tracks.update([row()], frame(2))[0]["track_id"]
    changed = tracks.update([row(category=2)], frame(3))[0]
    assert changed["track_id"] != original
    for i in range(4, 20):
        tracks.update([row(i * 500 + n * 100) for n in range(8)], frame(i))
        assert len(tracks.tracker.tracked_objects) <= 3


@pytest.mark.parametrize("ambiguous", [False, True])
def test_appearance_recovery_requires_unique_compatible_feature(ambiguous):
    tracks = VisualTracks("entity", ttl=3)

    def featured(x):
        return dict(row(x), _embedding=vector(), _embedding_space="test-v1")

    for i in range(1, 4):
        observed = tracks.update(
            [featured(0), featured(100)] if ambiguous else [featured(0)], frame(i)
        )
    old_ids = {r["track_id"] for r in observed}
    for i in range(4, 9):
        tracks.update([], frame(i))
    tracks.update([featured(300)], frame(9))
    recovered = tracks.update([featured(300)], frame(10))[0]["track_id"]
    assert recovered
    assert (recovered in old_ids) is not ambiguous


def test_depth_uses_only_valid_registered_target_pixels():
    mask = np.zeros((10, 10), bool)
    mask[:5] = True
    depth = np.full((10, 10), 9.0, np.float32)
    depth[:5] = 1.5
    depth[0, :3] = [np.nan, 0, 20]
    value = SimpleNamespace(depth_m=depth, depth_valid=np.ones((10, 10), bool))
    result = range_from_mask(value, mask)
    assert result["meters"] == 1.5 and result["pixels"] == 47
    assert result["quantity"] == "aligned_native_depth_z_m"
    value.depth_valid[:] = False
    assert range_from_mask(value, mask)["reason"] == "insufficient_depth"
    with pytest.raises(ValueError, match="geometry"):
        range_from_mask(value, mask[:2])
    value.depth_m = None
    assert range_from_mask(value, mask)["reason"] == "depth_unavailable"


def test_vision_api_routes_validation_and_unsupported_backend():
    calls = []

    def enroll(target, label):
        if target == "stale":
            raise ValueError("Face observation expired")
        calls.append((target, label))
        return {"label": label}

    api = {
        "identities": lambda: [],
        "enroll": enroll,
        "remove": lambda label: label == "sample",
        "target": lambda key: {"track_id": key},
    }
    with TestClient(create_app(Broadcaster(), vision=api)) as client:
        assert client.get("/api/vision/identities").json() == []
        assert client.post(
            "/api/vision/identities", json={"target_id": "current", "label": "sample"}
        ).json() == {"label": "sample"}
        assert calls == [("current", "sample")]
        assert (
            client.post(
                "/api/vision/identities", json={"target_id": "stale", "label": "sample"}
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/vision/identities",
                json={"target_id": "current", "label": "x" * 65},
            ).status_code
            == 422
        )
        assert (
            client.request(
                "DELETE", "/api/vision/identities", json={"label": "sample"}
            ).json()
            is True
        )
        assert client.post("/api/vision/target", json={"track_id": None}).json() == {
            "track_id": None
        }
    with TestClient(create_app(Broadcaster())) as client:
        assert client.get("/api/vision/identities").status_code == 503
