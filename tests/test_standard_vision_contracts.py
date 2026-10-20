# SPDX-License-Identifier: Apache-2.0
"""A replacement must preserve gestures and isolate incompatible face vectors."""

from copy import deepcopy

import numpy as np
import pytest

from app.config import GestureConfig
from app.gestures import GestureRecognizer
from app.identity_gallery import IdentityGallery
from app.vision_models import contract_hash
from test_gesture_rules import hand, scene


@pytest.mark.parametrize(
    "positions, expected",
    [([0, 50, 100, 50, 0, 50, 100], "wave"), ([0] * 8, "stop")],
)
def test_two_dimensional_hand_coordinates_preserve_temporal_policy(positions, expected):
    observed = []
    for dimensions in (2, 3):
        recognizer = GestureRecognizer(GestureConfig())
        events = []
        for i, x in enumerate(positions):
            value = hand(x=x)
            value["landmarks"] = [point[:dimensions] for point in value["landmarks"]]
            events += recognizer.update(
                scene(i, 1000 + i * 0.25, [value]), now=1000 + i * 0.25
            )
        observed.append(
            [(e["kind"], e["first_frame_id"], e["frame_id"]) for e in events]
        )
    assert observed[0] == observed[1]
    assert [kind for kind, *_ in observed[0]] == [expected]


def test_new_face_dimension_keeps_old_labels_and_refuses_cross_space_matching(tmp_path):
    path = tmp_path / "identities.sqlite"
    old_vector = np.eye(1, 128, dtype=np.float32)[0]
    new_vector = np.eye(1, 512, dtype=np.float32)[0]
    old = IdentityGallery(path)
    old.enroll("old-label", "sface:old-hash", old_vector)
    old.close()
    replacement = IdentityGallery(path, dimensions=512, threshold=0.5)
    try:
        assert replacement.match("arcface:new-hash", new_vector)["status"] == "unknown"
        with pytest.raises(ValueError, match="embedding"):
            replacement.enroll("wrong-size", "arcface:new-hash", old_vector)
        replacement.enroll("new-label", "arcface:new-hash", new_vector)
        assert replacement.match("arcface:new-hash", new_vector)["label"] == "new-label"
        assert {row["label"] for row in replacement.list()} == {
            "old-label",
            "new-label",
        }
        with pytest.raises(ValueError, match="embedding"):
            replacement.match("sface:old-hash", new_vector)
    finally:
        replacement.close()
    restored = IdentityGallery(path)
    try:
        assert restored.match("sface:old-hash", old_vector)["label"] == "old-label"
    finally:
        restored.close()


def test_new_adapter_hash_covers_feature_dimensions_and_matching_policy():
    entry = {
        "sha256": "model-hash",
        "adapter": "arcface_mobilenet_v1",
        "gallery_policy": {"dimensions": 512, "threshold": 0.5, "margin": 0.05},
    }
    original = contract_hash(entry)
    for key, value in (("dimensions", 128), ("threshold", 0.363)):
        changed = deepcopy(entry)
        changed["gallery_policy"][key] = value
        assert contract_hash(changed) != original
