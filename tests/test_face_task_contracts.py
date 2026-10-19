# SPDX-License-Identifier: Apache-2.0
"""Invalid face geometry/features must not become a confident identity result."""

import numpy as np
import pytest

from app.face_tasks import sface_transform, normalize_embedding, expression_scores


def test_degenerate_face_landmarks_cannot_produce_an_aligned_identity_crop():
    with pytest.raises(ValueError, match="landmarks"):
        sface_transform(np.zeros((5, 2), np.float32))
    with pytest.raises(ValueError, match="Degenerate"):
        sface_transform(np.column_stack((np.arange(5) * 10, np.zeros(5))))


@pytest.mark.parametrize("value", [0, np.nan, np.inf])
def test_invalid_embedding_does_not_become_an_unknown_or_matching_zero_vector(value):
    with pytest.raises(ValueError, match="embedding"):
        normalize_embedding(np.full(128, value, np.float32))


def test_expression_scores_stay_finite_for_large_logits_and_keep_public_labels():
    scores = expression_scores(
        np.array([-10000, -10000, -10000, 10000, -10000, -10000, -10000], np.float32)
    )
    assert scores["happy"] == 1 and sum(scores.values()) == 1
    assert "exaggeration" not in scores
