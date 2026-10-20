# SPDX-License-Identifier: Apache-2.0
"""Frozen benchmark inputs must also work in an independent restored checkout."""

import json
import pytest

from app.config import Config
from benchmarks import vision_b1_support as support


@pytest.mark.parametrize("external_audio", [False, True])
def test_restored_baseline_rebases_only_owned_paths_and_preserves_bytes(
    tmp_path, monkeypatch, external_audio
):
    old = tmp_path / "previous/05_runtime"
    restored = tmp_path / "restored"
    recorded = Config.load("02_config/settings.orin-nx-b1.yaml").effective()
    recorded["backends"]["stack_root"] = str(old)
    recorded["audio"]["output_directory"] = (
        "/separate/audio" if external_audio else str(old / "runtime-data/audio")
    )
    recorded["actions"]["database"] = str(old / "runtime-data/actions.sqlite")
    path = tmp_path / "effective.json"
    path.write_text(json.dumps(recorded))
    frozen_bytes = path.read_bytes()
    monkeypatch.setattr(support, "ROOT", restored)

    cfg = support.relocated_config(path)

    assert cfg.backends.stack_root == str(restored / "05_runtime")
    assert cfg.audio.output_directory == (
        "/separate/audio"
        if external_audio
        else str(restored / "05_runtime/runtime-data/audio")
    )
    assert cfg.actions.database == str(
        restored / "05_runtime/runtime-data/actions.sqlite"
    )
    assert cfg.perception.identity_gallery == recorded["perception"]["identity_gallery"]
    for section in ("vision", "llm", "stt", "tts", "gestures"):
        assert cfg.effective()[section] == recorded[section]
    assert path.read_bytes() == frozen_bytes
