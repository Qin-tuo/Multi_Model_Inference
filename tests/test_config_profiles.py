# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import pytest
from app.config import Config


def test_default_profile_is_local_4b_without_robot_or_rag(monkeypatch):
    monkeypatch.delenv("REACHY_ASSISTANT_CONFIG", raising=False)
    config = Config.load()
    assert config.llm.model == "assistant-vlm"
    assert config.llm.base_url == "http://127.0.0.1:8088"
    assert config.llm.context_tokens == 2048
    assert config.llm.max_tokens == 128
    assert config.llm.chat_template_kwargs == {"enable_thinking": False}
    assert config.vision.frames == 1
    assert not hasattr(config, "reachy") and not hasattr(config, "rag")
    assert config.face.enabled


@pytest.mark.parametrize(
    "content",
    [
        "unknown: {}",
        "llm: {modle: wrong}",
        "llm: {max_tokens: true}",
        "llm: {context_tokens: 128}",
        "vision: {frames: 3}",
        "llm: {base_url: 'https://example.com'}",
        "rag: {enabled: true}",
        "reachy: {enabled: true}",
        "audio: {echo_cancellation: 'false'}",
    ],
)
def test_bad_config_fails_instead_of_falling_back(tmp_path, content):
    # Supply a complete platform overlay, then introduce one invalid field.
    import yaml

    base = yaml.safe_load(Path("02_config/settings.orin-nx-headless.yaml").read_text())
    for key, value in yaml.safe_load(content).items():
        if isinstance(value, dict):
            base.setdefault(key, {}).update(value)
        else:
            base[key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(base))
    with pytest.raises(ValueError):
        Config.load(str(path))


def test_missing_explicit_profile_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.load(str(tmp_path / "missing.yaml"))


def test_environment_selects_profile(monkeypatch):
    monkeypatch.setenv(
        "REACHY_ASSISTANT_CONFIG", "02_config/settings.orin-nx-headless.yaml"
    )
    assert Config.load().llm.model == "assistant-vlm"


def test_cli_import_does_not_initialize_robot_memory_or_models():
    from project_paths import SOURCE_DIR

    code = f"import sys;sys.path.insert(0,{str(SOURCE_DIR)!r});import app.cli,json; print(json.dumps(sorted(sys.modules)))"
    loaded = json.loads(
        subprocess.check_output([sys.executable, "-c", code], text=True)
    )
    for forbidden in [
        "reachy_mini",
        "chromadb",
        "mem0",
        "pipecat",
        "faster_whisper",
        "kokoro_onnx",
    ]:
        assert not any(m == forbidden or m.startswith(forbidden + ".") for m in loaded)
