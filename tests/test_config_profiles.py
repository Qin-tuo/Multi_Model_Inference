# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import pytest
from app.config import Config


def test_default_profile_is_local_4b_without_robot_or_rag(monkeypatch):
    monkeypatch.delenv("EDGE_ASSISTANT_CONFIG", raising=False)
    config = Config.load()
    assert config.llm.model == "assistant-vlm"
    assert config.llm.base_url == "http://127.0.0.1:8088"
    assert config.llm.context_tokens == 2048
    assert config.llm.max_tokens == 128
    assert config.llm.chat_template_kwargs == {"enable_thinking": False}
    assert config.vision.frames == 1
    assert not hasattr(config, "reachy") and not hasattr(config, "rag")
    assert config.face.enabled
    assert config.vision.backend == "realsense"
    assert config.perception.model_manifest == "03_models/vision-runtime.json"


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

    base = yaml.safe_load(Path("02_config/settings.orin-nx.yaml").read_text())
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
        "EDGE_ASSISTANT_CONFIG", "02_config/settings.orin-nx.yaml"
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


def test_shell_python_and_service_defaults_match_outside_repository(
    monkeypatch, tmp_path
):
    from project_paths import ROOT, DEFAULT_CONFIG
    from deploy import runtime
    from app.vision_models import VisionModels

    monkeypatch.delenv("EDGE_ASSISTANT_CONFIG", raising=False)
    expected = Config.load(str(DEFAULT_CONFIG)).effective()
    assert runtime.config().effective() == expected
    assert VisionModels().manifest["model_set"] == "B2-standard-2d"
    for command in (
        [str(ROOT / "run.sh"), "config"],
        [sys.executable, str(ROOT / "main.py"), "config"],
    ):
        value = subprocess.check_output(command, cwd=tmp_path, text=True)
        assert json.loads(value) == expected


def test_explicit_profile_overrides_environment_without_changing_defaults(
    monkeypatch, tmp_path
):
    import yaml
    from project_paths import DEFAULT_CONFIG

    base = yaml.safe_load(DEFAULT_CONFIG.read_text())
    environment, explicit = tmp_path / "environment.yaml", tmp_path / "explicit.yaml"
    base["llm"]["max_tokens"] = 64
    environment.write_text(yaml.safe_dump(base))
    base["llm"]["max_tokens"] = 96
    explicit.write_text(yaml.safe_dump(base))
    monkeypatch.setenv("EDGE_ASSISTANT_CONFIG", str(environment))
    assert Config.load().llm.max_tokens == 64
    assert Config.load(str(explicit)).llm.max_tokens == 96
