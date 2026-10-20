# SPDX-License-Identifier: Apache-2.0
"""A new runtime revision must preserve old inputs and reject changed evidence."""

import json
import subprocess
import numpy as np
import pytest

from app.config import Config
from benchmarks import vision_b1_support as support
from deploy import runtime
import freeze_b1_inputs as freezing


def test_input_revision_reuses_identical_pixels_without_reference_experiment(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(freezing, "ROOT", tmp_path)
    source = tmp_path / "05_runtime/old/fixed.npy"
    source.parent.mkdir(parents=True)
    image = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
    np.save(source, image, allow_pickle=False)
    original = source.read_bytes()
    names = ["public-person", "public-person-dimmed", "bus", "bus-dimmed", "blank"]
    parent = tmp_path / "parent.json"
    parent.write_text(
        json.dumps(
            {
                "inputs": {
                    name: {
                        "path": str(source.relative_to(tmp_path)),
                        "sha256": runtime.sha256(source),
                    }
                    for name in names
                }
            }
        )
    )
    output = tmp_path / "05_runtime/new"
    output.mkdir()
    inputs, images = freezing.input_images(output, parent)
    assert source.read_bytes() == original
    assert set(inputs) == set(names)
    for name in names:
        assert np.array_equal(images[name], image)
        assert (tmp_path / inputs[name]["path"]).read_bytes() == original
    source.write_bytes(b"changed input")
    failed = tmp_path / "05_runtime/failed"
    failed.mkdir()
    with pytest.raises(ValueError, match="missing or changed"):
        freezing.input_images(failed, parent)
    assert not list(failed.iterdir())


def test_explicit_protocol_checks_its_own_runtime_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(support, "ROOT", tmp_path)
    code = tmp_path / "runtime.py"
    code.write_text("revision one\n")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(Config.load("02_config/settings.orin-nx-b1.yaml").effective())
    )
    protocol = tmp_path / "revision.json"
    protocol.write_text(
        json.dumps(
            {
                "locked_files": {"runtime.py": runtime.sha256(code)},
                "inputs": {},
                "effective_config": "config.json",
            }
        )
    )
    loaded, cfg = support.validate(protocol)
    assert (
        loaded["effective_config"] == "config.json"
        and cfg.perception.backend == "tensorrt_b1"
    )
    code.write_text("revision two\n")
    with pytest.raises(ValueError, match="frozen file changed"):
        support.validate(protocol)


def test_freeze_allows_unrelated_drafts_but_rejects_uncommitted_sources(
    tmp_path, monkeypatch
):
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True
        )

    git("init")
    (tmp_path / "01_src").mkdir()
    source = tmp_path / "01_src/runtime.py"
    source.write_text("original\n")
    (tmp_path / "README.md").write_text("original\n")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    monkeypatch.setattr(freezing, "ROOT", tmp_path)
    (tmp_path / "README.md").write_text("unrelated user edit\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/draft.md").write_text("user draft\n")
    (tmp_path / "01_src/layout.md").write_text("deployment prose only\n")
    assert freezing.uncommitted_sources() == []
    source.write_text("changed\n")
    (tmp_path / "01_src/new.py").write_text("untracked implementation\n")
    assert freezing.uncommitted_sources() == ["01_src/new.py", "01_src/runtime.py"]
