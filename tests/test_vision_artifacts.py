# SPDX-License-Identifier: Apache-2.0
"""Candidate preparation must preserve source identity and other writers' data."""

import hashlib
import io

import pytest

import prepare_vision_b1 as prepare


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    data, card = b"model-data", b"pinned model card"
    path = tmp_path / "05_runtime/models/b1/test/model.onnx"
    path.parent.mkdir(parents=True)
    (path.parent / "MODEL_CARD.md").write_bytes(card)
    entry = {
        "path": str(path.relative_to(tmp_path)),
        "filename": "model.onnx",
        "repository": "https://example.com/model",
        "revision": "fixed",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "model_card_sha256": hashlib.sha256(card).hexdigest(),
    }
    return entry, path, data


def test_check_only_rejects_a_changed_source_card(tmp_path, monkeypatch):
    entry, path, data = fixture(tmp_path, monkeypatch)
    path.write_bytes(data)
    (path.parent / "MODEL_CARD.md").write_bytes(b"changed license/provenance")
    with pytest.raises(ValueError, match="model card"):
        prepare.prepare("test", entry, check_only=True)
    assert path.read_bytes() == data


def test_checksum_failure_cannot_publish_a_partial_model(tmp_path, monkeypatch):
    entry, path, data = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        prepare.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"bad")
    )
    with pytest.raises(ValueError, match="checksum/size"):
        prepare.prepare("test", entry)
    assert not path.exists() and not path.with_suffix(".onnx.partial").exists()


def test_other_writer_final_artifact_is_not_overwritten(tmp_path, monkeypatch):
    entry, path, data = fixture(tmp_path, monkeypatch)

    def response(*args, **kwargs):
        path.write_bytes(b"other writer's model")
        return io.BytesIO(data)

    monkeypatch.setattr(prepare.urllib.request, "urlopen", response)
    with pytest.raises(FileExistsError):
        prepare.prepare("test", entry)
    assert path.read_bytes() == b"other writer's model"
    assert not path.with_suffix(".onnx.partial").exists()


def test_cached_model_cannot_bypass_a_changed_upstream_helper(tmp_path, monkeypatch):
    entry, path, data = fixture(tmp_path, monkeypatch)
    path.write_bytes(data)
    helper = tmp_path / "05_runtime/frameworks/b1/test/helper.py"
    helper.parent.mkdir(parents=True)
    helper.write_bytes(b"changed code")
    entry["prepost_source"] = {
        "files": {
            "helper.py": {
                "path": str(helper.relative_to(tmp_path)),
                "url": "https://example.com/helper.py",
                "sha256": hashlib.sha256(b"original code").hexdigest(),
            }
        }
    }
    with pytest.raises(ValueError, match="helper checksum"):
        prepare.prepare("test", entry, check_only=True)
    assert helper.read_bytes() == b"changed code" and path.read_bytes() == data
