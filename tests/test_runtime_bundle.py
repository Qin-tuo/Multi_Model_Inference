# SPDX-License-Identifier: Apache-2.0
import io
import json
from pathlib import Path
import tarfile

import pytest

from app.config import Config
from project_paths import LEGACY_STACK_DIR, RUNTIME_DIR, ROOT, load_artifact_manifest
from runtime_bundle import digest, inventory, pack, unpack, verify


@pytest.mark.parametrize("previous", [LEGACY_STACK_DIR, ROOT / "11_runtime"])
def test_legacy_config_rebases_only_runtime_paths(tmp_path, previous):
    profile = tmp_path / "settings.json"
    recorded = Config.load().effective()
    recorded["backends"]["stack_root"] = str(previous)
    recorded["actions"]["database"] = str(
        previous / "runtime-data/reachy-actions/actions.sqlite"
    )
    recorded["audio"]["output_directory"] = str(previous / "runtime-data/reachy-audio")
    profile.write_text(json.dumps(recorded))
    cfg = Config.load(str(profile))
    assert cfg.backends.stack_root == str(RUNTIME_DIR)
    assert cfg.actions.database.startswith(str(RUNTIME_DIR))
    assert cfg.audio.output_directory.startswith(str(RUNTIME_DIR))
    assert cfg.llm.context_tokens == 2048 and cfg.llm.max_tokens == 128


def test_manifest_paths_rebase_without_rewriting_evidence(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "files": [{"path": str(LEGACY_STACK_DIR / "models/test.onnx")}],
                "url": "https://example.com/model",
                "sha256": "fixed",
            }
        )
    )
    before = digest(path)
    data = load_artifact_manifest(path)
    assert data["files"][0]["path"] == str(RUNTIME_DIR / "models/test.onnx")
    assert data["sha256"] == "fixed" and data["url"] == "https://example.com/model"
    assert digest(path) == before


def prepared(root):
    for name in ("reachy-headless", "speech"):
        env = root / "environments" / name
        (env / "bin").mkdir(parents=True)
        (env / "lib/python3.12/site-packages").mkdir(parents=True)
        (env / "bin/python3").symlink_to("/usr/bin/python3")
        (env / "bin/python").symlink_to("python3")
        (env / "lib64").symlink_to("lib")
        (env / "bin/activate").write_text('VIRTUAL_ENV="' + str(env) + '"\n')
        (env / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (root / "models").mkdir()
    (root / "models/test.bin").write_bytes(b"immutable-model")
    (root / "runtime-data").mkdir()
    (root / "runtime-data/private.db").write_bytes(b"private-state")


def test_archive_roundtrip_preserves_artifacts_and_excludes_private_state(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "restored"
    prepared(source)
    inventory(source)
    archive = tmp_path / "runtime.tar"
    result = pack(source, archive)
    unpack(archive, result["sha256"], destination)
    assert (destination / "models/test.bin").read_bytes() == b"immutable-model"
    assert not (destination / "runtime-data").exists()
    assert (
        str(destination)
        in (destination / "environments/speech/bin/activate").read_text()
    )
    assert (destination / "environments/speech/bin/python3").readlink() == Path(
        "/usr/bin/python3"
    )
    assert (destination / "environments/speech/bin/python").resolve() == Path(
        "/usr/bin/python3"
    ).resolve()
    verify(destination)
    (destination / "models/test.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="mismatch"):
        verify(destination)


def test_restore_rejects_bad_digest_and_path_escape(tmp_path):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("../escaped")
        info.size = 4
        tar.addfile(info, io.BytesIO(b"evil"))
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="SHA256"):
        unpack(archive, "0" * 64, destination)
    assert not destination.exists()
    with pytest.raises(tarfile.FilterError):
        unpack(archive, digest(archive), destination)
    assert not destination.exists() and not (tmp_path / "escaped").exists()


def test_inventory_rejects_links_back_to_external_stack(tmp_path):
    (tmp_path / "escape").symlink_to(LEGACY_STACK_DIR / "models")
    with pytest.raises(ValueError, match="external"):
        inventory(tmp_path)
