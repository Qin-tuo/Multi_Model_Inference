# SPDX-License-Identifier: Apache-2.0
"""Directory migration must preserve imports and frozen artifact identity."""

import hashlib
import json
import subprocess
import sys

import pytest

from project_paths import ROOT, repo_path


def test_frozen_baseline_locks_resolve_without_rewriting_records():
    def historical(name):
        return subprocess.check_output(
            ["git", "show", "before-mainline-consolidation-20261020:" + str(repo_path(name).relative_to(ROOT))],
            cwd=ROOT,
        )

    protocol = json.loads(historical("benchmarks/manifests/b0-sw-v1.json"))
    for name, digest in protocol["locks"].items():
        # Historical evidence belongs to its preserved source revision. Current
        # manifests can evolve without rewriting the old benchmark's locks.
        saved = historical(name)
        assert hashlib.sha256(saved).hexdigest() == digest
    assert (
        hashlib.sha256(historical(protocol["input_manifest"])).hexdigest()
        == protocol["input_manifest_sha256"]
    )


def test_source_and_data_locations_are_unambiguous():
    assert repo_path("app/config.py") == ROOT / "01_src/app/config.py"
    assert (
        repo_path("benchmarks/vision_modules.py")
        == ROOT / "01_src/benchmarks/vision_modules.py"
    )
    assert (
        repo_path("benchmarks/results/reachy-b0-sw.json")
        == ROOT / "benchmarks/results/b0-sw.json"
    )
    assert repo_path("02_config/settings.yaml") == ROOT / "02_config/settings.yaml"
    for name in ("../outside", "/etc/passwd"):
        with pytest.raises(ValueError):
            repo_path(name)


def test_public_cli_and_benchmark_entry_work_outside_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "config"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    assert json.loads(result.stdout)["web"]["port"] == 8090
    subprocess.run(
        [sys.executable, str(repo_path("benchmarks/vision_modules.py")), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
