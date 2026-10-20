# SPDX-License-Identifier: Apache-2.0
"""A frozen measurement must remain usable beside unrelated runtime edits."""

from contextlib import contextmanager
import json
import sys

import pytest

from app.config import Config
from benchmarks import vision_modules as modules
from benchmarks import vision_protocol as support
from deploy import runtime


def test_historical_lock_validation_does_not_parse_old_runtime_config(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"perception": {"backend": "tensorrt_b1"}}))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "locked_files": {"config.json": runtime.sha256(config)},
        "inputs": {},
        "effective_config": "config.json",
    }))
    frozen = config.read_bytes()
    assert support.validate_locks(protocol, source_root=tmp_path)["effective_config"] == "config.json"
    assert config.read_bytes() == frozen
    config.write_text("{}")
    with pytest.raises(ValueError, match="frozen file changed"):
        support.validate_locks(protocol, source_root=tmp_path)


@pytest.mark.parametrize("changed", ["runtime.py", "input.npy"])
def test_selected_checkout_keeps_code_and_input_guards(tmp_path, monkeypatch, changed):
    current, frozen = tmp_path / "current", tmp_path / "frozen"
    config = Config.load("02_config/settings.orin-nx.yaml").effective()
    for root in (current, frozen):
        root.mkdir()
        (root / "runtime.py").write_text("frozen implementation\n")
        (root / "input.npy").write_bytes(b"frozen input")
        (root / "config.json").write_text(json.dumps(config))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "locked_files": {"runtime.py": runtime.sha256(frozen / "runtime.py")},
        "inputs": {"fixed": {
            "path": "input.npy", "sha256": runtime.sha256(frozen / "input.npy")
        }},
        "effective_config": "config.json",
    }))
    monkeypatch.setattr(support, "ROOT", current)
    (current / changed).write_bytes(b"active development")

    with pytest.raises(ValueError, match="changed"):
        support.validate(protocol)
    _, selected = support.validate(protocol, source_root=frozen)
    assert selected.backends.stack_root == str(frozen / "05_runtime")
    assert (current / changed).read_bytes() == b"active development"

    (frozen / changed).write_bytes(b"changed frozen checkout")
    with pytest.raises(ValueError, match="changed"):
        support.validate(protocol, source_root=frozen)


@pytest.mark.parametrize("historical", [False, True])
def test_coordinator_dispatches_frozen_worker_and_keeps_service_owner(
    tmp_path, monkeypatch, historical
):
    frozen = tmp_path / "frozen"
    worker = frozen / "01_src/benchmarks/vision_modules.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# frozen worker\n")
    if historical:
        worker = worker.with_name("vision_b1_modules.py")
        worker.write_text("# ten-model historical worker; sibling is B0\n")
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n")
    output = tmp_path / "measurement"
    production_config = object()
    observed = {}

    def validate_locks(path, *, source_root):
        assert path == protocol
        assert source_root == frozen
        return {}

    @contextmanager
    def exclusive_build(cfg, folder):
        assert cfg is production_config
        assert folder == output
        observed["service_owner"] = modules.services.ROOT
        yield

    def run_worker(argv, log, seconds):
        assert argv[1] == str(worker)
        assert seconds == 300
        observed["worker"] = argv[1]
        folder = output / "detection_segmentation"
        runtime.write_json(folder / "report.json", {"status": "MEASURED"})
        return 0

    monkeypatch.setattr(modules, "validate_locks", validate_locks)
    monkeypatch.setattr(modules, "environment", lambda out, **kw: None)
    monkeypatch.setattr(modules, "exclusive_build", exclusive_build)
    monkeypatch.setattr(modules, "run_worker", run_worker)
    monkeypatch.setattr(modules, "summarize", lambda out: None)
    monkeypatch.setattr(modules, "ROOT", tmp_path)
    monkeypatch.setattr(modules.services, "ROOT", tmp_path)
    monkeypatch.setattr(modules.runtime, "config", lambda path=None: production_config)
    monkeypatch.setattr(modules.services, "profile_path", lambda cfg: tmp_path / "live.json")
    monkeypatch.setattr(sys, "argv", [
        "vision_modules.py", "--output", str(output),
        "--source-root", str(frozen), "--protocol", str(protocol),
        "--role", "detection_segmentation",
    ])

    modules.main()

    report = json.loads((output / "report.json").read_text())
    assert report["runtime_source_root"] == str(frozen)
    assert report["worker_source_sha256"] == runtime.sha256(worker)
    assert observed["worker"] == str(worker)
    assert observed["service_owner"] == modules.ROOT
    assert report["status"] == "MEASURED_MODULES_ONLY_PIPELINE_PENDING"


@pytest.mark.parametrize("trace_exists", [False, True])
def test_nsight_only_wraps_worker_and_requires_its_report(
    tmp_path, monkeypatch, trace_exists
):
    from pathlib import Path

    script = tmp_path / "01_src/benchmarks/vision_modules.py"
    script.parent.mkdir(parents=True)
    script.write_text("# worker\n")
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n")
    output = tmp_path / "capture"
    calls = []
    cfg = object()

    @contextmanager
    def exclusive(actual, folder, *, allow_failed_stop):
        assert actual is cfg and allow_failed_stop
        calls.append("stop")
        try:
            yield
        finally:
            calls.append("restore")

    def run_worker(argv, log, seconds):
        assert calls == ["stop"]
        assert argv[:2] == ["/fake/nsys", "profile"]
        assert str(script) in argv and "--worker" in argv and "--nvtx" in argv
        folder = Path(argv[argv.index("--output") + 1])
        runtime.write_json(folder / "report.json", {"status": "MEASURED"})
        if trace_exists:
            (folder / "timeline.nsys-rep").write_bytes(b"fake trace")
        calls.append("profile-worker")
        return 0

    monkeypatch.setattr(modules, "ROOT", tmp_path)
    monkeypatch.setattr(modules, "validate_locks", lambda *a, **kw: {})
    monkeypatch.setattr(modules, "environment", lambda *a, **kw: None)
    monkeypatch.setattr(modules, "profiling_tools", lambda: {"nsys": "/fake/nsys"})
    monkeypatch.setattr(modules, "exclusive_build", exclusive)
    monkeypatch.setattr(modules, "run_worker", run_worker)
    monkeypatch.setattr(modules, "summarize", lambda out: None)
    monkeypatch.setattr(modules.runtime, "config", lambda *a: cfg)
    monkeypatch.setattr(modules.services, "profile_path", lambda _: tmp_path / "live.json")
    monkeypatch.setattr(sys, "argv", [
        "vision_modules.py", "--nsys", "--source-root", str(tmp_path),
        "--protocol", str(protocol), "--role", "detection_segmentation",
        "--output", str(output),
    ])
    if trace_exists:
        modules.main()
    else:
        with pytest.raises(RuntimeError, match="did not produce"):
            modules.main()
    assert calls == ["stop", "profile-worker", "restore"]
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == (
        "PROFILED_MODULES_ONLY_PIPELINE_PENDING" if trace_exists else "FAILED"
    )
