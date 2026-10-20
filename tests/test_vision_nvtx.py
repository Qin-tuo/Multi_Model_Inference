# SPDX-License-Identifier: Apache-2.0
"""Diagnostic ranges preserve calls/results and are removed on every exit path."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from app import segmentation, tensorrt_runner
from benchmarks import vision_nvtx as profiling


@pytest.mark.parametrize("fail", [False, True])
def test_annotations_preserve_output_and_restore_all_functions(monkeypatch, fail):
    events = []

    @contextmanager
    def annotate(message, domain):
        assert domain == "vision-inference"
        events.append(("push", message))
        try:
            yield
        finally:
            events.append(("pop", message))

    monkeypatch.setattr(profiling, "load_nvtx", lambda: SimpleNamespace(annotate=annotate))
    monkeypatch.setattr(segmentation, "prepare_segmentation", lambda x: ({"x": x}, {}))
    monkeypatch.setattr(tensorrt_runner, "validate_inputs", lambda x, specs: x)
    monkeypatch.setattr(segmentation, "native_masks", lambda x, _: x + 1)

    def decode(x, transform):
        if fail:
            raise ValueError("decode failed")
        return segmentation.native_masks(x, transform)

    monkeypatch.setattr(segmentation, "decode_segmentation", decode)

    class Native:
        def __init__(self):
            self.common = SimpleNamespace(_do_inference_base=lambda x: x["x"] * 2)

        def infer(self, x):
            return self.common._do_inference_base(tensorrt_runner.validate_inputs(x, {}))

    class Adapter:
        def __init__(self):
            self.model = Native()

        def infer(self, x):
            values, transform = segmentation.prepare_segmentation(x)
            return segmentation.decode_segmentation(self.model.infer(values), transform)

    adapter = Adapter()
    original = segmentation.prepare_segmentation
    original_helper = adapter.model.common._do_inference_base
    with profiling.model_annotations(adapter, "detection_segmentation") as marker:
        with marker("frame/public-person/plain/0"):
            if fail:
                with pytest.raises(ValueError, match="decode failed"):
                    adapter.infer(3)
            else:
                assert adapter.infer(3) == 7
    assert segmentation.prepare_segmentation is original
    assert segmentation.decode_segmentation is decode
    assert adapter.model.common._do_inference_base is original_helper
    assert "infer" not in vars(adapter)
    assert "infer" not in vars(adapter.model)
    stack = []
    for kind, message in events:
        if kind == "push":
            stack.append(message)
        else:
            assert stack.pop() == message
    assert stack == []
    assert {message for kind, message in events if kind == "push"} >= {
        "adapter_total", "preprocess", "native_sync", "postprocess",
        "input_validation", "runtime_helper",
    }


def test_nsight_launches_the_worker_with_nvtx(tmp_path):
    worker = ["python", "worker.py", "--worker", "detection_segmentation"]
    argv = profiling.profile_command(worker, tmp_path, "/usr/bin/nsys")
    assert argv[:2] == ["/usr/bin/nsys", "profile"]
    assert argv[-len(worker)-1:] == [*worker, "--nvtx"]
    assert "--output=" + str(tmp_path / "timeline") in argv
