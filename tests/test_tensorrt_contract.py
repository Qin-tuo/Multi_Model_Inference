# SPDX-License-Identifier: Apache-2.0
"""Input rejection must occur before any work reaches a model's GPU buffers."""

import numpy as np
import pytest
import threading
from types import SimpleNamespace

from app.tensorrt_runner import TensorRTModel, validate_inputs


def test_stereo_inputs_require_both_views_and_correct_dtype():
    shape = (1, 3, 4, 6)
    specs = {name: (shape, np.dtype("float32")) for name in ("left", "right")}
    left = np.zeros(shape, np.float32)
    with pytest.raises(ValueError, match="names"):
        validate_inputs({"left": left}, specs)
    with pytest.raises(ValueError, match="shape/dtype"):
        validate_inputs({"left": left, "right": left.astype(np.float64)}, specs)
    with pytest.raises(ValueError, match="shape/dtype"):
        validate_inputs({"left": left, "right": left[:, :, :1]}, specs)


def test_noncontiguous_roi_is_preserved_and_nan_is_rejected():
    crop = np.arange(120, dtype=np.float32).reshape(1, 3, 4, 10)[..., ::2]
    spec = {"image": (crop.shape, crop.dtype)}
    actual = validate_inputs({"image": crop}, spec)["image"]
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, crop)
    crop[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        validate_inputs({"image": crop}, spec)


@pytest.mark.parametrize("failure", ["buffer", "stream"])
def test_suppressed_helper_release_failure_keeps_owner_and_disables_inference(failure):
    model = TensorRTModel.__new__(TensorRTModel)
    model._lock = threading.RLock()
    model._closing = False
    model.context = model.engine = model.runtime = object()
    model._select_device = lambda: None
    buffer = SimpleNamespace(
        _device_mem=SimpleNamespace(device_ptr=10),
        _host_mem=SimpleNamespace(_host_ptr=20),
    )
    model.inputs, model.outputs = [buffer], []
    model.common = SimpleNamespace(free_buffers=lambda *args: None)
    stream = SimpleNamespace(_stream=30, synchronize=lambda: None, free=lambda: None)
    model.stream = stream
    if failure == "stream":
        buffer._device_mem.device_ptr = buffer._host_mem._host_ptr = None
    with pytest.raises(RuntimeError, match="release is unconfirmed"):
        model.close()
    assert model.context is not None and model.stream is stream
    with pytest.raises(RuntimeError, match="closed or closing"):
        model.infer({})
    buffer._device_mem.device_ptr = buffer._host_mem._host_ptr = None
    stream._stream = None
    model.close()
    assert model.context is None and model.stream is None
    assert model.inputs == [] and model.outputs == []
