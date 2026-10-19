# SPDX-License-Identifier: Apache-2.0
"""Observer hooks must preserve the helper and release their own CUDA events."""

from types import SimpleNamespace

import pytest

from benchmarks.vision_b1_timing import TensorRTTrace


def fixture():
    order, destroyed = [], []
    counter = iter(range(4))

    class Cuda:
        def cudaEventCreate(self):
            return (next(counter),)

        def cudaEventRecord(self, event, stream):
            order.append(event)
            return (None,)

        def cudaEventElapsedTime(self, start, end):
            return (0.125,)

        def cudaEventDestroy(self, event):
            destroyed.append(event)
            return (None,)

        def cudaStreamSynchronize(self, stream):
            order.append("sync")
            return (None,)

    common = SimpleNamespace(cudart=Cuda(), cuda_call=lambda result: result[0])

    def helper(inputs, outputs, stream, execute):
        order.append("h2d")
        execute()
        order.append("d2h")
        common.cuda_call(common.cudart.cudaStreamSynchronize(stream))
        return inputs + 1

    common._do_inference_base = helper

    class Native:
        def __init__(self):
            self.common = common

        def _select_device(self):
            pass

        def infer(self, value):
            return common._do_inference_base(
                value, None, 0, lambda: order.append("execute")
            )

    class Adapter:
        def __init__(self):
            self.model = Native()

        def infer(self, value):
            if value is None:
                raise ValueError("invalid input")
            return self.model.infer(value) * 2

    return Adapter(), common, order, destroyed


def test_timing_preserves_execution_order_outputs_and_restores_hooks():
    adapter, common, order, destroyed = fixture()
    cuda, helper = common.cudart, common._do_inference_base
    with TensorRTTrace(adapter) as observer:
        output, times = observer.call(4)
        assert output == 10
        assert order == [0, "h2d", 1, "execute", 2, "d2h", 3, "sync"]
        assert all(value >= 0 for value in times.values())
        assert times["adapter_ms"] == pytest.approx(
            times["preprocess_ms"] + times["native_sync_ms"] + times["postprocess_ms"]
        )
        assert common.cudart is cuda and common._do_inference_base is helper
        assert "infer" not in vars(adapter.model)
    assert sorted(destroyed) == list(range(4))


def test_failed_call_restores_inference_and_observer_remains_usable():
    adapter, common, order, destroyed = fixture()
    cuda, helper = common.cudart, common._do_inference_base
    with TensorRTTrace(adapter) as observer:
        with pytest.raises(ValueError, match="invalid input"):
            observer.call(None)
        assert common.cudart is cuda and common._do_inference_base is helper
        assert observer.call(2)[0] == 6
    assert len(destroyed) == 4
