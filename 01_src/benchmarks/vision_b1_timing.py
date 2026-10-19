# SPDX-License-Identifier: Apache-2.0
"""Measurement-only adapter/stream spans around the unchanged NVIDIA helper.

CUDA events measure intervals on the inference stream, including any host-submit
gaps. They are not isolated kernel timings, and must not be added to host wall
times. The application never installs these hooks. Compare plain/instrumented
calls on identical inputs to expose observer overhead and output differences.
"""

import time


class TensorRTTrace:
    def __init__(self, adapter):
        self.adapter, self.native = adapter, adapter.model
        self.common = self.native.common
        self.cuda, self.check = self.common.cudart, self.common.cuda_call
        self.events = []
        self.active = False
        self.native._select_device()
        try:
            for _ in range(4):
                self.events.append(self.check(self.cuda.cudaEventCreate()))
        except BaseException:
            self.close()
            raise

    def call(self, *args, **kwargs):
        if self.active or len(self.events) != 4:
            raise RuntimeError("Timing observer is busy or closed")
        self.active = True
        clock = time.perf_counter
        marks = {}
        original_infer = self.native.infer
        had_infer = "infer" in vars(self.native)
        original_helper = self.common._do_inference_base
        cuda, check, events = self.cuda, self.check, self.events
        observer = self

        class CudaProxy:
            def __getattr__(self, name):
                return getattr(cuda, name)

            def cudaStreamSynchronize(self, stream):
                check(cuda.cudaEventRecord(events[3], stream))
                result = cuda.cudaStreamSynchronize(stream)
                marks["sync_exit"] = clock()
                return result

        def helper(inputs, outputs, stream, execute):
            if "helper_enter" in marks:
                raise RuntimeError("Expected one native execution per adapter call")
            marks["helper_enter"] = clock()
            check(cuda.cudaEventRecord(events[0], stream))

            def measured_execute():
                check(cuda.cudaEventRecord(events[1], stream))
                result = execute()
                check(cuda.cudaEventRecord(events[2], stream))
                return result

            result = original_helper(inputs, outputs, stream, measured_execute)
            marks["helper_exit"] = clock()
            return result

        def native_infer(values):
            marks["native_enter"] = clock()
            result = original_infer(values)
            marks["native_exit"] = clock()
            return result

        try:
            self.native.infer = native_infer
            self.common.cudart = CudaProxy()
            self.common._do_inference_base = helper
            marks["adapter_enter"] = clock()
            value = self.adapter.infer(*args, **kwargs)
            marks["adapter_exit"] = clock()
            spans = {
                "adapter_ms": ("adapter_enter", "adapter_exit"),
                "preprocess_ms": ("adapter_enter", "native_enter"),
                "native_sync_ms": ("native_enter", "native_exit"),
                "postprocess_ms": ("native_exit", "adapter_exit"),
                "input_validation_host_copy_ms": ("native_enter", "helper_enter"),
                "output_host_copy_ms": ("sync_exit", "helper_exit"),
                "output_validation_ms": ("helper_exit", "native_exit"),
            }
            timings = {
                key: (marks[end] - marks[start]) * 1000
                for key, (start, end) in spans.items()
            }
            for i, label in enumerate(
                ("cuda_h2d_ms", "cuda_execution_ms", "cuda_d2h_ms")
            ):
                timings[label] = float(
                    check(cuda.cudaEventElapsedTime(events[i], events[i + 1]))
                )
            return value, timings
        finally:
            if had_infer:
                self.native.infer = original_infer
            else:
                del self.native.infer
            self.common.cudart = cuda
            self.common._do_inference_base = original_helper
            observer.active = False

    def close(self):
        errors = []
        while self.events:
            event = self.events.pop()
            try:
                self.check(self.cuda.cudaEventDestroy(event))
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("CUDA timing event release failed", errors)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
