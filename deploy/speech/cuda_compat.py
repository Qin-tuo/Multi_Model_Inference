"""Process-local CUDA 13 namespace bridge for pinned voxedge 0.0.15a3."""
from __future__ import annotations

import importlib
import sys


def install_cuda_runtime_alias():
    try:
        return importlib.import_module("cuda.cudart")
    except ModuleNotFoundError:
        import cuda
        from cuda.bindings import runtime
        cuda.cudart = runtime
        sys.modules["cuda.cudart"] = runtime
        return runtime
