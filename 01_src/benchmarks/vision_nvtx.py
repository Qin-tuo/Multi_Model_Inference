# SPDX-License-Identifier: Apache-2.0
"""Optional NVTX annotations around production functions in a measurement worker."""

from contextlib import contextmanager, ExitStack
from functools import wraps
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
import shutil
import sys

from project_paths import ROOT


def load_nvtx():
    overlay = ROOT / "05_runtime/builds/profiling-python"
    if not (overlay / "nvtx").is_dir():
        raise RuntimeError("Missing NVTX profiling overlay; install requirements-profiling.lock")
    if str(overlay) not in sys.path:
        sys.path.insert(0, str(overlay))
    module = import_module("nvtx")
    if not Path(module.__file__).resolve().is_relative_to(overlay.resolve()):
        raise RuntimeError("NVTX was loaded outside the dedicated profiling overlay")
    return module


def profiling_tools():
    nsys = shutil.which("nsys")
    if nsys is None:
        raise RuntimeError("nsys is not on PATH")
    load_nvtx()
    return {"nsys": nsys, "nvtx_version": version("nvtx")}


def profile_command(worker_argv, folder, nsys):
    return [
        nsys, "profile", "--trace=cuda,nvtx,osrt", "--sample=none",
        "--cpuctxsw=process-tree", "--wait=primary",
        "--output=" + str(folder / "timeline"),
        *worker_argv, "--nvtx",
    ]


def _wrap(owner, name, label, annotate, stack):
    owned = name in vars(owner)
    original = getattr(owner, name)

    @wraps(original)
    def call(*args, **kwargs):
        with annotate(label):
            return original(*args, **kwargs)

    def restore():
        if owned:
            setattr(owner, name, original)
        else:
            delattr(owner, name)

    setattr(owner, name, call)
    stack.callback(restore)


@contextmanager
def model_annotations(adapter, role):
    nvtx = load_nvtx()

    def annotate(message):
        return nvtx.annotate(message=message, domain="vision-inference")

    from app import tensorrt_runner

    with ExitStack() as stack:
        _wrap(adapter, "infer", "adapter_total", annotate, stack)
        _wrap(adapter.model, "infer", "native_sync", annotate, stack)
        _wrap(tensorrt_runner, "validate_inputs", "input_validation", annotate, stack)
        _wrap(adapter.model.common, "_do_inference_base", "runtime_helper", annotate, stack)
        if role == "detection_segmentation":
            from app import segmentation

            _wrap(segmentation, "prepare_segmentation", "preprocess", annotate, stack)
            _wrap(segmentation, "decode_segmentation", "postprocess", annotate, stack)
            _wrap(segmentation, "native_masks", "mask_decode", annotate, stack)
        yield annotate
