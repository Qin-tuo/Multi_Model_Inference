# SPDX-License-Identifier: AGPL-3.0-only
"""Hash-checked local artifacts; extracted from the original body adapter."""

import hashlib
import importlib.util
from pathlib import Path


def load_source(path, expected_sha, name):
    # 核对辅助代码的 SHA256，确保运行的是清单锁定的版本。
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError(f"Source hash mismatch: {path.name}")
    # 从指定文件动态导入模块，使用其中的 CUDA 缓冲区和推理辅助函数。
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_file(path, expected):
    # 校验模型或引擎文件内容；文件名相同并不代表权重内容相同。
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Model artifact hash mismatch: {path.name}")
    return path
