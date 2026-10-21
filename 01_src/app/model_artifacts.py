# SPDX-License-Identifier: AGPL-3.0-only
"""校验并加载本地模型工件与辅助代码，阻止同名文件被悄悄替换。"""

import hashlib
import importlib.util
from pathlib import Path


def load_source(path, expected_sha, name):
    # 核对辅助代码的 SHA256，确保运行的是清单锁定的版本。
    """核对辅助源码的 SHA256 后动态导入，返回指定文件对应的模块对象。

    校验失败直接抛错，调用方只会使用清单锁定版本的几何变换等辅助实现。
    """
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError(f"Source hash mismatch: {path.name}")
    # 从指定文件动态导入几何裁剪、关键点解码等辅助实现。
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_file(path, expected):
    # 校验模型或引擎文件内容；文件名相同并不代表权重内容相同。
    """验证模型或引擎文件的 SHA256，成功时返回 Path，内容不符时拒绝使用。"""
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Model artifact hash mismatch: {path.name}")
    return path
