# SPDX-License-Identifier: Apache-2.0
"""校验图运行时依赖；只在显式创建视觉任务图时启用。"""

from importlib.metadata import version
import resource

from project_paths import MODEL_DIR, load_artifact_manifest


def require_holoscan_runtime():
    """验证 CUDA13 发行包及栈限制，避免混用另一套 SDK 或静默更换数值依赖。"""
    settings = load_artifact_manifest(MODEL_DIR / "runtime-components.json")["holoscan"]
    for distribution, expected in settings["runtime_versions"].items():
        if version(distribution) != expected:
            raise RuntimeError("Holoscan runtime dependency differs: " + distribution)
    soft, _ = resource.getrlimit(resource.RLIMIT_STACK)
    if soft != resource.RLIM_INFINITY and soft < settings["minimum_stack_bytes"]:
        raise RuntimeError("Holoscan requires a process stack limit of at least 32 MiB before graph startup")
    return settings
