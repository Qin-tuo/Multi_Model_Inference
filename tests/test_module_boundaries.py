# SPDX-License-Identifier: Apache-2.0
"""Keep reusable code independent of executable scripts and retired adapters."""

import ast
import json
import subprocess
import sys

from project_paths import ROOT, SOURCE_DIR


def test_formal_modules_do_not_import_executable_scripts():
    scripts = {path.stem for path in (ROOT / "scripts").glob("*.py")}
    violations = []
    for path in SOURCE_DIR.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = (
                [node.module]
                if isinstance(node, ast.ImportFrom) and node.module and not node.level
                else [value.name for value in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            for name in modules:
                if name.split(".")[0] in scripts:
                    violations.append((str(path.relative_to(ROOT)), node.lineno, name))
    assert violations == []


def test_current_pose_and_face_helpers_do_not_load_old_detector_or_pose():
    code = (
        f"import sys,json;sys.path.insert(0,{str(SOURCE_DIR)!r});"
        "import app.wholebody_pose2d,app.face_tasks;"
        "print(json.dumps(sorted(sys.modules)))"
    )
    loaded = json.loads(
        subprocess.check_output([sys.executable, "-c", code], text=True)
    )
    assert "app.body_detector" not in loaded
    assert "app.wholebody_pose" not in loaded


def test_current_frame_worker_has_no_legacy_model_owner():
    code = (
        f"import sys,json;sys.path.insert(0,{str(SOURCE_DIR)!r});"
        "import app.perception_b1;"
        "print(json.dumps(sorted(sys.modules)))"
    )
    loaded = json.loads(
        subprocess.check_output([sys.executable, "-c", code], text=True)
    )
    assert "app.perception" not in loaded
    assert "app.body_detector" not in loaded
