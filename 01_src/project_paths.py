# SPDX-License-Identifier: Apache-2.0
"""Current project paths; runtime files live inside this checkout."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "01_src"
CONFIG_DIR = ROOT / "02_config"
DEFAULT_CONFIG = CONFIG_DIR / "settings.orin-nx.yaml"
MODEL_DIR = ROOT / "03_models"
WEB_DIR = ROOT / "04_web"
RUNTIME_DIR = ROOT / "05_runtime"


def artifact_path(value):
    """Expand explicit paths; Config.load resolves relative paths against ROOT."""
    return Path(value).expanduser()


def load_artifact_manifest(path):
    return json.loads(Path(path).read_text())
