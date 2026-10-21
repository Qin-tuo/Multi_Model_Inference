#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PYTHON="$PROJECT_DIR/05_runtime/environments/edge-assistant/bin/python"
export PYTHONDONTWRITEBYTECODE=1
if [[ ! -x "$PROJECT_PYTHON" ]]; then
    echo "缺少项目运行环境：请查看 docs/operations.md 中的运行依赖说明。" >&2
    exit 1
fi
# Config.load owns the shared default. Explicit --config and the inherited
# EDGE_ASSISTANT_CONFIG retain priority for both shell and Python entrypoints.
exec "$PROJECT_PYTHON" "$PROJECT_DIR/main.py" "$@"
