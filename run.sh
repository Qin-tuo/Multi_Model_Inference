#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PYTHON="$PROJECT_DIR/05_runtime/environments/edge-assistant/bin/python"
case "${1:-}" in
    llama-bench|llama-cli)
        PROJECT_TOOL="$1"
        shift
        export LD_LIBRARY_PATH="$PROJECT_DIR/05_runtime/builds/llama/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        exec "$PROJECT_DIR/05_runtime/builds/llama/bin/$PROJECT_TOOL" "$@"
        ;;
esac
if [[ ! -x "$PROJECT_PYTHON" ]]; then
    echo "缺少项目运行环境：请先运行 scripts/import_runtime.py 或恢复运行依赖包。" >&2
    exit 1
fi
exec "$PROJECT_PYTHON" "$PROJECT_DIR/main.py" "$@"
