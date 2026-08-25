#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:?Usage: serve_sglang.sh MODEL_PATH}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

python "$SCRIPT_DIR/patch_sglang_video.py"
exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --is-embedding \
  --enable-precise-embedding-interpolation
