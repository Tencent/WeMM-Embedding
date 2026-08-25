#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:?Usage: serve_vllm.sh MODEL_PATH}

exec vllm serve "$MODEL_PATH" \
  --runner pooling \
  --chat-template "$MODEL_PATH/embedding_chat_template.jinja"
