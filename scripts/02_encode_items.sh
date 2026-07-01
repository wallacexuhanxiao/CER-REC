#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-beauty}"
PYTHON_BIN="${PYTHON:-python}"
if [[ "$dataset" != "beauty" ]]; then
  echo "Only beauty is wired for this stage." >&2
  exit 1
fi

"$PYTHON_BIN" -m src.embeddings.encode_qwen_items \
  --processed-dir data/processed/beauty \
  --model Qwen/Qwen3-Embedding-0.6B \
  --batch-size 64 \
  --max-text-tokens 256

