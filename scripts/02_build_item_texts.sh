#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-beauty}"
PYTHON_BIN="${PYTHON:-python}"
if [[ "$dataset" != "beauty" ]]; then
  echo "Only beauty is wired for this stage." >&2
  exit 1
fi

"$PYTHON_BIN" -m src.data.build_item_texts \
  --raw-dir data/raw \
  --processed-dir data/processed/beauty \
  --max-text-tokens 256

"$PYTHON_BIN" -m src.data.check_item_texts --processed-dir data/processed/beauty

