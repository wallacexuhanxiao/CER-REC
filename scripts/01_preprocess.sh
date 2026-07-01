#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-beauty}"
PYTHON_BIN="${PYTHON:-python}"
if [[ "$dataset" != "beauty" ]]; then
  echo "Only beauty is wired for today's milestone." >&2
  exit 1
fi

"$PYTHON_BIN" -m src.data.preprocess_beauty \
  --raw-dir data/raw \
  --output-dir data/processed/beauty \
  --seed 2026 \
  --num-negatives 100 \
  --min-user-interactions 5 \
  --min-item-interactions 5

"$PYTHON_BIN" -m src.data.sanity_check --data-dir data/processed/beauty
