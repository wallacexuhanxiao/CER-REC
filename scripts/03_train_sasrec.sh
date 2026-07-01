#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-beauty}"
seed="${2:-2026}"
PYTHON_BIN="${PYTHON:-python}"
if [[ "$dataset" != "beauty" ]]; then
  echo "Only beauty is wired for today's milestone." >&2
  exit 1
fi

"$PYTHON_BIN" -m src.trainers.train_sasrec \
  --data-dir data/processed/beauty \
  --output-dir "outputs/beauty/sasrec_seed${seed}" \
  --seed "$seed" \
  --hidden-dim 64 \
  --max-history-length 50 \
  --num-layers 2 \
  --num-heads 2 \
  --dropout 0.2 \
  --batch-size 256 \
  --learning-rate 0.001 \
  --weight-decay 0.00001 \
  --train-negatives 1 \
  --early-stop-patience 10
