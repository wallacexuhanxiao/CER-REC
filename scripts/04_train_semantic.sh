#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-beauty}"
seed="${2:-2026}"
PYTHON_BIN="${PYTHON:-python}"
if [[ "$dataset" != "beauty" ]]; then
  echo "Only beauty is wired for this stage." >&2
  exit 1
fi

"$PYTHON_BIN" -m src.trainers.train_semantic_sasrec \
  --data-dir data/processed/beauty \
  --embedding-path data/processed/beauty/item_semantic_embeddings.fp16.npy \
  --output-dir "outputs/beauty/semantic_sasrec_seed${seed}" \
  --seed "$seed" \
  --hidden-dim 64 \
  --projection-hidden-dim 256 \
  --max-history-length 50 \
  --num-layers 2 \
  --num-heads 2 \
  --dropout 0.2 \
  --batch-size 256 \
  --learning-rate 0.001 \
  --weight-decay 0.00001 \
  --early-stop-patience 10

