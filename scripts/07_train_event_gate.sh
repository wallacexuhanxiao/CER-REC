#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-beauty}"
PYTHON_BIN="${PYTHON:-python}"
if [[ "$dataset" != "beauty" ]]; then
  echo "Only beauty is wired for this stage." >&2
  exit 1
fi

"$PYTHON_BIN" -m src.trainers.train_event_gate \
  --data-dir data/processed/beauty \
  --cf-checkpoint outputs/beauty/sasrec_seed2026/best.pt \
  --semantic-checkpoint outputs/beauty/semantic_sasrec_seed2026/best.pt \
  --semantic-embedding-path data/processed/beauty/item_semantic_embeddings.fp16.npy \
  --output-dir outputs/beauty/event_gate \
  --route-mode both \
  --train-negatives 32 \
  --batch-size 128 \
  --eval-batch-size 512 \
  --relation-hidden-dim 64 \
  --route-hidden-dim 128 \
  --learning-rate 0.0003 \
  --weight-decay 0.00001 \
  --dropout 0.1 \
  --max-epochs 100 \
  --early-stop-patience 10 \
  --candidate-chunk-size 16 \
  --seed 2026

