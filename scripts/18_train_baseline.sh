#!/usr/bin/env bash
set -euo pipefail

model="${1:?model required: gru4rec or bert4rec}"
seed="${2:-2026}"
PYTHON_BIN="${PYTHON:-python}"

case "$model" in
  gru4rec)
    "$PYTHON_BIN" -m src.trainers.train_gru4rec \
      --data-dir data/processed/beauty \
      --output-dir "outputs/beauty/gru4rec_seed${seed}" \
      --seed "$seed" \
      --hidden-dim 64 \
      --max-history-length 50 \
      --num-layers 1 \
      --dropout 0.2 \
      --batch-size 256 \
      --learning-rate 0.001 \
      --weight-decay 0.00001 \
      --train-negatives 1 \
      --early-stop-patience 10
    ;;
  bert4rec)
    "$PYTHON_BIN" -m src.trainers.train_bert4rec \
      --data-dir data/processed/beauty \
      --output-dir "outputs/beauty/bert4rec_seed${seed}" \
      --seed "$seed" \
      --hidden-dim 64 \
      --max-history-length 50 \
      --num-layers 2 \
      --num-heads 2 \
      --dropout 0.2 \
      --mask-prob 0.2 \
      --batch-size 256 \
      --learning-rate 0.001 \
      --weight-decay 0.00001 \
      --early-stop-patience 10
    ;;
  *)
    echo "Unknown baseline: $model" >&2
    exit 1
    ;;
esac
