#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python}"
"$PYTHON_BIN" -m src.evaluation.build_route_teacher \
  --data-dir data/processed/beauty \
  --cf-checkpoint outputs/beauty/sasrec_seed2026/best.pt \
  --semantic-checkpoint outputs/beauty/semantic_sasrec_seed2026/best.pt \
  --semantic-embedding-path data/processed/beauty/item_semantic_embeddings.fp16.npy \
  --output-dir /root/autodl-tmp/cer-rec/beauty/route_teacher \
  --train-negatives 32 \
  --batch-size 64 \
  --mask-chunk-size 1024 \
  --seed 2026
