#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-beauty}"
PYTHON_BIN="${PYTHON:-python}"
if [[ "$dataset" != "beauty" ]]; then
  echo "Only beauty is wired for this stage." >&2
  exit 1
fi

"$PYTHON_BIN" -m src.evaluation.expert_complementarity \
  --data-dir data/processed/beauty \
  --cf-checkpoint outputs/beauty/sasrec_seed2026/best.pt \
  --semantic-checkpoint outputs/beauty/semantic_sasrec_seed2026/best.pt \
  --semantic-embedding-path data/processed/beauty/item_semantic_embeddings.fp16.npy \
  --output-dir outputs/beauty/expert_predictions \
  --max-history-length 50 \
  --batch-size 4096

