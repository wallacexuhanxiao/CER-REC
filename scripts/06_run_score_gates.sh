#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-beauty}"
PYTHON_BIN="${PYTHON:-python}"
if [[ "$dataset" != "beauty" ]]; then
  echo "Only beauty is wired for this stage." >&2
  exit 1
fi

"$PYTHON_BIN" -m src.evaluation.export_expert_features
"$PYTHON_BIN" -m src.evaluation.audit_repeat_targets
"$PYTHON_BIN" -m src.evaluation.audit_item_text_leakage
"$PYTHON_BIN" -m src.evaluation.audit_tail_buckets
"$PYTHON_BIN" -m src.evaluation.build_semantic_hard_negatives
"$PYTHON_BIN" -m src.evaluation.evaluate_hard_negatives
"$PYTHON_BIN" -m src.evaluation.calibrate_expert_scores
"$PYTHON_BIN" -m src.trainers.train_score_router

