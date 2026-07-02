#!/usr/bin/env bash
set -euo pipefail
seed="${1:-2026}"
PYTHON_BIN="${PYTHON:-python}"
DATA_DIR="data/processed/beauty"
EMB_PATH="$DATA_DIR/item_semantic_embeddings.fp16.npy"
REPO_ROOT="outputs/beauty/seed${seed}"
TMP_ROOT="/root/autodl-tmp/cer-rec/beauty/seed${seed}"
CF_DIR="outputs/beauty/sasrec_seed${seed}"
SEM_DIR="outputs/beauty/semantic_sasrec_seed${seed}"
CF_CKPT="$CF_DIR/best.pt"
SEM_CKPT="$SEM_DIR/best.pt"
mkdir -p "$REPO_ROOT" "$TMP_ROOT"

if [[ ! -f "$CF_CKPT" ]]; then
  bash scripts/03_train_sasrec.sh beauty "$seed"
fi
if [[ ! -f "$SEM_CKPT" ]]; then
  bash scripts/04_train_semantic.sh beauty "$seed"
fi

"$PYTHON_BIN" -m src.evaluation.expert_complementarity \
  --data-dir "$DATA_DIR" \
  --cf-checkpoint "$CF_CKPT" \
  --semantic-checkpoint "$SEM_CKPT" \
  --semantic-embedding-path "$EMB_PATH" \
  --output-dir "$REPO_ROOT/expert_predictions" \
  --max-history-length 50 \
  --batch-size 4096

"$PYTHON_BIN" -m src.evaluation.export_expert_features \
  --data-dir "$DATA_DIR" \
  --cf-checkpoint "$CF_CKPT" \
  --semantic-checkpoint "$SEM_CKPT" \
  --semantic-embedding-path "$EMB_PATH" \
  --output-dir "$REPO_ROOT/expert_features" \
  --max-history-length 50 \
  --batch-size 4096 \
  --num-train-negatives 100 \
  --seed "$seed"

"$PYTHON_BIN" -m src.evaluation.calibrate_expert_scores \
  --feature-dir "$REPO_ROOT/expert_features" \
  --output-dir "$REPO_ROOT/calibration"

"$PYTHON_BIN" -m src.trainers.train_score_router \
  --feature-dir "$REPO_ROOT/expert_features" \
  --data-dir "$DATA_DIR" \
  --calibration-dir "$REPO_ROOT/calibration" \
  --output-dir "$REPO_ROOT/score_routers" \
  --batch-size 512 \
  --eval-batch-size 4096 \
  --learning-rate 0.001 \
  --weight-decay 0.00001 \
  --early-stop-patience 5 \
  --max-epochs 50

cf_temp=$("$PYTHON_BIN" - <<PY
import json
print(json.load(open('$REPO_ROOT/calibration/cf_temperature.json'))['temperature'])
PY
)
sem_temp=$("$PYTHON_BIN" - <<PY
import json
print(json.load(open('$REPO_ROOT/calibration/semantic_temperature.json'))['temperature'])
PY
)

"$PYTHON_BIN" -m src.trainers.train_event_gate \
  --data-dir "$DATA_DIR" \
  --cf-checkpoint "$CF_CKPT" \
  --semantic-checkpoint "$SEM_CKPT" \
  --semantic-embedding-path "$EMB_PATH" \
  --output-dir "$TMP_ROOT/event_gate_no_teacher_multiprefix5" \
  --route-mode learned \
  --lambda-route 0.0 \
  --max-prefixes-per-user 5 \
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
  --cf-temperature "$cf_temp" \
  --semantic-temperature "$sem_temp" \
  --seed "$seed"

"$PYTHON_BIN" -m src.evaluation.event_gate_analysis \
  --data-dir "$DATA_DIR" \
  --event-dir "$TMP_ROOT/event_gate_no_teacher_multiprefix5" \
  --expert-dir "$REPO_ROOT/expert_predictions" \
  --score-router-summary "$REPO_ROOT/score_routers/summary.json" \
  --learned-label EventGate-NoTeacher-MultiPrefix

"$PYTHON_BIN" -m src.evaluation.build_route_teacher \
  --data-dir "$DATA_DIR" \
  --cf-checkpoint "$CF_CKPT" \
  --semantic-checkpoint "$SEM_CKPT" \
  --semantic-embedding-path "$EMB_PATH" \
  --output-dir "$TMP_ROOT/route_teacher_multiprefix5" \
  --max-prefixes-per-user 5 \
  --train-negatives 32 \
  --batch-size 64 \
  --mask-chunk-size 1024 \
  --seed "$seed"

"$PYTHON_BIN" -m src.trainers.train_event_gate \
  --data-dir "$DATA_DIR" \
  --cf-checkpoint "$CF_CKPT" \
  --semantic-checkpoint "$SEM_CKPT" \
  --semantic-embedding-path "$EMB_PATH" \
  --output-dir "$TMP_ROOT/cer_rec_multiprefix5" \
  --route-mode learned \
  --route-teacher-dir "$TMP_ROOT/route_teacher_multiprefix5" \
  --lambda-route 0.3 \
  --max-prefixes-per-user 5 \
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
  --cf-temperature "$cf_temp" \
  --semantic-temperature "$sem_temp" \
  --seed "$seed"

"$PYTHON_BIN" -m src.evaluation.event_gate_analysis \
  --data-dir "$DATA_DIR" \
  --event-dir "$TMP_ROOT/cer_rec_multiprefix5" \
  --expert-dir "$REPO_ROOT/expert_predictions" \
  --score-router-summary "$REPO_ROOT/score_routers/summary.json" \
  --learned-label CER-Rec-MultiPrefix
