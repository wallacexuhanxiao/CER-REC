#!/usr/bin/env bash
set -euo pipefail

dataset="${1:?dataset required}"
seed="${2:-2026}"
PYTHON_BIN="${PYTHON:-python}"
DATA_DIR="data/processed/${dataset}"
EMB_PATH="$DATA_DIR/item_semantic_embeddings.fp16.npy"
REPO_ROOT="outputs/${dataset}/seed${seed}"
TMP_ROOT="/root/autodl-tmp/cer-rec/${dataset}/seed${seed}"
CF_DIR="outputs/${dataset}/sasrec_seed${seed}"
SEM_DIR="outputs/${dataset}/semantic_sasrec_seed${seed}"
CF_CKPT="$CF_DIR/best.pt"
SEM_CKPT="$SEM_DIR/best.pt"
mkdir -p "$REPO_ROOT" "$TMP_ROOT"

if [[ ! -f "$EMB_PATH" ]]; then
  "$PYTHON_BIN" -m src.embeddings.encode_qwen_items \
    --processed-dir "$DATA_DIR" \
    --model Qwen/Qwen3-Embedding-0.6B \
    --batch-size 64 \
    --max-text-tokens 256
fi

if [[ ! -f "$CF_CKPT" ]]; then
  "$PYTHON_BIN" -m src.trainers.train_sasrec \
    --data-dir "$DATA_DIR" \
    --output-dir "$CF_DIR" \
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
fi

if [[ ! -f "$SEM_CKPT" ]]; then
  "$PYTHON_BIN" -m src.trainers.train_semantic_sasrec \
    --data-dir "$DATA_DIR" \
    --embedding-path "$EMB_PATH" \
    --output-dir "$SEM_DIR" \
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

"$PYTHON_BIN" -m src.evaluation.build_route_teacher \
  --data-dir "$DATA_DIR" \
  --cf-checkpoint "$CF_CKPT" \
  --semantic-checkpoint "$SEM_CKPT" \
  --semantic-embedding-path "$EMB_PATH" \
  --output-dir "$TMP_ROOT/route_teacher_multiprefix5" \
  --max-prefixes-per-user 5 \
  --train-negatives 32 \
  --batch-size 32 \
  --max-history-length 50 \
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

"$PYTHON_BIN" -m src.evaluation.expert_recovery_metrics \
  --expert-dir "$REPO_ROOT/expert_predictions" \
  --model-name "CER-Rec-MultiPrefix" \
  --model-score-path "$TMP_ROOT/cer_rec_multiprefix5/learned/test_scores.npy" \
  --route-stats-path "$TMP_ROOT/cer_rec_multiprefix5/learned/route_stats.json" \
  --output-path "$TMP_ROOT/formal_summary.json"
