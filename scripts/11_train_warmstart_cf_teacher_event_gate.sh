#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python}"
TEACHER_DIR="${TEACHER_DIR:-/root/autodl-tmp/cer-rec/beauty/route_teacher}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/cer-rec/beauty/event_gate_cf_teacher_warmstart}"
"$PYTHON_BIN" -m src.trainers.train_event_gate \
  --data-dir data/processed/beauty \
  --cf-checkpoint outputs/beauty/sasrec_seed2026/best.pt \
  --semantic-checkpoint outputs/beauty/semantic_sasrec_seed2026/best.pt \
  --semantic-embedding-path data/processed/beauty/item_semantic_embeddings.fp16.npy \
  --output-dir "$OUTPUT_DIR" \
  --route-mode learned \
  --route-teacher-dir "$TEACHER_DIR" \
  --init-checkpoint outputs/beauty/event_gate/learned/best.pt \
  --lambda-route 0.3 \
  --train-route-gate-only \
  --freeze-event-relation \
  --train-negatives 32 \
  --batch-size 128 \
  --eval-batch-size 512 \
  --relation-hidden-dim 64 \
  --route-hidden-dim 128 \
  --learning-rate 0.0001 \
  --weight-decay 0.00001 \
  --dropout 0.1 \
  --max-epochs 30 \
  --early-stop-patience 8 \
  --candidate-chunk-size 16 \
  --seed 2026
"$PYTHON_BIN" -m src.evaluation.event_gate_analysis \
  --data-dir data/processed/beauty \
  --event-dir "$OUTPUT_DIR" \
  --expert-dir outputs/beauty/expert_predictions \
  --score-router-summary outputs/beauty/score_routers/summary.json \
  --learned-label EventGate-CFTeacher-WarmStart
