#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-beauty}"
PYTHON_BIN="${PYTHON:-python}"
if [[ "$dataset" != "beauty" ]]; then
  echo "Only beauty is wired for this stage." >&2
  exit 1
fi

"$PYTHON_BIN" -m src.evaluation.event_gate_analysis \
  --data-dir data/processed/beauty \
  --event-dir outputs/beauty/event_gate \
  --expert-dir outputs/beauty/expert_predictions \
  --score-router-summary outputs/beauty/score_routers/summary.json

