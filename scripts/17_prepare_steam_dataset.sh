#!/usr/bin/env bash
set -euo pipefail

seed="${1:-2026}"
PYTHON_BIN="${PYTHON:-python}"
RAW_ROOT="${RAW_ROOT:-/root/autodl-tmp/cer-rec/raw_datasets/steam}"
processed_dir="data/processed/steam"

"$PYTHON_BIN" -m src.data.preprocess_steam \
  --reviews-path "$RAW_ROOT/steam_reviews.json.gz" \
  --output-dir "$processed_dir" \
  --seed "$seed" \
  --num-negatives 100 \
  --min-user-interactions 5 \
  --min-item-interactions 5

"$PYTHON_BIN" -m src.data.build_amazon_item_texts \
  --metadata-path "$RAW_ROOT/steam_games.json.gz" \
  --metadata-format python \
  --processed-dir "$processed_dir" \
  --item-id-field id \
  --fallback-text "Unknown Steam game" \
  --max-text-tokens 256

"$PYTHON_BIN" -m src.data.check_item_texts --processed-dir "$processed_dir"
