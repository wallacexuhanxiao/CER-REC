#!/usr/bin/env bash
set -euo pipefail

dataset="${1:?dataset name required, e.g. fashion}"
review_path="${2:?review path required}"
metadata_path="${3:?metadata path required}"
item_field="${4:-parent_asin}"
seed="${5:-2026}"
PYTHON_BIN="${PYTHON:-python}"
processed_dir="data/processed/${dataset}"

"$PYTHON_BIN" -m src.data.preprocess_amazon \
  --input-path "$review_path" \
  --input-format jsonl \
  --output-dir "$processed_dir" \
  --dataset-name "Amazon ${dataset}" \
  --seed "$seed" \
  --num-negatives 100 \
  --min-user-interactions 5 \
  --min-item-interactions 5 \
  --user-field user_id \
  --item-field "$item_field" \
  --timestamp-field timestamp \
  --rating-field rating

"$PYTHON_BIN" -m src.data.build_amazon_item_texts \
  --metadata-path "$metadata_path" \
  --metadata-format jsonl \
  --processed-dir "$processed_dir" \
  --item-id-field "$item_field" \
  --fallback-text "Unknown ${dataset} product" \
  --max-text-tokens 256

"$PYTHON_BIN" -m src.data.check_item_texts --processed-dir "$processed_dir"
