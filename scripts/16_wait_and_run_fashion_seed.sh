#!/usr/bin/env bash
set -euo pipefail

seed="${1:-2026}"
RAW_ROOT="${RAW_ROOT:-/root/autodl-tmp/cer-rec/raw_datasets/amazon_fashion_2023}"
review_path="$RAW_ROOT/Amazon_Fashion.jsonl.gz"
metadata_path="$RAW_ROOT/meta_Amazon_Fashion.jsonl.gz"

while true; do
  if [[ -f "$review_path" && -f "$metadata_path" ]] && gzip -t "$review_path" && gzip -t "$metadata_path"; then
    break
  fi
  echo "[$(date)] waiting for Fashion review/meta under $RAW_ROOT"
  sleep 120
done

bash scripts/14_prepare_amazon_dataset.sh fashion "$review_path" "$metadata_path" parent_asin "$seed"
bash scripts/15_run_formal_seed.sh fashion "$seed"
