import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/beauty")
    args = parser.parse_args()
    processed_dir = Path(args.processed_dir)
    stats = json.loads((processed_dir / "dataset_stats.json").read_text())
    rows = [json.loads(line) for line in (processed_dir / "item_texts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == stats["num_items"]
    for expected_id, row in enumerate(rows, start=1):
        assert row["item_id"] == expected_id
        assert row["text"]
        assert row["has_text"] in (0, 1)
    print(json.dumps({"status": "ok", "num_item_texts": len(rows), "padding_item": "reserved_zero_row"}))


if __name__ == "__main__":
    main()

