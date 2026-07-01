import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from src.evaluation.expert_complementarity import tertile_labels


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/beauty")
    parser.add_argument("--output-dir", default="outputs/beauty/audits")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_pop = load_pickle(data_dir / "train_item_popularity.pkl")
    test = load_pickle(data_dir / "test.pkl")
    num_items = json.loads((data_dir / "dataset_stats.json").read_text())["num_items"]
    item_freq = np.asarray([train_pop.get(item, 0) for item in range(1, num_items + 1)])
    item_labels = tertile_labels(item_freq, ["Tail", "Mid", "Head"])
    target_items = np.asarray([sample["target"] for sample in test.values()])
    target_freq = np.asarray([train_pop.get(int(item), 0) for item in target_items])
    target_labels = tertile_labels(target_freq, ["Tail", "Mid", "Head"])
    rows = []
    for label in ["Tail", "Mid", "Head"]:
        item_mask = item_labels == label
        test_mask = target_labels == label
        rows.append(
            {
                "bucket": label,
                "num_items": int(item_mask.sum()),
                "num_test_samples": int(test_mask.sum()),
                "avg_train_frequency_items": float(item_freq[item_mask].mean()),
                "avg_train_frequency_test_targets": float(target_freq[test_mask].mean()),
                "min_train_frequency_test_targets": int(target_freq[test_mask].min()),
                "max_train_frequency_test_targets": int(target_freq[test_mask].max()),
            }
        )
    result = {"frequency_source": "train_item_popularity.pkl", "buckets": rows}
    (output_dir / "tail_bucket_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

