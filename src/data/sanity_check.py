import argparse
import json
import pickle
from pathlib import Path


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/beauty")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    train = load_pickle(data_dir / "train.pkl")
    valid = load_pickle(data_dir / "valid.pkl")
    test = load_pickle(data_dir / "test.pkl")
    valid_negatives = load_pickle(data_dir / "valid_negatives.pkl")
    test_negatives = load_pickle(data_dir / "test_negatives.pkl")
    popularity = load_pickle(data_dir / "train_item_popularity.pkl")
    stats = json.loads((data_dir / "dataset_stats.json").read_text())

    for uid in train:
        all_seen = set(test[uid]["history"]) | {test[uid]["target"]}
        assert valid[uid]["target"] not in valid[uid]["history"]
        assert test[uid]["target"] not in test[uid]["history"]
        assert not (set(valid_negatives[uid]) & all_seen)
        assert not (set(test_negatives[uid]) & all_seen)
        assert len(valid_negatives[uid]) == stats["num_test_negatives"]
        assert len(test_negatives[uid]) == stats["num_test_negatives"]
    assert sum(popularity.values()) == stats["num_train_interactions"]
    print(json.dumps({"status": "ok", "checks": "data leakage, negatives, train-only popularity"}))


if __name__ == "__main__":
    main()

