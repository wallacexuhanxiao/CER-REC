import argparse
import ast
import gzip
import json
from datetime import datetime
from pathlib import Path

from src.data.preprocess_amazon import (
    assert_protocol,
    build_mappings,
    iterative_kcore,
    make_splits,
    sample_negatives,
    save_pickle,
)


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="ignore") if path.suffix == ".gz" else path.open("rt", encoding="utf-8", errors="ignore")


def parse_date(value):
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return int(datetime.strptime(str(value), fmt).timestamp())
        except ValueError:
            continue
    return None


def iter_reviews(path: Path):
    with open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = ast.literal_eval(line)
            user = row.get("username")
            item = row.get("product_id")
            timestamp = parse_date(row.get("date"))
            if user is None or item is None or timestamp is None:
                continue
            yield str(user), str(item), timestamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-negatives", type=int, default=100)
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-item-interactions", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = list(iter_reviews(Path(args.reviews_path)))
    filtered = iterative_kcore(raw, args.min_user_interactions, args.min_item_interactions)
    user_sequences = {}
    for user, item, timestamp in filtered:
        user_sequences.setdefault(user, []).append((item, timestamp))
    user_sequences = {
        user: sorted(seq, key=lambda x: (x[1], x[0]))
        for user, seq in user_sequences.items()
        if len(seq) >= 5
    }

    user2id, item2id = build_mappings(user_sequences)
    train, valid, test, all_user_items, train_item_popularity, timestamps = make_splits(
        user_sequences, user2id, item2id
    )
    num_items = len(item2id)
    valid_negatives = sample_negatives(valid, all_user_items, num_items, args.num_negatives, args.seed + 17)
    test_negatives = sample_negatives(test, all_user_items, num_items, args.num_negatives, args.seed + 29)
    assert_protocol(train, valid, test, valid_negatives, test_negatives, all_user_items, timestamps)

    lengths = [len(seq) for seq in train.values()]
    stats = {
        "dataset": "Steam",
        "source": str(args.reviews_path),
        "input_format": "python-dict-gzip",
        "split": "leave-one-out by timestamp",
        "validation": "second-to-last item",
        "test": "last item",
        "num_test_negatives": args.num_negatives,
        "negative_seed_valid": args.seed + 17,
        "negative_seed_test": args.seed + 29,
        "num_users": len(user2id),
        "num_items": len(item2id),
        "num_interactions": sum(len(seq) + 2 for seq in train.values()),
        "num_train_interactions": sum(len(seq) for seq in train.values()),
        "avg_sequence_length": sum(lengths) / len(lengths),
        "median_sequence_length": float(__import__("statistics").median(lengths)),
        "max_sequence_length": max(lengths),
        "min_user_interactions": args.min_user_interactions,
        "min_item_interactions": args.min_item_interactions,
        "user_field": "username",
        "item_field": "product_id",
        "timestamp_field": "date",
    }

    save_pickle(output_dir / "train.pkl", train)
    save_pickle(output_dir / "valid.pkl", valid)
    save_pickle(output_dir / "test.pkl", test)
    save_pickle(output_dir / "valid_negatives.pkl", valid_negatives)
    save_pickle(output_dir / "test_negatives.pkl", test_negatives)
    save_pickle(output_dir / "train_item_popularity.pkl", dict(train_item_popularity))
    (output_dir / "user2id.json").write_text(json.dumps(user2id, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "item2id.json").write_text(json.dumps(item2id, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
