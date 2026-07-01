import argparse
import csv
import gzip
import json
import pickle
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlretrieve


RAW_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv"


def download_raw(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "ratings_Beauty.csv"
    gz_path = raw_dir / "ratings_Beauty.csv.gz"
    if csv_path.exists():
        return csv_path
    if gz_path.exists():
        return gz_path
    print(f"Downloading Amazon Beauty ratings from {RAW_URL}")
    urlretrieve(RAW_URL, csv_path)
    return csv_path


def load_ratings(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            user, item, rating, timestamp = row[:4]
            yield user, item, float(rating), int(float(timestamp))


def iterative_kcore(interactions, min_user: int, min_item: int):
    filtered = list(interactions)
    changed = True
    while changed:
        changed = False
        user_counts = Counter(u for u, _, _ in filtered)
        item_counts = Counter(i for _, i, _ in filtered)
        kept = [
            (u, i, t)
            for u, i, t in filtered
            if user_counts[u] >= min_user and item_counts[i] >= min_item
        ]
        if len(kept) != len(filtered):
            changed = True
            filtered = kept
    return filtered


def build_mappings(user_sequences):
    users = sorted(user_sequences)
    item_counts = Counter(i for seq in user_sequences.values() for i, _ in seq)
    items = sorted(item_counts)
    user2id = {u: idx + 1 for idx, u in enumerate(users)}
    item2id = {i: idx + 1 for idx, i in enumerate(items)}
    return user2id, item2id


def make_splits(user_sequences, user2id, item2id):
    train, valid, test = {}, {}, {}
    all_user_items = {}
    train_item_popularity = Counter()
    timestamps = {}

    for raw_user, seq in user_sequences.items():
        uid = user2id[raw_user]
        mapped = [(item2id[item], ts) for item, ts in seq]
        item_ids = [item for item, _ in mapped]
        all_user_items[uid] = set(item_ids)
        timestamps[uid] = [ts for _, ts in mapped]

        train_prefix = item_ids[:-2]
        valid_history = item_ids[:-2]
        valid_target = item_ids[-2]
        test_history = item_ids[:-1]
        test_target = item_ids[-1]

        train[uid] = train_prefix
        valid[uid] = {"history": valid_history, "target": valid_target}
        test[uid] = {"history": test_history, "target": test_target}
        train_item_popularity.update(train_prefix)

    return train, valid, test, all_user_items, train_item_popularity, timestamps


def sample_negatives(split, all_user_items, num_items, num_negatives, seed):
    rng = random.Random(seed)
    all_items = list(range(1, num_items + 1))
    negatives = {}
    for uid, sample in split.items():
        forbidden = set(all_user_items[uid])
        forbidden.add(sample["target"])
        pool = [item for item in all_items if item not in forbidden]
        if len(pool) < num_negatives:
            raise ValueError(f"User {uid} has only {len(pool)} available negatives.")
        negatives[uid] = rng.sample(pool, num_negatives)
    return negatives


def assert_protocol(train, valid, test, valid_negatives, test_negatives, all_user_items, timestamps):
    for uid in train:
        if valid[uid]["target"] in valid[uid]["history"]:
            raise AssertionError(f"validation target leaked into history for user {uid}")
        if test[uid]["target"] in test[uid]["history"]:
            raise AssertionError(f"test target leaked into history for user {uid}")
        if set(valid_negatives[uid]) & all_user_items[uid]:
            raise AssertionError(f"validation negatives overlap user history for user {uid}")
        if set(test_negatives[uid]) & all_user_items[uid]:
            raise AssertionError(f"test negatives overlap user history for user {uid}")
        if valid[uid]["target"] in valid_negatives[uid]:
            raise AssertionError(f"validation target appears in negatives for user {uid}")
        if test[uid]["target"] in test_negatives[uid]:
            raise AssertionError(f"test target appears in negatives for user {uid}")
        if timestamps[uid] != sorted(timestamps[uid]):
            raise AssertionError(f"timestamps are not sorted for user {uid}")


def save_pickle(path: Path, obj):
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed/beauty")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-negatives", type=int, default=100)
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-item-interactions", type=int, default=5)
    parser.add_argument("--rating-threshold", type=float, default=None)
    args = parser.parse_args()

    raw_path = download_raw(Path(args.raw_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = []
    for user, item, rating, ts in load_ratings(raw_path):
        if args.rating_threshold is not None and rating <= args.rating_threshold:
            continue
        raw.append((user, item, ts))

    filtered = iterative_kcore(raw, args.min_user_interactions, args.min_item_interactions)
    user_sequences = defaultdict(list)
    for user, item, ts in filtered:
        user_sequences[user].append((item, ts))
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
        "dataset": "Amazon Beauty",
        "source": "amazon_2014_ratings_fallback_llm_esr_style_cold_start_filter",
        "raw_file": str(raw_path),
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
        "median_sequence_length": statistics.median(lengths),
        "max_sequence_length": max(lengths),
        "min_user_interactions": args.min_user_interactions,
        "min_item_interactions": args.min_item_interactions,
        "rating_threshold": args.rating_threshold,
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

