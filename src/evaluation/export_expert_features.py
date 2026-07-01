import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from src.models.sasrec import SASRec
from src.models.semantic_sasrec import SemanticSASRec


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def sample_train_candidates(train, valid, test, num_items, num_negatives, seed):
    rng = random.Random(seed)
    split, negatives = {}, {}
    all_items = list(range(1, num_items + 1))
    for uid, seq in train.items():
        if len(seq) < 2:
            continue
        history = seq[:-1]
        target = seq[-1]
        forbidden = set(seq)
        forbidden.add(valid[uid]["target"])
        forbidden.add(test[uid]["target"])
        pool = [item for item in all_items if item not in forbidden and item != target]
        negatives[uid] = rng.sample(pool, num_negatives)
        split[uid] = {"history": history, "target": target}
    return split, negatives


@torch.no_grad()
def export_split(cf_model, sem_model, split, negatives, max_history_length, device, batch_size):
    cf_model.eval()
    sem_model.eval()
    users = list(split.keys())
    arrays = {k: [] for k in ["user_ids", "candidate_ids", "cf_scores", "semantic_scores", "cf_user", "semantic_user", "history_length"]}
    target_indices = []
    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        histories, candidates, hist_lens = [], [], []
        for uid in batch_users:
            sample = split[uid]
            history = sample["history"][-max_history_length:]
            histories.append([0] * (max_history_length - len(history)) + history)
            candidates.append([sample["target"]] + negatives[uid])
            hist_lens.append(len(sample["history"]))
        seq = torch.tensor(histories, dtype=torch.long, device=device)
        cand = torch.tensor(candidates, dtype=torch.long, device=device)
        cf_states = cf_model.encode(seq)[:, -1, :]
        sem_states = sem_model.encode(seq)[:, -1, :]
        cf_scores = cf_model.predict(seq, cand)
        sem_scores = sem_model.predict(seq, cand)
        arrays["user_ids"].append(np.asarray(batch_users, dtype=np.int32))
        arrays["candidate_ids"].append(np.asarray(candidates, dtype=np.int32))
        arrays["cf_scores"].append(cf_scores.detach().cpu().numpy().astype(np.float32))
        arrays["semantic_scores"].append(sem_scores.detach().cpu().numpy().astype(np.float32))
        arrays["cf_user"].append(cf_states.detach().cpu().numpy().astype(np.float32))
        arrays["semantic_user"].append(sem_states.detach().cpu().numpy().astype(np.float32))
        arrays["history_length"].append(np.asarray(hist_lens, dtype=np.float32))
        target_indices.append(np.zeros(len(batch_users), dtype=np.int32))
    return {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()} | {"target_indices": np.concatenate(target_indices)}


@torch.no_grad()
def export_item_features(cf_model, sem_model, num_items, device, batch_size):
    cf_items = cf_model.item_embedding.weight.detach().cpu().numpy().astype(np.float32)
    sem_parts = []
    for start in range(0, num_items + 1, batch_size):
        ids = torch.arange(start, min(num_items + 1, start + batch_size), dtype=torch.long, device=device)
        sem_parts.append(sem_model.project_items(ids).detach().cpu().numpy().astype(np.float32))
    return cf_items, np.vstack(sem_parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/beauty")
    parser.add_argument("--cf-checkpoint", default="outputs/beauty/sasrec_seed2026/best.pt")
    parser.add_argument("--semantic-checkpoint", default="outputs/beauty/semantic_sasrec_seed2026/best.pt")
    parser.add_argument("--semantic-embedding-path", default="data/processed/beauty/item_semantic_embeddings.fp16.npy")
    parser.add_argument("--output-dir", default="outputs/beauty/expert_features")
    parser.add_argument("--max-history-length", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-train-negatives", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads((data_dir / "dataset_stats.json").read_text())
    train = load_pickle(data_dir / "train.pkl")
    valid = load_pickle(data_dir / "valid.pkl")
    test = load_pickle(data_dir / "test.pkl")
    valid_neg = load_pickle(data_dir / "valid_negatives.pkl")
    test_neg = load_pickle(data_dir / "test_negatives.pkl")
    train_split, train_neg = sample_train_candidates(train, valid, test, stats["num_items"], args.num_train_negatives, args.seed + 101)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cf_model = SASRec(stats["num_items"], 64, args.max_history_length, 2, 2, 0.2).to(device)
    cf_model.load_state_dict(torch.load(args.cf_checkpoint, map_location=device)["model"])
    sem_model = SemanticSASRec(args.semantic_embedding_path, 64, 256, args.max_history_length, 2, 2, 0.2).to(device)
    sem_model.load_state_dict(torch.load(args.semantic_checkpoint, map_location=device)["model"])

    for name, split, neg in [("train", train_split, train_neg), ("valid", valid, valid_neg), ("test", test, test_neg)]:
        exported = export_split(cf_model, sem_model, split, neg, args.max_history_length, device, args.batch_size)
        np.savez_compressed(output_dir / f"{name}.npz", **exported)
        print(json.dumps({"split": name, "num_samples": int(exported["candidate_ids"].shape[0])}))

    cf_items, sem_items = export_item_features(cf_model, sem_model, stats["num_items"], device, args.batch_size)
    np.save(output_dir / "cf_item_embeddings.npy", cf_items)
    np.save(output_dir / "semantic_item_embeddings.npy", sem_items)
    meta = {"num_items": stats["num_items"], "num_candidates": 101, "seed": args.seed}
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

