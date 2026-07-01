import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch

from src.models.sasrec import SASRec
from src.models.semantic_sasrec import SemanticSASRec


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def ranks_from_scores(scores):
    order = np.argsort(-scores, axis=1)
    return np.where(order == 0)[1] + 1


def metrics_from_ranks(ranks, k=10):
    hits = ranks <= k
    ndcg = np.where(hits, 1.0 / np.log2(ranks + 1), 0.0)
    return float(hits.mean()), float(ndcg.mean())


def tertile_labels(values, names):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    labels = np.empty(len(values), dtype=object)
    n = len(values)
    labels[order[: n // 3]] = names[0]
    labels[order[n // 3 : (2 * n) // 3]] = names[1]
    labels[order[(2 * n) // 3 :]] = names[2]
    return labels


@torch.no_grad()
def score_model(model, split, negatives, max_history_length, device, batch_size):
    model.eval()
    users = list(split.keys())
    all_scores, all_candidates = [], []
    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        histories, candidates = [], []
        for uid in batch_users:
            sample = split[uid]
            history = sample["history"][-max_history_length:]
            histories.append([0] * (max_history_length - len(history)) + history)
            candidates.append([sample["target"]] + negatives[uid])
        seq = torch.tensor(histories, dtype=torch.long, device=device)
        cand = torch.tensor(candidates, dtype=torch.long, device=device)
        all_scores.append(model.predict(seq, cand).detach().cpu().numpy().astype(np.float32))
        all_candidates.append(np.asarray(candidates, dtype=np.int32))
    return np.vstack(all_scores), np.vstack(all_candidates)


def bucket_rows(bucket_name, labels, cf_ranks, sem_ranks):
    rows = []
    for label in list(dict.fromkeys(labels.tolist())):
        mask = labels == label
        cf_hits = cf_ranks[mask] <= 10
        sem_hits = sem_ranks[mask] <= 10
        cf_hr, cf_ndcg = metrics_from_ranks(cf_ranks[mask])
        sem_hr, sem_ndcg = metrics_from_ranks(sem_ranks[mask])
        rows.append(
            {
                "bucket_type": bucket_name,
                "bucket": str(label),
                "num_samples": int(mask.sum()),
                "cf_HR@10": cf_hr,
                "cf_NDCG@10": cf_ndcg,
                "semantic_HR@10": sem_hr,
                "semantic_NDCG@10": sem_ndcg,
                "cf_only": int((cf_hits & ~sem_hits).sum()),
                "semantic_only": int((~cf_hits & sem_hits).sum()),
                "cf_only_ratio": float((cf_hits & ~sem_hits).mean()),
                "semantic_only_ratio": float((~cf_hits & sem_hits).mean()),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/beauty")
    parser.add_argument("--cf-checkpoint", default="outputs/beauty/sasrec_seed2026/best.pt")
    parser.add_argument("--semantic-checkpoint", default="outputs/beauty/semantic_sasrec_seed2026/best.pt")
    parser.add_argument("--semantic-embedding-path", default="data/processed/beauty/item_semantic_embeddings.fp16.npy")
    parser.add_argument("--output-dir", default="outputs/beauty/expert_predictions")
    parser.add_argument("--max-history-length", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads((data_dir / "dataset_stats.json").read_text())
    test = load_pickle(data_dir / "test.pkl")
    test_negatives = load_pickle(data_dir / "test_negatives.pkl")
    train_popularity = load_pickle(data_dir / "train_item_popularity.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cf_model = SASRec(stats["num_items"], 64, args.max_history_length, 2, 2, 0.2).to(device)
    cf_model.load_state_dict(torch.load(args.cf_checkpoint, map_location=device)["model"])
    sem_model = SemanticSASRec(args.semantic_embedding_path, 64, 256, args.max_history_length, 2, 2, 0.2).to(device)
    sem_model.load_state_dict(torch.load(args.semantic_checkpoint, map_location=device)["model"])

    cf_scores, candidate_ids = score_model(cf_model, test, test_negatives, args.max_history_length, device, args.batch_size)
    semantic_scores, semantic_candidate_ids = score_model(sem_model, test, test_negatives, args.max_history_length, device, args.batch_size)
    assert np.array_equal(candidate_ids, semantic_candidate_ids)
    target_indices = np.zeros(candidate_ids.shape[0], dtype=np.int32)

    np.save(output_dir / "cf_scores.npy", cf_scores)
    np.save(output_dir / "semantic_scores.npy", semantic_scores)
    np.save(output_dir / "candidate_ids.npy", candidate_ids)
    np.save(output_dir / "target_indices.npy", target_indices)

    cf_ranks = ranks_from_scores(cf_scores)
    sem_ranks = ranks_from_scores(semantic_scores)
    cf_hits = cf_ranks <= 10
    sem_hits = sem_ranks <= 10
    oracle_ranks = np.minimum(cf_ranks, sem_ranks)
    oracle_hr, oracle_ndcg = metrics_from_ranks(oracle_ranks)
    cf_hr, cf_ndcg = metrics_from_ranks(cf_ranks)
    sem_hr, sem_ndcg = metrics_from_ranks(sem_ranks)

    target_items = candidate_ids[:, 0]
    target_pop = np.asarray([train_popularity.get(int(item), 0) for item in target_items])
    history_lengths = np.asarray([len(test[uid]["history"]) for uid in test.keys()])
    pop_labels = tertile_labels(target_pop, ["Tail", "Mid", "Head"])
    hist_labels = tertile_labels(history_lengths, ["Short", "Medium", "Long"])
    bucket_table = bucket_rows("target_popularity", pop_labels, cf_ranks, sem_ranks)
    bucket_table.extend(bucket_rows("history_length", hist_labels, cf_ranks, sem_ranks))

    summary = {
        "num_samples": int(candidate_ids.shape[0]),
        "cf_HR@10": cf_hr,
        "cf_NDCG@10": cf_ndcg,
        "semantic_HR@10": sem_hr,
        "semantic_NDCG@10": sem_ndcg,
        "both_correct": int((cf_hits & sem_hits).sum()),
        "cf_only_correct": int((cf_hits & ~sem_hits).sum()),
        "semantic_only_correct": int((~cf_hits & sem_hits).sum()),
        "both_wrong": int((~cf_hits & ~sem_hits).sum()),
        "cf_only_ratio": float((cf_hits & ~sem_hits).mean()),
        "semantic_only_ratio": float((~cf_hits & sem_hits).mean()),
        "oracle_HR@10": oracle_hr,
        "oracle_NDCG@10": oracle_ndcg,
        "oracle_gain_over_cf_HR@10": oracle_hr - cf_hr,
        "oracle_gain_over_cf_NDCG@10": oracle_ndcg - cf_ndcg,
        "bucket_table": bucket_table,
    }
    (output_dir.parent / "expert_complementarity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

