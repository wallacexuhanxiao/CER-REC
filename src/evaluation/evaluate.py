import pickle
from pathlib import Path

import torch

from src.evaluation.metrics import hr_ndcg_at_k


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


@torch.no_grad()
def evaluate_model(model, split, negatives, max_history_length, device, k=10, batch_size=512):
    model.eval()
    users = list(split.keys())
    total_hr = 0.0
    total_ndcg = 0.0

    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        histories = []
        candidates = []
        targets = []
        for uid in batch_users:
            sample = split[uid]
            history = sample["history"][-max_history_length:]
            padded = [0] * (max_history_length - len(history)) + history
            cand = [sample["target"]] + negatives[uid]
            histories.append(padded)
            candidates.append(cand)
            targets.append(sample["target"])

        seq = torch.tensor(histories, dtype=torch.long, device=device)
        cand = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = model.predict(seq, cand).detach().cpu()
        for row, cand_items, target in zip(scores, candidates, targets):
            order = torch.argsort(row, descending=True).tolist()
            ranked = [cand_items[idx] for idx in order]
            hr, ndcg = hr_ndcg_at_k(ranked, target, k)
            total_hr += hr
            total_ndcg += ndcg

    n = len(users)
    return {"HR@10": total_hr / n, "NDCG@10": total_ndcg / n}

