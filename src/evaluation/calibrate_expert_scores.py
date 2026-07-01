import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.evaluation.expert_complementarity import metrics_from_ranks, ranks_from_scores


def optimize_temperature(scores):
    x = torch.tensor(scores, dtype=torch.float32)
    y = torch.zeros(x.shape[0], dtype=torch.long)
    log_temp = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        temp = torch.exp(log_temp).clamp(0.05, 100.0)
        loss = F.cross_entropy(x / temp, y)
        loss.backward()
        return loss

    opt.step(closure)
    temp = float(torch.exp(log_temp).detach().clamp(0.05, 100.0).item())
    before = float(F.cross_entropy(x, y).item())
    after = float(F.cross_entropy(x / temp, y).item())
    return temp, before, after


def eval_fusion(cf_scores, sem_scores, cf_temp, sem_temp, g):
    scores = g * (cf_scores / cf_temp) + (1.0 - g) * (sem_scores / sem_temp)
    ranks = ranks_from_scores(scores)
    return metrics_from_ranks(ranks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", default="outputs/beauty/expert_features")
    parser.add_argument("--output-dir", default="outputs/beauty/calibration")
    args = parser.parse_args()
    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = np.load(feature_dir / "valid.npz")
    test = np.load(feature_dir / "test.npz")
    cf_temp, cf_ce_before, cf_ce_after = optimize_temperature(valid["cf_scores"])
    sem_temp, sem_ce_before, sem_ce_after = optimize_temperature(valid["semantic_scores"])
    (output_dir / "cf_temperature.json").write_text(json.dumps({"temperature": cf_temp}, indent=2))
    (output_dir / "semantic_temperature.json").write_text(json.dumps({"temperature": sem_temp}, indent=2))
    grid = []
    best = None
    for g_i in range(11):
        g = g_i / 10
        valid_hr, valid_ndcg = eval_fusion(valid["cf_scores"], valid["semantic_scores"], cf_temp, sem_temp, g)
        test_hr, test_ndcg = eval_fusion(test["cf_scores"], test["semantic_scores"], cf_temp, sem_temp, g)
        row = {"g": g, "valid_HR@10": valid_hr, "valid_NDCG@10": valid_ndcg, "test_HR@10": test_hr, "test_NDCG@10": test_ndcg}
        grid.append(row)
        if best is None or row["valid_NDCG@10"] > best["valid_NDCG@10"]:
            best = row
    metrics = {
        "cf_temperature": cf_temp,
        "semantic_temperature": sem_temp,
        "cf_ce_before": cf_ce_before,
        "cf_ce_after": cf_ce_after,
        "semantic_ce_before": sem_ce_before,
        "semantic_ce_after": sem_ce_after,
        "static_fusion_grid": grid,
        "best_static_fusion": best,
    }
    (output_dir / "calibration_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

