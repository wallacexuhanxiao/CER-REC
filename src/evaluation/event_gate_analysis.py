import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from src.evaluation.expert_complementarity import metrics_from_ranks, ranks_from_scores


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def load_json(path):
    return json.loads(Path(path).read_text())


def recovery(scores, cf_scores, sem_scores):
    ranks = ranks_from_scores(scores)
    cf_ranks = ranks_from_scores(cf_scores)
    sem_ranks = ranks_from_scores(sem_scores)
    hits = ranks <= 10
    cf_hits = cf_ranks <= 10
    sem_hits = sem_ranks <= 10
    cf_only = cf_hits & ~sem_hits
    sem_only = ~cf_hits & sem_hits
    oracle_hr = metrics_from_ranks(np.minimum(cf_ranks, sem_ranks))[0]
    sem_hr = metrics_from_ranks(sem_ranks)[0]
    hr, ndcg = metrics_from_ranks(ranks)
    return {
        "HR@10": hr,
        "NDCG@10": ndcg,
        "CF-Recovery": float((hits & cf_only).sum() / max(1, cf_only.sum())),
        "Semantic-Retention": float((hits & sem_only).sum() / max(1, sem_only.sum())),
        "Oracle-Gap-Capture": float((hr - sem_hr) / max(1e-12, oracle_hr - sem_hr)),
    }


def gate_stats(gates, masks, cf_scores, sem_scores, target_items, train_pop):
    cf_hits = ranks_from_scores(cf_scores) <= 10
    sem_hits = ranks_from_scores(sem_scores) <= 10
    cf_only = cf_hits & ~sem_hits
    sem_only = ~cf_hits & sem_hits
    both = cf_hits & sem_hits
    wrong = ~cf_hits & ~sem_hits
    valid = masks.astype(bool)
    flat = gates[valid]
    target_mean = (gates * valid).sum(axis=1) / np.maximum(1, valid.sum(axis=1))
    freqs = np.asarray([train_pop.get(int(item), 0) for item in target_items])
    order = np.argsort(freqs, kind="stable")
    n = len(freqs)
    buckets = {
        "Tail": order[: n // 3],
        "Mid": order[n // 3 : (2 * n) // 3],
        "Head": order[(2 * n) // 3 :],
    }
    result = {
        "mean_gate": float(flat.mean()),
        "std_gate": float(flat.std()),
        "p10": float(np.quantile(flat, 0.10)),
        "p25": float(np.quantile(flat, 0.25)),
        "p50": float(np.quantile(flat, 0.50)),
        "p75": float(np.quantile(flat, 0.75)),
        "p90": float(np.quantile(flat, 0.90)),
        "CF-only_mean_gate": float(target_mean[cf_only].mean()) if np.any(cf_only) else None,
        "Semantic-only_mean_gate": float(target_mean[sem_only].mean()) if np.any(sem_only) else None,
        "Both-correct_mean_gate": float(target_mean[both].mean()) if np.any(both) else None,
        "Both-wrong_mean_gate": float(target_mean[wrong].mean()) if np.any(wrong) else None,
    }
    for name, idx in buckets.items():
        result[f"{name}_mean_gate"] = float(target_mean[idx].mean())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/beauty")
    parser.add_argument("--event-dir", default="outputs/beauty/event_gate")
    parser.add_argument("--expert-dir", default="outputs/beauty/expert_predictions")
    parser.add_argument("--score-router-summary", default="outputs/beauty/score_routers/summary.json")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    event_dir = Path(args.event_dir)
    expert_dir = Path(args.expert_dir)
    cf_scores = np.load(expert_dir / "cf_scores.npy")
    sem_scores = np.load(expert_dir / "semantic_scores.npy")
    candidate_ids = np.load(expert_dir / "candidate_ids.npy")
    train_pop = load_pickle(data_dir / "train_item_popularity.pkl")
    cf_metrics = recovery(cf_scores, cf_scores, sem_scores)
    sem_metrics = recovery(sem_scores, cf_scores, sem_scores)
    oracle_ranks = np.minimum(ranks_from_scores(cf_scores), ranks_from_scores(sem_scores))
    oracle = metrics_from_ranks(oracle_ranks)
    report = {
        "CF": cf_metrics,
        "Semantic-only": sem_metrics,
        "Oracle": {"HR@10": oracle[0], "NDCG@10": oracle[1]},
        "ScoreRouters": load_json(args.score_router_summary),
        "EventModels": {},
    }
    for name, label in [("fixed_half", "DualEvent-NoRoute"), ("learned", "EventGate-NoTeacher")]:
        model_dir = event_dir / name
        scores = np.load(model_dir / "test_scores.npy")
        metrics = recovery(scores, cf_scores, sem_scores)
        if (model_dir / "target_event_gates.npy").exists():
            gates = np.load(model_dir / "target_event_gates.npy")
            masks = np.load(model_dir / "history_masks.npy")
            metrics["gate_stats"] = gate_stats(gates, masks, cf_scores, sem_scores, candidate_ids[:, 0], train_pop)
        report["EventModels"][label] = metrics
    output_path = event_dir / "event_gate_analysis.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

