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


def route_stats(gates, masks, branch_mass, cf_scores, sem_scores, target_items, train_pop):
    cf_hits = ranks_from_scores(cf_scores) <= 10
    sem_hits = ranks_from_scores(sem_scores) <= 10
    cf_only = cf_hits & ~sem_hits
    sem_only = ~cf_hits & sem_hits
    both = cf_hits & sem_hits
    wrong = ~cf_hits & ~sem_hits
    valid = masks.astype(bool)
    flat_gate = gates[valid]
    raw_gate_mean = (gates * valid).sum(axis=1) / np.maximum(1, valid.sum(axis=1))
    cf_mass = branch_mass
    freqs = np.asarray([train_pop.get(int(item), 0) for item in target_items])
    order = np.argsort(freqs, kind="stable")
    n = len(freqs)
    buckets = {
        "Tail": order[: n // 3],
        "Mid": order[n // 3 : (2 * n) // 3],
        "Head": order[(2 * n) // 3 :],
    }

    def maybe_mean(values, mask):
        return float(values[mask].mean()) if np.any(mask) else None

    result = {
        "cf_branch_mass_mean": float(cf_mass.mean()),
        "cf_branch_mass_std": float(cf_mass.std()),
        "cf_branch_mass_p10": float(np.quantile(cf_mass, 0.10)),
        "cf_branch_mass_p25": float(np.quantile(cf_mass, 0.25)),
        "cf_branch_mass_p50": float(np.quantile(cf_mass, 0.50)),
        "cf_branch_mass_p75": float(np.quantile(cf_mass, 0.75)),
        "cf_branch_mass_p90": float(np.quantile(cf_mass, 0.90)),
        "CF-only_cf_branch_mass": maybe_mean(cf_mass, cf_only),
        "Semantic-only_cf_branch_mass": maybe_mean(cf_mass, sem_only),
        "Both-correct_cf_branch_mass": maybe_mean(cf_mass, both),
        "Both-wrong_cf_branch_mass": maybe_mean(cf_mass, wrong),
        "raw_gate_mean": float(flat_gate.mean()),
        "raw_gate_std": float(flat_gate.std()),
        "raw_gate_p10": float(np.quantile(flat_gate, 0.10)),
        "raw_gate_p50": float(np.quantile(flat_gate, 0.50)),
        "raw_gate_p90": float(np.quantile(flat_gate, 0.90)),
        "CF-only_raw_gate": maybe_mean(raw_gate_mean, cf_only),
        "Semantic-only_raw_gate": maybe_mean(raw_gate_mean, sem_only),
    }
    for name, idx in buckets.items():
        result[f"{name}_cf_branch_mass"] = float(cf_mass[idx].mean())
        result[f"{name}_raw_gate"] = float(raw_gate_mean[idx].mean())
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
        if (model_dir / "target_event_gates.npy").exists() and (model_dir / "target_branch_masses.npy").exists():
            gates = np.load(model_dir / "target_event_gates.npy")
            masks = np.load(model_dir / "history_masks.npy")
            branch_mass = np.load(model_dir / "target_branch_masses.npy")
            metrics["route_stats"] = route_stats(
                gates, branch_mass=branch_mass, masks=masks, cf_scores=cf_scores,
                sem_scores=sem_scores, target_items=candidate_ids[:, 0], train_pop=train_pop
            )
        report["EventModels"][label] = metrics
    output_path = event_dir / "event_gate_analysis.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

