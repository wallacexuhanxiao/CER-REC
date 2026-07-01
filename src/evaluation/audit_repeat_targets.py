import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from src.evaluation.expert_complementarity import metrics_from_ranks, ranks_from_scores


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def subset_metrics(mask, cf_ranks, sem_ranks):
    if int(mask.sum()) == 0:
        return {
            "num_samples": 0,
            "cf_HR@10": None,
            "cf_NDCG@10": None,
            "semantic_HR@10": None,
            "semantic_NDCG@10": None,
            "oracle_HR@10": None,
            "oracle_NDCG@10": None,
        }
    oracle = np.minimum(cf_ranks[mask], sem_ranks[mask])
    cf = metrics_from_ranks(cf_ranks[mask])
    sem = metrics_from_ranks(sem_ranks[mask])
    ora = metrics_from_ranks(oracle)
    return {
        "num_samples": int(mask.sum()),
        "cf_HR@10": cf[0],
        "cf_NDCG@10": cf[1],
        "semantic_HR@10": sem[0],
        "semantic_NDCG@10": sem[1],
        "oracle_HR@10": ora[0],
        "oracle_NDCG@10": ora[1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/beauty")
    parser.add_argument("--prediction-dir", default="outputs/beauty/expert_predictions")
    parser.add_argument("--output-dir", default="outputs/beauty/audits")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    pred_dir = Path(args.prediction_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test = load_pickle(data_dir / "test.pkl")
    cf_scores = np.load(pred_dir / "cf_scores.npy")
    sem_scores = np.load(pred_dir / "semantic_scores.npy")
    cf_ranks = ranks_from_scores(cf_scores)
    sem_ranks = ranks_from_scores(sem_scores)
    repeats = np.asarray([sample["target"] in set(sample["history"]) for sample in test.values()])
    result = {
        "repeat_target_count": int(repeats.sum()),
        "repeat_target_ratio": float(repeats.mean()),
        "all": subset_metrics(np.ones_like(repeats, dtype=bool), cf_ranks, sem_ranks),
        "non_repeat_only": subset_metrics(~repeats, cf_ranks, sem_ranks),
        "repeat_only": subset_metrics(repeats, cf_ranks, sem_ranks),
    }
    (output_dir / "repeat_target_analysis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
