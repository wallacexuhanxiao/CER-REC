import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from src.evaluation.expert_complementarity import metrics_from_ranks, ranks_from_scores
from src.evaluation.export_expert_features import export_split
from src.models.sasrec import SASRec
from src.models.semantic_sasrec import SemanticSASRec


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/beauty")
    parser.add_argument("--negatives-path", default="data/processed/beauty/test_negatives_semantic_hard.pkl")
    parser.add_argument("--cf-checkpoint", default="outputs/beauty/sasrec_seed2026/best.pt")
    parser.add_argument("--semantic-checkpoint", default="outputs/beauty/semantic_sasrec_seed2026/best.pt")
    parser.add_argument("--semantic-embedding-path", default="data/processed/beauty/item_semantic_embeddings.fp16.npy")
    parser.add_argument("--output-dir", default="outputs/beauty/audits")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads((data_dir / "dataset_stats.json").read_text())
    test = load_pickle(data_dir / "test.pkl")
    negatives = load_pickle(args.negatives_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cf_model = SASRec(stats["num_items"], 64, 50, 2, 2, 0.2).to(device)
    cf_model.load_state_dict(torch.load(args.cf_checkpoint, map_location=device)["model"])
    sem_model = SemanticSASRec(args.semantic_embedding_path, 64, 256, 50, 2, 2, 0.2).to(device)
    sem_model.load_state_dict(torch.load(args.semantic_checkpoint, map_location=device)["model"])
    exported = export_split(cf_model, sem_model, test, negatives, 50, device, 4096)
    cf_ranks = ranks_from_scores(exported["cf_scores"])
    sem_ranks = ranks_from_scores(exported["semantic_scores"])
    cf_hits = cf_ranks <= 10
    sem_hits = sem_ranks <= 10
    oracle_ranks = np.minimum(cf_ranks, sem_ranks)
    result = {
        "num_samples": int(len(cf_ranks)),
        "cf_HR@10": metrics_from_ranks(cf_ranks)[0],
        "cf_NDCG@10": metrics_from_ranks(cf_ranks)[1],
        "semantic_HR@10": metrics_from_ranks(sem_ranks)[0],
        "semantic_NDCG@10": metrics_from_ranks(sem_ranks)[1],
        "both_correct": int((cf_hits & sem_hits).sum()),
        "cf_only_correct": int((cf_hits & ~sem_hits).sum()),
        "semantic_only_correct": int((~cf_hits & sem_hits).sum()),
        "both_wrong": int((~cf_hits & ~sem_hits).sum()),
        "semantic_only_ratio": float((~cf_hits & sem_hits).mean()),
        "oracle_HR@10": metrics_from_ranks(oracle_ranks)[0],
        "oracle_NDCG@10": metrics_from_ranks(oracle_ranks)[1],
    }
    (output_dir / "hard_negative_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

