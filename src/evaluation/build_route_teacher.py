import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.sasrec import SASRec
from src.models.semantic_sasrec import SemanticSASRec
from src.trainers.train_event_gate import file_sha256, sample_train_candidates, set_seed


class TeacherDataset(Dataset):
    def __init__(self, split, negatives, max_history_length):
        self.users = list(split.keys())
        histories, candidates = [], []
        for uid in self.users:
            sample = split[uid]
            history = sample["history"][-max_history_length:]
            histories.append([0] * (max_history_length - len(history)) + history)
            candidates.append([sample["target"]] + negatives[uid])
        self.histories = torch.tensor(histories, dtype=torch.long)
        self.candidates = torch.tensor(candidates, dtype=torch.long)

    def __len__(self):
        return self.histories.shape[0]

    def __getitem__(self, index):
        return index, self.histories[index], self.candidates[index]


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def q_margin(scores):
    return scores[:, 0] - torch.logsumexp(scores, dim=1)


@torch.no_grad()
def masked_margins(model, seq, candidates, valid_positions, chunk_size):
    base_scores = model.predict(seq, candidates)
    base_q = q_margin(base_scores)
    flat_rows, flat_pos = valid_positions.nonzero(as_tuple=True)
    masked_q = torch.empty(flat_rows.shape[0], dtype=base_q.dtype, device=base_q.device)
    for start in range(0, flat_rows.shape[0], chunk_size):
        end = min(flat_rows.shape[0], start + chunk_size)
        rows = flat_rows[start:end]
        pos = flat_pos[start:end]
        masked_seq = seq[rows].clone()
        masked_seq[torch.arange(end - start, device=seq.device), pos] = 0
        masked_scores = model.predict(masked_seq, candidates[rows])
        masked_q[start:end] = q_margin(masked_scores)
    deltas = torch.zeros_like(seq, dtype=base_q.dtype)
    deltas[flat_rows, flat_pos] = base_q[flat_rows] - masked_q
    return deltas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/beauty")
    parser.add_argument("--cf-checkpoint", default="outputs/beauty/sasrec_seed2026/best.pt")
    parser.add_argument("--semantic-checkpoint", default="outputs/beauty/semantic_sasrec_seed2026/best.pt")
    parser.add_argument("--semantic-embedding-path", default="data/processed/beauty/item_semantic_embeddings.fp16.npy")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/cer-rec/beauty/route_teacher")
    parser.add_argument("--max-history-length", type=int, default=50)
    parser.add_argument("--train-negatives", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mask-chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads((data_dir / "dataset_stats.json").read_text())
    train = load_pickle(data_dir / "train.pkl")
    valid = load_pickle(data_dir / "valid.pkl")
    test = load_pickle(data_dir / "test.pkl")
    train_split, train_neg = sample_train_candidates(train, valid, test, stats["num_items"], args.train_negatives, args.seed + 503)
    dataset = TeacherDataset(train_split, train_neg, args.max_history_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cf_model = SASRec(stats["num_items"], 64, args.max_history_length, 2, 2, 0.2).to(device)
    cf_model.load_state_dict(torch.load(args.cf_checkpoint, map_location=device)["model"])
    sem_model = SemanticSASRec(args.semantic_embedding_path, 64, 256, args.max_history_length, 2, 2, 0.2).to(device)
    sem_model.load_state_dict(torch.load(args.semantic_checkpoint, map_location=device)["model"])
    cf_model.eval()
    sem_model.eval()

    delta_cf = np.zeros((len(dataset), args.max_history_length), dtype=np.float32)
    delta_sem = np.zeros_like(delta_cf)
    valid_history = np.zeros((len(dataset), args.max_history_length), dtype=bool)
    print(json.dumps({"device": str(device), "num_samples": len(dataset), "num_batches": len(loader)}), flush=True)
    for batch_idx, (indices, seq, candidates) in enumerate(loader, start=1):
        seq = seq.to(device)
        candidates = candidates.to(device)
        valid_positions = seq.ne(0)
        cf_delta = masked_margins(cf_model, seq, candidates, valid_positions, args.mask_chunk_size)
        sem_delta = masked_margins(sem_model, seq, candidates, valid_positions, args.mask_chunk_size)
        idx = indices.numpy()
        delta_cf[idx] = cf_delta.cpu().numpy()
        delta_sem[idx] = sem_delta.cpu().numpy()
        valid_history[idx] = valid_positions.cpu().numpy()
        if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == len(loader):
            print(json.dumps({"batch": batch_idx, "num_batches": len(loader)}), flush=True)

    cf_scale = float(np.quantile(np.abs(delta_cf[valid_history]), 0.75))
    sem_scale = float(np.quantile(np.abs(delta_sem[valid_history]), 0.75))
    norm_cf = delta_cf / (cf_scale + args.eps)
    norm_sem = delta_sem / (sem_scale + args.eps)
    u_cf = np.maximum(norm_cf, 0.0)
    u_sem = np.maximum(norm_sem, 0.0)
    utility = u_cf + u_sem
    route_valid = valid_history & (utility > 0.0)
    targets = np.zeros_like(delta_cf, dtype=np.float32)
    weights = np.zeros_like(delta_cf, dtype=np.float32)
    targets[route_valid] = u_cf[route_valid] / (utility[route_valid] + args.eps)
    weights[route_valid] = np.abs(u_cf[route_valid] - u_sem[route_valid]) / (utility[route_valid] + args.eps)

    np.save(output_dir / "route_targets.float16.npy", targets.astype(np.float16))
    np.save(output_dir / "route_weights.float16.npy", weights.astype(np.float16))
    np.save(output_dir / "route_valid_mask.npy", route_valid)
    stats_out = {
        "num_samples": len(dataset),
        "max_history_length": args.max_history_length,
        "train_negatives": args.train_negatives,
        "seed": args.seed,
        "negative_seed": args.seed + 503,
        "cf_abs_delta_q75": cf_scale,
        "semantic_abs_delta_q75": sem_scale,
        "valid_history_events": int(valid_history.sum()),
        "route_valid_events": int(route_valid.sum()),
        "route_valid_ratio": float(route_valid.sum() / max(1, valid_history.sum())),
        "target_mean": float(targets[route_valid].mean()) if route_valid.any() else None,
        "target_p10": float(np.quantile(targets[route_valid], 0.10)) if route_valid.any() else None,
        "target_p50": float(np.quantile(targets[route_valid], 0.50)) if route_valid.any() else None,
        "target_p90": float(np.quantile(targets[route_valid], 0.90)) if route_valid.any() else None,
        "weight_mean": float(weights[route_valid].mean()) if route_valid.any() else None,
        "weight_p50": float(np.quantile(weights[route_valid], 0.50)) if route_valid.any() else None,
        "checksums": {
            "cf_checkpoint": file_sha256(args.cf_checkpoint),
            "semantic_checkpoint": file_sha256(args.semantic_checkpoint),
            "semantic_embedding": file_sha256(args.semantic_embedding_path),
            "train": file_sha256(data_dir / "train.pkl"),
            "valid": file_sha256(data_dir / "valid.pkl"),
            "test": file_sha256(data_dir / "test.pkl"),
        },
    }
    (output_dir / "teacher_stats.json").write_text(json.dumps(stats_out, indent=2), encoding="utf-8")
    print(json.dumps(stats_out, indent=2), flush=True)


if __name__ == "__main__":
    main()
