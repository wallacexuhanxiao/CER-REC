import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.evaluation.expert_complementarity import metrics_from_ranks, ranks_from_scores
from src.models.event_gate import DualEventFusion
from src.models.sasrec import SASRec
from src.models.semantic_sasrec import SemanticSASRec


class CandidateDataset(Dataset):
    def __init__(self, split, negatives, max_history_length):
        self.users = list(split.keys())
        histories, candidates, user_lengths = [], [], []
        for uid in self.users:
            sample = split[uid]
            full_history = sample["history"]
            history = full_history[-max_history_length:]
            histories.append([0] * (max_history_length - len(history)) + history)
            candidates.append([sample["target"]] + negatives[uid])
            user_lengths.append(len(full_history))
        self.histories = torch.tensor(histories, dtype=torch.long)
        self.candidates = torch.tensor(candidates, dtype=torch.long)
        self.user_lengths = torch.tensor(user_lengths, dtype=torch.float32)

    def __len__(self):
        return self.histories.shape[0]

    def __getitem__(self, index):
        return self.histories[index], self.candidates[index], self.user_lengths[index]


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_train_candidates(train, valid, test, num_items, num_negatives, seed):
    rng = random.Random(seed)
    all_items = list(range(1, num_items + 1))
    split, negatives = {}, {}
    for uid, seq in train.items():
        if len(seq) < 2:
            continue
        split[uid] = {"history": seq[:-1], "target": seq[-1]}
        forbidden = set(seq) | {valid[uid]["target"], test[uid]["target"]}
        pool = [item for item in all_items if item not in forbidden and item != seq[-1]]
        negatives[uid] = rng.sample(pool, num_negatives)
    return split, negatives


def build_frequency_tensor(data_dir, num_items):
    popularity = load_pickle(Path(data_dir) / "train_item_popularity.pkl")
    values = torch.zeros(num_items + 1, dtype=torch.float32)
    max_log = 1.0
    if popularity:
        max_log = np.log1p(max(popularity.values()))
    for item, count in popularity.items():
        values[item] = float(np.log1p(count) / max_log)
    return values, popularity


def recency_features(seq):
    mask = seq.ne(0)
    positions = mask.long().cumsum(dim=1).float()
    lengths = positions.max(dim=1, keepdim=True).values.clamp_min(1.0)
    return (positions / lengths).masked_fill(~mask, 0.0)


@torch.no_grad()
def frozen_features(cf_model, sem_model, seq, candidates):
    cf_states = cf_model.encode(seq)
    sem_states = sem_model.encode(seq)
    cf_candidates = cf_model.item_embedding(candidates)
    sem_candidates = sem_model.project_items(candidates)
    return cf_states.detach(), sem_states.detach(), cf_candidates.detach(), sem_candidates.detach()


def run_model(model, cf_model, sem_model, loader, freq_tensor, args, device, return_gates=False):
    model.eval()
    all_scores, all_gate_means, all_gates, all_masks, all_candidates = [], [], [], [], []
    with torch.no_grad():
        for seq, candidates, user_lengths in loader:
            seq, candidates, user_lengths = seq.to(device), candidates.to(device), user_lengths.to(device)
            cf_states, sem_states, cf_candidates, sem_candidates = frozen_features(cf_model, sem_model, seq, candidates)
            history_mask = seq.ne(0)
            target_freq = freq_tensor[candidates]
            hist_freq = freq_tensor[seq]
            user_len = torch.log1p(user_lengths) / np.log(51)
            recency = recency_features(seq)
            scores, gates = model(
                cf_states,
                sem_states,
                cf_candidates,
                sem_candidates,
                history_mask,
                target_freq,
                hist_freq,
                user_len,
                recency,
                args.cf_temperature,
                args.semantic_temperature,
            )
            all_scores.append(scores.cpu().numpy().astype(np.float32))
            target_gate = gates[:, 0, :]
            gate_mean = (target_gate * history_mask.float()).sum(dim=1) / history_mask.float().sum(dim=1).clamp_min(1.0)
            all_gate_means.append(gate_mean.cpu().numpy().astype(np.float32))
            if return_gates:
                all_gates.append(target_gate.cpu().numpy().astype(np.float32))
                all_masks.append(history_mask.cpu().numpy())
                all_candidates.append(candidates.cpu().numpy().astype(np.int32))
    result = {"scores": np.vstack(all_scores), "target_gate_mean": np.concatenate(all_gate_means)}
    if return_gates:
        result["target_event_gates"] = np.vstack(all_gates)
        result["history_masks"] = np.vstack(all_masks)
        result["candidate_ids"] = np.vstack(all_candidates)
    return result


def train_one(args, route_mode):
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / route_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads((data_dir / "dataset_stats.json").read_text())
    train = load_pickle(data_dir / "train.pkl")
    valid = load_pickle(data_dir / "valid.pkl")
    test = load_pickle(data_dir / "test.pkl")
    valid_neg = load_pickle(data_dir / "valid_negatives.pkl")
    test_neg = load_pickle(data_dir / "test_negatives.pkl")
    train_split, train_neg = sample_train_candidates(train, valid, test, stats["num_items"], args.train_negatives, args.seed + 503)
    train_ds = CandidateDataset(train_split, train_neg, args.max_history_length)
    valid_ds = CandidateDataset(valid, valid_neg, args.max_history_length)
    test_ds = CandidateDataset(test, test_neg, args.max_history_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.eval_batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cf_model = SASRec(stats["num_items"], 64, args.max_history_length, 2, 2, 0.2).to(device)
    cf_model.load_state_dict(torch.load(args.cf_checkpoint, map_location=device)["model"])
    sem_model = SemanticSASRec(args.semantic_embedding_path, 64, 256, args.max_history_length, 2, 2, 0.2).to(device)
    sem_model.load_state_dict(torch.load(args.semantic_checkpoint, map_location=device)["model"])
    cf_model.eval()
    sem_model.eval()
    for param in cf_model.parameters():
        param.requires_grad_(False)
    for param in sem_model.parameters():
        param.requires_grad_(False)

    freq_tensor, _ = build_frequency_tensor(data_dir, stats["num_items"])
    freq_tensor = freq_tensor.to(device)
    model = DualEventFusion(
        hidden_dim=64,
        relation_hidden_dim=args.relation_hidden_dim,
        route_hidden_dim=args.route_hidden_dim,
        dropout=args.dropout,
        route_mode=route_mode,
        candidate_chunk_size=args.candidate_chunk_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_ndcg, best_state, best_epoch, patience = -1.0, None, 0, 0
    logs = []
    print(json.dumps({"route_mode": route_mode, "device": str(device), "num_batches": len(train_loader)}), flush=True)
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for seq, candidates, user_lengths in train_loader:
            seq, candidates, user_lengths = seq.to(device), candidates.to(device), user_lengths.to(device)
            with torch.no_grad():
                cf_states, sem_states, cf_candidates, sem_candidates = frozen_features(cf_model, sem_model, seq, candidates)
            history_mask = seq.ne(0)
            scores, _ = model(
                cf_states,
                sem_states,
                cf_candidates,
                sem_candidates,
                history_mask,
                freq_tensor[candidates],
                freq_tensor[seq],
                torch.log1p(user_lengths) / np.log(51),
                recency_features(seq),
                args.cf_temperature,
                args.semantic_temperature,
            )
            loss = F.cross_entropy(scores, torch.zeros(scores.shape[0], dtype=torch.long, device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        valid_output = run_model(model, cf_model, sem_model, valid_loader, freq_tensor, args, device)
        valid_ranks = ranks_from_scores(valid_output["scores"])
        valid_hr, valid_ndcg = metrics_from_ranks(valid_ranks)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_HR@10": valid_hr, "valid_NDCG@10": valid_ndcg}
        logs.append(row)
        print(json.dumps({"route_mode": route_mode, **row}), flush=True)
        if valid_ndcg > best_ndcg:
            best_ndcg, best_epoch, patience = valid_ndcg, epoch, 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"model": best_state, "args": vars(args)}, output_dir / "best.pt")
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                break

    model.load_state_dict(best_state)
    test_output = run_model(model, cf_model, sem_model, test_loader, freq_tensor, args, device, return_gates=True)
    ranks = ranks_from_scores(test_output["scores"])
    hr, ndcg = metrics_from_ranks(ranks)
    metrics = {
        "route_mode": route_mode,
        "best_epoch": best_epoch,
        "HR@10": hr,
        "NDCG@10": ndcg,
    }
    (output_dir / "train_log.jsonl").write_text("\n".join(json.dumps(x) for x in logs) + "\n")
    np.save(output_dir / "test_scores.npy", test_output["scores"])
    np.save(output_dir / "target_event_gates.npy", test_output["target_event_gates"])
    np.save(output_dir / "history_masks.npy", test_output["history_masks"])
    save_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def write_manifest(args):
    output_path = Path(args.output_dir) / "stage2_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cf_checkpoint": args.cf_checkpoint,
        "semantic_checkpoint": args.semantic_checkpoint,
        "semantic_embedding_path": args.semantic_embedding_path,
        "data_dir": args.data_dir,
        "cf_temperature": args.cf_temperature,
        "semantic_temperature": args.semantic_temperature,
        "checksums": {
            "cf_checkpoint": file_sha256(args.cf_checkpoint),
            "semantic_checkpoint": file_sha256(args.semantic_checkpoint),
            "semantic_embedding": file_sha256(args.semantic_embedding_path),
            "train": file_sha256(Path(args.data_dir) / "train.pkl"),
            "valid": file_sha256(Path(args.data_dir) / "valid.pkl"),
            "test": file_sha256(Path(args.data_dir) / "test.pkl"),
            "valid_negatives": file_sha256(Path(args.data_dir) / "valid_negatives.pkl"),
            "test_negatives": file_sha256(Path(args.data_dir) / "test_negatives.pkl"),
        },
    }
    save_json(output_path, manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/beauty")
    parser.add_argument("--cf-checkpoint", default="outputs/beauty/sasrec_seed2026/best.pt")
    parser.add_argument("--semantic-checkpoint", default="outputs/beauty/semantic_sasrec_seed2026/best.pt")
    parser.add_argument("--semantic-embedding-path", default="data/processed/beauty/item_semantic_embeddings.fp16.npy")
    parser.add_argument("--output-dir", default="outputs/beauty/event_gate")
    parser.add_argument("--route-mode", choices=["fixed_half", "learned", "both"], default="both")
    parser.add_argument("--max-history-length", type=int, default=50)
    parser.add_argument("--train-negatives", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--relation-hidden-dim", type=int, default=64)
    parser.add_argument("--route-hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--candidate-chunk-size", type=int, default=16)
    parser.add_argument("--cf-temperature", type=float, default=1.5190560817718506)
    parser.add_argument("--semantic-temperature", type=float, default=1.35512113571167)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    write_manifest(args)
    modes = ["fixed_half", "learned"] if args.route_mode == "both" else [args.route_mode]
    summary = {mode: train_one(args, mode) for mode in modes}
    save_json(Path(args.output_dir) / "summary.json", summary)


if __name__ == "__main__":
    main()

