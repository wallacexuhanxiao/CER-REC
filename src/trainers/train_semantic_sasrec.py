import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation.evaluate import evaluate_model
from src.models.semantic_sasrec import SemanticSASRec


class SequenceDataset(Dataset):
    def __init__(self, train, valid, test, num_items, max_history_length, seed):
        inputs, positives, negatives_all = [], [], []
        rng = random.Random(seed)
        for uid, seq in train.items():
            if len(seq) < 2:
                continue
            seq = seq[-(max_history_length + 1) :]
            input_seq = seq[:-1]
            pos_seq = seq[1:]
            pad_len = max_history_length - len(input_seq)
            input_seq = [0] * pad_len + input_seq
            pos_seq = [0] * pad_len + pos_seq
            forbidden = set(seq)
            if uid in valid:
                forbidden.add(valid[uid]["target"])
            if uid in test:
                forbidden.add(test[uid]["target"])
            neg_seq = [0] * pad_len
            for target in pos_seq[pad_len:]:
                while True:
                    item = rng.randint(1, num_items)
                    if item not in forbidden and item != target:
                        neg_seq.append(item)
                        break
            inputs.append(input_seq)
            positives.append(pos_seq)
            negatives_all.append(neg_seq)
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.positives = torch.tensor(positives, dtype=torch.long)
        self.negatives = torch.tensor(negatives_all, dtype=torch.long)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return self.inputs[index], self.positives[index], self.negatives[index]


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


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


def projected_embedding_stats(model, device):
    rng = torch.Generator(device=device).manual_seed(2026)
    item_ids = torch.randint(1, model.semantic_embedding_table.shape[0], (1024,), generator=rng, device=device)
    with torch.no_grad():
        projected = model.project_items(item_ids)
    norms = projected.norm(dim=-1)
    return {
        "projected_variance": float(projected.var().item()),
        "projected_norm_mean": float(norms.mean().item()),
        "projected_norm_max": float(norms.max().item()),
        "projected_pairwise_std": float(projected.std(dim=0).mean().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/beauty")
    parser.add_argument("--embedding-path", default="data/processed/beauty/item_semantic_embeddings.fp16.npy")
    parser.add_argument("--output-dir", default="outputs/beauty/semantic_sasrec_seed2026")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--projection-hidden-dim", type=int, default=256)
    parser.add_argument("--max-history-length", type=int, default=50)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_checksum_before = file_sha256(args.embedding_path)

    train = load_pickle(data_dir / "train.pkl")
    valid = load_pickle(data_dir / "valid.pkl")
    test = load_pickle(data_dir / "test.pkl")
    valid_negatives = load_pickle(data_dir / "valid_negatives.pkl")
    test_negatives = load_pickle(data_dir / "test_negatives.pkl")
    stats = json.loads((data_dir / "dataset_stats.json").read_text())
    num_items = stats["num_items"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SequenceDataset(train, valid, test, num_items, args.max_history_length, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = SemanticSASRec(
        semantic_embedding_path=args.embedding_path,
        hidden_dim=args.hidden_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        max_history_length=args.max_history_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    assert model.semantic_embedding_table.requires_grad is False
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_ndcg, best_epoch, patience = -1.0, 0, 0
    history = []
    print(json.dumps({"device": str(device), "num_training_sequences": len(dataset), "num_batches": len(loader)}), flush=True)

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for seq, pos, neg in loader:
            seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()
            pos_logits, neg_logits = model.sequence_logits(seq, pos, neg)
            mask = pos.gt(0)
            loss = criterion(pos_logits[mask], torch.ones_like(pos_logits[mask]))
            loss = loss + criterion(neg_logits[mask], torch.zeros_like(neg_logits[mask]))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        valid_metrics = evaluate_model(model, valid, valid_negatives, args.max_history_length, device, batch_size=args.eval_batch_size)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_HR@10": valid_metrics["HR@10"],
            "validation_NDCG@10": valid_metrics["NDCG@10"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if valid_metrics["NDCG@10"] > best_ndcg:
            if not all(torch.isfinite(param).all() for param in model.parameters()):
                raise FloatingPointError("model parameters contain NaN or Inf")
            best_ndcg, best_epoch, patience = valid_metrics["NDCG@10"], epoch, 0
            torch.save({"model": model.state_dict(), "args": vars(args), "stats": stats}, output_dir / "best.pt")
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    valid_metrics = evaluate_model(model, valid, valid_negatives, args.max_history_length, device, batch_size=args.eval_batch_size)
    test_metrics = evaluate_model(model, test, test_negatives, args.max_history_length, device, batch_size=args.eval_batch_size)
    embedding_checksum_after = file_sha256(args.embedding_path)
    assert embedding_checksum_before == embedding_checksum_after
    collapse_stats = projected_embedding_stats(model, device)
    summary = {
        "seed": args.seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_NDCG@10": best_ndcg,
        "validation_HR@10": valid_metrics["HR@10"],
        "validation_NDCG@10": valid_metrics["NDCG@10"],
        "test_HR@10": test_metrics["HR@10"],
        "test_NDCG@10": test_metrics["NDCG@10"],
        "semantic_embedding_frozen": True,
        "embedding_checksum_before": embedding_checksum_before,
        "embedding_checksum_after": embedding_checksum_after,
        **collapse_stats,
    }
    (output_dir / "train_log.jsonl").write_text("\n".join(json.dumps(x) for x in history) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "config.yaml").write_text("\n".join(f"{k}: {v}" for k, v in vars(args).items()) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

