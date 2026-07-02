import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation.evaluate import evaluate_model
from src.models.bert4rec import BERT4Rec
from src.trainers.train_sasrec import load_pickle, set_seed


class MaskedSequenceDataset(Dataset):
    def __init__(self, train, max_history_length, mask_token_id, mask_prob, seed):
        self.rows = []
        rng = random.Random(seed)
        for seq in train.values():
            if len(seq) < 2:
                continue
            seq = seq[-max_history_length:]
            pad_len = max_history_length - len(seq)
            padded = [0] * pad_len + seq
            valid_positions = [idx for idx, item in enumerate(padded) if item > 0]
            masked_positions = [idx for idx in valid_positions if rng.random() < mask_prob]
            if not masked_positions:
                masked_positions = [rng.choice(valid_positions)]
            labels = [-100] * max_history_length
            inputs = list(padded)
            for idx in masked_positions:
                labels[idx] = inputs[idx]
                inputs[idx] = mask_token_id
            self.rows.append((inputs, labels))

        self.inputs = torch.tensor([x for x, _ in self.rows], dtype=torch.long)
        self.labels = torch.tensor([y for _, y in self.rows], dtype=torch.long)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/beauty")
    parser.add_argument("--output-dir", default="outputs/beauty/bert4rec_seed2026")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-history-length", type=int, default=50)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mask-prob", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = load_pickle(data_dir / "train.pkl")
    valid = load_pickle(data_dir / "valid.pkl")
    test = load_pickle(data_dir / "test.pkl")
    valid_negatives = load_pickle(data_dir / "valid_negatives.pkl")
    test_negatives = load_pickle(data_dir / "test_negatives.pkl")
    stats = json.loads((data_dir / "dataset_stats.json").read_text())
    num_items = stats["num_items"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERT4Rec(
        num_items=num_items,
        hidden_dim=args.hidden_dim,
        max_history_length=args.max_history_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    dataset = MaskedSequenceDataset(train, args.max_history_length, model.mask_token_id, args.mask_prob, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(json.dumps({"model": "BERT4Rec", "device": str(device), "num_training_sequences": len(dataset), "num_batches": len(loader)}), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    best_ndcg = -1.0
    best_epoch = 0
    patience = 0
    history = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for seq, labels in loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model.logits(seq)
            loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
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
            best_ndcg = valid_metrics["NDCG@10"]
            best_epoch = epoch
            patience = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "stats": stats}, output_dir / "best.pt")
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    valid_metrics = evaluate_model(model, valid, valid_negatives, args.max_history_length, device, batch_size=args.eval_batch_size)
    test_metrics = evaluate_model(model, test, test_negatives, args.max_history_length, device, batch_size=args.eval_batch_size)
    summary = {
        "model": "BERT4Rec",
        "seed": args.seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_NDCG@10": best_ndcg,
        "validation_HR@10": valid_metrics["HR@10"],
        "validation_NDCG@10": valid_metrics["NDCG@10"],
        "test_HR@10": test_metrics["HR@10"],
        "test_NDCG@10": test_metrics["NDCG@10"],
    }
    (output_dir / "train_log.jsonl").write_text("\n".join(json.dumps(x) for x in history) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
