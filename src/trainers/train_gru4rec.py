import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.evaluate import evaluate_model
from src.models.gru4rec import GRU4Rec
from src.trainers.train_sasrec import SequenceDataset, load_pickle, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/beauty")
    parser.add_argument("--output-dir", default="outputs/beauty/gru4rec_seed2026")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-history-length", type=int, default=50)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--train-negatives", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.train_negatives != 1:
        raise ValueError("This trainer currently supports train_negatives=1.")

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
    dataset = SequenceDataset(train, valid, test, num_items, args.max_history_length, args.train_negatives, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(json.dumps({"model": "GRU4Rec", "device": str(device), "num_training_sequences": len(dataset), "num_batches": len(loader)}), flush=True)

    model = GRU4Rec(num_items, args.hidden_dim, args.num_layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_ndcg = -1.0
    best_epoch = 0
    patience = 0
    history = []
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
        "model": "GRU4Rec",
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
