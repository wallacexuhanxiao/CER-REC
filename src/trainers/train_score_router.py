import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.evaluation.expert_complementarity import metrics_from_ranks, ranks_from_scores
from src.models.score_fusion import ItemGate, UserGate, UserTargetGate


class FeatureDataset(Dataset):
    def __init__(self, split, user_features, item_features, item_ids, cf_scores, sem_scores):
        self.user_features = user_features.astype(np.float32)
        self.item_features = item_features.astype(np.float32)
        self.item_ids = item_ids.astype(np.int64)
        self.cf_scores = cf_scores.astype(np.float32)
        self.sem_scores = sem_scores.astype(np.float32)

    def __len__(self):
        return self.cf_scores.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.user_features[idx]),
            torch.from_numpy(self.item_features[idx]),
            torch.from_numpy(self.cf_scores[idx]),
            torch.from_numpy(self.sem_scores[idx]),
        )


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def load_json(path):
    return json.loads(Path(path).read_text())


def make_user_features(split_npz):
    cf_user = split_npz["cf_user"]
    sem_user = split_npz["semantic_user"]
    hist = split_npz["history_length"][:, None]
    hist_scaled = np.log1p(hist) / np.log(51)
    diff = np.abs(cf_user - sem_user)
    return np.concatenate([hist_scaled, hist_scaled, cf_user, sem_user, diff], axis=1)


def bucket_one_hot(values):
    order = np.argsort(values, kind="stable")
    labels = np.zeros(len(values), dtype=np.int64)
    n = len(values)
    labels[order[n // 3 : (2 * n) // 3]] = 1
    labels[order[(2 * n) // 3 :]] = 2
    return np.eye(3, dtype=np.float32)[labels]


def make_item_feature_table(feature_dir, data_dir):
    cf_item = np.load(feature_dir / "cf_item_embeddings.npy")
    sem_item = np.load(feature_dir / "semantic_item_embeddings.npy")
    train_pop = load_pickle(data_dir / "train_item_popularity.pkl")
    item_texts = [json.loads(line) for line in (data_dir / "item_texts.jsonl").read_text(encoding="utf-8").splitlines()]
    num_items = cf_item.shape[0] - 1
    freqs = np.asarray([0] + [train_pop.get(item, 0) for item in range(1, num_items + 1)], dtype=np.float32)
    has_text = np.zeros(num_items + 1, dtype=np.float32)
    for row in item_texts:
        has_text[row["item_id"]] = row["has_text"]
    buckets = np.vstack([np.zeros((1, 3), dtype=np.float32), bucket_one_hot(freqs[1:])])
    freq_scaled = (np.log1p(freqs) / max(1.0, np.log1p(freqs.max())))[:, None]
    return np.concatenate([freq_scaled, buckets, has_text[:, None], cf_item, sem_item], axis=1).astype(np.float32)


def item_features_for_candidates(item_table, candidate_ids, cf_scores, sem_scores, cf_temp, sem_temp, include_scores):
    feats = item_table[candidate_ids]
    if include_scores:
        score_feats = np.stack([cf_scores / cf_temp, sem_scores / sem_temp], axis=-1)
        feats = np.concatenate([feats, score_feats.astype(np.float32)], axis=-1)
    return feats


def eval_scores(scores):
    ranks = ranks_from_scores(scores)
    return ranks, metrics_from_ranks(ranks)


def gate_diagnostics(gates, cf_hits, sem_hits, candidate_ids, train_pop):
    target_gate = gates[:, 0]
    freqs = np.asarray([train_pop.get(int(item), 0) for item in candidate_ids[:, 0]])
    order = np.argsort(freqs, kind="stable")
    n = len(freqs)
    masks = {
        "tail_mean_gate": order[: n // 3],
        "mid_mean_gate": order[n // 3 : (2 * n) // 3],
        "head_mean_gate": order[(2 * n) // 3 :],
    }
    result = {
        "mean_gate": float(target_gate.mean()),
        "gate_std": float(target_gate.std()),
        "cf_only_mean_gate": float(target_gate[cf_hits & ~sem_hits].mean()) if np.any(cf_hits & ~sem_hits) else None,
        "semantic_only_mean_gate": float(target_gate[~cf_hits & sem_hits].mean()) if np.any(~cf_hits & sem_hits) else None,
    }
    for key, idx in masks.items():
        result[key] = float(target_gate[idx].mean())
    return result


def recovery_metrics(router_scores, cf_scores, sem_scores):
    router_ranks = ranks_from_scores(router_scores)
    cf_ranks = ranks_from_scores(cf_scores)
    sem_ranks = ranks_from_scores(sem_scores)
    router_hits = router_ranks <= 10
    cf_hits = cf_ranks <= 10
    sem_hits = sem_ranks <= 10
    cf_only = cf_hits & ~sem_hits
    sem_only = ~cf_hits & sem_hits
    oracle_hr = metrics_from_ranks(np.minimum(cf_ranks, sem_ranks))[0]
    sem_hr = metrics_from_ranks(sem_ranks)[0]
    router_hr, router_ndcg = metrics_from_ranks(router_ranks)
    denom = max(1e-12, oracle_hr - sem_hr)
    return {
        "HR@10": router_hr,
        "NDCG@10": router_ndcg,
        "CF-Recovery": float((router_hits & cf_only).sum() / max(1, cf_only.sum())),
        "Semantic-Retention": float((router_hits & sem_only).sum() / max(1, sem_only.sum())),
        "Oracle-Gap-Capture": float((router_hr - sem_hr) / denom),
    }, cf_hits, sem_hits


def predict_router(model, loader, device, gate_type, cf_temp, sem_temp):
    model.eval()
    scores, gates = [], []
    with torch.no_grad():
        for user_feat, item_feat, cf_scores, sem_scores in loader:
            user_feat, item_feat = user_feat.to(device), item_feat.to(device)
            cf_scores, sem_scores = cf_scores.to(device), sem_scores.to(device)
            g = model(user_feat, item_feat)
            fused = g * (cf_scores / cf_temp) + (1 - g) * (sem_scores / sem_temp)
            scores.append(fused.cpu().numpy())
            gates.append(g.expand_as(fused).cpu().numpy() if gate_type == "user" else g.cpu().numpy())
    return np.vstack(scores), np.vstack(gates)


def train_one(gate_type, train_ds, valid_ds, test_ds, valid_raw, test_raw, train_pop, args, dims):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gate_type == "user":
        model = UserGate(dims["user"]).to(device)
    elif gate_type == "item":
        model = ItemGate(dims["item"]).to(device)
    else:
        model = UserTargetGate(dims["user"] + dims["item"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.eval_batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size)
    best_ndcg, best_state, patience = -1.0, None, 0
    logs = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for user_feat, item_feat, cf_scores, sem_scores in train_loader:
            user_feat, item_feat = user_feat.to(device), item_feat.to(device)
            cf_scores, sem_scores = cf_scores.to(device), sem_scores.to(device)
            opt.zero_grad()
            g = model(user_feat, item_feat)
            fused = g * (cf_scores / args.cf_temperature) + (1 - g) * (sem_scores / args.semantic_temperature)
            loss = F.cross_entropy(fused, torch.zeros(fused.shape[0], dtype=torch.long, device=device))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        valid_scores, _ = predict_router(model, valid_loader, device, gate_type, args.cf_temperature, args.semantic_temperature)
        ranks, (_, ndcg) = eval_scores(valid_scores)
        hr = float((ranks <= 10).mean())
        logs.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_HR@10": hr, "valid_NDCG@10": ndcg})
        print(json.dumps({"gate": gate_type, **logs[-1]}), flush=True)
        if ndcg > best_ndcg:
            best_ndcg, best_state, patience = ndcg, {k: v.detach().cpu() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                break
    model.load_state_dict(best_state)
    test_scores, test_gates = predict_router(model, test_loader, device, gate_type, args.cf_temperature, args.semantic_temperature)
    metrics, cf_hits, sem_hits = recovery_metrics(test_scores, test_raw["cf_scores"], test_raw["semantic_scores"])
    metrics.update(gate_diagnostics(test_gates, cf_hits, sem_hits, test_raw["candidate_ids"], train_pop))
    return model, logs, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", default="outputs/beauty/expert_features")
    parser.add_argument("--data-dir", default="data/processed/beauty")
    parser.add_argument("--calibration-dir", default="outputs/beauty/calibration")
    parser.add_argument("--output-dir", default="outputs/beauty/score_routers")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=50)
    args = parser.parse_args()
    feature_dir, data_dir, output_dir = Path(args.feature_dir), Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.cf_temperature = load_json(Path(args.calibration_dir) / "cf_temperature.json")["temperature"]
    args.semantic_temperature = load_json(Path(args.calibration_dir) / "semantic_temperature.json")["temperature"]
    train_raw = dict(np.load(feature_dir / "train.npz"))
    valid_raw = dict(np.load(feature_dir / "valid.npz"))
    test_raw = dict(np.load(feature_dir / "test.npz"))
    train_pop = load_pickle(data_dir / "train_item_popularity.pkl")
    item_table = make_item_feature_table(feature_dir, data_dir)

    user_train, user_valid, user_test = make_user_features(train_raw), make_user_features(valid_raw), make_user_features(test_raw)
    item_train = item_features_for_candidates(item_table, train_raw["candidate_ids"], train_raw["cf_scores"], train_raw["semantic_scores"], args.cf_temperature, args.semantic_temperature, False)
    item_valid = item_features_for_candidates(item_table, valid_raw["candidate_ids"], valid_raw["cf_scores"], valid_raw["semantic_scores"], args.cf_temperature, args.semantic_temperature, False)
    item_test = item_features_for_candidates(item_table, test_raw["candidate_ids"], test_raw["cf_scores"], test_raw["semantic_scores"], args.cf_temperature, args.semantic_temperature, False)
    ut_item_train = item_features_for_candidates(item_table, train_raw["candidate_ids"], train_raw["cf_scores"], train_raw["semantic_scores"], args.cf_temperature, args.semantic_temperature, True)
    ut_item_valid = item_features_for_candidates(item_table, valid_raw["candidate_ids"], valid_raw["cf_scores"], valid_raw["semantic_scores"], args.cf_temperature, args.semantic_temperature, True)
    ut_item_test = item_features_for_candidates(item_table, test_raw["candidate_ids"], test_raw["cf_scores"], test_raw["semantic_scores"], args.cf_temperature, args.semantic_temperature, True)

    results = {}
    cf_cal = test_raw["cf_scores"] / args.cf_temperature
    sem_cal = test_raw["semantic_scores"] / args.semantic_temperature
    static = load_json(Path(args.calibration_dir) / "calibration_metrics.json")["best_static_fusion"]["g"]
    static_scores = static * cf_cal + (1 - static) * sem_cal
    static_metrics, cf_hits, sem_hits = recovery_metrics(static_scores, test_raw["cf_scores"], test_raw["semantic_scores"])
    static_gates = np.full_like(static_scores, static, dtype=np.float32)
    static_metrics.update(gate_diagnostics(static_gates, cf_hits, sem_hits, test_raw["candidate_ids"], train_pop))
    results["StaticFusion"] = static_metrics | {"g": static}

    specs = {
        "UserGate": ("user", user_train, item_train, user_valid, item_valid, user_test, item_test),
        "ItemGate": ("item", user_train, item_train, user_valid, item_valid, user_test, item_test),
        "UserTargetGate": ("user_target", user_train, ut_item_train, user_valid, ut_item_valid, user_test, ut_item_test),
    }
    for name, (gate_type, u_tr, i_tr, u_va, i_va, u_te, i_te) in specs.items():
        train_ds = FeatureDataset("train", u_tr, i_tr, train_raw["candidate_ids"], train_raw["cf_scores"], train_raw["semantic_scores"])
        valid_ds = FeatureDataset("valid", u_va, i_va, valid_raw["candidate_ids"], valid_raw["cf_scores"], valid_raw["semantic_scores"])
        test_ds = FeatureDataset("test", u_te, i_te, test_raw["candidate_ids"], test_raw["cf_scores"], test_raw["semantic_scores"])
        dims = {"user": u_tr.shape[1], "item": i_tr.shape[2]}
        model, logs, metrics = train_one(gate_type, train_ds, valid_ds, test_ds, valid_raw, test_raw, train_pop, args, dims)
        gate_dir = output_dir / name
        gate_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), gate_dir / "best.pt")
        (gate_dir / "train_log.jsonl").write_text("\n".join(json.dumps(x) for x in logs) + "\n")
        (gate_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        results[name] = metrics
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

