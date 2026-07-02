#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


SEEDS = (2024, 2025, 2026)


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ranks(scores):
    order = np.argsort(-scores, axis=1)
    return np.where(order == 0)[1] + 1


def metrics_from_scores(scores):
    r = ranks(scores)
    return {
        "HR@10": float((r <= 10).mean()),
        "NDCG@10": float(np.where(r <= 10, 1 / np.log2(r + 1), 0).mean()),
        "ranks": r,
    }


def read_metric(path):
    data = load_json(path)
    if data is None:
        return None
    hr = data.get("test_HR@10", data.get("HR@10"))
    ndcg = data.get("test_NDCG@10", data.get("NDCG@10"))
    if hr is None or ndcg is None:
        return None
    return {"HR@10": float(hr), "NDCG@10": float(ndcg), "raw": data}


def recovery_for(seed, score_path, root):
    score_path = Path(score_path)
    expert_dir = root / f"outputs/beauty/seed{seed}/expert_predictions"
    if not score_path.exists() or not expert_dir.exists():
        return {}
    model_scores = np.load(score_path)
    cf_scores = np.load(expert_dir / "cf_scores.npy")
    sem_scores = np.load(expert_dir / "semantic_scores.npy")
    model = metrics_from_scores(model_scores)
    cf = metrics_from_scores(cf_scores)
    sem = metrics_from_scores(sem_scores)
    model_hit = model["ranks"] <= 10
    cf_hit = cf["ranks"] <= 10
    sem_hit = sem["ranks"] <= 10
    cf_only = cf_hit & ~sem_hit
    sem_only = ~cf_hit & sem_hit
    oracle_hr = float((cf_hit | sem_hit).mean())
    return {
        "CF-Recovery": float((model_hit & cf_only).sum() / cf_only.sum()) if cf_only.any() else None,
        "Semantic-Retention": float((model_hit & sem_only).sum() / sem_only.sum()) if sem_only.any() else None,
        "Oracle-Gap-Capture": float((model["HR@10"] - sem["HR@10"]) / (oracle_hr - sem["HR@10"])) if oracle_hr > sem["HR@10"] else None,
        "cf_only": int(cf_only.sum()),
        "semantic_only": int(sem_only.sum()),
        "oracle_HR@10": oracle_hr,
    }


def rows_to_summary(rows):
    out = []
    grouped = {}
    for row in rows:
        grouped.setdefault(row["Model"], []).append(row)
    for model, vals in grouped.items():
        seeds = [v["Seed"] for v in vals]
        item = {"Model": model, "Seeds": ",".join(str(x) for x in seeds)}
        for key in ("HR@10", "NDCG@10", "CF-Recovery", "Semantic-Retention", "Oracle-Gap-Capture"):
            xs = [v[key] for v in vals if v.get(key) is not None]
            if xs:
                item[f"{key}_mean"] = float(np.mean(xs))
                item[f"{key}_std"] = float(np.std(xs, ddof=0))
                item[key] = f"{np.mean(xs):.4f} ± {np.std(xs, ddof=0):.4f}" if len(xs) > 1 else f"{xs[0]:.4f}"
            else:
                item[f"{key}_mean"] = None
                item[f"{key}_std"] = None
                item[key] = ""
        out.append(item)
    return out


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, fields):
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--tmp-root", default="/root/autodl-tmp/cer-rec")
    parser.add_argument("--output-dir", default="results/beauty")
    args = parser.parse_args()

    root = Path(args.repo_root)
    tmp = Path(args.tmp_root)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    main_specs = {
        "GRU4Rec": lambda s: root / f"outputs/beauty/gru4rec_seed{s}/metrics.json",
        "BERT4Rec": lambda s: root / f"outputs/beauty/bert4rec_seed{s}/metrics.json",
        "SASRec": lambda s: root / f"outputs/beauty/sasrec_seed{s}/metrics.json",
        "Semantic-only SASRec": lambda s: root / f"outputs/beauty/semantic_sasrec_seed{s}/metrics.json",
        "UserTargetGate": lambda s: root / f"outputs/beauty/seed{s}/score_routers/UserTargetGate/metrics.json",
        "EventGate-NoTeacher": lambda s: tmp / f"beauty/seed{s}/event_gate_no_teacher_multiprefix5/learned/metrics.json",
        "CER-Rec": lambda s: tmp / f"beauty/seed{s}/cer_rec_multiprefix5/learned/metrics.json",
    }

    detailed = []
    for model, path_fn in main_specs.items():
        for seed in SEEDS:
            metric = read_metric(path_fn(seed))
            if metric is None:
                continue
            row = {"Model": model, "Seed": seed, "HR@10": metric["HR@10"], "NDCG@10": metric["NDCG@10"]}
            if model in {"UserTargetGate", "EventGate-NoTeacher", "CER-Rec"}:
                row.update(metric["raw"])
                if model in {"EventGate-NoTeacher", "CER-Rec"}:
                    rel = "event_gate_no_teacher_multiprefix5" if model == "EventGate-NoTeacher" else "cer_rec_multiprefix5"
                    row.update(recovery_for(seed, tmp / f"beauty/seed{seed}/{rel}/learned/test_scores.npy", root))
            detailed.append(row)

    summary = rows_to_summary(detailed)
    main_fields = ["Model", "Seeds", "HR@10", "NDCG@10", "CF-Recovery", "Semantic-Retention", "Oracle-Gap-Capture"]
    write_csv(output_dir / "main_results_summary.csv", summary, main_fields)
    write_markdown(output_dir / "main_results_summary.md", summary, main_fields)
    write_csv(output_dir / "main_results_by_seed.csv", detailed, sorted({k for row in detailed for k in row}))

    ablation = []
    for seed in SEEDS:
        calib = load_json(root / f"outputs/beauty/seed{seed}/calibration/calibration_metrics.json")
        if calib and "best_static_fusion" in calib:
            row = {"Model": "StaticFusion", "Seed": seed}
            row["HR@10"] = float(calib["best_static_fusion"]["test_HR@10"])
            row["NDCG@10"] = float(calib["best_static_fusion"]["test_NDCG@10"])
            ablation.append(row)
        for gate in ("UserGate", "ItemGate", "UserTargetGate"):
            metric = read_metric(root / f"outputs/beauty/seed{seed}/score_routers/{gate}/metrics.json")
            if metric:
                row = {"Model": gate, "Seed": seed, "HR@10": metric["HR@10"], "NDCG@10": metric["NDCG@10"]}
                row.update({k: metric["raw"].get(k) for k in ("CF-Recovery", "Semantic-Retention", "Oracle-Gap-Capture")})
                ablation.append(row)
        for model, rel in (
            ("EventGate-NoTeacher", "event_gate_no_teacher_multiprefix5"),
            ("CER-Rec Multi-prefix", "cer_rec_multiprefix5"),
        ):
            metric = read_metric(tmp / f"beauty/seed{seed}/{rel}/learned/metrics.json")
            if metric:
                row = {"Model": model, "Seed": seed, "HR@10": metric["HR@10"], "NDCG@10": metric["NDCG@10"]}
                row.update(recovery_for(seed, tmp / f"beauty/seed{seed}/{rel}/learned/test_scores.npy", root))
                ablation.append(row)

    single_seed_specs = {
        "Teacher single-prefix lambda=0.1": tmp / "beauty/event_gate_cf_teacher/lambda_0.1/learned/metrics.json",
        "Teacher single-prefix lambda=0.3": tmp / "beauty/event_gate_cf_teacher/lambda_0.3/learned/metrics.json",
        "Teacher single-prefix lambda=1.0": tmp / "beauty/event_gate_cf_teacher/lambda_1.0/learned/metrics.json",
        "Teacher multi-prefix early": tmp / "beauty/event_gate_cf_teacher_multiprefix5/learned/metrics.json",
        "WarmStart Teacher": tmp / "beauty/event_gate_cf_teacher_warmstart/learned/metrics.json",
    }
    for model, path in single_seed_specs.items():
        metric = read_metric(path)
        if metric:
            row = {"Model": model, "Seed": 2026, "HR@10": metric["HR@10"], "NDCG@10": metric["NDCG@10"]}
            score_path = path.parent / "test_scores.npy"
            row.update(recovery_for(2026, score_path, root))
            ablation.append(row)

    ablation_summary = rows_to_summary(ablation)
    write_csv(output_dir / "ablation_results_summary.csv", ablation_summary, main_fields)
    write_markdown(output_dir / "ablation_results_summary.md", ablation_summary, main_fields)
    write_csv(output_dir / "ablation_results_by_seed.csv", ablation, sorted({k for row in ablation for k in row}))

    manifest = {
        "dataset": "Amazon Beauty",
        "seeds": list(SEEDS),
        "generated_files": [
            "main_results_summary.csv",
            "main_results_summary.md",
            "main_results_by_seed.csv",
            "ablation_results_summary.csv",
            "ablation_results_summary.md",
            "ablation_results_by_seed.csv",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
