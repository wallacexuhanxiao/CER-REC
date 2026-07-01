# CER-Rec

Minimal reproducible pipeline for Amazon Beauty sequential recommendation.

Today's scope:

- Freeze the data protocol.
- Build Amazon Beauty train/valid/test splits and fixed negatives.
- Train and evaluate a standard SASRec baseline.

Out of scope for this milestone: LLM embeddings, semantic experts, gates, and counterfactual teachers.

## Quick Start

```bash
bash scripts/01_preprocess.sh beauty
bash scripts/03_train_sasrec.sh beauty 2026
```

