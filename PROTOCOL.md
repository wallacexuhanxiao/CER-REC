# CER-Rec Data and Evaluation Protocol

## Dataset

- Dataset: Amazon Beauty
- Task: Sequential next-item recommendation
- Primary source priority:
  1. Reuse official LLM-ESR or HSUGA Beauty processed data when available.
  2. Otherwise process Amazon Beauty raw interactions with the same cold-start filtering style used by LLM-ESR/related sequential recommendation code.
- Current implementation source: fallback raw `ratings_Beauty.csv` processing, with a documented source tag in `dataset_stats.json`.

## Split

- Split: leave-one-out by timestamp
- Validation: second-to-last item
- Test: last item
- Max history length: 50
- For each user sequence `(i_1, ..., i_{T-2}, i_{T-1}, i_T)`:
  - Training predicts the next item inside the prefix sequence.
  - Validation uses `(i_1, ..., i_{T-2})` to predict `i_{T-1}`.
  - Test uses `(i_1, ..., i_{T-1})` to predict `i_T`.

## Evaluation

- Candidate set: 1 positive + 100 fixed negatives
- Metrics: HR@10, NDCG@10
- Seeds: 2024, 2025, 2026
- First baseline seed: 2026

## Negative Sampling

- Each validation/test user has 100 fixed negatives.
- Negatives are sampled from items the user never interacted with.
- Negatives never include the positive target.
- Generated negative files are saved and reused by every model.
- Evaluation code must load the saved negative files; it must not resample during evaluation.

## Leakage Rules

- Test target must not appear in the corresponding test history.
- Validation target must not appear in the corresponding validation history.
- Test and validation negatives must not include any user history item or target item.
- Item popularity and all frequency features must be computed from training interactions only.
- User interactions must remain timestamp-sorted.

## SASRec Baseline

- hidden_dim: 64
- max_history_length: 50
- num_layers: 2
- num_heads: 2
- dropout: 0.2
- batch_size: 256
- learning_rate: 0.001
- weight_decay: 0.00001
- train_negatives: 1
- early_stop_patience: 10

