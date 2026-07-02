# External Baseline Preparation

This file tracks the external baselines that should be run after the CER-Rec formal seed runs are stable. Large third-party repositories and downloaded raw datasets must stay on the data disk, not in this Git repository.

## Storage layout

- Third-party source root: `/root/autodl-tmp/cer-rec/external_baselines/`
- Local fallback archive: `/Volumes/sd卡/cer-rec/external_baselines_sources.tgz`
- Server fallback archive target: `/root/autodl-tmp/cer-rec/external_baselines_sources.tgz`
- Raw dataset root on server: `/root/autodl-tmp/cer-rec/raw_datasets/`

## Baselines

| Baseline | Source | Current status | Notes |
| --- | --- | --- | --- |
| LLM-ESR | `https://github.com/Applied-Machine-Learning-Lab/LLM-ESR.git` | Prepared locally; server GitHub clone may timeout | Requires its handled data files: `inter.txt`, `itm_emb_np.pkl`, `usr_emb_np.pkl`, `pca64_itm_emb_np.pkl`, `sim_user_100.pkl`. We should adapt it to CER-Rec frozen split and fixed negatives before reporting. |
| LLM-ESR mirror | `https://github.com/liuqidong07/LLM-ESR.git` | Prepared locally | Same contents as above mirror, kept as fallback. |
| RLMRec | `https://github.com/HKUDS/RLMRec.git` | Prepared locally | Official datasets are Amazon-book/Yelp/Steam. For fair comparison, use CER-Rec split/negatives or clearly label as official-protocol auxiliary result. |
| HSUGA | Paper only / code not found | Pending | No public repository found during preparation. Needs author code or a faithful reimplementation before inclusion. |

## Fair-comparison rule

All external baselines used in the main table must consume the same CER-Rec protocol artifacts where possible:

- `train.pkl`, `valid.pkl`, `test.pkl`
- fixed validation/test candidates or fixed negatives
- identical user/item mapping when reporting Beauty/Fashion/Steam
- metrics: HR@10 and NDCG@10 on the same 101-candidate evaluation

If a baseline cannot be adapted to this protocol, record it separately as an official-protocol reference, not as a directly comparable main-table number.

## Next run checklist

1. Run `bash scripts/prepare_external_baselines.sh` on the server.
2. If GitHub is slow on the server, copy `/Volumes/sd卡/cer-rec/external_baselines_sources.tgz` to `/root/autodl-tmp/cer-rec/external_baselines_sources.tgz` and rerun with `BASELINE_ARCHIVE=/root/autodl-tmp/cer-rec/external_baselines_sources.tgz`.
3. Inspect each baseline's expected data format.
4. Write CER-Rec-to-baseline data adapters without changing the frozen split or fixed negatives.
5. Run only after Beauty/Fashion/Steam CER-Rec formal experiments are locked.
