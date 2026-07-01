import math


def hr_ndcg_at_k(ranked_items, target, k=10):
    topk = ranked_items[:k]
    if target not in topk:
        return 0.0, 0.0
    rank = topk.index(target)
    return 1.0, 1.0 / math.log2(rank + 2)

