import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_pickle(path, obj):
    with Path(path).open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/beauty")
    parser.add_argument("--embedding-path", default="data/processed/beauty/item_semantic_embeddings.fp16.npy")
    parser.add_argument("--output-path", default="data/processed/beauty/test_negatives_semantic_hard.pkl")
    parser.add_argument("--random-count", type=int, default=50)
    parser.add_argument("--hard-count", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=1024)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    test = load_pickle(data_dir / "test.pkl")
    random_negatives = load_pickle(data_dir / "test_negatives.pkl")
    embeddings = np.load(args.embedding_path).astype(np.float32)
    item_matrix = torch.from_numpy(embeddings).cuda() if torch.cuda.is_available() else torch.from_numpy(embeddings)
    item_matrix[0].zero_()
    users = list(test.keys())
    output = {}
    for start in range(0, len(users), args.chunk_size):
        batch_users = users[start : start + args.chunk_size]
        targets = torch.tensor([test[uid]["target"] for uid in batch_users], dtype=torch.long, device=item_matrix.device)
        sims = item_matrix[targets] @ item_matrix.T
        for row_idx, uid in enumerate(batch_users):
            forbidden = set(test[uid]["history"]) | {test[uid]["target"]} | set(random_negatives[uid][: args.random_count]) | {0}
            sims[row_idx, list(forbidden)] = -1e9
        topk = torch.topk(sims, k=args.hard_count, dim=1).indices.detach().cpu().numpy()
        for uid, hard in zip(batch_users, topk):
            output[uid] = list(random_negatives[uid][: args.random_count]) + [int(x) for x in hard.tolist()]
            assert len(output[uid]) == args.random_count + args.hard_count
            assert test[uid]["target"] not in output[uid]
            assert not (set(output[uid]) & set(test[uid]["history"]))
    save_pickle(args.output_path, output)
    result = {"num_users": len(output), "random_count": args.random_count, "hard_count": args.hard_count, "output_path": args.output_path}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

