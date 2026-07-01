import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/beauty")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-text-tokens", type=int, default=256)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    processed_dir = Path(args.processed_dir)
    item2id_path = processed_dir / "item2id.json"
    stats = json.loads((processed_dir / "dataset_stats.json").read_text())
    rows = [json.loads(line) for line in (processed_dir / "item_texts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == stats["num_items"]
    for expected_id, row in enumerate(rows, start=1):
        assert row["item_id"] == expected_id

    texts = [row["text"] for row in rows]
    model = SentenceTransformer(args.model, trust_remote_code=True)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float16)
    embeddings = np.vstack([np.zeros((1, embeddings.shape[1]), dtype=np.float16), embeddings])
    assert embeddings.shape[0] == stats["num_items"] + 1
    assert not np.isnan(embeddings).any()
    assert np.allclose(embeddings[0], 0)

    output_path = processed_dir / "item_semantic_embeddings.fp16.npy"
    np.save(output_path, embeddings)
    meta = {
        "model": "Qwen3-Embedding-0.6B",
        "model_id": args.model,
        "num_items": stats["num_items"],
        "embedding_dim": int(embeddings.shape[1]),
        "dtype": "float16",
        "normalized": True,
        "max_text_tokens": args.max_text_tokens,
        "item_mapping_checksum": file_sha256(item2id_path),
        "embedding_file": str(output_path),
    }
    (processed_dir / "item_semantic_embedding_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rng = np.random.default_rng(2026)
    sample_pairs = []
    for _ in range(5):
        a, b = rng.choice(np.arange(1, embeddings.shape[0]), size=2, replace=False)
        sim = float(np.dot(embeddings[a].astype(np.float32), embeddings[b].astype(np.float32)))
        sample_pairs.append({"item_a": int(a), "item_b": int(b), "cosine": sim})
    print(json.dumps({"meta": meta, "sample_cosine_pairs": sample_pairs}, indent=2))


if __name__ == "__main__":
    main()

