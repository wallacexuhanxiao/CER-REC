import argparse
import ast
import gzip
import json
from pathlib import Path
from urllib.request import urlretrieve


META_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz"
FALLBACK_TEXT = "Unknown beauty product"


def download_metadata(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "meta_Beauty.json.gz"
    if path.exists():
        return path
    print(f"Downloading Amazon Beauty metadata from {META_URL}")
    urlretrieve(META_URL, path)
    return path


def clean_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"none", "nan", "null"}:
            return ""
        return " ".join(text.split())
    if isinstance(value, (list, tuple)):
        return " ".join(clean_text(v) for v in value if clean_text(v)).strip()
    return clean_text(str(value))


def category_path(value):
    if not value:
        return ""
    paths = []
    for path in value:
        if isinstance(path, (list, tuple)):
            text = " > ".join(clean_text(x) for x in path if clean_text(x))
        else:
            text = clean_text(path)
        if text:
            paths.append(text)
    return " | ".join(paths)


def load_metadata(path: Path, wanted_raw_ids):
    meta = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            row = ast.literal_eval(line)
            asin = row.get("asin")
            if asin in wanted_raw_ids:
                meta[asin] = row
    return meta


def build_text(row):
    title = clean_text(row.get("title"))
    brand = clean_text(row.get("brand"))
    category = category_path(row.get("categories"))
    features = clean_text(row.get("feature") or row.get("features"))
    description = clean_text(row.get("description"))
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if brand:
        parts.append(f"Brand: {brand}")
    if category:
        parts.append(f"Category: {category}")
    if features:
        parts.append(f"Features: {features}")
    if description:
        parts.append(f"Description: {description}")
    text = "\n".join(parts)
    return text if text else FALLBACK_TEXT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed/beauty")
    parser.add_argument("--max-text-tokens", type=int, default=256)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    item2id = json.loads((processed_dir / "item2id.json").read_text(encoding="utf-8"))
    id2raw = {idx: raw for raw, idx in item2id.items()}
    meta_path = download_metadata(Path(args.raw_dir))
    metadata = load_metadata(meta_path, set(item2id.keys()))

    rows = []
    missing = []
    title_count = brand_count = category_count = description_count = 0
    no_text_count = 0
    text_lengths = []
    overlong_count = 0

    for item_id in range(1, len(item2id) + 1):
        raw_item_id = id2raw[item_id]
        row = metadata.get(raw_item_id)
        if row is None:
            missing.append(raw_item_id)
            text = FALLBACK_TEXT
            has_text = 0
        else:
            title = clean_text(row.get("title"))
            brand = clean_text(row.get("brand"))
            category = category_path(row.get("categories"))
            description = clean_text(row.get("description"))
            title_count += bool(title)
            brand_count += bool(brand)
            category_count += bool(category)
            description_count += bool(description)
            text = build_text(row)
            has_text = int(text != FALLBACK_TEXT)
            no_text_count += int(not has_text)

        tokens = text.split()
        text_lengths.append(len(tokens))
        if len(tokens) > args.max_text_tokens:
            overlong_count += 1
            text = " ".join(tokens[: args.max_text_tokens])

        rows.append({"item_id": item_id, "raw_item_id": raw_item_id, "text": text, "has_text": has_text})

    audit = {
        "num_items": len(item2id),
        "metadata_file": str(meta_path),
        "title_ratio": title_count / len(item2id),
        "category_ratio": category_count / len(item2id),
        "brand_ratio": brand_count / len(item2id),
        "description_ratio": description_count / len(item2id),
        "no_text_count": no_text_count + len(missing),
        "avg_text_length": sum(text_lengths) / len(text_lengths),
        "overlong_text_ratio": overlong_count / len(item2id),
        "max_text_tokens": args.max_text_tokens,
        "missing_metadata_count": len(missing),
        "missing_metadata_raw_item_ids": missing,
    }

    with (processed_dir / "item_texts.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (processed_dir / "item_metadata_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

