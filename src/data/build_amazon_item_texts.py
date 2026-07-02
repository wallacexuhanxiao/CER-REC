import argparse
import ast
import gzip
import json
import re
from pathlib import Path


ASIN_RE = re.compile(r"\bB[0-9A-Z]{9}\b")


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="ignore") if path.suffix == ".gz" else path.open("rt", encoding="utf-8", errors="ignore")


def clean_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"none", "nan", "null", "[]", "{}"}:
            return ""
        return " ".join(text.split())
    if isinstance(value, dict):
        return " ".join(clean_text(v) for v in value.values() if clean_text(v)).strip()
    if isinstance(value, (list, tuple)):
        return " ".join(clean_text(v) for v in value if clean_text(v)).strip()
    return clean_text(str(value))


def category_path(row):
    values = []
    for key in ("categories", "category", "main_category"):
        value = row.get(key)
        if not value:
            continue
        if isinstance(value, str):
            values.append(clean_text(value))
        else:
            for path in value:
                if isinstance(path, (list, tuple)):
                    values.append(" > ".join(clean_text(x) for x in path if clean_text(x)))
                else:
                    values.append(clean_text(path))
    return " | ".join(v for v in values if v)


def iter_metadata(path: Path, metadata_format: str):
    with open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line) if metadata_format == "jsonl" else ast.literal_eval(line)


def load_metadata(path: Path, wanted_raw_ids, metadata_format: str, item_id_field: str):
    meta = {}
    for row in iter_metadata(path, metadata_format):
        raw_id = row.get(item_id_field)
        if raw_id in wanted_raw_ids:
            meta[raw_id] = row
    return meta


def build_text(row, fallback_text: str):
    title = clean_text(row.get("title"))
    brand = clean_text(row.get("brand") or row.get("store"))
    category = category_path(row)
    features = clean_text(row.get("feature") or row.get("features"))
    description = clean_text(row.get("description"))
    details = clean_text(row.get("details"))
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
    if details:
        parts.append(f"Details: {details}")
    text = ASIN_RE.sub("", "\n".join(parts))
    text = " ".join(text.split())
    return text if text else fallback_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--metadata-format", choices=["jsonl", "python"], required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--item-id-field", default="parent_asin")
    parser.add_argument("--fallback-text", default="Unknown product")
    parser.add_argument("--max-text-tokens", type=int, default=256)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    item2id = json.loads((processed_dir / "item2id.json").read_text(encoding="utf-8"))
    id2raw = {idx: raw for raw, idx in item2id.items()}
    metadata = load_metadata(Path(args.metadata_path), set(item2id), args.metadata_format, args.item_id_field)

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
            text = args.fallback_text
            has_text = 0
        else:
            title = clean_text(row.get("title"))
            brand = clean_text(row.get("brand") or row.get("store"))
            category = category_path(row)
            description = clean_text(row.get("description"))
            title_count += bool(title)
            brand_count += bool(brand)
            category_count += bool(category)
            description_count += bool(description)
            text = build_text(row, args.fallback_text)
            has_text = int(text != args.fallback_text)
            no_text_count += int(not has_text)

        tokens = text.split()
        text_lengths.append(len(tokens))
        if len(tokens) > args.max_text_tokens:
            overlong_count += 1
            text = " ".join(tokens[: args.max_text_tokens])

        rows.append({"item_id": item_id, "raw_item_id": raw_item_id, "text": text, "has_text": has_text})

    audit = {
        "num_items": len(item2id),
        "metadata_file": str(args.metadata_path),
        "metadata_format": args.metadata_format,
        "item_id_field": args.item_id_field,
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
