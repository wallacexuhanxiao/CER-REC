import argparse
import json
import re
from pathlib import Path


ASIN_RE = re.compile(r"\bB[0-9A-Z]{9}\b")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/beauty")
    parser.add_argument("--output-dir", default="outputs/beauty/audits")
    args = parser.parse_args()
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (processed_dir / "item_texts.jsonl").read_text(encoding="utf-8").splitlines()]
    suspicious = {
        "raw_item_id_in_text_count": 0,
        "asin_pattern_count": 0,
        "review_marker_count": 0,
        "rating_marker_count": 0,
        "interaction_marker_count": 0,
        "username_marker_count": 0,
    }
    examples = []
    for row in rows:
        text = row["text"]
        lower = text.lower()
        flags = []
        if row["raw_item_id"] in text:
            suspicious["raw_item_id_in_text_count"] += 1
            flags.append("raw_item_id")
        if ASIN_RE.search(text):
            suspicious["asin_pattern_count"] += 1
            flags.append("asin_pattern")
        for key, patterns in {
            "review_marker_count": [r"\breviewtext\b", r"\breviewer\b", r"\bsummary:"],
            "rating_marker_count": [r"\boverall:", r"\brating:", r"\bstars:"],
            "interaction_marker_count": [r"\binteraction\b", r"\binteraction count\b", r"\bsalesrank\b"],
            "username_marker_count": [r"\busername\b", r"\breviewerid\b", r"\buserid\b"],
        }.items():
            if any(re.search(p, lower) for p in patterns):
                suspicious[key] += 1
                flags.append(key)
        if flags and len(examples) < 20:
            examples.append({"item_id": row["item_id"], "raw_item_id": row["raw_item_id"], "flags": flags, "text_preview": text[:240]})
    result = {
        **suspicious,
        "num_items": len(rows),
        "allowed_fields": ["Title", "Brand", "Category", "Features", "Description"],
        "examples": examples,
        "verdict": "pass" if suspicious["raw_item_id_in_text_count"] == 0 and suspicious["asin_pattern_count"] == 0 and suspicious["review_marker_count"] == 0 else "review",
    }
    (output_dir / "item_text_leakage_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
