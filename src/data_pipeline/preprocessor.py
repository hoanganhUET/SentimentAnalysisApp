from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import html
import importlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


REVIEW_KEEP_COLS = {
    "reviewerID",
    "asin",
    "overall",
    "reviewText",
    "summary",
    "unixReviewTime",
}
META_KEEP_COLS = {"asin", "title", "categories"}

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NON_WORD_RE = re.compile(r"[^a-z0-9_\s]")
_MULTISPACE_RE = re.compile(r"\s+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U0001F300-\U0001FAFF"  # Symbols and pictographs
    "\U00002600-\U000026FF"  # Misc symbols
    "\U00002700-\U000027BF"  # Dingbats
    "]+",
    flags=re.UNICODE,
)


def _load_stopwords() -> set[str]:
    try:
        nltk_module = importlib.import_module("nltk")
        stopwords_module = importlib.import_module("nltk.corpus").stopwords

        try:
            return set(stopwords_module.words("english"))
        except LookupError:
            nltk_module.download("stopwords", quiet=True)
            return set(stopwords_module.words("english"))
    except Exception:
        # Fallback set keeps preprocessing running when NLTK is unavailable.
        return {
            "a",
            "an",
            "the",
            "and",
            "or",
            "is",
            "are",
            "was",
            "were",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "this",
            "that",
            "it",
        }


STOP_WORDS = _load_stopwords()


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, "r", encoding="utf-8")


def _parse_line(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None

    try:
        data = json.loads(line)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    try:
        data = ast.literal_eval(line)
        return data if isinstance(data, dict) else None
    except (ValueError, SyntaxError):
        return None


def load_data_in_chunks(file_path, chunk_size=10000, data_type="review"):
    """
    Đọc dữ liệu theo từng chunk và lọc các trường không cần thiết để tránh OOM.
    data_type có thể là "review" hoặc "metadata".
    """
    path = Path(file_path)
    keep_cols = REVIEW_KEEP_COLS if data_type == "review" else META_KEEP_COLS
    chunk = []

    with _open_text(path) as handle:
        for line in handle:
            raw_item = _parse_line(line)
            if not raw_item:
                continue

            filtered_item = {k: v for k, v in raw_item.items() if k in keep_cols}
            chunk.append(filtered_item)

            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

    if chunk:
        yield chunk


def _clean_text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _replace_emojis_with_text(text: str) -> str:
    def _repl(match: re.Match) -> str:
        chars = match.group(0)
        labels = []
        for char in chars:
            label = unicodedata.name(char, "")
            if label:
                labels.append(label.lower().replace(" ", "_"))
        return f" {' '.join(labels)} " if labels else " "

    return _EMOJI_RE.sub(_repl, text)


def _normalize_review_text(value) -> str:
    if value is None:
        return ""

    text = str(value)
    if not text.strip():
        return ""

    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _replace_emojis_with_text(text)
    text = text.lower()
    text = _NON_WORD_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()

    if not text:
        return ""

    tokens = [token for token in text.split(" ") if token and token not in STOP_WORDS]
    return " ".join(tokens)


def _rating_to_sentiment(rating: float) -> str:
    if rating >= 4.0:
        return "Positive"
    if rating <= 2.0:
        return "Negative"
    return "Neutral"


def _flatten_categories(raw_categories) -> list[str]:
    if not isinstance(raw_categories, list) or not raw_categories:
        return []

    if isinstance(raw_categories[0], list):
        selected = raw_categories[0]
    else:
        selected = raw_categories

    flattened = []
    for item in selected:
        text = _clean_text(item)
        if text:
            flattened.append(text)
    return flattened


def process_reviews(reviews_file_path: Path, processed_dir: Path, chunk_size: int = 10000) -> dict:
    reviews_out = processed_dir / "reviews_clean.jsonl"
    interactions_out = processed_dir / "interactions.jsonl"

    stats = {
        "review_rows": 0,
        "interaction_rows": 0,
        "invalid_rows": 0,
        "empty_text_rows": 0,
    }

    with open(reviews_out, "w", encoding="utf-8") as reviews_f, open(
        interactions_out, "w", encoding="utf-8"
    ) as interactions_f:
        for chunk in load_data_in_chunks(reviews_file_path, chunk_size=chunk_size, data_type="review"):
            for row in chunk:
                reviewer_id = _clean_text(row.get("reviewerID"))
                asin = _clean_text(row.get("asin"))
                timestamp = row.get("unixReviewTime")

                try:
                    rating = float(row.get("overall"))
                except (TypeError, ValueError):
                    stats["invalid_rows"] += 1
                    continue

                if not reviewer_id or not asin:
                    stats["invalid_rows"] += 1
                    continue

                summary = _normalize_review_text(row.get("summary"))
                review_text = _normalize_review_text(row.get("reviewText"))
                merged_text = " ".join(part for part in [summary, review_text] if part)

                interaction_record = {
                    "reviewer_id": reviewer_id,
                    "asin": asin,
                    "rating": rating,
                    "timestamp": timestamp,
                }
                interactions_f.write(json.dumps(interaction_record, ensure_ascii=False) + "\n")
                stats["interaction_rows"] += 1

                if merged_text:
                    review_record = {
                        "reviewer_id": reviewer_id,
                        "asin": asin,
                        "rating": rating,
                        "sentiment": _rating_to_sentiment(rating),
                        "text": merged_text,
                        "timestamp": timestamp,
                    }
                    reviews_f.write(json.dumps(review_record, ensure_ascii=False) + "\n")
                    stats["review_rows"] += 1
                else:
                    stats["empty_text_rows"] += 1

    return stats


def process_metadata(metadata_file_path: Path, processed_dir: Path, chunk_size: int = 10000) -> dict:
    metadata_out = processed_dir / "products_meta.jsonl"
    stats = {
        "metadata_rows": 0,
        "invalid_rows": 0,
    }

    with open(metadata_out, "w", encoding="utf-8") as meta_f:
        for chunk in load_data_in_chunks(metadata_file_path, chunk_size=chunk_size, data_type="metadata"):
            for row in chunk:
                asin = _clean_text(row.get("asin"))
                if not asin:
                    stats["invalid_rows"] += 1
                    continue

                category_path = _flatten_categories(row.get("categories"))
                record = {
                    "asin": asin,
                    "title": _clean_text(row.get("title")),
                    "category_path": category_path,
                    "category_leaf": category_path[-1] if category_path else "",
                }

                meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["metadata_rows"] += 1

    return stats


def _build_asin_category_map(products_meta_path: Path) -> dict[str, str]:
    asin_to_category = {}
    with open(products_meta_path, "r", encoding="utf-8") as meta_f:
        for line in meta_f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            asin = row.get("asin")
            if not asin:
                continue
            category = row.get("category_leaf") or "Unknown"
            asin_to_category[asin] = category
    return asin_to_category


def analyze_sentiment_distribution_by_category(processed_dir: Path) -> dict:
    reviews_path = processed_dir / "reviews_clean.jsonl"
    meta_path = processed_dir / "products_meta.jsonl"
    output_csv = processed_dir / "category_sentiment_distribution.csv"

    if not reviews_path.exists():
        raise FileNotFoundError(f"Missing file: {reviews_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing file: {meta_path}")

    asin_to_category = _build_asin_category_map(meta_path)

    category_sentiment_counts = defaultdict(Counter)
    category_totals = Counter()

    with open(reviews_path, "r", encoding="utf-8") as reviews_f:
        for line in reviews_f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            asin = row.get("asin")
            sentiment = row.get("sentiment")

            if sentiment not in {"Negative", "Neutral", "Positive"}:
                try:
                    sentiment = _rating_to_sentiment(float(row.get("rating")))
                except (TypeError, ValueError):
                    continue

            category = asin_to_category.get(asin, "Unknown")
            category_sentiment_counts[category][sentiment] += 1
            category_totals[category] += 1

    rows = []
    for category, total in category_totals.items():
        negative_count = category_sentiment_counts[category]["Negative"]
        neutral_count = category_sentiment_counts[category]["Neutral"]
        positive_count = category_sentiment_counts[category]["Positive"]

        rows.append(
            {
                "category": category,
                "total_reviews": total,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
                "positive_count": positive_count,
                "negative_ratio_pct": round((negative_count / total) * 100, 4) if total else 0.0,
                "neutral_ratio_pct": round((neutral_count / total) * 100, 4) if total else 0.0,
                "positive_ratio_pct": round((positive_count / total) * 100, 4) if total else 0.0,
            }
        )

    rows.sort(key=lambda x: x["total_reviews"], reverse=True)

    fieldnames = [
        "category",
        "total_reviews",
        "negative_count",
        "neutral_count",
        "positive_count",
        "negative_ratio_pct",
        "neutral_ratio_pct",
        "positive_ratio_pct",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "category_count": len(rows),
        "distribution_file": str(output_csv),
    }


def _compute_class_weight(label_counts: Counter) -> dict[str, float]:
    total = sum(label_counts.values())
    n_classes = len(label_counts)
    if total == 0 or n_classes == 0:
        return {}

    class_weight = {}
    for label, count in label_counts.items():
        if count <= 0:
            continue
        class_weight[label] = round(total / (n_classes * count), 6)
    return class_weight


def analyze_class_imbalance_and_strategy(processed_dir: Path, severe_ratio_threshold: float = 2.5) -> dict:
    reviews_path = processed_dir / "reviews_clean.jsonl"
    output_json = processed_dir / "class_imbalance_report.json"

    if not reviews_path.exists():
        raise FileNotFoundError(f"Missing file: {reviews_path}")

    label_counts: Counter = Counter()
    with open(reviews_path, "r", encoding="utf-8") as reviews_f:
        for line in reviews_f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            sentiment = row.get("sentiment")
            if sentiment not in {"Negative", "Neutral", "Positive"}:
                try:
                    sentiment = _rating_to_sentiment(float(row.get("rating")))
                except (TypeError, ValueError):
                    continue
            label_counts[sentiment] += 1

    if not label_counts:
        raise ValueError("No valid sentiment labels found for class imbalance analysis")

    total_samples = sum(label_counts.values())
    majority_label, majority_count = max(label_counts.items(), key=lambda x: x[1])
    minority_label, minority_count = min(label_counts.items(), key=lambda x: x[1])
    imbalance_ratio = round(majority_count / minority_count, 6) if minority_count else float("inf")

    class_weight = _compute_class_weight(label_counts)

    use_smote = imbalance_ratio >= severe_ratio_threshold and minority_count >= 6
    recommended_strategy = "SMOTE" if use_smote else "class_weight"
    smote_k_neighbors = min(5, max(1, minority_count - 1))

    report = {
        "total_samples": total_samples,
        "class_distribution": dict(label_counts),
        "majority_class": majority_label,
        "minority_class": minority_label,
        "imbalance_ratio": imbalance_ratio,
        "is_imbalanced": imbalance_ratio > 1.5,
        "recommended_strategy": recommended_strategy,
        "class_weight": class_weight,
        "logistic_regression": {
            "class_weight": class_weight,
            "max_iter": 2000,
        },
        "svm": {
            "class_weight": class_weight,
            "kernel": "linear",
        },
        "smote": {
            "enabled": use_smote,
            "sampling_strategy": "not majority",
            "random_state": 42,
            "k_neighbors": smote_k_neighbors,
        },
    }

    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    return {
        "imbalance_ratio": imbalance_ratio,
        "recommended_strategy": recommended_strategy,
        "imbalance_report_file": str(output_json),
    }


def apply_smote_if_available(X, y, random_state: int = 42, k_neighbors: int = 5):
    """Apply SMOTE for downstream training if imbalanced-learn is installed."""
    try:
        smote_cls = importlib.import_module("imblearn.over_sampling").SMOTE
    except Exception as exc:
        raise ImportError(
            "SMOTE requires imbalanced-learn. Install with: pip install imbalanced-learn"
        ) from exc

    smote = smote_cls(
        sampling_strategy="not majority",
        random_state=random_state,
        k_neighbors=k_neighbors,
    )
    return smote.fit_resample(X, y)


def _select_split_bucket(row: dict, seed: int, modulo: int = 10000) -> int:
    key = (
        f"{seed}|{row.get('reviewer_id', '')}|{row.get('asin', '')}|"
        f"{row.get('timestamp', '')}|{row.get('rating', '')}|{row.get('sentiment', '')}"
    )
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def export_train_val_test_csv(
    processed_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(f"train_ratio + val_ratio + test_ratio must be 1.0, got {total_ratio}")

    reviews_path = processed_dir / "reviews_clean.jsonl"
    if not reviews_path.exists():
        raise FileNotFoundError(f"Missing file: {reviews_path}")

    train_out = processed_dir / "train.csv"
    val_out = processed_dir / "val.csv"
    test_out = processed_dir / "test.csv"

    fieldnames = ["reviewer_id", "asin", "rating", "sentiment", "text", "timestamp"]

    train_cut = int(train_ratio * 10000)
    val_cut = train_cut + int(val_ratio * 10000)
    split_counts = Counter()

    with open(train_out, "w", encoding="utf-8", newline="") as train_f, open(
        val_out, "w", encoding="utf-8", newline=""
    ) as val_f, open(test_out, "w", encoding="utf-8", newline="") as test_f:
        train_writer = csv.DictWriter(train_f, fieldnames=fieldnames)
        val_writer = csv.DictWriter(val_f, fieldnames=fieldnames)
        test_writer = csv.DictWriter(test_f, fieldnames=fieldnames)

        train_writer.writeheader()
        val_writer.writeheader()
        test_writer.writeheader()

        with open(reviews_path, "r", encoding="utf-8") as reviews_f:
            for line in reviews_f:
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                cleaned_row = {k: row.get(k, "") for k in fieldnames}
                cleaned_row["text"] = _clean_text(cleaned_row.get("text"))

                if not cleaned_row["text"] or not cleaned_row.get("sentiment"):
                    continue

                bucket = _select_split_bucket(cleaned_row, seed=seed)
                if bucket < train_cut:
                    train_writer.writerow(cleaned_row)
                    split_counts["train_rows"] += 1
                elif bucket < val_cut:
                    val_writer.writerow(cleaned_row)
                    split_counts["val_rows"] += 1
                else:
                    test_writer.writerow(cleaned_row)
                    split_counts["test_rows"] += 1

    total_rows = split_counts["train_rows"] + split_counts["val_rows"] + split_counts["test_rows"]
    return {
        "train_file": str(train_out),
        "val_file": str(val_out),
        "test_file": str(test_out),
        "train_rows": split_counts["train_rows"],
        "val_rows": split_counts["val_rows"],
        "test_rows": split_counts["test_rows"],
        "split_total_rows": total_rows,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Process raw Amazon product data into processed JSONL files.")
    parser.add_argument(
        "--raw-dir",
        default=str(project_root / "data" / "raw"),
        help="Raw data directory containing reviews_5_core.json.gz and meta_data.json.gz.",
    )
    parser.add_argument(
        "--processed-dir",
        default=str(project_root / "data" / "processed"),
        help="Output directory for processed files.",
    )
    parser.add_argument("--chunk-size", type=int, default=10000, help="Number of rows processed per chunk.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    processed_dir = Path(args.processed_dir).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    reviews_file_path = raw_dir / "reviews_5_core.json.gz"
    metadata_file_path = raw_dir / "meta_data.json.gz"

    if not reviews_file_path.exists():
        raise FileNotFoundError(f"Missing file: {reviews_file_path}")
    if not metadata_file_path.exists():
        raise FileNotFoundError(f"Missing file: {metadata_file_path}")

    review_stats = process_reviews(reviews_file_path, processed_dir, chunk_size=args.chunk_size)
    metadata_stats = process_metadata(metadata_file_path, processed_dir, chunk_size=args.chunk_size)
    sentiment_distribution_stats = analyze_sentiment_distribution_by_category(processed_dir)
    imbalance_stats = analyze_class_imbalance_and_strategy(processed_dir)
    split_stats = export_train_val_test_csv(processed_dir)

    print("Processed files:")
    print(f"- {processed_dir / 'reviews_clean.jsonl'}")
    print(f"- {processed_dir / 'interactions.jsonl'}")
    print(f"- {processed_dir / 'products_meta.jsonl'}")
    print(f"- {processed_dir / 'category_sentiment_distribution.csv'}")
    print(f"- {processed_dir / 'class_imbalance_report.json'}")
    print(f"- {processed_dir / 'train.csv'}")
    print(f"- {processed_dir / 'val.csv'}")
    print(f"- {processed_dir / 'test.csv'}")
    print("\nStats:")
    print(
        json.dumps(
            {
                **review_stats,
                **metadata_stats,
                **sentiment_distribution_stats,
                **imbalance_stats,
                **split_stats,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

