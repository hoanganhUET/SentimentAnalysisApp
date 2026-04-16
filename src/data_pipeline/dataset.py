from __future__ import annotations

import importlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _try_import(module_name: str):
	try:
		return importlib.import_module(module_name)
	except Exception:
		return None


_torch_utils_data = _try_import("torch.utils.data")
_DatasetBase = _torch_utils_data.Dataset if _torch_utils_data is not None else object


@dataclass
class DatasetConfig:
	data_path: str | Path = "data/processed/reviews_clean.jsonl"
	tokenizer_name: str = "distilbert-base-uncased"
	text_column: str = "text"
	label_column: str = "sentiment"
	train_ratio: float = 0.8
	val_ratio: float = 0.1
	test_ratio: float = 0.1
	seed: int = 42
	max_length: int = 256
	batch_size: int = 16
	num_workers: int = 0
	stratify_by_label: bool = False


def _validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
	total = train_ratio + val_ratio + test_ratio
	if abs(total - 1.0) > 1e-8:
		raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def _normalize_text(value: Any) -> str:
	if value is None:
		return ""
	return " ".join(str(value).strip().split())


def read_reviews_jsonl(
	path: str | Path,
	text_column: str = "text",
	label_column: str = "sentiment",
) -> list[dict[str, Any]]:
	file_path = Path(path)
	if not file_path.exists():
		raise FileNotFoundError(f"Missing dataset file: {file_path}")

	records: list[dict[str, Any]] = []
	with open(file_path, "r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue

			row = json.loads(line)
			text_value = _normalize_text(row.get(text_column))
			label_value = row.get(label_column)

			if not text_value or label_value is None:
				continue

			records.append(
				{
					"text": text_value,
					"sentiment": str(label_value),
					"rating": row.get("rating"),
					"asin": row.get("asin"),
					"reviewer_id": row.get("reviewer_id"),
					"timestamp": row.get("timestamp"),
				}
			)

	if not records:
		raise ValueError(f"No valid records found in: {file_path}")
	return records


def build_label_mapping(records: list[dict[str, Any]]) -> dict[str, int]:
	labels = sorted({row["sentiment"] for row in records})
	return {label: idx for idx, label in enumerate(labels)}


def _split_one_group(
	group_records: list[dict[str, Any]],
	train_ratio: float,
	val_ratio: float,
	seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
	rng = random.Random(seed)
	shuffled = group_records[:]
	rng.shuffle(shuffled)

	n_total = len(shuffled)
	n_train = int(n_total * train_ratio)
	n_val = int(n_total * val_ratio)
	n_test = n_total - n_train - n_val

	train_split = shuffled[:n_train]
	val_split = shuffled[n_train : n_train + n_val]
	test_split = shuffled[n_train + n_val : n_train + n_val + n_test]
	return train_split, val_split, test_split


def split_records(
	records: list[dict[str, Any]],
	train_ratio: float = 0.8,
	val_ratio: float = 0.1,
	test_ratio: float = 0.1,
	seed: int = 42,
	stratify_by_label: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
	_validate_split_ratios(train_ratio, val_ratio, test_ratio)

	if not stratify_by_label:
		train_split, val_split, test_split = _split_one_group(records, train_ratio, val_ratio, seed)
		return train_split, val_split, test_split

	grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in records:
		grouped[row["sentiment"]].append(row)

	train_split: list[dict[str, Any]] = []
	val_split: list[dict[str, Any]] = []
	test_split: list[dict[str, Any]] = []

	for index, label in enumerate(sorted(grouped)):
		g_train, g_val, g_test = _split_one_group(
			grouped[label],
			train_ratio=train_ratio,
			val_ratio=val_ratio,
			seed=seed + index,
		)
		train_split.extend(g_train)
		val_split.extend(g_val)
		test_split.extend(g_test)

	rng = random.Random(seed)
	rng.shuffle(train_split)
	rng.shuffle(val_split)
	rng.shuffle(test_split)
	return train_split, val_split, test_split


class SentimentTorchDataset(_DatasetBase):
	"""PyTorch-style dataset for sentiment classification."""

	def __init__(
		self,
		records: list[dict[str, Any]],
		tokenizer,
		label2id: dict[str, int],
		max_length: int = 256,
	):
		self.records = records
		self.tokenizer = tokenizer
		self.label2id = label2id
		self.max_length = max_length

	def __len__(self) -> int:
		return len(self.records)

	def __getitem__(self, index: int) -> dict[str, Any]:
		row = self.records[index]
		text = row["text"]
		label = row["sentiment"]

		encoded = self.tokenizer(
			text,
			truncation=True,
			max_length=self.max_length,
			return_attention_mask=True,
		)
		encoded["labels"] = self.label2id[label]
		return encoded


class TorchDynamicPadCollator:
	"""Create padded mini-batches for tokenized rows."""

	def __init__(self, tokenizer):
		self.tokenizer = tokenizer
		self.torch = _try_import("torch")
		if self.torch is None:
			raise ImportError("torch is required for TorchDynamicPadCollator")

	def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
		labels = [item["labels"] for item in batch]
		features = [{k: v for k, v in item.items() if k != "labels"} for item in batch]
		padded = self.tokenizer.pad(features, padding=True, return_tensors="pt")
		padded["labels"] = self.torch.tensor(labels, dtype=self.torch.long)
		return padded


def create_torch_dataloaders(
	config: DatasetConfig,
) -> tuple[dict[str, Any], Any, dict[str, int], dict[int, str]]:
	torch_module = _try_import("torch")
	transformers_module = _try_import("transformers")
	if torch_module is None:
		raise ImportError("torch is required to create PyTorch DataLoaders")
	if transformers_module is None:
		raise ImportError("transformers is required to tokenize text")

	records = read_reviews_jsonl(config.data_path, config.text_column, config.label_column)
	train_records, val_records, test_records = split_records(
		records,
		train_ratio=config.train_ratio,
		val_ratio=config.val_ratio,
		test_ratio=config.test_ratio,
		seed=config.seed,
		stratify_by_label=config.stratify_by_label,
	)

	label2id = build_label_mapping(records)
	id2label = {idx: label for label, idx in label2id.items()}

	tokenizer = transformers_module.AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)
	collator = TorchDynamicPadCollator(tokenizer)

	datasets = {
		"train": SentimentTorchDataset(train_records, tokenizer, label2id, max_length=config.max_length),
		"val": SentimentTorchDataset(val_records, tokenizer, label2id, max_length=config.max_length),
		"test": SentimentTorchDataset(test_records, tokenizer, label2id, max_length=config.max_length),
	}

	dataloaders = {
		"train": _torch_utils_data.DataLoader(
			datasets["train"],
			batch_size=config.batch_size,
			shuffle=True,
			collate_fn=collator,
			num_workers=config.num_workers,
		),
		"val": _torch_utils_data.DataLoader(
			datasets["val"],
			batch_size=config.batch_size,
			shuffle=False,
			collate_fn=collator,
			num_workers=config.num_workers,
		),
		"test": _torch_utils_data.DataLoader(
			datasets["test"],
			batch_size=config.batch_size,
			shuffle=False,
			collate_fn=collator,
			num_workers=config.num_workers,
		),
	}

	return dataloaders, tokenizer, label2id, id2label


def create_hf_dataset_wrapper(
	config: DatasetConfig,
) -> tuple[Any, Any, Any, dict[str, int], dict[int, str]]:
	datasets_module = _try_import("datasets")
	transformers_module = _try_import("transformers")
	if datasets_module is None:
		raise ImportError("datasets is required to create HuggingFace Dataset wrapper")
	if transformers_module is None:
		raise ImportError("transformers is required to tokenize text")

	records = read_reviews_jsonl(config.data_path, config.text_column, config.label_column)
	train_records, val_records, test_records = split_records(
		records,
		train_ratio=config.train_ratio,
		val_ratio=config.val_ratio,
		test_ratio=config.test_ratio,
		seed=config.seed,
		stratify_by_label=config.stratify_by_label,
	)

	label2id = build_label_mapping(records)
	id2label = {idx: label for label, idx in label2id.items()}
	tokenizer = transformers_module.AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)

	def to_hf_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
		return [{"text": row["text"], "labels": label2id[row["sentiment"]]} for row in rows]

	hf_dict = datasets_module.DatasetDict(
		{
			"train": datasets_module.Dataset.from_list(to_hf_rows(train_records)),
			"validation": datasets_module.Dataset.from_list(to_hf_rows(val_records)),
			"test": datasets_module.Dataset.from_list(to_hf_rows(test_records)),
		}
	)

	def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
		return tokenizer(
			batch["text"],
			truncation=True,
			max_length=config.max_length,
		)

	tokenized = hf_dict.map(tokenize_batch, batched=True)
	tokenized = tokenized.remove_columns(["text"])
	tokenized.set_format(type="torch")

	data_collator = transformers_module.DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
	return tokenized, tokenizer, data_collator, label2id, id2label


__all__ = [
	"DatasetConfig",
	"SentimentTorchDataset",
	"TorchDynamicPadCollator",
	"build_label_mapping",
	"create_hf_dataset_wrapper",
	"create_torch_dataloaders",
	"read_reviews_jsonl",
	"split_records",
]

