"""Load and validate real MedNLI and PubMedQA dataset files.

The loader intentionally does not fabricate examples. Experiments must point to
real JSONL/CSV files on disk. Tests create tiny temporary fixtures that follow
these schemas, but production runs should use the actual MedNLI and PubMedQA
exports that the user is licensed to access.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DatasetRecord:
    """One normalized supervised text-classification example."""

    text: str
    label: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DatasetBundle:
    """Train/validation/test splits loaded from real input files."""

    name: str
    train: list[DatasetRecord]
    validation: list[DatasetRecord]
    test: list[DatasetRecord]
    labels: tuple[str, ...]

    @property
    def split_sizes(self) -> dict[str, int]:
        """Return the number of examples in each split."""

        return {"train": len(self.train), "validation": len(self.validation), "test": len(self.test)}


def load_dataset_bundle(config: dict[str, Any]) -> DatasetBundle:
    """Load a MedNLI or PubMedQA dataset from configured local files."""

    name = str(config.get("name", "")).lower()
    paths = config.get("paths", {})
    if name not in {"mednli", "pubmedqa"}:
        raise ValueError("dataset.name must be either 'mednli' or 'pubmedqa'.")
    train = _load_split(name, paths.get("train"), "train")
    validation = _load_split(name, paths.get("validation"), "validation")
    test = _load_split(name, paths.get("test"), "test")
    labels = tuple(sorted({record.label for record in [*train, *validation, *test]}))
    if len(labels) < 2:
        raise ValueError("Dataset must contain at least two labels across splits.")
    return DatasetBundle(name=name, train=train, validation=validation, test=test, labels=labels)


def _load_split(name: str, path_value: object, split: str) -> list[DatasetRecord]:
    if not path_value:
        raise ValueError(f"Missing path for dataset split: {split}.")
    path = Path(str(path_value))
    if not path.exists():
        raise FileNotFoundError(
            f"Configured {name} {split} split does not exist: {path}. "
            "Place the real dataset file there or update configs/experiment.yaml."
        )
    if path.suffix.lower() == ".jsonl":
        rows = _read_jsonl(path)
    elif path.suffix.lower() == ".csv":
        rows = _read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset file extension for {path}; use .jsonl or .csv.")
    records = [_normalize_record(name, row, split, index) for index, row in enumerate(rows)]
    if not records:
        raise ValueError(f"Dataset split {split} at {path} is empty.")
    return records


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped:
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc


def _read_csv(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _normalize_record(name: str, row: dict[str, Any], split: str, index: int) -> DatasetRecord:
    if name == "mednli":
        premise = _first_present(row, ["sentence1", "premise"])
        hypothesis = _first_present(row, ["sentence2", "hypothesis"])
        label = _first_present(row, ["gold_label", "label"])
        text = f"premise: {premise}\nhypothesis: {hypothesis}"
    else:
        question = _first_present(row, ["question"])
        context = _pubmedqa_context(row)
        label = _first_present(row, ["final_decision", "label", "answer"])
        text = f"question: {question}\ncontext: {context}"
    if not str(label).strip():
        raise ValueError(f"Missing label in {name} {split} record {index}.")
    return DatasetRecord(text=text, label=str(label).strip(), metadata={"split": split, "index": index})


def _pubmedqa_context(row: dict[str, Any]) -> str:
    context = row.get("context") or row.get("contexts") or row.get("abstract")
    if isinstance(context, dict):
        parts = context.get("contexts") or context.get("text") or context.get("labels") or []
        if isinstance(parts, list):
            return " ".join(str(part) for part in parts)
    if isinstance(context, list):
        return " ".join(str(part) for part in context)
    return str(context or "")


def _first_present(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"Missing required field; expected one of {keys}.")
