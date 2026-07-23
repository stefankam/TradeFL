import json

import pytest

from tradefl.data import load_dataset_bundle


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_mednli_loader_requires_real_files(tmp_path):
    cfg = {"name": "mednli", "paths": {"train": tmp_path / "missing.jsonl", "validation": tmp_path / "v.jsonl", "test": tmp_path / "t.jsonl"}}
    with pytest.raises(FileNotFoundError):
        load_dataset_bundle(cfg)


def test_mednli_loader_normalizes_records(tmp_path):
    rows = [
        {"sentence1": "A patient has fever.", "sentence2": "The patient is ill.", "gold_label": "entailment"},
        {"sentence1": "A patient has fever.", "sentence2": "The patient is healthy.", "gold_label": "contradiction"},
    ]
    for split in ["train", "validation", "test"]:
        write_jsonl(tmp_path / f"{split}.jsonl", rows)
    bundle = load_dataset_bundle({"name": "mednli", "paths": {"train": tmp_path / "train.jsonl", "validation": tmp_path / "validation.jsonl", "test": tmp_path / "test.jsonl"}})
    assert bundle.name == "mednli"
    assert bundle.labels == ("contradiction", "entailment")
    assert "premise:" in bundle.train[0].text


def test_pubmedqa_loader_normalizes_records(tmp_path):
    rows = [
        {"question": "Does treatment help?", "context": {"contexts": ["Trial improved outcomes."]}, "final_decision": "yes"},
        {"question": "Is it harmful?", "context": ["No signal was observed."], "final_decision": "no"},
    ]
    for split in ["train", "validation", "test"]:
        write_jsonl(tmp_path / f"{split}.jsonl", rows)
    bundle = load_dataset_bundle({"name": "pubmedqa", "paths": {"train": tmp_path / "train.jsonl", "validation": tmp_path / "validation.jsonl", "test": tmp_path / "test.jsonl"}})
    assert bundle.name == "pubmedqa"
    assert set(bundle.labels) == {"no", "yes"}
    assert "question:" in bundle.train[0].text
