import json

from tradefl.data import load_dataset_bundle
from tradefl.federation.simulator import run_to_target
from tradefl.plans.lora import build


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_dataset_backed_plan_runs_to_target(tmp_path):
    rows = [
        {"sentence1": "Patient has pneumonia", "sentence2": "Patient is sick", "gold_label": "entailment"},
        {"sentence1": "Patient has pneumonia", "sentence2": "Patient is healthy", "gold_label": "contradiction"},
        {"sentence1": "No fracture", "sentence2": "Bone is broken", "gold_label": "contradiction"},
        {"sentence1": "Lab is positive", "sentence2": "Test result is abnormal", "gold_label": "entailment"},
    ]
    for split in ["train", "validation", "test"]:
        write_jsonl(tmp_path / f"{split}.jsonl", rows)
    dataset = load_dataset_bundle({"name": "mednli", "paths": {"train": tmp_path / "train.jsonl", "validation": tmp_path / "validation.jsonl", "test": tmp_path / "test.jsonl"}})
    plan = build(plan_id="lora_rank_8", type="lora", rank=8, privacy_risk=0.25, seed=42, dataset=dataset, experiment={"batch_size": 4, "local_epochs": 1, "primary_metric": "accuracy"})
    rounds, summary = run_to_target(plan, target_quality=0.5, max_rounds=3)
    assert rounds
    assert summary["target_reached"]
    assert summary["validation_utility"] >= 0.5
    assert summary["communication_to_target_bytes"] > 0
