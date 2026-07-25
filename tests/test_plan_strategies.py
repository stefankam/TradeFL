import importlib
import json

from tradefl.data import load_dataset_bundle


def _dataset(tmp_path):
    rows = [
        {"sentence1": "alpha beta gamma delta", "sentence2": "alpha", "gold_label": "entailment"},
        {"sentence1": "epsilon zeta eta theta", "sentence2": "different", "gold_label": "contradiction"},
        {"sentence1": "iota kappa lambda mu", "sentence2": "iota", "gold_label": "entailment"},
    ]
    for split in ["train", "validation", "test"]:
        (tmp_path / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
    return load_dataset_bundle(
        {
            "name": "mednli",
            "paths": {split: tmp_path / f"{split}.jsonl" for split in ["train", "validation", "test"]},
        }
    )


def _build(plan_type, dataset, **overrides):
    module = importlib.import_module(f"tradefl.plans.{plan_type}")
    config = {
        "plan_id": plan_type,
        "type": plan_type,
        "privacy_risk": 0.25,
        "seed": 42,
        "dataset": dataset,
        "experiment": {"batch_size": 2, "local_epochs": 1, "primary_metric": "accuracy"},
        **overrides,
    }
    return module.build(**config)


def test_plan_modules_construct_distinct_training_strategies(tmp_path):
    dataset = _dataset(tmp_path)
    full = _build("full_finetuning", dataset)
    lora = _build("lora", dataset, rank=2)
    qlora = _build("qlora", dataset, rank=2, quantization_bits=4)
    split = _build("splitfed", dataset, split_layer=2, activation_compression=True)
    distillation = _build("distillation", dataset)

    for plan in [full, lora, qlora, split, distillation]:
        plan.setup()
        plan.run_round(0)
        assert sum(plan.backend.class_counts.values()) == len(dataset.train)

    assert "alpha" in full.backend.vocabulary
    assert all(feature.startswith("adapter_") for feature in lora.backend.vocabulary)
    assert qlora.backend.serialized_size_bytes() < lora.backend.serialized_size_bytes()
    assert any(feature.startswith("client_") for feature in split.backend.vocabulary)
    assert any(feature.startswith("server_") for feature in split.backend.vocabulary)
    assert sum(distillation.teacher.class_counts.values()) == len(dataset.train)
