import json
from types import SimpleNamespace

import numpy as np

from scripts import run_real_federated_experiment as runner
from tradefl.data.loaders import DatasetRecord
from tradefl.federation.huggingface import ClientUpdate, pubmedqa_prompt, tensor_state_nbytes


def test_pubmedqa_prompt_constrains_output_labels():
    record = DatasetRecord("question: Does it work?\ncontext: Trial text", "yes", {})

    prompt = pubmedqa_prompt(record)

    assert "Does it work?" in prompt
    assert "yes, no, or maybe" in prompt
    assert prompt.endswith("answer:")


def test_tensor_state_bytes_are_real_array_payload_bytes():
    state = {"a": np.zeros((2, 3), dtype=np.float32), "b": np.zeros(4, dtype=np.int16)}

    assert tensor_state_nbytes(state) == 32


def test_end_to_end_runner_aggregates_clients_and_writes_real_provenance(tmp_path, monkeypatch):
    class FakeTrainer:
        def __init__(self, model_config, labels, training):
            self.labels = labels

        def initial_state(self):
            return {"adapter": np.zeros(1, dtype=np.float32)}

        def train_client(self, records, global_state):
            value = float(sum(record.metadata["index"] for record in records))
            state = {"adapter": np.array([value], dtype=np.float32)}
            return ClientUpdate(state, len(records), 0.1, 128, tensor_state_nbytes(state))

        def evaluate(self, records, global_state):
            return {"accuracy": 0.75, "macro_f1": 0.7}

    monkeypatch.setattr(runner, "HuggingFaceClientTrainer", FakeTrainer)
    records = [DatasetRecord(f"example {index}", "yes" if index % 2 else "no", {"index": index}) for index in range(10)]
    dataset = SimpleNamespace(train=records, validation=records[:2], test=records[2:4], labels=("no", "yes"), name="pubmedqa")
    round_path = tmp_path / "round_metrics.jsonl"

    summary = runner.run_model_experiment(
        {"experiment_id": "test_model", "model_id": "test/model", "architecture": "causal_lm", "method": "lora"},
        {},
        {
            "num_clients": 2,
            "client_sampling_ratio": 1.0,
            "max_rounds": 1,
            "target_quality": 0.7,
            "primary_metric": "accuracy",
            "reference_utility": 0.9,
        },
        dataset,
        42,
        round_path,
    )

    row = json.loads(round_path.read_text())
    assert row["training_mode"] == "real_federated"
    assert row["selected_clients"] == [0, 1]
    assert row["bytes_uploaded"] == 8
    assert row["bytes_downloaded"] == 8
    assert summary["training_mode"] == "real_federated"
    assert summary["target_reached"] is True
