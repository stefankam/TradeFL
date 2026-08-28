"""Centralized OpenAI API baseline evaluation for PubMedQA."""
from __future__ import annotations

import importlib
import time
from typing import Any

from tradefl.data.loaders import DatasetRecord
from tradefl.federation.huggingface import pubmedqa_prompt


def evaluate_openai_baseline(
    model_config: dict[str, Any],
    validation: list[DatasetRecord],
    test: list[DatasetRecord],
    labels: list[str],
    client: Any | None = None,
) -> dict[str, Any]:
    """Evaluate an API model centrally; no API weights participate in FedAvg."""

    if client is None:
        openai = importlib.import_module("openai")
        client = openai.OpenAI()
    started = time.perf_counter()
    validation_predictions, validation_bytes = _predict(client, model_config["model_id"], validation, labels)
    test_predictions, test_bytes = _predict(client, model_config["model_id"], test, labels)
    latency = time.perf_counter() - started
    validation_metrics = _metrics(validation_predictions, validation, labels)
    test_metrics = _metrics(test_predictions, test, labels)
    return {
        "plan_id": model_config.get("experiment_id", f"{model_config['model_id']}_centralized_baseline"),
        "model_id": model_config["model_id"],
        "method": "centralized_api",
        "seed": 42,
        "dataset": "pubmedqa",
        "training_mode": "centralized_api",
        "aggregation": "none",
        "validation_utility": validation_metrics["accuracy"],
        "validation_macro_f1": validation_metrics["macro_f1"],
        "test_utility": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "rounds_completed": 0,
        "rounds_to_target": None,
        "target_reached": False,
        "peak_memory_bytes": 0,
        "compute_to_target_seconds": latency,
        "communication_to_target_bytes": validation_bytes + test_bytes,
        "latency_to_target_seconds": latency,
        "mean_round_latency_seconds": latency,
        "privacy_risk": float(model_config.get("privacy_risk", 0.75)),
        "energy_to_target_joules": None,
    }


def _predict(client: Any, model_id: str, records: list[DatasetRecord], labels: list[str]) -> tuple[list[str], int]:
    predictions = []
    transferred = 0
    for record in records:
        prompt = pubmedqa_prompt(record) + "\nReturn exactly one label: yes, no, or maybe."
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4,
        )
        answer = str(response.choices[0].message.content or "").strip().lower()
        predictions.append(next((label for label in labels if answer == label or label in answer.split()), labels[0]))
        transferred += len(prompt.encode("utf-8")) + len(answer.encode("utf-8"))
    return predictions, transferred


def _metrics(predictions: list[str], records: list[DatasetRecord], labels: list[str]) -> dict[str, float]:
    accuracy = sum(prediction == record.label for prediction, record in zip(predictions, records)) / len(records)
    f1s = []
    for label in labels:
        tp = sum(pred == label and row.label == label for pred, row in zip(predictions, records))
        fp = sum(pred == label and row.label != label for pred, row in zip(predictions, records))
        fn = sum(pred != label and row.label == label for pred, row in zip(predictions, records))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": accuracy, "macro_f1": sum(f1s) / len(f1s)}
