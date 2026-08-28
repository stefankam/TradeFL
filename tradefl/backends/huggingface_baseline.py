"""Centralized local Hugging Face baseline evaluation."""
from __future__ import annotations

import time
from typing import Any

from tradefl.federation.huggingface import HuggingFaceClientTrainer
from tradefl.measurement.energy import EnergyMeter


def evaluate_huggingface_baseline(model_config: dict[str, Any], dataset, training: dict[str, Any]) -> dict[str, Any]:
    """Evaluate an open-weight model locally without training or FedAvg."""

    trainer = HuggingFaceClientTrainer(model_config, dataset.labels, training)
    meter = EnergyMeter()
    meter.start()
    started = time.perf_counter()
    metrics = trainer.evaluate_centralized_splits(dataset.validation, dataset.test)
    latency = time.perf_counter() - started
    energy = meter.stop()
    return {
        "plan_id": model_config.get("experiment_id", f"{model_config['model_id']}_centralized_baseline"),
        "model_id": model_config["model_id"],
        "method": "centralized_huggingface",
        "seed": 42,
        "dataset": dataset.name,
        "training_mode": "centralized_local",
        "aggregation": "none",
        "validation_utility": metrics["validation"]["accuracy"],
        "validation_macro_f1": metrics["validation"]["macro_f1"],
        "test_utility": metrics["test"]["accuracy"],
        "test_macro_f1": metrics["test"]["macro_f1"],
        "rounds_completed": 0,
        "rounds_to_target": None,
        "target_reached": False,
        "peak_memory_bytes": metrics["peak_memory_bytes"],
        "compute_to_target_seconds": latency,
        "communication_to_target_bytes": 0,
        "latency_to_target_seconds": latency,
        "mean_round_latency_seconds": latency,
        "privacy_risk": float(model_config.get("privacy_risk", 0.05)),
        "energy_to_target_joules": energy,
    }

