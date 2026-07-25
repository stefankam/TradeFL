#!/usr/bin/env python
"""Validate model roles for the planned real federated experiments."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradefl.utils.config import load_yaml


LOCAL_ROLES = {"federated_participant"}
SUPPORTED_ARCHITECTURES = {"causal_lm", "sequence_classification"}
SUPPORTED_METHODS = {"full_finetuning", "lora", "qlora"}


def validate_config(config: dict[str, Any]) -> list[str]:
    """Return configuration errors without importing heavyweight ML packages."""

    errors: list[str] = []
    experiments = config.get("federated_experiments", [])
    external = config.get("external_models", [])
    if not experiments:
        errors.append("at least one federated_experiment is required")

    experiment_ids: set[str] = set()
    for item in experiments:
        experiment_id = str(item.get("experiment_id", ""))
        if not experiment_id:
            errors.append("every federated experiment requires experiment_id")
        elif experiment_id in experiment_ids:
            errors.append(f"duplicate experiment_id: {experiment_id}")
        experiment_ids.add(experiment_id)
        if item.get("role") not in LOCAL_ROLES:
            errors.append(f"{experiment_id or '<unnamed>'}: federated models must use role=federated_participant")
        if item.get("architecture") not in SUPPORTED_ARCHITECTURES:
            errors.append(f"{experiment_id or '<unnamed>'}: unsupported architecture")
        if item.get("method") not in SUPPORTED_METHODS:
            errors.append(f"{experiment_id or '<unnamed>'}: unsupported training method")
        if not item.get("model_id"):
            errors.append(f"{experiment_id or '<unnamed>'}: model_id is required")

    for item in external:
        model_id = str(item.get("model_id", "<unnamed>"))
        if item.get("federated_trainable") is not False:
            errors.append(f"{model_id}: API-only external models cannot be marked federated_trainable")
        if item.get("role") == "federated_participant":
            errors.append(f"{model_id}: external API models cannot participate in FedAvg")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_federated_models.yaml")
    args = parser.parse_args()
    errors = validate_config(load_yaml(Path(args.config)))
    if errors:
        raise SystemExit("Invalid real-federated config:\n- " + "\n- ".join(errors))
    print(f"Valid real-federated model-role configuration: {args.config}")


if __name__ == "__main__":
    main()
