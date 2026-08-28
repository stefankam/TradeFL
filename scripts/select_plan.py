#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradefl.selection.feasibility import Constraints
from tradefl.selection.normalization import Budgets
from tradefl.selection.scoring import TradeFLWeights
from tradefl.selection.selector import select_best, write_selection
from tradefl.utils.config import load_yaml

SELECT_PLAN_API_VERSION = 3

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--budgets", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--scenario")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--weight-method", choices=["configured", "entropy"], default="configured")

    args = parser.parse_args()
    rows = pd.read_csv(args.results).to_dict("records")
    budget_cfg = load_yaml(args.budgets)
    weight_cfg = load_yaml(args.weights)
    experiment_cfg = load_yaml(args.config)
    scenario = args.scenario or weight_cfg.get("default_scenario", "equal")
    weights = TradeFLWeights(**weight_cfg["scenarios"][scenario])
    if args.weight_method == "entropy":
        weights = entropy_weights(rows, budget_cfg.get("enabled_metrics", {}), weights)

    result = select_best(
        rows,
        Budgets(**budget_cfg["budgets"]),
        weights,
        Constraints(**experiment_cfg["constraints"]),
        budget_cfg.get("enabled_metrics", {}),
        experiment_cfg["experiment"].get("reference_utility", 1.0),
    )


    write_selection(result, args.output_dir or experiment_cfg["experiment"].get("output_dir", "outputs"))


def entropy_weights(rows: list[dict], enabled: dict[str, bool], fallback: TradeFLWeights) -> TradeFLWeights:
    """Derive objective weights from cross-plan metric information diversity."""

    columns = {
        "memory": "peak_memory_bytes",
        "compute": "compute_to_target_seconds",
        "communication": "communication_to_target_bytes",
        "energy": "energy_to_target_joules",
        "latency": "latency_to_target_seconds",
        "privacy": "privacy_risk",
        "accuracy": "accuracy_loss",
    }
    divergences = {}
    for metric, column in columns.items():
        values = pd.to_numeric(pd.Series([row.get(column) for row in rows]), errors="coerce").dropna()
        if not enabled.get(metric, True) or len(values) < 2 or values.max() == values.min():
            divergences[metric] = 0.0
            continue
        scaled = (values - values.min()) / (values.max() - values.min())
        probabilities = scaled / scaled.sum() if scaled.sum() else scaled
        entropy = -sum(value * math.log(value) for value in probabilities if value > 0) / math.log(len(values))
        divergences[metric] = 1.0 - entropy
    total = sum(divergences.values())
    if total <= 0:
        return fallback
    return TradeFLWeights(**{metric: value / total for metric, value in divergences.items()})


if __name__ == "__main__":
    main()
