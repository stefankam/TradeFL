#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradefl.selection.feasibility import Constraints
from tradefl.selection.normalization import Budgets
from tradefl.selection.scoring import TradeFLWeights
from tradefl.selection.selector import select_best, write_selection
from tradefl.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--budgets", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--scenario")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    rows = pd.read_csv(args.results).to_dict("records")
    budget_cfg = load_yaml(args.budgets)
    weight_cfg = load_yaml(args.weights)
    experiment_cfg = load_yaml(args.config)
    scenario = args.scenario or weight_cfg.get("default_scenario", "equal")
    result = select_best(
        rows,
        Budgets(**budget_cfg["budgets"]),
        TradeFLWeights(**weight_cfg["scenarios"][scenario]),
        Constraints(**experiment_cfg["constraints"]),
        budget_cfg.get("enabled_metrics", {}),
        experiment_cfg["experiment"].get("reference_utility", 1.0),
    )
    write_selection(result, experiment_cfg["experiment"].get("output_dir", "outputs"))


if __name__ == "__main__":
    main()
