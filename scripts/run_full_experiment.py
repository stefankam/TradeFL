#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradefl.data import load_dataset_bundle
from tradefl.federation.simulator import run_to_target
from tradefl.utils.config import load_yaml
from tradefl.utils.seeds import set_seed


def build_plan(plan_id: str, seed: int, plans_cfg: list[dict], dataset, experiment: dict):
    cfg = next(plan for plan in plans_cfg if plan["plan_id"] == plan_id)
    module = importlib.import_module(f"tradefl.plans.{cfg['type']}")
    return module.build(seed=seed, dataset=dataset, experiment=experiment, **cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    exp = cfg["experiment"]
    dataset = load_dataset_bundle(cfg["dataset"])
    plans = load_yaml(exp.get("plans_config", "configs/plans.yaml"))["plans"]
    set_seed(args.seed)
    plan = build_plan(args.plan, args.seed, plans, dataset, exp)
    rounds, summary = run_to_target(plan, exp["target_quality"], exp["max_rounds"])
    out = Path(exp.get("output_dir", "outputs"))
    out.mkdir(exist_ok=True)
    with (out / "round_metrics.jsonl").open("a", encoding="utf-8") as handle:
        for row in rounds:
            handle.write(json.dumps(row, default=str) + "\n")
    summary.update(
        {
            "plan_id": args.plan,
            "seed": args.seed,
            "dataset": dataset.name,
            "accuracy_loss": max(0, exp.get("reference_utility", 1.0) - summary["validation_utility"]),
        }
    )
    path = out / "raw_metrics.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(summary)


if __name__ == "__main__":
    main()
