#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradefl.data import load_dataset_bundle
from tradefl.federation.simulator import run_to_target
from tradefl.utils.config import load_yaml
from scripts.obtain_data import ensure_dataset_available
from tradefl.utils.seeds import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    exp = cfg["experiment"]
    ensure_dataset_available(cfg["dataset"], seed=int(exp.get("seeds", [42])[0]) if isinstance(exp.get("seeds", [42]), list) else 42)
    dataset = load_dataset_bundle(cfg["dataset"])
    plans = load_yaml(exp.get("plans_config", "configs/plans.yaml"))["plans"]
    out = Path(exp.get("output_dir", "outputs"))
    out.mkdir(exist_ok=True)
    if exp.get("overwrite", False):
        for name in ["raw_metrics.csv", "round_metrics.jsonl"]:
            (out / name).unlink(missing_ok=True)
    summaries = []
    for seed in exp["seeds"]:
        set_seed(seed)
        for plan_cfg in plans:
            module = importlib.import_module(f"tradefl.plans.{plan_cfg['type']}")
            plan = module.build(seed=seed, dataset=dataset, experiment=exp, **plan_cfg)
            rounds, summary = run_to_target(plan, exp["target_quality"], exp["max_rounds"])
            with (out / "round_metrics.jsonl").open("a", encoding="utf-8") as handle:
                for row in rounds:
                    handle.write(json.dumps(row, default=str) + "\n")
            summary.update(
                {
                    "plan_id": plan_cfg["plan_id"],
                    "seed": seed,
                    "dataset": dataset.name,
                    "accuracy_loss": max(0, exp.get("reference_utility", 1.0) - summary["validation_utility"]),
                }
            )
            summaries.append(summary)
    pd.DataFrame(summaries).to_csv(out / "raw_metrics.csv", index=False)


if __name__ == "__main__":
    main()
