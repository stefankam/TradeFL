"""Dataset-backed fine-tuning plan builder."""
from .base import build_dataset_plan

def build(**kwargs):
    plan_cfg = {key: value for key, value in kwargs.items() if key not in {"dataset", "seed", "experiment"}}
    return build_dataset_plan(plan_cfg, kwargs["dataset"], kwargs["seed"], kwargs["experiment"])
