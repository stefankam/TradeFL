"""Plotting utilities for TokenOS experiments."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import importlib.util

PLOTS = {"utility":"utility.png", "accuracy":"accuracy.png", "energy":"energy.png", "latency":"latency.png", "memory_pressure":"memory_pressure.png", "communication":"communication.png", "successful_clients":"successful_clients.png"}

def generate_plots(metrics: Any, output: str) -> None:
    """Generate comparison plots, or placeholder files when matplotlib is unavailable."""
    out=Path(output); out.mkdir(parents=True, exist_ok=True)
    if not importlib.util.find_spec("matplotlib"):
        for filename in PLOTS.values():
            (out/filename).write_text("matplotlib is unavailable in this environment; install requirements.txt to render plots.\n", encoding="utf-8")
        return
    import matplotlib.pyplot as plt
    for metric, filename in PLOTS.items():
        plt.figure(figsize=(8,5))
        for system, group in metrics.groupby("system"):
            plt.plot(group["round"], group[metric], label=system, linewidth=1.6)
        plt.xlabel("Round"); plt.ylabel(metric.replace("_"," ").title()); plt.title(f"{metric.replace('_',' ').title()} over rounds")
        plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(out/filename, dpi=150); plt.close()
