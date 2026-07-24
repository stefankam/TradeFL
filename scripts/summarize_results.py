#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

GRAPH_SPECS = [
    ("01_tradefl_score_by_plan.pdf", "tradefl_score", "TradeFL score by plan", "TradeFL score", "bar"),
    ("02_peak_memory_by_plan.pdf", "peak_memory_bytes", "Peak memory by plan", "Peak memory (bytes)", "bar"),
    ("03_compute_to_target_by_plan.pdf", "compute_to_target_seconds", "Compute to target by plan", "Compute seconds", "bar"),
    ("04_communication_to_target_by_plan.pdf", "communication_to_target_bytes", "Communication to target by plan", "Communication bytes", "bar"),
    ("05_latency_to_target_by_plan.pdf", "latency_to_target_seconds", "Latency to target by plan", "Latency seconds", "bar"),
    ("06_validation_utility_by_plan.pdf", "validation_utility", "Validation utility by plan", "Validation utility", "bar"),
    ("07_accuracy_loss_by_plan.pdf", "accuracy_loss", "Accuracy loss by plan", "Accuracy loss", "bar"),
    ("08_privacy_risk_by_plan.pdf", "privacy_risk", "Privacy risk by plan", "Privacy risk", "bar"),
    ("09_rounds_to_target_by_plan.pdf", "rounds_to_target", "Rounds to target by plan", "Rounds", "bar"),
    ("10_feasibility_by_plan.pdf", "feasible", "Feasibility by plan", "Feasible rate", "bar"),
    ("11_score_vs_accuracy_loss.pdf", "accuracy_loss", "TradeFL score vs accuracy loss", "Accuracy loss", "scatter_score"),
    ("12_memory_vs_communication.pdf", "peak_memory_bytes", "Memory vs communication", "Peak memory bytes", "scatter_comm"),
]

SUMMARY_AGGREGATIONS = {
    "tradefl_score": "mean",
    "peak_memory_bytes": "mean",
    "compute_to_target_seconds": "mean",
    "communication_to_target_bytes": "mean",
    "latency_to_target_seconds": "mean",
    "validation_utility": "mean",
    "accuracy_loss": "mean",
    "privacy_risk": "mean",
    "rounds_to_target": "mean",
    "feasible": "max",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summarize_results(Path(args.input), Path(args.output))


def summarize_results(input_path: Path, output_prefix: Path) -> None:
    """Write CSV summary and at least ten PDF graphs from a plan_summary CSV."""

    df = pd.read_csv(input_path)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_frame(df)
    summary.to_csv(str(output_prefix) + ".csv", index=False)
    graph_dir = Path(str(output_prefix) + "_graphs")
    graph_dir.mkdir(parents=True, exist_ok=True)
    written = write_pdf_graphs(df, summary, graph_dir)
    manifest = pd.DataFrame({"pdf_graph": [path.name for path in written], "path": [str(path) for path in written]})
    manifest.to_csv(graph_dir / "graph_manifest.csv", index=False)


def summarize_frame(df: pd.DataFrame) -> pd.DataFrame:
    aggregations = {column: agg for column, agg in SUMMARY_AGGREGATIONS.items() if column in df.columns}
    if "plan_id" not in df.columns:
        raise ValueError("Input CSV must include a plan_id column.")
    if not aggregations:
        return df[["plan_id"]].drop_duplicates().sort_values("plan_id")
    return df.groupby("plan_id", as_index=False).agg(aggregations)


def write_pdf_graphs(df: pd.DataFrame, summary: pd.DataFrame, graph_dir: Path) -> list[Path]:
    """Create at least ten auditable PDF graph files."""

    written: list[Path] = []
    for filename, metric, title, ylabel, kind in GRAPH_SPECS:
        path = graph_dir / filename
        if kind == "scatter_score":
            _scatter_plot(df, x=metric, y="tradefl_score", title=title, xlabel=ylabel, ylabel="TradeFL score", path=path)
        elif kind == "scatter_comm":
            _scatter_plot(df, x=metric, y="communication_to_target_bytes", title=title, xlabel=ylabel, ylabel="Communication bytes", path=path)
        else:
            _bar_plot(summary, metric=metric, title=title, ylabel=ylabel, path=path)
        written.append(path)
    return written


def _bar_plot(summary: pd.DataFrame, metric: str, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if metric in summary.columns and not summary.empty:
        plot_data = summary[["plan_id", metric]].copy().sort_values(metric, na_position="last")
        if metric == "feasible":
            plot_data[metric] = plot_data[metric].astype(float)
        ax.bar(plot_data["plan_id"].astype(str), plot_data[metric].fillna(0.0))
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
    else:
        _no_data(ax, f"Missing metric: {metric}")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def _scatter_plot(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if x in df.columns and y in df.columns and not df.empty:
        ax.scatter(pd.to_numeric(df[x], errors="coerce"), pd.to_numeric(df[y], errors="coerce"))
        if "plan_id" in df.columns:
            for _, row in df.iterrows():
                x_value = pd.to_numeric(pd.Series([row[x]]), errors="coerce").iloc[0]
                y_value = pd.to_numeric(pd.Series([row[y]]), errors="coerce").iloc[0]
                if pd.notna(x_value) and pd.notna(y_value):
                    ax.annotate(str(row["plan_id"]), (x_value, y_value), fontsize=7, alpha=0.75)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        _no_data(ax, f"Missing metrics: {x}, {y}")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def _no_data(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    main()
