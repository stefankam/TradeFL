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

from tradefl.utils.config import load_yaml

GRAPH_SPECS = [
    ("01_tradefl_score_by_plan.pdf", "tradefl_score", "Deployment-specific TradeFL score by plan", "TradeFL score", "bar"),
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
    ("13_model_quality_comparison.pdf", "validation_utility", "Validation/test utility by model plan", "Utility", "quality"),
]

PARETO_SPECS = [
    ("16_pareto_memory_communication.pdf", "memory_communication", "peak_memory_bytes", "communication_to_target_bytes", "Peak accelerator memory (MiB)", "Communication to target (MiB)", "min", "min", 2**20, 2**20),
    ("17_pareto_latency_energy.pdf", "latency_energy", "latency_to_target_seconds", "energy_to_target_joules", "Latency to target (seconds)", "Energy to target (joules)", "min", "min", 1.0, 1.0),
    ("18_pareto_privacy_accuracy.pdf", "privacy_accuracy", "privacy_risk", "validation_utility", "Privacy risk (unitless; lower is better)", "Validation utility (accuracy; higher is better)", "min", "max", 1.0, 1.0),
    ("19_pareto_time_communication.pdf", "time_communication", "compute_to_target_seconds", "communication_to_target_bytes", "Compute time to target (seconds)", "Communication to target (MiB)", "min", "min", 1.0, 2**20),
]

RAW_VALUE_COLUMNS = [
    "plan_id", "peak_memory_bytes", "compute_to_target_seconds", "communication_to_target_bytes",
    "latency_to_target_seconds", "energy_to_target_joules", "privacy_risk", "validation_utility",
    "test_utility", "rounds_to_target", "tradefl_score", "feasible", "status",
]

SUMMARY_AGGREGATIONS = {
    "tradefl_score": "mean",
    "peak_memory_bytes": "mean",
    "compute_to_target_seconds": "mean",
    "communication_to_target_bytes": "mean",
    "latency_to_target_seconds": "mean",
    "energy_to_target_joules": "mean",
    "validation_utility": "mean",
    "test_utility": "mean",
    "accuracy_loss": "mean",
    "privacy_risk": "mean",
    "rounds_to_target": "mean",
    "feasible": "max",
}

SUMMARIZE_RESULTS_API_VERSION = 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models-config", help="Show every configured real plan, marking plans without results N/A.")
    args = parser.parse_args()
    expected_plan_ids = None
    expected_status = None
    if args.models_config:
        model_config = load_yaml(args.models_config)
        expected_plan_ids = [item["experiment_id"] for item in model_config["federated_experiments"]]
        expected_status = {plan_id: "not_run" for plan_id in expected_plan_ids}
        for item in model_config.get("external_models", []):
            model_id = str(item.get("experiment_id", item["model_id"]))
            expected_plan_ids.append(model_id)
            expected_status[model_id] = "external_not_federated"
    summarize_results(
        Path(args.input),
        Path(args.output),
        expected_plan_ids=expected_plan_ids,
        expected_status=expected_status,
    )


def summarize_results(
    input_path: Path,
    output_prefix: Path,
    expected_plan_ids: list[str] | None = None,
    expected_status: dict[str, str] | None = None,
) -> None:
    """Write graphs only from results proven to come from real federated training."""

    df = pd.read_csv(input_path)
    allowed_modes = {"real_federated", "centralized_api", "centralized_local"}
    if "training_mode" not in df.columns:
        raise ValueError(
            "Input CSV must include training_mode='real_federated' or a recognized centralized mode; "
            "reference-model graphs are intentionally disabled."
        )
    unexpected = sorted(set(df["training_mode"].dropna().astype(str)) - allowed_modes)
    if unexpected or df["training_mode"].isna().any():
        raise ValueError(
            "Only real_federated, centralized_api, or centralized_local results can be graphed; "
            f"found {unexpected or ['<missing>']}"
        )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_frame(df)
    if expected_plan_ids:
        observed_ids = set(summary["plan_id"].astype(str))
        unexpected_ids = sorted(observed_ids - set(expected_plan_ids))
        ordered_ids = [*expected_plan_ids, *unexpected_ids]
        summary = summary.set_index("plan_id").reindex(ordered_ids).rename_axis("plan_id").reset_index()
        summary["observed"] = summary["plan_id"].isin(observed_ids)
        summary["status"] = [
            "completed" if plan_id in observed_ids else (expected_status or {}).get(plan_id, "not_run")
            for plan_id in summary["plan_id"]
        ]
    summary.to_csv(str(output_prefix) + ".csv", index=False)
    graph_dir = Path(str(output_prefix) + "_graphs")
    graph_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating PDF graphs in: {graph_dir.resolve()}", flush=True)
    written = write_pdf_graphs(df, summary, graph_dir)
    write_raw_value_tables(summary, graph_dir)
    pareto_rows = []
    for filename, tradeoff, x, y, xlabel, ylabel, x_direction, y_direction, x_scale, y_scale in PARETO_SPECS:
        path = graph_dir / filename
        pareto_rows.extend(
            _pareto_plot(summary, tradeoff, x, y, xlabel, ylabel, x_direction, y_direction, x_scale, y_scale, path)
        )
        written.append(path)
    invalid_graphs = [path for path in written if not path.exists() or path.stat().st_size == 0]
    if invalid_graphs:
        raise RuntimeError(f"Graph generation did not produce valid PDFs: {[str(path) for path in invalid_graphs]}")
    pd.DataFrame(pareto_rows).to_csv(graph_dir / "pareto_status.csv", index=False)
    coverage = summary[["plan_id"]].copy()
    coverage["observed"] = summary.get("observed", True)
    coverage["status"] = summary.get("status", coverage["observed"].map({True: "completed", False: "not_run"}))
    coverage.to_csv(graph_dir / "plan_coverage.csv", index=False)
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
        elif kind == "quality":
            _quality_plot(summary, title=title, path=path)
        else:
            _bar_plot(summary, metric=metric, title=title, ylabel=ylabel, path=path)
        written.append(path)
    raw_table = graph_dir / "14_raw_values_by_plan.pdf"
    _raw_value_table_plot(summary, raw_table)
    written.append(raw_table)
    resource_comparison = graph_dir / "15_model_resource_comparison.pdf"
    _resource_comparison_plot(summary, resource_comparison)
    written.append(resource_comparison)
    return written


def write_raw_value_tables(summary: pd.DataFrame, graph_dir: Path) -> None:
    """Write auditable values and units separately from visual encodings."""

    columns = [column for column in RAW_VALUE_COLUMNS if column in summary.columns]
    summary[columns].to_csv(graph_dir / "raw_values_by_plan.csv", index=False)
    units = {
        "peak_memory_bytes": "bytes", "compute_to_target_seconds": "seconds",
        "communication_to_target_bytes": "bytes", "latency_to_target_seconds": "seconds",
        "energy_to_target_joules": "joules", "privacy_risk": "unitless",
        "validation_utility": "accuracy fraction", "test_utility": "accuracy fraction",
        "rounds_to_target": "rounds", "tradefl_score": "unitless composite (deployment-specific)",
        "feasible": "boolean/rate",
    }
    pd.DataFrame([{"metric": metric, "unit": unit} for metric, unit in units.items()]).to_csv(
        graph_dir / "metric_units.csv", index=False
    )


def _raw_value_table_plot(summary: pd.DataFrame, path: Path) -> None:
    display_columns = [
        "plan_id", "validation_utility", "test_utility", "peak_memory_bytes",
        "communication_to_target_bytes", "compute_to_target_seconds", "latency_to_target_seconds",
        "energy_to_target_joules", "privacy_risk",
    ]
    table = summary[[column for column in display_columns if column in summary.columns]].copy()
    rename = {
        "plan_id": "Plan", "validation_utility": "Val. accuracy", "test_utility": "Test accuracy",
        "peak_memory_bytes": "Memory (MiB)", "communication_to_target_bytes": "Communication (MiB)",
        "compute_to_target_seconds": "Compute (s)", "latency_to_target_seconds": "Latency (s)",
        "energy_to_target_joules": "Energy (J)", "privacy_risk": "Privacy risk",
    }
    for column in ("peak_memory_bytes", "communication_to_target_bytes"):
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="coerce") / 2**20
    for column in table.columns:
        if column != "plan_id":
            table[column] = pd.to_numeric(table[column], errors="coerce").map(lambda value: "N/A" if pd.isna(value) else f"{value:.4g}")
    table = table.rename(columns=rename)
    fig, ax = plt.subplots(figsize=(16, max(3.5, 0.55 * len(table) + 2)))
    ax.axis("off")
    ax.set_title("Raw mean values by model plan (N/A = not measured)", pad=16)
    ax.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="center").auto_set_font_size(False)
    fig.axes[0].tables[0].set_fontsize(7)
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def _resource_comparison_plot(summary: pd.DataFrame, path: Path) -> None:
    specs = [
        ("peak_memory_bytes", "Peak accelerator memory (MiB)", 2**20),
        ("communication_to_target_bytes", "Communication to target (MiB)", 2**20),
        ("compute_to_target_seconds", "Compute time to target (seconds)", 1.0),
        ("latency_to_target_seconds", "Latency to target (seconds)", 1.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, (metric, title, scale) in zip(axes.flat, specs):
        values = pd.to_numeric(summary.get(metric, pd.Series(float("nan"), index=summary.index)), errors="coerce") / scale
        missing = values.isna()
        bars = ax.bar(summary["plan_id"].astype(str), values.fillna(0), color=["lightgray" if value else "tab:blue" for value in missing])
        for bar, is_missing in zip(bars, missing):
            if is_missing:
                bar.set_hatch("//")
                ax.annotate("N/A", (bar.get_x() + bar.get_width() / 2, 0), ha="center", va="bottom", fontsize=7)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
    fig.suptitle("LLaMA, DeepSeek, and Bio_ClinicalBERT resource comparison")
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def pareto_classification(frame: pd.DataFrame, x: str, y: str, x_direction: str, y_direction: str) -> dict[str, tuple[str, str]]:
    """Return plan -> (status, dominating plans) for a two-objective view."""

    numeric = frame[["plan_id"]].copy()
    numeric[x] = frame[x] if x in frame else float("nan")
    numeric[y] = frame[y] if y in frame else float("nan")
    numeric[x] = pd.to_numeric(numeric[x], errors="coerce")
    numeric[y] = pd.to_numeric(numeric[y], errors="coerce")
    result = {}
    for _, candidate in numeric.iterrows():
        plan_id = str(candidate["plan_id"])
        if pd.isna(candidate[x]) or pd.isna(candidate[y]):
            result[plan_id] = ("not_available", "")
            continue
        dominators = []
        for _, other in numeric.iterrows():
            if str(other["plan_id"]) == plan_id or pd.isna(other[x]) or pd.isna(other[y]):
                continue
            x_no_worse = other[x] <= candidate[x] if x_direction == "min" else other[x] >= candidate[x]
            y_no_worse = other[y] <= candidate[y] if y_direction == "min" else other[y] >= candidate[y]
            x_better = other[x] < candidate[x] if x_direction == "min" else other[x] > candidate[x]
            y_better = other[y] < candidate[y] if y_direction == "min" else other[y] > candidate[y]
            if x_no_worse and y_no_worse and (x_better or y_better):
                dominators.append(str(other["plan_id"]))
        result[plan_id] = ("dominated" if dominators else "pareto", ";".join(sorted(dominators)))
    return result


def _pareto_plot(summary, tradeoff, x, y, xlabel, ylabel, x_direction, y_direction, x_scale, y_scale, path):
    classification = pareto_classification(summary, x, y, x_direction, y_direction)
    fig, ax = plt.subplots(figsize=(10, 7))
    rows = []
    for _, row in summary.iterrows():
        plan_id = str(row["plan_id"])
        status, dominated_by = classification[plan_id]
        x_value = pd.to_numeric(pd.Series([row.get(x)]), errors="coerce").iloc[0]
        y_value = pd.to_numeric(pd.Series([row.get(y)]), errors="coerce").iloc[0]
        rows.append({"tradeoff": tradeoff, "plan_id": plan_id, "status": status, "dominated_by": dominated_by})
        if status == "not_available":
            continue
        marker, color = (("x", "tab:red") if status == "dominated" else ("o", "tab:green"))
        ax.scatter(x_value / x_scale, y_value / y_scale, marker=marker, color=color, s=75, label=status)
        ax.annotate(plan_id, (x_value / x_scale, y_value / y_scale), fontsize=7, xytext=(4, 4), textcoords="offset points")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), title="Dominance")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        _no_data(ax, f"No plans have both {x} and {y}")
    missing = [plan_id for plan_id, (status, _) in classification.items() if status == "not_available"]
    if missing:
        ax.text(0.01, 0.01, "N/A: " + ", ".join(missing), transform=ax.transAxes, fontsize=7, va="bottom")
    ax.set_title(f"Pareto view: {tradeoff.replace('_', ' ')}")
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return rows


def _bar_plot(summary: pd.DataFrame, metric: str, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if metric in summary.columns and not summary.empty:
        columns = ["plan_id", metric] + [column for column in ("observed", "status") if column in summary.columns]
        plot_data = summary[columns].copy()
        if "observed" not in plot_data.columns:
            plot_data = plot_data.sort_values(metric, na_position="last")
            plot_data["observed"] = True
        if metric == "feasible":
            plot_data[metric] = plot_data[metric].astype(float)
        missing = ~plot_data["observed"] | plot_data[metric].isna()
        bars = ax.bar(
            plot_data["plan_id"].astype(str),
            plot_data[metric].fillna(0.0),
            color=["lightgray" if value else "tab:blue" for value in missing],
        )
        for index, (bar, is_missing) in enumerate(zip(bars, missing)):
            if is_missing:
                bar.set_hatch("//")
                status = plot_data.iloc[index].get("status", "not_run")
                label = "N/A\nexternal" if status == "external_not_federated" else "N/A\nnot run"
                ax.annotate(label, (bar.get_x() + bar.get_width() / 2, 0), ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
    else:
        _no_data(ax, f"Missing metric: {metric}")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def _quality_plot(summary: pd.DataFrame, title: str, path: Path) -> None:
    """Compare validation and test utility across local and external model roles."""

    fig, ax = plt.subplots(figsize=(12, 6.5))
    positions = list(range(len(summary)))
    validation = pd.to_numeric(
        summary["validation_utility"] if "validation_utility" in summary else pd.Series(float("nan"), index=summary.index),
        errors="coerce",
    )
    test = pd.to_numeric(
        summary["test_utility"] if "test_utility" in summary else pd.Series(float("nan"), index=summary.index),
        errors="coerce",
    )
    width = 0.36
    ax.bar([position - width / 2 for position in positions], validation.fillna(0), width, label="Validation utility")
    ax.bar([position + width / 2 for position in positions], test.fillna(0), width, label="Test utility")
    status = summary.get("status", pd.Series(["completed"] * len(summary)))
    for position, state in zip(positions, status):
        if state != "completed":
            label = "external API\nnot FedAvg" if state == "external_not_federated" else "N/A\nnot run"
            ax.annotate(label, (position, 0), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions, summary["plan_id"].astype(str), rotation=35, ha="right")
    ax.set_ylabel("Utility")
    ax.set_title(title)
    ax.legend()
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
