#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from tradefl.scheduling import replay_schedulers
from tradefl.utils.config import load_yaml

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 13,
        "figure.titlesize": 22,
    }
)

ANNOTATION_FONT_SIZE = 13
PARETO_ANNOTATION_FONT_SIZE = 16

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
    "mean_round_latency_seconds", "p95_round_latency_seconds", "latency_to_target_seconds", "energy_to_target_joules", "privacy_risk", "validation_utility",
    "test_utility", "rounds_to_target", "tradefl_score", "feasible", "status",
]

SUMMARY_AGGREGATIONS = {
    "tradefl_score": "mean",
    "peak_memory_bytes": "mean",
    "compute_to_target_seconds": "mean",
    "communication_to_target_bytes": "mean",
    "latency_to_target_seconds": "mean",
    "mean_round_latency_seconds": "mean",
    "p95_round_latency_seconds": "mean",
    "energy_to_target_joules": "mean",
    "validation_utility": "mean",
    "test_utility": "mean",
    "accuracy_loss": "mean",
    "privacy_risk": "mean",
    "rounds_to_target": "mean",
    "feasible": "max",
}

SUMMARIZE_RESULTS_API_VERSION = 3

SCHEDULER_GRAPH_SPECS = [
    ("20_scheduler_overall_task_utility.pdf", "overall_task_utility", "Overall task utility", "Mean test utility"),
    ("20b_scheduler_validation_utility.pdf", "validation_task_utility", "Validation utility used for selection", "Mean validation utility"),
    ("20c_scheduler_mean_round_latency.pdf", "mean_round_latency_seconds", "Mean round latency", "Seconds"),
    ("20d_scheduler_p95_round_latency.pdf", "p95_round_latency_seconds", "P95 round latency", "Seconds"),
    ("20e_scheduler_peak_client_memory.pdf", "peak_client_memory_bytes", "Peak client memory", "Bytes"),
    ("20f_scheduler_rounds_to_target.pdf", "rounds_to_target", "Rounds to target or censoring", "Observed rounds"),
    ("21_scheduler_deadline_violation_rate.pdf", "deadline_slo_violation_rate", "Deadline/SLO violation rate", "Violation rate"),
    ("22_scheduler_resource_violation_rate.pdf", "memory_resource_violation_rate", "Memory/resource violation rate", "Violation rate"),
    ("23_scheduler_communication_cost.pdf", "communication_cost_bytes", "Communication cost", "Mean bytes per decision"),
    ("24_scheduler_energy_consumption.pdf", "energy_consumption_joules", "Energy consumption", "Mean joules per decision"),
    ("25_scheduler_time_to_target.pdf", "attained_time_to_target_seconds", "Attained time to target", "Mean seconds among attained tasks"),
    ("26_scheduler_scheduling_overhead.pdf", "scheduling_overhead_microseconds", "Scheduling overhead", "Mean microseconds per decision"),
    ("27_scheduler_target_attainment_rate.pdf", "target_attainment_rate", "Target attainment rate", "Attainment rate"),
    ("28_scheduler_censored_horizon.pdf", "censored_observation_horizon_seconds", "Censored observation horizon", "Mean seconds among unattained tasks"),
    ("32_scheduler_feasible_selection_rate.pdf", "feasible_selection_rate", "Feasible-plan selection rate", "Feasible selection rate"),
    ("33_scheduler_test_regret_to_oracle.pdf", "test_utility_regret_to_oracle", "Test-utility regret relative to oracle", "Oracle test utility - policy test utility"),
]

CLIENT_POPULATION_GRAPH_SPECS = [
    ("40_client_population_test_utility.pdf", "test_utility", "Test utility", "Mean test utility"),
    ("41_client_population_rounds_completed.pdf", "rounds_completed", "Rounds completed", "Mean rounds"),
    ("42_client_population_peak_memory.pdf", "peak_memory_bytes", "Peak memory", "Mean peak memory (bytes)"),
    ("43_client_population_compute.pdf", "compute_to_target_seconds", "Compute", "Mean compute seconds"),
    ("44_client_population_communication.pdf", "communication_to_target_bytes", "Communication", "Mean communication bytes"),
    ("45_client_population_latency.pdf", "latency_to_target_seconds", "Latency", "Mean latency seconds"),
    ("46_client_population_energy.pdf", "energy_to_target_joules", "Energy", "Mean energy (joules)"),
]




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models-config", help="Show every configured real plan, marking plans without results N/A.")
    parser.add_argument("--config", help="Experiment config supplying replay resource and SLO constraints.")
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
        constraints=load_yaml(args.config).get("constraints", {}) if args.config else None,
    )


def summarize_results(
    input_path: Path,
    output_prefix: Path,
    expected_plan_ids: list[str] | None = None,
    expected_status: dict[str, str] | None = None,
    constraints: dict | None = None,
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
    presentation_df = plan_presentation_frame(df)
    summary = summarize_frame(presentation_df)
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
    written = write_pdf_graphs(presentation_df, summary, graph_dir)
    write_raw_value_tables(summary, graph_dir)
    pareto_rows = []
    for filename, tradeoff, x, y, xlabel, ylabel, x_direction, y_direction, x_scale, y_scale in PARETO_SPECS:
        path = graph_dir / filename
        pareto_rows.extend(
            _pareto_plot(summary, tradeoff, x, y, xlabel, ylabel, x_direction, y_direction, x_scale, y_scale, path)
        )
        written.append(path)
    written.extend(write_scheduler_replay_graphs(df, graph_dir, constraints or {}))
    written.extend(write_client_population_graphs(df, graph_dir))
    write_scalability_diagnostics(df, graph_dir)
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



def write_client_population_graphs(df: pd.DataFrame, graph_dir: Path) -> list[Path]:
    """Compare no-selection experiments in which every logical client participates."""

    required = {"client_population", "participation_mode"}
    if not required.issubset(df.columns):
        return []
    rows = df.loc[df["participation_mode"] == "full_participation"].copy()
    rows["client_population"] = pd.to_numeric(rows["client_population"], errors="coerce")
    rows = rows.dropna(subset=["client_population"])
    if rows.empty:
        return []
    populations = sorted(rows["client_population"].astype(int).unique())
    comparison_rows = []
    written = []
    for filename, metric, title, ylabel in CLIENT_POPULATION_GRAPH_SPECS:
        if metric not in rows or pd.to_numeric(rows[metric], errors="coerce").notna().sum() == 0:
            continue
        means, lows, highs = [], [], []
        for population in populations:
            values = pd.to_numeric(
                rows.loc[rows["client_population"] == population, metric], errors="coerce",
            ).dropna()
            mean = float(values.mean()) if not values.empty else float("nan")
            half = 1.96 * float(values.std(ddof=1)) / len(values) ** 0.5 if len(values) > 1 else 0.0
            means.append(mean)
            lows.append(mean - half)
            highs.append(mean + half)
            comparison_rows.append({
                "client_population": population,
                "participation_mode": "full_participation",
                "metric": metric,
                "mean": mean,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
                "observation_count": len(values),
            })
        fig, ax = plt.subplots(figsize=(10, 7))
        values = pd.Series(means, dtype=float)
        errors = [values - pd.Series(lows), pd.Series(highs) - values]
        bars = ax.bar([str(value) for value in populations], values, color="tab:green", alpha=0.85)
        ax.errorbar(range(len(populations)), values, yerr=errors, fmt="none", color="black", capsize=6)
        for index, population in enumerate(populations):
            points = pd.to_numeric(
                rows.loc[rows["client_population"] == population, metric], errors="coerce",
            ).dropna()
            ax.scatter([index] * len(points), points, color="white", edgecolor="black", s=50, zorder=4)
        for bar, value in zip(bars, values):
            ax.annotate(
                "N/A" if pd.isna(value) else f"{value:.3g}",
                (bar.get_x() + bar.get_width() / 2, 0 if pd.isna(value) else value),
                ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE,
            )
        ax.set_title(f"Full participation: {title} by client population")
        ax.set_xlabel("Logical clients (all participate each round)")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        path = graph_dir / filename
        fig.savefig(path, format="pdf")
        plt.close(fig)
        written.append(path)
    pd.DataFrame(comparison_rows).to_csv(graph_dir / "client_population_comparison.csv", index=False)
    return written


def write_scalability_diagnostics(df: pd.DataFrame, graph_dir: Path) -> None:
    """Flag population-study measurements that require provenance review."""

    required = {"client_population", "peak_memory_bytes", "participation_mode"}
    if not required.issubset(df.columns):
        return
    rows = df.loc[df["participation_mode"] == "full_participation"].copy()
    if rows.empty:
        return
    plan_column = "base_experiment_id" if "base_experiment_id" in rows else "plan_id"
    rows["client_population"] = pd.to_numeric(rows["client_population"], errors="coerce")
    rows["peak_memory_bytes"] = pd.to_numeric(rows["peak_memory_bytes"], errors="coerce")
    diagnostics = []
    for plan_id, group in rows.dropna(subset=["client_population", "peak_memory_bytes"]).groupby(plan_column):
        means = group.groupby("client_population")["peak_memory_bytes"].mean().sort_index()
        for (lower_population, lower), (upper_population, upper) in zip(means.items(), list(means.items())[1:]):
            ratio = upper / lower if lower > 0 else float("nan")
            diagnostics.append({
                "plan_id": plan_id,
                "lower_client_population": int(lower_population),
                "upper_client_population": int(upper_population),
                "lower_mean_peak_memory_bytes": lower,
                "upper_mean_peak_memory_bytes": upper,
                "upper_to_lower_memory_ratio": ratio,
                "requires_review": bool(pd.notna(ratio) and (ratio < 0.75 or ratio > 1.25)),
                "reason": (
                    "peak accelerator memory changed by more than 25%; verify model/method provenance and raw per-client peaks"
                    if pd.notna(ratio) and (ratio < 0.75 or ratio > 1.25) else "within tolerance"
                ),
            })
    pd.DataFrame(diagnostics).to_csv(graph_dir / "client_population_memory_diagnostics.csv", index=False)



def summarize_frame(df: pd.DataFrame) -> pd.DataFrame:
    aggregations = {column: agg for column, agg in SUMMARY_AGGREGATIONS.items() if column in df.columns}
    if "plan_id" not in df.columns:
        raise ValueError("Input CSV must include a plan_id column.")
    if not aggregations:
        return df[["plan_id"]].drop_duplicates().sort_values("plan_id")
    return df.groupby("plan_id", as_index=False).agg(aggregations)


def plan_presentation_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return one scheduler treatment per base model for model-plan figures.

    Scheduler policies remain separate experimental rows in raw outputs and in
    scheduler figures. Model-plan figures preserve their historical model-only
    x-axis by preferring the dynamic TradeFL treatment and falling back to the
    first available treatment for an incomplete live run.
    """

    if "base_experiment_id" not in df.columns or "scheduler_policy" not in df.columns:
        return df.copy()
    federated = df.loc[df["training_mode"] == "real_federated"].copy()
    external = df.loc[df["training_mode"] != "real_federated"].copy()
    priority = {
        "tradefl_dynamic": 0,
        "tradefl_fixed": 1,
        "independent": 2,
        "static_weighted_sum": 3,
        "greedy": 4,
        "seeded_random": 5,
    }
    federated["_scheduler_priority"] = federated["scheduler_policy"].map(priority).fillna(99)
    keys = ["base_experiment_id"] + (["seed"] if "seed" in federated.columns else [])
    federated = federated.sort_values("_scheduler_priority").drop_duplicates(keys).drop(columns="_scheduler_priority")
    federated["plan_id"] = federated["base_experiment_id"].fillna(federated["plan_id"])
    return pd.concat([federated, external], ignore_index=True, sort=False)


def write_scheduler_replay_graphs(df: pd.DataFrame, graph_dir: Path, constraints: dict) -> list[Path]:
    """Plot executed online treatments; retain replay only as a diagnostic."""

    replay = replay_schedulers(df, constraints)
    online = _online_training_scheduler_results(df, constraints)
    if online is None and replay is None:
        return []
    if replay is not None:
        replay.decisions.to_csv(graph_dir / "offline_replay_decisions.csv", index=False)
        replay.outcomes.to_csv(graph_dir / "offline_replay_comparison.csv", index=False)
        replay.per_seed_outcomes.to_csv(graph_dir / "offline_replay_comparison_per_seed.csv", index=False)
        replay.price_trace.to_csv(graph_dir / "offline_replay_shadow_price_trace.csv", index=False)
    if online is None:
        outcomes, per_seed, definitions = replay.outcomes, replay.per_seed_outcomes, replay.definitions
        outcomes.insert(1, "result_source", "offline_replay_diagnostic")
        per_seed.insert(2, "result_source", "offline_replay_diagnostic")
    else:
        outcomes, per_seed, definitions = online
    outcomes.to_csv(graph_dir / "scheduler_comparison.csv", index=False)
    per_seed.to_csv(graph_dir / "scheduler_comparison_per_seed.csv", index=False)
    definitions.to_csv(graph_dir / "scheduler_definitions.csv", index=False)
    pd.DataFrame([{
        "artifact": "20-33 scheduler PDFs and scheduler_comparison*.csv",
        "result_source": str(outcomes["result_source"].iloc[0]),
        "proof_column": "scheduler_policy",
        "note": "Rows are aggregated from separately executed training treatments; offline replay is stored only under offline_replay_*.",
    }]).to_csv(graph_dir / "scheduler_result_provenance.csv", index=False)
    pd.DataFrame([{
        "seed_count": int(per_seed["seed"].nunique()),
        "recommended_minimum_seed_count": 10,
        "intervals_are_descriptive": True,
        "adequate_for_inference": bool(per_seed["seed"].nunique() >= 10),
        "note": "Paired Student-t intervals describe observed seeds; no hypothesis test is claimed.",
    }]).to_csv(graph_dir / "scheduler_statistical_scope.csv", index=False)
    written = []
    for filename, metric, title, ylabel in SCHEDULER_GRAPH_SPECS:
        path = graph_dir / filename
        _scheduler_bar_plot(outcomes, per_seed, metric, title + " — online training", ylabel, path)
        written.append(path)
    dashboard = graph_dir / "29_scheduler_comparison_dashboard.pdf"
    _scheduler_dashboard(outcomes, per_seed, dashboard)
    written.append(dashboard)
    actions = graph_dir / "30_scheduler_selected_actions.pdf"
    if replay is not None:
        _scheduler_action_plot(replay.decisions, actions)
        written.append(actions)
        prices = graph_dir / "31_scheduler_shadow_price_trajectory.pdf"
        _shadow_price_trajectory_plot(replay.price_trace, prices)
        written.append(prices)
    return written


def _online_training_scheduler_results(df: pd.DataFrame, constraints: dict):
    """Aggregate only separately executed training treatments, never replay choices."""

    required = {"scheduler_policy", "seed", "test_utility", "validation_utility"}
    if not required.issubset(df.columns):
        return None
    rows = df.loc[df["scheduler_policy"].notna()].copy()
    rows = rows.loc[~rows["scheduler_policy"].isin(["seeded_random", "centralized_baseline"])]
    if rows.empty:
        return None
    labels = {
        "random_feasible": "Random feasible", "fedcs": "FedCS", "oort": "Oort",
        "pedpc": "PEDPC", "greedy": "Greedy", "independent": "Independent",
        "static_weighted_sum": "Static weighted-sum", "tradefl_fixed": "TradeFL fixed prices",
        "tradefl_dynamic": "TradeFL",
    }
    rows["scheduler"] = rows["scheduler_policy"].map(labels).fillna(rows["scheduler_policy"])
    numeric = {
        "overall_task_utility": "test_utility", "validation_task_utility": "validation_utility",
        "mean_round_latency_seconds": "mean_round_latency_seconds",
        "p95_round_latency_seconds": "p95_round_latency_seconds",
        "peak_client_memory_bytes": "peak_memory_bytes", "rounds_to_target": "rounds_completed",
        "communication_cost_bytes": "communication_to_target_bytes",
        "energy_consumption_joules": "energy_to_target_joules",
        "attained_time_to_target_seconds": "attained_time_to_target_seconds",
        "censored_observation_horizon_seconds": "censored_observation_horizon_seconds",
        "scheduling_overhead_microseconds": "scheduling_overhead_microseconds",
    }
    records = []
    for (scheduler, seed), group in rows.groupby(["scheduler", "seed"], dropna=False):
        record = {"scheduler": scheduler, "seed": seed, "result_source": "online_training"}
        for metric, source in numeric.items():
            values = pd.to_numeric(group[source], errors="coerce").dropna() if source in group else pd.Series(dtype=float)
            record[metric] = values.mean() if not values.empty else float("nan")
        latency_limit = constraints.get("maximum_round_latency_seconds")
        memory_limit = constraints.get("memory_capacity_bytes")
        mean_latency = pd.to_numeric(group.get("mean_round_latency_seconds"), errors="coerce")
        memory = pd.to_numeric(group.get("peak_memory_bytes"), errors="coerce")
        record["deadline_slo_violation_rate"] = (mean_latency > float(latency_limit)).mean() if latency_limit is not None else float("nan")
        record["memory_resource_violation_rate"] = (memory > float(memory_limit)).mean() if memory_limit is not None else float("nan")
        reached = group.get("target_reached", pd.Series(False, index=group.index)).astype(str).str.lower().isin(["true", "1"])
        record["target_attainment_rate"] = reached.mean()
        record["time_to_target_censoring_rate"] = 1.0 - reached.mean()
        feasible = group.get("feasible", pd.Series(float("nan"), index=group.index))
        feasible = feasible.astype(str).str.lower().map({"true": 1.0, "false": 0.0})
        record["feasible_selection_rate"] = feasible.mean()
        record["test_utility_regret_to_oracle"] = float("nan")
        records.append(record)
    per_seed = pd.DataFrame(records)
    aggregates = []
    for scheduler, group in per_seed.groupby("scheduler", dropna=False):
        result = {"scheduler": scheduler, "result_source": "online_training", "seed_count": group["seed"].nunique()}
        for metric in [c for c in per_seed if c not in {"scheduler", "seed", "result_source"}]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            mean = values.mean() if not values.empty else float("nan")
            critical = {2: 12.706, 3: 4.303}.get(len(values), 1.96)
            half = critical * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            result.update({metric: mean, f"{metric}_ci95_low": mean - half, f"{metric}_ci95_high": mean + half,
                           f"{metric}_n": len(values), f"{metric}_ci95_method": "descriptive Student-t" if len(values) > 1 else "not estimable"})
        aggregates.append(result)
    definitions = pd.DataFrame([{
        "scheduler": label, "result_source": "online_training",
        "definition": "Executed as a separate federated training treatment; see scheduler_policy in raw_metrics.csv.",
    } for label in labels.values() if label in set(per_seed["scheduler"])])
    return pd.DataFrame(aggregates), per_seed, definitions


def _scheduler_bar_plot(
    outcomes: pd.DataFrame, per_seed: pd.DataFrame, metric: str, title: str, ylabel: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    values = pd.to_numeric(
        outcomes.get(metric, pd.Series(float("nan"), index=outcomes.index)),
        errors="coerce",
    )
    if values.notna().sum() == 0:
        ax.text(
            0.5, 0.5,
            "No feasible selections are available yet.\n"
            "This live metric will populate when a feasible plan completes.",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=16, fontweight="bold",
        )
        ax.set_title(f"Scheduler: {title}")
        ax.set_ylabel(ylabel)
        ax.set_xticks([])
        fig.tight_layout()
        fig.savefig(path, format="pdf")
        plt.close(fig)
        return
    bars = ax.bar(outcomes["scheduler"], values.fillna(0.0), color="tab:purple")
    low = pd.to_numeric(outcomes.get(f"{metric}_ci95_low"), errors="coerce")
    high = pd.to_numeric(outcomes.get(f"{metric}_ci95_high"), errors="coerce")
    if low is not None and high is not None:
        finite = values.notna() & low.notna() & high.notna()
        if finite.any():
            positions = [index for index, keep in enumerate(finite) if keep]
            central = values.loc[finite]
            errors = [central - low.loc[finite], high.loc[finite] - central]
            ax.errorbar(positions, central, yerr=errors, fmt="none", color="black", capsize=5, zorder=4)
    scheduler_positions = {scheduler: index for index, scheduler in enumerate(outcomes["scheduler"])}
    for _, point in per_seed.iterrows():
        value = pd.to_numeric(point.get(metric), errors="coerce")
        if pd.notna(value) and np.isfinite(float(value)):
            # Use the categorical scheduler label consistently with ax.bar;
            # mixing numeric offsets into a categorical axis fails for masked
            # values on recent Matplotlib releases.
            ax.scatter(str(point["scheduler"]), float(value), color="white", edgecolor="black", s=45, zorder=5)
    finite_values = values.dropna()
    label_offset = max(float(finite_values.max()) * 0.015, 0.01) if not finite_values.empty else 0.01
    for bar, value, missing in zip(bars, values, values.isna()):
        if missing:
            bar.set_hatch("//")
            ax.annotate(
                "N/A", (bar.get_x() + bar.get_width() / 2, 0),
                ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE,
            )
        else:
            if value == 0:
                ax.scatter([bar.get_x() + bar.get_width() / 2], [0], marker="_", s=180, color="black", zorder=3)
            ax.annotate(
                f"{value:.3g}",
                (bar.get_x() + bar.get_width() / 2, max(float(value), 0) + label_offset),
                ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE,
            )
    if metric.endswith("_rate"):
        ax.set_ylim(0, 1.08)
    ax.set_title(f"Scheduler: {title}")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def _scheduler_dashboard(outcomes: pd.DataFrame, per_seed: pd.DataFrame, path: Path) -> None:
    """Render a compact comparison of all requested scheduler policies."""

    metrics = [
        ("overall_task_utility", "Task utility"),
        ("target_attainment_rate", "Target attainment"),
        ("deadline_slo_violation_rate", "Deadline/SLO violations"),
        ("memory_resource_violation_rate", "Memory violations"),
        ("attained_time_to_target_seconds", "Attained time to target (s)"),
        ("scheduling_overhead_microseconds", "Scheduling overhead (µs)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    positions = {scheduler: index for index, scheduler in enumerate(outcomes["scheduler"])}
    for ax, (metric, title) in zip(axes.flat, metrics):
        values = pd.to_numeric(
            outcomes.get(metric, pd.Series(float("nan"), index=outcomes.index)),
            errors="coerce",
        )
        ax.bar(outcomes["scheduler"], values.fillna(0), color="tab:purple", alpha=0.85)
        for _, point in per_seed.iterrows():
            value = pd.to_numeric(point.get(metric), errors="coerce")
            if pd.notna(value):
                ax.scatter(positions[point["scheduler"]], value, color="white", edgecolor="black", s=32, zorder=3)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35, labelsize=9)
        if metric.endswith("_rate"):
            ax.set_ylim(0, 1.05)
    fig.suptitle("Scheduler comparison across common actions and constraints")
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def _scheduler_action_plot(decisions: pd.DataFrame, path: Path) -> None:
    """Show the actual action-selection frequency for each scheduler."""

    table = pd.crosstab(decisions["scheduler"], decisions["selected_action"])
    fig, ax = plt.subplots(figsize=(max(14, 0.6 * len(table.columns) + 8), 8))
    image = ax.imshow(table.to_numpy(), aspect="auto", cmap="Purples")
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xlabel("Selected action / plan")
    ax.set_ylabel("Scheduler")
    ax.set_title("Scheduler action-selection frequency")
    for row in range(len(table.index)):
        for column in range(len(table.columns)):
            ax.text(column, row, str(table.iat[row, column]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, label="Selection count")
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def _shadow_price_trajectory_plot(price_trace: pd.DataFrame, path: Path) -> None:
    """Plot dynamic TradeFL prices by seed, task, and constrained resource."""

    fig, ax = plt.subplots(figsize=(14, 8))
    if price_trace.empty:
        _no_data(ax, "No dynamic shadow-price trace available")
    else:
        for (seed, resource), group in price_trace.groupby(["seed", "resource"], dropna=False):
            ordered = group.sort_values("task_index")
            ax.plot(
                ordered["task_index"], ordered["price_after"], marker="o",
                label=f"seed={seed}, {resource}",
            )
        ax.set_xlabel("Workload task index")
        ax.set_ylabel("Shadow price after reconciliation")
        ax.legend(fontsize=9, ncol=2)
    ax.set_title("Dynamic TradeFL shadow-price trajectories")
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)




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
    fig, ax = plt.subplots(figsize=(18, max(5.0, 0.72 * len(table) + 2.5)))
    ax.axis("off")
    ax.set_title("Raw mean values by model plan (N/A = not measured)", pad=16)
    ax.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="center").auto_set_font_size(False)
    fig.axes[0].tables[0].set_fontsize(10)
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
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    for ax, (metric, title, scale) in zip(axes.flat, specs):
        values = pd.to_numeric(summary.get(metric, pd.Series(float("nan"), index=summary.index)), errors="coerce") / scale
        missing = values.isna()
        bars = ax.bar(summary["plan_id"].astype(str), values.fillna(0), color=["lightgray" if value else "tab:blue" for value in missing])
        for bar, is_missing in zip(bars, missing):
            if is_missing:
                bar.set_hatch("//")
                ax.annotate(
                    "N/A", (bar.get_x() + bar.get_width() / 2, 0),
                    ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE,
                )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35, labelsize=11)
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
    fig, ax = plt.subplots(figsize=(13, 9))
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
        ax.annotate(
            plan_id,
            (x_value / x_scale, y_value / y_scale),
            fontsize=PARETO_ANNOTATION_FONT_SIZE,
            fontweight="bold",
            xytext=(7, 7),
            textcoords="offset points",
        )
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
        ax.text(
            0.01, 0.01, "N/A: " + ", ".join(missing),
            transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom",
        )
    ax.set_title(f"Pareto view: {tradeoff.replace('_', ' ')}")
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return rows


def _bar_plot(summary: pd.DataFrame, metric: str, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7.5))
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
                ax.annotate(
                    label, (bar.get_x() + bar.get_width() / 2, 0),
                    ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE,
                )
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

    fig, ax = plt.subplots(figsize=(14, 8))
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
            ax.annotate(label, (position, 0), ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE)
    ax.set_xticks(positions, summary["plan_id"].astype(str), rotation=35, ha="right")
    ax.set_ylabel("Utility")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def _scatter_plot(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    if x in df.columns and y in df.columns and not df.empty:
        ax.scatter(pd.to_numeric(df[x], errors="coerce"), pd.to_numeric(df[y], errors="coerce"))
        if "plan_id" in df.columns:
            for _, row in df.iterrows():
                x_value = pd.to_numeric(pd.Series([row[x]]), errors="coerce").iloc[0]
                y_value = pd.to_numeric(pd.Series([row[y]]), errors="coerce").iloc[0]
                if pd.notna(x_value) and pd.notna(y_value):
                    ax.annotate(
                        str(row["plan_id"]), (x_value, y_value),
                        fontsize=ANNOTATION_FONT_SIZE, alpha=0.85,
                    )
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
