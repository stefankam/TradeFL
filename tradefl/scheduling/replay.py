"""Offline scheduler policies and replay engine over measured plan candidates."""
from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd


SCHEDULER_DEFINITIONS = [
    {"scheduler": "TradeFL", "definition": "Lagrangian replay with projected subgradient shadow-price updates after each seed: lambda=max(0, lambda+0.25*(usage/capacity-1))."},
    {"scheduler": "Independent/per-resource", "definition": "Minimize the worst independently normalized resource cost."},
    {"scheduler": "Static weighted-sum", "definition": "Minimize an equal, fixed weighted sum without feasibility filtering."},
    {"scheduler": "Greedy", "definition": "Maximize current validation utility without look-ahead."},
    {"scheduler": "TradeFL fixed prices", "definition": "Same Lagrangian score and initial unit prices as TradeFL, but prices remain fixed at 1.0 for every seed."},
    {"scheduler": "Oracle (enumeration)", "definition": "Hindsight maximum test utility among feasible observed candidates; not an online policy."},
]


@dataclass(frozen=True)
class SchedulerReplayResult:
    decisions: pd.DataFrame
    outcomes: pd.DataFrame
    price_trace: pd.DataFrame
    definitions: pd.DataFrame


def replay_schedulers(frame: pd.DataFrame, constraints: dict) -> SchedulerReplayResult | None:
    """Run every scheduling policy over existing measurements without training."""

    required = {"plan_id", "validation_utility", "peak_memory_bytes", "latency_to_target_seconds"}
    if not required.issubset(frame.columns) or frame.empty:
        return None
    candidates = _prepare_candidates(frame)
    cost_columns = _cost_columns(candidates)
    fixed_prices = {resource: 1.0 for resource in shadow_resource_columns(constraints)}
    shadow_prices = dict(fixed_prices)
    price_rows: list[dict] = []
    decisions: list[dict] = []
    for epoch, (seed, group) in enumerate(candidates.groupby("seed", dropna=False)):
        normalized = normalize_costs(group, cost_columns)
        feasible = group.loc[group.apply(lambda row: is_feasible(row, constraints), axis=1)]
        eligible = feasible if not feasible.empty else group
        ratios = shadow_resource_ratios(group, constraints)
        policies = (
            ("TradeFL", lambda: select_tradefl(group, normalized, ratios, shadow_prices)),
            ("Independent/per-resource", lambda: select_independent(group, normalized, cost_columns)),
            ("Static weighted-sum", lambda: select_static_weighted_sum(group, normalized, cost_columns)),
            ("Greedy", lambda: select_greedy(group)),
            ("TradeFL fixed prices", lambda: select_fixed_price(group, normalized, ratios, fixed_prices)),
            ("Oracle (enumeration)", lambda: select_oracle(eligible)),
        )
        selected_tradefl = None
        for scheduler, policy in policies:
            started = time.perf_counter_ns()
            selected = policy()
            overhead_us = (time.perf_counter_ns() - started) / 1000.0
            if scheduler == "TradeFL":
                selected_tradefl = selected
            decisions.append(decision_row(scheduler, seed, selected, overhead_us, constraints))
        assert selected_tradefl is not None
        selected_ratios = ratios.loc[selected_tradefl.name]
        for resource, old_price in list(shadow_prices.items()):
            violation = float(selected_ratios.get(resource, 0.0)) - 1.0
            new_price = max(0.0, old_price + 0.25 * violation)
            price_rows.append({
                "epoch": epoch, "seed": seed, "resource": resource,
                "price_before": old_price, "selected_ratio": float(selected_ratios.get(resource, 0.0)),
                "subgradient": violation, "step_size": 0.25, "price_after": new_price,
            })
            shadow_prices[resource] = new_price
    decision_frame = pd.DataFrame(decisions)
    outcomes = decision_frame.groupby("scheduler", as_index=False).agg(
        overall_task_utility=("task_utility", "mean"),
        deadline_slo_violation_rate=("deadline_slo_violation", "mean"),
        memory_resource_violation_rate=("memory_resource_violation", "mean"),
        communication_cost_bytes=("communication_cost_bytes", "mean"),
        energy_consumption_joules=("energy_consumption_joules", "mean"),
        time_to_target_seconds=("time_to_target_seconds", "mean"),
        time_to_target_censoring_rate=("time_to_target_censored", "mean"),
        scheduling_overhead_microseconds=("scheduling_overhead_microseconds", "mean"),
        target_attainment_rate=("target_reached", "mean"),
    )
    return SchedulerReplayResult(
        decision_frame,
        outcomes,
        pd.DataFrame(price_rows),
        pd.DataFrame(SCHEDULER_DEFINITIONS),
    )


def select_tradefl(group, normalized, ratios, prices):
    """Select the candidate minimizing quality loss plus dynamic resource prices."""
    return _minimum_shadow_price_score(group, normalized, ratios, prices)


def select_fixed_price(group, normalized, ratios, prices):
    """Select with the TradeFL objective while leaving prices unchanged."""
    return _minimum_shadow_price_score(group, normalized, ratios, prices)


def select_independent(group, normalized, columns):
    """Minimize the worst independently normalized resource demand."""
    return group.loc[normalized.loc[group.index, columns].max(axis=1).idxmin()]


def select_static_weighted_sum(group, normalized, columns):
    """Minimize an equal fixed weighted sum of normalized costs."""
    scores = sum(normalized.loc[group.index, column] for column in columns)
    return group.loc[scores.idxmin()]


def select_greedy(group):
    """Select the candidate with maximum current validation utility."""
    return group.loc[pd.to_numeric(group["validation_utility"], errors="coerce").idxmax()]


def select_oracle(eligible):
    """Select hindsight maximum test utility among feasible observed candidates."""
    return eligible.sort_values(["test_utility", "latency_to_target_seconds"], ascending=[False, True]).iloc[0]


def _prepare_candidates(frame):
    candidates = frame.copy()
    if "seed" not in candidates:
        candidates["seed"] = 0
    if "test_utility" not in candidates:
        candidates["test_utility"] = candidates["validation_utility"]
    if "accuracy_loss" not in candidates:
        candidates["accuracy_loss"] = 1.0 - pd.to_numeric(candidates["validation_utility"], errors="coerce")
    return candidates


def _cost_columns(candidates):
    return [
        column for column in (
            "peak_memory_bytes", "compute_to_target_seconds", "communication_to_target_bytes",
            "energy_to_target_joules", "latency_to_target_seconds", "privacy_risk", "accuracy_loss",
        ) if column in candidates.columns and pd.to_numeric(candidates[column], errors="coerce").notna().any()
    ]


def normalize_costs(frame, columns):
    normalized = pd.DataFrame(index=frame.index)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        span = values.max() - values.min()
        normalized[column] = (values - values.min()) / span if pd.notna(span) and span > 0 else 0.0
        normalized[column] = normalized[column].fillna(0.0)
    return normalized


def shadow_resource_columns(constraints):
    mapping = {
        "memory": "memory_capacity_bytes", "round_latency": "maximum_round_latency_seconds",
        "total_latency": "maximum_time_to_target_seconds", "privacy": "maximum_privacy_risk",
        "quality_floor": "minimum_validation_utility",
    }
    return [resource for resource, key in mapping.items() if key in constraints and float(constraints[key]) > 0]


def shadow_resource_ratios(group, constraints):
    ratios = pd.DataFrame(index=group.index)
    if "memory_capacity_bytes" in constraints:
        ratios["memory"] = pd.to_numeric(group["peak_memory_bytes"], errors="coerce") / float(constraints["memory_capacity_bytes"])
    if "maximum_round_latency_seconds" in constraints:
        source = group["mean_round_latency_seconds"] if "mean_round_latency_seconds" in group else group["latency_to_target_seconds"]
        ratios["round_latency"] = pd.to_numeric(source, errors="coerce") / float(constraints["maximum_round_latency_seconds"])
    if "maximum_time_to_target_seconds" in constraints:
        ratios["total_latency"] = pd.to_numeric(group["latency_to_target_seconds"], errors="coerce") / float(constraints["maximum_time_to_target_seconds"])
    if "maximum_privacy_risk" in constraints and "privacy_risk" in group:
        ratios["privacy"] = pd.to_numeric(group["privacy_risk"], errors="coerce") / float(constraints["maximum_privacy_risk"])
    if "minimum_validation_utility" in constraints:
        floor = float(constraints["minimum_validation_utility"])
        ratios["quality_floor"] = (2 * floor - pd.to_numeric(group["validation_utility"], errors="coerce")) / floor
    return ratios.fillna(0.0)


def _minimum_shadow_price_score(group, normalized, ratios, prices):
    accuracy = normalized["accuracy_loss"] if "accuracy_loss" in normalized else pd.Series(0.0, index=group.index)
    lagrangian = accuracy.copy()
    for resource, price in prices.items():
        lagrangian = lagrangian + price * (ratios[resource] - 1.0)
    return group.loc[lagrangian.idxmin()]


def is_feasible(row, constraints):
    memory = _number(row.get("peak_memory_bytes"), 0)
    round_latency = _number(row.get("mean_round_latency_seconds", row.get("latency_to_target_seconds")), 0)
    total_latency = _number(row.get("latency_to_target_seconds"), 0)
    utility = _number(row.get("validation_utility"), 0)
    privacy = _number(row.get("privacy_risk"), 0)
    return (
        memory <= float(constraints.get("memory_capacity_bytes", float("inf")))
        and round_latency <= float(constraints.get("maximum_round_latency_seconds", float("inf")))
        and total_latency <= float(constraints.get("maximum_time_to_target_seconds", float("inf")))
        and utility >= float(constraints.get("minimum_validation_utility", 0))
        and privacy <= float(constraints.get("maximum_privacy_risk", float("inf")))
    )


def decision_row(scheduler, seed, row, overhead_us, constraints):
    memory = _number(row.get("peak_memory_bytes"), 0)
    round_latency = _number(row.get("mean_round_latency_seconds", row.get("latency_to_target_seconds")), 0)
    total_latency = _number(row.get("latency_to_target_seconds"), 0)
    target_reached = str(row.get("target_reached", False)).strip().lower() in {"true", "1", "yes"}
    return {
        "scheduler": scheduler, "seed": seed, "selected_plan": row["plan_id"],
        "task_utility": float(pd.to_numeric(row.get("test_utility", row["validation_utility"]), errors="coerce")),
        "deadline_slo_violation": float(
            round_latency > float(constraints.get("maximum_round_latency_seconds", float("inf")))
            or total_latency > float(constraints.get("maximum_time_to_target_seconds", float("inf")))
        ),
        "memory_resource_violation": float(memory > float(constraints.get("memory_capacity_bytes", float("inf")))),
        "communication_cost_bytes": pd.to_numeric(row.get("communication_to_target_bytes"), errors="coerce"),
        "energy_consumption_joules": pd.to_numeric(row.get("energy_to_target_joules"), errors="coerce"),
        "time_to_target_seconds": total_latency,
        "time_to_target_censored": float(not target_reached),
        "scheduling_overhead_microseconds": overhead_us,
        "target_reached": float(target_reached),
    }


def _number(value, default=float("nan")):
    converted = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(converted) else float(converted)
