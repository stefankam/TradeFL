"""Offline scheduler policies and replay engine over measured plan candidates."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import pandas as pd


SCHEDULER_DEFINITIONS = [
    {"scheduler": "TradeFL", "definition": "Online Lagrangian replay over every workload task; prices reset per seed and update after each task."},
    {"scheduler": "Independent/per-resource", "definition": "Minimize the worst independently normalized resource cost."},
    {"scheduler": "Static weighted-sum", "definition": "Minimize an equal, fixed weighted sum without feasibility filtering."},
    {"scheduler": "Greedy", "definition": "Maximize current validation utility without look-ahead."},
    {"scheduler": "TradeFL fixed prices", "definition": "Same feasible net-gain objective and 1/capacity warm start as TradeFL, but prices remain fixed for every seed."},
    {"scheduler": "Oracle (enumeration)", "definition": "Hindsight enumeration over the common feasible action set: target attainment first, then test utility, then latency."},
]


@dataclass(frozen=True)
class SchedulerReplayResult:
    decisions: pd.DataFrame
    outcomes: pd.DataFrame
    per_seed_outcomes: pd.DataFrame
    price_trace: pd.DataFrame
    definitions: pd.DataFrame


def replay_schedulers(frame: pd.DataFrame, constraints: dict) -> SchedulerReplayResult | None:
    """Replay a multi-task workload independently within every seed."""

    required = {"plan_id", "validation_utility", "peak_memory_bytes", "latency_to_target_seconds"}
    if not required.issubset(frame.columns) or frame.empty:
        return None
    candidates = _prepare_candidates(frame)
    cost_columns = _cost_columns(candidates)
    price_rows, decisions = [], []
    capacities = shadow_resource_capacities(constraints)
    fixed_prices = {resource: 1.0 / capacity for resource, capacity in capacities.items()}
    shadow_prices = dict(fixed_prices)
    trajectory_index = 0
    for seed, seed_group in candidates.groupby("seed", dropna=False):
        for task_id, group in seed_group.groupby("workload_task_id", dropna=False):
            normalized = normalize_costs(group, cost_columns)
            feasible = group.loc[group.apply(lambda row: is_feasible(row, constraints), axis=1)]
            oracle_pool = feasible if not feasible.empty else group
            ratios = shadow_resource_ratios(group, constraints)
            policies = (
                ("TradeFL", lambda: select_tradefl(group, normalized, ratios, shadow_prices)),
                ("Independent/per-resource", lambda: select_independent(group, normalized, cost_columns)),
                ("Static weighted-sum", lambda: select_static_weighted_sum(group, normalized, cost_columns)),
                ("Greedy", lambda: select_greedy(group)),
                ("TradeFL fixed prices", lambda: select_fixed_price(group, normalized, ratios, fixed_prices)),
                ("Oracle (enumeration)", lambda: select_oracle(oracle_pool)),
            )
            selected_tradefl = None
            for scheduler, policy in policies:
                started = time.perf_counter_ns()
                selected = policy()
                overhead_us = (time.perf_counter_ns() - started) / 1000.0
                if scheduler == "TradeFL":
                    selected_tradefl = selected
                decisions.append(decision_row(scheduler, seed, task_id, selected, overhead_us, constraints))
            assert selected_tradefl is not None
            selected_ratios = ratios.loc[selected_tradefl.name]
            for resource, old_price in list(shadow_prices.items()):
                selected_ratio = float(selected_ratios.get(resource, 0.0))
                capacity = capacities[resource]
                selected_demand = selected_ratio * capacity
                step_size = 0.25 / (capacity * capacity * math.sqrt(trajectory_index + 1))
                violation = selected_demand - capacity
                new_price = min(10.0 / capacity, max(0.0, old_price + step_size * violation))
                price_rows.append({
                    "task_index": trajectory_index, "workload_task_id": task_id, "seed": seed,
                    "resource": resource, "price_before": old_price,
                    "selected_ratio": selected_ratio, "selected_demand": selected_demand,
                    "capacity": capacity, "subgradient": violation,
                    "step_size": step_size, "price_after": new_price,
                })
                shadow_prices[resource] = new_price
            trajectory_index += 1
    decision_frame = pd.DataFrame(decisions)
    per_seed = decision_frame.groupby(["scheduler", "seed"], as_index=False, dropna=False).agg(
        overall_task_utility=("task_utility", "mean"),
        deadline_slo_violation_rate=("deadline_slo_violation", "mean"),
        memory_resource_violation_rate=("memory_resource_violation", "mean"),
        communication_cost_bytes=("communication_cost_bytes", "mean"),
        energy_consumption_joules=("energy_consumption_joules", "mean"),
        attained_time_to_target_seconds=("attained_time_to_target_seconds", "mean"),
        censored_observation_horizon_seconds=("censored_observation_horizon_seconds", "mean"),
        time_to_target_censoring_rate=("time_to_target_censored", "mean"),
        scheduling_overhead_microseconds=("scheduling_overhead_microseconds", "mean"),
        feasible_selection_rate=("selected_feasible", "mean"),
    )
    metrics = [column for column in per_seed if column not in {"scheduler", "seed"}]
    rows = []
    for scheduler, group in per_seed.groupby("scheduler", dropna=False):
        result = {"scheduler": scheduler, "seed_count": group["seed"].nunique()}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            mean = values.mean() if not values.empty else float("nan")
            half = _critical_95(len(values)) * values.std(ddof=1) / (len(values) ** 0.5) if len(values) > 1 else 0.0
            result.update({metric: mean, f"{metric}_ci95_low": mean - half, f"{metric}_ci95_high": mean + half})
        rows.append(result)
    return SchedulerReplayResult(
        decision_frame, pd.DataFrame(rows), per_seed, pd.DataFrame(price_rows), pd.DataFrame(SCHEDULER_DEFINITIONS)
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
    """Hindsight enumeration: attain target first, then utility, then latency."""
    target = eligible.loc[eligible.apply(_target_reached, axis=1)]
    pool = target if not target.empty else eligible
    return pool.sort_values(["test_utility", "latency_to_target_seconds"], ascending=[False, True]).iloc[0]


def _prepare_candidates(frame):
    candidates = frame.copy()
    if "seed" not in candidates:
        candidates["seed"] = 0
    if "test_utility" not in candidates:
        candidates["test_utility"] = candidates["validation_utility"]
    if "predicted_utility" not in candidates:
        # Replay-time validation utility is the prediction available to an
        # online policy; test utility remains reserved for outcome reporting.
        candidates["predicted_utility"] = candidates["validation_utility"]
    if "accuracy_loss" not in candidates:
        candidates["accuracy_loss"] = 1.0 - pd.to_numeric(candidates["validation_utility"], errors="coerce")
    if "workload_task_id" not in candidates:
        # Aggregate run summaries have no explicit task dimension. Treat each
        # independently measured seed as one task instead of copying every seed
        # into several synthetic epochs: seed 42 -> task_0, seed 43 -> task_1,
        # and so on in sorted seed order.
        task_indices = candidates.groupby("seed", dropna=False, sort=True).ngroup()
        candidates["workload_task_id"] = task_indices.map(lambda index: f"task_{index}")
        candidates["workload_task_source"] = "inferred_from_seed"
    elif "workload_task_source" not in candidates:
        candidates["workload_task_source"] = "observed"
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
    return list(shadow_resource_capacities(constraints))


def shadow_resource_capacities(constraints):
    mapping = {
        "memory": "memory_capacity_bytes",
        "round_latency": "maximum_round_latency_seconds",
        "total_latency": "maximum_time_to_target_seconds",
        "communication": "maximum_communication_bytes",
        "energy": "maximum_energy_joules",
    }
    return {
        resource: float(constraints[key])
        for resource, key in mapping.items()
        if key in constraints and float(constraints[key]) > 0
    }

def shadow_resource_ratios(group, constraints):
    ratios = pd.DataFrame(index=group.index)
    ratios.attrs["capacities"] = shadow_resource_capacities(constraints)
    if "memory_capacity_bytes" in constraints:
        ratios["memory"] = pd.to_numeric(group["peak_memory_bytes"], errors="coerce") / float(constraints["memory_capacity_bytes"])
    if "maximum_round_latency_seconds" in constraints:
        source = group["mean_round_latency_seconds"] if "mean_round_latency_seconds" in group else group["latency_to_target_seconds"]
        ratios["round_latency"] = pd.to_numeric(source, errors="coerce") / float(constraints["maximum_round_latency_seconds"])
    if "maximum_time_to_target_seconds" in constraints:
        ratios["total_latency"] = pd.to_numeric(group["latency_to_target_seconds"], errors="coerce") / float(constraints["maximum_time_to_target_seconds"])
    if "maximum_communication_bytes" in constraints and "communication_to_target_bytes" in group:
        ratios["communication"] = pd.to_numeric(group["communication_to_target_bytes"], errors="coerce") / float(constraints["maximum_communication_bytes"])
    if "maximum_energy_joules" in constraints and "energy_to_target_joules" in group:
        ratios["energy"] = pd.to_numeric(group["energy_to_target_joules"], errors="coerce") / float(constraints["maximum_energy_joules"])
    return ratios.fillna(0.0)


def _minimum_shadow_price_score(group, normalized, ratios, prices):
    utility = pd.to_numeric(group["predicted_utility"], errors="coerce")
    net_gain = utility.fillna(float("-inf"))
    capacities = ratios.attrs.get("capacities", {})
    for resource, price in prices.items():
        net_gain = net_gain - price * ratios.loc[group.index, resource] * capacities.get(resource, 1.0)
    return group.loc[net_gain.idxmax()]


def is_feasible(row, constraints):
    memory = _number(row.get("peak_memory_bytes"), 0)
    round_latency = _number(row.get("mean_round_latency_seconds", row.get("latency_to_target_seconds")), 0)
    total_latency = _number(row.get("latency_to_target_seconds"), 0)
    utility = _number(row.get("validation_utility"), 0)
    privacy = _number(row.get("privacy_risk"), 0)
    communication = _number(row.get("communication_to_target_bytes"), 0)
    energy = _number(row.get("energy_to_target_joules"), 0)
    return (
        bool(row.get("policy_compatible", True))
        and
        memory <= float(constraints.get("memory_capacity_bytes", float("inf")))
        and round_latency <= float(constraints.get("maximum_round_latency_seconds", float("inf")))
        and total_latency <= float(constraints.get("maximum_time_to_target_seconds", float("inf")))
        and utility >= float(constraints.get("minimum_validation_utility", 0))
        and privacy <= float(constraints.get("maximum_privacy_risk", float("inf")))
        and communication <= float(constraints.get("maximum_communication_bytes", float("inf")))
        and energy <= float(constraints.get("maximum_energy_joules", float("inf")))
    )


def decision_row(scheduler, seed, task_id, row, overhead_us, constraints):
    memory = _number(row.get("peak_memory_bytes"), 0)
    round_latency = _number(row.get("mean_round_latency_seconds", row.get("latency_to_target_seconds")), 0)
    total_latency = _number(row.get("latency_to_target_seconds"), 0)
    target_reached = _target_reached(row)
    return {
        "scheduler": scheduler, "seed": seed, "workload_task_id": task_id,
        "workload_task_source": row.get("workload_task_source", "observed"),
        "decision_status": "selected" if is_feasible(row, constraints) else "selected_infeasible",
        "selected_feasible": float(is_feasible(row, constraints)),
        "selected_plan": row["plan_id"], "selected_action": row["plan_id"],
        "task_utility": float(pd.to_numeric(row.get("predicted_utility", row["validation_utility"]), errors="coerce")),
        "deadline_slo_violation": float(
            round_latency > float(constraints.get("maximum_round_latency_seconds", float("inf")))
            or total_latency > float(constraints.get("maximum_time_to_target_seconds", float("inf")))
        ),
        "memory_resource_violation": float(memory > float(constraints.get("memory_capacity_bytes", float("inf")))),
        "communication_cost_bytes": pd.to_numeric(row.get("communication_to_target_bytes"), errors="coerce"),
        "energy_consumption_joules": pd.to_numeric(row.get("energy_to_target_joules"), errors="coerce"),
        "attained_time_to_target_seconds": total_latency if target_reached else float("nan"),
        "censored_observation_horizon_seconds": total_latency if not target_reached else float("nan"),
        "time_to_target_censored": float(not target_reached),
        "scheduling_overhead_microseconds": overhead_us,
        "target_reached": float(target_reached),
        "target_quality": _number(row.get("target_quality")),
        "constraint_provenance": row.get("constraint_provenance", "{}"),
    }


def no_feasible_decision_row(scheduler, seed, task_id, overhead_us, constraints):
    """Represent an empty constrained action set without fabricating an outcome."""

    return {
        "scheduler": scheduler, "seed": seed, "workload_task_id": task_id,
        "decision_status": "no_feasible_plan",
        "selected_plan": "NO_FEASIBLE_PLAN", "selected_action": "NO_FEASIBLE_PLAN",
        "task_utility": float("nan"),
        "deadline_slo_violation": float("nan"),
        "memory_resource_violation": float("nan"),
        "communication_cost_bytes": float("nan"),
        "energy_consumption_joules": float("nan"),
        "attained_time_to_target_seconds": float("nan"),
        "censored_observation_horizon_seconds": float("nan"),
        "time_to_target_censored": 1.0,
        "scheduling_overhead_microseconds": overhead_us,
        "target_reached": 0.0,
        "target_quality": float("nan"),
        "constraint_provenance": str(constraints),
    }




def _target_reached(row) -> bool:
    recorded = str(row.get("target_reached", "")).strip().lower()
    if recorded in {"true", "1", "yes"}:
        return True
    if recorded in {"false", "0", "no"}:
        return False
    target = _number(row.get("target_quality"))
    utility = _number(row.get("validation_utility"))
    return pd.notna(target) and pd.notna(utility) and utility >= target


def _number(value, default=float("nan")):
    converted = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(converted) else float(converted)


def _critical_95(sample_count: int) -> float:
    """Two-sided Student-t 95% critical value, with a normal large-n tail."""

    values = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    return values.get(sample_count, 1.96 if sample_count > 30 else 2.042)
