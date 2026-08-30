import pandas as pd

from tradefl.scheduling.replay import (
    replay_schedulers,
    select_fixed_price,
    select_greedy,
    select_independent,
    select_oracle,
    select_static_weighted_sum,
    select_tradefl,
)


def candidates():
    return pd.DataFrame(
        [
            {
                "plan_id": "cheap", "seed": 42, "validation_utility": 0.7, "test_utility": 0.68,
                "accuracy_loss": 0.2, "peak_memory_bytes": 4, "compute_to_target_seconds": 2,
                "communication_to_target_bytes": 10, "energy_to_target_joules": 5,
                "latency_to_target_seconds": 3, "mean_round_latency_seconds": 3,
                "privacy_risk": 0.2, "target_reached": True,
            },
            {
                "plan_id": "accurate", "seed": 42, "validation_utility": 0.9, "test_utility": 0.88,
                "accuracy_loss": 0.0, "peak_memory_bytes": 12, "compute_to_target_seconds": 8,
                "communication_to_target_bytes": 30, "energy_to_target_joules": 15,
                "latency_to_target_seconds": 9, "mean_round_latency_seconds": 9,
                "privacy_risk": 0.2, "target_reached": True,
            },
        ]
    )


def test_replay_module_owns_and_runs_all_scheduling_methods():
    constraints = {
        "memory_capacity_bytes": 8,
        "maximum_round_latency_seconds": 6,
        "maximum_time_to_target_seconds": 20,
        "minimum_validation_utility": 0.4,
        "maximum_privacy_risk": 0.5,
    }

    replay = replay_schedulers(candidates(), constraints)

    assert replay is not None
    assert set(replay.outcomes["scheduler"]) == {
        "TradeFL", "Independent/per-resource", "Static weighted-sum", "Greedy",
        "TradeFL fixed prices", "Oracle (enumeration)",
    }
    assert set(replay.definitions["scheduler"]) == set(replay.outcomes["scheduler"])
    assert not replay.price_trace.empty


def test_each_policy_has_a_public_scheduler_function():
    assert all(callable(policy) for policy in (
        select_tradefl,
        select_fixed_price,
        select_independent,
        select_static_weighted_sum,
        select_greedy,
        select_oracle,
    ))
