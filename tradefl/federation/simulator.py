"""Federation runner for common fine-tuning plan interface."""
from __future__ import annotations

from tradefl.measurement.latency import summarize_latencies


def run_to_target(plan, target_quality: float, max_rounds: int) -> tuple[list[dict], dict]:
    """Run a plan until target validation utility or max_rounds is reached."""

    plan.setup()
    rounds = []
    reached = False
    rounds_to_target = None
    for round_index in range(max_rounds):
        row = plan.run_round(round_index)
        rounds.append(row)
        if row["validation_utility"] >= target_quality:
            reached = True
            rounds_to_target = round_index + 1
            break
    evals = plan.evaluate()
    plan.cleanup()
    used = rounds[: rounds_to_target or len(rounds)]
    if not used:
        raise ValueError("No rounds were executed; max_rounds must be positive.")
    summary = {
        **evals,
        "rounds_completed": len(used),
        "rounds_to_target": rounds_to_target,
        "target_reached": reached,
        "peak_memory_bytes": max(x["peak_memory_bytes"] for x in used),
        "compute_to_target_seconds": sum(x["compute_time_seconds"] for x in used),
        "communication_to_target_bytes": sum(x["bytes_uploaded"] + x["bytes_downloaded"] for x in used),
        "energy_to_target_joules": None,
        "latency_to_target_seconds": sum(x["latency_seconds"] for x in used),
        "privacy_risk": max(x["privacy_risk"] for x in used),
        "mean_round_latency_seconds": summarize_latencies([x["latency_seconds"] for x in used])["mean_round_latency_seconds"],
        "median_round_latency_seconds": summarize_latencies([x["latency_seconds"] for x in used])["median_round_latency_seconds"],
        "p95_round_latency_seconds": summarize_latencies([x["latency_seconds"] for x in used])["p95_round_latency_seconds"],
        "metadata": used[-1].get("metadata", {}),
    }
    return rounds, summary
