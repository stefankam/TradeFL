"""Latency aggregation helpers."""
import statistics
def summarize_latencies(values:list[float])->dict[str,float]:
    if not values: return {'mean_round_latency_seconds':0,'median_round_latency_seconds':0,'p95_round_latency_seconds':0}
    ordered=sorted(values); idx=min(len(ordered)-1, int(round(0.95*(len(ordered)-1))))
    return {'mean_round_latency_seconds':statistics.mean(values),'median_round_latency_seconds':statistics.median(values),'p95_round_latency_seconds':ordered[idx]}
