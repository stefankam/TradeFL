"""Hard-constraint feasibility filtering before scoring."""
from __future__ import annotations
from dataclasses import dataclass
@dataclass
class FeasibilityResult:
    feasible: bool; violations: list[str]
@dataclass
class Constraints:
    memory_capacity_bytes: float; maximum_round_latency_seconds: float; maximum_time_to_target_seconds: float; minimum_validation_utility: float; maximum_privacy_risk: float; hospital_edge_required: bool=False; public_cloud_allowed: bool=True; require_target_reached: bool=True; edge_available: bool=True; uses_public_cloud: bool=False
def check_feasibility(metrics: dict[str,object], constraints: Constraints) -> FeasibilityResult:
    v=[]
    if float(metrics.get('peak_memory_bytes',0)) > constraints.memory_capacity_bytes: v.append('memory_capacity_exceeded')
    if float(metrics.get('mean_round_latency_seconds', metrics.get('latency_to_target_seconds',0))) > constraints.maximum_round_latency_seconds: v.append('round_latency_exceeded')
    if float(metrics.get('latency_to_target_seconds',0)) > constraints.maximum_time_to_target_seconds: v.append('time_to_target_exceeded')
    if float(metrics.get('validation_utility',0)) < constraints.minimum_validation_utility: v.append('minimum_validation_utility_not_met')
    if float(metrics.get('privacy_risk',0)) > constraints.maximum_privacy_risk: v.append('privacy_risk_exceeded')
    if constraints.require_target_reached and not bool(metrics.get('target_reached',False)): v.append('target_quality_not_reached')
    if constraints.hospital_edge_required and not constraints.edge_available: v.append('required_edge_infrastructure_unavailable')
    if not constraints.public_cloud_allowed and constraints.uses_public_cloud: v.append('public_cloud_prohibited')
    return FeasibilityResult(not v, v)
