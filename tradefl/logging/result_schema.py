"""Machine-readable TradeFL output schemas."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
@dataclass
class PlanSummary:
    plan_id:str; seed:int; peak_memory_bytes:float; compute_to_target_seconds:float; communication_to_target_bytes:int; energy_to_target_joules:float|None; latency_to_target_seconds:float; privacy_risk:float; validation_utility:float; accuracy_loss:float; rounds_to_target:int|None; target_reached:bool; feasible:bool; violations:str; normalized_memory:float|None=None; normalized_compute:float|None=None; normalized_communication:float|None=None; normalized_energy:float|None=None; normalized_latency:float|None=None; normalized_privacy:float|None=None; normalized_accuracy:float|None=None; tradefl_score:float|None=None; metadata:dict[str,Any]=field(default_factory=dict)
    def to_dict(self)->dict[str,Any]: return asdict(self)
