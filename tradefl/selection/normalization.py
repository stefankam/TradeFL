"""Budget-based normalization for TradeFL costs."""
from __future__ import annotations
from dataclasses import dataclass, asdict
@dataclass
class Budgets:
    memory_bytes: float; compute_seconds: float; communication_bytes: float; energy_joules: float|None; latency_seconds: float; privacy_risk: float; accuracy_loss: float
    def validate(self) -> None:
        for k,v in asdict(self).items():
            if v is not None and v <= 0: raise ValueError(f'Budget {k} must be positive when enabled.')
def normalize_metrics(metrics: dict[str,float|None], budgets: Budgets, enabled: dict[str,bool]|None=None) -> dict[str,float|None]:
    budgets.validate(); enabled=enabled or {}
    mapping={'memory':('peak_memory_bytes','memory_bytes'),'compute':('compute_to_target_seconds','compute_seconds'),'communication':('communication_to_target_bytes','communication_bytes'),'energy':('energy_to_target_joules','energy_joules'),'latency':('latency_to_target_seconds','latency_seconds'),'privacy':('privacy_risk','privacy_risk'),'accuracy':('accuracy_loss','accuracy_loss')}
    out={}
    bd=asdict(budgets)
    for name,(mk,bk) in mapping.items():
        if enabled.get(name, True) is False: out[name]=None; continue
        value=metrics.get(mk); budget=bd[bk]
        if value is None or budget is None: raise ValueError(f'Missing enabled metric or budget: {name}')
        out[name]=float(value)/float(budget)
    return out
