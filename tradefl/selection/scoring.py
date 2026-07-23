"""TradeFL weighted objective calculation."""
from __future__ import annotations
from dataclasses import dataclass, asdict
METRICS = ("memory","compute","communication","energy","latency","privacy","accuracy")
@dataclass
class TradeFLWeights:
    memory: float; compute: float; communication: float; energy: float; latency: float; privacy: float; accuracy: float
    def validate(self) -> None:
        vals=list(asdict(self).values())
        if any(v < 0 for v in vals): raise ValueError('All weights must be non-negative.')
        if abs(sum(vals)-1.0)>1e-6: raise ValueError('TradeFL weights must sum to 1.')
    def asdict(self) -> dict[str,float]: return asdict(self)
def renormalize_weights(weights: TradeFLWeights, enabled: dict[str,bool]) -> TradeFLWeights:
    vals={k:v for k,v in weights.asdict().items() if enabled.get(k, True)}; total=sum(vals.values())
    if total <= 0: raise ValueError('At least one enabled weight must be positive.')
    allv={k:(vals.get(k,0.0)/total if enabled.get(k, True) else 0.0) for k in METRICS}
    out=TradeFLWeights(**allv); out.validate(); return out
def calculate_score(normalized_costs: dict[str,float|None], weights: TradeFLWeights) -> float:
    weights.validate(); total=0.0
    for k,w in weights.asdict().items():
        if w and normalized_costs.get(k) is None: raise ValueError(f'Missing enabled normalized metric: {k}')
        total += w * float(normalized_costs.get(k) or 0.0)
    return total
