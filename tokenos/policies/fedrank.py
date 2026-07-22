from dataclasses import dataclass, field
from typing import Dict, List
from tokenos.device import EdgeDevice

@dataclass
class FedRankPolicy:
    """Rank clients using data value plus compute/communication/latency tokens."""
    weights: Dict[str, float] = field(default_factory=lambda: {"data": .4, "compute": .25, "communication": .25, "latency": .1})

    def rank(self, devices: List[EdgeDevice]) -> List[EdgeDevice]:
        return sorted(devices, key=self.score, reverse=True)

    def score(self, device: EdgeDevice) -> float:
        t = device.current_token_budget
        return (self.weights["data"] * device.data_value + self.weights["compute"] * t.get("compute", 0) +
                self.weights["communication"] * t.get("communication", 0) + self.weights["latency"] * t.get("latency", 0))
