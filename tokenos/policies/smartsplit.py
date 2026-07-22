from dataclasses import dataclass
from typing import Dict

@dataclass
class SmartSplitPolicy:
    """Choose a split layer from memory, communication, and latency tokens."""
    num_layers: int = 12

    def choose_split(self, tokens: Dict[str, float]) -> int:
        memory = tokens.get("memory", 0); comm = tokens.get("communication", 0); latency = tokens.get("latency", 0)
        ratio = (comm + latency) / max(1.0, memory + comm + latency)
        return int(max(1, min(self.num_layers, round(1 + ratio * (self.num_layers - 1)))))
