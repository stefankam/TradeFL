from dataclasses import dataclass
from typing import Dict

@dataclass
class FedHybridPolicy:
    """Select a tensor memory strategy."""
    def plan(self, tokens: Dict[str, float]) -> str:
        if tokens.get("memory", 0) >= 18 and tokens.get("compute", 0) >= 10:
            return "keep activation"
        if tokens.get("accuracy", 0) < 8:
            return "checkpoint"
        if tokens.get("compute", 0) > 16:
            return "recompute"
        return "compress"
