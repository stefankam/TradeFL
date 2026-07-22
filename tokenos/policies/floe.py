from dataclasses import dataclass
from typing import Dict

@dataclass
class FloePolicy:
    """Choose cloud-edge collaboration mode."""
    def choose_mode(self, tokens: Dict[str, float]) -> str:
        comm = tokens.get("communication", 0); privacy = tokens.get("privacy", 0); acc = tokens.get("accuracy", 0)
        if privacy > 18 or comm < 8:
            return "local SLM only"
        if comm > 17 and privacy < 10 and acc < 16:
            return "cloud LLM only"
        return "hybrid SLM + cloud LLM"
