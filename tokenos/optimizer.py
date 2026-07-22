"""Unified TokenOS optimizer."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
from .device import EdgeDevice
from .tokens import ResourceTokenManager
from .policies import FedRankPolicy, SmartSplitPolicy, FedHybridPolicy, ClonePolicy, FloePolicy

@dataclass
class TokenOptimizer:
    """Evaluates policy decisions and emits an execution plan per round."""
    token_manager: ResourceTokenManager = field(default_factory=ResourceTokenManager)
    fedrank: FedRankPolicy = field(default_factory=FedRankPolicy)
    smartsplit: SmartSplitPolicy = field(default_factory=SmartSplitPolicy)
    fedhybrid: FedHybridPolicy = field(default_factory=FedHybridPolicy)
    clone: ClonePolicy = field(default_factory=ClonePolicy)
    floe: FloePolicy = field(default_factory=FloePolicy)
    participation_rate: float = 0.2

    def plan_round(self, devices: List[EdgeDevice], round_id: int) -> Dict:
        for d in devices:
            self.token_manager.update_allocation(d)
        selected = self.fedrank.rank(devices)[:max(1, int(len(devices) * self.participation_rate))]
        plans = []
        totals = {"utility": 0.0, "accuracy": 0.0, "energy": 0.0, "latency": 0.0, "memory_pressure": 0.0, "communication": 0.0}
        for d in selected:
            tokens = d.current_token_budget
            split = self.smartsplit.choose_split(tokens)
            memory_strategy = self.fedhybrid.plan(tokens)
            config = self.clone.configure(tokens)
            mode = self.floe.choose_mode(tokens)
            metrics = self.score(tokens, mode, memory_strategy, config)
            for k in totals:
                totals[k] += metrics[k]
            plans.append({"device_id": d.device_id, "tokens": tokens, "split_layer": split, "memory_strategy": memory_strategy,
                          "inference_config": config, "execution_mode": mode, "metrics": metrics})
        n = max(1, len(selected))
        aggregate = {k: v / n for k, v in totals.items()}
        aggregate["successful_clients"] = len(selected)
        return {"round": round_id, "selected_clients": [d.device_id for d in selected], "device_plans": plans, "metrics": aggregate}

    def score(self, tokens: Dict[str, float], mode: str, memory_strategy: str, config: Dict[str, str]) -> Dict[str, float]:
        accuracy_gain = tokens["accuracy"] * 0.55 + (8 if mode == "cloud LLM only" else 5 if mode.startswith("hybrid") else 2)
        privacy_gain = tokens["privacy"] * 0.4 + (5 if mode == "local SLM only" else -3 if mode == "cloud LLM only" else 1)
        latency_cost = max(0.0, 25 - tokens["latency"]) * 0.22 + (4 if mode == "cloud LLM only" else 2 if mode.startswith("hybrid") else 1)
        energy_cost = max(0.0, 22 - tokens["energy"]) * 0.20 + (1 if config.get("dvfs_level") == "low" else 3)
        communication_cost = max(0.0, 20 - tokens["communication"]) * 0.18 + (8 if mode == "cloud LLM only" else 4 if mode.startswith("hybrid") else 0.5)
        memory_pressure = max(0.0, 20 - tokens["memory"]) / 20
        if memory_strategy == "compress":
            accuracy_gain -= 1.5; memory_pressure *= 0.75
        elif memory_strategy == "checkpoint":
            energy_cost += 1.0; memory_pressure *= 0.65
        elif memory_strategy == "recompute":
            energy_cost += 1.5; memory_pressure *= 0.8
        utility = accuracy_gain + privacy_gain - latency_cost - energy_cost - communication_cost
        return {"utility": utility, "accuracy": accuracy_gain, "energy": energy_cost, "latency": latency_cost,
                "memory_pressure": memory_pressure, "communication": communication_cost}
