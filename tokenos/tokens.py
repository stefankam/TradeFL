"""Unified 100-token resource abstraction."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable
from .device import EdgeDevice

RESOURCE_KEYS = ("memory", "compute", "communication", "energy", "latency", "privacy", "accuracy", "storage")


@dataclass
class ResourceTokenManager:
    """Normalizes raw device resources and maintains per-device 100-token budgets."""

    weights: Dict[str, float] = field(default_factory=lambda: {k: 1.0 for k in RESOURCE_KEYS})

    def initialize(self, devices: Iterable[EdgeDevice]) -> None:
        for device in devices:
            device.current_token_budget = self.normalize_device(device)

    def normalize_device(self, device: EdgeDevice) -> Dict[str, float]:
        raw = {
            "memory": max(0.01, device.memory_capacity * (1.0 - device.memory_pressure)),
            "compute": max(0.01, device.compute_capacity * (1.0 - device.cpu_load)),
            "communication": max(0.01, device.effective_bandwidth()),
            "energy": max(0.01, device.battery_level),
            "latency": max(0.01, device.latency_budget),
            "privacy": max(0.01, device.privacy_level),
            "accuracy": max(0.01, device.data_value),
            "storage": max(0.01, device.storage_capacity),
        }
        weighted = {k: raw[k] * self.weights.get(k, 1.0) for k in RESOURCE_KEYS}
        total = sum(weighted.values())
        tokens = {k: 100.0 * weighted[k] / total for k in RESOURCE_KEYS}
        return self._rebalance(tokens)

    def update_allocation(self, device: EdgeDevice) -> Dict[str, float]:
        device.current_token_budget = self.normalize_device(device)
        return device.current_token_budget

    def apply_exchange(self, tokens: Dict[str, float], rule: str) -> Dict[str, float]:
        updated = dict(tokens)
        exchanges = {
            "checkpoint": {"memory": 10, "compute": -4},
            "compression": {"memory": 15, "accuracy": -2},
            "cloud_offload": {"accuracy": 10, "communication": -8, "privacy": -5},
            "local_inference": {"privacy": 8, "memory": -5, "compute": -5},
            "dvfs": {"energy": 8, "latency": -5},
        }
        for key, delta in exchanges.get(rule, {}).items():
            updated[key] = max(0.0, updated.get(key, 0.0) + delta)
        return self._rebalance(updated)

    def _rebalance(self, tokens: Dict[str, float]) -> Dict[str, float]:
        clipped = {k: max(0.0, float(tokens.get(k, 0.0))) for k in RESOURCE_KEYS}
        total = sum(clipped.values()) or 1.0
        return {k: 100.0 * clipped[k] / total for k in RESOURCE_KEYS}
