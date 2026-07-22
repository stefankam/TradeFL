"""Device model for heterogeneous federated edge clients."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class EdgeDevice:
    """A simulated edge device with raw capacities and runtime state."""

    device_id: str
    memory_capacity: float
    compute_capacity: float
    bandwidth: float
    battery_level: float
    latency_budget: float
    privacy_level: float
    storage_capacity: float
    data_value: float
    current_token_budget: Dict[str, float] = field(default_factory=dict)
    memory_pressure: float = 0.0
    cpu_load: float = 0.0
    network_quality: float = 1.0

    def drain_battery(self, amount: float) -> None:
        self.battery_level = max(0.0, self.battery_level - amount)

    def degrade_network(self, factor: float) -> None:
        self.network_quality = min(1.0, max(0.05, self.network_quality * factor))

    def apply_memory_pressure(self, pressure: float) -> None:
        self.memory_pressure = min(1.0, max(0.0, pressure))

    def update_cpu_load(self, load: float) -> None:
        self.cpu_load = min(1.0, max(0.0, load))

    def effective_bandwidth(self) -> float:
        return self.bandwidth * self.network_quality

    def randomize_runtime(self, rng) -> None:
        """Apply deterministic random runtime variance from a numpy Generator."""
        self.drain_battery(float(rng.uniform(0.05, 1.5)))
        self.degrade_network(float(rng.uniform(0.85, 1.08)))
        self.apply_memory_pressure(float(np_clip(rng.normal(self.memory_pressure, 0.12))))
        self.update_cpu_load(float(np_clip(rng.normal(0.45, 0.25))))


def np_clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
