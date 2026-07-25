"""Framework-independent primitives for genuine sample-weighted FedAvg."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


TensorState = Mapping[str, np.ndarray]


def iid_partition_indices(num_examples: int, num_clients: int, seed: int) -> list[np.ndarray]:
    """Assign every training example to exactly one deterministic IID client."""

    if num_examples < 1:
        raise ValueError("num_examples must be positive")
    if num_clients < 2:
        raise ValueError("real federated training requires at least two clients")
    if num_clients > num_examples:
        raise ValueError("num_clients cannot exceed num_examples")
    indices = np.arange(num_examples)
    np.random.default_rng(seed).shuffle(indices)
    return [partition.copy() for partition in np.array_split(indices, num_clients)]


def sample_weighted_fedavg(updates: Sequence[tuple[TensorState, int]]) -> dict[str, np.ndarray]:
    """Aggregate matching client tensors using client example counts as weights."""

    if len(updates) < 2:
        raise ValueError("FedAvg requires updates from at least two clients")
    states, counts = zip(*updates)
    if any(count <= 0 for count in counts):
        raise ValueError("client example counts must be positive")
    names = set(states[0])
    if any(set(state) != names for state in states[1:]):
        raise ValueError("all clients must return identical tensor names")
    total = sum(counts)
    averaged: dict[str, np.ndarray] = {}
    for name in sorted(names):
        tensors = [np.asarray(state[name]) for state in states]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError(f"client tensor shapes do not match for {name}")
        accumulator = np.zeros(tensors[0].shape, dtype=np.float64)
        for tensor, count in zip(tensors, counts):
            accumulator += tensor.astype(np.float64) * (count / total)
        averaged[name] = accumulator.astype(tensors[0].dtype)
    return averaged
