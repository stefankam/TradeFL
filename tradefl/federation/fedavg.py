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


def dirichlet_label_partition_indices(
    labels: Sequence[object], num_clients: int, seed: int, alpha: float,
) -> list[np.ndarray]:
    """Partition examples with reproducible label skew and no empty clients."""

    if len(labels) < num_clients:
        raise ValueError("num_clients cannot exceed number of labeled examples")
    if num_clients < 2:
        raise ValueError("real federated training requires at least two clients")
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive")
    rng = np.random.default_rng(seed)
    clients: list[list[int]] = [[] for _ in range(num_clients)]
    labels_array = np.asarray(labels, dtype=object)
    for label in sorted(set(labels), key=str):
        indices = np.flatnonzero(labels_array == label)
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        counts = rng.multinomial(len(indices), proportions)
        cursor = 0
        for client, count in enumerate(counts):
            clients[client].extend(indices[cursor:cursor + count].tolist())
            cursor += count
    # A very small alpha can leave clients empty. Move one record at a time
    # from the largest client so every logical client performs real work.
    for empty in [index for index, rows in enumerate(clients) if not rows]:
        donor = max(range(num_clients), key=lambda index: len(clients[index]))
        if len(clients[donor]) <= 1:
            raise ValueError("could not construct non-empty Dirichlet client partitions")
        clients[empty].append(clients[donor].pop())
    for rows in clients:
        rng.shuffle(rows)
    return [np.asarray(rows, dtype=int) for rows in clients]


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
