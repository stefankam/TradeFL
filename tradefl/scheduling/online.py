"""Common interface and online policies for federated client-subset scheduling."""
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from tradefl.scheduling.shadow_pricing import ResourceSpec


ONLINE_POLICIES = (
    "tradefl_dynamic",
    "tradefl_fixed",
    "independent",
    "static_weighted_sum",
    "greedy",
)


@dataclass(frozen=True)
class CandidateAction:
    action_id: str
    clients: tuple[int, ...]
    predicted_demand: dict[str, float]
    predicted_utility: float


class OnlineScheduler(ABC):
    """Shared state, action space, and telemetry reconciliation for policies."""

    policy_name = "abstract"

    def __init__(
        self,
        num_clients: int,
        clients_per_round: int,
        config: dict[str, Any],
        seed: int,
        client_utilities: dict[int, float] | None = None,
    ) -> None:
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.ema_alpha = float(config.get("demand_ema_alpha", 0.5))
        self.learning_rate = float(config.get("learning_rate", 0.25))
        self.resources = _resource_specs(config)
        self.prices = {name: spec.initial_price for name, spec in self.resources.items()}
        self.client_utilities = client_utilities or {client: 1.0 for client in range(num_clients)}
        self._predictions: dict[int, dict[str, float]] = {}
        self._rng = np.random.default_rng(seed)

    def select_clients(self) -> tuple[list[int], dict[str, Any]]:
        actions = self.candidate_actions()
        unseen = [sum(client not in self._predictions for client in action.clients) for action in actions]
        maximum_unseen = max(unseen)
        eligible = [action for action, count in zip(actions, unseen) if count == maximum_unseen]
        started_prices = dict(self.prices)
        selected, score = self.choose(eligible)
        return list(selected.clients), {
            "policy": self.policy_name,
            "selected_action_id": selected.action_id,
            "candidate_action_count": len(actions),
            "candidate_action_ids": [action.action_id for action in actions],
            "predicted_demand": selected.predicted_demand,
            "predicted_utility": selected.predicted_utility,
            "prices_before": started_prices,
            "policy_score": float(score),
            "token_cost": float(sum(started_prices[name] * selected.predicted_demand[name] for name in self.resources)),
        }

    def candidate_actions(self) -> list[CandidateAction]:
        return [
            CandidateAction(
                action_id="clients:" + ",".join(map(str, clients)),
                clients=clients,
                predicted_demand=self._predict_action(clients),
                predicted_utility=sum(self.client_utilities.get(client, 0.0) for client in clients),
            )
            for clients in itertools.combinations(range(self.num_clients), self.clients_per_round)
        ]

    @abstractmethod
    def choose(self, actions: list[CandidateAction]) -> tuple[CandidateAction, float]:
        """Choose exactly one action from the shared candidate set."""

    def reconcile(self, client_demands: dict[int, dict[str, float]]) -> dict[str, Any]:
        for client, demand in client_demands.items():
            previous = self._predictions.get(client)
            self._predictions[client] = {
                name: float(demand[name]) if previous is None else (
                    self.ema_alpha * float(demand[name]) + (1.0 - self.ema_alpha) * previous[name]
                )
                for name in self.resources
            }
        realized = self._aggregate(client_demands.values())
        before = dict(self.prices)
        self._update_prices(realized)
        return {
            "realized_demand": realized,
            "availability": {name: spec.availability for name, spec in self.resources.items()},
            "prices_after": dict(self.prices),
            "price_updated": before != self.prices,
        }

    def _update_prices(self, realized: dict[str, float]) -> None:
        return None

    def _predict_action(self, clients: tuple[int, ...]) -> dict[str, float]:
        measured = list(self._predictions.values())
        fallback = {
            name: float(np.mean([row[name] for row in measured])) if measured else 0.0
            for name in self.resources
        }
        return self._aggregate(self._predictions.get(client, fallback) for client in clients)

    def _aggregate(self, demands) -> dict[str, float]:
        rows = list(demands)
        return {
            name: (
                max((float(row[name]) for row in rows), default=0.0)
                if spec.reducer == "max"
                else sum(float(row[name]) for row in rows)
            )
            for name, spec in self.resources.items()
        }

    def _pick_minimum(self, actions: list[CandidateAction], scores: list[float]):
        minimum = min(scores)
        tied = [index for index, score in enumerate(scores) if np.isclose(score, minimum)]
        index = int(self._rng.choice(tied))
        return actions[index], scores[index]


class DynamicPriceScheduler(OnlineScheduler):
    policy_name = "tradefl_dynamic"

    def choose(self, actions):
        return self._pick_minimum(actions, [
            sum(self.prices[name] * action.predicted_demand[name] for name in self.resources)
            for action in actions
        ])

    def _update_prices(self, realized):
        for name, spec in self.resources.items():
            eta = self.learning_rate / (spec.availability * spec.availability)
            self.prices[name] = min(
                spec.maximum_price,
                max(0.0, self.prices[name] + eta * (realized[name] - spec.availability)),
            )


class FixedPriceScheduler(DynamicPriceScheduler):
    policy_name = "tradefl_fixed"

    def _update_prices(self, realized):
        return None


class IndependentResourceScheduler(OnlineScheduler):
    policy_name = "independent"

    def choose(self, actions):
        scores = [
            max(action.predicted_demand[name] / spec.availability for name, spec in self.resources.items())
            for action in actions
        ]
        return self._pick_minimum(actions, scores)


class StaticWeightedSumScheduler(OnlineScheduler):
    policy_name = "static_weighted_sum"

    def choose(self, actions):
        scores = [
            sum(action.predicted_demand[name] / spec.availability for name, spec in self.resources.items())
            for action in actions
        ]
        return self._pick_minimum(actions, scores)


class GreedyUtilityScheduler(OnlineScheduler):
    policy_name = "greedy"

    def choose(self, actions):
        maximum = max(action.predicted_utility for action in actions)
        candidates = [action for action in actions if np.isclose(action.predicted_utility, maximum)]
        selected = candidates[int(self._rng.integers(len(candidates)))]
        return selected, -selected.predicted_utility


def build_online_scheduler(policy, num_clients, clients_per_round, config, seed, client_utilities=None):
    classes = {
        "tradefl_dynamic": DynamicPriceScheduler,
        "tradefl_fixed": FixedPriceScheduler,
        "independent": IndependentResourceScheduler,
        "static_weighted_sum": StaticWeightedSumScheduler,
        "greedy": GreedyUtilityScheduler,
    }
    if policy not in classes:
        raise ValueError(f"Unknown online scheduler policy {policy!r}; choose one of {ONLINE_POLICIES}")
    return classes[policy](num_clients, clients_per_round, config, seed, client_utilities)


def _resource_specs(config):
    default_initial = float(config.get("initial_price", 1.0))
    default_maximum = float(config.get("maximum_price", 10.0))
    resources = {
        name: ResourceSpec(
            float(spec["availability"]),
            str(spec.get("reducer", "sum")),
            float(spec.get("initial_price", default_initial)),
            float(spec.get("maximum_price", default_maximum)),
        )
        for name, spec in config.get("resources", {}).items()
    }
    if not resources:
        raise ValueError("online scheduling requires at least one configured physical resource")
    for name, spec in resources.items():
        if spec.availability <= 0:
            raise ValueError(f"online scheduling availability for {name} must be positive")
        if spec.reducer not in {"sum", "max"}:
            raise ValueError(f"online scheduling reducer for {name} must be 'sum' or 'max'")
        if spec.initial_price < 0 or spec.maximum_price < spec.initial_price:
            raise ValueError(f"online scheduling bounds for {name} are invalid")
    return resources
