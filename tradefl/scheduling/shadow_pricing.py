"""Online shadow-price client scheduling for federated training."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any
import math
import numpy as np


@dataclass(frozen=True)
class ResourceSpec:
    availability: float
    reducer: str = "sum"
    initial_price: float = 0.0
    maximum_price: float = 10.0


class ShadowPriceScheduler:
    """Select client subsets and update physical-resource dual variables.

    A subset is a candidate action. Its predicted resource demand is built from
    per-client exponential moving averages, and its token cost is ``p dot d``.
    After the action executes, prices receive the projected dual update from
    the TradeFL formulation. Unmeasured clients are explored before cost-based
    selection so a client cannot be permanently excluded due to missing data.
    """

    def __init__(self, num_clients: int, clients_per_round: int, config: dict[str, Any], seed: int) -> None:
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.learning_rate = float(config.get("learning_rate", 0.25))
        self.ema_alpha = float(config.get("demand_ema_alpha", 0.5))
        self.update_prices = bool(config.get("update_prices", True))
        default_initial = float(config.get("initial_price", 0.0))
        default_maximum = float(config.get("maximum_price", 10.0))
        self.resources = {
            name: ResourceSpec(
                float(spec["availability"]),
                str(spec.get("reducer", "sum")),
                float(spec.get("initial_price", default_initial)),
                float(spec.get("maximum_price", default_maximum)),
            )
            for name, spec in config.get("resources", {}).items()
        }
        if not self.resources:
            raise ValueError("shadow_pricing.resources must define at least one physical resource")
        for name, spec in self.resources.items():
            if spec.availability <= 0:
                raise ValueError(f"shadow pricing availability for {name} must be positive")
            if spec.reducer not in {"sum", "max"}:
                raise ValueError(f"shadow pricing reducer for {name} must be 'sum' or 'max'")
            if spec.initial_price < 0 or spec.maximum_price < spec.initial_price:
                raise ValueError(f"shadow pricing bounds for {name} must satisfy 0 <= initial_price <= maximum_price")
        self.prices = {name: spec.initial_price for name, spec in self.resources.items()}
        self._predictions: dict[int, dict[str, float]] = {}
        self._rng = np.random.default_rng(seed)
        self._epoch = 0


    def select_clients(self) -> tuple[list[int], dict[str, Any]]:
        """Return the minimum-token-cost candidate subset and its reservation."""

        candidates = list(itertools.combinations(range(self.num_clients), self.clients_per_round))
        unseen_counts = [sum(client not in self._predictions for client in action) for action in candidates]
        maximum_unseen = max(unseen_counts)
        eligible = [action for action, unseen in zip(candidates, unseen_counts) if unseen == maximum_unseen]
        reservations = [self._predict_action(action) for action in eligible]
        costs = [sum(self.prices[name] * demand[name] for name in self.resources) for demand in reservations]
        minimum = min(costs)
        tied = [index for index, cost in enumerate(costs) if np.isclose(cost, minimum)]
        chosen_index = int(self._rng.choice(tied))
        action = eligible[chosen_index]
        return list(action), {
            "candidate_action_count": len(candidates),
            "predicted_demand": reservations[chosen_index],
            "prices_before": dict(self.prices),
            "token_cost": costs[chosen_index],
        }

    def reconcile(
        self,
        client_demands: dict[int, dict[str, float]],
        realized_utility_gain: float | None = None,
    ) -> dict[str, Any]:
        """Reconcile reservation with realization and apply the projected update."""
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
        if self.update_prices:
            for name, spec in self.resources.items():
                # With raw-demand price p=lambda/A, eta_t=learning_rate/A^2
                # is exactly the paper's eta_t*(D-A) projected update.
                eta = self.learning_rate / (
                    spec.availability * spec.availability * math.sqrt(self._epoch + 1)
                )
                self.prices[name] = min(
                    spec.maximum_price,
                    max(0.0, before[name] + eta * (realized[name] - spec.availability)),
                )
            self._epoch += 1
        return {
            "realized_demand": realized,
            "availability": {name: spec.availability for name, spec in self.resources.items()},
            "prices_after": dict(self.prices),
            "price_updated": self.update_prices,
        }

    def _predict_action(self, action: tuple[int, ...]) -> dict[str, float]:
        measured = list(self._predictions.values())
        fallback = {
            name: (float(np.mean([row[name] for row in measured])) if measured else 0.0)
            for name in self.resources
        }
        return self._aggregate(self._predictions.get(client, fallback) for client in action)

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
