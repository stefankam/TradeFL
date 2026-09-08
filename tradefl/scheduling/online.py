"""Common interface and online policies for federated client-subset scheduling."""
from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from tradefl.scheduling.shadow_pricing import ResourceSpec


ONLINE_POLICIES = (
    "random_feasible",
    "fedcs",
    "oort",
    "pedpc",
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
    uses_tradefl_objective = False

    def __init__(
        self,
        num_clients: int,
        clients_per_round: int,
        config: dict[str, Any],
        seed: int,
        client_utilities: dict[int, float] | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> None:
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.ema_alpha = float(config.get("demand_ema_alpha", 0.5))
        self.learning_rate = float(config.get("learning_rate", 0.25))
        self.utility_ema_alpha = float(config.get("utility_ema_alpha", 0.5))
        if not 0.0 <= self.ema_alpha <= 1.0 or not 0.0 <= self.utility_ema_alpha <= 1.0:
            raise ValueError("demand and utility EMA alpha values must be between zero and one")
        self.resources = _resource_specs(config)
        self.constraints = dict(constraints or {})
        self.prices = {name: spec.initial_price for name, spec in self.resources.items()}
        self.client_utilities = client_utilities or {client: 1.0 for client in range(num_clients)}
        policy = config.get("policy_constraints", {})
        self.required_clients = frozenset(map(int, policy.get("required_clients", [])))
        self.forbidden_clients = frozenset(map(int, policy.get("forbidden_clients", [])))
        self.incompatible_pairs = {
            frozenset(map(int, pair)) for pair in policy.get("incompatible_client_pairs", [])
        }
        bank = config.get("token_bank", {})
        balance = bank.get("initial_balance")
        self.token_balance = float(balance) if balance is not None else float("inf")
        if self.token_balance < 0:
            raise ValueError("token-bank initial balance must be nonnegative")
        self._predictions: dict[int, dict[str, float]] = {}
        self._pending_reservation: dict[str, Any] | None = None
        self._epoch = 0
        self._rng = np.random.default_rng(seed)
        self.selection_counts = {client: 0 for client in range(num_clients)}
        self.last_selected_epoch = {client: -1 for client in range(num_clients)}
        self.client_losses: dict[int, float] = {}



    def select_clients(self) -> tuple[list[int], dict[str, Any]]:
        if self._pending_reservation is not None:
            raise RuntimeError("the previous token reservation must be reconciled before selecting another plan")
        actions = self.candidate_actions()
        feasible = [action for action in actions if self._is_feasible(action)]
        affordable = [action for action in feasible if self._token_cost(action) <= self.token_balance]
        if not affordable:
            reason = "no_feasible_plan" if not feasible else "insufficient_token_balance"
            return [], {
                "policy": self.policy_name,
                "decision_status": reason,
                "selected_action_id": "NO_FEASIBLE_PLAN",
                "candidate_action_count": len(actions),
                "feasible_action_count": len(feasible),
                "affordable_action_count": 0,
                "candidate_action_ids": [action.action_id for action in actions],
                "predicted_demand": {},
                "predicted_utility": None,
                "prices_before": dict(self.prices),
                "prices_after": dict(self.prices),
                "price_updated": False,
                "token_cost": None,
                "reserved_charge": None,
                "token_balance_before": self.token_balance,
                "token_balance_after_reservation": self.token_balance,
                "token_balance_after_reconciliation": self.token_balance,
            }
        started_prices = dict(self.prices)
        selected, score = self.choose(affordable)
        token_cost = self._token_cost(selected)
        balance_before = self.token_balance
        self.token_balance -= token_cost
        self._pending_reservation = {
            "clients": selected.clients,
            "predicted_demand": dict(selected.predicted_demand),
            "prices": started_prices,
            "reserved_charge": token_cost,
        }
        for client in selected.clients:
            self.selection_counts[client] += 1
            self.last_selected_epoch[client] = self._epoch
        telemetry = {
            "policy": self.policy_name,
            "decision_status": "selected",
            "selected_action_id": selected.action_id,
            "candidate_action_count": len(actions),
            "feasible_action_count": len(feasible),
            "affordable_action_count": len(affordable),
            "candidate_action_ids": [action.action_id for action in actions],
            "predicted_demand": selected.predicted_demand,
            "predicted_utility": selected.predicted_utility,
            "prices_before": started_prices,
            "policy_score": float(score),
            "token_cost": token_cost,
            "reserved_charge": token_cost,
            "token_balance_before": balance_before,
            "token_balance_after_reservation": self.token_balance,
        }
        if self.uses_tradefl_objective:
            telemetry["net_plan_score"] = selected.predicted_utility - token_cost
        return list(selected.clients), telemetry





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

    def reconcile(
        self,
        client_demands: dict[int, dict[str, float]],
        realized_utility_gain: float | None = None,
        client_losses: dict[int, float] | None = None,
    ) -> dict[str, Any]:
        if self._pending_reservation is None:
            raise RuntimeError("cannot reconcile without a pending token reservation")
        if client_losses:
            for client, loss in client_losses.items():
                old = self.client_losses.get(client, float(loss))
                self.client_losses[client] = self.utility_ema_alpha * float(loss) + (1.0 - self.utility_ema_alpha) * old
        for client, demand in client_demands.items():
            previous = self._predictions.get(client)
            self._predictions[client] = {
                name: float(demand[name]) if previous is None else (
                    self.ema_alpha * float(demand[name]) + (1.0 - self.ema_alpha) * previous[name]
                )
                for name in self.resources
            }
        realized = self._aggregate(client_demands.values())
        reservation = self._pending_reservation
        refunds = {
            name: reservation["prices"][name]
            * max(reservation["predicted_demand"][name] - realized[name], 0.0)
            for name in self.resources
        }
        overruns = {
            name: reservation["prices"][name]
            * max(realized[name] - reservation["predicted_demand"][name], 0.0)
            for name in self.resources
        }
        refund = sum(refunds.values())
        overrun = sum(overruns.values())
        self.token_balance += refund - overrun
        if realized_utility_gain is not None:
            per_client_gain = float(realized_utility_gain) / len(reservation["clients"])
            for client in reservation["clients"]:
                old = self.client_utilities.get(client, per_client_gain)
                self.client_utilities[client] = (
                    self.utility_ema_alpha * per_client_gain + (1.0 - self.utility_ema_alpha) * old
                )
        before = dict(self.prices)
        self._update_prices(realized)
        self._epoch += 1
        self._pending_reservation = None
        return {
            "realized_demand": realized,
            "availability": {name: spec.availability for name, spec in self.resources.items()},
            "prices_after": dict(self.prices),
            "price_updated": before != self.prices,
            "refunds": refunds,
            "overruns": overruns,
            "refund_charge": refund,
            "overrun_charge": overrun,
            "reconciled_charge": reservation["reserved_charge"] - refund + overrun,
            "token_balance_after_reconciliation": self.token_balance,
            "realized_utility_gain": realized_utility_gain,
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

    def _token_cost(self, action: CandidateAction) -> float:
        return float(sum(self.prices[name] * action.predicted_demand[name] for name in self.resources))

    def _is_feasible(self, action: CandidateAction) -> bool:
        clients = frozenset(action.clients)
        policy_compatible = (
            self.required_clients.issubset(clients)
            and not clients.intersection(self.forbidden_clients)
            and not any(pair.issubset(clients) for pair in self.incompatible_pairs)
        )
        utility_feasible = action.predicted_utility >= float(
            self.constraints.get("minimum_validation_utility", float("-inf"))
        )
        return policy_compatible and utility_feasible and all(
            action.predicted_demand[name] <= spec.availability for name, spec in self.resources.items()
        )


class DynamicPriceScheduler(OnlineScheduler):
    policy_name = "tradefl_dynamic"
    uses_tradefl_objective = True

    def choose(self, actions):
        gains = [action.predicted_utility - self._token_cost(action) for action in actions]
        maximum = max(gains)
        tied = [index for index, gain in enumerate(gains) if np.isclose(gain, maximum)]
        index = int(self._rng.choice(tied))
        return actions[index], gains[index]

    def _update_prices(self, realized):
        for name, spec in self.resources.items():
            eta = self.learning_rate / (
                spec.availability * spec.availability * math.sqrt(self._epoch + 1)
            )
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
        scores = [self.score_action(action) for action in actions]
        return self._pick_minimum(actions, scores)

    def score_action(self, action):
        return sum(action.predicted_demand[name] / spec.availability for name, spec in self.resources.items())


class GreedyUtilityScheduler(OnlineScheduler):
    policy_name = "greedy"

    def choose(self, actions):
        maximum = max(self.score_action(action) for action in actions)
        candidates = [action for action in actions if np.isclose(self.score_action(action), maximum)]
        selected = candidates[int(self._rng.integers(len(candidates)))]
        return selected, selected.predicted_utility

    def score_action(self, action):
        return action.predicted_utility


class RandomFeasibleScheduler(OnlineScheduler):
    policy_name = "random_feasible"

    def choose(self, actions):
        index = int(self._rng.integers(len(actions)))
        return actions[index], 0.0


class FedCSScheduler(OnlineScheduler):
    policy_name = "fedcs"

    def choose(self, actions):
        """FedCS selection phase instantiated with measured local completion time.

        The common experiment fixes the selected cardinality, so maximizing the
        number completed before the deadline reduces to choosing the action with
        the smallest predicted completion time.
        """
        latency_resource = "compute_time_seconds" if "compute_time_seconds" in self.resources else next(iter(self.resources))
        scores = [action.predicted_demand[latency_resource] for action in actions]
        return self._pick_minimum(actions, scores)


class OortScheduler(OnlineScheduler):
    policy_name = "oort"

    def choose(self, actions):
        """Oort-style utility, system-efficiency, exploration, and staleness score."""
        latency_resource = "compute_time_seconds" if "compute_time_seconds" in self.resources else next(iter(self.resources))
        exploration = float(self.constraints.get("oort_exploration_factor", 0.3))
        scores = []
        for action in actions:
            client_scores = []
            for client in action.clients:
                # Oort's statistical utility is loss based. Before a client has
                # a measured loss, the label-representativeness prior is used.
                statistical = self.client_losses.get(client, self.client_utilities.get(client, 1.0))
                duration = self._predictions.get(client, {}).get(latency_resource, 0.0)
                duration = max(duration, 1e-12)
                uncertainty = math.sqrt(math.log(self._epoch + 2.0) / (self.selection_counts[client] + 1.0))
                staleness = max(0, self._epoch - self.last_selected_epoch[client])
                client_scores.append(statistical / duration + exploration * uncertainty * math.sqrt(staleness + 1.0))
            scores.append(sum(client_scores))
        maximum = max(scores)
        tied = [index for index, score in enumerate(scores) if np.isclose(score, maximum)]
        index = int(self._rng.choice(tied))
        return actions[index], scores[index]


class PEDPCScheduler(DynamicPriceScheduler):
    policy_name = "pedpc"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pedpc = args[2].get("pedpc", {}) if len(args) >= 3 else {}
        self.pedpc_v = float(pedpc.get("v", 1.0))
        self.energy_budget = float(pedpc.get("client_energy_budget_joules", 5000.0))
        self.energy_queues = {client: 0.0 for client in range(self.num_clients)}
        self.energy_predictions: dict[int, float] = {}

    def choose(self, actions):
        scores = []
        for action in actions:
            queue_penalty = sum(
                self.energy_queues[client]
                * (self.energy_predictions.get(client, self.energy_budget) - self.energy_budget)
                for client in action.clients
            )
            scores.append(queue_penalty + self.pedpc_v * (1.0 - action.predicted_utility))
        return self._pick_minimum(actions, scores)

    def reconcile(self, client_demands, realized_utility_gain=None, client_losses=None):
        for client, demand in client_demands.items():
            energy = demand.get("energy_joules")
            if energy is not None:
                old = self.energy_predictions.get(client, float(energy))
                self.energy_predictions[client] = self.ema_alpha * float(energy) + (1.0 - self.ema_alpha) * old
                self.energy_queues[client] = max(
                    0.0, self.energy_queues[client] + float(energy) - self.energy_budget
                )
        result = super().reconcile(client_demands, realized_utility_gain, client_losses)
        result["pedpc_energy_deficit_queues"] = dict(self.energy_queues)
        return result

    def _update_prices(self, realized):
        # PEDPC updates virtual energy-deficit queues, not TradeFL token prices.
        return None


def build_online_scheduler(
    policy, num_clients, clients_per_round, config, seed, client_utilities=None, constraints=None,
):
    classes = {
        "random_feasible": RandomFeasibleScheduler,
        "fedcs": FedCSScheduler,
        "oort": OortScheduler,
        "pedpc": PEDPCScheduler,
        "tradefl_dynamic": DynamicPriceScheduler,
        "tradefl_fixed": FixedPriceScheduler,
        "independent": IndependentResourceScheduler,
        "static_weighted_sum": StaticWeightedSumScheduler,
        "greedy": GreedyUtilityScheduler,
    }
    if policy not in classes:
        raise ValueError(f"Unknown online scheduler policy {policy!r}; choose one of {ONLINE_POLICIES}")
    return classes[policy](num_clients, clients_per_round, config, seed, client_utilities, constraints)


def _resource_specs(config):
    default_initial = float(config.get("initial_price", 0.0))
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
