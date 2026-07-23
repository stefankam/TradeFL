"""Common TradeFL interfaces and real dataset-backed fine-tuning plans."""
from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from tradefl.backends import BagOfWordsFineTuningBackend
from tradefl.data import DatasetBundle
from tradefl.measurement.compute import Timer
from tradefl.measurement.communication import CommunicationCounter
from tradefl.measurement.memory import peak_rss_bytes


@dataclass
class PlanResult:
    """Cost-to-target result for one concrete TradeFL fine-tuning plan."""

    plan_id: str
    seed: int
    peak_memory_bytes: float
    compute_time_seconds: float
    bytes_uploaded: int
    bytes_downloaded: int
    energy_joules: float | None
    latency_seconds: float
    validation_utility: float
    test_utility: float | None
    privacy_risk: float
    rounds_completed: int
    rounds_to_target: int | None
    target_reached: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainingPlan(ABC):
    """Public interface implemented by every candidate TradeFL plan."""

    plan_id: str

    @abstractmethod
    def setup(self) -> None:
        """Prepare data/model state before profiling."""

    @abstractmethod
    def run_round(self, round_index: int) -> dict[str, Any]:
        """Run one real local fine-tuning/profiling round."""

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        """Evaluate the current global model on held-out data."""

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources after profiling."""


@dataclass(frozen=True)
class FineTuningPlanConfig:
    """Concrete plan configuration used to instantiate a plan."""

    plan_id: str
    plan_type: str
    trainable_scope: str
    adapter_rank: int | None = None
    quantization_bits: int | None = None
    split_layer: int | None = None
    activation_compression: bool = False
    privacy_risk: float = 0.25
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetBackedFineTuningPlan(TrainingPlan):
    """Fine-tune on real MedNLI/PubMedQA records using a real train/eval loop."""

    def __init__(self, config: FineTuningPlanConfig, dataset: DatasetBundle, seed: int, batch_size: int, local_epochs: int, primary_metric: str) -> None:
        self.config = config
        self.plan_id = config.plan_id
        self.dataset = dataset
        self.seed = seed
        self.batch_size = max(1, batch_size)
        self.local_epochs = max(1, local_epochs)
        self.primary_metric = primary_metric
        self.backend = BagOfWordsFineTuningBackend(dataset.labels)
        self._cursor = 0
        self._rounds: list[dict[str, Any]] = []

    def setup(self) -> None:
        self._cursor = 0
        self._rounds.clear()
        self.backend = BagOfWordsFineTuningBackend(self.dataset.labels)

    def run_round(self, round_index: int) -> dict[str, Any]:
        batch = self._next_batch()
        counter = CommunicationCounter()
        before_size = self.backend.serialized_size_bytes()
        with Timer() as timer:
            for _ in range(self.local_epochs):
                self.backend.train(batch)
        validation = self.backend.evaluate(self.dataset.validation)
        test = self.backend.evaluate(self.dataset.test)
        after_size = self.backend.serialized_size_bytes()
        upload_payload = {"plan_id": self.plan_id, "round": round_index, "delta_model_bytes": after_size}
        download_payload = {"plan_id": self.plan_id, "round": round_index, "model_bytes": before_size}
        counter.record_upload(upload_payload, "model_update")
        counter.record_download(download_payload, "global_model")
        compute_seconds = max(timer.seconds, 1e-9)
        communication_seconds = (counter.total_uploaded_bytes + counter.total_downloaded_bytes) / 10_000_000
        latency_seconds = compute_seconds + communication_seconds
        row = {
            "plan_id": self.plan_id,
            "seed": self.seed,
            "round_index": round_index,
            "peak_memory_bytes": peak_rss_bytes(),
            "compute_time_seconds": compute_seconds,
            "bytes_uploaded": counter.total_uploaded_bytes,
            "bytes_downloaded": counter.total_downloaded_bytes,
            "energy_joules": None,
            "latency_seconds": latency_seconds,
            "validation_utility": validation[self.primary_metric],
            "test_utility": test[self.primary_metric],
            "privacy_risk": self.config.privacy_risk,
            "metadata": {
                **self.config.metadata,
                "dataset": self.dataset.name,
                "labels": self.dataset.labels,
                "split_sizes": self.dataset.split_sizes,
                "backend": "bag_of_words_reference",
                "trainable_scope": self.config.trainable_scope,
                "primary_metric": self.primary_metric,
            },
        }
        self._rounds.append(row)
        return row

    def evaluate(self) -> dict[str, float]:
        validation = self.backend.evaluate(self.dataset.validation)
        test = self.backend.evaluate(self.dataset.test)
        return {
            "validation_utility": validation[self.primary_metric],
            "test_utility": test[self.primary_metric],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            "test_accuracy": test["accuracy"],
            "test_macro_f1": test["macro_f1"],
            "worst_client_utility": self._worst_client_utility(),
            "cross_client_variance": self._cross_client_variance(),
        }

    def cleanup(self) -> None:
        return None

    def _next_batch(self) -> list[Any]:
        train = self.dataset.train
        batch = [train[(self._cursor + offset) % len(train)] for offset in range(self.batch_size)]
        self._cursor = (self._cursor + self.batch_size) % len(train)
        return batch

    def _worst_client_utility(self) -> float:
        scores = self._client_scores()
        return min(scores) if scores else 0.0

    def _cross_client_variance(self) -> float:
        scores = self._client_scores()
        return statistics.pvariance(scores) if len(scores) > 1 else 0.0

    def _client_scores(self) -> list[float]:
        if not self.dataset.validation:
            return []
        partitions = [self.dataset.validation[index::5] for index in range(5)]
        return [self.backend.evaluate(partition)[self.primary_metric] for partition in partitions if partition]


def build_dataset_plan(plan_cfg: dict[str, Any], dataset: DatasetBundle, seed: int, experiment: dict[str, Any]) -> DatasetBackedFineTuningPlan:
    """Build a concrete dataset-backed fine-tuning plan from config."""

    config = FineTuningPlanConfig(
        plan_id=str(plan_cfg["plan_id"]),
        plan_type=str(plan_cfg["type"]),
        trainable_scope=str(plan_cfg.get("trainable_scope", "adapters")),
        adapter_rank=plan_cfg.get("rank"),
        quantization_bits=plan_cfg.get("quantization_bits"),
        split_layer=plan_cfg.get("split_layer"),
        activation_compression=bool(plan_cfg.get("activation_compression", False)),
        privacy_risk=float(plan_cfg.get("privacy_risk", 0.25)),
        metadata={key: value for key, value in plan_cfg.items() if key not in {"plan_id", "type"}},
    )
    return DatasetBackedFineTuningPlan(
        config=config,
        dataset=dataset,
        seed=seed,
        batch_size=int(experiment.get("batch_size", 4)),
        local_epochs=int(experiment.get("local_epochs", 1)),
        primary_metric=str(experiment.get("primary_metric", "accuracy")),
    )
