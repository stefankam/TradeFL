"""Quantized LoRA-style feature-adapter reference plan."""
from tradefl.backends import HashedAdapterBackend

from .base import DatasetBackedFineTuningPlan, plan_constructor_kwargs


class QLoRAPlan(DatasetBackedFineTuningPlan):
    """Train a rank-bounded adapter with quantized communication accounting."""

    def create_backend(self) -> HashedAdapterBackend:
        rank = self.config.adapter_rank or 8
        bits = self.config.quantization_bits or 4
        return HashedAdapterBackend(self.dataset.labels, feature_buckets=rank * 32, quantization_bits=bits)


def build(**kwargs):
    return QLoRAPlan(**plan_constructor_kwargs(kwargs))
