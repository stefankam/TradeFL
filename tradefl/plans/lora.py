"""LoRA-style bounded feature-adapter reference plan."""
from tradefl.backends import HashedAdapterBackend

from .base import DatasetBackedFineTuningPlan, plan_constructor_kwargs


class LoRAPlan(DatasetBackedFineTuningPlan):
    """Train only a rank-sized hashed feature adapter."""

    def create_backend(self) -> HashedAdapterBackend:
        rank = self.config.adapter_rank or 8
        return HashedAdapterBackend(self.dataset.labels, feature_buckets=rank * 32)

def build(**kwargs):
    return LoRAPlan(**plan_constructor_kwargs(kwargs))
