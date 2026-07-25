"""SplitFed-style client/server feature-partition reference plan."""
from tradefl.backends import SplitFeatureBackend

from .base import DatasetBackedFineTuningPlan, plan_constructor_kwargs


class SplitFedPlan(DatasetBackedFineTuningPlan):
    """Partition extracted features at the configured client/server cut."""

    def create_backend(self) -> SplitFeatureBackend:
        return SplitFeatureBackend(
            self.dataset.labels,
            split_layer=self.config.split_layer or 8,
            activation_compression=self.config.activation_compression,
        )


def build(**kwargs):
    return SplitFedPlan(**plan_constructor_kwargs(kwargs))
