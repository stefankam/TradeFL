"""Teacher/student distillation reference plan."""
from tradefl.backends import BagOfWordsFineTuningBackend, HashedAdapterBackend
from tradefl.data.loaders import DatasetRecord

from .base import DatasetBackedFineTuningPlan, plan_constructor_kwargs


class DistillationPlan(DatasetBackedFineTuningPlan):
    """Train a compact student from predictions made by a full teacher."""

    def create_backend(self) -> HashedAdapterBackend:
        return HashedAdapterBackend(self.dataset.labels, feature_buckets=64)

    def setup(self) -> None:
        super().setup()
        self.teacher = BagOfWordsFineTuningBackend(self.dataset.labels)
        self.teacher.train(self.dataset.train)

    def train_batch(self, batch: list[DatasetRecord]) -> None:
        distilled = [
            DatasetRecord(record.text, self.teacher.predict(record.text), {**record.metadata, "distilled": True})
            for record in batch
        ]
        self.backend.train(distilled)


def build(**kwargs):
    return DistillationPlan(**plan_constructor_kwargs(kwargs))
