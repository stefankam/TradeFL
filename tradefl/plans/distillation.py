"""Reference plan plus genuine soft-logit Transformer distillation."""
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


def transformer_distillation_trainer(base_trainer, torch_module, functional):
    """Return a genuine soft-logit, temperature-scaled Transformer trainer."""

    class DistillationTrainer(base_trainer):
        def __init__(self, *args, teacher_model, temperature, alpha, **kwargs):
            super().__init__(*args, **kwargs)
            self.teacher_model = teacher_model
            self.temperature = temperature
            self.alpha = alpha

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            with torch_module.no_grad():
                teacher_logits = self.teacher_model(**inputs).logits
            temperature = self.temperature
            soft_loss = functional.kl_div(
                functional.log_softmax(outputs.logits / temperature, dim=-1),
                functional.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature**2)
            hard_loss = functional.cross_entropy(outputs.logits, inputs["labels"])
            loss = self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss
            return (loss, outputs) if return_outputs else loss

    return DistillationTrainer
