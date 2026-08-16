"""Reference plan plus genuine PEFT LoRA injection."""
from tradefl.backends import HashedAdapterBackend

from .base import DatasetBackedFineTuningPlan, plan_constructor_kwargs


class LoRAPlan(DatasetBackedFineTuningPlan):
    """Train only a rank-sized hashed feature adapter."""

    def create_backend(self) -> HashedAdapterBackend:
        rank = self.config.adapter_rank or 8
        return HashedAdapterBackend(self.dataset.labels, feature_buckets=rank * 32)


def build(**kwargs):
    return LoRAPlan(**plan_constructor_kwargs(kwargs))


def attach_peft_lora(model, peft_module, model_config: dict, architecture: str):
    """Inject genuine trainable PEFT LoRA matrices into a Transformer."""

    config = peft_module.LoraConfig(
        task_type="SEQ_CLS" if architecture == "sequence_classification" else "CAUSAL_LM",
        r=int(model_config.get("lora_rank", 8)),
        lora_alpha=int(model_config.get("lora_alpha", 16)),
        lora_dropout=float(model_config.get("lora_dropout", 0.05)),
        target_modules=model_config.get("target_modules"),
    )
    return peft_module.get_peft_model(model, config)
