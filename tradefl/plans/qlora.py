"""Reference plan plus genuine bitsandbytes NF4 QLoRA configuration."""
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


def quantization_config(transformers_module, torch_module, training: dict):
    """Build genuine bitsandbytes NF4 QLoRA loading configuration."""

    return transformers_module.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=getattr(torch_module, training.get("compute_dtype", "bfloat16")),
    )
