"""Reference plan plus genuine Transformer full-state FedAvg helpers."""
import numpy as np

from .base import DatasetBackedFineTuningPlan, plan_constructor_kwargs


class FullFineTuningPlan(DatasetBackedFineTuningPlan):
    """Update the complete bag-of-words model without adapter constraints."""


def build(**kwargs):
    return FullFineTuningPlan(**plan_constructor_kwargs(kwargs))


def extract_transformer_state(model) -> dict[str, np.ndarray]:
    """Extract every floating-point tensor for genuine full-model FedAvg."""

    return {
        name: tensor.detach().cpu().numpy().copy()
        for name, tensor in model.state_dict().items()
        if tensor.is_floating_point()
    }


def load_transformer_state(model, torch_module, state: dict[str, np.ndarray]) -> None:
    """Load a FedAvg full-model state into a Transformer."""

    model.load_state_dict({name: torch_module.from_numpy(value) for name, value in state.items()}, strict=False)
