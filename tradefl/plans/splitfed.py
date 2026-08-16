"""Reference plan plus a genuine Transformer activation/gradient cut."""
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


def register_transformer_boundary(model, split_layer: int, counter: dict[str, int]):
    """Register the BERT cut that transports activations and their gradients."""

    encoder = getattr(getattr(model, "bert", None), "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        raise ValueError("splitfed currently requires a BERT-family sequence-classification model")
    if not 1 <= split_layer < len(layers):
        raise ValueError(f"split_layer must be between 1 and {len(layers) - 1}")

    def boundary_hook(module, inputs, output):
        activation = output[0] if isinstance(output, tuple) else output
        transported = activation.clone()
        counter["uploaded_bytes"] += transported.numel() * transported.element_size()
        if transported.requires_grad:
            transported.register_hook(
                lambda gradient: counter.__setitem__(
                    "downloaded_bytes",
                    counter["downloaded_bytes"] + gradient.numel() * gradient.element_size(),
                )
            )
        if isinstance(output, tuple):
            return (transported, *output[1:])
        return transported

    return layers[split_layer - 1].register_forward_hook(boundary_hook)
