"""Full-vocabulary supervised fine-tuning reference plan."""
from .base import DatasetBackedFineTuningPlan, plan_constructor_kwargs

class FullFineTuningPlan(DatasetBackedFineTuningPlan):
    """Update the complete bag-of-words model without adapter constraints."""


def build(**kwargs):
    return FullFineTuningPlan(**plan_constructor_kwargs(kwargs))
