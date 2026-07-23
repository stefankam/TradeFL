"""Deprecated module kept for import compatibility.

TradeFL now profiles fine-tuning plans only. Configure adapter, quantized,
SplitFed, or distillation plans in configs/plans.yaml instead of full training.
"""

def build(**_: object):
    raise ValueError("full_finetuning is not supported; TradeFL now runs fine-tuning plans only.")
