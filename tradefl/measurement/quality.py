"""Utility metric helpers."""
def accuracy_loss(reference_utility:float, plan_utility:float)->float: return max(0.0, reference_utility-plan_utility)
