import pytest
from tradefl.selection.normalization import Budgets, normalize_metrics

def test_budget_normalization():
    metrics={'peak_memory_bytes':5,'compute_to_target_seconds':4,'communication_to_target_bytes':3,'energy_to_target_joules':2,'latency_to_target_seconds':6,'privacy_risk':1,'accuracy_loss':.25}
    out=normalize_metrics(metrics, Budgets(10,10,10,10,10,2,.5))
    assert out['memory']==.5 and out['accuracy']==.5

def test_missing_metric_handling():
    with pytest.raises(ValueError): normalize_metrics({}, Budgets(1,1,1,None,1,1,1))

