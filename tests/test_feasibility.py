from tradefl.selection.feasibility import Constraints, check_feasibility

def test_feasibility_filtering_passes():
    c=Constraints(10,10,100,.8,.5)
    r=check_feasibility({'peak_memory_bytes':5,'mean_round_latency_seconds':2,'latency_to_target_seconds':5,'validation_utility':.85,'privacy_risk':.2,'target_reached':True}, c)
    assert r.feasible

def test_hard_constraint_violations():
    c=Constraints(1,1,1,.9,.1, public_cloud_allowed=False, uses_public_cloud=True)
    r=check_feasibility({'peak_memory_bytes':5,'mean_round_latency_seconds':2,'latency_to_target_seconds':5,'validation_utility':.5,'privacy_risk':.2,'target_reached':False}, c)
    assert not r.feasible
    assert 'memory_capacity_exceeded' in r.violations
    assert 'target_quality_not_reached' in r.violations
