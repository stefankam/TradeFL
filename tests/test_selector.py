import pytest
from tradefl.selection.feasibility import Constraints
from tradefl.selection.normalization import Budgets
from tradefl.selection.scoring import TradeFLWeights
from tradefl.selection.selector import select_best
from tradefl.utils.seeds import set_seed
import random

def rows():
    base={'seed':42,'energy_to_target_joules':None,'target_reached':True,'mean_round_latency_seconds':10,'privacy_risk':.2,'validation_utility':.85,'accuracy_loss':.05}
    return [dict(base, plan_id='a', peak_memory_bytes=5, compute_to_target_seconds=5, communication_to_target_bytes=5, latency_to_target_seconds=5), dict(base, plan_id='b', peak_memory_bytes=2, compute_to_target_seconds=2, communication_to_target_bytes=2, latency_to_target_seconds=2)]

def test_plan_ranking_and_nearest_alternatives():
    res=select_best(rows(), Budgets(10,10,10,None,10,1,.1), TradeFLWeights(.2,.2,.2,0,.2,.1,.1), Constraints(10,20,100,.8,.5), {'energy':False})
    assert res['selected_plan']=='b'
    assert res['alternatives'][0]['plan_id']=='a'

def test_cost_to_target_aggregation_prefers_lower_total():
    res=select_best(rows(), Budgets(10,10,10,None,10,1,.1), TradeFLWeights(.2,.2,.2,0,.2,.1,.1), Constraints(10,20,100,.8,.5), {'energy':False})
    assert res['score'] < res['alternatives'][0]['score']

def test_deterministic_seed_handling():
    set_seed(7); a=random.random(); set_seed(7); assert random.random()==pytest.approx(a)
