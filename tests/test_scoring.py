import pytest
from tradefl.selection.scoring import TradeFLWeights, calculate_score, renormalize_weights

def test_tradefl_score():
    costs={'memory':0.5,'compute':0.4,'communication':0.3,'energy':0.2,'latency':0.6,'privacy':0.1,'accuracy':0.25}
    weights=TradeFLWeights(.2,.1,.1,.1,.2,.1,.2)
    assert calculate_score(costs, weights)==pytest.approx(.2*.5+.1*.4+.1*.3+.1*.2+.2*.6+.1*.1+.2*.25)

def test_weight_validation():
    with pytest.raises(ValueError): TradeFLWeights(.5,.5,.5,0,0,0,0).validate()
    with pytest.raises(ValueError): TradeFLWeights(-1,1,1,0,0,0,0).validate()

def test_disabled_energy_metric():
    w=renormalize_weights(TradeFLWeights(.2,.2,.2,.2,.1,.05,.05), {'energy':False,'memory':True,'compute':True,'communication':True,'latency':True,'privacy':True,'accuracy':True})
    assert w.energy == 0
    assert sum(w.asdict().values()) == pytest.approx(1.0)
