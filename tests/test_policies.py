from tokenos.device import EdgeDevice
from tokenos.tokens import ResourceTokenManager
from tokenos.policies import FedRankPolicy, SmartSplitPolicy, FedHybridPolicy, ClonePolicy, FloePolicy


def test_policies_return_valid_outputs():
    d1=EdgeDevice("a",8,8,50,80,100,5,64,20); d2=EdgeDevice("b",2,2,5,20,30,9,16,5)
    ResourceTokenManager().initialize([d1,d2])
    assert FedRankPolicy().rank([d2,d1])[0].device_id in {"a","b"}
    split=SmartSplitPolicy(num_layers=12).choose_split(d1.current_token_budget)
    assert 1 <= split <= 12
    assert FedHybridPolicy().plan(d1.current_token_budget) in {"keep activation","checkpoint","compress","recompute"}
    cfg=ClonePolicy().configure(d1.current_token_budget)
    assert {"model","lora_rank","dvfs_level"} <= set(cfg)
    assert FloePolicy().choose_mode(d1.current_token_budget) in {"local SLM only","cloud LLM only","hybrid SLM + cloud LLM"}
