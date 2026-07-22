from tokenos.device import EdgeDevice
from tokenos.tokens import ResourceTokenManager


def device():
    return EdgeDevice("d", 8, 4, 10, 90, 100, 5, 64, 10)


def test_initialize_creates_100_tokens():
    d = device(); tm = ResourceTokenManager(); tm.initialize([d])
    assert round(sum(d.current_token_budget.values()), 6) == 100
    assert set(d.current_token_budget) == {"memory","compute","communication","energy","latency","privacy","accuracy","storage"}


def test_exchange_rebalances():
    d = device(); tm = ResourceTokenManager(); tokens = tm.normalize_device(d)
    exchanged = tm.apply_exchange(tokens, "checkpoint")
    assert round(sum(exchanged.values()), 6) == 100
    assert exchanged["memory"] > tokens["memory"]
