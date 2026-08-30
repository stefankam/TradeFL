import pytest

from tradefl.scheduling.shadow_pricing import ShadowPriceScheduler


def config(update_prices=True):
    return {
        "learning_rate": 0.25,
        "initial_price": 1.0,
        "maximum_price": 2.0,
        "update_prices": update_prices,
        "resources": {"compute": {"availability": 10.0, "reducer": "sum"}},
    }


def test_projected_dual_update_uses_realized_minus_available_demand():
    scheduler = ShadowPriceScheduler(2, 2, config(), seed=42)
    clients, reservation = scheduler.select_clients()

    reconciliation = scheduler.reconcile({0: {"compute": 8.0}, 1: {"compute": 7.0}})

    assert clients == [0, 1]
    assert reservation["token_cost"] == 0.0
    assert reconciliation["realized_demand"] == {"compute": 15.0}
    assert reconciliation["prices_after"]["compute"] == pytest.approx(1.0125)


def test_price_projection_and_fixed_price_ablation():
    dynamic = ShadowPriceScheduler(1, 1, config(), seed=42)
    fixed = ShadowPriceScheduler(1, 1, config(update_prices=False), seed=42)

    assert dynamic.reconcile({0: {"compute": 1000.0}})["prices_after"]["compute"] == 2.0
    assert fixed.reconcile({0: {"compute": 1000.0}})["prices_after"]["compute"] == 1.0


def test_scheduler_explores_unmeasured_clients_then_uses_token_cost():
    scheduler = ShadowPriceScheduler(3, 1, config(), seed=7)
    first, _ = scheduler.select_clients()
    scheduler.reconcile({first[0]: {"compute": 9.0}})
    second, _ = scheduler.select_clients()

    assert second != first


def test_training_action_minimizes_predicted_shadow_token_cost():
    scheduler = ShadowPriceScheduler(3, 1, config(), seed=7)
    scheduler.reconcile({0: {"compute": 8.0}, 1: {"compute": 2.0}, 2: {"compute": 6.0}})

    selected, action = scheduler.select_clients()

    assert selected == [1]
    assert action["predicted_demand"] == {"compute": 2.0}
    assert action["token_cost"] == pytest.approx(action["prices_before"]["compute"] * 2.0)


def test_invalid_resource_configuration_is_rejected():
    with pytest.raises(ValueError, match="availability"):
        ShadowPriceScheduler(2, 1, {"resources": {"compute": {"availability": 0}}}, seed=1)
