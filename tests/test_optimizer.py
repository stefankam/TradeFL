from tokenos.simulator import Simulation
from tokenos.optimizer import TokenOptimizer


def test_optimizer_plan_has_metrics():
    devices = Simulation(rounds=1, num_devices=10, seed=1).create_devices()
    plan = TokenOptimizer().plan_round(devices, 0)
    assert plan["metrics"]["successful_clients"] == 2
    assert len(plan["device_plans"]) == 2
    assert "utility" in plan["metrics"]
