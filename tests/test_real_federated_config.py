from scripts.validate_real_federated_config import validate_config
from tradefl.utils.config import load_yaml


def test_checked_in_real_federated_roles_are_valid():
    assert validate_config(load_yaml("configs/real_federated_models.yaml")) == []


def test_api_model_cannot_be_a_fedavg_participant():
    config = {
        "federated_experiments": [
            {
                "experiment_id": "local",
                "model_id": "local/model",
                "architecture": "causal_lm",
                "role": "federated_participant",
            }
        ],
        "external_models": [
            {
                "model_id": "gpt-4o",
                "role": "federated_participant",
                "federated_trainable": True,
            }
        ],
    }

    errors = validate_config(config)

    assert "gpt-4o: API-only external models cannot be marked federated_trainable" in errors
    assert "gpt-4o: external API models cannot participate in FedAvg" in errors
