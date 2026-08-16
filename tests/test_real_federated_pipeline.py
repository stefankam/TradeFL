from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_real_federated_pipeline import execute_commands, pipeline_commands, validate_component_versions


def arguments(**overrides):
    values = {
        "experiment_config": "configs/experiment_pubmedqa_real.yaml",
        "models_config": "configs/real_federated_models.yaml",
        "output_dir": "outputs/custom",
        "budgets": "configs/budgets.yaml",
        "weights": "configs/weights.yaml",
        "weight_method": "configured",
        "experiment_id": None,
        "cpu_smoke_test": False,
        "allow_slow_cpu": False,
        "strict_hardware": False,
        "skip_training": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_pipeline_connects_all_three_stages_to_same_output_directory():
    commands = pipeline_commands(arguments(strict_hardware=True))

    assert len(commands) == 3
    assert Path(commands[0][1]).name == "run_real_federated_experiment.py"
    assert "--strict-hardware" in commands[0]
    assert Path(commands[1][1]).name == "select_plan.py"
    assert commands[1][commands[1].index("--results") + 1] == str(Path("outputs/custom/raw_metrics.csv"))
    assert commands[1][commands[1].index("--output-dir") + 1] == "outputs/custom"
    assert commands[1][commands[1].index("--weight-method") + 1] == "configured"
    assert Path(commands[2][1]).name == "summarize_results.py"
    assert commands[2][commands[2].index("--input") + 1] == str(Path("outputs/custom/plan_summary.csv"))
    assert commands[2][commands[2].index("--models-config") + 1] == "configs/real_federated_models.yaml"


def test_pipeline_can_resume_at_selection_without_training():
    commands = pipeline_commands(arguments(skip_training=True))

    assert len(commands) == 2
    assert Path(commands[0][1]).name == "select_plan.py"
    assert Path(commands[1][1]).name == "summarize_results.py"


def test_pipeline_forwards_smoke_and_selected_experiment_flags():
    commands = pipeline_commands(
        arguments(cpu_smoke_test=True, experiment_id=["bio_clinicalbert_baseline"])
    )

    assert "--cpu-smoke-test" in commands[0]
    position = commands[0].index("--experiment-id")
    assert commands[0][position + 1] == "bio_clinicalbert_baseline"


def test_strict_hardware_failure_has_actionable_message_without_called_process_traceback(monkeypatch):
    monkeypatch.setattr(
        "scripts.run_real_federated_pipeline.subprocess.run",
        lambda command, check, cwd: SimpleNamespace(returncode=1),
    )

    with pytest.raises(SystemExit, match="--cpu-smoke-test") as error:
        execute_commands([["python", "training.py", "--strict-hardware"]])

    assert "CalledProcessError" not in str(error.value)


def test_non_hardware_stage_failure_identifies_stage(monkeypatch):
    results = iter([SimpleNamespace(returncode=0), SimpleNamespace(returncode=2)])
    monkeypatch.setattr(
        "scripts.run_real_federated_pipeline.subprocess.run",
        lambda command, check, cwd: next(results),
    )

    with pytest.raises(SystemExit, match="stage 2/2 failed with exit code 2"):
        execute_commands([["python", "training.py"], ["python", "selection.py"]])


def test_mixed_pipeline_and_selection_versions_fail_before_training(monkeypatch):
    monkeypatch.setattr(
        "scripts.run_real_federated_pipeline._load_sibling_select_plan",
        lambda: SimpleNamespace(SELECT_PLAN_API_VERSION=1),
    )

    with pytest.raises(SystemExit, match="Update the whole repository") as error:
        validate_component_versions()

    assert "--output-dir" in str(error.value)
    assert "--weight-method" in str(error.value)


def test_current_selection_component_matches_pipeline_contract():
    validate_component_versions()
